import argparse
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import torch
from tensordict import TensorDict

from libraries import constants
from libraries.classes.Planner import Planner
from libraries.classes.SumoSimulator import Simulator
from libraries.constants import EDGE_DATA_FILE_PATH, PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH
from libraries.utils.preprocessingUtils import generateEdgeDataFile
from taz_rl.rlenv.local_taz_env_v10 import SumoTazEnvV10
from taz_rl.training_script_ppo_v10 import (
    ACTION_BINS,
    BASE_DEMAND,
    ENV_V10_CONFIG,
    HOURLY_DEMAND_PROFILE,
    ROUTE_RANDOM_TRIP_SEED,
    ROUTE_SAMPLER_SEED,
    ROUTE_SAMPLER_THREADS,
    ActorCriticV10,
    _resolve_route_folder,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints_v10", "checkpoint_ppo_v10_best.pt")
REPORT_ROOT = os.path.join(SCRIPT_DIR, "evaluation_reports_v10")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def _safe_parse_xml_root(xml_path: str):
    try:
        return ET.parse(xml_path).getroot()
    except Exception:
        return None


def _parse_timeslot(timeslot: str) -> tuple[int, str]:
    match = re.fullmatch(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", timeslot.strip())
    if match is None:
        raise ValueError("timeslot must have format HH:MM-HH:MM")

    start_hour, start_minute, end_hour, end_minute = [int(value) for value in match.groups()]
    if start_minute != 0 or end_minute != 0:
        raise ValueError("only full-hour slots are supported, for example 08:00-09:00")
    if end_hour != (start_hour + 1) % 24:
        raise ValueError("timeslot must span exactly one hour")
    return start_hour, timeslot.replace(":", "-")


def _prepare_route_folder(simulation_date: str, timeslot: str, planner: Planner) -> str:
    hour, timeslot_clean = _parse_timeslot(timeslot)
    total_cars = int(BASE_DEMAND * HOURLY_DEMAND_PROFILE[hour])
    route_folder_path, should_generate_routes = _resolve_route_folder(
        simulation_date=simulation_date,
        timeslot_clean=timeslot_clean,
        total_cars_random=total_cars,
    )

    if not should_generate_routes:
        print(f"[ROUTE CACHE] Reusing deterministic trip/route files from {route_folder_path}")
        return route_folder_path

    print(f"[ROUTE GEN] Generating deterministic trip/route files at {route_folder_path}")
    os.makedirs(os.path.join(route_folder_path, "output"), exist_ok=True)
    generateEdgeDataFile(PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH, date=simulation_date, time_slot=timeslot)
    planner.scenarioGenerator.generateRoute(
        inputEdgePath=EDGE_DATA_FILE_PATH,
        timeSlot=timeslot_clean,
        totalCount=total_cars,
        custom=False,
        outputFolder=route_folder_path,
        randomTripSeed=ROUTE_RANDOM_TRIP_SEED,
        routeSamplerSeed=ROUTE_SAMPLER_SEED,
        routeSamplerThreads=ROUTE_SAMPLER_THREADS,
    )
    return route_folder_path


def _parse_waiting_metrics(tripinfo_path: str) -> dict:
    root = _safe_parse_xml_root(tripinfo_path)
    if root is None:
        return {
            "trip_count": 0,
            "avg_waiting_time": 0.0,
            "total_waiting_time": 0.0,
            "ok": False,
            "path": tripinfo_path,
        }

    waiting_times = []
    for trip in root.findall("tripinfo"):
        waiting_times.append(_safe_float(trip.get("waitingTime"), 0.0))

    if not waiting_times:
        return {
            "trip_count": 0,
            "avg_waiting_time": 0.0,
            "total_waiting_time": 0.0,
            "ok": False,
            "path": tripinfo_path,
        }

    total_waiting_time = float(sum(waiting_times))
    return {
        "trip_count": int(len(waiting_times)),
        "avg_waiting_time": float(total_waiting_time / len(waiting_times)),
        "total_waiting_time": total_waiting_time,
        "ok": True,
        "path": tripinfo_path,
    }


def _parse_emission_total(emission_path: str, metric: str) -> dict:
    root = _safe_parse_xml_root(emission_path)
    if root is None:
        return {"total": 0.0, "ok": False, "path": emission_path}

    total = 0.0
    saw_value = False
    for timestep in root.findall("timestep"):
        vehicles = timestep.findall("vehicle")
        if vehicles:
            for vehicle in vehicles:
                total += _safe_float(vehicle.get(metric), 0.0)
            saw_value = True
            continue

        value = timestep.get(metric)
        if value is not None:
            total += _safe_float(value, 0.0)
            saw_value = True

    return {"total": float(total), "ok": bool(saw_value), "path": emission_path}


def _collect_run_metrics(run_root: str) -> dict:
    output_dir = os.path.join(run_root, "output")
    tripinfo_path = os.path.join(output_dir, "tripinfos.xml")
    emission_path = os.path.join(output_dir, "emission-output.xml")

    waiting = _parse_waiting_metrics(tripinfo_path)
    co2 = _parse_emission_total(emission_path, "CO2")
    nox = _parse_emission_total(emission_path, "NOx")

    return {
        "trip_count": int(waiting["trip_count"]),
        "avg_waiting_time": float(waiting["avg_waiting_time"]),
        "total_waiting_time": float(waiting["total_waiting_time"]),
        "total_co2": float(co2["total"]),
        "total_nox": float(nox["total"]),
        "ok": bool(waiting["ok"] and co2["ok"] and nox["ok"]),
        "tripinfo_path": waiting["path"],
        "emission_path": emission_path,
    }


def _load_policy(checkpoint_path: str, env: SumoTazEnvV10) -> tuple[ActorCriticV10, dict]:
    runtime_device = torch.device(env.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    policy = ActorCriticV10(env.agent_obs_dim, env.max_tls_per_taz, ACTION_BINS).to(runtime_device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy, checkpoint


def _run_rl_episode(env: SumoTazEnvV10, policy: ActorCriticV10, hour: int):
    env.set_baseline_penalty(None)
    env.set_episode_context(hour=hour)
    td = env.reset()

    with torch.no_grad():
        while True:
            obs = td["observation"]
            action_mask = td["action_mask"].bool()
            action_index, _, _ = policy.act(obs, action_mask, deterministic=True)
            step_td = env.step(TensorDict({"action": action_index}, batch_size=[], device=env.device))
            next_td = step_td["next"] if "next" in step_td.keys() else step_td

            terminated = bool(next_td["terminated"].item())
            truncated = bool(next_td["truncated"].item())
            if terminated or truncated:
                break
            td = next_td


def _metric_comparison(baseline_value: float, rl_value: float) -> dict:
    delta_rl_minus_baseline = float(rl_value - baseline_value)
    improvement_vs_baseline = float(baseline_value - rl_value)
    if abs(baseline_value) > 1e-9:
        improvement_pct = float((baseline_value - rl_value) / baseline_value * 100.0)
    else:
        improvement_pct = None
    return {
        "baseline": float(baseline_value),
        "rl_best": float(rl_value),
        "delta_rl_minus_baseline": delta_rl_minus_baseline,
        "improvement_vs_baseline": improvement_vs_baseline,
        "improvement_pct": improvement_pct,
    }


def _print_summary(report: dict):
    comparison = report["comparison"]
    print("")
    print("Metric | Baseline | RL Best | Delta (RL-Baseline) | Improvement")
    print("avg_waiting_time | "
          f"{comparison['avg_waiting_time']['baseline']:.4f} | "
          f"{comparison['avg_waiting_time']['rl_best']:.4f} | "
          f"{comparison['avg_waiting_time']['delta_rl_minus_baseline']:.4f} | "
          f"{comparison['avg_waiting_time']['improvement_vs_baseline']:.4f}")
    print("total_waiting_time | "
          f"{comparison['total_waiting_time']['baseline']:.4f} | "
          f"{comparison['total_waiting_time']['rl_best']:.4f} | "
          f"{comparison['total_waiting_time']['delta_rl_minus_baseline']:.4f} | "
          f"{comparison['total_waiting_time']['improvement_vs_baseline']:.4f}")
    print("total_co2 | "
          f"{comparison['total_co2']['baseline']:.4f} | "
          f"{comparison['total_co2']['rl_best']:.4f} | "
          f"{comparison['total_co2']['delta_rl_minus_baseline']:.4f} | "
          f"{comparison['total_co2']['improvement_vs_baseline']:.4f}")
    print("total_nox | "
          f"{comparison['total_nox']['baseline']:.4f} | "
          f"{comparison['total_nox']['rl_best']:.4f} | "
          f"{comparison['total_nox']['delta_rl_minus_baseline']:.4f} | "
          f"{comparison['total_nox']['improvement_vs_baseline']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare a no-agent baseline run against the best v10 checkpoint on the same date and timeslot."
    )
    parser.add_argument("--date", required=True, help="Simulation date in YYYY-MM-DD format.")
    parser.add_argument("--timeslot", required=True, help="Timeslot in HH:MM-HH:MM format, for example 08:00-09:00.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH, help="Path to the v10 checkpoint to evaluate.")
    parser.add_argument("--report-root", default=REPORT_ROOT, help="Root folder where evaluation outputs and report are saved.")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    hour, timeslot_clean = _parse_timeslot(args.timeslot)
    report_dir = os.path.join(os.path.abspath(args.report_root), args.date, timeslot_clean)
    baseline_run_dir = os.path.join(report_dir, "baseline")
    rl_run_dir = os.path.join(report_dir, "rl_best")
    os.makedirs(os.path.join(baseline_run_dir, "output"), exist_ok=True)
    os.makedirs(os.path.join(rl_run_dir, "output"), exist_ok=True)

    sumo_standalone_dir = os.path.join(constants.SUMO_PATH, "standalone")
    log_file = os.path.join(sumo_standalone_dir, "compare_best_v10_vs_baseline.log")
    sumo = Simulator(configurationPath=sumo_standalone_dir, logFile=log_file, tazTlsMapFile=constants.TAZ_FILE)
    planner = Planner(simulator=sumo)
    sumo.changeDetectorPath(detectorPath=constants.SUMO_NETWORK_PATH)
    env = SumoTazEnvV10(sumoSimulator=sumo, stepSize=300, **ENV_V10_CONFIG)

    try:
        route_folder_path = _prepare_route_folder(args.date, args.timeslot, planner)

        print("[BASELINE] Running no-agent baseline evaluation.")
        sumo.changeRouteFilePath(route_folder_path)
        sumo.changeTypePath(baseline_run_dir)
        env.run_reference_episode_no_agent(hour=hour)
        baseline_metrics = _collect_run_metrics(baseline_run_dir)

        print("[RL BEST] Loading checkpoint and running deterministic policy evaluation.")
        policy, checkpoint = _load_policy(args.checkpoint, env)
        sumo.changeRouteFilePath(route_folder_path)
        sumo.changeTypePath(rl_run_dir)
        _run_rl_episode(env, policy, hour=hour)
        rl_metrics = _collect_run_metrics(rl_run_dir)

        comparison = {
            "avg_waiting_time": _metric_comparison(baseline_metrics["avg_waiting_time"], rl_metrics["avg_waiting_time"]),
            "total_waiting_time": _metric_comparison(baseline_metrics["total_waiting_time"], rl_metrics["total_waiting_time"]),
            "total_co2": _metric_comparison(baseline_metrics["total_co2"], rl_metrics["total_co2"]),
            "total_nox": _metric_comparison(baseline_metrics["total_nox"], rl_metrics["total_nox"]),
        }

        report = {
            "date": args.date,
            "timeslot": args.timeslot,
            "hour": int(hour),
            "checkpoint_path": os.path.abspath(args.checkpoint),
            "checkpoint_episode": int(checkpoint.get("episode", -1)),
            "route_folder_path": route_folder_path,
            "baseline_run_dir": baseline_run_dir,
            "rl_run_dir": rl_run_dir,
            "baseline": baseline_metrics,
            "rl_best": rl_metrics,
            "comparison": comparison,
        }

        report_path = os.path.join(report_dir, "comparison.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        _print_summary(report)
        print("")
        print(f"Report saved to {report_path}")
    finally:
        try:
            if env.sumo.isLoaded():
                env.sumo.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()
