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
from taz_rl.rlenv.local_taz_env_v11 import SumoTazEnvV11
from taz_rl.training_script_ppo_v10 import ActorCriticV10
from taz_rl.training_script_ppo_v11 import (
    ACTION_BINS,
    BASE_DEMAND,
    ENV_V11_CONFIG,
    HOURLY_DEMAND_PROFILE,
    ROUTE_RANDOM_TRIP_SEED,
    ROUTE_SAMPLER_SEED,
    ROUTE_SAMPLER_THREADS,
    _resolve_route_folder,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints_v11", "checkpoint_ppo_v11_best.pt")
REPORT_ROOT = os.path.join(SCRIPT_DIR, "evaluation_reports_v11")


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


def _collect_run_metrics(run_root: str, waiting_components: dict | None = None) -> dict:
    output_dir = os.path.join(run_root, "output")
    emission_path = os.path.join(output_dir, "emission-output.xml")
    waiting = dict(waiting_components or {})
    co2 = _parse_emission_total(emission_path, "CO2")
    nox = _parse_emission_total(emission_path, "NOx")

    return {
        "trip_count": int(waiting.get("trip_count", 0)),
        "avg_waiting_time": float(waiting.get("avg_waiting_time", 0.0)),
        "total_waiting_time": float(waiting.get("total_waiting_time", 0.0)),
        "waiting_metric_source": str(waiting.get("waiting_metric_source", "unknown")),
        "waiting_time_by_taz": dict(waiting.get("waiting_time_by_taz", {}) or {}),
        "vehicle_count_by_taz": dict(waiting.get("vehicle_count_by_taz", {}) or {}),
        "total_co2": float(co2["total"]),
        "total_nox": float(nox["total"]),
        "ok": bool(waiting.get("parse_ok", False) and co2["ok"] and nox["ok"]),
        "tripinfo_path": str(waiting.get("tripinfo_path", os.path.join(output_dir, "tripinfos.xml"))),
        "emission_path": emission_path,
    }


def _load_policy(checkpoint: dict, env: SumoTazEnvV11) -> ActorCriticV10:
    runtime_device = torch.device(env.device)
    policy = ActorCriticV10(
        env.agent_obs_dim,
        env.max_control_groups_per_taz,
        ACTION_BINS,
    ).to(runtime_device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy


def _run_rl_episode(
    env: SumoTazEnvV11,
    policy: ActorCriticV10,
    hour: int,
    deterministic: bool,
) -> dict:
    env.set_baseline_penalty(None)
    env.set_episode_context(hour=hour)
    td = env.reset()

    with torch.no_grad():
        while True:
            obs = td["observation"]
            action_mask = td["action_mask"].bool()
            action_index, _, _ = policy.act(obs, action_mask, deterministic=deterministic)
            step_td = env.step(TensorDict({"action": action_index}, batch_size=[], device=env.device))
            next_td = step_td["next"] if "next" in step_td.keys() else step_td

            terminated = bool(next_td["terminated"].item())
            truncated = bool(next_td["truncated"].item())
            if terminated or truncated:
                break
            td = next_td

    return {
        "terminal_reward": dict((env.last_reward_components or {}).get("terminal_reward", {}) or {}),
        "group_action_details": dict((env.last_reward_components or {}).get("group_action_details", {}) or {}),
        "group_signal_by_taz": dict((env.last_reward_components or {}).get("group_signal_by_taz", {}) or {}),
    }


def _metric_comparison(baseline_value: float, rl_value: float) -> dict:
    delta_rl_minus_baseline = float(rl_value - baseline_value)
    improvement_vs_baseline = float(baseline_value - rl_value)
    improvement_pct = float((baseline_value - rl_value) / baseline_value * 100.0) if abs(baseline_value) > 1e-9 else None
    return {
        "baseline": float(baseline_value),
        "rl_best": float(rl_value),
        "delta_rl_minus_baseline": delta_rl_minus_baseline,
        "improvement_vs_baseline": improvement_vs_baseline,
        "improvement_pct": improvement_pct,
    }


def _build_comparison(baseline_metrics: dict, rl_metrics: dict) -> dict:
    return {
        "avg_waiting_time": _metric_comparison(baseline_metrics["avg_waiting_time"], rl_metrics["avg_waiting_time"]),
        "total_waiting_time": _metric_comparison(baseline_metrics["total_waiting_time"], rl_metrics["total_waiting_time"]),
        "total_co2": _metric_comparison(baseline_metrics["total_co2"], rl_metrics["total_co2"]),
        "total_nox": _metric_comparison(baseline_metrics["total_nox"], rl_metrics["total_nox"]),
    }


def _aggregate_metrics(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {
            "run_count": 0,
            "trip_count": 0.0,
            "avg_waiting_time": 0.0,
            "total_waiting_time": 0.0,
            "total_co2": 0.0,
            "total_nox": 0.0,
            "ok": False,
            "waiting_metric_source": "none",
            "run_dirs": [],
        }

    def _mean(key: str) -> float:
        return float(np.mean([_safe_float(metrics.get(key), 0.0) for metrics in metrics_list]))

    sources = {str(metrics.get("waiting_metric_source", "unknown")) for metrics in metrics_list}
    return {
        "run_count": int(len(metrics_list)),
        "trip_count": _mean("trip_count"),
        "avg_waiting_time": _mean("avg_waiting_time"),
        "total_waiting_time": _mean("total_waiting_time"),
        "total_co2": _mean("total_co2"),
        "total_nox": _mean("total_nox"),
        "ok": bool(all(bool(metrics.get("ok", False)) for metrics in metrics_list)),
        "waiting_metric_source": next(iter(sources)) if len(sources) == 1 else "mixed",
        "run_dirs": [str(metrics.get("run_dir", "")) for metrics in metrics_list],
    }


def _print_summary(report: dict):
    for section_name in ("comparison_greedy", "comparison_stochastic_mean"):
        section = report[section_name]
        print(section_name)
        for metric_name, values in section.items():
            improvement_pct = values.get("improvement_pct")
            improvement_pct_text = "n/a" if improvement_pct is None else f"{improvement_pct:.2f}%"
            print(
                f"{metric_name}: baseline={values['baseline']:.4f} | "
                f"rl={values['rl_best']:.4f} | "
                f"delta={values['delta_rl_minus_baseline']:.4f} | "
                f"improvement={values['improvement_vs_baseline']:.4f} | "
                f"improvement_pct={improvement_pct_text}"
            )
        print("")


def main():
    parser = argparse.ArgumentParser(description="Compare a no-agent baseline run against a v11 checkpoint.")
    parser.add_argument("--date", required=True, help="Simulation date in YYYY-MM-DD format.")
    parser.add_argument("--timeslot", required=True, help="Timeslot in HH:MM-HH:MM format, for example 08:00-09:00.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH, help="Path to the v11 checkpoint to evaluate.")
    parser.add_argument("--report-root", default=REPORT_ROOT, help="Root folder where evaluation outputs and report are saved.")
    parser.add_argument("--stochastic-runs", type=int, default=1, help="Number of sampled-policy evaluation runs to average.")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.stochastic_runs < 1:
        raise ValueError("--stochastic-runs must be >= 1")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint_env_config = dict(checkpoint.get("env_config", ENV_V11_CONFIG))
    single_taz_id = checkpoint.get("single_taz_id")
    selected_taz_ids = [str(single_taz_id)] if single_taz_id else None

    hour, timeslot_clean = _parse_timeslot(args.timeslot)
    report_dir = os.path.join(os.path.abspath(args.report_root), args.date, timeslot_clean)
    baseline_run_dir = os.path.join(report_dir, "baseline")
    rl_greedy_run_dir = os.path.join(report_dir, "rl_greedy")
    rl_stochastic_root_dir = os.path.join(report_dir, "rl_stochastic")
    os.makedirs(os.path.join(baseline_run_dir, "output"), exist_ok=True)
    os.makedirs(os.path.join(rl_greedy_run_dir, "output"), exist_ok=True)
    os.makedirs(rl_stochastic_root_dir, exist_ok=True)

    sumo_standalone_dir = os.path.join(constants.SUMO_PATH, "standalone")
    log_file = os.path.join(sumo_standalone_dir, "compare_best_v11_vs_baseline.log")
    sumo = Simulator(configurationPath=sumo_standalone_dir, logFile=log_file, tazTlsMapFile=constants.TAZ_FILE)
    planner = Planner(simulator=sumo)
    sumo.changeDetectorPath(detectorPath=constants.SUMO_NETWORK_PATH)
    env = SumoTazEnvV11(
        sumoSimulator=sumo,
        stepSize=300,
        selected_taz_ids=selected_taz_ids,
        **checkpoint_env_config,
    )

    try:
        route_folder_path = _prepare_route_folder(args.date, args.timeslot, planner)
        policy = _load_policy(checkpoint, env)

        print("[BASELINE] Running no-agent baseline evaluation.")
        sumo.changeRouteFilePath(route_folder_path)
        sumo.changeTypePath(baseline_run_dir)
        _, baseline_components = env.run_reference_episode_no_agent(hour=hour)
        baseline_metrics = _collect_run_metrics(baseline_run_dir, waiting_components=baseline_components)
        baseline_metrics["run_dir"] = baseline_run_dir

        print("[RL GREEDY] Running deterministic v11 policy evaluation.")
        sumo.changeRouteFilePath(route_folder_path)
        sumo.changeTypePath(rl_greedy_run_dir)
        rl_greedy_rollout = _run_rl_episode(env, policy, hour=hour, deterministic=True)
        rl_greedy_metrics = _collect_run_metrics(rl_greedy_run_dir, waiting_components=rl_greedy_rollout["terminal_reward"])
        rl_greedy_metrics["run_dir"] = rl_greedy_run_dir
        rl_greedy_metrics["group_action_details"] = rl_greedy_rollout["group_action_details"]
        rl_greedy_metrics["group_signal_by_taz"] = rl_greedy_rollout["group_signal_by_taz"]

        rl_stochastic_runs = []
        for run_idx in range(args.stochastic_runs):
            run_dir = os.path.join(rl_stochastic_root_dir, f"run_{run_idx + 1:02d}")
            os.makedirs(os.path.join(run_dir, "output"), exist_ok=True)
            print(f"[RL STOCHASTIC] Run {run_idx + 1}/{args.stochastic_runs}.")
            sumo.changeRouteFilePath(route_folder_path)
            sumo.changeTypePath(run_dir)
            rl_rollout = _run_rl_episode(env, policy, hour=hour, deterministic=False)
            rl_metrics = _collect_run_metrics(run_dir, waiting_components=rl_rollout["terminal_reward"])
            rl_metrics["run_dir"] = run_dir
            rl_metrics["run_index"] = int(run_idx + 1)
            rl_metrics["group_action_details"] = rl_rollout["group_action_details"]
            rl_metrics["group_signal_by_taz"] = rl_rollout["group_signal_by_taz"]
            rl_stochastic_runs.append(rl_metrics)

        rl_stochastic_mean = _aggregate_metrics(rl_stochastic_runs)
        comparison_greedy = _build_comparison(baseline_metrics, rl_greedy_metrics)
        comparison_stochastic_mean = _build_comparison(baseline_metrics, rl_stochastic_mean)

        report = {
            "date": args.date,
            "timeslot": args.timeslot,
            "hour": int(hour),
            "checkpoint_path": os.path.abspath(args.checkpoint),
            "checkpoint_episode": int(checkpoint.get("episode", -1)),
            "checkpoint_selection_metric": checkpoint.get("checkpoint_selection_metric"),
            "single_taz_id": single_taz_id,
            "route_folder_path": route_folder_path,
            "control_groups_by_taz": checkpoint.get("control_groups_by_taz"),
            "baseline_run_dir": baseline_run_dir,
            "rl_greedy_run_dir": rl_greedy_run_dir,
            "rl_stochastic_root_dir": rl_stochastic_root_dir,
            "baseline": baseline_metrics,
            "rl_greedy": rl_greedy_metrics,
            "rl_stochastic_runs": rl_stochastic_runs,
            "rl_stochastic_mean": rl_stochastic_mean,
            "comparison_greedy": comparison_greedy,
            "comparison_stochastic_mean": comparison_stochastic_mean,
        }

        report_path = os.path.join(report_dir, "comparison.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        _print_summary(report)
        print(f"Report saved to {report_path}")
    finally:
        try:
            if env.sumo.isLoaded():
                env.sumo.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()
