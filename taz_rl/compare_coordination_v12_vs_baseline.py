import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import fields

import numpy as np
import torch
from tensordict import TensorDict

# Allow direct execution with `python taz_rl/compare_coordination_v12_vs_baseline.py`
# from the repository root, without requiring users to export PYTHONPATH.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from libraries import constants
from libraries.classes.Planner import Planner
from libraries.classes.SumoSimulator import Simulator
from libraries.constants import EDGE_DATA_FILE_PATH, PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH
from libraries.utils.preprocessingUtils import generateEdgeDataFile
from taz_rl.global_coordination_v12 import (
    COORDINATION_PRICE_BINS,
    CoordinationObservationConfig,
    CoordinationRewardConfig,
    augment_local_observation_with_prices,
    build_coordination_observation,
    build_taz_adjacency_matrix,
    compute_coordination_step_rewards,
    extract_taz_features,
    summarize_step_actions,
)
from taz_rl.rlenv.local_taz_env_v12_coordination import SumoTazEnvV12Coordination
from taz_rl.training_script_coordination_v12 import (
    LOCAL_GLOBAL_CONTEXT_DIM,
    TAZ_ADJACENCY_DISTANCE_THRESHOLD,
    TAZ_ADJACENCY_FALLBACK_K,
)
from taz_rl.training_script_ppo_v11 import (
    ACTION_BINS,
    BASE_DEMAND,
    ENV_V11_CONFIG,
    HOURLY_DEMAND_PROFILE,
    ROUTE_RANDOM_TRIP_SEED,
    ROUTE_SAMPLER_SEED,
    ROUTE_SAMPLER_THREADS,
    ActorCriticV10,
    _resolve_route_folder,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_PATH = os.path.join(
    SCRIPT_DIR,
    "checkpoints_coordination_v12",
    "checkpoint_coordination_v12_best.pt",
)
REPORT_ROOT = os.path.join(SCRIPT_DIR, "evaluation_reports_coordination_v12")


TRIPINFO_EMISSION_ATTRS = {
    "total_co2": "CO2_abs",
    "total_nox": "NOx_abs",
    "total_fuel": "fuel_abs",
    "total_co": "CO_abs",
}


# The comparison uses tripinfo totals because they are already accumulated per
# vehicle by SUMO. Summing these fields avoids double-counting timestep samples
# from emission-output.xml.
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


def _parse_tripinfo_emission_totals(tripinfo_path: str) -> dict:
    root = _safe_parse_xml_root(tripinfo_path)
    totals = {metric_name: 0.0 for metric_name in TRIPINFO_EMISSION_ATTRS}
    if root is None:
        return {
            **totals,
            "tripinfo_count": 0,
            "emission_trip_count": 0,
            "ok": False,
            "path": tripinfo_path,
        }

    seen_metrics = {metric_name: False for metric_name in TRIPINFO_EMISSION_ATTRS}
    tripinfo_count = 0
    emission_trip_count = 0
    for trip in root.findall("tripinfo"):
        tripinfo_count += 1
        emissions = trip.find("emissions")
        if emissions is None:
            continue

        saw_emission_value = False
        for metric_name, attr_name in TRIPINFO_EMISSION_ATTRS.items():
            value = emissions.get(attr_name)
            if value is None:
                continue
            totals[metric_name] += _safe_float(value, 0.0)
            seen_metrics[metric_name] = True
            saw_emission_value = True

        if saw_emission_value:
            emission_trip_count += 1

    return {
        **{metric_name: float(value) for metric_name, value in totals.items()},
        "tripinfo_count": int(tripinfo_count),
        "emission_trip_count": int(emission_trip_count),
        "ok": bool(emission_trip_count > 0 and all(seen_metrics.values())),
        "path": tripinfo_path,
    }


def _collect_run_metrics(run_root: str, waiting_components: dict | None = None) -> dict:
    output_dir = os.path.join(run_root, "output")
    waiting = dict(waiting_components or {})
    tripinfo_path = str(waiting.get("tripinfo_path") or os.path.join(output_dir, "tripinfos.xml"))
    emissions = _parse_tripinfo_emission_totals(tripinfo_path)

    return {
        "trip_count": int(waiting.get("trip_count", 0)),
        "tripinfo_count": int(emissions["tripinfo_count"]),
        "emission_trip_count": int(emissions["emission_trip_count"]),
        "avg_waiting_time": float(waiting.get("avg_waiting_time", 0.0)),
        "total_waiting_time": float(waiting.get("total_waiting_time", 0.0)),
        "waiting_metric_source": str(waiting.get("waiting_metric_source", "unknown")),
        "emission_metric_source": "tripinfo_emissions_abs",
        "waiting_time_by_taz": dict(waiting.get("waiting_time_by_taz", {}) or {}),
        "vehicle_count_by_taz": dict(waiting.get("vehicle_count_by_taz", {}) or {}),
        "total_co2": float(emissions["total_co2"]),
        "total_nox": float(emissions["total_nox"]),
        "total_fuel": float(emissions["total_fuel"]),
        "total_co": float(emissions["total_co"]),
        "ok": bool(waiting.get("parse_ok", False) and emissions["ok"]),
        "tripinfo_path": tripinfo_path,
        "emission_path": tripinfo_path,
        "emission_output_path": os.path.join(output_dir, "emission-output.xml"),
    }


def _load_dataclass_config(raw: dict | None, config_cls):
    raw = dict(raw or {})
    valid_keys = {field.name for field in fields(config_cls)}
    return config_cls(**{key: raw[key] for key in raw if key in valid_keys})


def _build_global_context_tensor(
    hour: int,
    baseline_penalty: float | None,
    device: torch.device,
) -> torch.Tensor:
    # Keep the same compact context used during v12 training: hour, demand,
    # baseline-enabled flag, and baseline terminal penalty.
    total_cars = int(BASE_DEMAND * HOURLY_DEMAND_PROFILE[hour])
    baseline_value = 0.0 if baseline_penalty is None else float(np.clip(float(baseline_penalty), -5.0, 5.0))
    return torch.tensor(
        [
            float(hour) / 23.0,
            float(total_cars) / max(float(BASE_DEMAND), 1.0),
            1.0,
            baseline_value,
        ],
        dtype=torch.float32,
        device=device,
    )


def _build_coordination_bundle(checkpoint: dict, env: SumoTazEnvV12Coordination, adjacency_matrix: torch.Tensor) -> dict:
    # Rebuild both policies from checkpoint metadata so the evaluator stays
    # compatible with future observation/reward configuration changes.
    runtime_device = torch.device(env.device)
    taz_ids = env.get_taz_ids()
    num_taz = len(taz_ids)
    obs_config = _load_dataclass_config(
        checkpoint.get("coordination_observation_config"),
        CoordinationObservationConfig,
    )
    reward_config = _load_dataclass_config(
        checkpoint.get("coordination_reward_config"),
        CoordinationRewardConfig,
    )
    price_bins = tuple(checkpoint.get("coordination_price_bins", COORDINATION_PRICE_BINS))
    dummy_taz_features = torch.zeros((num_taz, env.per_taz_feature_dim), dtype=torch.float32, device=runtime_device)
    dummy_flow = torch.zeros((num_taz, num_taz), dtype=torch.float32, device=runtime_device)
    dummy_prices = torch.zeros(num_taz, dtype=torch.float32, device=runtime_device)
    dummy_context = torch.zeros(4, dtype=torch.float32, device=runtime_device)
    global_obs_dim = int(
        build_coordination_observation(
            dummy_taz_features,
            adjacency_matrix,
            previous_flow_matrix=dummy_flow,
            previous_prices=dummy_prices,
            previous_action_summary=None,
            taz_ids=taz_ids,
            extra_context=dummy_context,
            config=obs_config,
        ).shape[-1]
    )

    local_obs_dim = int(checkpoint.get("augmented_obs_dim", env.agent_obs_dim + LOCAL_GLOBAL_CONTEXT_DIM))
    local_policy = ActorCriticV10(local_obs_dim, env.max_control_groups_per_taz, ACTION_BINS).to(runtime_device)
    local_policy.load_state_dict(checkpoint["local_model_state_dict"])
    local_policy.eval()

    global_policy = ActorCriticV10(global_obs_dim, num_taz, price_bins).to(runtime_device)
    global_policy.load_state_dict(checkpoint["global_model_state_dict"])
    global_policy.eval()

    return {
        "local_policy": local_policy,
        "global_policy": global_policy,
        "observation_config": obs_config,
        "reward_config": reward_config,
        "price_bins": price_bins,
        "global_obs_dim": int(global_obs_dim),
        "local_obs_dim": int(local_obs_dim),
    }


def _matrix_to_nested_dict(matrix: torch.Tensor, taz_ids: list[str]) -> dict:
    matrix_cpu = matrix.detach().cpu()
    return {
        src_taz: {
            dst_taz: float(matrix_cpu[src_idx, dst_idx].item())
            for dst_idx, dst_taz in enumerate(taz_ids)
            if float(matrix_cpu[src_idx, dst_idx].item()) > 0.0
        }
        for src_idx, src_taz in enumerate(taz_ids)
    }


def _run_coordination_episode(
    env: SumoTazEnvV12Coordination,
    bundle: dict,
    adjacency_matrix: torch.Tensor,
    hour: int,
    deterministic: bool,
    baseline_penalty: float | None = None,
) -> dict:
    runtime_device = torch.device(env.device)
    taz_ids = env.get_taz_ids()
    num_taz = len(taz_ids)
    local_policy = bundle["local_policy"]
    global_policy = bundle["global_policy"]
    obs_config = bundle["observation_config"]
    reward_config = bundle["reward_config"]

    env.set_baseline_penalty(None)
    env.set_episode_context(hour=hour)
    td = env.reset()

    previous_flow_matrix = torch.zeros((num_taz, num_taz), dtype=torch.float32, device=runtime_device)
    previous_prices = torch.zeros(num_taz, dtype=torch.float32, device=runtime_device)
    previous_action_summary = None
    step_prices = []
    step_diagnostics = []
    step_flows = []

    with torch.no_grad():
        while True:
            # Global policy first emits one coordination price per TAZ. The
            # local policy receives these prices appended to its local state.
            local_obs = td["observation"]
            action_mask = td["action_mask"].bool()
            global_obs = build_coordination_observation(
                extract_taz_features(local_obs, env),
                adjacency_matrix,
                previous_flow_matrix=previous_flow_matrix,
                previous_prices=previous_prices,
                previous_action_summary=previous_action_summary,
                taz_ids=taz_ids,
                extra_context=_build_global_context_tensor(hour, baseline_penalty, runtime_device),
                config=obs_config,
            )
            global_action_mask = torch.ones((1, num_taz), dtype=torch.bool, device=runtime_device)
            global_action_index, _, _ = global_policy.act(
                global_obs,
                global_action_mask,
                deterministic=deterministic,
            )
            price_values = global_policy.action_values(global_action_index).squeeze(0)
            augmented_obs = augment_local_observation_with_prices(local_obs, price_values, adjacency_matrix)
            action_index, _, _ = local_policy.act(augmented_obs, action_mask, deterministic=deterministic)

            step_td = env.step(TensorDict({"action": action_index}, batch_size=[], device=runtime_device))
            next_td = step_td["next"] if "next" in step_td.keys() else step_td
            reward_components = dict(getattr(env, "last_reward_components", {}) or {})
            terminal_reward = dict(reward_components.get("terminal_reward", {}) or {})
            terminated = bool(next_td["terminated"].item())
            truncated = bool(next_td["truncated"].item())
            done = terminated or truncated
            terminal_penalty = (
                float(terminal_reward.get("penalty", 0.0))
                if done and bool(terminal_reward.get("parse_ok", False))
                else None
            )

            flow_matrix = torch.tensor(env.get_last_step_flow_matrix(), dtype=torch.float32, device=runtime_device)
            current_action_summary = summarize_step_actions(
                taz_ids=taz_ids,
                applied_actions=next_td["applied_action"],
                applied_deltas=next_td["applied_duration_delta"],
                action_mask=action_mask,
            )
            _, global_reward_value, coordination_diag = compute_coordination_step_rewards(
                taz_ids=taz_ids,
                price_values=price_values,
                reward_components=reward_components,
                flow_matrix=flow_matrix,
                action_summary_by_taz=current_action_summary,
                baseline_penalty=baseline_penalty if done else None,
                terminal_penalty=terminal_penalty if done else None,
                config=reward_config,
            )
            coordination_diag["global_reward"] = float(global_reward_value)

            # Store the coordination trace in the JSON report. This is useful
            # for checking whether one TAZ is exporting congestion to neighbors.
            step_prices.append({
                taz: float(price_values[idx].item())
                for idx, taz in enumerate(taz_ids)
            })
            step_diagnostics.append(coordination_diag)
            step_flows.append(_matrix_to_nested_dict(flow_matrix, taz_ids))

            previous_flow_matrix = flow_matrix
            previous_prices = price_values.detach()
            previous_action_summary = current_action_summary

            if done:
                break
            td = next_td

    return {
        "terminal_reward": dict((env.last_reward_components or {}).get("terminal_reward", {}) or {}),
        "group_action_details": dict((env.last_reward_components or {}).get("group_action_details", {}) or {}),
        "group_signal_by_taz": dict((env.last_reward_components or {}).get("group_signal_by_taz", {}) or {}),
        "coordination_rollout": {
            "enabled": True,
            "affects_rollout_directly": True,
            "price_by_step": step_prices,
            "flow_by_step": step_flows,
            "step_diagnostics": step_diagnostics,
            "episode_flow_by_taz": env.get_coordination_flow_diagnostics().get("episode_flow_by_taz", {}),
        },
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
        "total_fuel": _metric_comparison(baseline_metrics["total_fuel"], rl_metrics["total_fuel"]),
        "total_co": _metric_comparison(baseline_metrics["total_co"], rl_metrics["total_co"]),
    }


def _aggregate_metrics(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {
            "run_count": 0,
            "trip_count": 0.0,
            "tripinfo_count": 0.0,
            "emission_trip_count": 0.0,
            "avg_waiting_time": 0.0,
            "total_waiting_time": 0.0,
            "total_co2": 0.0,
            "total_nox": 0.0,
            "total_fuel": 0.0,
            "total_co": 0.0,
            "ok": False,
            "waiting_metric_source": "none",
            "emission_metric_source": "none",
            "run_dirs": [],
        }

    def _mean(key: str) -> float:
        return float(np.mean([_safe_float(metrics.get(key), 0.0) for metrics in metrics_list]))

    sources = {str(metrics.get("waiting_metric_source", "unknown")) for metrics in metrics_list}
    emission_sources = {str(metrics.get("emission_metric_source", "unknown")) for metrics in metrics_list}
    return {
        "run_count": int(len(metrics_list)),
        "trip_count": _mean("trip_count"),
        "tripinfo_count": _mean("tripinfo_count"),
        "emission_trip_count": _mean("emission_trip_count"),
        "avg_waiting_time": _mean("avg_waiting_time"),
        "total_waiting_time": _mean("total_waiting_time"),
        "total_co2": _mean("total_co2"),
        "total_nox": _mean("total_nox"),
        "total_fuel": _mean("total_fuel"),
        "total_co": _mean("total_co"),
        "ok": bool(all(bool(metrics.get("ok", False)) for metrics in metrics_list)),
        "waiting_metric_source": next(iter(sources)) if len(sources) == 1 else "mixed",
        "emission_metric_source": next(iter(emission_sources)) if len(emission_sources) == 1 else "mixed",
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
    parser = argparse.ArgumentParser(
        description="Compare a no-agent baseline against a v12 coordination RL checkpoint."
    )
    parser.add_argument("--date", required=True, help="Simulation date in YYYY-MM-DD format.")
    parser.add_argument("--timeslot", required=True, help="Timeslot in HH:MM-HH:MM format, for example 08:00-09:00.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH, help="Path to the coordination v12 checkpoint.")
    parser.add_argument("--report-root", default=REPORT_ROOT, help="Root folder where evaluation outputs and report are saved.")
    parser.add_argument("--stochastic-runs", type=int, default=1, help="Number of sampled-policy evaluation runs to average.")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.stochastic_runs < 1:
        raise ValueError("--stochastic-runs must be >= 1")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("local_model_state_dict") is None or checkpoint.get("global_model_state_dict") is None:
        raise KeyError("Coordination checkpoint must contain local_model_state_dict and global_model_state_dict.")

    checkpoint_env_config = dict(checkpoint.get("env_config", ENV_V11_CONFIG))
    checkpoint_taz_ids = [str(taz) for taz in checkpoint.get("taz_ids", [])]
    selected_taz_ids = checkpoint_taz_ids if len(checkpoint_taz_ids) == 1 else None

    hour, timeslot_clean = _parse_timeslot(args.timeslot)
    report_dir = os.path.join(os.path.abspath(args.report_root), args.date, timeslot_clean)
    baseline_run_dir = os.path.join(report_dir, "baseline")
    rl_greedy_run_dir = os.path.join(report_dir, "rl_coordination_greedy")
    rl_stochastic_root_dir = os.path.join(report_dir, "rl_coordination_stochastic")
    os.makedirs(os.path.join(baseline_run_dir, "output"), exist_ok=True)
    os.makedirs(os.path.join(rl_greedy_run_dir, "output"), exist_ok=True)
    os.makedirs(rl_stochastic_root_dir, exist_ok=True)

    sumo_standalone_dir = os.path.join(constants.SUMO_PATH, "standalone")
    log_file = os.path.join(sumo_standalone_dir, "compare_coordination_v12_vs_baseline.log")
    sumo = Simulator(configurationPath=sumo_standalone_dir, logFile=log_file, tazTlsMapFile=constants.TAZ_FILE)
    planner = Planner(simulator=sumo)
    sumo.changeDetectorPath(detectorPath=constants.SUMO_NETWORK_PATH)
    env = SumoTazEnvV12Coordination(
        sumoSimulator=sumo,
        stepSize=600,
        selected_taz_ids=selected_taz_ids,
        **checkpoint_env_config,
    )

    try:
        route_folder_path = _prepare_route_folder(args.date, args.timeslot, planner)
        runtime_device = torch.device(env.device)
        # Adjacency is inferred from the SUMO TAZ polygons and reused by both
        # the global observation builder and local observation augmentation.
        adjacency_matrix, adjacency_metadata = build_taz_adjacency_matrix(
            taz_ids=env.get_taz_ids(),
            taz_additional_path=constants.TAZ_ADDITIONAL_FILE_PATH,
            distance_threshold=float(
                checkpoint.get("adjacency_metadata", {}).get(
                    "distance_threshold",
                    TAZ_ADJACENCY_DISTANCE_THRESHOLD,
                )
            ),
            fallback_k=int(
                checkpoint.get("adjacency_metadata", {}).get(
                    "fallback_k",
                    TAZ_ADJACENCY_FALLBACK_K,
                )
            ),
            device=runtime_device,
        )
        bundle = _build_coordination_bundle(checkpoint, env, adjacency_matrix)

        print("[BASELINE] Running no-agent baseline evaluation.")
        sumo.changeRouteFilePath(route_folder_path)
        sumo.changeTypePath(baseline_run_dir)
        _, baseline_components = env.run_reference_episode_no_agent(hour=hour)
        baseline_metrics = _collect_run_metrics(baseline_run_dir, waiting_components=baseline_components)
        baseline_metrics["run_dir"] = baseline_run_dir

        baseline_penalty = (
            float(baseline_components.get("penalty", 0.0))
            if baseline_components.get("parse_ok", False)
            else None
        )

        # The greedy rollout is the main reproducible comparison point. The
        # stochastic runs are optional and show sampled-policy variability.
        print("[COORDINATION GREEDY] Running deterministic v12 coordination policy evaluation.")
        sumo.changeRouteFilePath(route_folder_path)
        sumo.changeTypePath(rl_greedy_run_dir)
        rl_greedy_rollout = _run_coordination_episode(
            env,
            bundle,
            adjacency_matrix,
            hour=hour,
            deterministic=True,
            baseline_penalty=baseline_penalty,
        )
        rl_greedy_metrics = _collect_run_metrics(rl_greedy_run_dir, waiting_components=rl_greedy_rollout["terminal_reward"])
        rl_greedy_metrics["run_dir"] = rl_greedy_run_dir
        rl_greedy_metrics["group_action_details"] = rl_greedy_rollout["group_action_details"]
        rl_greedy_metrics["group_signal_by_taz"] = rl_greedy_rollout["group_signal_by_taz"]
        rl_greedy_metrics["coordination_rollout"] = rl_greedy_rollout["coordination_rollout"]

        rl_stochastic_runs = []
        for run_idx in range(args.stochastic_runs):
            run_dir = os.path.join(rl_stochastic_root_dir, f"run_{run_idx + 1:02d}")
            os.makedirs(os.path.join(run_dir, "output"), exist_ok=True)
            print(f"[COORDINATION STOCHASTIC] Run {run_idx + 1}/{args.stochastic_runs}.")
            sumo.changeRouteFilePath(route_folder_path)
            sumo.changeTypePath(run_dir)
            rl_rollout = _run_coordination_episode(
                env,
                bundle,
                adjacency_matrix,
                hour=hour,
                deterministic=False,
                baseline_penalty=baseline_penalty,
            )
            rl_metrics = _collect_run_metrics(run_dir, waiting_components=rl_rollout["terminal_reward"])
            rl_metrics["run_dir"] = run_dir
            rl_metrics["run_index"] = int(run_idx + 1)
            rl_metrics["group_action_details"] = rl_rollout["group_action_details"]
            rl_metrics["group_signal_by_taz"] = rl_rollout["group_signal_by_taz"]
            rl_metrics["coordination_rollout"] = rl_rollout["coordination_rollout"]
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
            "training_mode": checkpoint.get("training_mode"),
            "route_folder_path": route_folder_path,
            "taz_ids": env.get_taz_ids(),
            "tls_ids": env.get_tls_ids(),
            "control_groups_by_taz": checkpoint.get("control_groups_by_taz"),
            "coordination_price_bins": list(bundle["price_bins"]),
            "coordination_observation_config": vars(bundle["observation_config"]),
            "coordination_reward_config": vars(bundle["reward_config"]),
            "adjacency_metadata": adjacency_metadata,
            "global_obs_dim": int(bundle["global_obs_dim"]),
            "local_obs_dim": int(bundle["local_obs_dim"]),
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

        report_path = os.path.join(report_dir, "comparison_coordination_v12.json")
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
