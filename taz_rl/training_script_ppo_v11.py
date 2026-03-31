import csv
import json
import math
import os
import random
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict

from libraries import constants
from libraries.classes.Planner import Planner
from libraries.classes.SumoSimulator import Simulator
from libraries.constants import EDGE_DATA_FILE_PATH, PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH
from libraries.utils.preprocessingUtils import generateEdgeDataFile
from taz_rl.rlenv.local_taz_env_v11 import SumoTazEnvV11
from taz_rl.training_script_ppo_v10 import (
    ActorCriticV10,
    ENV_V10_CONFIG,
    BASE_DEMAND,
    CLIP_RATIO,
    ENTROPY_COEF,
    ENTROPY_COEF_FINAL,
    ENTROPY_WARMUP_RATIO,
    GAMMA,
    LAMBDA,
    LEGACY_ROUTE_CACHE_ROOTS,
    LR,
    MINIBATCH_SIZE,
    PPO_UPDATE_EPOCHS,
    ROUTE_CACHE_ROOT,
    ROUTE_RANDOM_TRIP_SEED,
    ROUTE_SAMPLER_SEED,
    ROUTE_SAMPLER_THREADS,
    TARGET_KL,
    VALUE_COEF,
    HOURLY_DEMAND_PROFILE,
    _clone_state_dict_to_cpu,
    _move_optimizer_state_to_device,
    compute_gae,
    ppo_update,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


TRAIN_START_DATE = datetime(2024, 2, 1)
N_TRAIN_DAYS = 200
FOCUS_HOURS = [8]
TRAIN_ON_SAME_DAY = True
SINGLE_TAZ_ID = None
SINGLE_TAZ_WAITING_ONLY_REWARD = False

DEMAND_NOISE_RANGE = (1.0, 1.0)
GLOBAL_SEED = 42
REUSE_DETERMINISTIC_ROUTE_FILES = True

ACTION_BINS = tuple(float(x) for x in SumoTazEnvV11.ACTION_BIN_VALUES)

ENV_V11_CONFIG = dict(ENV_V10_CONFIG)
ENV_V11_CONFIG.update(
    coordination_distance=350.0,
    coordination_path_length=900.0,
    coordination_group_max_size=3,
    group_axis_align_only=True,
)

LOAD_MODEL = False
RUN_SUFFIX = f"_single_{SINGLE_TAZ_ID}" if SINGLE_TAZ_ID else ""
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, f"checkpoints_v11{RUN_SUFFIX}", "checkpoint_ppo_v11_final.pt")
RESUME_FROM_CHECKPOINT_EPISODE = True
START_EPISODE_OVERRIDE = None


def _masked_values(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = tensor[mask]
    if valid.numel() > 0:
        return valid
    return torch.zeros(1, dtype=tensor.dtype, device=tensor.device)


def _masked_mean_std(tensor: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    valid = _masked_values(tensor, mask)
    return float(valid.mean().item()), float(valid.std(unbiased=False).item())


def _count_discrete_actions(applied_actions: torch.Tensor, action_mask: torch.Tensor) -> dict:
    flat = _masked_values(applied_actions, action_mask).reshape(-1)
    return {
        "action_count_neg10": int((flat == -10.0).sum().item()),
        "action_count_zero": int((flat == 0.0).sum().item()),
        "action_count_pos10": int((flat == 10.0).sum().item()),
    }


def _build_action_summary_by_taz(
    selected_actions: torch.Tensor,
    applied_deltas: torch.Tensor,
    action_mask: torch.Tensor,
    taz_ids: list[str],
) -> dict:
    summary = {}
    for taz_idx, taz in enumerate(taz_ids):
        if action_mask.ndim == 3:
            valid_mask = action_mask[:, taz_idx, :].bool().reshape(-1)
        else:
            valid_mask = action_mask[taz_idx].bool().unsqueeze(0).expand(selected_actions.shape[0], -1).reshape(-1)

        selected_valid = selected_actions[:, taz_idx, :].reshape(-1)[valid_mask]
        delta_valid = applied_deltas[:, taz_idx, :].reshape(-1)[valid_mask]
        if selected_valid.numel() == 0:
            summary[taz] = {
                "avg_action_selected": 0.0,
                "avg_applied_duration_delta": 0.0,
                "applied_duration_nonzero_ratio": 0.0,
                "action_counts": {},
            }
            continue

        summary[taz] = {
            "avg_action_selected": float(selected_valid.mean().item()),
            "avg_applied_duration_delta": float(delta_valid.mean().item()),
            "applied_duration_nonzero_ratio": float((delta_valid.abs() > 1e-3).float().mean().item()),
            "action_counts": {
                "neg10": int((selected_valid == -10.0).sum().item()),
                "zero": int((selected_valid == 0.0).sum().item()),
                "pos10": int((selected_valid == 10.0).sum().item()),
            },
        }
    return summary


def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_route_cache_folder(root: str, simulation_date: str, timeslot_clean: str, total_cars_random: int) -> str:
    return os.path.join(
        root,
        simulation_date,
        timeslot_clean,
        f"count_{int(total_cars_random)}",
        f"tripseed_{ROUTE_RANDOM_TRIP_SEED}_sampleseed_{ROUTE_SAMPLER_SEED}_threads_{ROUTE_SAMPLER_THREADS}",
    )


def _route_files_ready(route_folder_path: str) -> bool:
    required = (
        "randomTrips.rou.xml",
        "trips.rou.xml",
        "generatedRoutes.rou.xml",
    )
    return all(os.path.exists(os.path.join(route_folder_path, name)) for name in required)


def _resolve_route_folder(simulation_date: str, timeslot_clean: str, total_cars_random: int) -> tuple[str, bool]:
    roots = [ROUTE_CACHE_ROOT] + list(LEGACY_ROUTE_CACHE_ROOTS)
    for root in roots:
        folder = _build_route_cache_folder(root, simulation_date, timeslot_clean, total_cars_random)
        if _route_files_ready(folder):
            return folder, False
    return _build_route_cache_folder(ROUTE_CACHE_ROOT, simulation_date, timeslot_clean, total_cars_random), True


def _sample_demand_noise(episode_idx: int) -> float:
    low, high = DEMAND_NOISE_RANGE
    if float(low) == float(high):
        return float(low)
    rng = random.Random(GLOBAL_SEED + int(episode_idx))
    return float(rng.uniform(low, high))


def _build_episode_schedule():
    episodes = []
    if TRAIN_ON_SAME_DAY:
        for episode_idx in range(N_TRAIN_DAYS):
            hour = FOCUS_HOURS[episode_idx % len(FOCUS_HOURS)]
            episodes.append((TRAIN_START_DATE, int(hour)))
        return episodes

    for day_offset in range(N_TRAIN_DAYS):
        day = TRAIN_START_DATE + timedelta(days=day_offset)
        for hour in FOCUS_HOURS:
            episodes.append((day, int(hour)))
    return episodes


def _maybe_load_checkpoint(policy, optim, scheduler):
    start_episode = 0
    if not LOAD_MODEL:
        return start_episode

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    policy.load_state_dict(checkpoint["model_state_dict"])
    if checkpoint.get("optimizer_state_dict") is not None:
        try:
            optim.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as exc:
            print(f"[WARN] Could not load optimizer state: {exc}")
    if checkpoint.get("scheduler_state_dict") is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as exc:
            print(f"[WARN] Could not load scheduler state: {exc}")

    if RESUME_FROM_CHECKPOINT_EPISODE:
        start_episode = int(checkpoint.get("episode", -1)) + 1
    if START_EPISODE_OVERRIDE is not None:
        start_episode = int(START_EPISODE_OVERRIDE)
    return start_episode


def _load_existing_history(json_path: str) -> list[dict]:
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _rewrite_history_csv(csv_path: str, csv_cols: list[str], history: list[dict]):
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_cols)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in csv_cols})


def _extract_terminal_penalty(row: dict) -> Optional[float]:
    if not bool(row.get("terminal_parse_ok", False)):
        return None
    try:
        penalty = float(row.get("terminal_penalty", ""))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(penalty):
        return None
    return penalty


def _prepare_history_files(csv_path: str, json_path: str, csv_cols: list[str], start_episode: int):
    history = _load_existing_history(json_path) if LOAD_MODEL else []
    history = [row for row in history if int(row.get("episode", -1)) < int(start_episode)]

    if history:
        _rewrite_history_csv(csv_path, csv_cols, history)
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=csv_cols).writeheader()

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    if history:
        scored_rows = [
            (penalty, row)
            for row in history
            for penalty in [_extract_terminal_penalty(row)]
            if penalty is not None
        ]
        if scored_rows:
            best_terminal_penalty, best_row = min(scored_rows, key=lambda item: item[0])
            best_episode = int(best_row.get("episode", -1))
        else:
            best_terminal_penalty = float("inf")
            best_episode = -1
    else:
        best_terminal_penalty = float("inf")
        best_episode = -1
    return history, best_terminal_penalty, best_episode


def _ensure_baseline_reference(env, sumo, hour: int, route_folder_path: str, cache: dict) -> tuple[Optional[float], dict, bool]:
    if TRAIN_ON_SAME_DAY:
        key = ("same_day", int(hour))
    else:
        key = (route_folder_path, int(hour))
    if key in cache:
        penalty, components = cache[key]
        return penalty, dict(components), False

    sumo.changeRouteFilePath(route_folder_path)
    sumo.changeTypePath(route_folder_path)
    penalty, components = env.run_reference_episode_no_agent(hour=hour)
    if not bool(components.get("parse_ok", False)):
        penalty = None
    cache[key] = (penalty, dict(components))
    return penalty, dict(components), True


def main():
    _set_global_seed(GLOBAL_SEED)

    sumo_standalone_dir = os.path.join(constants.SUMO_PATH, "standalone")
    log_file = os.path.join(sumo_standalone_dir, "command_log_v11.txt")
    sumo = Simulator(configurationPath=sumo_standalone_dir, logFile=log_file, tazTlsMapFile=constants.TAZ_FILE)
    planner = Planner(simulator=sumo)
    sumo.changeDetectorPath(detectorPath=constants.SUMO_NETWORK_PATH)

    selected_taz_ids = [str(SINGLE_TAZ_ID)] if SINGLE_TAZ_ID else None
    env_config = dict(ENV_V11_CONFIG)
    if SINGLE_TAZ_ID and SINGLE_TAZ_WAITING_ONLY_REWARD:
        env_config.update(
            waiting_reward_weight=0.0,
            emission_reward_weight=1.0,
            jam_reward_weight=0.0,
        )
    env = SumoTazEnvV11(
        sumoSimulator=sumo,
        stepSize=600,
        selected_taz_ids=selected_taz_ids,
        **env_config,
    )
    runtime_device = torch.device(env.device)
    taz_ids = env.get_taz_ids()
    tls_ids = env.get_tls_ids()
    control_groups_by_taz = env.get_control_groups_by_taz()
    control_group_counts = {
        taz: len(groups)
        for taz, groups in control_groups_by_taz.items()
    }
    num_taz = len(taz_ids)
    obs_dim = int(env.agent_obs_dim)
    act_dim = int(env.max_control_groups_per_taz)

    policy = ActorCriticV10(obs_dim, act_dim, ACTION_BINS).to(runtime_device)
    policy.train()
    optim = torch.optim.Adam(policy.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.7, patience=15, threshold=0.005, threshold_mode="rel", min_lr=1e-5
    )
    start_episode = _maybe_load_checkpoint(policy, optim, scheduler)
    _move_optimizer_state_to_device(optim, runtime_device)

    episode_schedule = _build_episode_schedule()
    n_episodes = len(episode_schedule)
    if start_episode >= n_episodes:
        print(f"[INFO] start_episode={start_episode} >= n_episodes={n_episodes}. Nothing to train.")
        return

    csv_path = os.path.join(SCRIPT_DIR, f"training_history_v11{RUN_SUFFIX}.csv")
    json_path = os.path.join(SCRIPT_DIR, f"training_history_v11{RUN_SUFFIX}.json")
    ckpt_dir = os.path.join(SCRIPT_DIR, f"checkpoints_v11{RUN_SUFFIX}")
    detail_dir = os.path.join(SCRIPT_DIR, f"training_details_v11{RUN_SUFFIX}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(detail_dir, exist_ok=True)

    csv_cols = [
        "episode", "date", "timeslot", "num_taz", "num_tls", "num_control_groups_max", "num_steps", "reward_basis",
        "episode_reward_mean", "episode_reward_std", "episode_reward_min", "episode_reward_max", "episode_reward_sum",
        "final_penalty_mean", "final_penalty_std", "final_penalty_min", "final_penalty_max",
        "avg_action_selected", "sample_action_std_selected",
        "action_count_neg10", "action_count_zero", "action_count_pos10",
        "avg_applied_duration_delta", "sample_applied_duration_delta_std", "applied_duration_nonzero_ratio", "action_enabled_ratio",
        "returns_mean", "returns_std", "avg_value_estimate",
        "learning_rate", "learning_rate_after_step",
        "ppo_total_loss", "ppo_policy_loss", "ppo_value_loss",
        "policy_clip_fraction", "approx_kl", "policy_entropy_mean",
        "ppo_early_stop", "ppo_epochs_performed", "ppo_epochs_planned",
        "baseline_parse_ok", "baseline_penalty",
        "terminal_parse_ok", "terminal_penalty", "terminal_bonus", "comparison_delta_penalty",
        "details_path",
    ]
    history, best_terminal_penalty, best_episode = _prepare_history_files(csv_path, json_path, csv_cols, start_episode)
    interrupted = False
    last_completed_episode = start_episode - 1
    baseline_cache = {}

    print(f"[INFO] v11 TAZ count: {len(taz_ids)} | IDs: {taz_ids}")
    print(f"[INFO] v11 TLS count: {len(tls_ids)}")
    print(f"[INFO] Observation dim per TAZ: {obs_dim} | control-group slots per TAZ: {act_dim}")
    print(f"[INFO] Control groups per TAZ: {control_group_counts}")
    print(
        f"[INFO] Coordination: distance={env_config['coordination_distance']} | "
        f"path_length={env_config['coordination_path_length']} | "
        f"group_max_size={env_config['coordination_group_max_size']} | "
        f"group_axis_align_only={env_config['group_axis_align_only']}"
    )
    print(
        f"[INFO] Reward weights: waiting={env_config['waiting_reward_weight']} | "
        f"emission={env_config['emission_reward_weight']} | "
        f"jam={env_config['jam_reward_weight']} | "
        f"waiting_time_memory={env_config['waiting_time_memory']} | "
        f"action_signal_threshold={env_config['action_signal_threshold']} | "
        f"terminal_bonus_weight={env_config['terminal_bonus_weight']}"
    )

    for episode_idx in range(start_episode, n_episodes):
        episode_num = episode_idx + 1
        day, hour = episode_schedule[episode_idx]
        simulation_date = day.strftime("%Y-%m-%d")
        timeslot = f"{hour:02d}:00-{(hour + 1):02d}:00"
        timeslot_clean = timeslot.replace(":", "-")
        print(f"\n[RL V11 EP {episode_num}/{n_episodes}] Date={simulation_date} Slot={timeslot}")

        total_cars = int(BASE_DEMAND * HOURLY_DEMAND_PROFILE[hour])
        noise = _sample_demand_noise(episode_idx)
        total_cars_random = int(total_cars * noise)
        route_folder_path, should_generate_routes = _resolve_route_folder(
            simulation_date=simulation_date,
            timeslot_clean=timeslot_clean,
            total_cars_random=total_cars_random,
        )
        os.makedirs(os.path.join(route_folder_path, "output"), exist_ok=True)

        if REUSE_DETERMINISTIC_ROUTE_FILES and not should_generate_routes:
            print(f"[ROUTE CACHE] RL V11 EP {episode_num}/{n_episodes} | {route_folder_path}")
        else:
            print(f"[ROUTE GEN] RL V11 EP {episode_num}/{n_episodes} | {route_folder_path}")
            generateEdgeDataFile(PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH, date=simulation_date, time_slot=timeslot)
            planner.scenarioGenerator.generateRoute(
                inputEdgePath=EDGE_DATA_FILE_PATH,
                timeSlot=timeslot_clean,
                totalCount=total_cars_random,
                custom=False,
                outputFolder=route_folder_path,
                randomTripSeed=ROUTE_RANDOM_TRIP_SEED,
                routeSamplerSeed=ROUTE_SAMPLER_SEED,
                routeSamplerThreads=ROUTE_SAMPLER_THREADS,
            )

        baseline_penalty, baseline_components, baseline_ran = _ensure_baseline_reference(
            env=env,
            sumo=sumo,
            hour=hour,
            route_folder_path=route_folder_path,
            cache=baseline_cache,
        )
        baseline_parse_ok = bool(baseline_components.get("parse_ok", False))
        if env_config["comparison_reward_enabled"] and baseline_penalty is not None:
            env.set_baseline_penalty(baseline_penalty, baseline_components)
        else:
            env.set_baseline_penalty(None)
        print(
            f"[BASELINE REF] RL V11 EP {episode_num}/{n_episodes} | "
            f"{'Computed' if baseline_ran else 'Reused'} | "
            f"parse_ok={baseline_parse_ok} | penalty={baseline_penalty if baseline_penalty is not None else 'n/a'}"
        )

        sumo.changeRouteFilePath(route_folder_path)
        sumo.changeTypePath(route_folder_path)
        env.set_episode_context(hour=hour)
        print(f"[RL RUN] RL V11 EP {episode_num}/{n_episodes} | Starting coordinated-group RL simulation.")
        td = env.reset()

        rewards_list = []
        values_list = []
        logps_list = []
        obs_list = []
        action_masks_list = []
        acts_list = []
        applied_actions_list = []
        applied_deltas_list = []
        dones_list = []
        episode_invalid = False

        while True:
            obs = td["observation"]
            action_mask = td["action_mask"].bool()
            if not torch.isfinite(obs).all():
                print("[WARN] Non-finite observation. Skipping episode.")
                episode_invalid = True
                break

            action_index, logp, value = policy.act(obs, action_mask)
            if not (torch.isfinite(logp).all() and torch.isfinite(value).all()):
                print("[WARN] Non-finite tensors in policy output. Skipping episode.")
                episode_invalid = True
                break

            try:
                step_td = env.step(TensorDict({"action": action_index}, batch_size=[], device=runtime_device))
            except KeyboardInterrupt:
                interrupted = True
                episode_invalid = True
                print("[WARN] Interrupted during RL rollout.")
                break

            next_td = step_td["next"] if "next" in step_td.keys() else step_td
            reward_vec = next_td["reward"].detach().clone()
            terminated = bool(next_td["terminated"].item())
            truncated = bool(next_td["truncated"].item())
            done_vec = torch.full_like(reward_vec, 1.0 if (terminated or truncated) else 0.0)

            if not (
                torch.isfinite(reward_vec).all()
                and torch.isfinite(next_td["applied_action"]).all()
                and torch.isfinite(next_td["applied_duration_delta"]).all()
            ):
                print("[WARN] Non-finite environment outputs. Skipping episode.")
                episode_invalid = True
                break

            obs_list.append(obs.detach().clone())
            action_masks_list.append(action_mask.detach().clone())
            acts_list.append(action_index.detach().clone())
            applied_actions_list.append(next_td["applied_action"].detach().clone())
            applied_deltas_list.append(next_td["applied_duration_delta"].detach().clone())
            logps_list.append(logp.detach().clone())
            values_list.append(value.detach().clone())
            rewards_list.append(reward_vec)
            dones_list.append(done_vec)

            if terminated or truncated:
                break
            td = next_td

        if interrupted:
            break
        if episode_invalid or len(rewards_list) == 0:
            try:
                if env.sumo.isLoaded():
                    env.sumo.end()
            except Exception:
                pass
            continue

        rewards_t = torch.stack(rewards_list)
        values_t = torch.stack(values_list)
        dones_t = torch.stack(dones_list)
        adv_t, ret_t = compute_gae(rewards_t, values_t, dones_t, gamma=GAMMA, lam=LAMBDA)

        batch_obs_3d = torch.stack(obs_list)
        batch_mask_3d = torch.stack(action_masks_list)
        batch_act_3d = torch.stack(acts_list)
        batch_applied_act_3d = torch.stack(applied_actions_list)
        batch_applied_delta_3d = torch.stack(applied_deltas_list)
        batch_logp_2d = torch.stack(logps_list)

        batch_obs = batch_obs_3d.reshape(-1, obs_dim)
        batch_action_mask = batch_mask_3d.reshape(-1, act_dim)
        batch_act = batch_act_3d.reshape(-1, act_dim)
        batch_logp = batch_logp_2d.reshape(-1)
        batch_adv = adv_t.reshape(-1)
        batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std(unbiased=False) + 1e-8)
        batch_ret = ret_t.reshape(-1)
        rollout_policy_state = _clone_state_dict_to_cpu(policy.state_dict())

        progress = episode_idx / max(n_episodes - 1, 1)
        if progress <= ENTROPY_WARMUP_RATIO:
            entropy_coef_now = ENTROPY_COEF
        else:
            decay_progress = (progress - ENTROPY_WARMUP_RATIO) / max(1.0 - ENTROPY_WARMUP_RATIO, 1e-8)
            entropy_coef_now = ENTROPY_COEF + (ENTROPY_COEF_FINAL - ENTROPY_COEF) * decay_progress

        stats = ppo_update(
            policy,
            optim,
            batch_obs,
            batch_action_mask,
            batch_act,
            batch_logp,
            batch_adv,
            batch_ret,
            clip_ratio=CLIP_RATIO,
            ppo_epochs=PPO_UPDATE_EPOCHS,
            minibatch_size=MINIBATCH_SIZE,
            entropy_coef=float(entropy_coef_now),
            value_coef=VALUE_COEF,
            target_kl=TARGET_KL,
        )

        lr_before_step = float(optim.param_groups[0]["lr"])
        episode_reward_by_taz_tensor = rewards_t.sum(dim=0).cpu()
        reward_mean = float(episode_reward_by_taz_tensor.mean().item())
        scheduler.step(reward_mean)
        lr_after_step = float(optim.param_groups[0]["lr"])

        reward_components = dict(getattr(env, "last_reward_components", {}) or {})
        dense_reward_by_taz = dict(reward_components.get("dense_reward_by_taz", {}) or {})
        penalty_by_taz = dict(reward_components.get("penalty_by_taz", {}) or {})
        terminal_reward = dict(reward_components.get("terminal_reward", {}) or {})
        terminal_parse_ok = bool(terminal_reward.get("parse_ok", False))
        terminal_penalty = float(terminal_reward.get("penalty", 0.0)) if terminal_parse_ok else ""
        terminal_reward_value = float(reward_components.get("terminal_bonus", 0.0))
        comparison_delta_penalty = (
            float(baseline_penalty - terminal_penalty)
            if (baseline_penalty is not None and terminal_parse_ok)
            else ""
        )

        action_mask_stats = batch_mask_3d.bool()
        avg_action_selected, std_action_selected = _masked_mean_std(batch_applied_act_3d, action_mask_stats)
        avg_applied_delta, std_applied_delta = _masked_mean_std(batch_applied_delta_3d, action_mask_stats)
        nonzero_ratio = float((_masked_values(batch_applied_delta_3d.abs(), action_mask_stats) > 1e-3).float().mean().item())
        action_enabled_ratio = float(action_mask_stats.float().mean().item())
        action_counts = _count_discrete_actions(batch_applied_act_3d, action_mask_stats)
        action_summary_by_taz = _build_action_summary_by_taz(
            selected_actions=batch_applied_act_3d,
            applied_deltas=batch_applied_delta_3d,
            action_mask=batch_mask_3d,
            taz_ids=taz_ids,
        )

        detail_path = os.path.join(detail_dir, f"episode_{episode_idx:04d}.json")
        with open(detail_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "episode": int(episode_idx),
                    "episode_num": int(episode_num),
                    "date": simulation_date,
                    "timeslot": timeslot,
                    "num_steps": int(len(rewards_list)),
                    "reward_by_taz": {
                        taz: float(episode_reward_by_taz_tensor[idx].item())
                        for idx, taz in enumerate(taz_ids)
                    },
                    "dense_reward_by_taz": dense_reward_by_taz,
                    "final_penalty_by_taz": penalty_by_taz,
                    "delta_penalty_by_taz": dict(reward_components.get("delta_penalty_by_taz", {}) or {}),
                    "baseline_reference_components": baseline_components,
                    "terminal_reward_components": terminal_reward,
                    "control_groups_by_taz": control_groups_by_taz,
                    "group_action_details": dict(reward_components.get("group_action_details", {}) or {}),
                    "group_signal_by_taz": dict(reward_components.get("group_signal_by_taz", {}) or {}),
                    "action_summary_by_taz": action_summary_by_taz,
                },
                handle,
                indent=2,
            )

        penalty_vector = torch.tensor([float(penalty_by_taz.get(taz, 0.0)) for taz in taz_ids], dtype=torch.float32)
        row = {
            "episode": int(episode_idx),
            "date": simulation_date,
            "timeslot": timeslot,
            "num_taz": int(num_taz),
            "num_tls": int(len(tls_ids)),
            "num_control_groups_max": int(act_dim),
            "num_steps": int(len(rewards_list)),
            "reward_basis": str(reward_components.get("reward_basis", "taz_step_abs+delta_waiting_with_terminal_baseline_compare")),
            "episode_reward_mean": reward_mean,
            "episode_reward_std": float(episode_reward_by_taz_tensor.std(unbiased=False).item()),
            "episode_reward_min": float(episode_reward_by_taz_tensor.min().item()),
            "episode_reward_max": float(episode_reward_by_taz_tensor.max().item()),
            "episode_reward_sum": float(episode_reward_by_taz_tensor.sum().item()),
            "final_penalty_mean": float(penalty_vector.mean().item()),
            "final_penalty_std": float(penalty_vector.std(unbiased=False).item()),
            "final_penalty_min": float(penalty_vector.min().item()),
            "final_penalty_max": float(penalty_vector.max().item()),
            "avg_action_selected": avg_action_selected,
            "sample_action_std_selected": std_action_selected,
            **action_counts,
            "avg_applied_duration_delta": avg_applied_delta,
            "sample_applied_duration_delta_std": std_applied_delta,
            "applied_duration_nonzero_ratio": nonzero_ratio,
            "action_enabled_ratio": action_enabled_ratio,
            "returns_mean": float(batch_ret.mean().item()),
            "returns_std": float(batch_ret.std(unbiased=False).item()),
            "avg_value_estimate": float(values_t.mean().item()),
            "learning_rate": lr_before_step,
            "learning_rate_after_step": lr_after_step,
            "ppo_total_loss": float(stats.get("total_loss", 0.0)),
            "ppo_policy_loss": float(stats.get("policy_loss", 0.0)),
            "ppo_value_loss": float(stats.get("value_loss", 0.0)),
            "policy_clip_fraction": float(stats.get("clip_fraction", 0.0)),
            "approx_kl": float(stats.get("approx_kl", 0.0)),
            "policy_entropy_mean": float(stats.get("entropy_mean", 0.0)),
            "ppo_early_stop": bool(stats.get("early_stop", False)),
            "ppo_epochs_performed": int(stats.get("epochs_performed", 0)),
            "ppo_epochs_planned": int(stats.get("epochs_planned", PPO_UPDATE_EPOCHS)),
            "baseline_parse_ok": bool(baseline_parse_ok),
            "baseline_penalty": float(baseline_penalty) if baseline_penalty is not None else "",
            "terminal_parse_ok": terminal_parse_ok,
            "terminal_penalty": terminal_penalty,
            "terminal_bonus": terminal_reward_value,
            "comparison_delta_penalty": comparison_delta_penalty,
            "details_path": detail_path,
        }
        history.append(row)
        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_cols)
            writer.writerow({key: row.get(key, "") for key in csv_cols})
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        checkpoint_payload = {
            "episode": int(episode_idx),
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "reward_mean": reward_mean,
            "terminal_penalty": float(terminal_penalty) if terminal_parse_ok else None,
            "checkpoint_selection_metric": "lowest_terminal_penalty",
            "env_config": env_config,
            "focus_hours": FOCUS_HOURS,
            "taz_ids": taz_ids,
            "tls_ids": tls_ids,
            "tls_count": len(tls_ids),
            "max_control_groups_per_taz": act_dim,
            "control_groups_by_taz": control_groups_by_taz,
            "action_bins": ACTION_BINS,
            "single_taz_id": SINGLE_TAZ_ID,
            "ppo": stats,
            "interrupted": False,
            "policy_stage": "post_update_training_policy",
        }
        torch.save(checkpoint_payload, os.path.join(ckpt_dir, f"checkpoint_ppo_v11_episode{episode_idx}.pt"))

        if terminal_parse_ok and float(terminal_penalty) < best_terminal_penalty:
            best_terminal_penalty = float(terminal_penalty)
            best_episode = int(episode_idx)
            best_payload = dict(checkpoint_payload)
            best_payload["model_state_dict"] = rollout_policy_state
            best_payload["optimizer_state_dict"] = None
            best_payload["scheduler_state_dict"] = None
            best_payload["policy_stage"] = "pre_update_rollout_policy"
            best_payload["is_best"] = True
            torch.save(best_payload, os.path.join(ckpt_dir, "checkpoint_ppo_v11_best.pt"))

        last_completed_episode = episode_idx
        best_taz = max(penalty_by_taz, key=lambda key: dense_reward_by_taz.get(key, float("-inf")))
        worst_taz = max(penalty_by_taz, key=lambda key: penalty_by_taz[key])
        msg = (
            f"[RL V11 RESULT] EP {episode_num}/{n_episodes} | "
            f"RewardMean {row['episode_reward_mean']:.4f} | RewardStd {row['episode_reward_std']:.4f} | "
            f"PenaltyMean {row['final_penalty_mean']:.4f} | ActionMean {row['avg_action_selected']:.2f} | "
            f"AppliedNZ {row['applied_duration_nonzero_ratio']:.2f} | "
            f"ActionEnabled {row['action_enabled_ratio']:.2f} | KL {row['approx_kl']:.4f} | "
            f"Entropy {row['policy_entropy_mean']:.4f} | "
            f"BestDenseTAZ {best_taz}={dense_reward_by_taz.get(best_taz, 0.0):.4f} | "
            f"WorstPenaltyTAZ {worst_taz}={penalty_by_taz[worst_taz]:.4f} | "
            f"TerminalReward {terminal_reward_value:.4f}"
        )
        if comparison_delta_penalty != "":
            msg += f" | VsBaseline {comparison_delta_penalty:.4f}"
        print(msg)

    try:
        env.sumo.end()
    except Exception:
        pass

    torch.save(
        {
            "episode": int(last_completed_episode),
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history_len": len(history),
            "env_config": env_config,
            "focus_hours": FOCUS_HOURS,
            "taz_ids": taz_ids,
            "tls_ids": tls_ids,
            "tls_count": len(tls_ids),
            "action_bins": ACTION_BINS,
            "max_control_groups_per_taz": act_dim,
            "control_groups_by_taz": control_groups_by_taz,
            "single_taz_id": SINGLE_TAZ_ID,
            "best_episode": best_episode,
            "best_terminal_penalty": best_terminal_penalty if math.isfinite(best_terminal_penalty) else None,
            "checkpoint_selection_metric": "lowest_terminal_penalty",
            "interrupted": bool(interrupted),
        },
        os.path.join(ckpt_dir, "checkpoint_ppo_v11_final.pt"),
    )

    if interrupted:
        print("[INFO] v11 training interrupted. Partial checkpoint saved.")
    else:
        print("[INFO] v11 training complete.")


if __name__ == "__main__":
    main()
