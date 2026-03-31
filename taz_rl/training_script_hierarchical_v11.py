import csv
import json
import math
import os
from typing import Optional

import torch
from tensordict import TensorDict

from libraries import constants
from libraries.classes.Planner import Planner
from libraries.classes.SumoSimulator import Simulator
from libraries.constants import EDGE_DATA_FILE_PATH, PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH
from libraries.utils.preprocessingUtils import generateEdgeDataFile
from taz_rl.global_agent_v10 import (
    GLOBAL_PRIORITY_BINS,
    GlobalObservationConfig,
    GlobalRewardConfig,
    build_global_action_mask,
    build_global_observation,
    compute_priority_rewards,
)
from taz_rl.rlenv.local_taz_env_v11 import SumoTazEnvV11
from taz_rl.training_script_ppo_v11 import (
    ACTION_BINS,
    ActorCriticV10,
    BASE_DEMAND,
    CLIP_RATIO,
    ENTROPY_COEF,
    ENTROPY_COEF_FINAL,
    ENTROPY_WARMUP_RATIO,
    ENV_V11_CONFIG,
    FOCUS_HOURS,
    GAMMA,
    GLOBAL_SEED,
    HOURLY_DEMAND_PROFILE,
    LAMBDA,
    LR,
    MINIBATCH_SIZE,
    N_TRAIN_DAYS,
    PPO_UPDATE_EPOCHS,
    ROUTE_RANDOM_TRIP_SEED,
    ROUTE_SAMPLER_SEED,
    ROUTE_SAMPLER_THREADS,
    RUN_SUFFIX,
    SINGLE_TAZ_ID,
    SINGLE_TAZ_WAITING_ONLY_REWARD,
    TARGET_KL,
    TRAIN_ON_SAME_DAY,
    TRAIN_START_DATE,
    VALUE_COEF,
    _build_action_summary_by_taz,
    _build_episode_schedule,
    _clone_state_dict_to_cpu,
    _count_discrete_actions,
    _ensure_baseline_reference,
    _masked_mean_std,
    _move_optimizer_state_to_device,
    _resolve_route_folder,
    _sample_demand_noise,
    _set_global_seed,
    compute_gae,
    ppo_update,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


TRAINING_MODE = "global_only"  # "global_only" or "joint_finetune"

LOAD_LOCAL_POLICY = True
LOCAL_POLICY_CHECKPOINT_PATH = os.path.join(
    SCRIPT_DIR,
    f"checkpoints_v11{RUN_SUFFIX}",
    "checkpoint_ppo_v11_best.pt",
)

LOAD_HIERARCHICAL_CHECKPOINT = False
HIERARCHICAL_CHECKPOINT_PATH = os.path.join(
    SCRIPT_DIR,
    f"checkpoints_hierarchical_v11{RUN_SUFFIX}",
    "checkpoint_hierarchical_v11_final.pt",
)
RESUME_FROM_HIERARCHICAL_EPISODE = True
START_EPISODE_OVERRIDE = None

GLOBAL_LR = 1e-4
GLOBAL_CLIP_RATIO = 0.10
GLOBAL_ENTROPY_COEF = 0.010
GLOBAL_ENTROPY_COEF_FINAL = 0.002
GLOBAL_VALUE_COEF = 0.5
GLOBAL_TARGET_KL = 0.015
GLOBAL_PPO_UPDATE_EPOCHS = 4
GLOBAL_MINIBATCH_SIZE = 32
GLOBAL_GAMMA = 0.99
GLOBAL_LAMBDA = 0.95
GLOBAL_BATCH_EPISODES = 8

GLOBAL_OBSERVATION_CONFIG = GlobalObservationConfig(
    per_taz_feature_dim=14,
    include_city_mean=True,
    include_city_std=True,
    include_city_max=True,
    include_city_min=False,
    include_action_density=False,
)
GLOBAL_REWARD_CONFIG = GlobalRewardConfig(
    local_bonus_scale=0.75,
    alignment_reward_weight=0.35,
    delta_penalty_weight=0.70,
    residual_penalty_weight=0.30,
    effort_weight=0.20,
    effort_delta_ref=10.0,
    local_bonus_clip=1.5,
    global_reward_clip=4.0,
    city_reward_weight=1.0,
)


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages
    std = advantages.std(unbiased=False)
    if float(std.item()) < 1e-8:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (std + 1e-8)


def _entropy_coef_now(episode_idx: int, total_episodes: int, start_value: float, end_value: float) -> float:
    progress = episode_idx / max(total_episodes - 1, 1)
    if progress <= ENTROPY_WARMUP_RATIO:
        return float(start_value)
    decay_progress = (progress - ENTROPY_WARMUP_RATIO) / max(1.0 - ENTROPY_WARMUP_RATIO, 1e-8)
    return float(start_value + (end_value - start_value) * decay_progress)


def _safe_load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_policy_weights(policy: torch.nn.Module, checkpoint_path: str, label: str) -> dict:
    checkpoint = _safe_load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    policy.load_state_dict(state_dict)
    print(f"[INFO] Loaded {label} weights from {checkpoint_path}")
    return checkpoint


def _maybe_load_hierarchical_checkpoint(
    local_policy: ActorCriticV10,
    global_policy: ActorCriticV10,
    local_optim: Optional[torch.optim.Optimizer],
    global_optim: torch.optim.Optimizer,
    local_scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    global_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
) -> int:
    start_episode = 0
    if not LOAD_HIERARCHICAL_CHECKPOINT:
        return start_episode

    checkpoint = _safe_load_checkpoint(HIERARCHICAL_CHECKPOINT_PATH)
    if checkpoint.get("local_model_state_dict") is not None:
        local_policy.load_state_dict(checkpoint["local_model_state_dict"])
    if checkpoint.get("global_model_state_dict") is not None:
        global_policy.load_state_dict(checkpoint["global_model_state_dict"])

    if local_optim is not None and checkpoint.get("local_optimizer_state_dict") is not None:
        try:
            local_optim.load_state_dict(checkpoint["local_optimizer_state_dict"])
        except Exception as exc:
            print(f"[WARN] Could not load local optimizer state: {exc}")
    if checkpoint.get("global_optimizer_state_dict") is not None:
        try:
            global_optim.load_state_dict(checkpoint["global_optimizer_state_dict"])
        except Exception as exc:
            print(f"[WARN] Could not load global optimizer state: {exc}")

    if local_scheduler is not None and checkpoint.get("local_scheduler_state_dict") is not None:
        try:
            local_scheduler.load_state_dict(checkpoint["local_scheduler_state_dict"])
        except Exception as exc:
            print(f"[WARN] Could not load local scheduler state: {exc}")
    if checkpoint.get("global_scheduler_state_dict") is not None:
        try:
            global_scheduler.load_state_dict(checkpoint["global_scheduler_state_dict"])
        except Exception as exc:
            print(f"[WARN] Could not load global scheduler state: {exc}")

    if RESUME_FROM_HIERARCHICAL_EPISODE:
        start_episode = int(checkpoint.get("episode", -1)) + 1
    if START_EPISODE_OVERRIDE is not None:
        start_episode = int(START_EPISODE_OVERRIDE)

    print(f"[INFO] Resumed hierarchical checkpoint from episode {start_episode}")
    return start_episode


def _load_history(json_path: str) -> list[dict]:
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _extract_terminal_penalty(row: dict) -> Optional[float]:
    if not bool(row.get("terminal_parse_ok", False)):
        return None
    try:
        value = float(row.get("terminal_penalty", ""))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _prepare_history(csv_path: str, json_path: str, csv_cols: list[str], start_episode: int) -> tuple[list[dict], float, int]:
    history = _load_history(json_path) if LOAD_HIERARCHICAL_CHECKPOINT else []
    history = [row for row in history if int(row.get("episode", -1)) < int(start_episode)]

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_cols)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in csv_cols})

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

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
    return history, best_terminal_penalty, best_episode


def _append_history_row(history: list[dict], row: dict, csv_path: str, json_path: str, csv_cols: list[str]):
    history.append(row)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_cols)
        writer.writerow({key: row.get(key, "") for key in csv_cols})
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def _build_global_context_tensor(
    hour: int,
    total_cars_random: int,
    demand_noise: float,
    baseline_penalty: Optional[float],
    device: torch.device,
) -> torch.Tensor:
    baseline_value = 0.0 if baseline_penalty is None else float(max(min(baseline_penalty, 5.0), -5.0))
    return torch.tensor(
        [
            float(hour) / 23.0,
            float(total_cars_random) / max(float(BASE_DEMAND), 1.0),
            float(demand_noise),
            baseline_value,
        ],
        dtype=torch.float32,
        device=device,
    )


def _build_env_config() -> dict:
    env_config = dict(ENV_V11_CONFIG)
    if SINGLE_TAZ_ID and SINGLE_TAZ_WAITING_ONLY_REWARD:
        env_config.update(
            waiting_reward_weight=1.0,
            emission_reward_weight=0.0,
            jam_reward_weight=0.0,
        )
    return env_config


def _extract_global_local_obs(local_observation: torch.Tensor, env: SumoTazEnvV11) -> torch.Tensor:
    if local_observation.ndim != 2:
        raise ValueError(
            f"Expected local observation with shape [num_taz, obs_dim], got {tuple(local_observation.shape)}."
        )
    start_idx = int(env.max_tls_per_taz * env.per_tls_feature_dim)
    end_idx = start_idx + int(env.per_taz_feature_dim)
    return local_observation[:, start_idx:end_idx]


def _run_global_batch_update(
    batch_entries: list[dict],
    global_policy: ActorCriticV10,
    global_optim: torch.optim.Optimizer,
    global_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    global_obs_dim: int,
    num_taz: int,
    total_episodes: int,
    device: torch.device,
) -> tuple[dict, float, float]:
    batch_obs = torch.cat([entry["obs"] for entry in batch_entries], dim=0).to(device=device, dtype=torch.float32)
    batch_action_mask = torch.cat([entry["action_mask"] for entry in batch_entries], dim=0).to(device=device)
    batch_action_index = torch.cat([entry["action_index"] for entry in batch_entries], dim=0).to(device=device)
    batch_logp = torch.cat([entry["logp"] for entry in batch_entries], dim=0).to(device=device, dtype=torch.float32).reshape(-1)
    batch_values = torch.cat([entry["value"] for entry in batch_entries], dim=0).to(device=device, dtype=torch.float32).reshape(-1, 1)
    batch_rewards = torch.tensor(
        [[float(entry["reward"])] for entry in batch_entries],
        dtype=torch.float32,
        device=device,
    )
    batch_dones = torch.ones_like(batch_rewards)
    batch_adv, batch_ret = compute_gae(
        batch_rewards,
        batch_values,
        batch_dones,
        gamma=GLOBAL_GAMMA,
        lam=GLOBAL_LAMBDA,
    )
    entropy_coef = float(sum(
        _entropy_coef_now(
            episode_idx=int(entry["episode_idx"]),
            total_episodes=total_episodes,
            start_value=GLOBAL_ENTROPY_COEF,
            end_value=GLOBAL_ENTROPY_COEF_FINAL,
        )
        for entry in batch_entries
    ) / max(len(batch_entries), 1))
    stats = ppo_update(
        global_policy,
        global_optim,
        batch_obs.reshape(-1, global_obs_dim),
        batch_action_mask.reshape(-1, num_taz),
        batch_action_index.reshape(-1, num_taz),
        batch_logp,
        _normalize_advantages(batch_adv.reshape(-1)),
        batch_ret.reshape(-1),
        clip_ratio=GLOBAL_CLIP_RATIO,
        ppo_epochs=GLOBAL_PPO_UPDATE_EPOCHS,
        minibatch_size=GLOBAL_MINIBATCH_SIZE,
        entropy_coef=entropy_coef,
        value_coef=GLOBAL_VALUE_COEF,
        target_kl=GLOBAL_TARGET_KL,
    )
    lr_before_step = float(global_optim.param_groups[0]["lr"])
    global_scheduler.step(float(batch_rewards.mean().item()))
    lr_after_step = float(global_optim.param_groups[0]["lr"])
    return stats, lr_before_step, lr_after_step


def main():
    if TRAINING_MODE not in {"global_only", "joint_finetune"}:
        raise ValueError(f"Unsupported TRAINING_MODE={TRAINING_MODE}")
    if SINGLE_TAZ_ID:
        print("[WARN] SINGLE_TAZ_ID is enabled. The global priority agent is only meaningful with multiple TAZs.")

    _set_global_seed(GLOBAL_SEED)

    sumo_standalone_dir = os.path.join(constants.SUMO_PATH, "standalone")
    log_file = os.path.join(sumo_standalone_dir, "command_log_hierarchical_v11.txt")
    sumo = Simulator(configurationPath=sumo_standalone_dir, logFile=log_file, tazTlsMapFile=constants.TAZ_FILE)
    planner = Planner(simulator=sumo)
    sumo.changeDetectorPath(detectorPath=constants.SUMO_NETWORK_PATH)

    selected_taz_ids = [str(SINGLE_TAZ_ID)] if SINGLE_TAZ_ID else None
    env_config = _build_env_config()
    env = SumoTazEnvV11(
        sumoSimulator=sumo,
        stepSize=600,
        selected_taz_ids=selected_taz_ids,
        **env_config,
    )
    runtime_device = torch.device(env.device)
    taz_ids = env.get_taz_ids()
    tls_ids = env.get_tls_ids()
    num_taz = len(taz_ids)
    obs_dim = int(env.agent_obs_dim)
    act_dim = int(env.max_control_groups_per_taz)

    local_policy = ActorCriticV10(obs_dim, act_dim, ACTION_BINS).to(runtime_device)
    if LOAD_LOCAL_POLICY:
        _load_policy_weights(local_policy, LOCAL_POLICY_CHECKPOINT_PATH, "local v11 policy")
    local_policy.train(mode=(TRAINING_MODE == "joint_finetune"))
    if TRAINING_MODE == "global_only":
        for parameter in local_policy.parameters():
            parameter.requires_grad_(False)

    dummy_local_obs = torch.zeros((num_taz, env.per_taz_feature_dim), dtype=torch.float32, device=runtime_device)
    dummy_context = torch.zeros(4, dtype=torch.float32, device=runtime_device)
    global_obs_dim = int(
        build_global_observation(
            dummy_local_obs,
            extra_context=dummy_context,
            config=GLOBAL_OBSERVATION_CONFIG,
        ).shape[-1]
    )
    global_policy = ActorCriticV10(global_obs_dim, num_taz, GLOBAL_PRIORITY_BINS).to(runtime_device)
    global_policy.train()

    local_optim = None
    local_scheduler = None
    if TRAINING_MODE == "joint_finetune":
        local_optim = torch.optim.Adam(local_policy.parameters(), lr=LR)
        local_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            local_optim,
            mode="max",
            factor=0.7,
            patience=15,
            threshold=0.005,
            threshold_mode="rel",
            min_lr=1e-5,
        )

    global_optim = torch.optim.Adam(global_policy.parameters(), lr=GLOBAL_LR)
    global_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        global_optim,
        mode="max",
        factor=0.7,
        patience=15,
        threshold=0.005,
        threshold_mode="rel",
        min_lr=1e-5,
    )

    start_episode = _maybe_load_hierarchical_checkpoint(
        local_policy=local_policy,
        global_policy=global_policy,
        local_optim=local_optim,
        global_optim=global_optim,
        local_scheduler=local_scheduler,
        global_scheduler=global_scheduler,
    )
    if local_optim is not None:
        _move_optimizer_state_to_device(local_optim, runtime_device)
    _move_optimizer_state_to_device(global_optim, runtime_device)

    episode_schedule = _build_episode_schedule()
    n_episodes = len(episode_schedule)
    if start_episode >= n_episodes:
        print(f"[INFO] start_episode={start_episode} >= n_episodes={n_episodes}. Nothing to train.")
        return

    csv_path = os.path.join(SCRIPT_DIR, f"training_history_hierarchical_v11{RUN_SUFFIX}.csv")
    json_path = os.path.join(SCRIPT_DIR, f"training_history_hierarchical_v11{RUN_SUFFIX}.json")
    ckpt_dir = os.path.join(SCRIPT_DIR, f"checkpoints_hierarchical_v11{RUN_SUFFIX}")
    detail_dir = os.path.join(SCRIPT_DIR, f"training_details_hierarchical_v11{RUN_SUFFIX}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(detail_dir, exist_ok=True)

    csv_cols = [
        "episode",
        "date",
        "timeslot",
        "training_mode",
        "num_taz",
        "num_tls",
        "num_control_groups_max",
        "num_steps",
        "baseline_parse_ok",
        "baseline_penalty",
        "terminal_parse_ok",
        "terminal_penalty",
        "final_penalty_mean",
        "base_local_reward_mean",
        "base_local_reward_sum",
        "shaped_local_reward_mean",
        "shaped_local_reward_sum",
        "local_bonus_mean",
        "local_bonus_std",
        "local_bonus_sum",
        "global_reward",
        "global_city_reward",
        "global_alignment_reward",
        "global_priority_abs_mean",
        "global_priority_std",
        "global_update_applied",
        "global_batch_size",
        "global_buffer_size_pending",
        "global_batch_reward_mean",
        "global_learning_rate",
        "global_learning_rate_after_step",
        "global_ppo_total_loss",
        "global_ppo_policy_loss",
        "global_ppo_value_loss",
        "global_policy_clip_fraction",
        "global_approx_kl",
        "global_policy_entropy_mean",
        "global_ppo_early_stop",
        "local_learning_rate",
        "local_learning_rate_after_step",
        "local_ppo_total_loss",
        "local_ppo_policy_loss",
        "local_ppo_value_loss",
        "local_policy_clip_fraction",
        "local_approx_kl",
        "local_policy_entropy_mean",
        "details_path",
    ]
    history, best_terminal_penalty, best_episode = _prepare_history(csv_path, json_path, csv_cols, start_episode)
    baseline_cache = {}
    interrupted = False
    last_completed_episode = start_episode - 1
    global_rollout_buffers: dict[int, list[dict]] = {}
    remaining_global_episodes_by_hour = {}
    for _, scheduled_hour in episode_schedule[start_episode:]:
        remaining_global_episodes_by_hour[scheduled_hour] = remaining_global_episodes_by_hour.get(scheduled_hour, 0) + 1

    print(f"[INFO] Hierarchical v11 mode: {TRAINING_MODE}")
    print(f"[INFO] TAZ count: {len(taz_ids)} | IDs: {taz_ids}")
    print(f"[INFO] TLS count: {len(tls_ids)}")
    print(f"[INFO] Local obs dim per TAZ: {obs_dim} | local control groups: {act_dim}")
    print(f"[INFO] Global obs dim: {global_obs_dim} | priority bins: {GLOBAL_PRIORITY_BINS}")
    print(f"[INFO] Global batch episodes per hour slot: {GLOBAL_BATCH_EPISODES}")
    print(
        f"[INFO] Same-day training: {TRAIN_ON_SAME_DAY} | "
        f"start_date={TRAIN_START_DATE.strftime('%Y-%m-%d')} | focus_hours={FOCUS_HOURS} | n_days={N_TRAIN_DAYS}"
    )

    for episode_idx in range(start_episode, n_episodes):
        episode_num = episode_idx + 1
        day, hour = episode_schedule[episode_idx]
        simulation_date = day.strftime("%Y-%m-%d")
        timeslot = f"{hour:02d}:00-{(hour + 1):02d}:00"
        timeslot_clean = timeslot.replace(":", "-")
        print(f"\n[HIER EP {episode_num}/{n_episodes}] Date={simulation_date} Slot={timeslot}")

        total_cars = int(BASE_DEMAND * HOURLY_DEMAND_PROFILE[hour])
        demand_noise = _sample_demand_noise(episode_idx)
        total_cars_random = int(total_cars * demand_noise)
        route_folder_path, should_generate_routes = _resolve_route_folder(
            simulation_date=simulation_date,
            timeslot_clean=timeslot_clean,
            total_cars_random=total_cars_random,
        )
        os.makedirs(os.path.join(route_folder_path, "output"), exist_ok=True)

        if should_generate_routes:
            print(f"[ROUTE GEN] {route_folder_path}")
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
        else:
            print(f"[ROUTE CACHE] {route_folder_path}")

        baseline_penalty, baseline_components, baseline_ran = _ensure_baseline_reference(
            env=env,
            sumo=sumo,
            hour=hour,
            route_folder_path=route_folder_path,
            cache=baseline_cache,
        )
        baseline_parse_ok = bool(baseline_components.get("parse_ok", False))
        print(
            f"[BASELINE REF] {'Computed' if baseline_ran else 'Reused'} | "
            f"parse_ok={baseline_parse_ok} | penalty={baseline_penalty if baseline_penalty is not None else 'n/a'}"
        )

        sumo.changeRouteFilePath(route_folder_path)
        sumo.changeTypePath(route_folder_path)
        if env_config["comparison_reward_enabled"] and baseline_penalty is not None:
            env.set_baseline_penalty(baseline_penalty, baseline_components)
        else:
            env.set_baseline_penalty(None)
        env.set_episode_context(hour=hour)
        td = env.reset()

        initial_local_obs = td["observation"].detach().clone()
        global_context = _build_global_context_tensor(
            hour=hour,
            total_cars_random=total_cars_random,
            demand_noise=demand_noise,
            baseline_penalty=baseline_penalty,
            device=runtime_device,
        )
        global_obs = build_global_observation(
            _extract_global_local_obs(initial_local_obs, env),
            action_mask=td["action_mask"].bool(),
            extra_context=global_context,
            config=GLOBAL_OBSERVATION_CONFIG,
        )
        global_action_mask = build_global_action_mask(num_taz=num_taz, device=runtime_device)
        global_action_index, global_logp, global_value = global_policy.act(global_obs, global_action_mask)
        global_action_values = global_policy.action_values(global_action_index).squeeze(0)
        print(
            f"[GLOBAL ACT] mean={global_action_values.float().mean().item():.3f} | "
            f"abs_mean={global_action_values.float().abs().mean().item():.3f}"
        )

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

        local_rollout_state = _clone_state_dict_to_cpu(local_policy.state_dict())
        global_rollout_state = _clone_state_dict_to_cpu(global_policy.state_dict())

        while True:
            obs = td["observation"]
            action_mask = td["action_mask"].bool()
            if not torch.isfinite(obs).all():
                print("[WARN] Non-finite local observation. Skipping episode.")
                episode_invalid = True
                break

            action_index, logp, value = local_policy.act(obs, action_mask)
            if not (torch.isfinite(logp).all() and torch.isfinite(value).all()):
                print("[WARN] Non-finite local policy outputs. Skipping episode.")
                episode_invalid = True
                break

            try:
                step_td = env.step(TensorDict({"action": action_index}, batch_size=[], device=runtime_device))
            except KeyboardInterrupt:
                interrupted = True
                episode_invalid = True
                print("[WARN] Interrupted during rollout.")
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

        reward_components = dict(getattr(env, "last_reward_components", {}) or {})
        terminal_reward = dict(reward_components.get("terminal_reward", {}) or {})
        terminal_parse_ok = bool(terminal_reward.get("parse_ok", False))
        terminal_penalty = float(terminal_reward.get("penalty", 0.0)) if terminal_parse_ok else None

        batch_mask_3d = torch.stack(action_masks_list)
        batch_applied_act_3d = torch.stack(applied_actions_list)
        batch_applied_delta_3d = torch.stack(applied_deltas_list)
        action_summary_by_taz = _build_action_summary_by_taz(
            selected_actions=batch_applied_act_3d,
            applied_deltas=batch_applied_delta_3d,
            action_mask=batch_mask_3d,
            taz_ids=taz_ids,
        )

        local_bonus_vec, global_reward_value, global_reward_diag = compute_priority_rewards(
            taz_ids=taz_ids,
            priority_action_values=global_action_values,
            reward_components=reward_components,
            action_summary_by_taz=action_summary_by_taz,
            baseline_penalty=baseline_penalty,
            terminal_penalty=terminal_penalty,
            config=GLOBAL_REWARD_CONFIG,
        )

        rewards_t = torch.stack(rewards_list)
        base_rewards_t = rewards_t.clone()
        if TRAINING_MODE == "joint_finetune":
            rewards_t[-1] = rewards_t[-1] + local_bonus_vec.to(device=runtime_device, dtype=rewards_t.dtype)

        base_episode_reward_by_taz = base_rewards_t.sum(dim=0).cpu()
        shaped_episode_reward_by_taz = rewards_t.sum(dim=0).cpu()
        local_bonus_cpu = local_bonus_vec.detach().cpu()

        local_stats = {
            "total_loss": "",
            "policy_loss": "",
            "value_loss": "",
            "clip_fraction": "",
            "approx_kl": "",
            "entropy_mean": "",
        }
        local_lr_before_step = ""
        local_lr_after_step = ""

        if TRAINING_MODE == "joint_finetune":
            values_t = torch.stack(values_list)
            dones_t = torch.stack(dones_list)
            adv_t, ret_t = compute_gae(rewards_t, values_t, dones_t, gamma=GAMMA, lam=LAMBDA)

            batch_obs_3d = torch.stack(obs_list)
            batch_act_3d = torch.stack(acts_list)
            batch_logp_2d = torch.stack(logps_list)

            batch_obs = batch_obs_3d.reshape(-1, obs_dim)
            batch_action_mask = batch_mask_3d.reshape(-1, act_dim)
            batch_act = batch_act_3d.reshape(-1, act_dim)
            batch_logp = batch_logp_2d.reshape(-1)
            batch_adv = _normalize_advantages(adv_t.reshape(-1))
            batch_ret = ret_t.reshape(-1)
            local_entropy_coef_now = _entropy_coef_now(
                episode_idx=episode_idx,
                total_episodes=n_episodes,
                start_value=ENTROPY_COEF,
                end_value=ENTROPY_COEF_FINAL,
            )

            local_stats = ppo_update(
                local_policy,
                local_optim,
                batch_obs,
                batch_action_mask,
                batch_act,
                batch_logp,
                batch_adv,
                batch_ret,
                clip_ratio=CLIP_RATIO,
                ppo_epochs=PPO_UPDATE_EPOCHS,
                minibatch_size=MINIBATCH_SIZE,
                entropy_coef=local_entropy_coef_now,
                value_coef=VALUE_COEF,
                target_kl=TARGET_KL,
            )
            local_lr_before_step = float(local_optim.param_groups[0]["lr"])
            local_scheduler.step(float(shaped_episode_reward_by_taz.mean().item()))
            local_lr_after_step = float(local_optim.param_groups[0]["lr"])

        global_buffer = global_rollout_buffers.setdefault(int(hour), [])
        global_buffer.append(
            {
                "episode_idx": int(episode_idx),
                "obs": global_obs.detach().cpu().reshape(1, global_obs_dim),
                "action_mask": global_action_mask.detach().cpu().reshape(1, num_taz),
                "action_index": global_action_index.detach().cpu().reshape(1, num_taz),
                "logp": global_logp.detach().cpu().reshape(1),
                "value": global_value.detach().cpu().reshape(1),
                "reward": float(global_reward_value),
            }
        )
        remaining_global_episodes_by_hour[hour] = max(remaining_global_episodes_by_hour.get(hour, 1) - 1, 0)

        global_update_applied = False
        global_batch_size = 0
        global_batch_reward_mean = ""
        global_lr_before_step = ""
        global_lr_after_step = ""
        global_stats = {
            "total_loss": "",
            "policy_loss": "",
            "value_loss": "",
            "clip_fraction": "",
            "approx_kl": "",
            "entropy_mean": "",
            "early_stop": "",
        }

        should_flush_global_buffer = (
            len(global_buffer) >= GLOBAL_BATCH_EPISODES
            or remaining_global_episodes_by_hour.get(hour, 0) == 0
        )
        if should_flush_global_buffer:
            global_batch_size = int(len(global_buffer))
            global_batch_reward_mean = float(sum(entry["reward"] for entry in global_buffer) / max(global_batch_size, 1))
            global_stats, global_lr_before_step, global_lr_after_step = _run_global_batch_update(
                batch_entries=global_buffer,
                global_policy=global_policy,
                global_optim=global_optim,
                global_scheduler=global_scheduler,
                global_obs_dim=global_obs_dim,
                num_taz=num_taz,
                total_episodes=n_episodes,
                device=runtime_device,
            )
            global_buffer.clear()
            global_update_applied = True

        global_buffer_size_pending = int(len(global_buffer))

        penalty_by_taz = {
            taz: float((reward_components.get("penalty_by_taz", {}) or {}).get(taz, 0.0))
            for taz in taz_ids
        }
        penalty_vector = torch.tensor([penalty_by_taz[taz] for taz in taz_ids], dtype=torch.float32)
        avg_action_selected, _ = _masked_mean_std(batch_applied_act_3d, batch_mask_3d.bool())
        avg_applied_delta, _ = _masked_mean_std(batch_applied_delta_3d, batch_mask_3d.bool())
        action_counts = _count_discrete_actions(batch_applied_act_3d, batch_mask_3d.bool())

        detail_path = os.path.join(detail_dir, f"episode_{episode_idx:04d}.json")
        with open(detail_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "episode": int(episode_idx),
                    "episode_num": int(episode_num),
                    "date": simulation_date,
                    "timeslot": timeslot,
                    "training_mode": TRAINING_MODE,
                    "num_steps": int(len(rewards_list)),
                    "num_control_groups_max": int(act_dim),
                    "baseline_reference_components": baseline_components,
                    "terminal_reward_components": terminal_reward,
                    "base_reward_by_taz": {
                        taz: float(base_episode_reward_by_taz[idx].item())
                        for idx, taz in enumerate(taz_ids)
                    },
                    "shaped_reward_by_taz": {
                        taz: float(shaped_episode_reward_by_taz[idx].item())
                        for idx, taz in enumerate(taz_ids)
                    },
                    "final_penalty_by_taz": penalty_by_taz,
                    "global_priority_action_by_taz": {
                        taz: float(global_action_values[idx].item())
                        for idx, taz in enumerate(taz_ids)
                    },
                    "global_update_applied": bool(global_update_applied),
                    "global_batch_size": int(global_batch_size),
                    "global_buffer_size_pending": int(global_buffer_size_pending),
                    "global_batch_reward_mean": global_batch_reward_mean,
                    "global_reward_diagnostics": global_reward_diag,
                    "local_bonus_by_taz": {
                        taz: float(local_bonus_cpu[idx].item())
                        for idx, taz in enumerate(taz_ids)
                    },
                    "group_action_details": dict(reward_components.get("group_action_details", {}) or {}),
                    "group_signal_by_taz": dict(reward_components.get("group_signal_by_taz", {}) or {}),
                    "action_summary_by_taz": action_summary_by_taz,
                    "rollout_summary": {
                        "avg_action_selected": float(avg_action_selected),
                        "avg_applied_duration_delta": float(avg_applied_delta),
                        **action_counts,
                    },
                },
                handle,
                indent=2,
            )

        row = {
            "episode": int(episode_idx),
            "date": simulation_date,
            "timeslot": timeslot,
            "training_mode": TRAINING_MODE,
            "num_taz": int(num_taz),
            "num_tls": int(len(tls_ids)),
            "num_control_groups_max": int(act_dim),
            "num_steps": int(len(rewards_list)),
            "baseline_parse_ok": bool(baseline_parse_ok),
            "baseline_penalty": float(baseline_penalty) if baseline_penalty is not None else "",
            "terminal_parse_ok": bool(terminal_parse_ok),
            "terminal_penalty": float(terminal_penalty) if terminal_penalty is not None else "",
            "final_penalty_mean": float(penalty_vector.mean().item()),
            "base_local_reward_mean": float(base_episode_reward_by_taz.mean().item()),
            "base_local_reward_sum": float(base_episode_reward_by_taz.sum().item()),
            "shaped_local_reward_mean": float(shaped_episode_reward_by_taz.mean().item()),
            "shaped_local_reward_sum": float(shaped_episode_reward_by_taz.sum().item()),
            "local_bonus_mean": float(local_bonus_cpu.mean().item()),
            "local_bonus_std": float(local_bonus_cpu.std(unbiased=False).item()),
            "local_bonus_sum": float(local_bonus_cpu.sum().item()),
            "global_reward": float(global_reward_value),
            "global_city_reward": float(global_reward_diag.get("city_reward", 0.0)),
            "global_alignment_reward": float(global_reward_diag.get("alignment_reward", 0.0)),
            "global_priority_abs_mean": float(global_action_values.abs().mean().item()),
            "global_priority_std": float(global_action_values.std(unbiased=False).item()),
            "global_update_applied": bool(global_update_applied),
            "global_batch_size": int(global_batch_size),
            "global_buffer_size_pending": int(global_buffer_size_pending),
            "global_batch_reward_mean": global_batch_reward_mean,
            "global_learning_rate": global_lr_before_step,
            "global_learning_rate_after_step": global_lr_after_step,
            "global_ppo_total_loss": global_stats["total_loss"],
            "global_ppo_policy_loss": global_stats["policy_loss"],
            "global_ppo_value_loss": global_stats["value_loss"],
            "global_policy_clip_fraction": global_stats["clip_fraction"],
            "global_approx_kl": global_stats["approx_kl"],
            "global_policy_entropy_mean": global_stats["entropy_mean"],
            "global_ppo_early_stop": global_stats["early_stop"],
            "local_learning_rate": local_lr_before_step,
            "local_learning_rate_after_step": local_lr_after_step,
            "local_ppo_total_loss": local_stats["total_loss"],
            "local_ppo_policy_loss": local_stats["policy_loss"],
            "local_ppo_value_loss": local_stats["value_loss"],
            "local_policy_clip_fraction": local_stats["clip_fraction"],
            "local_approx_kl": local_stats["approx_kl"],
            "local_policy_entropy_mean": local_stats["entropy_mean"],
            "details_path": detail_path,
        }
        _append_history_row(history, row, csv_path, json_path, csv_cols)

        checkpoint_payload = {
            "episode": int(episode_idx),
            "training_mode": TRAINING_MODE,
            "local_model_state_dict": local_policy.state_dict(),
            "global_model_state_dict": global_policy.state_dict(),
            "local_optimizer_state_dict": local_optim.state_dict() if local_optim is not None else None,
            "global_optimizer_state_dict": global_optim.state_dict(),
            "local_scheduler_state_dict": local_scheduler.state_dict() if local_scheduler is not None else None,
            "global_scheduler_state_dict": global_scheduler.state_dict(),
            "env_config": env_config,
            "focus_hours": FOCUS_HOURS,
            "taz_ids": taz_ids,
            "tls_ids": tls_ids,
            "max_control_groups_per_taz": act_dim,
            "control_groups_by_taz": env.get_control_groups_by_taz(),
            "local_action_bins": ACTION_BINS,
            "global_priority_bins": GLOBAL_PRIORITY_BINS,
            "global_observation_config": vars(GLOBAL_OBSERVATION_CONFIG),
            "global_reward_config": vars(GLOBAL_REWARD_CONFIG),
            "global_batch_episodes": GLOBAL_BATCH_EPISODES,
            "best_episode": best_episode,
            "best_terminal_penalty": best_terminal_penalty if math.isfinite(best_terminal_penalty) else None,
        }
        torch.save(
            checkpoint_payload,
            os.path.join(ckpt_dir, f"checkpoint_hierarchical_v11_episode{episode_idx}.pt"),
        )

        if terminal_penalty is not None and float(terminal_penalty) < best_terminal_penalty:
            best_terminal_penalty = float(terminal_penalty)
            best_episode = int(episode_idx)
            best_payload = dict(checkpoint_payload)
            best_payload["local_model_state_dict"] = local_rollout_state
            best_payload["global_model_state_dict"] = global_rollout_state
            best_payload["local_optimizer_state_dict"] = None
            best_payload["global_optimizer_state_dict"] = None
            best_payload["local_scheduler_state_dict"] = None
            best_payload["global_scheduler_state_dict"] = None
            best_payload["policy_stage"] = "pre_update_rollout_policy"
            torch.save(best_payload, os.path.join(ckpt_dir, "checkpoint_hierarchical_v11_best.pt"))

        last_completed_episode = episode_idx
        print(
            f"[HIER RESULT] EP {episode_num}/{n_episodes} | "
            f"TerminalPenalty {terminal_penalty if terminal_penalty is not None else 'n/a'} | "
            f"BaseLocalMean {row['base_local_reward_mean']:.4f} | "
            f"ShapedLocalMean {row['shaped_local_reward_mean']:.4f} | "
            f"LocalBonusMean {row['local_bonus_mean']:.4f} | "
            f"GlobalReward {row['global_reward']:.4f} | "
            f"PriorityAbsMean {row['global_priority_abs_mean']:.3f} | "
            f"GlobalUpdate {global_update_applied} | "
            f"BatchSize {global_batch_size} | "
            f"PendingHourBuffer {global_buffer_size_pending}"
        )

    final_payload = {
        "episode": last_completed_episode,
        "training_mode": TRAINING_MODE,
        "local_model_state_dict": local_policy.state_dict(),
        "global_model_state_dict": global_policy.state_dict(),
        "local_optimizer_state_dict": local_optim.state_dict() if local_optim is not None else None,
        "global_optimizer_state_dict": global_optim.state_dict(),
        "local_scheduler_state_dict": local_scheduler.state_dict() if local_scheduler is not None else None,
        "global_scheduler_state_dict": global_scheduler.state_dict(),
        "history_len": len(history),
        "env_config": env_config,
        "focus_hours": FOCUS_HOURS,
        "taz_ids": taz_ids,
        "tls_ids": tls_ids,
        "max_control_groups_per_taz": act_dim,
        "control_groups_by_taz": env.get_control_groups_by_taz(),
        "local_action_bins": ACTION_BINS,
        "global_priority_bins": GLOBAL_PRIORITY_BINS,
        "global_observation_config": vars(GLOBAL_OBSERVATION_CONFIG),
        "global_reward_config": vars(GLOBAL_REWARD_CONFIG),
        "global_batch_episodes": GLOBAL_BATCH_EPISODES,
        "best_episode": best_episode,
        "best_terminal_penalty": best_terminal_penalty if math.isfinite(best_terminal_penalty) else None,
        "interrupted": bool(interrupted),
    }
    torch.save(final_payload, os.path.join(ckpt_dir, "checkpoint_hierarchical_v11_final.pt"))

    if interrupted:
        print("[INFO] Hierarchical training interrupted. Partial checkpoint saved.")
    else:
        print("[INFO] Hierarchical v11 training complete.")


if __name__ == "__main__":
    main()
