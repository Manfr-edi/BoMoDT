import csv
import json
import math
import os
from typing import Optional

"""
Train the v12 coordination architecture.

The local v11 policy still controls SUMO traffic-light groups. The new global
policy runs at every local step and emits per-TAZ prices. Local observations are
augmented with those prices, and local rewards are corrected when observed
TAZ-to-TAZ flows indicate harmful spillover into worsening downstream TAZs.
"""

import numpy as np
import torch
from tensordict import TensorDict

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
from taz_rl.training_script_ppo_v11 import (
    ACTION_BINS,
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
    REUSE_DETERMINISTIC_ROUTE_FILES,
    ROUTE_RANDOM_TRIP_SEED,
    ROUTE_SAMPLER_SEED,
    ROUTE_SAMPLER_THREADS,
    SINGLE_TAZ_ID,
    SINGLE_TAZ_WAITING_ONLY_REWARD,
    TARGET_KL,
    TRAIN_DAY_FILTER,
    TRAIN_ON_SAME_DAY,
    VALUE_COEF,
    ActorCriticV10,
    _build_episode_schedule,
    _clone_state_dict_to_cpu,
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


# By default, train both the global price policy and the augmented local policy.
# "global_only" is useful for diagnostics, but it cannot affect traffic unless
# the local policy later learns to respond to the price features.
TRAINING_MODE = "joint_finetune"  # "joint_finetune" or "global_only"
RUN_SUFFIX = "_coordination_v12"

# The v12 local policy has three extra observation features, so this loader copies
# all compatible v11 weights and initializes the new input columns to zero.
LOAD_LOCAL_POLICY = True
LOCAL_POLICY_CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints_v11", "checkpoint_ppo_v11_best.pt")

LOAD_COORDINATION_CHECKPOINT = False
COORDINATION_CHECKPOINT_PATH = os.path.join(
    SCRIPT_DIR,
    f"checkpoints{RUN_SUFFIX}",
    "checkpoint_coordination_v12_final.pt",
)
RESUME_FROM_COORDINATION_EPISODE = True
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

LOCAL_GLOBAL_CONTEXT_DIM = 3
TAZ_ADJACENCY_DISTANCE_THRESHOLD = 80.0
TAZ_ADJACENCY_FALLBACK_K = 3

# Global observations combine per-TAZ state, neighbor state, previous flows,
# previous prices, previous local-action intensity, and episode context.
COORDINATION_OBSERVATION_CONFIG = CoordinationObservationConfig(
    per_taz_feature_dim=14,
    include_neighbor_mean=True,
    include_flow_features=True,
    include_previous_prices=True,
    include_city_mean=True,
    include_city_std=True,
    include_city_max=True,
    flow_ref=100.0,
)
# Reward weights encode the coordination objective: improve city emissions,
# avoid uneven TAZ burden, and penalize local spillover into worsening TAZs.
COORDINATION_REWARD_CONFIG = CoordinationRewardConfig(
    city_delta_weight=1.0,
    terminal_delta_weight=1.0,
    imbalance_weight=0.25,
    spillover_weight=0.85,
    local_externality_weight=0.75,
    local_global_share_weight=0.10,
    local_bonus_clip=1.5,
    global_reward_clip=4.0,
    price_effort_weight=0.03,
    action_effort_weight=0.25,
    flow_ref=100.0,
)


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Normalize PPO advantages while handling tiny batches safely."""

    if advantages.numel() <= 1:
        return advantages
    std = advantages.std(unbiased=False)
    if float(std.item()) < 1e-8:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (std + 1e-8)


def _entropy_coef_now(episode_idx: int, total_episodes: int, start_value: float, end_value: float) -> float:
    """Use the same entropy decay schedule as the existing local PPO scripts."""

    progress = episode_idx / max(total_episodes - 1, 1)
    if progress <= ENTROPY_WARMUP_RATIO:
        return float(start_value)
    decay_progress = (progress - ENTROPY_WARMUP_RATIO) / max(1.0 - ENTROPY_WARMUP_RATIO, 1e-8)
    return float(start_value + (end_value - start_value) * decay_progress)


def _build_env_config() -> dict:
    """Reuse v11 environment configuration and keep the single-TAZ override compatible."""

    env_config = dict(ENV_V11_CONFIG)
    if SINGLE_TAZ_ID and SINGLE_TAZ_WAITING_ONLY_REWARD:
        env_config.update(
            waiting_reward_weight=1.0,
            emission_reward_weight=0.0,
            jam_reward_weight=0.0,
        )
    return env_config


def _build_global_context_tensor(
    hour: int,
    total_cars_random: int,
    demand_noise: float,
    baseline_penalty: Optional[float],
    device: torch.device,
) -> torch.Tensor:
    """Small episode-level context appended to every global observation."""

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


def _load_checkpoint(path: str) -> dict:
    """Load a Torch checkpoint with a clear error when the path is wrong."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_augmented_local_policy_weights(policy: ActorCriticV10, checkpoint_path: str, base_obs_dim: int):
    """
    Load a v11 local checkpoint into the v12 augmented local policy.

    The v12 policy has three extra input features. Existing weights are copied
    where shapes match; the new input weights are zeroed so the initial behavior
    matches the v11 policy as closely as possible.
    """

    checkpoint = _load_checkpoint(checkpoint_path)
    source = checkpoint.get("model_state_dict", checkpoint.get("local_model_state_dict", checkpoint))
    target = policy.state_dict()
    patched = {}
    for key, target_tensor in target.items():
        source_tensor = source.get(key)
        if source_tensor is None:
            patched[key] = target_tensor
            continue
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            patched[key] = source_tensor
            continue

        next_tensor = target_tensor.clone()
        if key == "extractor.net.0.weight" and source_tensor.ndim == 1:
            next_tensor[:base_obs_dim] = source_tensor[:base_obs_dim]
        elif key == "extractor.net.0.bias" and source_tensor.ndim == 1:
            next_tensor[:base_obs_dim] = source_tensor[:base_obs_dim]
        elif key == "extractor.net.1.weight" and source_tensor.ndim == 2:
            next_tensor[:, :base_obs_dim] = source_tensor[:, :base_obs_dim]
            next_tensor[:, base_obs_dim:] = 0.0
        else:
            print(f"[WARN] Skipping incompatible local checkpoint tensor: {key}")
        patched[key] = next_tensor
    policy.load_state_dict(patched)
    print(f"[INFO] Loaded local v11 weights into augmented coordination policy from {checkpoint_path}")


def _maybe_load_coordination_checkpoint(
    local_policy: ActorCriticV10,
    global_policy: ActorCriticV10,
    local_optim: Optional[torch.optim.Optimizer],
    global_optim: torch.optim.Optimizer,
    local_scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    global_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
) -> int:
    """Resume a previous v12 coordination run when configured."""

    if not LOAD_COORDINATION_CHECKPOINT:
        return 0

    checkpoint = _load_checkpoint(COORDINATION_CHECKPOINT_PATH)
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

    start_episode = int(checkpoint.get("episode", -1)) + 1 if RESUME_FROM_COORDINATION_EPISODE else 0
    if START_EPISODE_OVERRIDE is not None:
        start_episode = int(START_EPISODE_OVERRIDE)
    print(f"[INFO] Resumed coordination checkpoint from episode {start_episode}")
    return start_episode


def _write_history_header(csv_path: str, csv_cols: list[str]):
    """Create a fresh CSV history for this run."""

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=csv_cols).writeheader()


def _append_history(row: dict, csv_path: str, json_path: str, csv_cols: list[str], history: list[dict]):
    """Append one episode row to both CSV and JSON histories."""

    history.append(row)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_cols)
        writer.writerow({key: row.get(key, "") for key in csv_cols})
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def _global_ppo_update(
    global_policy: ActorCriticV10,
    global_optim: torch.optim.Optimizer,
    global_obs: torch.Tensor,
    global_action_mask: torch.Tensor,
    global_actions: torch.Tensor,
    global_logp: torch.Tensor,
    global_values: torch.Tensor,
    global_rewards: torch.Tensor,
    global_dones: torch.Tensor,
    episode_idx: int,
    total_episodes: int,
) -> dict:
    """Run one PPO update over the global policy rollout collected in an episode."""

    advantages, returns = compute_gae(
        global_rewards,
        global_values,
        global_dones,
        gamma=GLOBAL_GAMMA,
        lam=GLOBAL_LAMBDA,
    )
    entropy_coef = _entropy_coef_now(
        episode_idx=episode_idx,
        total_episodes=total_episodes,
        start_value=GLOBAL_ENTROPY_COEF,
        end_value=GLOBAL_ENTROPY_COEF_FINAL,
    )
    return ppo_update(
        global_policy,
        global_optim,
        global_obs,
        global_action_mask,
        global_actions,
        global_logp,
        _normalize_advantages(advantages.reshape(-1)),
        returns.reshape(-1),
        clip_ratio=GLOBAL_CLIP_RATIO,
        ppo_epochs=GLOBAL_PPO_UPDATE_EPOCHS,
        minibatch_size=GLOBAL_MINIBATCH_SIZE,
        entropy_coef=entropy_coef,
        value_coef=GLOBAL_VALUE_COEF,
        target_kl=GLOBAL_TARGET_KL,
    )


def main():
    """Build SUMO, policies, route schedule, and run the coordination training loop."""

    if TRAINING_MODE not in {"joint_finetune", "global_only"}:
        raise ValueError(f"Unsupported TRAINING_MODE={TRAINING_MODE}")
    if TRAINING_MODE == "global_only":
        print("[WARN] global_only trains prices but cannot change behavior unless local weights are later fine-tuned.")

    _set_global_seed(GLOBAL_SEED)

    sumo_standalone_dir = os.path.join(constants.SUMO_PATH, "standalone")
    log_file = os.path.join(sumo_standalone_dir, "command_log_coordination_v12.txt")
    sumo = Simulator(configurationPath=sumo_standalone_dir, logFile=log_file, tazTlsMapFile=constants.TAZ_FILE)
    planner = Planner(simulator=sumo)
    sumo.changeDetectorPath(detectorPath=constants.SUMO_NETWORK_PATH)

    selected_taz_ids = [str(SINGLE_TAZ_ID)] if SINGLE_TAZ_ID else None
    env_config = _build_env_config()
    env = SumoTazEnvV12Coordination(
        sumoSimulator=sumo,
        stepSize=600,
        selected_taz_ids=selected_taz_ids,
        **env_config,
    )
    runtime_device = torch.device(env.device)
    taz_ids = env.get_taz_ids()
    num_taz = len(taz_ids)
    base_obs_dim = int(env.agent_obs_dim)
    # Local observations are v11 observations plus own/neighbor/delta price.
    augmented_obs_dim = base_obs_dim + LOCAL_GLOBAL_CONTEXT_DIM
    act_dim = int(env.max_control_groups_per_taz)

    # TAZ adjacency is derived from SUMO TAZ polygons, not hardcoded.
    adjacency_matrix, adjacency_metadata = build_taz_adjacency_matrix(
        taz_ids=taz_ids,
        taz_additional_path=constants.TAZ_ADDITIONAL_FILE_PATH,
        distance_threshold=TAZ_ADJACENCY_DISTANCE_THRESHOLD,
        fallback_k=TAZ_ADJACENCY_FALLBACK_K,
        device=runtime_device,
    )

    dummy_taz_features = torch.zeros((num_taz, env.per_taz_feature_dim), dtype=torch.float32, device=runtime_device)
    dummy_context = torch.zeros(4, dtype=torch.float32, device=runtime_device)
    dummy_flow = torch.zeros((num_taz, num_taz), dtype=torch.float32, device=runtime_device)
    dummy_prices = torch.zeros(num_taz, dtype=torch.float32, device=runtime_device)
    global_obs_dim = int(
        build_coordination_observation(
            dummy_taz_features,
            adjacency_matrix,
            previous_flow_matrix=dummy_flow,
            previous_prices=dummy_prices,
            taz_ids=taz_ids,
            extra_context=dummy_context,
            config=COORDINATION_OBSERVATION_CONFIG,
        ).shape[-1]
    )

    local_policy = ActorCriticV10(augmented_obs_dim, act_dim, ACTION_BINS).to(runtime_device)
    if LOAD_LOCAL_POLICY:
        _load_augmented_local_policy_weights(local_policy, LOCAL_POLICY_CHECKPOINT_PATH, base_obs_dim=base_obs_dim)
    local_policy.train(mode=(TRAINING_MODE == "joint_finetune"))
    if TRAINING_MODE == "global_only":
        for parameter in local_policy.parameters():
            parameter.requires_grad_(False)

    global_policy = ActorCriticV10(global_obs_dim, num_taz, COORDINATION_PRICE_BINS).to(runtime_device)
    global_policy.train()

    local_optim = None
    local_scheduler = None
    if TRAINING_MODE == "joint_finetune":
        local_optim = torch.optim.Adam(local_policy.parameters(), lr=LR)
        local_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            local_optim, mode="max", factor=0.7, patience=15, threshold=0.005, threshold_mode="rel", min_lr=1e-5
        )
    global_optim = torch.optim.Adam(global_policy.parameters(), lr=GLOBAL_LR)
    global_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        global_optim, mode="max", factor=0.7, patience=15, threshold=0.005, threshold_mode="rel", min_lr=1e-5
    )

    start_episode = _maybe_load_coordination_checkpoint(
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

    ckpt_dir = os.path.join(SCRIPT_DIR, f"checkpoints{RUN_SUFFIX}")
    detail_dir = os.path.join(SCRIPT_DIR, f"training_details{RUN_SUFFIX}")
    csv_path = os.path.join(SCRIPT_DIR, f"training_history{RUN_SUFFIX}.csv")
    json_path = os.path.join(SCRIPT_DIR, f"training_history{RUN_SUFFIX}.json")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(detail_dir, exist_ok=True)

    csv_cols = [
        "episode",
        "date",
        "timeslot",
        "training_mode",
        "terminal_penalty",
        "terminal_parse_ok",
        "base_local_reward_mean",
        "shaped_local_reward_mean",
        "coordination_adjustment_mean",
        "global_reward_mean",
        "global_price_mean",
        "global_price_std",
        "spillover_mean",
        "imbalance_mean",
        "episode_flow_total",
        "local_ppo_total_loss",
        "global_ppo_total_loss",
        "local_learning_rate",
        "global_learning_rate",
    ]
    _write_history_header(csv_path, csv_cols)
    history = []
    baseline_cache = {}
    best_terminal_penalty = float("inf")
    best_episode = -1
    interrupted = False
    last_completed_episode = start_episode - 1

    print(
        f"[INFO] Coordination v12 | episodes={n_episodes} | taz={num_taz} | "
        f"base_obs_dim={base_obs_dim} | augmented_obs_dim={augmented_obs_dim} | global_obs_dim={global_obs_dim}"
    )
    print(f"[INFO] TAZ adjacency: {adjacency_metadata['neighbors_by_taz']}")

    for episode_idx in range(start_episode, n_episodes):
        episode_num = episode_idx + 1
        day, hour = episode_schedule[episode_idx]
        simulation_date = day.strftime("%Y-%m-%d")
        timeslot = f"{hour:02d}:00-{(hour + 1):02d}:00"
        timeslot_clean = timeslot.replace(":", "-")
        print(f"\n[COORD V12 EP {episode_num}/{n_episodes}] Date={simulation_date} Slot={timeslot}")

        total_cars = int(BASE_DEMAND * HOURLY_DEMAND_PROFILE[hour])
        demand_noise = _sample_demand_noise(episode_idx)
        total_cars_random = int(total_cars * demand_noise)
        route_folder_path, should_generate_routes = _resolve_route_folder(
            simulation_date=simulation_date,
            timeslot_clean=timeslot_clean,
            total_cars_random=total_cars_random,
        )
        os.makedirs(os.path.join(route_folder_path, "output"), exist_ok=True)
        if REUSE_DETERMINISTIC_ROUTE_FILES and not should_generate_routes:
            print(f"[ROUTE CACHE] {route_folder_path}")
        else:
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

        baseline_penalty, baseline_components, baseline_ran = _ensure_baseline_reference(
            env=env,
            sumo=sumo,
            hour=hour,
            route_folder_path=route_folder_path,
            cache=baseline_cache,
        )
        print(
            f"[BASELINE REF] {'Computed' if baseline_ran else 'Reused'} | "
            f"parse_ok={bool(baseline_components.get('parse_ok', False))} | "
            f"penalty={baseline_penalty if baseline_penalty is not None else 'n/a'}"
        )

        sumo.changeRouteFilePath(route_folder_path)
        sumo.changeTypePath(route_folder_path)
        if env_config["comparison_reward_enabled"] and baseline_penalty is not None:
            env.set_baseline_penalty(baseline_penalty, baseline_components)
        else:
            env.set_baseline_penalty(None)
        env.set_episode_context(hour=hour)
        td = env.reset()

        # Local rollout tensors are stored per local step and flattened before PPO.
        local_obs_list = []
        local_action_mask_list = []
        local_action_list = []
        local_logp_list = []
        local_value_list = []
        local_reward_list = []
        local_base_reward_list = []
        done_list = []

        global_obs_list = []
        global_action_mask_list = []
        global_action_list = []
        global_logp_list = []
        global_value_list = []
        global_reward_list = []
        global_done_list = []

        # The first global decision has no previous flow/action context.
        previous_flow_matrix = torch.zeros((num_taz, num_taz), dtype=torch.float32, device=runtime_device)
        previous_prices = torch.zeros(num_taz, dtype=torch.float32, device=runtime_device)
        previous_action_summary = None
        step_diagnostics = []
        episode_invalid = False
        local_rollout_state = _clone_state_dict_to_cpu(local_policy.state_dict())
        global_rollout_state = _clone_state_dict_to_cpu(global_policy.state_dict())

        while True:
            local_obs = td["observation"]
            action_mask = td["action_mask"].bool()
            taz_features = extract_taz_features(local_obs, env)
            global_context = _build_global_context_tensor(
                hour=hour,
                total_cars_random=total_cars_random,
                demand_noise=demand_noise,
                baseline_penalty=baseline_penalty,
                device=runtime_device,
            )
            global_obs = build_coordination_observation(
                taz_features,
                adjacency_matrix,
                previous_flow_matrix=previous_flow_matrix,
                previous_prices=previous_prices,
                previous_action_summary=previous_action_summary,
                taz_ids=taz_ids,
                extra_context=global_context,
                config=COORDINATION_OBSERVATION_CONFIG,
            )
            global_action_mask = torch.ones((1, num_taz), dtype=torch.bool, device=runtime_device)
            global_action_index, global_logp, global_value = global_policy.act(global_obs, global_action_mask)
            price_values = global_policy.action_values(global_action_index).squeeze(0)
            # Prices are injected into the local state before the local action is sampled.
            augmented_local_obs = augment_local_observation_with_prices(local_obs, price_values, adjacency_matrix)

            action_index, local_logp, local_value = local_policy.act(augmented_local_obs, action_mask)
            try:
                step_td = env.step(TensorDict({"action": action_index}, batch_size=[], device=runtime_device))
            except KeyboardInterrupt:
                interrupted = True
                episode_invalid = True
                print("[WARN] Interrupted during rollout.")
                break

            next_td = step_td["next"] if "next" in step_td.keys() else step_td
            base_reward_vec = next_td["reward"].detach().clone()
            terminated = bool(next_td["terminated"].item())
            truncated = bool(next_td["truncated"].item())
            done_value = 1.0 if (terminated or truncated) else 0.0
            reward_components = dict(getattr(env, "last_reward_components", {}) or {})
            terminal_reward = dict(reward_components.get("terminal_reward", {}) or {})
            terminal_penalty = float(terminal_reward.get("penalty", 0.0)) if bool(terminal_reward.get("parse_ok", False)) else None

            flow_matrix = torch.tensor(env.get_last_step_flow_matrix(), dtype=torch.float32, device=runtime_device)
            current_action_summary = summarize_step_actions(
                taz_ids=taz_ids,
                applied_actions=next_td["applied_action"],
                applied_deltas=next_td["applied_duration_delta"],
                action_mask=action_mask,
            )
            local_adjustment, global_reward_value, coordination_diag = compute_coordination_step_rewards(
                taz_ids=taz_ids,
                price_values=price_values,
                reward_components=reward_components,
                flow_matrix=flow_matrix,
                action_summary_by_taz=current_action_summary,
                baseline_penalty=baseline_penalty if done_value > 0.0 else None,
                terminal_penalty=terminal_penalty if done_value > 0.0 else None,
                config=COORDINATION_REWARD_CONFIG,
            )
            shaped_reward_vec = base_reward_vec
            if TRAINING_MODE == "joint_finetune":
                # This is where the global coordinator changes local learning.
                shaped_reward_vec = shaped_reward_vec + local_adjustment.to(device=runtime_device, dtype=base_reward_vec.dtype)

            local_obs_list.append(augmented_local_obs.detach().clone())
            local_action_mask_list.append(action_mask.detach().clone())
            local_action_list.append(action_index.detach().clone())
            local_logp_list.append(local_logp.detach().clone())
            local_value_list.append(local_value.detach().clone())
            local_reward_list.append(shaped_reward_vec.detach().clone())
            local_base_reward_list.append(base_reward_vec.detach().clone())
            done_list.append(torch.full_like(base_reward_vec, done_value))

            global_obs_list.append(global_obs.detach().clone())
            global_action_mask_list.append(global_action_mask.detach().clone())
            global_action_list.append(global_action_index.detach().clone())
            global_logp_list.append(global_logp.detach().clone())
            global_value_list.append(global_value.detach().clone())
            global_reward_list.append(torch.tensor([global_reward_value], dtype=torch.float32, device=runtime_device))
            global_done_list.append(torch.tensor([done_value], dtype=torch.float32, device=runtime_device))

            step_diagnostics.append(coordination_diag)
            # Feed the next global decision with the latest coordination context.
            previous_flow_matrix = flow_matrix
            previous_prices = price_values.detach()
            previous_action_summary = current_action_summary

            if terminated or truncated:
                break
            td = next_td

        if interrupted:
            break
        if episode_invalid or not local_reward_list:
            try:
                if env.sumo.isLoaded():
                    env.sumo.end()
            except Exception:
                pass
            continue

        local_stats = {
            "total_loss": "",
            "policy_loss": "",
            "value_loss": "",
            "clip_fraction": "",
            "approx_kl": "",
            "entropy_mean": "",
            "early_stop": "",
        }
        local_lr = ""
        if TRAINING_MODE == "joint_finetune":
            # The local policy learns from emission reward plus spillover correction.
            local_rewards_t = torch.stack(local_reward_list)
            local_values_t = torch.stack(local_value_list)
            dones_t = torch.stack(done_list)
            adv_t, ret_t = compute_gae(local_rewards_t, local_values_t, dones_t, gamma=GAMMA, lam=LAMBDA)
            local_entropy_coef = _entropy_coef_now(
                episode_idx=episode_idx,
                total_episodes=n_episodes,
                start_value=ENTROPY_COEF,
                end_value=ENTROPY_COEF_FINAL,
            )
            local_stats = ppo_update(
                local_policy,
                local_optim,
                torch.stack(local_obs_list).reshape(-1, augmented_obs_dim),
                torch.stack(local_action_mask_list).reshape(-1, act_dim),
                torch.stack(local_action_list).reshape(-1, act_dim),
                torch.stack(local_logp_list).reshape(-1),
                _normalize_advantages(adv_t.reshape(-1)),
                ret_t.reshape(-1),
                clip_ratio=CLIP_RATIO,
                ppo_epochs=PPO_UPDATE_EPOCHS,
                minibatch_size=MINIBATCH_SIZE,
                entropy_coef=local_entropy_coef,
                value_coef=VALUE_COEF,
                target_kl=TARGET_KL,
            )
            local_lr = float(local_optim.param_groups[0]["lr"])
            local_scheduler.step(float(torch.stack(local_reward_list).sum(dim=0).mean().item()))

        global_rewards_t = torch.stack(global_reward_list).reshape(-1, 1)
        # The global policy learns which TAZ prices reduce city penalty, imbalance, and spillover.
        global_stats = _global_ppo_update(
            global_policy=global_policy,
            global_optim=global_optim,
            global_obs=torch.cat(global_obs_list, dim=0).reshape(-1, global_obs_dim),
            global_action_mask=torch.cat(global_action_mask_list, dim=0).reshape(-1, num_taz),
            global_actions=torch.cat(global_action_list, dim=0).reshape(-1, num_taz),
            global_logp=torch.cat(global_logp_list, dim=0).reshape(-1),
            global_values=torch.cat(global_value_list, dim=0).reshape(-1, 1),
            global_rewards=global_rewards_t,
            global_dones=torch.stack(global_done_list).reshape(-1, 1),
            episode_idx=episode_idx,
            total_episodes=n_episodes,
        )
        global_lr = float(global_optim.param_groups[0]["lr"])
        global_scheduler.step(float(global_rewards_t.mean().item()))

        reward_components = dict(getattr(env, "last_reward_components", {}) or {})
        terminal_reward = dict(reward_components.get("terminal_reward", {}) or {})
        terminal_parse_ok = bool(terminal_reward.get("parse_ok", False))
        terminal_penalty = float(terminal_reward.get("penalty", 0.0)) if terminal_parse_ok else None

        base_local_reward = torch.stack(local_base_reward_list).sum(dim=0)
        shaped_local_reward = torch.stack(local_reward_list).sum(dim=0)
        adjustment_mean = float((shaped_local_reward - base_local_reward).mean().item())
        global_reward_mean = float(global_rewards_t.mean().item())
        price_values_all = torch.cat([
            global_policy.action_values(actions).reshape(-1).detach().cpu()
            for actions in global_action_list
        ])
        spillover_mean = float(np.mean([diag.get("spillover", 0.0) for diag in step_diagnostics]))
        imbalance_mean = float(np.mean([diag.get("imbalance", 0.0) for diag in step_diagnostics]))
        episode_flow_total = float(env.get_episode_flow_matrix().sum())

        row = {
            "episode": int(episode_idx),
            "date": simulation_date,
            "timeslot": timeslot,
            "training_mode": TRAINING_MODE,
            "terminal_penalty": terminal_penalty if terminal_penalty is not None else "",
            "terminal_parse_ok": bool(terminal_parse_ok),
            "base_local_reward_mean": float(base_local_reward.mean().item()),
            "shaped_local_reward_mean": float(shaped_local_reward.mean().item()),
            "coordination_adjustment_mean": adjustment_mean,
            "global_reward_mean": global_reward_mean,
            "global_price_mean": float(price_values_all.float().mean().item()),
            "global_price_std": float(price_values_all.float().std(unbiased=False).item()),
            "spillover_mean": spillover_mean,
            "imbalance_mean": imbalance_mean,
            "episode_flow_total": episode_flow_total,
            "local_ppo_total_loss": local_stats["total_loss"],
            "global_ppo_total_loss": global_stats["total_loss"],
            "local_learning_rate": local_lr,
            "global_learning_rate": global_lr,
        }
        _append_history(row, csv_path, json_path, csv_cols, history)

        detail_path = os.path.join(detail_dir, f"episode_{episode_idx:04d}.json")
        with open(detail_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "episode": int(episode_idx),
                    "date": simulation_date,
                    "timeslot": timeslot,
                    "route_folder_path": route_folder_path,
                    "adjacency": adjacency_metadata,
                    "baseline": baseline_components,
                    "terminal_reward": terminal_reward,
                    "coordination_step_diagnostics": step_diagnostics,
                    "flow_diagnostics": env.get_coordination_flow_diagnostics(),
                    "local_stats": local_stats,
                    "global_stats": global_stats,
                },
                handle,
                indent=2,
            )

        checkpoint_payload = {
            # Store both policies and the metadata needed to replay/evaluate the architecture.
            "episode": int(episode_idx),
            "training_mode": TRAINING_MODE,
            "local_model_state_dict": local_policy.state_dict(),
            "global_model_state_dict": global_policy.state_dict(),
            "local_optimizer_state_dict": local_optim.state_dict() if local_optim is not None else None,
            "global_optimizer_state_dict": global_optim.state_dict(),
            "local_scheduler_state_dict": local_scheduler.state_dict() if local_scheduler is not None else None,
            "global_scheduler_state_dict": global_scheduler.state_dict(),
            "env_config": env_config,
            "taz_ids": taz_ids,
            "tls_ids": env.get_tls_ids(),
            "control_groups_by_taz": env.get_control_groups_by_taz(),
            "base_obs_dim": int(base_obs_dim),
            "augmented_obs_dim": int(augmented_obs_dim),
            "local_global_context_dim": int(LOCAL_GLOBAL_CONTEXT_DIM),
            "local_action_bins": ACTION_BINS,
            "coordination_price_bins": COORDINATION_PRICE_BINS,
            "coordination_observation_config": vars(COORDINATION_OBSERVATION_CONFIG),
            "coordination_reward_config": vars(COORDINATION_REWARD_CONFIG),
            "adjacency_metadata": adjacency_metadata,
            "best_episode": best_episode,
            "best_terminal_penalty": best_terminal_penalty if math.isfinite(best_terminal_penalty) else None,
        }
        torch.save(checkpoint_payload, os.path.join(ckpt_dir, f"checkpoint_coordination_v12_episode{episode_idx}.pt"))

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
            torch.save(best_payload, os.path.join(ckpt_dir, "checkpoint_coordination_v12_best.pt"))

        last_completed_episode = int(episode_idx)
        print(
            f"[COORD RESULT] EP {episode_num}/{n_episodes} | "
            f"TerminalPenalty {terminal_penalty if terminal_penalty is not None else 'n/a'} | "
            f"BaseLocalMean {row['base_local_reward_mean']:.4f} | "
            f"ShapedLocalMean {row['shaped_local_reward_mean']:.4f} | "
            f"GlobalRewardMean {global_reward_mean:.4f} | "
            f"PriceMean {row['global_price_mean']:.3f} | "
            f"Spillover {spillover_mean:.4f} | "
            f"Flow {episode_flow_total:.0f}"
        )

    final_payload = {
        "episode": int(last_completed_episode),
        "training_mode": TRAINING_MODE,
        "local_model_state_dict": local_policy.state_dict(),
        "global_model_state_dict": global_policy.state_dict(),
        "local_optimizer_state_dict": local_optim.state_dict() if local_optim is not None else None,
        "global_optimizer_state_dict": global_optim.state_dict(),
        "local_scheduler_state_dict": local_scheduler.state_dict() if local_scheduler is not None else None,
        "global_scheduler_state_dict": global_scheduler.state_dict(),
        "env_config": env_config,
        "taz_ids": taz_ids,
        "tls_ids": env.get_tls_ids(),
        "control_groups_by_taz": env.get_control_groups_by_taz(),
        "base_obs_dim": int(base_obs_dim),
        "augmented_obs_dim": int(augmented_obs_dim),
        "local_global_context_dim": int(LOCAL_GLOBAL_CONTEXT_DIM),
        "local_action_bins": ACTION_BINS,
        "coordination_price_bins": COORDINATION_PRICE_BINS,
        "coordination_observation_config": vars(COORDINATION_OBSERVATION_CONFIG),
        "coordination_reward_config": vars(COORDINATION_REWARD_CONFIG),
        "adjacency_metadata": adjacency_metadata,
        "best_episode": best_episode,
        "best_terminal_penalty": best_terminal_penalty if math.isfinite(best_terminal_penalty) else None,
        "interrupted": bool(interrupted),
    }
    torch.save(final_payload, os.path.join(ckpt_dir, "checkpoint_coordination_v12_final.pt"))

    try:
        if env.sumo.isLoaded():
            env.sumo.end()
    except Exception:
        pass


if __name__ == "__main__":
    main()
