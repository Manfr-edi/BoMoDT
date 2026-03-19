import csv
import json
import math
import os
import random
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Categorical

from libraries import constants
from libraries.classes.Planner import Planner
from libraries.classes.SumoSimulator import Simulator
from libraries.constants import EDGE_DATA_FILE_PATH, PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH, SUMO_PATH
from libraries.utils.preprocessingUtils import generateEdgeDataFile
from taz_rl.rlenv.local_taz_env_v10 import SumoTazEnvV10


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================== CONFIG ====================
TRAIN_START_DATE = datetime(2024, 2, 2)
N_TRAIN_DAYS = 200
FOCUS_HOURS = [8]
TRAIN_ON_SAME_DAY = True

HOURLY_DEMAND_PROFILE = {
    0: 0.2, 1: 0.15, 2: 0.1, 3: 0.1, 4: 0.2,
    5: 0.4, 6: 0.7, 7: 1.2, 8: 1.5,
    9: 1.0, 10: 0.8, 11: 0.9,
    12: 1.1, 13: 1.0, 14: 0.9, 15: 1.0,
    16: 1.3, 17: 1.6, 18: 1.4, 19: 1.0,
    20: 0.8, 21: 0.6, 22: 0.4, 23: 0.3,
}
BASE_DEMAND = 5000
DEMAND_NOISE_RANGE = (1.0, 1.0)
GLOBAL_SEED = 42
REUSE_DETERMINISTIC_ROUTE_FILES = True
ROUTE_RANDOM_TRIP_SEED = 42
ROUTE_SAMPLER_SEED = 42
ROUTE_SAMPLER_THREADS = 1
ROUTE_CACHE_ROOT = os.path.join(SUMO_PATH, "routes_v10_deterministic")
LEGACY_ROUTE_CACHE_ROOTS = [
    os.path.join(SUMO_PATH, "routes_v9_deterministic"),
    os.path.join(SUMO_PATH, "routes_v8_deterministic"),
    os.path.join(SUMO_PATH, "routes_v7_deterministic"),
]

ACTION_BINS = (-10.0, -5.0, 0.0, 5.0, 10.0)
ZERO_ACTION_INDEX = ACTION_BINS.index(0.0)

ENV_V10_CONFIG = dict(
    speed_norm=10.0,
    jam_norm=10.0,
    warmupSteps=0,
    cooldownSteps=12,
    min_green=25,
    max_green=120,
    metric_clip=2.0,
    reward_clip=4.0,
    dense_abs_penalty_weight=1.0,
    dense_delta_reward_weight=0.40,
    terminal_bonus_weight=1.0,
    comparison_reward_enabled=True,
    waiting_reward_weight=0.70,
    emission_reward_weight=0.20,
    jam_reward_weight=0.10,
    emission_co2_mix_weight=0.50,
    emission_nox_mix_weight=0.50,
    tls_waiting_time_ref=1000.0,
    tls_co2_ref=80000.0,
    tls_nox_ref=40.0,
    tls_active_vehicle_ref=80.0,
    taz_waiting_time_ref=6000.0,
    taz_co2_ref=300000.0,
    taz_nox_ref=150.0,
    terminal_waiting_time_ref=500000.0,
    terminal_co2_ref=5000000000.0,
    terminal_nox_ref=1800000.0,
    tls_veh_total_ref=120.0,
    tls_pressure_ref=10000.0,
    taz_veh_total_ref=2500.0,
    taz_std_occupancy_ref=0.25,
    max_jam_len_ref=120.0,
)

# PPO
GAMMA = 0.99
LAMBDA = 0.95
LR = 2e-4
CLIP_RATIO = 0.12
ENTROPY_COEF = 0.008
ENTROPY_COEF_FINAL = 0.001
ENTROPY_WARMUP_RATIO = 0.25
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.015
PPO_UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 64

EXTRACTOR_HIDDEN = 256
EXTRACTOR_OUT = 128

# Checkpoints
LOAD_MODEL = False
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints_v10", "checkpoint_ppo_v10_final.pt")
RESUME_FROM_CHECKPOINT_EPISODE = True
START_EPISODE_OVERRIDE = None


# ==================== MODEL ====================
class FeatureExtractor(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ActorCriticV10(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, action_bins: tuple[float, ...]):
        super().__init__()
        self.num_bins = int(len(action_bins))
        self.zero_action_index = int(action_bins.index(0.0))
        self.extractor = FeatureExtractor(obs_dim, hidden_dim=EXTRACTOR_HIDDEN, out_dim=EXTRACTOR_OUT)
        self.actor = nn.Sequential(
            nn.Linear(EXTRACTOR_OUT, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.logits_head = nn.Linear(32, act_dim * self.num_bins)
        self.critic = nn.Sequential(
            nn.Linear(EXTRACTOR_OUT, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.act_dim = int(act_dim)
        self.register_buffer("action_bins", torch.tensor(action_bins, dtype=torch.float32), persistent=False)

        nn.init.orthogonal_(self.logits_head.weight, gain=0.01)
        nn.init.constant_(self.logits_head.bias, 0.0)

    def _apply_action_mask(self, logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        if action_mask is None:
            return logits
        valid_mask = action_mask.bool().unsqueeze(-1)
        forced_logits = torch.full_like(logits, -1e9)
        forced_logits[..., self.zero_action_index] = 0.0
        return torch.where(valid_mask, logits, forced_logits)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None):
        feat = self.extractor(obs)
        logits = self.logits_head(self.actor(feat)).reshape(-1, self.act_dim, self.num_bins)
        logits = self._apply_action_mask(logits, action_mask)
        value = self.critic(feat).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor, action_mask: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(obs, action_mask)
        dist = Categorical(logits=logits)
        if deterministic:
            action_index = logits.argmax(dim=-1)
        else:
            action_index = dist.sample()
        valid_mask = action_mask.bool()
        action_index = torch.where(
            valid_mask,
            action_index,
            torch.full_like(action_index, self.zero_action_index),
        )
        logp = (dist.log_prob(action_index) * valid_mask.to(dtype=logits.dtype)).sum(dim=-1)
        return action_index, logp, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor):
        logits, value = self.forward(obs, action_mask)
        dist = Categorical(logits=logits)
        valid_mask = action_mask.bool()
        safe_actions = torch.where(
            valid_mask,
            actions.long(),
            torch.full_like(actions.long(), self.zero_action_index),
        )
        logp = (dist.log_prob(safe_actions) * valid_mask.to(dtype=logits.dtype)).sum(dim=-1)
        entropy = (dist.entropy() * valid_mask.to(dtype=logits.dtype)).sum(dim=-1)
        return logp, value, entropy

    def action_values(self, action_indices: torch.Tensor) -> torch.Tensor:
        safe_indices = action_indices.long().clamp(0, self.num_bins - 1)
        return self.action_bins[safe_indices]


# ==================== RL UTILS ====================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    t_max, n_agents = rewards.shape
    adv = torch.zeros_like(rewards)
    gae = torch.zeros(n_agents, dtype=rewards.dtype, device=rewards.device)
    next_value = torch.zeros(n_agents, dtype=rewards.dtype, device=rewards.device)

    for t_idx in reversed(range(t_max)):
        mask = 1.0 - dones[t_idx]
        delta = rewards[t_idx] + gamma * next_value * mask - values[t_idx]
        gae = delta + gamma * lam * mask * gae
        adv[t_idx] = gae
        next_value = values[t_idx]
    returns = adv + values
    return adv, returns


def ppo_update(
    policy,
    optim,
    obs,
    action_mask,
    act,
    logp_old,
    adv,
    ret,
    clip_ratio=0.2,
    ppo_epochs=4,
    minibatch_size=256,
    entropy_coef=0.01,
    value_coef=0.5,
    target_kl=None,
):
    n_samples = obs.shape[0]
    idx = torch.arange(n_samples, device=obs.device)
    clip_fracs = []
    kls = []
    entropies = []
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_total_loss = 0.0

    batch_size = min(int(minibatch_size), n_samples)
    planned_updates = int(ppo_epochs) * max(int(math.ceil(n_samples / batch_size)), 1)
    performed_updates = 0
    early_stop = False
    epochs_performed = 0

    for epoch_idx in range(int(ppo_epochs)):
        perm = idx[torch.randperm(n_samples)]
        epoch_kls = []
        for start in range(0, n_samples, batch_size):
            mb_idx = perm[start:start + batch_size]
            logp, values, entropy = policy.evaluate(obs[mb_idx], act[mb_idx], action_mask[mb_idx])
            ratio = torch.exp(torch.clamp(logp - logp_old[mb_idx], -20.0, 20.0))
            surr1 = ratio * adv[mb_idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values, ret[mb_idx])
            entropy_loss = -entropy.mean() * entropy_coef
            total_loss = policy_loss + value_coef * value_loss + entropy_loss

            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optim.step()
            performed_updates += 1

            clipped = (torch.abs(ratio - 1.0) > clip_ratio).float().mean().item()
            approx_kl = (logp_old[mb_idx] - logp).mean().item()
            epoch_kls.append(float(approx_kl))
            clip_fracs.append(float(clipped))
            kls.append(float(approx_kl))
            entropies.append(float(entropy.mean().item()))
            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_total_loss = float(total_loss.item())

        epochs_performed += 1
        epoch_mean_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
        if target_kl is not None and epoch_mean_kl > float(target_kl) * 1.25 and epoch_idx >= 1:
            early_stop = True
            break

    return {
        "total_loss": float(last_total_loss),
        "policy_loss": float(last_policy_loss),
        "value_loss": float(last_value_loss),
        "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        "approx_kl": float(np.mean(kls)) if kls else 0.0,
        "entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "early_stop": bool(early_stop),
        "epochs_performed": int(epochs_performed),
        "epochs_planned": int(ppo_epochs),
        "update_fraction": float(performed_updates / max(planned_updates, 1)),
    }


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
        "action_count_neg5": int((flat == -5.0).sum().item()),
        "action_count_zero": int((flat == 0.0).sum().item()),
        "action_count_pos5": int((flat == 5.0).sum().item()),
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
        valid_mask = action_mask[taz_idx].bool()
        selected_valid = selected_actions[:, taz_idx, :][:, valid_mask].reshape(-1)
        delta_valid = applied_deltas[:, taz_idx, :][:, valid_mask].reshape(-1)
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
                "neg5": int((selected_valid == -5.0).sum().item()),
                "zero": int((selected_valid == 0.0).sum().item()),
                "pos5": int((selected_valid == 5.0).sum().item()),
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


def _move_optimizer_state_to_device(optim: torch.optim.Optimizer, device: torch.device):
    for state in optim.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


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

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    policy.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        try:
            optim.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as exc:
            print(f"[WARN] Could not load optimizer state: {exc}")
    if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as exc:
            print(f"[WARN] Could not load scheduler state: {exc}")

    if RESUME_FROM_CHECKPOINT_EPISODE:
        start_episode = int(ckpt.get("episode", -1)) + 1
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


def _prepare_history_files(csv_path: str, json_path: str, csv_cols: list[str], start_episode: int):
    history = _load_existing_history(json_path) if LOAD_MODEL else []
    history = [
        row for row in history
        if int(row.get("episode", -1)) < int(start_episode)
    ]

    if history:
        _rewrite_history_csv(csv_path, csv_cols, history)
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=csv_cols).writeheader()

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    if history:
        best_row = max(history, key=lambda row: float(row.get("episode_reward_mean", float("-inf"))))
        best_reward = float(best_row.get("episode_reward_mean", float("-inf")))
        best_episode = int(best_row.get("episode", -1))
    else:
        best_reward = float("-inf")
        best_episode = -1
    return history, best_reward, best_episode


def _ensure_baseline_reference(env, sumo, hour: int, route_folder_path: str, cache: dict) -> tuple[Optional[float], dict, bool]:
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


# ==================== MAIN ====================
def main():
    _set_global_seed(GLOBAL_SEED)

    sumo_standalone_dir = os.path.join(constants.SUMO_PATH, "standalone")
    log_file = os.path.join(sumo_standalone_dir, "command_log_v10.txt")
    sumo = Simulator(configurationPath=sumo_standalone_dir, logFile=log_file, tazTlsMapFile=constants.TAZ_FILE)
    planner = Planner(simulator=sumo)
    sumo.changeDetectorPath(detectorPath=constants.SUMO_NETWORK_PATH)

    env = SumoTazEnvV10(sumoSimulator=sumo, stepSize=300, **ENV_V10_CONFIG)
    runtime_device = torch.device(env.device)
    taz_ids = env.get_taz_ids()
    tls_ids = env.get_tls_ids()
    num_taz = len(taz_ids)
    obs_dim = int(env.agent_obs_dim)
    act_dim = int(env.max_tls_per_taz)

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

    csv_path = os.path.join(SCRIPT_DIR, "training_history_v10.csv")
    json_path = os.path.join(SCRIPT_DIR, "training_history_v10.json")
    ckpt_dir = os.path.join(SCRIPT_DIR, "checkpoints_v10")
    detail_dir = os.path.join(SCRIPT_DIR, "training_details_v10")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(detail_dir, exist_ok=True)

    csv_cols = [
        "episode", "date", "timeslot", "num_taz", "num_tls", "num_steps", "reward_basis",
        "episode_reward_mean", "episode_reward_std", "episode_reward_min", "episode_reward_max", "episode_reward_sum",
        "final_penalty_mean", "final_penalty_std", "final_penalty_min", "final_penalty_max",
        "avg_action_selected", "sample_action_std_selected",
        "action_count_neg10", "action_count_neg5", "action_count_zero", "action_count_pos5", "action_count_pos10",
        "avg_applied_duration_delta", "sample_applied_duration_delta_std", "applied_duration_nonzero_ratio",
        "returns_mean", "returns_std", "avg_value_estimate",
        "learning_rate", "learning_rate_after_step",
        "ppo_total_loss", "ppo_policy_loss", "ppo_value_loss",
        "policy_clip_fraction", "approx_kl", "policy_entropy_mean",
        "ppo_early_stop", "ppo_epochs_performed", "ppo_epochs_planned",
        "baseline_parse_ok", "baseline_penalty",
        "terminal_parse_ok", "terminal_penalty", "terminal_bonus", "comparison_delta_penalty",
        "details_path",
    ]
    history, best_reward, best_episode = _prepare_history_files(csv_path, json_path, csv_cols, start_episode)
    interrupted = False
    last_completed_episode = start_episode - 1
    baseline_cache = {}

    print(f"[INFO] v10 TAZ count: {len(taz_ids)} | IDs: {taz_ids}")
    print(f"[INFO] v10 TLS count: {len(tls_ids)}")
    print(f"[INFO] Observation dim per TAZ: {obs_dim} | TLS slots per TAZ: {act_dim}")
    print(f"[INFO] Global seed: {GLOBAL_SEED}")
    print(
        f"[INFO] Same-day training: {TRAIN_ON_SAME_DAY} | "
        f"start_date={TRAIN_START_DATE.strftime('%Y-%m-%d')} | focus_hours={FOCUS_HOURS}"
    )
    print(
        f"[INFO] Deterministic route cache: {REUSE_DETERMINISTIC_ROUTE_FILES} | "
        f"primary={ROUTE_CACHE_ROOT} | randomTrips seed={ROUTE_RANDOM_TRIP_SEED} | "
        f"routeSampler seed={ROUTE_SAMPLER_SEED} | threads={ROUTE_SAMPLER_THREADS}"
    )

    for episode_idx in range(start_episode, n_episodes):
        episode_num = episode_idx + 1
        day, hour = episode_schedule[episode_idx]
        simulation_date = day.strftime("%Y-%m-%d")
        timeslot = f"{hour:02d}:00-{(hour + 1):02d}:00"
        timeslot_clean = timeslot.replace(":", "-")
        print(f"\n[RL EP {episode_num}/{n_episodes}] Date={simulation_date} Slot={timeslot}")

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
            print(
                f"[ROUTE CACHE] RL EP {episode_num}/{n_episodes} | "
                f"Reusing deterministic trip/route files from {route_folder_path}"
            )
        else:
            print(
                f"[ROUTE GEN] RL EP {episode_num}/{n_episodes} | "
                f"Generating deterministic trip/route files at {route_folder_path}"
            )
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
        if baseline_penalty is not None:
            env.set_baseline_penalty(baseline_penalty, baseline_components)
            baseline_msg = f"penalty={baseline_penalty:.4f}"
        else:
            env.set_baseline_penalty(None)
            baseline_msg = "penalty unavailable"
        print(
            f"[BASELINE REF] RL EP {episode_num}/{n_episodes} | "
            f"{'Computed' if baseline_ran else 'Reused'} | parse_ok={baseline_parse_ok} | {baseline_msg}"
        )

        sumo.changeRouteFilePath(route_folder_path)
        sumo.changeTypePath(route_folder_path)
        env.set_episode_context(hour=hour)
        print(f"[RL RUN] RL EP {episode_num}/{n_episodes} | Starting per-TAZ RL simulation.")
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
            entropy_coef=entropy_coef_now,
            value_coef=VALUE_COEF,
            target_kl=TARGET_KL,
        )
        lr_before_step = float(optim.param_groups[0]["lr"])
        episode_reward_by_taz_tensor = rewards_t.sum(dim=0).cpu()
        reward_mean = float(episode_reward_by_taz_tensor.mean().item())
        scheduler.step(reward_mean)
        lr_after_step = float(optim.param_groups[0]["lr"])

        reward_components = dict(getattr(env, "last_reward_components", {}) or {})
        penalty_by_taz = {
            taz: float((reward_components.get("penalty_by_taz", {}) or {}).get(taz, 0.0))
            for taz in taz_ids
        }
        penalty_details_by_taz = dict(reward_components.get("penalty_details_by_taz", {}) or {})
        dense_reward_by_taz = {
            taz: float((reward_components.get("dense_reward_by_taz", {}) or {}).get(taz, 0.0))
            for taz in taz_ids
        }
        delta_penalty_by_taz = {
            taz: float((reward_components.get("delta_penalty_by_taz", {}) or {}).get(taz, 0.0))
            for taz in taz_ids
        }
        penalty_vector = torch.tensor([penalty_by_taz[taz] for taz in taz_ids], dtype=torch.float32)
        terminal_reward = dict(reward_components.get("terminal_reward", {}) or {})
        terminal_parse_ok = bool(terminal_reward.get("parse_ok", False))
        terminal_penalty = float(terminal_reward.get("penalty", 0.0)) if terminal_parse_ok else ""
        terminal_bonus = float(reward_components.get("terminal_bonus", 0.0))
        comparison_delta_penalty = float(terminal_reward.get("delta_penalty", 0.0)) if terminal_parse_ok else ""

        action_mask_stats = batch_mask_3d.bool()
        avg_action_selected, std_action_selected = _masked_mean_std(batch_applied_act_3d, action_mask_stats)
        avg_applied_delta, std_applied_delta = _masked_mean_std(batch_applied_delta_3d, action_mask_stats)
        nonzero_ratio = float((_masked_values(batch_applied_delta_3d.abs(), action_mask_stats) > 1e-3).float().mean().item())
        action_counts = _count_discrete_actions(batch_applied_act_3d, action_mask_stats)
        action_summary_by_taz = _build_action_summary_by_taz(
            selected_actions=batch_applied_act_3d,
            applied_deltas=batch_applied_delta_3d,
            action_mask=batch_mask_3d[0],
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
                    "delta_penalty_by_taz": delta_penalty_by_taz,
                    "final_penalty_details_by_taz": penalty_details_by_taz,
                    "baseline_reference_components": baseline_components,
                    "terminal_reward_components": terminal_reward,
                    "action_summary_by_taz": action_summary_by_taz,
                },
                handle,
                indent=2,
            )

        row = {
            "episode": int(episode_idx),
            "date": simulation_date,
            "timeslot": timeslot,
            "num_taz": int(num_taz),
            "num_tls": int(len(tls_ids)),
            "num_steps": int(len(rewards_list)),
            "reward_basis": str(reward_components.get("reward_basis", "taz_absolute_penalty_plus_delta_with_terminal_baseline_bonus")),
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
            "returns_mean": float(batch_ret.mean().item()),
            "returns_std": float(batch_ret.std(unbiased=False).item()),
            "avg_value_estimate": float(values_t.mean().item()),
            "learning_rate": lr_before_step,
            "learning_rate_after_step": lr_after_step,
            "ppo_total_loss": stats["total_loss"],
            "ppo_policy_loss": stats["policy_loss"],
            "ppo_value_loss": stats["value_loss"],
            "policy_clip_fraction": stats["clip_fraction"],
            "approx_kl": stats["approx_kl"],
            "policy_entropy_mean": stats["entropy_mean"],
            "ppo_early_stop": stats["early_stop"],
            "ppo_epochs_performed": stats["epochs_performed"],
            "ppo_epochs_planned": stats["epochs_planned"],
            "baseline_parse_ok": baseline_parse_ok,
            "baseline_penalty": float(baseline_penalty) if baseline_penalty is not None else "",
            "terminal_parse_ok": terminal_parse_ok,
            "terminal_penalty": terminal_penalty,
            "terminal_bonus": terminal_bonus,
            "comparison_delta_penalty": comparison_delta_penalty,
            "details_path": detail_path,
        }
        history.append(row)

        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=csv_cols).writerow({key: row.get(key, "") for key in csv_cols})
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        ckpt_payload = {
            "episode": episode_idx,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "reward_mean": reward_mean,
            "env_config": ENV_V10_CONFIG,
            "focus_hours": FOCUS_HOURS,
            "taz_ids": taz_ids,
            "tls_ids": tls_ids,
            "tls_count": len(tls_ids),
            "max_tls_per_taz": act_dim,
            "action_bins": ACTION_BINS,
            "ppo": stats,
            "interrupted": False,
        }
        torch.save(ckpt_payload, os.path.join(ckpt_dir, f"checkpoint_ppo_v10_episode{episode_idx}.pt"))

        if reward_mean > best_reward:
            best_reward = float(reward_mean)
            best_episode = int(episode_idx)
            best_payload = dict(ckpt_payload)
            best_payload["is_best"] = True
            torch.save(best_payload, os.path.join(ckpt_dir, "checkpoint_ppo_v10_best.pt"))

        last_completed_episode = episode_idx
        best_taz = max(penalty_by_taz, key=lambda key: dense_reward_by_taz.get(key, float("-inf")))
        worst_taz = max(penalty_by_taz, key=lambda key: penalty_by_taz[key])
        print(
            f"[RL RESULT] RL EP {episode_num}/{n_episodes} | "
            f"RewardMean {row['episode_reward_mean']:.4f} | RewardStd {row['episode_reward_std']:.4f} | "
            f"PenaltyMean {row['final_penalty_mean']:.4f} | ActionMean {row['avg_action_selected']:.2f} | "
            f"AppliedNZ {row['applied_duration_nonzero_ratio']:.2f} | KL {row['approx_kl']:.4f} | "
            f"Entropy {row['policy_entropy_mean']:.4f} | "
            f"BestDenseTAZ {best_taz}={dense_reward_by_taz.get(best_taz, 0.0):.4f} | "
            f"WorstPenaltyTAZ {worst_taz}={penalty_by_taz[worst_taz]:.4f} | "
            f"TerminalBonus {terminal_bonus:.4f}"
        )

    try:
        env.sumo.end()
    except Exception:
        pass

    torch.save(
        {
            "episode": last_completed_episode,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history_len": len(history),
            "env_config": ENV_V10_CONFIG,
            "focus_hours": FOCUS_HOURS,
            "taz_ids": taz_ids,
            "tls_ids": tls_ids,
            "tls_count": len(tls_ids),
            "action_bins": ACTION_BINS,
            "best_episode": best_episode,
            "best_reward": best_reward,
            "interrupted": bool(interrupted),
        },
        os.path.join(ckpt_dir, "checkpoint_ppo_v10_final.pt"),
    )

    if interrupted:
        print("[INFO] Training interrupted. Partial checkpoint saved.")
    else:
        print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
