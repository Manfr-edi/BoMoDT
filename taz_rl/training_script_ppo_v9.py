import csv
import json
import math
import os
import random
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from torch.distributions.transforms import AffineTransform, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from libraries import constants
from libraries.classes.Planner import Planner
from libraries.classes.SumoSimulator import Simulator
from libraries.constants import EDGE_DATA_FILE_PATH, PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH, SUMO_PATH
from libraries.utils.preprocessingUtils import generateEdgeDataFile
from taz_rl.rlenv.local_taz_env_v9 import SumoTazEnvV9


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
ROUTE_CACHE_ROOT = os.path.join(SUMO_PATH, "routes_v9_deterministic")
LEGACY_ROUTE_CACHE_ROOTS = [
    os.path.join(SUMO_PATH, "routes_v8_deterministic"),
    os.path.join(SUMO_PATH, "routes_v7_deterministic"),
]

ENV_V9_CONFIG = dict(
    speed_norm=10.0,
    jam_norm=10.0,
    warmupSteps=0,
    cooldownSteps=12,
    min_green=25,
    max_green=120,
    reward_clip=2.0,
    waiting_reward_weight=0.35,
    emission_reward_weight=0.30,
    jam_reward_weight=0.35,
    emission_co2_mix_weight=0.50,
    emission_nox_mix_weight=0.50,
    tls_waiting_time_ref=1000.0,
    tls_co2_ref=80000.0,
    tls_nox_ref=40.0,
    tls_active_vehicle_ref=80.0,
    taz_waiting_time_ref=6000.0,
    taz_co2_ref=300000.0,
    taz_nox_ref=150.0,
    tls_veh_total_ref=120.0,
    tls_pressure_ref=10000.0,
    taz_veh_total_ref=2500.0,
    taz_std_occupancy_ref=0.25,
    max_jam_len_ref=120.0,
)

# PPO
GAMMA = 0.99
LAMBDA = 0.95
LR = 4e-4
CLIP_RATIO = 0.2
ENTROPY_COEF = 0.0025
ENTROPY_COEF_FINAL = 0.00015
ENTROPY_WARMUP_RATIO = 0.20
VALUE_COEF = 0.3
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.06
PPO_UPDATE_EPOCHS = 6
MINIBATCH_SIZE = 64
ACTION_LIMIT = 15.0

EXTRACTOR_HIDDEN = 256
EXTRACTOR_OUT = 128

# Checkpoints
LOAD_MODEL = False
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints_v9", "checkpoint_ppo_v9_final.pt")
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


class ActorCriticV9(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, action_limit: float = 15.0):
        super().__init__()
        self.action_limit = float(action_limit)
        self.extractor = FeatureExtractor(obs_dim, hidden_dim=EXTRACTOR_HIDDEN, out_dim=EXTRACTOR_OUT)

        self.actor = nn.Sequential(
            nn.Linear(EXTRACTOR_OUT, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(32, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.35))

        self.critic = nn.Sequential(
            nn.Linear(EXTRACTOR_OUT, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)

    @staticmethod
    def _mask_to_float(action_mask: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if action_mask is None:
            return torch.ones_like(ref)
        return action_mask.to(dtype=ref.dtype, device=ref.device)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None):
        feat = self.extractor(obs)
        mean = self.mean_head(self.actor(feat))
        mask = self._mask_to_float(action_mask, mean)
        mean = mean * mask
        std = torch.exp(self.log_std).clamp(0.05, 1.5).unsqueeze(0).expand_as(mean)
        value = self.critic(feat).squeeze(-1)
        return mean, std, value

    def dist(self, mean: torch.Tensor, std: torch.Tensor):
        base = Normal(mean, std)
        dist = TransformedDistribution(
            base,
            [TanhTransform(cache_size=1), AffineTransform(loc=0.0, scale=self.action_limit)],
        )
        return dist, base

    def act(self, obs: torch.Tensor, action_mask: torch.Tensor, deterministic: bool = False):
        mean, std, value = self.forward(obs, action_mask)
        dist, _ = self.dist(mean, std)
        if deterministic:
            action = torch.tanh(mean) * self.action_limit
        else:
            action = dist.sample()
        mask = action_mask.to(dtype=action.dtype, device=action.device)
        action = action * mask
        logp = (dist.log_prob(action) * mask).sum(-1)
        return action, logp, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor):
        mean, std, value = self.forward(obs, action_mask)
        dist, base = self.dist(mean, std)
        mask = action_mask.to(dtype=actions.dtype, device=actions.device)
        actions = actions * mask
        logp = (dist.log_prob(actions) * mask).sum(-1)
        entropy = (base.entropy() * mask).sum(-1)
        return logp, value, entropy

    def std_mean(self) -> float:
        return float(torch.exp(self.log_std).clamp(0.05, 1.5).mean().item())


# ==================== RL UTILS ====================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    t_max, n_agents = rewards.shape
    adv = torch.zeros_like(rewards)
    gae = torch.zeros(n_agents, dtype=rewards.dtype, device=rewards.device)
    next_value = torch.zeros(n_agents, dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(t_max)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
        next_value = values[t]
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
    n = obs.shape[0]
    idx = torch.arange(n, device=obs.device)
    clip_fracs = []
    kls = []
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_total_loss = 0.0

    mb = min(int(minibatch_size), n)
    planned_updates = int(ppo_epochs) * max(int(math.ceil(n / mb)), 1)
    performed_updates = 0
    early_stop = False
    epochs_performed = 0

    for epoch_idx in range(int(ppo_epochs)):
        perm = idx[torch.randperm(n)]
        epoch_kls = []
        for start in range(0, n, mb):
            mb_idx = perm[start:start + mb]
            logp, v, ent = policy.evaluate(obs[mb_idx], act[mb_idx], action_mask[mb_idx])
            ratio = torch.exp(torch.clamp(logp - logp_old[mb_idx], -20.0, 20.0))
            surr1 = ratio * adv[mb_idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(v, ret[mb_idx])
            entropy_loss = -ent.mean() * entropy_coef
            total_loss = policy_loss + value_coef * value_loss + entropy_loss

            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optim.step()
            performed_updates += 1

            clipped = (torch.abs(ratio - 1.0) > clip_ratio).float().mean().item()
            approx_kl = (logp_old[mb_idx] - logp).mean().item()
            epoch_kls.append(float(approx_kl))
            clip_fracs.append(clipped)
            kls.append(approx_kl)
            last_policy_loss = policy_loss.item()
            last_value_loss = value_loss.item()
            last_total_loss = total_loss.item()

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
        "action_count_neg15": int((flat == -15.0).sum().item()),
        "action_count_neg10": int((flat == -10.0).sum().item()),
        "action_count_neg5": int((flat == -5.0).sum().item()),
        "action_count_zero": int((flat == 0.0).sum().item()),
        "action_count_pos5": int((flat == 5.0).sum().item()),
        "action_count_pos10": int((flat == 10.0).sum().item()),
        "action_count_pos15": int((flat == 15.0).sum().item()),
    }


def _build_action_summary_by_taz(
    raw_actions: torch.Tensor,
    applied_actions: torch.Tensor,
    applied_deltas: torch.Tensor,
    action_mask: torch.Tensor,
    taz_ids: list[str],
) -> dict:
    summary = {}
    for taz_idx, taz in enumerate(taz_ids):
        valid_mask = action_mask[taz_idx].bool()
        raw_valid = raw_actions[:, taz_idx, :][:, valid_mask].reshape(-1)
        applied_valid = applied_actions[:, taz_idx, :][:, valid_mask].reshape(-1)
        delta_valid = applied_deltas[:, taz_idx, :][:, valid_mask].reshape(-1)
        if raw_valid.numel() == 0:
            summary[taz] = {
                "avg_action_raw": 0.0,
                "avg_action_discrete": 0.0,
                "avg_applied_duration_delta": 0.0,
                "applied_duration_nonzero_ratio": 0.0,
                "action_counts": {},
            }
            continue

        summary[taz] = {
            "avg_action_raw": float(raw_valid.mean().item()),
            "avg_action_discrete": float(applied_valid.mean().item()),
            "avg_applied_duration_delta": float(delta_valid.mean().item()),
            "applied_duration_nonzero_ratio": float((delta_valid.abs() > 1e-3).float().mean().item()),
            "action_counts": {
                "neg15": int((applied_valid == -15.0).sum().item()),
                "neg10": int((applied_valid == -10.0).sum().item()),
                "neg5": int((applied_valid == -5.0).sum().item()),
                "zero": int((applied_valid == 0.0).sum().item()),
                "pos5": int((applied_valid == 5.0).sum().item()),
                "pos10": int((applied_valid == 10.0).sum().item()),
                "pos15": int((applied_valid == 15.0).sum().item()),
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
        for ep_idx in range(N_TRAIN_DAYS):
            hour = FOCUS_HOURS[ep_idx % len(FOCUS_HOURS)]
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


# ==================== MAIN ====================
def main():
    _set_global_seed(GLOBAL_SEED)

    sumo_standalone_dir = os.path.join(constants.SUMO_PATH, "standalone")
    log_file = os.path.join(sumo_standalone_dir, "command_log_v9.txt")
    sumo = Simulator(configurationPath=sumo_standalone_dir, logFile=log_file, tazTlsMapFile=constants.TAZ_FILE)
    twin_planner = Planner(simulator=sumo)
    sumo.changeDetectorPath(detectorPath=constants.SUMO_NETWORK_PATH)

    env = SumoTazEnvV9(sumoSimulator=sumo, stepSize=300, **ENV_V9_CONFIG)
    runtime_device = torch.device(env.device)
    taz_ids = env.get_taz_ids()
    tls_ids = env.get_tls_ids()
    num_taz = len(taz_ids)
    obs_dim = int(env.agent_obs_dim)
    act_dim = int(env.max_tls_per_taz)

    policy = ActorCriticV9(obs_dim, act_dim, action_limit=ACTION_LIMIT).to(runtime_device)
    policy.train()
    optim = torch.optim.Adam(policy.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.7, patience=12, threshold=0.01, threshold_mode="rel", min_lr=1e-5
    )
    start_episode = _maybe_load_checkpoint(policy, optim, scheduler)
    _move_optimizer_state_to_device(optim, runtime_device)

    episode_schedule = _build_episode_schedule()
    n_episodes = len(episode_schedule)
    if start_episode >= n_episodes:
        print(f"[INFO] start_episode={start_episode} >= n_episodes={n_episodes}. Nothing to train.")
        return

    csv_path = os.path.join(SCRIPT_DIR, "training_history_v9.csv")
    json_path = os.path.join(SCRIPT_DIR, "training_history_v9.json")
    ckpt_dir = os.path.join(SCRIPT_DIR, "checkpoints_v9")
    detail_dir = os.path.join(SCRIPT_DIR, "training_details_v9")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(detail_dir, exist_ok=True)

    csv_cols = [
        "episode", "date", "timeslot", "num_taz", "num_tls", "num_steps", "reward_basis",
        "episode_reward_mean", "episode_reward_std", "episode_reward_min", "episode_reward_max", "episode_reward_sum",
        "final_penalty_mean", "final_penalty_std", "final_penalty_min", "final_penalty_max",
        "avg_action_raw", "sample_action_std_raw", "avg_action_discrete", "sample_action_std_discrete",
        "action_count_neg15", "action_count_neg10", "action_count_neg5", "action_count_zero",
        "action_count_pos5", "action_count_pos10", "action_count_pos15",
        "avg_applied_duration_delta", "sample_applied_duration_delta_std", "applied_duration_nonzero_ratio",
        "returns_mean", "returns_std", "avg_value_estimate",
        "learning_rate", "learning_rate_after_step",
        "ppo_total_loss", "ppo_policy_loss", "ppo_value_loss",
        "policy_clip_fraction", "approx_kl", "ppo_early_stop", "ppo_epochs_performed", "ppo_epochs_planned",
        "policy_std_mean", "details_path",
    ]
    history = []
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=csv_cols).writeheader()

    best_reward = float("-inf")
    best_episode = -1
    interrupted = False
    last_completed_episode = start_episode - 1

    print(f"[INFO] v9 TAZ count: {len(taz_ids)} | IDs: {taz_ids}")
    print(f"[INFO] v9 TLS count: {len(tls_ids)}")
    print(f"[INFO] Observation dim per TAZ: {obs_dim} | Action dim per TAZ: {act_dim}")
    print(f"[INFO] Global seed: {GLOBAL_SEED}")
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
        print(f"\n[RL EP {episode_num}/{n_episodes}] Date={simulation_date} Slot={timeslot}")

        timeslot_clean = timeslot.replace(":", "-")
        hour_multiplier = HOURLY_DEMAND_PROFILE[hour]
        total_cars = int(BASE_DEMAND * hour_multiplier)
        noise = _sample_demand_noise(episode_idx)
        total_cars_random = int(total_cars * noise)
        route_folder_path, should_generate_routes = _resolve_route_folder(
            simulation_date,
            timeslot_clean,
            total_cars_random,
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
            twin_planner.scenarioGenerator.generateRoute(
                inputEdgePath=EDGE_DATA_FILE_PATH,
                timeSlot=timeslot_clean,
                totalCount=total_cars_random,
                custom=False,
                outputFolder=route_folder_path,
                randomTripSeed=ROUTE_RANDOM_TRIP_SEED,
                routeSamplerSeed=ROUTE_SAMPLER_SEED,
                routeSamplerThreads=ROUTE_SAMPLER_THREADS,
            )

        sumo.changeTypePath(typePath=route_folder_path)
        sumo.changeRouteFilePath(route_folder_path)

        env.set_episode_context(hour=hour)
        print(f"[RL RUN] RL EP {episode_num}/{n_episodes} | Starting per-TAZ RL simulation.")
        td = env.reset()

        rewards_list = []
        tls_rewards_list = []
        values_list = []
        logps_list = []
        obs_list = []
        action_masks_list = []
        acts_list = []
        applied_acts_list = []
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

            a, logp, v = policy.act(obs, action_mask)
            if not (torch.isfinite(a).all() and torch.isfinite(logp).all() and torch.isfinite(v).all()):
                print("[WARN] Non-finite tensors in policy output. Skipping episode.")
                episode_invalid = True
                break

            try:
                step_td = env.step(TensorDict({"action": a}, batch_size=[], device=runtime_device))
            except KeyboardInterrupt:
                interrupted = True
                episode_invalid = True
                print("[WARN] Interrupted during RL rollout.")
                break

            next_td = step_td["next"] if "next" in step_td.keys() else step_td
            reward_vec = next_td["reward"].detach().clone()
            tls_reward_vec = next_td["reward_tls"].detach().clone()
            terminated = bool(next_td["terminated"].item())
            truncated = bool(next_td["truncated"].item())
            done_vec = torch.full_like(reward_vec, 1.0 if (terminated or truncated) else 0.0)

            if not (
                torch.isfinite(reward_vec).all()
                and torch.isfinite(tls_reward_vec).all()
                and torch.isfinite(next_td["applied_action"]).all()
                and torch.isfinite(next_td["applied_duration_delta"]).all()
            ):
                print("[WARN] Non-finite environment outputs. Skipping episode.")
                episode_invalid = True
                break

            obs_list.append(obs.detach().clone())
            action_masks_list.append(action_mask.detach().clone())
            acts_list.append(a.detach().clone())
            applied_acts_list.append(next_td["applied_action"].detach().clone())
            applied_deltas_list.append(next_td["applied_duration_delta"].detach().clone())
            logps_list.append(logp.detach().clone())
            values_list.append(v.detach().clone())
            rewards_list.append(reward_vec)
            tls_rewards_list.append(tls_reward_vec)
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
        tls_rewards_t = torch.stack(tls_rewards_list)
        values_t = torch.stack(values_list)
        dones_t = torch.stack(dones_list)
        adv_t, ret_t = compute_gae(rewards_t, values_t, dones_t, gamma=GAMMA, lam=LAMBDA)

        batch_obs_3d = torch.stack(obs_list)
        batch_mask_3d = torch.stack(action_masks_list)
        batch_act_3d = torch.stack(acts_list)
        batch_applied_act_3d = torch.stack(applied_acts_list)
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
        episode_reward_by_tls_tensor = tls_rewards_t.sum(dim=0).cpu()
        reward_mean = float(episode_reward_by_taz_tensor.mean().item())
        scheduler.step(reward_mean)
        lr_after_step = float(optim.param_groups[0]["lr"])

        reward_components = dict(getattr(env, "last_reward_components", {}) or {})
        penalty_by_taz = {
            taz: float((reward_components.get("penalty_by_taz", {}) or {}).get(taz, 0.0))
            for taz in taz_ids
        }
        penalty_by_tls = {
            tls: float((reward_components.get("penalty_by_tls", {}) or {}).get(tls, 0.0))
            for tls in tls_ids
        }
        penalty_vector = torch.tensor([penalty_by_taz[taz] for taz in taz_ids], dtype=torch.float32)
        reward_by_taz = {
            taz: float(episode_reward_by_taz_tensor[idx].item())
            for idx, taz in enumerate(taz_ids)
        }
        reward_by_tls = {
            tls: float(episode_reward_by_tls_tensor[idx].item())
            for idx, tls in enumerate(tls_ids)
        }

        batch_mask_stats = batch_mask_3d.bool()
        avg_action_raw, std_action_raw = _masked_mean_std(batch_act_3d, batch_mask_stats)
        avg_action_discrete, std_action_discrete = _masked_mean_std(batch_applied_act_3d, batch_mask_stats)
        avg_applied_delta, std_applied_delta = _masked_mean_std(batch_applied_delta_3d, batch_mask_stats)
        nonzero_ratio = float((_masked_values(batch_applied_delta_3d.abs(), batch_mask_stats) > 1e-3).float().mean().item())
        action_counts = _count_discrete_actions(batch_applied_act_3d, batch_mask_stats)
        action_summary_by_taz = _build_action_summary_by_taz(
            raw_actions=batch_act_3d,
            applied_actions=batch_applied_act_3d,
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
                    "reward_by_taz": reward_by_taz,
                    "reward_by_tls": reward_by_tls,
                    "final_penalty_by_taz": penalty_by_taz,
                    "final_penalty_by_tls": penalty_by_tls,
                    "final_penalty_details_by_taz": reward_components.get("penalty_details_by_taz", {}),
                    "final_penalty_details_by_tls": reward_components.get("penalty_details_by_tls", {}),
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
            "reward_basis": str(reward_components.get("reward_basis", "per_taz_sum_tls_step_delta")),
            "episode_reward_mean": reward_mean,
            "episode_reward_std": float(episode_reward_by_taz_tensor.std(unbiased=False).item()),
            "episode_reward_min": float(episode_reward_by_taz_tensor.min().item()),
            "episode_reward_max": float(episode_reward_by_taz_tensor.max().item()),
            "episode_reward_sum": float(episode_reward_by_taz_tensor.sum().item()),
            "final_penalty_mean": float(penalty_vector.mean().item()),
            "final_penalty_std": float(penalty_vector.std(unbiased=False).item()),
            "final_penalty_min": float(penalty_vector.min().item()),
            "final_penalty_max": float(penalty_vector.max().item()),
            "avg_action_raw": avg_action_raw,
            "sample_action_std_raw": std_action_raw,
            "avg_action_discrete": avg_action_discrete,
            "sample_action_std_discrete": std_action_discrete,
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
            "ppo_early_stop": stats["early_stop"],
            "ppo_epochs_performed": stats["epochs_performed"],
            "ppo_epochs_planned": stats["epochs_planned"],
            "policy_std_mean": policy.std_mean(),
            "details_path": detail_path,
        }
        history.append(row)

        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=csv_cols).writerow({k: row.get(k, "") for k in csv_cols})
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        ckpt_payload = {
            "episode": episode_idx,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "reward_mean": reward_mean,
            "env_config": ENV_V9_CONFIG,
            "focus_hours": FOCUS_HOURS,
            "taz_ids": taz_ids,
            "tls_ids": tls_ids,
            "tls_count": len(tls_ids),
            "max_tls_per_taz": act_dim,
            "ppo": stats,
            "interrupted": False,
        }
        torch.save(ckpt_payload, os.path.join(ckpt_dir, f"checkpoint_ppo_v9_episode{episode_idx}.pt"))

        if reward_mean > best_reward:
            best_reward = float(reward_mean)
            best_episode = int(episode_idx)
            best_payload = dict(ckpt_payload)
            best_payload["is_best"] = True
            torch.save(best_payload, os.path.join(ckpt_dir, "checkpoint_ppo_v9_best.pt"))

        last_completed_episode = episode_idx
        best_taz = max(reward_by_taz, key=reward_by_taz.get)
        worst_taz = min(reward_by_taz, key=reward_by_taz.get)
        best_tls = max(reward_by_tls, key=reward_by_tls.get)
        worst_tls = min(reward_by_tls, key=reward_by_tls.get)
        print(
            f"[RL RESULT] RL EP {episode_num}/{n_episodes} | "
            f"RewardMean {row['episode_reward_mean']:.4f} | RewardStd {row['episode_reward_std']:.4f} | "
            f"PenaltyMean {row['final_penalty_mean']:.4f} | ActionMean {row['avg_action_discrete']:.2f} | "
            f"AppliedNZ {row['applied_duration_nonzero_ratio']:.2f} | KL {row['approx_kl']:.4f} | "
            f"BestTAZ {best_taz}={reward_by_taz[best_taz]:.4f} | "
            f"WorstTAZ {worst_taz}={reward_by_taz[worst_taz]:.4f} | "
            f"BestTLS {best_tls}={reward_by_tls[best_tls]:.4f} | "
            f"WorstTLS {worst_tls}={reward_by_tls[worst_tls]:.4f}"
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
            "env_config": ENV_V9_CONFIG,
            "focus_hours": FOCUS_HOURS,
            "taz_ids": taz_ids,
            "tls_ids": tls_ids,
            "tls_count": len(tls_ids),
            "best_episode": best_episode,
            "best_reward": best_reward,
            "interrupted": bool(interrupted),
        },
        os.path.join(ckpt_dir, "checkpoint_ppo_v9_final.pt"),
    )

    if interrupted:
        print("[INFO] Training interrupted. Partial checkpoint saved.")
    else:
        print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
