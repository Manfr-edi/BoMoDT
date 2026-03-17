# training_script_ppo_v7.py
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
from taz_rl.rlenv.local_taz_env_v7 import SumoTazEnvV7


# ==================== CONFIG ====================
TRAIN_START_DATE = datetime(2024, 2, 1)
N_TRAIN_DAYS = 200
FOCUS_HOURS = [8]
TRAIN_ON_SAME_DAY = True
# Reward mode:
# - False: train with one simulation (absolute reward from that episode)
# - True: train with delta reward vs no-agent baseline each episode
USE_BASELINE_COMPARISON_REWARD = True
# If training uses single-simulation reward, run no-agent baseline only every N episodes for logging.
BASELINE_EVAL_EVERY = 10

HOURLY_DEMAND_PROFILE = {
    0: 0.2, 1: 0.15, 2: 0.1, 3: 0.1, 4: 0.2,
    5: 0.4, 6: 0.7, 7: 1.2, 8: 1.5,
    9: 1.0, 10: 0.8, 11: 0.9,
    12: 1.1, 13: 1.0, 14: 0.9, 15: 1.0,
    16: 1.3, 17: 1.6, 18: 1.4, 19: 1.0,
    20: 0.8, 21: 0.6, 22: 0.4, 23: 0.3,
}
BASE_DEMAND = 9000
DEMAND_NOISE_RANGE = (1.0, 1.0)

ENV_V7_CONFIG = dict(
    speed_norm=10.0,
    jam_norm=10.0,
    warmupSteps=0,
    cooldownSteps=12,
    min_green=10,
    max_green=300,
    time_loss_weight=0.65,
    emission_weight=0.25,
    jam_weight=0.10,
    emission_co2_mix_weight=0.34,
    emission_nox_mix_weight=0.33,
    emission_fuel_mix_weight=0.33,
    time_loss_ref=2500000.0,
    co2_ref=11000000000.0,
    nox_ref=4000000.0,
    fuel_ref=3600000000.0,
    max_jam_len_ref=120.0,
    metric_clip=5.0,
    reward_clip=5.0,
    comparison_reward_enabled=USE_BASELINE_COMPARISON_REWARD,
    live_vehicle_sample=4096,
    active_vehicle_ref=2500.0,
    waiting_ref=60.0,
    vehicle_time_loss_ref=120.0,
    emission_vehicle_co2_ref=3500.0,
    emission_vehicle_nox_ref=1.5,
    emission_vehicle_fuel_ref=1200.0,
    tls_veh_total_ref=120.0,
    tls_pressure_ref=10000.0,
    taz_veh_total_ref=2500.0,
    taz_std_occupancy_ref=0.25,
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
MINIBATCH_SIZE = 32
ACTION_LIMIT = 20.0

EXTRACTOR_HIDDEN = 256
EXTRACTOR_OUT = 128

# Checkpoints
LOAD_MODEL = False
CHECKPOINT_PATH = os.path.join("checkpoints_v7", "checkpoint_ppo_v7_final.pt")
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


class ActorCriticV7(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, action_limit: float = 20.0):
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

    def forward(self, obs):
        feat = self.extractor(obs)
        mean = self.mean_head(self.actor(feat))
        std = torch.exp(self.log_std).clamp(0.05, 1.5)
        value = self.critic(feat).squeeze(-1)
        return mean, std, value

    def dist(self, mean, std):
        base = Normal(mean, std)
        dist = TransformedDistribution(
            base,
            [TanhTransform(cache_size=1), AffineTransform(loc=0.0, scale=self.action_limit)],
        )
        return dist, base

    @torch.no_grad()
    def act(self, obs):
        mean, std, value = self.forward(obs)
        d, _ = self.dist(mean, std)
        a = d.sample()
        logp = d.log_prob(a).sum(-1)
        return a, logp, value

    def evaluate(self, obs, actions):
        mean, std, value = self.forward(obs)
        d, base = self.dist(mean, std)
        logp = d.log_prob(actions).sum(-1)
        entropy = base.entropy().sum(-1)
        return logp, value, entropy


# ==================== RL UTILS ====================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    t_max = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = 0.0
    next_value = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)

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
            logp, v, ent = policy.evaluate(obs[mb_idx], act[mb_idx])
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


def _count_discrete_actions(applied_actions: torch.Tensor) -> dict:
    flat = applied_actions.reshape(-1)
    return {
        "action_count_neg15": int((flat == -15.0).sum().item()),
        "action_count_neg10": int((flat == -10.0).sum().item()),
        "action_count_neg5": int((flat == -5.0).sum().item()),
        "action_count_pos5": int((flat == 5.0).sum().item()),
        "action_count_pos10": int((flat == 10.0).sum().item()),
        "action_count_pos15": int((flat == 15.0).sum().item()),
    }


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

    ckpt_path = CHECKPOINT_PATH
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ckpt_path))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cuda:0", weights_only=True)
    policy.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        try:
            optim.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"[WARN] Could not load optimizer state: {e}")
    if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            print(f"[WARN] Could not load scheduler state: {e}")

    if RESUME_FROM_CHECKPOINT_EPISODE:
        start_episode = int(ckpt.get("episode", -1)) + 1
    if START_EPISODE_OVERRIDE is not None:
        start_episode = int(START_EPISODE_OVERRIDE)
    return start_episode


# ==================== MAIN ====================
def main():
    logFile = "../sumoenv/standalone/command_log.txt"
    sumo = Simulator(configurationPath="../sumoenv/standalone", logFile=logFile, tazTlsMapFile=constants.TAZ_FILE)
    twinPlanner = Planner(simulator=sumo)
    sumo.changeDetectorPath(detectorPath=constants.SUMO_NETWORK_PATH)

    env = SumoTazEnvV7(sumoSimulator=sumo, stepSize=300, **ENV_V7_CONFIG)
    taz_ids = env.get_taz_ids()
    tls_ids = env.get_tls_ids()

    # TorchRL may expose either Composite specs (with "observation"/"action")
    # or plain tensor specs. Infer robustly.
    try:
        obs_dim = int(env.observation_spec["observation"].shape[-1])
    except Exception:
        obs_shape = tuple(getattr(env.observation_spec, "shape", ()))
        if len(obs_shape) > 0:
            obs_dim = int(obs_shape[-1])
        else:
            td_probe = env.reset()
            obs_dim = int(td_probe["observation"].shape[-1])
            try:
                if env.sumo.isLoaded():
                    env.sumo.end()
            except Exception:
                pass

    try:
        act_dim = int(env.action_spec["action"].shape[-1])
    except Exception:
        act_shape = tuple(getattr(env.action_spec, "shape", ()))
        act_dim = int(act_shape[-1]) if len(act_shape) > 0 else len(env.get_tls_ids())

    policy = ActorCriticV7(obs_dim, act_dim, action_limit=ACTION_LIMIT)
    policy.train()
    optim = torch.optim.Adam(policy.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.7, patience=12, threshold=0.01, threshold_mode="rel", min_lr=1e-5
    )
    start_episode = _maybe_load_checkpoint(policy, optim, scheduler)

    episode_schedule = _build_episode_schedule()
    n_episodes = len(episode_schedule)
    if start_episode >= n_episodes:
        print(f"[INFO] start_episode={start_episode} >= n_episodes={n_episodes}. Nothing to train.")
        return

    csv_path = "training_history_v7.csv"
    json_path = "training_history_v7.json"
    ckpt_dir = "checkpoints_v7"
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_cols = [
        "episode", "date", "timeslot", "num_taz", "num_tls", "total_reward", "num_steps",
        "avg_action_raw", "sample_action_std_raw", "avg_action_discrete", "sample_action_std_discrete",
        "action_count_neg15", "action_count_neg10", "action_count_neg5",
        "action_count_pos5", "action_count_pos10", "action_count_pos15",
        "avg_applied_duration_delta", "sample_applied_duration_delta_std", "applied_duration_nonzero_ratio",
        "reward_basis", "reward_parse_ok", "comparison_reward_enabled", "baseline_penalty", "delta_penalty", "reward_penalty",
        "eval_ran", "eval_baseline_parse_ok", "eval_baseline_penalty", "eval_delta_penalty",
        "trip_count", "total_time_loss", "total_co2", "total_nox", "total_fuel",
        "episode_max_jam_len", "baseline_episode_max_jam_len", "delta_episode_max_jam_len",
        "time_loss_norm", "emission_norm", "jam_norm",
        "returns_mean", "returns_std", "avg_value_estimate",
        "learning_rate", "learning_rate_after_step",
        "ppo_total_loss", "ppo_policy_loss", "ppo_value_loss",
        "policy_clip_fraction", "approx_kl", "ppo_early_stop", "ppo_epochs_performed", "ppo_epochs_planned",
        "policy_std_mean",
    ]
    history = []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=csv_cols).writeheader()

    best_reward = float("-inf")
    best_episode = -1
    fixed_obs = torch.zeros((1, obs_dim), dtype=torch.float32)
    interrupted = False
    last_completed_episode = start_episode - 1

    print(f"[INFO] v7 TAZ count: {len(taz_ids)} | IDs: {taz_ids}")
    print(f"[INFO] v7 TLS count: {len(tls_ids)}")
    print(f"[INFO] Observation dim: {obs_dim} | Action dim: {act_dim}")
    print(f"[INFO] Comparison reward vs baseline: {USE_BASELINE_COMPARISON_REWARD}")
    if not USE_BASELINE_COMPARISON_REWARD:
        print(f"[INFO] Baseline evaluation cadence (episodes): {BASELINE_EVAL_EVERY}")

    for episode_idx in range(start_episode, n_episodes):
        day, hour = episode_schedule[episode_idx]
        simulation_date = day.strftime("%Y-%m-%d")
        timeslot = f"{hour:02d}:00-{(hour + 1):02d}:00"
        print(f"\n[EP {episode_idx + 1}/{n_episodes}] Date={simulation_date} Slot={timeslot}")

        generateEdgeDataFile(PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH, date=simulation_date, time_slot=timeslot)
        timeslot_clean = timeslot.replace(":", "-")
        route_folder_path = os.path.join(SUMO_PATH, "routes", timeslot_clean)
        os.makedirs(os.path.join(route_folder_path, "output"), exist_ok=True)

        hour_multiplier = HOURLY_DEMAND_PROFILE[hour]
        total_cars = int(BASE_DEMAND * hour_multiplier)
        noise = random.uniform(*DEMAND_NOISE_RANGE)
        total_cars_random = int(total_cars * noise)

        twinPlanner.scenarioGenerator.generateRoute(
            inputEdgePath=EDGE_DATA_FILE_PATH,
            timeSlot=timeslot_clean,
            totalCount=total_cars_random,
            custom=False,
        )
        sumo.changeTypePath(typePath=route_folder_path)
        sumo.changeRouteFilePath(route_folder_path)

        baseline_penalty = None
        baseline_components = {}
        if USE_BASELINE_COMPARISON_REWARD:
            try:
                baseline_penalty, baseline_components = env.run_reference_episode_no_agent(hour=hour)
            except KeyboardInterrupt:
                interrupted = True
                print("[WARN] Interrupted during baseline run.")
                break
            if not bool(baseline_components.get("parse_ok", False)):
                print("[WARN] Baseline parse failed. Falling back to absolute reward for this episode.")
                baseline_penalty = None
            env.set_baseline_penalty(baseline_penalty, baseline_components)
        else:
            env.set_baseline_penalty(None)

        env.set_episode_context(hour=hour)
        td = env.reset()

        rewards_list = []
        values_list = []
        logps_list = []
        obs_list = []
        acts_list = []
        applied_acts_list = []
        applied_deltas_list = []
        dones_list = []
        terminal_reward = None
        episode_invalid = False

        while True:
            obs = td["observation"].unsqueeze(0)
            a, logp, v = policy.act(obs)
            if not (torch.isfinite(a).all() and torch.isfinite(logp).all() and torch.isfinite(v).all()):
                print("[WARN] Non-finite tensors in policy output. Skipping episode.")
                episode_invalid = True
                break

            try:
                step_td = env.step(TensorDict({"action": a.squeeze(0)}, batch_size=[]))
            except KeyboardInterrupt:
                interrupted = True
                episode_invalid = True
                print("[WARN] Interrupted during RL rollout.")
                break

            next_td = step_td["next"] if "next" in step_td.keys() else step_td
            r = next_td["reward"].item()
            terminated = bool(next_td["terminated"].item())
            truncated = bool(next_td["truncated"].item())

            obs_list.append(obs.squeeze(0))
            acts_list.append(a.squeeze(0))
            applied_acts_list.append(next_td["applied_action"].detach().clone())
            applied_deltas_list.append(next_td["applied_duration_delta"].detach().clone())
            logps_list.append(logp.squeeze(0))
            values_list.append(v.squeeze(0))
            rewards_list.append(torch.tensor(r, dtype=torch.float32))
            dones_list.append(torch.tensor(1.0 if (terminated or truncated) else 0.0, dtype=torch.float32))
            if terminated or truncated:
                terminal_reward = float(r)

            if terminated or truncated:
                break
            td = next_td

        if interrupted:
            break
        if episode_invalid or len(rewards_list) == 0:
            continue

        total_reward = float(terminal_reward) if terminal_reward is not None else float(sum(x.item() for x in rewards_list))

        rewards_t = torch.stack(rewards_list)
        values_t = torch.stack(values_list)
        dones_t = torch.stack(dones_list)
        adv_t, ret_t = compute_gae(rewards_t, values_t, dones_t, gamma=GAMMA, lam=LAMBDA)

        batch_obs = torch.stack(obs_list)
        batch_act = torch.stack(acts_list)
        batch_applied_act = torch.stack(applied_acts_list)
        batch_applied_delta = torch.stack(applied_deltas_list)
        batch_logp = torch.stack(logps_list)
        batch_adv = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)
        batch_ret = ret_t

        progress = episode_idx / max(n_episodes - 1, 1)
        if progress <= ENTROPY_WARMUP_RATIO:
            entropy_coef_now = ENTROPY_COEF
        else:
            decay_progress = (progress - ENTROPY_WARMUP_RATIO) / max(1.0 - ENTROPY_WARMUP_RATIO, 1e-8)
            entropy_coef_now = ENTROPY_COEF + (ENTROPY_COEF_FINAL - ENTROPY_COEF) * decay_progress

        stats = ppo_update(
            policy, optim, batch_obs, batch_act, batch_logp, batch_adv, batch_ret,
            clip_ratio=CLIP_RATIO, ppo_epochs=PPO_UPDATE_EPOCHS, minibatch_size=MINIBATCH_SIZE,
            entropy_coef=entropy_coef_now, value_coef=VALUE_COEF, target_kl=TARGET_KL
        )
        lr_before_step = float(optim.param_groups[0]["lr"])
        scheduler.step(total_reward)
        lr_after_step = float(optim.param_groups[0]["lr"])

        with torch.no_grad():
            _, std_dbg, _ = policy.forward(fixed_obs)
            policy_std_mean = float(std_dbg.mean().item())

        reward_components = dict(getattr(env, "last_reward_components", {}))
        reward_penalty = float(reward_components.get("penalty", 0.0))

        # Optional periodic no-agent baseline evaluation (logging only, no training reward impact).
        eval_ran = False
        eval_baseline_penalty = None
        eval_baseline_components = {}
        eval_baseline_parse_ok = False
        if (not USE_BASELINE_COMPARISON_REWARD) and BASELINE_EVAL_EVERY > 0 and (episode_idx % BASELINE_EVAL_EVERY == 0):
            eval_ran = True
            try:
                eval_baseline_penalty, eval_baseline_components = env.run_reference_episode_no_agent(hour=hour)
            except KeyboardInterrupt:
                interrupted = True
                print("[WARN] Interrupted during periodic baseline evaluation run.")
                break
            eval_baseline_parse_ok = bool(eval_baseline_components.get("parse_ok", False))
            if not eval_baseline_parse_ok:
                eval_baseline_penalty = None
            env.set_baseline_penalty(None)

        if interrupted:
            break

        eval_delta_penalty = ""
        if eval_baseline_penalty is not None:
            eval_delta_penalty = float(eval_baseline_penalty - reward_penalty)

        baseline_components_for_log = baseline_components if USE_BASELINE_COMPARISON_REWARD else eval_baseline_components
        baseline_episode_max_jam_len = ""
        delta_episode_max_jam_len = ""
        if len(baseline_components_for_log) > 0:
            baseline_episode_max_jam_len = float(baseline_components_for_log.get("episode_max_jam_len", 0.0))
            delta_episode_max_jam_len = baseline_episode_max_jam_len - float(reward_components.get("episode_max_jam_len", 0.0))

        action_counts = _count_discrete_actions(batch_applied_act)
        row = {
            "episode": int(episode_idx),
            "date": simulation_date,
            "timeslot": timeslot,
            "num_taz": int(reward_components.get("num_taz", len(taz_ids))),
            "num_tls": int(reward_components.get("num_tls", len(tls_ids))),
            "total_reward": float(total_reward),
            "num_steps": int(len(rewards_list)),
            "avg_action_raw": float(batch_act.mean().item()),
            "sample_action_std_raw": float(batch_act.std(unbiased=False).item()),
            "avg_action_discrete": float(batch_applied_act.mean().item()),
            "sample_action_std_discrete": float(batch_applied_act.std(unbiased=False).item()),
            **action_counts,
            "avg_applied_duration_delta": float(batch_applied_delta.mean().item()),
            "sample_applied_duration_delta_std": float(batch_applied_delta.std(unbiased=False).item()),
            "applied_duration_nonzero_ratio": float((batch_applied_delta.abs() > 1e-3).float().mean().item()),
            "reward_basis": str(reward_components.get("reward_basis", "")),
            "reward_parse_ok": bool(reward_components.get("parse_ok", False)),
            "comparison_reward_enabled": bool(reward_components.get("comparison_reward_enabled", False)),
            "baseline_penalty": float(reward_components.get("baseline_penalty", 0.0)),
            "delta_penalty": float(reward_components.get("delta_penalty", 0.0)),
            "reward_penalty": reward_penalty,
            "eval_ran": bool(eval_ran),
            "eval_baseline_parse_ok": bool(eval_baseline_parse_ok) if eval_ran else "",
            "eval_baseline_penalty": float(eval_baseline_penalty) if eval_baseline_penalty is not None else "",
            "eval_delta_penalty": eval_delta_penalty,
            "trip_count": int(reward_components.get("trip_count", 0)),
            "total_time_loss": float(reward_components.get("total_time_loss", 0.0)),
            "total_co2": float(reward_components.get("total_co2", 0.0)),
            "total_nox": float(reward_components.get("total_nox", 0.0)),
            "total_fuel": float(reward_components.get("total_fuel", 0.0)),
            "episode_max_jam_len": float(reward_components.get("episode_max_jam_len", 0.0)),
            "baseline_episode_max_jam_len": baseline_episode_max_jam_len,
            "delta_episode_max_jam_len": delta_episode_max_jam_len,
            "time_loss_norm": float(reward_components.get("time_loss_norm", 0.0)),
            "emission_norm": float(reward_components.get("emission_norm", 0.0)),
            "jam_norm": float(reward_components.get("jam_norm", 0.0)),
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
            "policy_std_mean": policy_std_mean,
        }
        history.append(row)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=csv_cols).writerow({k: row.get(k, "") for k in csv_cols})
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        ckpt_payload = {
            "episode": episode_idx,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "total_reward": float(total_reward),
            "env_config": ENV_V7_CONFIG,
            "focus_hours": FOCUS_HOURS,
            "taz_ids": taz_ids,
            "tls_count": len(tls_ids),
            "ppo": stats,
            "interrupted": False,
        }
        torch.save(ckpt_payload, os.path.join(ckpt_dir, f"checkpoint_ppo_v7_episode{episode_idx}.pt"))

        if total_reward > best_reward:
            best_reward = float(total_reward)
            best_episode = int(episode_idx)
            best_payload = dict(ckpt_payload)
            best_payload["is_best"] = True
            torch.save(best_payload, os.path.join(ckpt_dir, "checkpoint_ppo_v7_best.pt"))

        last_completed_episode = episode_idx
        msg = (
            f"Episode {episode_idx} | Reward {total_reward:.4f} | BaselinePen {row['baseline_penalty']:.4f} | "
            f"Delta {row['delta_penalty']:.4f} | Penalty {row['reward_penalty']:.4f} | "
            f"TimeLoss {row['total_time_loss']:.0f} | CO2 {row['total_co2']:.0f} | "
            f"MaxJam {row['episode_max_jam_len']:.2f} | KL {row['approx_kl']:.4f}"
        )
        if eval_baseline_penalty is not None:
            msg += f" | EvalBaselinePen {eval_baseline_penalty:.4f} | EvalDelta {eval_delta_penalty:.4f}"
        elif eval_ran and not eval_baseline_parse_ok:
            msg += " | EvalBaselinePen parse_failed"
        print(msg)

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
            "env_config": ENV_V7_CONFIG,
            "focus_hours": FOCUS_HOURS,
            "taz_ids": taz_ids,
            "tls_count": len(tls_ids),
            "best_episode": best_episode,
            "best_reward": best_reward,
            "interrupted": bool(interrupted),
        },
        os.path.join(ckpt_dir, "checkpoint_ppo_v7_final.pt"),
    )

    if interrupted:
        print("[INFO] Training interrupted. Partial checkpoint saved.")
    else:
        print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
