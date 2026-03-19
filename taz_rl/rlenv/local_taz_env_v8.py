# local_taz_env_v8.py
from typing import Optional

import libtraci
import numpy as np
import torch
from tensordict import TensorDict

try:
    from torchrl.data import Bounded as BoundedSpec, Unbounded as UnboundedSpec
except Exception:
    from torchrl.data import BoundedTensorSpec as BoundedSpec
    from torchrl.data import UnboundedContinuousTensorSpec as UnboundedSpec

from libraries.constants import SUMO_NETWORK_PATH, TAZ_FILE
from taz_rl.rlenv.aggregation import aggregate_taz_metrics, aggregate_tls_metrics
from taz_rl.rlenv.local_taz_env_v7 import SumoTazEnvV7


class SumoTazEnvV8(SumoTazEnvV7):
    """
    v8 environment:
      - one local observation per TAZ
      - one local reward per TAZ
      - shared action interface with padded TLS slots per TAZ
      - step reward based on local TAZ detector aggregates only
    """

    def __init__(
        self,
        sumoSimulator,
        stepSize: int = 300,
        device: str = "cpu",
        taz_map_path: str = TAZ_FILE,
        speed_norm: float = 10.0,
        jam_norm: float = 10.0,
        warmupSteps: int = 0,
        cooldownSteps: int = 12,
        min_green: int = 10,
        max_green: int = 300,
        reward_clip: float = 2.0,
        waiting_reward_weight: float = 0.45,
        emission_reward_weight: float = 0.45,
        jam_reward_weight: float = 0.15,
        emission_co2_mix_weight: float = 0.50,
        emission_nox_mix_weight: float = 0.50,
        taz_waiting_time_ref: float = 6000.0,
        taz_co2_ref: float = 300000.0,
        taz_nox_ref: float = 150.0,
        tls_veh_total_ref: float = 120.0,
        tls_pressure_ref: float = 10000.0,
        taz_veh_total_ref: float = 2500.0,
        taz_std_occupancy_ref: float = 0.25,
        max_jam_len_ref: float = 120.0,
        tls_add_path: str = SUMO_NETWORK_PATH + "/optimized_tls.add.xml",
        **kwargs,
    ):
        super().__init__(
            sumoSimulator=sumoSimulator,
            stepSize=stepSize,
            device=device,
            taz_map_path=taz_map_path,
            speed_norm=speed_norm,
            jam_norm=jam_norm,
            warmupSteps=warmupSteps,
            cooldownSteps=cooldownSteps,
            min_green=min_green,
            max_green=max_green,
            reward_clip=reward_clip,
            comparison_reward_enabled=False,
            tls_veh_total_ref=tls_veh_total_ref,
            tls_pressure_ref=tls_pressure_ref,
            taz_veh_total_ref=taz_veh_total_ref,
            taz_std_occupancy_ref=taz_std_occupancy_ref,
            max_jam_len_ref=max_jam_len_ref,
            tls_add_path=tls_add_path,
            **kwargs,
        )

        self.waiting_reward_weight = float(waiting_reward_weight)
        self.emission_reward_weight = float(emission_reward_weight)
        self.jam_reward_weight = float(jam_reward_weight)
        self.emission_co2_mix_weight = float(emission_co2_mix_weight)
        self.emission_nox_mix_weight = float(emission_nox_mix_weight)
        self.taz_waiting_time_ref = max(float(taz_waiting_time_ref), 1e-6)
        self.taz_co2_ref = max(float(taz_co2_ref), 1e-6)
        self.taz_nox_ref = max(float(taz_nox_ref), 1e-6)
        reward_weight_sum = (
            self.waiting_reward_weight
            + self.emission_reward_weight
            + self.jam_reward_weight
        )
        if reward_weight_sum <= 0.0:
            self.waiting_reward_weight = 0.45
            self.emission_reward_weight = 0.45
            self.jam_reward_weight = 0.10
            reward_weight_sum = 1.0
        self.waiting_reward_weight /= reward_weight_sum
        self.emission_reward_weight /= reward_weight_sum
        self.jam_reward_weight /= reward_weight_sum
        emission_mix_sum = self.emission_co2_mix_weight + self.emission_nox_mix_weight
        if emission_mix_sum <= 0.0:
            self.emission_co2_mix_weight = 0.50
            self.emission_nox_mix_weight = 0.50
            emission_mix_sum = 1.0
        self.emission_co2_mix_weight /= emission_mix_sum
        self.emission_nox_mix_weight /= emission_mix_sum

        self.tls_by_taz = {taz: list(self.taz_tls_map[taz]) for taz in self.taz_ids}
        self.max_tls_per_taz = max(len(v) for v in self.tls_by_taz.values())
        self.per_tls_feature_dim = 1 + self.max_num_phases + 6
        self.per_taz_feature_dim = 10
        self.agent_obs_dim = self.max_tls_per_taz * self.per_tls_feature_dim + self.per_taz_feature_dim

        self.observation_spec = UnboundedSpec(
            shape=(len(self.taz_ids), self.agent_obs_dim),
            device=self.device,
        )
        self.action_spec = BoundedSpec(
            low=-15.0,
            high=15.0,
            shape=(len(self.taz_ids), self.max_tls_per_taz),
            device=self.device,
        )
        self.reward_spec = UnboundedSpec(shape=(len(self.taz_ids),), device=self.device)

        action_mask = []
        for taz in self.taz_ids:
            valid = len(self.tls_by_taz[taz])
            row = [True] * valid + [False] * (self.max_tls_per_taz - valid)
            action_mask.append(row)
        self._action_mask = torch.tensor(action_mask, dtype=torch.bool, device=self.device)

        self._prev_taz_penalties = np.zeros(len(self.taz_ids), dtype=np.float32)
        self._taz_lanes = {taz: set() for taz in self.taz_ids}
        self._lane_to_taz_idx = {}
        self._edge_to_taz_idx = {}

    @staticmethod
    def _zero_taz_metrics() -> dict:
        return {
            "mean_speed": 0.0,
            "max_jam_len": 0.0,
            "mean_occupancy": 0.0,
            "veh_total": 0.0,
            "critical_ratio": 0.0,
            "max_occupancy": 0.0,
            "std_occupancy": 0.0,
            "active_vehicle_count": 0.0,
            "total_waiting_time": 0.0,
            "avg_waiting_time": 0.0,
            "total_co2": 0.0,
            "total_nox": 0.0,
        }

    def get_action_mask(self) -> torch.Tensor:
        return self._action_mask.clone()

    @staticmethod
    def _lane_to_edge_id(lane_id: str) -> str:
        lane_id = str(lane_id or "")
        if not lane_id:
            return ""
        if lane_id.startswith(":"):
            return lane_id
        if "_" in lane_id:
            return lane_id.rsplit("_", 1)[0]
        return lane_id

    def _build_taz_lane_cache(self):
        self._taz_lanes = {taz: set() for taz in self.taz_ids}
        self._lane_to_taz_idx = {}
        self._edge_to_taz_idx = {}

        for taz_idx, taz in enumerate(self.taz_ids):
            for tls in self.tls_by_taz[taz]:
                try:
                    tls_lanes = set(self.sumo.get_tls_lanes(tls))
                except Exception:
                    tls_lanes = set()
                self._taz_lanes[taz].update(tls_lanes)

            for lane_id in self._taz_lanes[taz]:
                lane_id = str(lane_id)
                if lane_id not in self._lane_to_taz_idx:
                    self._lane_to_taz_idx[lane_id] = int(taz_idx)
                edge_id = self._lane_to_edge_id(lane_id)
                if edge_id and edge_id not in self._edge_to_taz_idx:
                    self._edge_to_taz_idx[edge_id] = int(taz_idx)

    def _collect_vehicle_metrics_by_taz(self) -> dict:
        metrics_by_taz = {
            taz: {
                "active_vehicle_count": 0.0,
                "total_waiting_time": 0.0,
                "avg_waiting_time": 0.0,
                "total_co2": 0.0,
                "total_nox": 0.0,
            }
            for taz in self.taz_ids
        }
        if not self.sumo.isRunning():
            return metrics_by_taz

        try:
            vehicle_ids = list(libtraci.vehicle.getIDList())
        except Exception:
            return metrics_by_taz

        for vid in vehicle_ids:
            try:
                lane_id = str(libtraci.vehicle.getLaneID(vid))
            except Exception:
                lane_id = ""
            taz_idx = self._lane_to_taz_idx.get(lane_id)

            if taz_idx is None:
                edge_id = self._lane_to_edge_id(lane_id)
                taz_idx = self._edge_to_taz_idx.get(edge_id)
            if taz_idx is None:
                try:
                    road_id = str(libtraci.vehicle.getRoadID(vid))
                except Exception:
                    road_id = ""
                taz_idx = self._edge_to_taz_idx.get(road_id)
            if taz_idx is None:
                continue

            try:
                waiting_time = self._safe_float(libtraci.vehicle.getWaitingTime(vid), 0.0)
                co2 = self._safe_float(libtraci.vehicle.getCO2Emission(vid), 0.0)
                nox = self._safe_float(libtraci.vehicle.getNOxEmission(vid), 0.0)
            except Exception:
                continue

            taz = self.taz_ids[int(taz_idx)]
            metrics_by_taz[taz]["active_vehicle_count"] += 1.0
            metrics_by_taz[taz]["total_waiting_time"] += float(waiting_time)
            metrics_by_taz[taz]["total_co2"] += float(co2)
            metrics_by_taz[taz]["total_nox"] += float(nox)

        for taz in self.taz_ids:
            count = metrics_by_taz[taz]["active_vehicle_count"]
            if count > 0:
                metrics_by_taz[taz]["avg_waiting_time"] = (
                    metrics_by_taz[taz]["total_waiting_time"] / float(count)
                )
        return metrics_by_taz

    def _build_taz_penalties(self):
        penalties = []
        details = {}
        for taz in self.taz_ids:
            tm = self._last_taz_metrics_by_id.get(taz, self._zero_taz_metrics())
            max_jam_len = self._safe_float(tm.get("max_jam_len", 0.0), 0.0)
            total_waiting_time = self._safe_float(tm.get("total_waiting_time", 0.0), 0.0)
            avg_waiting_time = self._safe_float(tm.get("avg_waiting_time", 0.0), 0.0)
            total_co2 = self._safe_float(tm.get("total_co2", 0.0), 0.0)
            total_nox = self._safe_float(tm.get("total_nox", 0.0), 0.0)

            waiting_penalty = float(np.clip(total_waiting_time / self.taz_waiting_time_ref, 0.0, 1.0))
            co2_penalty = float(np.clip(total_co2 / self.taz_co2_ref, 0.0, 1.0))
            nox_penalty = float(np.clip(total_nox / self.taz_nox_ref, 0.0, 1.0))
            emission_penalty = (
                self.emission_co2_mix_weight * co2_penalty
                + self.emission_nox_mix_weight * nox_penalty
            )
            jam_penalty = float(np.clip(max_jam_len / max(self.max_jam_len_ref, 1e-6), 0.0, 1.0))

            penalty = (
                self.waiting_reward_weight * waiting_penalty
                + self.emission_reward_weight * emission_penalty
                + self.jam_reward_weight * jam_penalty
            )
            penalties.append(float(penalty))
            details[taz] = {
                "penalty": float(penalty),
                "waiting_penalty": float(waiting_penalty),
                "emission_penalty": float(emission_penalty),
                "co2_penalty": float(co2_penalty),
                "nox_penalty": float(nox_penalty),
                "jam_penalty": float(jam_penalty),
                "avg_waiting_time": float(avg_waiting_time),
                "total_waiting_time": float(total_waiting_time),
                "total_co2": float(total_co2),
                "total_nox": float(total_nox),
                "max_jam_len": float(max_jam_len),
            }
        return np.asarray(penalties, dtype=np.float32), details

    def _collect_observation(self) -> torch.Tensor:
        if not self.sumo.isRunning():
            self._last_taz_metrics_by_id = {taz: self._zero_taz_metrics() for taz in self.taz_ids}
            return torch.zeros(self.observation_spec.shape, dtype=torch.float32, device=self.device)

        raw_by_taz = {}
        for taz in self.taz_ids:
            try:
                raw_by_taz[taz] = self.sumo.get_taz_e2_metrics(taz, interval="last", mode="dict")
            except Exception:
                raw_by_taz[taz] = {"tls_data": {}, "taz_avg": {}}
        vehicle_metrics_by_taz = self._collect_vehicle_metrics_by_taz()

        obs_rows = []
        self._last_taz_metrics_by_id = {}
        hour_sin, hour_cos = self._get_time_features()

        for taz in self.taz_ids:
            tls_data = raw_by_taz[taz].get("tls_data", {})
            tls_runtime_metrics = []
            tls_metrics_by_id = {}
            for tls in self.tls_by_taz[taz]:
                dets = tls_data.get(tls, [])
                aggregated = aggregate_tls_metrics(dets)
                runtime_metrics = self._collect_tls_runtime_metrics(tls, aggregated)
                tls_runtime_metrics.append(runtime_metrics)
                tls_metrics_by_id[tls] = runtime_metrics

            taz_metrics = aggregate_taz_metrics(tls_runtime_metrics)
            packed = {
                "mean_speed": self._safe_float(taz_metrics.get("mean_speed", 0.0), 0.0),
                "max_jam_len": self._safe_float(taz_metrics.get("max_jam_len", 0.0), 0.0),
                "mean_occupancy": self._safe_float(taz_metrics.get("mean_occupancy", 0.0), 0.0),
                "veh_total": self._safe_float(taz_metrics.get("veh_total", 0.0), 0.0),
                "critical_ratio": self._safe_float(taz_metrics.get("critical_ratio", 0.0), 0.0),
                "max_occupancy": self._safe_float(taz_metrics.get("max_occupancy", 0.0), 0.0),
                "std_occupancy": self._safe_float(taz_metrics.get("std_occupancy", 0.0), 0.0),
            }
            packed.update(vehicle_metrics_by_taz.get(taz, {}))
            self._last_taz_metrics_by_id[taz] = packed
            self._episode_max_jam_len = max(self._episode_max_jam_len, packed["max_jam_len"])

            row = []
            tls_ids = self.tls_by_taz[taz]
            for slot_idx in range(self.max_tls_per_taz):
                if slot_idx < len(tls_ids):
                    tls = tls_ids[slot_idx]
                    m = tls_metrics_by_id[tls]
                    phase_id = int(m.get("phase_id", 0))
                    n_phases = int(self.tls_num_phases.get(tls, self.max_num_phases))
                    one_hot = np.zeros(self.max_num_phases, dtype=np.float32)
                    if 0 <= phase_id < n_phases:
                        one_hot[phase_id] = 1.0

                    phase_duration = self._safe_float(m.get("phase_duration", 0.0), 0.0)
                    mean_speed = self._safe_float(m.get("mean_speed", 0.0), 0.0)
                    max_jam = self._safe_float(m.get("max_jam", 0.0), 0.0)
                    mean_occ = self._safe_float(m.get("mean_occupancy", 0.0), 0.0)
                    veh_total = self._safe_float(m.get("veh_total", 0.0), 0.0)
                    pressure = self._safe_float(m.get("pressure", 0.0), 0.0)

                    phase_duration_norm = np.clip(phase_duration / float(self.max_green), 0.0, 2.0)
                    mean_speed_norm = np.clip(mean_speed / self.speed_norm, 0.0, 2.0)
                    max_jam_norm = np.clip(max_jam / self.jam_norm, 0.0, 2.0)
                    occ_norm = mean_occ / 100.0 if mean_occ > 1.5 else mean_occ
                    occ_norm = self._clip01(occ_norm)
                    veh_total_norm = np.clip(veh_total / self.tls_veh_total_ref, 0.0, 5.0)
                    pressure_norm = np.clip(max(pressure, 0.0) / self.tls_pressure_ref, 0.0, 5.0)

                    row.extend(
                        [1.0]
                        + one_hot.tolist()
                        + [
                            float(phase_duration_norm),
                            float(mean_speed_norm),
                            float(max_jam_norm),
                            float(occ_norm),
                            float(veh_total_norm),
                            float(pressure_norm),
                        ]
                    )
                else:
                    row.extend([0.0] * self.per_tls_feature_dim)

            t_occ_norm = packed["mean_occupancy"] / 100.0 if packed["mean_occupancy"] > 1.5 else packed["mean_occupancy"]
            t_occ_norm = self._clip01(t_occ_norm)
            total_waiting_norm = np.clip(packed["total_waiting_time"] / self.taz_waiting_time_ref, 0.0, 5.0)
            avg_waiting_norm = np.clip(packed["avg_waiting_time"] / self.waiting_ref, 0.0, 5.0)
            total_co2_norm = np.clip(packed["total_co2"] / self.taz_co2_ref, 0.0, 5.0)
            total_nox_norm = np.clip(packed["total_nox"] / self.taz_nox_ref, 0.0, 5.0)

            row.extend(
                [
                    float(np.clip(packed["max_jam_len"] / self.jam_norm, 0.0, 2.0)),
                    float(np.clip(packed["mean_speed"] / self.speed_norm, 0.0, 2.0)),
                    float(t_occ_norm),
                    float(np.clip(packed["veh_total"] / self.taz_veh_total_ref, 0.0, 5.0)),
                    float(total_waiting_norm),
                    float(avg_waiting_norm),
                    float(total_co2_norm),
                    float(total_nox_norm),
                    hour_sin,
                    hour_cos,
                ]
            )
            obs_rows.append(row)

        obs_t = torch.tensor(obs_rows, dtype=torch.float32, device=self.device)
        if not torch.isfinite(obs_t).all():
            obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs_t

    def _reset(self, tensordict=None, **kwargs):
        if self.sumo.isLoaded():
            self.sumo.end()

        self.sumo.start(activeGui=True, logFilePath=self.sumo.logFile, rl_mode=True)
        self.current_step = 0
        self._last_green_tls_snapshot = {}
        self._phase_duration_memory = {}
        self._active_program_id = {}
        self._build_taz_lane_cache()
        for tls in self.tls_list:
            self._sync_phase_duration_memory(tls)
        self._reset_duration_diagnostics()
        self.last_reward_components = {}
        self.last_duration_diagnostics = {}
        self._episode_max_jam_len = 0.0
        self._last_taz_metrics_by_id = {taz: self._zero_taz_metrics() for taz in self.taz_ids}

        obs = self._collect_observation()
        penalties, _ = self._build_taz_penalties()
        self._prev_taz_penalties = penalties.copy()
        return TensorDict(
            {
                "observation": obs,
                "action_mask": self.get_action_mask(),
            },
            batch_size=[],
        )

    def _step(self, tensordict):
        action = tensordict["action"]
        if action.requires_grad:
            action = action.detach()
        action = action.to(self.device)

        low = float(torch.as_tensor(self.action_spec.low).min().item())
        high = float(torch.as_tensor(self.action_spec.high).max().item())
        action_clipped = torch.clamp(action, min=low, max=high)
        action_discrete = self._discretize_actions(action_clipped)
        action_discrete = torch.where(self._action_mask, action_discrete, torch.zeros_like(action_discrete))

        applied_action = torch.zeros_like(action_discrete)
        applied_duration_delta = torch.zeros_like(action_discrete)
        for taz_idx, taz in enumerate(self.taz_ids):
            for slot_idx, tls in enumerate(self.tls_by_taz[taz]):
                a = float(action_discrete[taz_idx, slot_idx].item())
                applied_action[taz_idx, slot_idx] = a
                applied_duration_delta[taz_idx, slot_idx] = float(self._apply_action_to_tls(tls, a))

        self.sumo.step(quantity=self.stepSize)
        self.current_step += 1
        sim_running = self.sumo.isRunning()
        time_limit_reached = self.current_step >= self.cooldownSteps

        obs = self._collect_observation()
        penalties, penalty_details = self._build_taz_penalties()

        if self.current_step <= self.warmupSteps:
            reward_vector = np.zeros_like(penalties, dtype=np.float32)
        else:
            reward_vector = np.clip(self._prev_taz_penalties - penalties, -self.reward_clip, self.reward_clip)
        self._prev_taz_penalties = penalties.copy()

        terminated = (not sim_running) and (not time_limit_reached)
        truncated = bool(time_limit_reached)
        if terminated or truncated:
            if self.sumo.isLoaded():
                self.sumo.end()

        reward_by_taz = {taz: float(reward_vector[idx]) for idx, taz in enumerate(self.taz_ids)}
        penalty_by_taz = {taz: float(penalties[idx]) for idx, taz in enumerate(self.taz_ids)}
        self.last_reward_components = {
            "reward_basis": "per_taz_wait_emission_jam_delta",
            "reward_by_taz": reward_by_taz,
            "penalty_by_taz": penalty_by_taz,
            "penalty_details_by_taz": penalty_details,
            "reward_mean": float(np.mean(reward_vector)),
            "reward_std": float(np.std(reward_vector)),
            "reward_min": float(np.min(reward_vector)),
            "reward_max": float(np.max(reward_vector)),
            "penalty_mean": float(np.mean(penalties)),
            "penalty_std": float(np.std(penalties)),
            "penalty_min": float(np.min(penalties)),
            "penalty_max": float(np.max(penalties)),
            "episode_steps": int(self.current_step),
            "num_taz": int(len(self.taz_ids)),
            "num_tls": int(len(self.tls_list)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "warmup": bool(self.current_step <= self.warmupSteps),
        }

        return TensorDict(
            {
                "observation": obs,
                "action_mask": self.get_action_mask(),
                "reward": torch.tensor(reward_vector, dtype=torch.float32, device=self.device),
                "applied_action": applied_action,
                "applied_duration_delta": applied_duration_delta,
                "terminated": torch.tensor(terminated, device=self.device),
                "truncated": torch.tensor(truncated, device=self.device),
            },
            batch_size=[],
        )
