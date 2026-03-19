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


class SumoTazEnvV9(SumoTazEnvV7):
    """
    v9 environment:
      - one agent per TAZ
      - one observation row per TAZ, containing all TLS data in that TAZ
      - one action vector per TAZ, one slot for each TLS in that TAZ
      - rewards are computed from per-TLS waiting/emission/jam signals and summed back to the TAZ
    """

    def __init__(
        self,
        sumoSimulator,
        stepSize: int = 300,
        device: str = "cuda:0",
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
        jam_reward_weight: float = 0.10,
        emission_co2_mix_weight: float = 0.50,
        emission_nox_mix_weight: float = 0.50,
        tls_waiting_time_ref: float = 1000.0,
        tls_co2_ref: float = 80000.0,
        tls_nox_ref: float = 40.0,
        tls_active_vehicle_ref: float = 80.0,
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
        self.tls_waiting_time_ref = max(float(tls_waiting_time_ref), 1e-6)
        self.tls_co2_ref = max(float(tls_co2_ref), 1e-6)
        self.tls_nox_ref = max(float(tls_nox_ref), 1e-6)
        self.tls_active_vehicle_ref = max(float(tls_active_vehicle_ref), 1e-6)
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
        self.per_tls_feature_dim = 1 + self.max_num_phases + 11
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

        self._prev_tls_penalties = np.zeros(len(self.tls_list), dtype=np.float32)
        self._tls_lanes = {tls: set() for tls in self.tls_list}
        self._lane_to_tls_idx = {}
        self._edge_to_tls_idx = {}
        self._last_tls_metrics_by_id = {
            tls: self._zero_tls_metrics(taz_id=self.tls_to_taz.get(tls))
            for tls in self.tls_list
        }

    @staticmethod
    def _zero_tls_metrics(taz_id: Optional[str] = None) -> dict:
        return {
            "taz_id": str(taz_id) if taz_id is not None else None,
            "phase_id": 0,
            "phase_duration": 0.0,
            "veh_total": 0.0,
            "mean_speed": 0.0,
            "mean_occupancy": 0.0,
            "max_jam": 0.0,
            "pressure": 0.0,
            "active_vehicle_count": 0.0,
            "total_waiting_time": 0.0,
            "avg_waiting_time": 0.0,
            "total_co2": 0.0,
            "total_nox": 0.0,
        }

    @staticmethod
    def _zero_taz_reward_metrics() -> dict:
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

    def _build_tls_lane_cache(self):
        self._tls_lanes = {tls: set() for tls in self.tls_list}
        self._lane_to_tls_idx = {}
        self._edge_to_tls_idx = {}

        for tls_idx, tls in enumerate(self.tls_list):
            try:
                tls_lanes = set(self.sumo.get_tls_lanes(tls))
            except Exception:
                tls_lanes = set()
            self._tls_lanes[tls] = tls_lanes

            for lane_id in tls_lanes:
                lane_id = str(lane_id)
                if lane_id not in self._lane_to_tls_idx:
                    self._lane_to_tls_idx[lane_id] = int(tls_idx)
                edge_id = self._lane_to_edge_id(lane_id)
                if edge_id and edge_id not in self._edge_to_tls_idx:
                    self._edge_to_tls_idx[edge_id] = int(tls_idx)

    def _collect_vehicle_metrics_by_tls(self) -> dict:
        metrics_by_tls = {
            tls: {
                "active_vehicle_count": 0.0,
                "total_waiting_time": 0.0,
                "avg_waiting_time": 0.0,
                "total_co2": 0.0,
                "total_nox": 0.0,
            }
            for tls in self.tls_list
        }
        if not self.sumo.isRunning():
            return metrics_by_tls

        try:
            vehicle_ids = list(libtraci.vehicle.getIDList())
        except Exception:
            return metrics_by_tls

        for vid in vehicle_ids:
            try:
                lane_id = str(libtraci.vehicle.getLaneID(vid))
            except Exception:
                lane_id = ""
            tls_idx = self._lane_to_tls_idx.get(lane_id)

            if tls_idx is None:
                edge_id = self._lane_to_edge_id(lane_id)
                tls_idx = self._edge_to_tls_idx.get(edge_id)
            if tls_idx is None:
                try:
                    road_id = str(libtraci.vehicle.getRoadID(vid))
                except Exception:
                    road_id = ""
                tls_idx = self._edge_to_tls_idx.get(road_id)
            if tls_idx is None:
                continue

            try:
                waiting_time = self._safe_float(libtraci.vehicle.getWaitingTime(vid), 0.0)
                co2 = self._safe_float(libtraci.vehicle.getCO2Emission(vid), 0.0)
                nox = self._safe_float(libtraci.vehicle.getNOxEmission(vid), 0.0)
            except Exception:
                continue

            tls_id = self.tls_list[int(tls_idx)]
            metrics_by_tls[tls_id]["active_vehicle_count"] += 1.0
            metrics_by_tls[tls_id]["total_waiting_time"] += float(waiting_time)
            metrics_by_tls[tls_id]["total_co2"] += float(co2)
            metrics_by_tls[tls_id]["total_nox"] += float(nox)

        for tls_id in self.tls_list:
            count = metrics_by_tls[tls_id]["active_vehicle_count"]
            if count > 0:
                metrics_by_tls[tls_id]["avg_waiting_time"] = (
                    metrics_by_tls[tls_id]["total_waiting_time"] / float(count)
                )
        return metrics_by_tls

    def _build_taz_metrics_from_tls(self) -> dict:
        metrics_by_taz = {
            taz: self._zero_taz_reward_metrics()
            for taz in self.taz_ids
        }

        for taz in self.taz_ids:
            tls_metrics = [
                self._last_tls_metrics_by_id.get(
                    tls,
                    self._zero_tls_metrics(taz_id=taz),
                )
                for tls in self.taz_tls_map[taz]
            ]
            detector_aggregate = aggregate_taz_metrics(tls_metrics)
            active_vehicle_count = float(sum(m.get("active_vehicle_count", 0.0) for m in tls_metrics))
            total_waiting_time = float(sum(m.get("total_waiting_time", 0.0) for m in tls_metrics))
            total_co2 = float(sum(m.get("total_co2", 0.0) for m in tls_metrics))
            total_nox = float(sum(m.get("total_nox", 0.0) for m in tls_metrics))
            avg_waiting_time = total_waiting_time / active_vehicle_count if active_vehicle_count > 0.0 else 0.0

            metrics_by_taz[taz] = {
                "mean_speed": self._safe_float(detector_aggregate.get("mean_speed", 0.0), 0.0),
                "max_jam_len": self._safe_float(detector_aggregate.get("max_jam_len", 0.0), 0.0),
                "mean_occupancy": self._safe_float(detector_aggregate.get("mean_occupancy", 0.0), 0.0),
                "veh_total": self._safe_float(detector_aggregate.get("veh_total", 0.0), 0.0),
                "critical_ratio": self._safe_float(detector_aggregate.get("critical_ratio", 0.0), 0.0),
                "max_occupancy": self._safe_float(detector_aggregate.get("max_occupancy", 0.0), 0.0),
                "std_occupancy": self._safe_float(detector_aggregate.get("std_occupancy", 0.0), 0.0),
                "active_vehicle_count": active_vehicle_count,
                "total_waiting_time": total_waiting_time,
                "avg_waiting_time": avg_waiting_time,
                "total_co2": total_co2,
                "total_nox": total_nox,
            }
        return metrics_by_taz

    def _build_tls_penalties(self):
        penalties = []
        details = {}
        for tls in self.tls_list:
            tm = self._last_tls_metrics_by_id.get(
                tls,
                self._zero_tls_metrics(taz_id=self.tls_to_taz.get(tls)),
            )
            max_jam = self._safe_float(tm.get("max_jam", 0.0), 0.0)
            total_waiting_time = self._safe_float(tm.get("total_waiting_time", 0.0), 0.0)
            avg_waiting_time = self._safe_float(tm.get("avg_waiting_time", 0.0), 0.0)
            total_co2 = self._safe_float(tm.get("total_co2", 0.0), 0.0)
            total_nox = self._safe_float(tm.get("total_nox", 0.0), 0.0)

            waiting_penalty = float(np.clip(total_waiting_time / self.tls_waiting_time_ref, 0.0, 1.0))
            co2_penalty = float(np.clip(total_co2 / self.tls_co2_ref, 0.0, 1.0))
            nox_penalty = float(np.clip(total_nox / self.tls_nox_ref, 0.0, 1.0))
            emission_penalty = (
                self.emission_co2_mix_weight * co2_penalty
                + self.emission_nox_mix_weight * nox_penalty
            )
            jam_penalty = float(np.clip(max_jam / max(self.max_jam_len_ref, 1e-6), 0.0, 1.0))

            penalty = (
                self.waiting_reward_weight * waiting_penalty
                + self.emission_reward_weight * emission_penalty
                + self.jam_reward_weight * jam_penalty
            )
            penalties.append(float(penalty))
            details[tls] = {
                "taz_id": self.tls_to_taz.get(tls),
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
                "max_jam": float(max_jam),
            }
        return np.asarray(penalties, dtype=np.float32), details

    def _summarize_taz_rewards_and_penalties(
        self,
        reward_vector: np.ndarray,
        penalties: np.ndarray,
        penalty_details_by_tls: dict,
    ) -> tuple[dict, dict, dict]:
        reward_by_taz = {taz: 0.0 for taz in self.taz_ids}
        penalty_by_taz = {taz: 0.0 for taz in self.taz_ids}
        details_by_taz = {
            taz: {
                "tls_count": 0,
                "penalty_sum": 0.0,
                "waiting_penalty_sum": 0.0,
                "emission_penalty_sum": 0.0,
                "jam_penalty_sum": 0.0,
                "total_waiting_time": 0.0,
                "total_co2": 0.0,
                "total_nox": 0.0,
                "max_jam_len": 0.0,
            }
            for taz in self.taz_ids
        }

        for tls_idx, tls in enumerate(self.tls_list):
            taz = self.tls_to_taz.get(tls)
            if taz is None:
                continue
            reward_by_taz[taz] += float(reward_vector[tls_idx])
            penalty_by_taz[taz] += float(penalties[tls_idx])

            tls_detail = penalty_details_by_tls.get(tls, {})
            taz_detail = details_by_taz[taz]
            taz_detail["tls_count"] += 1
            taz_detail["penalty_sum"] += float(tls_detail.get("penalty", 0.0))
            taz_detail["waiting_penalty_sum"] += float(tls_detail.get("waiting_penalty", 0.0))
            taz_detail["emission_penalty_sum"] += float(tls_detail.get("emission_penalty", 0.0))
            taz_detail["jam_penalty_sum"] += float(tls_detail.get("jam_penalty", 0.0))
            taz_detail["total_waiting_time"] += float(tls_detail.get("total_waiting_time", 0.0))
            taz_detail["total_co2"] += float(tls_detail.get("total_co2", 0.0))
            taz_detail["total_nox"] += float(tls_detail.get("total_nox", 0.0))
            taz_detail["max_jam_len"] = max(
                float(taz_detail.get("max_jam_len", 0.0)),
                float(tls_detail.get("max_jam", 0.0)),
            )

        return reward_by_taz, penalty_by_taz, details_by_taz

    def _collect_observation(self) -> torch.Tensor:
        if not self.sumo.isRunning():
            self._last_taz_metrics_by_id = {
                taz: self._zero_taz_reward_metrics()
                for taz in self.taz_ids
            }
            self._last_tls_metrics_by_id = {
                tls: self._zero_tls_metrics(taz_id=self.tls_to_taz.get(tls))
                for tls in self.tls_list
            }
            return torch.zeros(self.observation_spec.shape, dtype=torch.float32, device=self.device)

        raw_by_taz = {}
        for taz in self.taz_ids:
            try:
                raw_by_taz[taz] = self.sumo.get_taz_e2_metrics(taz, interval="last", mode="dict")
            except Exception:
                raw_by_taz[taz] = {"tls_data": {}, "taz_avg": {}}

        aggregated_by_tls = {}
        for taz in self.taz_ids:
            tls_data = raw_by_taz[taz].get("tls_data", {})
            for tls in self.taz_tls_map[taz]:
                dets = tls_data.get(tls, [])
                aggregated_by_tls[tls] = aggregate_tls_metrics(dets)

        vehicle_metrics_by_tls = self._collect_vehicle_metrics_by_tls()
        self._last_tls_metrics_by_id = {}
        self._episode_max_jam_len = 0.0

        for tls in self.tls_list:
            aggregated = aggregated_by_tls.get(
                tls,
                {
                    "veh_total": 0.0,
                    "mean_speed": 0.0,
                    "mean_occupancy": 0.0,
                    "max_jam": 0.0,
                    "pressure": 0.0,
                },
            )
            runtime_metrics = self._collect_tls_runtime_metrics(tls, aggregated)
            packed = dict(runtime_metrics)
            packed.update(vehicle_metrics_by_tls.get(tls, {}))
            packed["taz_id"] = self.tls_to_taz.get(tls)
            self._last_tls_metrics_by_id[tls] = packed
            self._episode_max_jam_len = max(
                self._episode_max_jam_len,
                self._safe_float(packed.get("max_jam", 0.0), 0.0),
            )

        self._last_taz_metrics_by_id = self._build_taz_metrics_from_tls()

        hour_sin, hour_cos = self._get_time_features()
        obs_rows = []
        for taz in self.taz_ids:
            row = []
            tls_ids = self.tls_by_taz[taz]
            for slot_idx in range(self.max_tls_per_taz):
                if slot_idx < len(tls_ids):
                    tls = tls_ids[slot_idx]
                    m = self._last_tls_metrics_by_id.get(
                        tls,
                        self._zero_tls_metrics(taz_id=taz),
                    )
                    phase_id = int(m.get("phase_id", 0))
                    n_phases = int(self.tls_num_phases.get(tls, self.max_num_phases))
                    one_hot = np.zeros(self.max_num_phases, dtype=np.float32)
                    if 0 <= phase_id < n_phases:
                        one_hot[phase_id] = 1.0

                    mean_occ = self._safe_float(m.get("mean_occupancy", 0.0), 0.0)
                    occ_norm = mean_occ / 100.0 if mean_occ > 1.5 else mean_occ
                    occ_norm = self._clip01(occ_norm)

                    row.extend(
                        [1.0]
                        + one_hot.tolist()
                        + [
                            float(np.clip(self._safe_float(m.get("phase_duration", 0.0), 0.0) / float(self.max_green), 0.0, 2.0)),
                            float(np.clip(self._safe_float(m.get("mean_speed", 0.0), 0.0) / self.speed_norm, 0.0, 2.0)),
                            float(np.clip(self._safe_float(m.get("max_jam", 0.0), 0.0) / self.jam_norm, 0.0, 2.0)),
                            float(occ_norm),
                            float(np.clip(self._safe_float(m.get("veh_total", 0.0), 0.0) / self.tls_veh_total_ref, 0.0, 5.0)),
                            float(np.clip(max(self._safe_float(m.get("pressure", 0.0), 0.0), 0.0) / self.tls_pressure_ref, 0.0, 5.0)),
                            float(np.clip(self._safe_float(m.get("active_vehicle_count", 0.0), 0.0) / self.tls_active_vehicle_ref, 0.0, 5.0)),
                            float(np.clip(self._safe_float(m.get("total_waiting_time", 0.0), 0.0) / self.tls_waiting_time_ref, 0.0, 5.0)),
                            float(np.clip(self._safe_float(m.get("avg_waiting_time", 0.0), 0.0) / self.waiting_ref, 0.0, 5.0)),
                            float(np.clip(self._safe_float(m.get("total_co2", 0.0), 0.0) / self.tls_co2_ref, 0.0, 5.0)),
                            float(np.clip(self._safe_float(m.get("total_nox", 0.0), 0.0) / self.tls_nox_ref, 0.0, 5.0)),
                        ]
                    )
                else:
                    row.extend([0.0] * self.per_tls_feature_dim)

            tm = self._last_taz_metrics_by_id.get(taz, self._zero_taz_reward_metrics())
            t_occ_norm = self._safe_float(tm.get("mean_occupancy", 0.0), 0.0)
            t_occ_norm = t_occ_norm / 100.0 if t_occ_norm > 1.5 else t_occ_norm
            t_occ_norm = self._clip01(t_occ_norm)
            row.extend(
                [
                    float(np.clip(self._safe_float(tm.get("max_jam_len", 0.0), 0.0) / self.jam_norm, 0.0, 2.0)),
                    float(np.clip(self._safe_float(tm.get("mean_speed", 0.0), 0.0) / self.speed_norm, 0.0, 2.0)),
                    float(t_occ_norm),
                    float(np.clip(self._safe_float(tm.get("veh_total", 0.0), 0.0) / self.taz_veh_total_ref, 0.0, 5.0)),
                    float(np.clip(self._safe_float(tm.get("active_vehicle_count", 0.0), 0.0) / self.taz_veh_total_ref, 0.0, 5.0)),
                    float(np.clip(self._safe_float(tm.get("total_waiting_time", 0.0), 0.0) / self.taz_waiting_time_ref, 0.0, 5.0)),
                    float(np.clip(self._safe_float(tm.get("total_co2", 0.0), 0.0) / self.taz_co2_ref, 0.0, 5.0)),
                    float(np.clip(self._safe_float(tm.get("total_nox", 0.0), 0.0) / self.taz_nox_ref, 0.0, 5.0)),
                    float(hour_sin),
                    float(hour_cos),
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

        self.sumo.start(activeGui=False, logFilePath=self.sumo.logFile, rl_mode=True)
        self.current_step = 0
        self._last_green_tls_snapshot = {}
        self._phase_duration_memory = {}
        self._active_program_id = {}
        self._build_tls_lane_cache()
        for tls in self.tls_list:
            self._sync_phase_duration_memory(tls)
        self._reset_duration_diagnostics()
        self.last_reward_components = {}
        self.last_duration_diagnostics = {}
        self._episode_max_jam_len = 0.0
        self._last_taz_metrics_by_id = {
            taz: self._zero_taz_reward_metrics()
            for taz in self.taz_ids
        }
        self._last_tls_metrics_by_id = {
            tls: self._zero_tls_metrics(taz_id=self.tls_to_taz.get(tls))
            for tls in self.tls_list
        }

        obs = self._collect_observation()
        tls_penalties, _ = self._build_tls_penalties()
        self._prev_tls_penalties = tls_penalties.copy()
        return TensorDict(
            {
                "observation": obs,
                "action_mask": self.get_action_mask(),
            },
            batch_size=[],
            device=self.device,
        )

    def _step(self, tensordict):
        action = tensordict["action"]
        if action.requires_grad:
            action = action.detach()
        action = action.to(self.device).reshape(len(self.taz_ids), self.max_tls_per_taz)

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
        tls_penalties, penalty_details_by_tls = self._build_tls_penalties()

        if self.current_step <= self.warmupSteps:
            tls_reward_vector = np.zeros_like(tls_penalties, dtype=np.float32)
        else:
            tls_reward_vector = np.clip(self._prev_tls_penalties - tls_penalties, -self.reward_clip, self.reward_clip)
        self._prev_tls_penalties = tls_penalties.copy()

        reward_by_taz, penalty_by_taz, penalty_details_by_taz = self._summarize_taz_rewards_and_penalties(
            reward_vector=tls_reward_vector,
            penalties=tls_penalties,
            penalty_details_by_tls=penalty_details_by_tls,
        )
        taz_reward_vector = np.asarray([reward_by_taz[taz] for taz in self.taz_ids], dtype=np.float32)
        taz_penalty_vector = np.asarray([penalty_by_taz[taz] for taz in self.taz_ids], dtype=np.float32)

        terminated = (not sim_running) and (not time_limit_reached)
        truncated = bool(time_limit_reached)
        if terminated or truncated:
            if self.sumo.isLoaded():
                self.sumo.end()

        reward_by_tls = {
            tls: float(tls_reward_vector[idx])
            for idx, tls in enumerate(self.tls_list)
        }
        penalty_by_tls = {
            tls: float(tls_penalties[idx])
            for idx, tls in enumerate(self.tls_list)
        }

        self.last_reward_components = {
            "reward_basis": "per_taz_sum_tls_wait_emission_jam_delta",
            "reward_by_tls": reward_by_tls,
            "penalty_by_tls": penalty_by_tls,
            "penalty_details_by_tls": penalty_details_by_tls,
            "reward_by_taz": reward_by_taz,
            "penalty_by_taz": penalty_by_taz,
            "penalty_details_by_taz": penalty_details_by_taz,
            "reward_mean": float(np.mean(taz_reward_vector)),
            "reward_std": float(np.std(taz_reward_vector)),
            "reward_min": float(np.min(taz_reward_vector)),
            "reward_max": float(np.max(taz_reward_vector)),
            "reward_sum": float(np.sum(taz_reward_vector)),
            "penalty_mean": float(np.mean(taz_penalty_vector)),
            "penalty_std": float(np.std(taz_penalty_vector)),
            "penalty_min": float(np.min(taz_penalty_vector)),
            "penalty_max": float(np.max(taz_penalty_vector)),
            "penalty_sum": float(np.sum(taz_penalty_vector)),
            "tls_reward_mean": float(np.mean(tls_reward_vector)),
            "tls_penalty_mean": float(np.mean(tls_penalties)),
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
                "reward": torch.tensor(taz_reward_vector, dtype=torch.float32, device=self.device),
                "reward_tls": torch.tensor(tls_reward_vector, dtype=torch.float32, device=self.device),
                "applied_action": applied_action,
                "applied_duration_delta": applied_duration_delta,
                "terminated": torch.tensor(terminated, device=self.device),
                "truncated": torch.tensor(truncated, device=self.device),
            },
            batch_size=[],
            device=self.device,
        )
