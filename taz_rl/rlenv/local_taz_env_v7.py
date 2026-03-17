# local_taz_env_v7.py
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Optional

import libtraci
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase

try:
    from torchrl.data import Bounded as BoundedSpec, Unbounded as UnboundedSpec
except Exception:
    from torchrl.data import BoundedTensorSpec as BoundedSpec
    from torchrl.data import UnboundedContinuousTensorSpec as UnboundedSpec

from libraries.constants import SUMO_NETWORK_PATH, TAZ_FILE
from taz_rl.rlenv.aggregation import aggregate_taz_metrics, aggregate_tls_metrics


class SumoTazEnvV7(EnvBase):
    """
    v7 environment:
      - controls all TAZes and all TLS in one synchronized env
      - one global action vector (all TLS) per step
      - one SUMO advance per step for all agents together
      - terminal reward computed once per episode
      - optional reward as delta vs baseline (no-agent) simulation
    """

    def __init__(
        self,
        sumoSimulator,
        stepSize: int = 300,
        device: str = "cpu",
        taz_map_path: str = TAZ_FILE,
        # observation normalization
        speed_norm: float = 10.0,
        jam_norm: float = 10.0,
        # episode control
        warmupSteps: int = 0,
        cooldownSteps: int = 12,
        # live feature extraction
        live_vehicle_sample: int = 4096,
        active_vehicle_ref: float = 2500.0,
        waiting_ref: float = 60.0,
        vehicle_time_loss_ref: float = 120.0,
        emission_vehicle_co2_ref: float = 3500.0,
        emission_vehicle_nox_ref: float = 1.5,
        emission_vehicle_fuel_ref: float = 1200.0,
        # aggregated E2 normalization refs
        tls_veh_total_ref: float = 120.0,
        tls_pressure_ref: float = 10000.0,
        taz_veh_total_ref: float = 2500.0,
        taz_std_occupancy_ref: float = 0.25,
        # phase clipping
        min_green: int = 10,
        max_green: int = 300,
        # terminal reward weights
        time_loss_weight: float = 0.65,
        emission_weight: float = 0.25,
        jam_weight: float = 0.10,
        emission_co2_mix_weight: float = 0.34,
        emission_nox_mix_weight: float = 0.33,
        emission_fuel_mix_weight: float = 0.33,
        # terminal reward references (1-hour totals)
        time_loss_ref: float = 2500000.0,
        co2_ref: float = 11000000000.0,
        nox_ref: float = 4000000.0,
        fuel_ref: float = 3600000000.0,
        max_jam_len_ref: float = 120.0,
        # reward clipping
        metric_clip: float = 5.0,
        reward_clip: float = 5.0,
        # reward mode
        comparison_reward_enabled: bool = False,
        # net file to read phases
        tls_add_path: str = SUMO_NETWORK_PATH + "/optimized_tls.add.xml",
        **kwargs,
    ):
        super().__init__(device=device)

        self.sumo = sumoSimulator
        self.stepSize = int(stepSize)
        self.speed_norm = float(speed_norm)
        self.jam_norm = float(jam_norm)
        self.warmupSteps = int(warmupSteps)
        self.cooldownSteps = int(cooldownSteps)
        self.tls_add_path = str(tls_add_path)

        # live feature settings
        self.live_vehicle_sample = max(int(live_vehicle_sample), 64)
        self.active_vehicle_ref = max(float(active_vehicle_ref), 1e-6)
        self.waiting_ref = max(float(waiting_ref), 1e-6)
        self.vehicle_time_loss_ref = max(float(vehicle_time_loss_ref), 1e-6)
        self.emission_vehicle_co2_ref = max(float(emission_vehicle_co2_ref), 1e-6)
        self.emission_vehicle_nox_ref = max(float(emission_vehicle_nox_ref), 1e-6)
        self.emission_vehicle_fuel_ref = max(float(emission_vehicle_fuel_ref), 1e-6)

        # E2 normalization refs
        self.tls_veh_total_ref = max(float(tls_veh_total_ref), 1e-6)
        self.tls_pressure_ref = max(float(tls_pressure_ref), 1e-6)
        self.taz_veh_total_ref = max(float(taz_veh_total_ref), 1e-6)
        self.taz_std_occupancy_ref = max(float(taz_std_occupancy_ref), 1e-6)

        # phase constraints
        self.min_green = int(min_green)
        self.max_green = int(max_green)

        # reward config
        self.time_loss_weight = float(time_loss_weight)
        self.emission_weight = float(emission_weight)
        self.jam_weight = float(jam_weight)
        self.emission_co2_mix_weight = float(emission_co2_mix_weight)
        self.emission_nox_mix_weight = float(emission_nox_mix_weight)
        self.emission_fuel_mix_weight = float(emission_fuel_mix_weight)
        self.time_loss_ref = max(float(time_loss_ref), 1e-6)
        self.co2_ref = max(float(co2_ref), 1e-6)
        self.nox_ref = max(float(nox_ref), 1e-6)
        self.fuel_ref = max(float(fuel_ref), 1e-6)
        self.max_jam_len_ref = max(float(max_jam_len_ref), 1e-6)
        self.metric_clip = max(float(metric_clip), 0.0)
        self.reward_clip = max(float(reward_clip), 1e-6)
        self.comparison_reward_enabled = bool(comparison_reward_enabled)

        # normalize top-level reward weights
        reward_weight_sum = self.time_loss_weight + self.emission_weight + self.jam_weight
        if reward_weight_sum <= 0.0:
            self.time_loss_weight = 0.65
            self.emission_weight = 0.25
            self.jam_weight = 0.10
            reward_weight_sum = 1.0
        self.time_loss_weight /= reward_weight_sum
        self.emission_weight /= reward_weight_sum
        self.jam_weight /= reward_weight_sum

        # normalize emission mix weights
        emission_mix_sum = (
            self.emission_co2_mix_weight
            + self.emission_nox_mix_weight
            + self.emission_fuel_mix_weight
        )
        if emission_mix_sum <= 0.0:
            self.emission_co2_mix_weight = 0.34
            self.emission_nox_mix_weight = 0.33
            self.emission_fuel_mix_weight = 0.33
            emission_mix_sum = 1.0
        self.emission_co2_mix_weight /= emission_mix_sum
        self.emission_nox_mix_weight /= emission_mix_sum
        self.emission_fuel_mix_weight /= emission_mix_sum

        # ---------- topology from TAZ map ----------
        self.taz_map_path = str(taz_map_path)
        self.taz_tls_map = self._load_taz_tls_map(self.taz_map_path)
        self.taz_ids = list(self.taz_tls_map.keys())
        self.tls_list = self._build_unique_tls_list(self.taz_tls_map)
        if len(self.tls_list) == 0:
            raise RuntimeError("No TLS found from TAZ mapping.")

        self.tls_to_taz = {}
        for taz in self.taz_ids:
            for tls in self.taz_tls_map[taz]:
                if tls not in self.tls_to_taz:
                    self.tls_to_taz[tls] = taz

        self.tls_num_phases, self.max_num_phases = self._compute_tls_phases_from_net(self.tls_add_path)
        self.max_num_phases = max(int(self.max_num_phases), 1)

        # observation shape
        n_features_per_tls = self.max_num_phases + 6
        n_tls_features = len(self.tls_list) * n_features_per_tls
        n_features_per_taz = 9
        n_taz_features = len(self.taz_ids) * n_features_per_taz
        self.live_feature_dim = 7
        state_dim = n_tls_features + n_taz_features + self.live_feature_dim

        self.observation_spec = UnboundedSpec(shape=(state_dim,))
        self.action_spec = BoundedSpec(low=-15.0, high=15.0, shape=(len(self.tls_list),))
        self.reward_spec = UnboundedSpec(shape=(1,))

        self.discrete_action_values = torch.tensor(
            [-15.0, -10.0, -5.0, 5.0, 10.0, 15.0],
            dtype=torch.float32,
            device=self.device,
        )

        # runtime state
        self.current_hour = 0
        self.current_step = 0
        self.output_subdir = "output"
        self.tripinfo_filename = "tripinfos.xml"
        self.emission_filename = "emission-output.xml"

        self._last_taz_metrics_by_id = {
            taz: self._zero_taz_metrics()
            for taz in self.taz_ids
        }
        self._last_green_tls_snapshot = {}
        self._phase_duration_memory = {}
        self._active_program_id = {}
        self._episode_max_jam_len = 0.0

        self.last_reward_components = {}
        self.reference_penalty = None
        self.reference_reward_components = {}

    # ---------- setup helpers ----------
    @staticmethod
    def _build_unique_tls_list(taz_tls_map: dict) -> list[str]:
        ordered = []
        seen = set()
        for taz in taz_tls_map:
            for tls in taz_tls_map[taz]:
                if tls in seen:
                    continue
                seen.add(tls)
                ordered.append(str(tls))
        return ordered

    def _load_taz_tls_map(self, taz_map_path: str) -> dict:
        sim_map = getattr(self.sumo, "tazTlsMap", None)
        if isinstance(sim_map, dict) and len(sim_map) > 0:
            return {
                str(k): [str(x) for x in v]
                for k, v in sim_map.items()
                if isinstance(v, list)
            }

        if not os.path.exists(taz_map_path):
            raise FileNotFoundError(f"TAZ map not found: {taz_map_path}")
        with open(taz_map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise RuntimeError("Invalid TAZ map content: expected dict.")
        return {
            str(k): [str(x) for x in v]
            for k, v in raw.items()
            if isinstance(v, list)
        }

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
        }

    # ---------- public helpers ----------
    def set_episode_context(self, hour: int):
        self.current_hour = int(hour) % 24

    def set_baseline_penalty(self, penalty: Optional[float], components: Optional[dict] = None):
        if penalty is None:
            self.reference_penalty = None
            self.reference_reward_components = {}
            return
        self.reference_penalty = float(penalty)
        self.reference_reward_components = dict(components or {})

    def get_taz_ids(self) -> list[str]:
        return list(self.taz_ids)

    def get_tls_ids(self) -> list[str]:
        return list(self.tls_list)

    # ---------- generic utils ----------
    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            v = float(value)
        except Exception:
            return default
        if not np.isfinite(v):
            return default
        return v

    @staticmethod
    def _get_attr_case_insensitive(elem: ET.Element, key: str):
        if key in elem.attrib:
            return elem.attrib[key]
        target = key.lower()
        for k, v in elem.attrib.items():
            if k.lower() == target:
                return v
        return None

    def _safe_parse_xml_root(self, xml_path: str):
        try:
            return ET.parse(xml_path).getroot()
        except Exception:
            pass

        try:
            with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            return None
        if text is None:
            return None

        text = text.replace("\ufeff", "")
        text = re.sub(r"<\?xml[^>]*\?>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        text = text.strip()
        if not text:
            return None

        root_tag = None
        for m in re.finditer(r"<([A-Za-z_][\w\-\.:]*)[^>]*>", text):
            tag = m.group(1)
            if tag.startswith("?") or tag.startswith("!"):
                continue
            if tag.lower() == "xml":
                continue
            root_tag = tag
            break
        if root_tag is None:
            return None

        closing = f"</{root_tag}>"
        close_pos = text.rfind(closing)
        if close_pos >= 0:
            candidate = text[: close_pos + len(closing)]
        else:
            last_gt = text.rfind(">")
            if last_gt <= 0:
                return None
            candidate = text[: last_gt + 1] + closing

        for _ in range(32):
            try:
                return ET.fromstring(candidate)
            except Exception:
                cut = candidate.rfind("<", 0, max(0, len(candidate) - len(closing)))
                if cut <= 0:
                    break
                candidate = candidate[:cut] + closing
        return None

    @staticmethod
    def _is_green_phase_state(phase_state: str) -> bool:
        return ("g" in phase_state) or ("G" in phase_state)

    def _get_time_features(self):
        angle = 2.0 * np.pi * (self.current_hour % 24) / 24.0
        return float(np.sin(angle)), float(np.cos(angle))

    def _collect_live_features(self) -> torch.Tensor:
        if not self.sumo.isRunning():
            return torch.zeros(self.live_feature_dim, dtype=torch.float32, device=self.device)

        try:
            vehicle_ids = list(libtraci.vehicle.getIDList())
        except Exception:
            return torch.zeros(self.live_feature_dim, dtype=torch.float32, device=self.device)
        if not vehicle_ids:
            return torch.zeros(self.live_feature_dim, dtype=torch.float32, device=self.device)

        if len(vehicle_ids) > self.live_vehicle_sample:
            vehicle_ids = vehicle_ids[: self.live_vehicle_sample]

        n = float(len(vehicle_ids))
        speed_sum = 0.0
        waiting_sum = 0.0
        time_loss_sum = 0.0
        co2_sum = 0.0
        nox_sum = 0.0
        fuel_sum = 0.0
        for vid in vehicle_ids:
            try:
                speed_sum += self._safe_float(libtraci.vehicle.getSpeed(vid), 0.0)
                waiting_sum += self._safe_float(libtraci.vehicle.getWaitingTime(vid), 0.0)
                time_loss_sum += self._safe_float(libtraci.vehicle.getTimeLoss(vid), 0.0)
                co2_sum += self._safe_float(libtraci.vehicle.getCO2Emission(vid), 0.0)
                nox_sum += self._safe_float(libtraci.vehicle.getNOxEmission(vid), 0.0)
                fuel_sum += self._safe_float(libtraci.vehicle.getFuelConsumption(vid), 0.0)
            except Exception:
                continue

        mean_speed = speed_sum / max(n, 1.0)
        mean_waiting = waiting_sum / max(n, 1.0)
        mean_time_loss = time_loss_sum / max(n, 1.0)
        mean_co2 = co2_sum / max(n, 1.0)
        mean_nox = nox_sum / max(n, 1.0)
        mean_fuel = fuel_sum / max(n, 1.0)

        live_features = [
            float(np.clip(n / self.active_vehicle_ref, 0.0, 5.0)),
            float(np.clip(mean_speed / max(self.speed_norm, 1e-6), 0.0, 3.0)),
            float(np.clip(mean_waiting / self.waiting_ref, 0.0, 5.0)),
            float(np.clip(mean_time_loss / self.vehicle_time_loss_ref, 0.0, 5.0)),
            float(np.clip(mean_co2 / self.emission_vehicle_co2_ref, 0.0, 5.0)),
            float(np.clip(mean_nox / self.emission_vehicle_nox_ref, 0.0, 5.0)),
            float(np.clip(mean_fuel / self.emission_vehicle_fuel_ref, 0.0, 5.0)),
        ]
        return torch.tensor(live_features, dtype=torch.float32, device=self.device)

    # ---------- net / tls helpers ----------
    def _compute_tls_phases_from_net(self, net_path: str):
        tls_num_phases = {}
        max_phases = 0
        try:
            tree = ET.parse(net_path)
            root = tree.getroot()
            for tl_logic in root.findall("tlLogic"):
                tls_id = str(tl_logic.get("id"))
                if tls_id not in self.tls_list:
                    continue
                n_phases = len(tl_logic.findall("phase"))
                if n_phases <= 0:
                    n_phases = 1
                tls_num_phases[tls_id] = int(n_phases)
                max_phases = max(max_phases, int(n_phases))
        except Exception:
            pass

        if max_phases <= 0:
            max_phases = 1
        for tls in self.tls_list:
            if tls not in tls_num_phases:
                tls_num_phases[tls] = max_phases
        return tls_num_phases, max_phases

    def _find_previous_green_phase_id(self, program, current_phase_id: int) -> Optional[int]:
        phases = list(getattr(program, "phases", []))
        n_phases = len(phases)
        if n_phases == 0:
            return None

        idx = (int(current_phase_id) - 1) % n_phases
        for _ in range(n_phases):
            if self._is_green_phase_state(getattr(phases[idx], "state", "")):
                return idx
            idx = (idx - 1) % n_phases
        return None

    def _discretize_actions(self, action_tensor: torch.Tensor) -> torch.Tensor:
        return torch.where(
            action_tensor < -13.0,
            torch.full_like(action_tensor, -15.0),
            torch.where(
                action_tensor < -3.0,
                torch.full_like(action_tensor, -10.0),
                torch.where(
                    action_tensor < 0.0,
                    torch.full_like(action_tensor, -5.0),
                    torch.where(
                        action_tensor < 3.0,
                        torch.full_like(action_tensor, 5.0),
                        torch.where(
                            action_tensor < 13.0,
                            torch.full_like(action_tensor, 10.0),
                            torch.full_like(action_tensor, 15.0),
                        ),
                    ),
                ),
            ),
        )

    def _get_active_program_logic(self, tls_id: str):
        all_programs = list(libtraci.trafficlight.getAllProgramLogics(tls_id))
        if not all_programs:
            return None, None
        active_program_id = str(libtraci.trafficlight.getProgram(tls_id))
        for prog in all_programs:
            if str(getattr(prog, "programID", "")) == active_program_id:
                return prog, active_program_id
        return all_programs[0], str(getattr(all_programs[0], "programID", ""))

    def _sync_phase_duration_memory(self, tls_id: str):
        try:
            program, program_id = self._get_active_program_logic(tls_id)
        except Exception:
            return
        if program is None:
            return
        durations = [self._safe_float(getattr(ph, "duration", 0.0), 0.0) for ph in list(program.phases)]
        if not durations:
            return
        self._phase_duration_memory[tls_id] = durations
        self._active_program_id[tls_id] = program_id

    def _apply_action_to_tls(self, tls_id: str, action_value: float) -> float:
        try:
            program, program_id = self._get_active_program_logic(tls_id)
        except Exception:
            return 0.0
        if program is None:
            return 0.0

        if (
            tls_id not in self._phase_duration_memory
            or tls_id not in self._active_program_id
            or self._active_program_id.get(tls_id) != program_id
            or len(self._phase_duration_memory.get(tls_id, [])) != len(list(program.phases))
        ):
            self._sync_phase_duration_memory(tls_id)

        phase_memory = self._phase_duration_memory.get(tls_id, [])
        if len(phase_memory) != len(list(program.phases)):
            phase_memory = [self._safe_float(getattr(ph, "duration", 0.0), 0.0) for ph in list(program.phases)]
            self._phase_duration_memory[tls_id] = phase_memory

        current_phase_id = int(program.currentPhaseIndex)
        target_phase_id = current_phase_id
        current_phase = program.phases[current_phase_id]
        if not self._is_green_phase_state(current_phase.state):
            prev_green_phase_id = self._find_previous_green_phase_id(program, current_phase_id)
            if prev_green_phase_id is None:
                return 0.0
            target_phase_id = int(prev_green_phase_id)

        base = self._safe_float(
            phase_memory[target_phase_id],
            self._safe_float(program.phases[target_phase_id].duration, 0.0),
        )
        new_dur = int(np.clip(base + float(action_value), self.min_green, self.max_green))
        self._phase_duration_memory[tls_id][target_phase_id] = float(new_dur)

        applied_duration = float(new_dur)
        try:
            for i, ph in enumerate(list(program.phases)):
                d = int(np.clip(self._phase_duration_memory[tls_id][i], self.min_green, self.max_green))
                ph.maxDur = d
                ph.minDur = d
                ph.duration = d
            try:
                program.currentPhaseIndex = current_phase_id
            except Exception:
                pass
            libtraci.trafficlight.setProgramLogic(tls_id, program)

            if target_phase_id == current_phase_id:
                try:
                    libtraci.trafficlight.setPhaseDuration(tls_id, float(new_dur))
                except Exception:
                    pass

            after_program, _ = self._get_active_program_logic(tls_id)
            if after_program is not None:
                applied_duration = self._safe_float(
                    after_program.phases[target_phase_id].duration,
                    float(new_dur),
                )
                self._phase_duration_memory[tls_id][target_phase_id] = float(applied_duration)
        except Exception:
            try:
                self.sumo.set_tls_phase_duration(tls_id, target_phase_id, new_dur)
            except Exception:
                pass

        self._last_green_tls_snapshot[tls_id] = {
            **self._last_green_tls_snapshot.get(tls_id, {}),
            "phase_id": target_phase_id,
            "phase_duration": float(applied_duration),
        }
        return float(applied_duration - base)

    # ---------- observation ----------
    def _collect_tls_runtime_metrics(self, tls_id: str, aggregated: dict) -> dict:
        try:
            program = libtraci.trafficlight.getAllProgramLogics(tls_id)[0]
            current_phase_id = int(program.currentPhaseIndex)
            current_phase = program.phases[current_phase_id]
            current_phase_state = getattr(current_phase, "state", "")
        except Exception:
            program = None
            current_phase_id = int(libtraci.trafficlight.getPhase(tls_id))
            current_phase_state = ""

        try:
            current_phase_duration = float(libtraci.trafficlight.getPhaseDuration(tls_id))
        except Exception:
            if program is not None and 0 <= current_phase_id < len(program.phases):
                current_phase_duration = self._safe_float(program.phases[current_phase_id].duration, 0.0)
            else:
                current_phase_duration = 0.0

        phase_id_for_obs = current_phase_id
        phase_duration_for_obs = current_phase_duration
        metrics_for_obs = dict(aggregated)

        if self._is_green_phase_state(current_phase_state):
            self._last_green_tls_snapshot[tls_id] = {
                "phase_id": phase_id_for_obs,
                "phase_duration": phase_duration_for_obs,
                "veh_total": self._safe_float(aggregated.get("veh_total", 0.0), 0.0),
                "mean_speed": self._safe_float(aggregated.get("mean_speed", 0.0), 0.0),
                "mean_occupancy": self._safe_float(aggregated.get("mean_occupancy", 0.0), 0.0),
                "max_jam": self._safe_float(aggregated.get("max_jam", 0.0), 0.0),
                "pressure": self._safe_float(aggregated.get("pressure", 0.0), 0.0),
            }
        else:
            if program is not None:
                prev_green_phase_id = self._find_previous_green_phase_id(program, current_phase_id)
                if prev_green_phase_id is not None:
                    phase_id_for_obs = int(prev_green_phase_id)
                    phase_duration_for_obs = self._safe_float(
                        program.phases[prev_green_phase_id].duration,
                        phase_duration_for_obs,
                    )

            cached_metrics = self._last_green_tls_snapshot.get(tls_id)
            if cached_metrics is not None:
                metrics_for_obs = {
                    "veh_total": self._safe_float(cached_metrics.get("veh_total", aggregated.get("veh_total", 0.0)), 0.0),
                    "mean_speed": self._safe_float(cached_metrics.get("mean_speed", aggregated.get("mean_speed", 0.0)), 0.0),
                    "mean_occupancy": self._safe_float(cached_metrics.get("mean_occupancy", aggregated.get("mean_occupancy", 0.0)), 0.0),
                    "max_jam": self._safe_float(cached_metrics.get("max_jam", aggregated.get("max_jam", 0.0)), 0.0),
                    "pressure": self._safe_float(cached_metrics.get("pressure", aggregated.get("pressure", 0.0)), 0.0),
                }
                phase_duration_for_obs = self._safe_float(
                    cached_metrics.get("phase_duration", phase_duration_for_obs),
                    phase_duration_for_obs,
                )

        return {
            "tls_id": tls_id,
            "phase_id": int(phase_id_for_obs),
            "phase_duration": float(phase_duration_for_obs),
            "veh_total": self._safe_float(metrics_for_obs.get("veh_total", 0.0), 0.0),
            "mean_speed": self._safe_float(metrics_for_obs.get("mean_speed", 0.0), 0.0),
            "mean_occupancy": self._safe_float(metrics_for_obs.get("mean_occupancy", 0.0), 0.0),
            "max_jam": self._safe_float(metrics_for_obs.get("max_jam", 0.0), 0.0),
            "pressure": self._safe_float(metrics_for_obs.get("pressure", 0.0), 0.0),
        }

    def _collect_observation(self) -> torch.Tensor:
        if not self.sumo.isRunning():
            self._last_taz_metrics_by_id = {taz: self._zero_taz_metrics() for taz in self.taz_ids}
            return torch.zeros(self.observation_spec.shape, dtype=torch.float32, device=self.device)

        # read E2 raw data for each TAZ once
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

        tls_metrics_by_id = {}
        for tls in self.tls_list:
            aggregated = aggregated_by_tls.get(
                tls,
                {"veh_total": 0.0, "mean_speed": 0.0, "mean_occupancy": 0.0, "max_jam": 0.0, "pressure": 0.0},
            )
            tls_metrics_by_id[tls] = self._collect_tls_runtime_metrics(tls, aggregated)

        # per-TAZ aggregates
        self._last_taz_metrics_by_id = {}
        for taz in self.taz_ids:
            tls_metrics = [tls_metrics_by_id[tls] for tls in self.taz_tls_map[taz] if tls in tls_metrics_by_id]
            taz_metrics = aggregate_taz_metrics(tls_metrics)
            packed = {
                "mean_speed": self._safe_float(taz_metrics.get("mean_speed", 0.0), 0.0),
                "max_jam_len": self._safe_float(taz_metrics.get("max_jam_len", 0.0), 0.0),
                "mean_occupancy": self._safe_float(taz_metrics.get("mean_occupancy", 0.0), 0.0),
                "veh_total": self._safe_float(taz_metrics.get("veh_total", 0.0), 0.0),
                "critical_ratio": self._safe_float(taz_metrics.get("critical_ratio", 0.0), 0.0),
                "max_occupancy": self._safe_float(taz_metrics.get("max_occupancy", 0.0), 0.0),
                "std_occupancy": self._safe_float(taz_metrics.get("std_occupancy", 0.0), 0.0),
            }
            self._last_taz_metrics_by_id[taz] = packed
            self._episode_max_jam_len = max(self._episode_max_jam_len, packed["max_jam_len"])

        obs = []

        # TLS block
        for tls in self.tls_list:
            m = tls_metrics_by_id[tls]
            phase_id = int(m.get("phase_id", 0))
            n_phases = int(self.tls_num_phases.get(tls, self.max_num_phases))

            one_hot = np.zeros(self.max_num_phases, dtype=np.float32)
            if 0 <= phase_id < n_phases:
                one_hot[phase_id] = 1.0
            obs.extend(one_hot.tolist())

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
            pressure_norm = np.clip(pressure / self.tls_pressure_ref, 0.0, 5.0)
            obs.extend([phase_duration_norm, mean_speed_norm, max_jam_norm, occ_norm, veh_total_norm, pressure_norm])

        # TAZ block
        hour_sin, hour_cos = self._get_time_features()
        for taz in self.taz_ids:
            tm = self._last_taz_metrics_by_id[taz]
            t_speed = tm["mean_speed"]
            t_jam = tm["max_jam_len"]
            t_occ = tm["mean_occupancy"]
            t_veh = tm["veh_total"]
            t_critical = tm["critical_ratio"]
            t_max_occ = tm["max_occupancy"]
            t_std_occ = tm["std_occupancy"]

            t_occ_norm = t_occ / 100.0 if t_occ > 1.5 else t_occ
            t_occ_norm = self._clip01(t_occ_norm)
            t_critical_norm = self._clip01(t_critical)
            t_max_occ_norm = self._clip01(t_max_occ)
            t_std_occ_norm = np.clip(t_std_occ / self.taz_std_occupancy_ref, 0.0, 5.0)

            obs.extend([
                float(np.clip(t_speed / self.speed_norm, 0.0, 2.0)),
                float(np.clip(t_jam / self.jam_norm, 0.0, 2.0)),
                float(t_occ_norm),
                float(np.clip(t_veh / self.taz_veh_total_ref, 0.0, 5.0)),
                float(t_critical_norm),
                float(t_max_occ_norm),
                float(t_std_occ_norm),
                hour_sin,
                hour_cos,
            ])

        # city-level live block
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        live_obs = self._collect_live_features()
        obs_t = torch.cat([obs_t, live_obs], dim=0)
        if not torch.isfinite(obs_t).all():
            obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs_t

    # ---------- terminal metrics / reward ----------
    def _get_output_dir(self) -> Optional[str]:
        root = getattr(self.sumo, "typePath", None)
        if root is None:
            root = getattr(self.sumo, "routeFilePath", None)
        if root is None:
            return None
        return os.path.join(str(root), self.output_subdir)

    def _parse_tripinfo_metrics(self, tripinfo_path: str) -> dict:
        if not os.path.exists(tripinfo_path):
            return {"trip_count": 0, "total_arrival": 0.0, "total_time_loss": 0.0, "ok": False}

        root = self._safe_parse_xml_root(tripinfo_path)
        if root is None:
            return {"trip_count": 0, "total_arrival": 0.0, "total_time_loss": 0.0, "ok": False}

        arrivals = []
        time_losses = []
        for trip in root.findall("tripinfo"):
            arrival = self._safe_float(trip.get("arrival"), np.nan)
            time_loss = self._safe_float(trip.get("timeLoss"), np.nan)
            if np.isfinite(arrival) and np.isfinite(time_loss):
                arrivals.append(arrival)
                time_losses.append(time_loss)

        if not arrivals:
            return {"trip_count": 0, "total_arrival": 0.0, "total_time_loss": 0.0, "ok": False}

        return {
            "trip_count": int(len(arrivals)),
            "total_arrival": float(sum(arrivals)),
            "total_time_loss": float(sum(time_losses)),
            "avg_arrival": float(sum(arrivals) / len(arrivals)),
            "avg_time_loss": float(sum(time_losses) / len(time_losses)),
            "ok": True,
        }

    def _parse_emission_metric(self, emission_path: str, metric: str) -> dict:
        if not os.path.exists(emission_path):
            return {"total_emission": 0.0, "avg_vehicle_emission": 0.0, "ok": False}

        root = self._safe_parse_xml_root(emission_path)
        if root is None:
            return {"total_emission": 0.0, "avg_vehicle_emission": 0.0, "ok": False}

        totals = []
        per_vehicle = []
        for timestep in root.findall("timestep"):
            vehicles = timestep.findall("vehicle")
            vehicle_count = len(vehicles)
            if vehicle_count == 0:
                for key in ("vehicleCount", "vehicles", "nVeh", "numVehicles", "count"):
                    value = self._get_attr_case_insensitive(timestep, key)
                    if value is None:
                        continue
                    vehicle_count = int(max(0, round(self._safe_float(value, 0.0))))
                    break

            total_attr = self._get_attr_case_insensitive(timestep, metric)
            if total_attr is not None:
                total_value = self._safe_float(total_attr, 0.0)
            else:
                total_value = sum(
                    self._safe_float(self._get_attr_case_insensitive(v, metric), 0.0)
                    for v in vehicles
                )

            totals.append(total_value)
            if vehicle_count > 0:
                per_vehicle.append(total_value / float(vehicle_count))

        if not totals:
            return {"total_emission": 0.0, "avg_vehicle_emission": 0.0, "ok": False}

        return {
            "total_emission": float(sum(totals)),
            "avg_vehicle_emission": float(sum(per_vehicle) / len(per_vehicle)) if per_vehicle else 0.0,
            "ok": True,
        }

    def _normalize_metric(self, value: float, ref: float) -> float:
        return float(np.clip(float(value) / float(ref), 0.0, self.metric_clip))

    def _compute_terminal_reward(self) -> tuple[float, dict]:
        output_dir = self._get_output_dir()
        if output_dir is None:
            return -self.reward_clip, {"parse_ok": False, "reason": "missing_output_dir"}

        tripinfo_path = os.path.join(output_dir, self.tripinfo_filename)
        emission_path = os.path.join(output_dir, self.emission_filename)

        trip = self._parse_tripinfo_metrics(tripinfo_path)
        co2 = self._parse_emission_metric(emission_path, "CO2")
        nox = self._parse_emission_metric(emission_path, "NOx")
        fuel = self._parse_emission_metric(emission_path, "fuel")
        parse_ok = bool(trip["ok"] and co2["ok"] and nox["ok"] and fuel["ok"])
        if not parse_ok:
            return -self.reward_clip, {
                "parse_ok": False,
                "reason": "missing_or_invalid_tripinfo_or_emission",
                "trip_count": int(trip.get("trip_count", 0)),
                "tripinfo_path": tripinfo_path,
                "emission_path": emission_path,
            }

        time_loss_norm = self._normalize_metric(trip["total_time_loss"], self.time_loss_ref)
        co2_norm = self._normalize_metric(co2["total_emission"], self.co2_ref)
        nox_norm = self._normalize_metric(nox["total_emission"], self.nox_ref)
        fuel_norm = self._normalize_metric(fuel["total_emission"], self.fuel_ref)
        emission_norm = (
            self.emission_co2_mix_weight * co2_norm
            + self.emission_nox_mix_weight * nox_norm
            + self.emission_fuel_mix_weight * fuel_norm
        )
        jam_norm = self._normalize_metric(self._episode_max_jam_len, self.max_jam_len_ref)
        penalty = (
            self.time_loss_weight * time_loss_norm
            + self.emission_weight * emission_norm
            + self.jam_weight * jam_norm
        )

        use_comparison = bool(self.comparison_reward_enabled and (self.reference_penalty is not None))
        baseline_penalty = float(self.reference_penalty) if use_comparison else 0.0
        reward_raw = baseline_penalty - penalty if use_comparison else -penalty
        reward = float(np.clip(reward_raw, -self.reward_clip, self.reward_clip))

        components = {
            "parse_ok": True,
            "reward_basis": "delta_vs_baseline" if use_comparison else "hour_totals_simple",
            "trip_count": int(trip["trip_count"]),
            "total_arrival": float(trip["total_arrival"]),
            "total_time_loss": float(trip["total_time_loss"]),
            "total_co2": float(co2["total_emission"]),
            "total_nox": float(nox["total_emission"]),
            "total_fuel": float(fuel["total_emission"]),
            "episode_max_jam_len": float(self._episode_max_jam_len),
            "time_loss_norm": float(time_loss_norm),
            "emission_norm": float(emission_norm),
            "jam_norm": float(jam_norm),
            "co2_norm": float(co2_norm),
            "nox_norm": float(nox_norm),
            "fuel_norm": float(fuel_norm),
            "time_loss_weight": float(self.time_loss_weight),
            "emission_weight": float(self.emission_weight),
            "jam_weight": float(self.jam_weight),
            "comparison_reward_enabled": bool(self.comparison_reward_enabled),
            "baseline_penalty": float(baseline_penalty),
            "delta_penalty": float(baseline_penalty - penalty) if use_comparison else 0.0,
            "reward_raw": float(reward_raw),
            "penalty": float(penalty),
            "total": float(reward),
        }
        return reward, components

    # ---------- reference run helper ----------
    def run_reference_episode_no_agent(self, hour: int) -> tuple[float, dict]:
        self.set_baseline_penalty(None)
        self.set_episode_context(hour)
        self.reset()

        while True:
            self.sumo.step(quantity=self.stepSize)
            self.current_step += 1
            self._collect_observation()
            sim_running = self.sumo.isRunning()
            time_limit_reached = self.current_step >= self.cooldownSteps
            terminated = (not sim_running) and (not time_limit_reached)
            truncated = bool(time_limit_reached)
            if terminated or truncated:
                break

        if self.sumo.isLoaded():
            self.sumo.end()

        _, baseline_components = self._compute_terminal_reward()
        baseline_penalty = float(baseline_components.get("penalty", 0.0))
        return baseline_penalty, baseline_components

    # ---------- torchrl required ----------
    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        return seed

    def _reset(self, tensordict=None, **kwargs):
        if self.sumo.isLoaded():
            self.sumo.end()

        self.sumo.start(activeGui=False, logFilePath=self.sumo.logFile, rl_mode=True)
        self.current_step = 0
        self._last_green_tls_snapshot = {}
        self._phase_duration_memory = {}
        self._active_program_id = {}
        for tls in self.tls_list:
            self._sync_phase_duration_memory(tls)
        self.last_reward_components = {}
        self._episode_max_jam_len = 0.0
        self._last_taz_metrics_by_id = {taz: self._zero_taz_metrics() for taz in self.taz_ids}

        obs = self._collect_observation()
        return TensorDict({"observation": obs}, batch_size=[])

    def _step(self, tensordict):
        action = tensordict["action"]
        if action.requires_grad:
            action = action.detach()
        action = action.to(self.device)

        low = float(torch.as_tensor(self.action_spec.low).min().item())
        high = float(torch.as_tensor(self.action_spec.high).max().item())
        action_clipped = torch.clamp(action, min=low, max=high)
        action_discrete = self._discretize_actions(action_clipped)
        action_np = action_discrete.cpu().numpy()

        applied_duration_delta = []
        for tls, a in zip(self.tls_list, action_np):
            applied_duration_delta.append(self._apply_action_to_tls(tls, a))

        # synchronized global simulation step
        self.sumo.step(quantity=self.stepSize)
        self.current_step += 1
        sim_running = self.sumo.isRunning()
        time_limit_reached = self.current_step >= self.cooldownSteps

        obs = self._collect_observation()
        terminated = (not sim_running) and (not time_limit_reached)
        truncated = bool(time_limit_reached)

        reward = 0.0
        if terminated or truncated:
            if self.sumo.isLoaded():
                self.sumo.end()
            reward, reward_components = self._compute_terminal_reward()
            reward_components.update(
                {
                    "is_terminal_reward": True,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "episode_steps": int(self.current_step),
                    "num_taz": int(len(self.taz_ids)),
                    "num_tls": int(len(self.tls_list)),
                }
            )
            self.last_reward_components = reward_components
        else:
            self.last_reward_components = {
                "is_terminal_reward": False,
                "total": 0.0,
                "episode_steps": int(self.current_step),
                "num_taz": int(len(self.taz_ids)),
                "num_tls": int(len(self.tls_list)),
            }

        return TensorDict(
            {
                "observation": obs,
                "reward": torch.tensor(reward, dtype=torch.float32, device=self.device),
                "applied_action": action_discrete,
                "applied_duration_delta": torch.tensor(applied_duration_delta, dtype=torch.float32, device=self.device),
                "terminated": torch.tensor(terminated, device=self.device),
                "truncated": torch.tensor(truncated, device=self.device),
            },
            batch_size=[],
        )
