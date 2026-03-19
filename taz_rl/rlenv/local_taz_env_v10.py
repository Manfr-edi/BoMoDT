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


class SumoTazEnvV10(EnvBase):
    """
    v10 environment:
      - one agent per TAZ
      - one observation row per TAZ
      - one discrete action per TLS slot inside the TAZ
      - dense reward from absolute TAZ penalty plus local penalty improvement
      - terminal bonus aligned with a no-agent baseline on the same route/hour
    """

    ACTION_BIN_VALUES = (-10.0, -5.0, 0.0, 5.0, 10.0)
    ZERO_ACTION_INDEX = 2

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
        min_green: int = 25,
        max_green: int = 120,
        metric_clip: float = 2.0,
        reward_clip: float = 4.0,
        dense_abs_penalty_weight: float = 1.0,
        dense_delta_reward_weight: float = 0.40,
        terminal_bonus_weight: float = 1.0,
        comparison_reward_enabled: bool = True,
        waiting_reward_weight: float = 0.70,
        emission_reward_weight: float = 0.20,
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
        terminal_waiting_time_ref: float = 500000.0,
        terminal_co2_ref: float = 5000000000.0,
        terminal_nox_ref: float = 1800000.0,
        tls_veh_total_ref: float = 120.0,
        tls_pressure_ref: float = 10000.0,
        taz_veh_total_ref: float = 2500.0,
        taz_std_occupancy_ref: float = 0.25,
        max_jam_len_ref: float = 120.0,
        tls_add_path: str = SUMO_NETWORK_PATH + "/optimized_tls.add.xml",
        **kwargs,
    ):
        super().__init__(device=device)

        self.sumo = sumoSimulator
        self.stepSize = int(stepSize)
        self.speed_norm = max(float(speed_norm), 1e-6)
        self.jam_norm = max(float(jam_norm), 1e-6)
        self.warmupSteps = int(warmupSteps)
        self.cooldownSteps = int(cooldownSteps)
        self.min_green = int(min_green)
        self.max_green = int(max_green)
        self.metric_clip = max(float(metric_clip), 1.0)
        self.reward_clip = max(float(reward_clip), 1e-6)
        self.dense_abs_penalty_weight = float(dense_abs_penalty_weight)
        self.dense_delta_reward_weight = float(dense_delta_reward_weight)
        self.terminal_bonus_weight = float(terminal_bonus_weight)
        self.comparison_reward_enabled = bool(comparison_reward_enabled)
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
        self.terminal_waiting_time_ref = max(float(terminal_waiting_time_ref), 1e-6)
        self.terminal_co2_ref = max(float(terminal_co2_ref), 1e-6)
        self.terminal_nox_ref = max(float(terminal_nox_ref), 1e-6)
        self.tls_veh_total_ref = max(float(tls_veh_total_ref), 1e-6)
        self.tls_pressure_ref = max(float(tls_pressure_ref), 1e-6)
        self.taz_veh_total_ref = max(float(taz_veh_total_ref), 1e-6)
        self.taz_std_occupancy_ref = max(float(taz_std_occupancy_ref), 1e-6)
        self.max_jam_len_ref = max(float(max_jam_len_ref), 1e-6)
        self.tls_add_path = str(tls_add_path)
        self.waiting_ref = 60.0

        reward_weight_sum = (
            self.waiting_reward_weight
            + self.emission_reward_weight
            + self.jam_reward_weight
        )
        if reward_weight_sum <= 0.0:
            self.waiting_reward_weight = 0.70
            self.emission_reward_weight = 0.20
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
            low=0,
            high=len(self.ACTION_BIN_VALUES) - 1,
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
        self.action_bins = torch.tensor(self.ACTION_BIN_VALUES, dtype=torch.float32, device=self.device)

        self.current_hour = 0
        self.current_step = 0
        self.output_subdir = "output"
        self.tripinfo_filename = "tripinfos.xml"
        self.emission_filename = "emission-output.xml"

        self.reference_penalty = None
        self.reference_reward_components = {}
        self._prev_taz_penalties = np.zeros(len(self.taz_ids), dtype=np.float32)
        self._tls_lanes = {tls: set() for tls in self.tls_list}
        self._lane_to_tls_idx = {}
        self._edge_to_tls_idx = {}
        self._last_tls_metrics_by_id = {
            tls: self._zero_tls_metrics(taz_id=self.tls_to_taz.get(tls))
            for tls in self.tls_list
        }
        self._last_taz_metrics_by_id = {
            taz: self._zero_taz_metrics()
            for taz in self.taz_ids
        }
        self._last_green_tls_snapshot = {}
        self._phase_duration_memory = {}
        self._active_program_id = {}
        self._episode_phase_initial_durations = {}
        self._episode_phase_cumulative_abs_deltas = {}
        self._episode_phase_signed_deltas = {}
        self._episode_phase_nonzero_updates = {}
        self._episode_phase_sign_flips = {}
        self._episode_phase_last_nonzero_sign = {}
        self._episode_duration_events = {}
        self._episode_max_jam_len = 0.0

        self.last_reward_components = {}
        self.last_duration_diagnostics = {}

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
        with open(taz_map_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise RuntimeError("Invalid TAZ map content: expected dict.")
        return {
            str(k): [str(x) for x in v]
            for k, v in raw.items()
            if isinstance(v, list)
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

    def get_action_mask(self) -> torch.Tensor:
        return self._action_mask.clone()

    def decode_action_indices(self, action_index: torch.Tensor) -> torch.Tensor:
        return self.action_bins[action_index.long().clamp(0, len(self.ACTION_BIN_VALUES) - 1)]

    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except Exception:
            return default
        if not np.isfinite(parsed):
            return default
        return parsed

    @staticmethod
    def _get_attr_case_insensitive(elem: ET.Element, key: str):
        if key in elem.attrib:
            return elem.attrib[key]
        target = key.lower()
        for existing_key, existing_value in elem.attrib.items():
            if existing_key.lower() == target:
                return existing_value
        return None

    def _safe_parse_xml_root(self, xml_path: str):
        try:
            return ET.parse(xml_path).getroot()
        except Exception:
            pass

        try:
            with open(xml_path, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
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
        for match in re.finditer(r"<([A-Za-z_][\w\-\.:]*)[^>]*>", text):
            tag = match.group(1)
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
    def _lane_to_edge_id(lane_id: str) -> str:
        lane_id = str(lane_id or "")
        if not lane_id:
            return ""
        if lane_id.startswith(":"):
            return lane_id
        if "_" in lane_id:
            return lane_id.rsplit("_", 1)[0]
        return lane_id

    @staticmethod
    def _is_green_phase_state(phase_state: str) -> bool:
        return ("g" in phase_state) or ("G" in phase_state)

    @staticmethod
    def _has_yellow_phase_state(phase_state: str) -> bool:
        return ("y" in phase_state) or ("Y" in phase_state)

    @classmethod
    def _is_editable_phase_state(cls, phase_state: str) -> bool:
        return cls._is_green_phase_state(phase_state) and not cls._has_yellow_phase_state(phase_state)

    @staticmethod
    def _normalize_phase_duration_value(duration: float, fallback: float = 1.0) -> int:
        try:
            value = float(duration)
        except Exception:
            value = float(fallback)
        if not np.isfinite(value):
            value = float(fallback)
        return max(int(round(value)), 1)

    def _get_time_features(self):
        angle = 2.0 * np.pi * (self.current_hour % 24) / 24.0
        return float(np.sin(angle)), float(np.cos(angle))

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
            for tls in self.tls_list:
                tls_num_phases[tls] = 1
            max_phases = 1
        else:
            for tls in self.tls_list:
                tls_num_phases.setdefault(tls, 1)
        return tls_num_phases, max_phases

    def _find_previous_editable_phase_id(self, program, current_phase_id: int) -> Optional[int]:
        phases = list(getattr(program, "phases", []))
        n_phases = len(phases)
        if n_phases == 0:
            return None

        idx = (int(current_phase_id) - 1) % n_phases
        for _ in range(n_phases):
            if self._is_editable_phase_state(getattr(phases[idx], "state", "")):
                return idx
            idx = (idx - 1) % n_phases
        return None

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

    def _reset_duration_diagnostics(self):
        self._episode_phase_initial_durations = {}
        self._episode_phase_cumulative_abs_deltas = {}
        self._episode_phase_signed_deltas = {}
        self._episode_phase_nonzero_updates = {}
        self._episode_phase_sign_flips = {}
        self._episode_phase_last_nonzero_sign = {}
        self._episode_duration_events = {}

        for tls_id, durations in self._phase_duration_memory.items():
            phase_count = len(durations)
            self._episode_phase_initial_durations[tls_id] = [float(x) for x in durations]
            self._episode_phase_cumulative_abs_deltas[tls_id] = [0.0 for _ in range(phase_count)]
            self._episode_phase_signed_deltas[tls_id] = [0.0 for _ in range(phase_count)]
            self._episode_phase_nonzero_updates[tls_id] = [0 for _ in range(phase_count)]
            self._episode_phase_sign_flips[tls_id] = [0 for _ in range(phase_count)]
            self._episode_phase_last_nonzero_sign[tls_id] = [0 for _ in range(phase_count)]
            self._episode_duration_events[tls_id] = []

    def _record_duration_event(
        self,
        tls_id: str,
        target_phase_id: int,
        requested_delta: float,
        base_duration: float,
        applied_duration: float,
    ):
        initial = self._episode_phase_initial_durations.get(tls_id)
        cumulative = self._episode_phase_cumulative_abs_deltas.get(tls_id)
        signed = self._episode_phase_signed_deltas.get(tls_id)
        nonzero = self._episode_phase_nonzero_updates.get(tls_id)
        flips = self._episode_phase_sign_flips.get(tls_id)
        last_signs = self._episode_phase_last_nonzero_sign.get(tls_id)
        if (
            initial is None
            or cumulative is None
            or signed is None
            or nonzero is None
            or flips is None
            or last_signs is None
            or target_phase_id < 0
            or target_phase_id >= len(initial)
        ):
            return

        applied_delta = float(applied_duration - base_duration)
        self._episode_duration_events.setdefault(tls_id, []).append(
            {
                "control_step": int(self.current_step),
                "phase_id": int(target_phase_id),
                "requested_delta": float(requested_delta),
                "applied_delta": float(applied_delta),
                "base_duration": float(base_duration),
                "applied_duration": float(applied_duration),
            }
        )

        if abs(applied_delta) <= 1e-9:
            return

        cumulative[target_phase_id] += abs(applied_delta)
        signed[target_phase_id] += applied_delta
        nonzero[target_phase_id] += 1

        sign = 1 if applied_delta > 0.0 else -1
        prev_sign = int(last_signs[target_phase_id])
        if prev_sign != 0 and prev_sign != sign:
            flips[target_phase_id] += 1
        last_signs[target_phase_id] = sign

    def get_episode_duration_diagnostics(self) -> dict:
        per_tls = {}
        summary_net_abs = []
        summary_cumulative_abs = []
        summary_ratio = []
        summary_sign_flips = []
        summary_nonzero_updates = []
        reverted_count = 0
        changed_count = 0

        for tls_id in self.tls_list:
            initial = [float(x) for x in self._episode_phase_initial_durations.get(tls_id, [])]
            final = [float(x) for x in self._phase_duration_memory.get(tls_id, initial)]
            phase_count = min(len(initial), len(final))
            if len(initial) != phase_count:
                initial = initial[:phase_count]
            if len(final) != phase_count:
                final = final[:phase_count]

            net_deltas = [float(f - i) for i, f in zip(initial, final)]
            cumulative_abs = [
                float(x)
                for x in self._episode_phase_cumulative_abs_deltas.get(tls_id, [0.0 for _ in range(phase_count)])
            ][:phase_count]
            signed_deltas = [
                float(x)
                for x in self._episode_phase_signed_deltas.get(tls_id, [0.0 for _ in range(phase_count)])
            ][:phase_count]
            nonzero_updates = [
                int(x)
                for x in self._episode_phase_nonzero_updates.get(tls_id, [0 for _ in range(phase_count)])
            ][:phase_count]
            sign_flips = [
                int(x)
                for x in self._episode_phase_sign_flips.get(tls_id, [0 for _ in range(phase_count)])
            ][:phase_count]

            total_net_abs = float(sum(abs(x) for x in net_deltas))
            total_cumulative_abs = float(sum(cumulative_abs))
            net_to_cumulative_ratio = (
                float(total_net_abs / total_cumulative_abs)
                if total_cumulative_abs > 1e-9 else 0.0
            )
            total_sign_flips = int(sum(sign_flips))
            total_nonzero_updates = int(sum(nonzero_updates))
            reverted_to_initial = bool(total_net_abs <= 1e-9)
            changed = bool(total_cumulative_abs > 1e-9)

            if reverted_to_initial:
                reverted_count += 1
            if changed:
                changed_count += 1

            summary_net_abs.append(total_net_abs)
            summary_cumulative_abs.append(total_cumulative_abs)
            summary_ratio.append(net_to_cumulative_ratio)
            summary_sign_flips.append(float(total_sign_flips))
            summary_nonzero_updates.append(float(total_nonzero_updates))

            per_tls[tls_id] = {
                "initial_phase_durations": initial,
                "final_phase_durations": final,
                "phase_net_deltas": net_deltas,
                "phase_signed_applied_deltas": signed_deltas,
                "phase_cumulative_abs_deltas": cumulative_abs,
                "phase_nonzero_updates": nonzero_updates,
                "phase_sign_flips": sign_flips,
                "total_net_abs_delta": total_net_abs,
                "total_cumulative_abs_delta": total_cumulative_abs,
                "net_to_cumulative_ratio": float(net_to_cumulative_ratio),
                "total_sign_flips": total_sign_flips,
                "total_nonzero_updates": total_nonzero_updates,
                "reverted_to_initial": reverted_to_initial,
                "changed": changed,
                "events": list(self._episode_duration_events.get(tls_id, [])),
            }

        tls_count = max(len(self.tls_list), 1)
        return {
            "summary": {
                "tls_count": int(len(self.tls_list)),
                "mean_total_net_abs_delta": float(sum(summary_net_abs) / tls_count) if summary_net_abs else 0.0,
                "mean_total_cumulative_abs_delta": float(sum(summary_cumulative_abs) / tls_count) if summary_cumulative_abs else 0.0,
                "mean_net_to_cumulative_ratio": float(sum(summary_ratio) / tls_count) if summary_ratio else 0.0,
                "mean_total_sign_flips": float(sum(summary_sign_flips) / tls_count) if summary_sign_flips else 0.0,
                "mean_total_nonzero_updates": float(sum(summary_nonzero_updates) / tls_count) if summary_nonzero_updates else 0.0,
                "reverted_tls_ratio": float(reverted_count / tls_count),
                "changed_tls_ratio": float(changed_count / tls_count),
            },
            "per_tls": per_tls,
        }

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
        if not self._is_editable_phase_state(current_phase.state):
            prev_editable_phase_id = self._find_previous_editable_phase_id(program, current_phase_id)
            if prev_editable_phase_id is None:
                return 0.0
            target_phase_id = int(prev_editable_phase_id)

        base = self._safe_float(
            phase_memory[target_phase_id],
            self._safe_float(program.phases[target_phase_id].duration, 0.0),
        )
        new_dur = int(np.clip(base + float(action_value), self.min_green, self.max_green))
        self._phase_duration_memory[tls_id][target_phase_id] = float(new_dur)

        applied_duration = float(new_dur)
        try:
            for idx, phase in enumerate(list(program.phases)):
                stored_duration = self._safe_float(
                    self._phase_duration_memory[tls_id][idx],
                    self._safe_float(getattr(phase, "duration", 0.0), 0.0),
                )
                if idx == target_phase_id and self._is_editable_phase_state(getattr(phase, "state", "")):
                    duration_value = self._normalize_phase_duration_value(new_dur, fallback=stored_duration)
                else:
                    duration_value = self._normalize_phase_duration_value(
                        stored_duration,
                        fallback=self._safe_float(getattr(phase, "duration", 0.0), 1.0),
                    )
                phase.maxDur = duration_value
                phase.minDur = duration_value
                phase.duration = duration_value
            try:
                program.currentPhaseIndex = current_phase_id
            except Exception:
                pass
            libtraci.trafficlight.setProgramLogic(tls_id, program)

            if target_phase_id == current_phase_id and self._is_editable_phase_state(current_phase.state):
                try:
                    libtraci.trafficlight.setPhaseDuration(tls_id, float(new_dur))
                except Exception:
                    pass

            after_program, _ = self._get_active_program_logic(tls_id)
            if after_program is not None:
                self._sync_phase_duration_memory(tls_id)
                synced_memory = self._phase_duration_memory.get(tls_id, [])
                applied_duration = self._safe_float(
                    synced_memory[target_phase_id] if target_phase_id < len(synced_memory) else float(new_dur),
                    self._safe_float(after_program.phases[target_phase_id].duration, float(new_dur)),
                )
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
        self._record_duration_event(
            tls_id=tls_id,
            target_phase_id=target_phase_id,
            requested_delta=float(action_value),
            base_duration=float(base),
            applied_duration=float(applied_duration),
        )
        return float(applied_duration - base)

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

        for vehicle_id in vehicle_ids:
            try:
                lane_id = str(libtraci.vehicle.getLaneID(vehicle_id))
            except Exception:
                lane_id = ""
            tls_idx = self._lane_to_tls_idx.get(lane_id)

            if tls_idx is None:
                edge_id = self._lane_to_edge_id(lane_id)
                tls_idx = self._edge_to_tls_idx.get(edge_id)
            if tls_idx is None:
                try:
                    road_id = str(libtraci.vehicle.getRoadID(vehicle_id))
                except Exception:
                    road_id = ""
                tls_idx = self._edge_to_tls_idx.get(road_id)
            if tls_idx is None:
                continue

            try:
                waiting_time = self._safe_float(libtraci.vehicle.getWaitingTime(vehicle_id), 0.0)
                co2 = self._safe_float(libtraci.vehicle.getCO2Emission(vehicle_id), 0.0)
                nox = self._safe_float(libtraci.vehicle.getNOxEmission(vehicle_id), 0.0)
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

    def _collect_tls_runtime_metrics(self, tls_id: str, aggregated: dict, vehicle_metrics: dict) -> dict:
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
        metrics_for_obs = {
            "veh_total": self._safe_float(aggregated.get("veh_total", 0.0), 0.0),
            "mean_speed": self._safe_float(aggregated.get("mean_speed", 0.0), 0.0),
            "mean_occupancy": self._safe_float(aggregated.get("mean_occupancy", 0.0), 0.0),
            "max_jam": self._safe_float(aggregated.get("max_jam", 0.0), 0.0),
            "pressure": self._safe_float(aggregated.get("pressure", 0.0), 0.0),
            "active_vehicle_count": self._safe_float(vehicle_metrics.get("active_vehicle_count", 0.0), 0.0),
            "total_waiting_time": self._safe_float(vehicle_metrics.get("total_waiting_time", 0.0), 0.0),
            "avg_waiting_time": self._safe_float(vehicle_metrics.get("avg_waiting_time", 0.0), 0.0),
            "total_co2": self._safe_float(vehicle_metrics.get("total_co2", 0.0), 0.0),
            "total_nox": self._safe_float(vehicle_metrics.get("total_nox", 0.0), 0.0),
        }

        if self._is_editable_phase_state(current_phase_state):
            self._last_green_tls_snapshot[tls_id] = {
                "phase_id": int(phase_id_for_obs),
                "phase_duration": float(phase_duration_for_obs),
                **metrics_for_obs,
            }
        else:
            if program is not None:
                prev_editable_phase_id = self._find_previous_editable_phase_id(program, current_phase_id)
                if prev_editable_phase_id is not None:
                    phase_id_for_obs = int(prev_editable_phase_id)
                    phase_duration_for_obs = self._safe_float(
                        program.phases[prev_editable_phase_id].duration,
                        phase_duration_for_obs,
                    )

            cached_metrics = self._last_green_tls_snapshot.get(tls_id)
            if cached_metrics is not None:
                metrics_for_obs = {
                    "veh_total": self._safe_float(cached_metrics.get("veh_total", metrics_for_obs["veh_total"]), 0.0),
                    "mean_speed": self._safe_float(cached_metrics.get("mean_speed", metrics_for_obs["mean_speed"]), 0.0),
                    "mean_occupancy": self._safe_float(cached_metrics.get("mean_occupancy", metrics_for_obs["mean_occupancy"]), 0.0),
                    "max_jam": self._safe_float(cached_metrics.get("max_jam", metrics_for_obs["max_jam"]), 0.0),
                    "pressure": self._safe_float(cached_metrics.get("pressure", metrics_for_obs["pressure"]), 0.0),
                    "active_vehicle_count": self._safe_float(cached_metrics.get("active_vehicle_count", metrics_for_obs["active_vehicle_count"]), 0.0),
                    "total_waiting_time": self._safe_float(cached_metrics.get("total_waiting_time", metrics_for_obs["total_waiting_time"]), 0.0),
                    "avg_waiting_time": self._safe_float(cached_metrics.get("avg_waiting_time", metrics_for_obs["avg_waiting_time"]), 0.0),
                    "total_co2": self._safe_float(cached_metrics.get("total_co2", metrics_for_obs["total_co2"]), 0.0),
                    "total_nox": self._safe_float(cached_metrics.get("total_nox", metrics_for_obs["total_nox"]), 0.0),
                }
                phase_duration_for_obs = self._safe_float(
                    cached_metrics.get("phase_duration", phase_duration_for_obs),
                    phase_duration_for_obs,
                )

        return {
            "taz_id": self.tls_to_taz.get(tls_id),
            "tls_id": tls_id,
            "phase_id": int(phase_id_for_obs),
            "phase_duration": float(phase_duration_for_obs),
            **metrics_for_obs,
        }

    def _build_taz_metrics_from_tls(self) -> dict:
        metrics_by_taz = {
            taz: self._zero_taz_metrics()
            for taz in self.taz_ids
        }

        for taz in self.taz_ids:
            tls_metrics = [
                self._last_tls_metrics_by_id.get(
                    tls,
                    self._zero_tls_metrics(taz_id=taz),
                )
                for tls in self.tls_by_taz[taz]
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

    def _build_taz_penalties(self):
        penalties = []
        details = {}
        for taz in self.taz_ids:
            metrics = self._last_taz_metrics_by_id.get(taz, self._zero_taz_metrics())
            max_jam_len = self._safe_float(metrics.get("max_jam_len", 0.0), 0.0)
            total_waiting_time = self._safe_float(metrics.get("total_waiting_time", 0.0), 0.0)
            avg_waiting_time = self._safe_float(metrics.get("avg_waiting_time", 0.0), 0.0)
            total_co2 = self._safe_float(metrics.get("total_co2", 0.0), 0.0)
            total_nox = self._safe_float(metrics.get("total_nox", 0.0), 0.0)

            waiting_penalty = float(np.clip(total_waiting_time / self.taz_waiting_time_ref, 0.0, self.metric_clip))
            co2_penalty = float(np.clip(total_co2 / self.taz_co2_ref, 0.0, self.metric_clip))
            nox_penalty = float(np.clip(total_nox / self.taz_nox_ref, 0.0, self.metric_clip))
            emission_penalty = (
                self.emission_co2_mix_weight * co2_penalty
                + self.emission_nox_mix_weight * nox_penalty
            )
            jam_penalty = float(np.clip(max_jam_len / self.max_jam_len_ref, 0.0, self.metric_clip))

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
                "active_vehicle_count": float(self._safe_float(metrics.get("active_vehicle_count", 0.0), 0.0)),
                "avg_waiting_time": float(avg_waiting_time),
                "total_waiting_time": float(total_waiting_time),
                "total_co2": float(total_co2),
                "total_nox": float(total_nox),
                "max_jam_len": float(max_jam_len),
            }
        return np.asarray(penalties, dtype=np.float32), details

    def _compute_dense_reward_vector(self, penalties: np.ndarray) -> np.ndarray:
        delta = self._prev_taz_penalties - penalties
        reward_vector = (
            -self.dense_abs_penalty_weight * penalties
            + self.dense_delta_reward_weight * delta
        )
        return np.clip(reward_vector, -self.reward_clip, self.reward_clip).astype(np.float32)

    def _collect_observation(self) -> torch.Tensor:
        if not self.sumo.isRunning():
            self._last_tls_metrics_by_id = {
                tls: self._zero_tls_metrics(taz_id=self.tls_to_taz.get(tls))
                for tls in self.tls_list
            }
            self._last_taz_metrics_by_id = {
                taz: self._zero_taz_metrics()
                for taz in self.taz_ids
            }
            return torch.zeros(self.observation_spec.shape, dtype=torch.float32, device=self.device)

        raw_by_taz = {}
        for taz in self.taz_ids:
            try:
                raw_by_taz[taz] = self.sumo.get_taz_e2_metrics(taz, interval="last", mode="dict")
            except Exception:
                raw_by_taz[taz] = {"tls_data": {}, "taz_avg": {}}

        vehicle_metrics_by_tls = self._collect_vehicle_metrics_by_tls()
        tls_metrics_by_id = {}
        for taz in self.taz_ids:
            tls_data = raw_by_taz[taz].get("tls_data", {})
            for tls in self.tls_by_taz[taz]:
                aggregated = aggregate_tls_metrics(tls_data.get(tls, []))
                tls_metrics_by_id[tls] = self._collect_tls_runtime_metrics(
                    tls_id=tls,
                    aggregated=aggregated,
                    vehicle_metrics=vehicle_metrics_by_tls.get(tls, {}),
                )

        for tls in self.tls_list:
            tls_metrics_by_id.setdefault(
                tls,
                self._zero_tls_metrics(taz_id=self.tls_to_taz.get(tls)),
            )
        self._last_tls_metrics_by_id = tls_metrics_by_id
        self._last_taz_metrics_by_id = self._build_taz_metrics_from_tls()

        obs_rows = []
        hour_sin, hour_cos = self._get_time_features()
        for taz in self.taz_ids:
            row = []
            tls_ids = self.tls_by_taz[taz]
            for slot_idx in range(self.max_tls_per_taz):
                if slot_idx < len(tls_ids):
                    tls = tls_ids[slot_idx]
                    metrics = tls_metrics_by_id[tls]
                    phase_id = int(metrics.get("phase_id", 0))
                    n_phases = int(self.tls_num_phases.get(tls, self.max_num_phases))
                    one_hot = np.zeros(self.max_num_phases, dtype=np.float32)
                    if 0 <= phase_id < n_phases:
                        one_hot[phase_id] = 1.0

                    mean_occ = self._safe_float(metrics.get("mean_occupancy", 0.0), 0.0)
                    occ_norm = mean_occ / 100.0 if mean_occ > 1.5 else mean_occ
                    occ_norm = self._clip01(occ_norm)

                    row.extend(
                        [1.0]
                        + one_hot.tolist()
                        + [
                            float(np.clip(self._safe_float(metrics.get("phase_duration", 0.0), 0.0) / float(self.max_green), 0.0, 2.0)),
                            float(np.clip(self._safe_float(metrics.get("mean_speed", 0.0), 0.0) / self.speed_norm, 0.0, 2.0)),
                            float(np.clip(self._safe_float(metrics.get("max_jam", 0.0), 0.0) / self.jam_norm, 0.0, 2.0)),
                            float(occ_norm),
                            float(np.clip(self._safe_float(metrics.get("veh_total", 0.0), 0.0) / self.tls_veh_total_ref, 0.0, 5.0)),
                            float(np.clip(max(self._safe_float(metrics.get("pressure", 0.0), 0.0), 0.0) / self.tls_pressure_ref, 0.0, 5.0)),
                            float(np.clip(self._safe_float(metrics.get("active_vehicle_count", 0.0), 0.0) / self.tls_active_vehicle_ref, 0.0, 5.0)),
                            float(np.clip(self._safe_float(metrics.get("total_waiting_time", 0.0), 0.0) / self.tls_waiting_time_ref, 0.0, 5.0)),
                            float(np.clip(self._safe_float(metrics.get("avg_waiting_time", 0.0), 0.0) / self.waiting_ref, 0.0, 5.0)),
                            float(np.clip(self._safe_float(metrics.get("total_co2", 0.0), 0.0) / self.tls_co2_ref, 0.0, 5.0)),
                            float(np.clip(self._safe_float(metrics.get("total_nox", 0.0), 0.0) / self.tls_nox_ref, 0.0, 5.0)),
                        ]
                    )
                else:
                    row.extend([0.0] * self.per_tls_feature_dim)

            taz_metrics = self._last_taz_metrics_by_id[taz]
            t_occ_norm = taz_metrics["mean_occupancy"] / 100.0 if taz_metrics["mean_occupancy"] > 1.5 else taz_metrics["mean_occupancy"]
            t_occ_norm = self._clip01(t_occ_norm)
            row.extend(
                [
                    float(np.clip(self._safe_float(taz_metrics.get("max_jam_len", 0.0), 0.0) / self.jam_norm, 0.0, 2.0)),
                    float(np.clip(self._safe_float(taz_metrics.get("mean_speed", 0.0), 0.0) / self.speed_norm, 0.0, 2.0)),
                    float(t_occ_norm),
                    float(np.clip(self._safe_float(taz_metrics.get("veh_total", 0.0), 0.0) / self.taz_veh_total_ref, 0.0, 5.0)),
                    float(np.clip(self._safe_float(taz_metrics.get("active_vehicle_count", 0.0), 0.0) / self.taz_veh_total_ref, 0.0, 5.0)),
                    float(np.clip(self._safe_float(taz_metrics.get("total_waiting_time", 0.0), 0.0) / self.taz_waiting_time_ref, 0.0, 5.0)),
                    float(np.clip(self._safe_float(taz_metrics.get("total_co2", 0.0), 0.0) / self.taz_co2_ref, 0.0, 5.0)),
                    float(np.clip(self._safe_float(taz_metrics.get("total_nox", 0.0), 0.0) / self.taz_nox_ref, 0.0, 5.0)),
                    hour_sin,
                    hour_cos,
                ]
            )
            self._episode_max_jam_len = max(
                self._episode_max_jam_len,
                self._safe_float(taz_metrics.get("max_jam_len", 0.0), 0.0),
            )
            obs_rows.append(row)

        obs_tensor = torch.tensor(obs_rows, dtype=torch.float32, device=self.device)
        if not torch.isfinite(obs_tensor).all():
            obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs_tensor

    def _get_output_dir(self) -> Optional[str]:
        root = getattr(self.sumo, "typePath", None)
        if root is None:
            root = getattr(self.sumo, "routeFilePath", None)
        if root is None:
            return None
        return os.path.join(str(root), self.output_subdir)

    def _parse_tripinfo_metrics(self, tripinfo_path: str) -> dict:
        if not os.path.exists(tripinfo_path):
            return {
                "trip_count": 0,
                "avg_waiting_time": 0.0,
                "total_waiting_time": 0.0,
                "ok": False,
            }

        root = self._safe_parse_xml_root(tripinfo_path)
        if root is None:
            return {
                "trip_count": 0,
                "avg_waiting_time": 0.0,
                "total_waiting_time": 0.0,
                "ok": False,
            }

        waiting_times = []
        for trip in root.findall("tripinfo"):
            waiting_times.append(self._safe_float(trip.get("waitingTime"), 0.0))

        if not waiting_times:
            return {
                "trip_count": 0,
                "avg_waiting_time": 0.0,
                "total_waiting_time": 0.0,
                "ok": False,
            }

        total_waiting_time = float(sum(waiting_times))
        return {
            "trip_count": int(len(waiting_times)),
            "avg_waiting_time": float(total_waiting_time / len(waiting_times)),
            "total_waiting_time": total_waiting_time,
            "ok": True,
        }

    def _parse_emission_metric(self, emission_path: str, metric: str) -> dict:
        if not os.path.exists(emission_path):
            return {"total_emission": 0.0, "ok": False}

        root = self._safe_parse_xml_root(emission_path)
        if root is None:
            return {"total_emission": 0.0, "ok": False}

        totals = []
        for timestep in root.findall("timestep"):
            vehicles = timestep.findall("vehicle")
            if vehicles:
                totals.append(
                    sum(
                        self._safe_float(self._get_attr_case_insensitive(vehicle, metric), 0.0)
                        for vehicle in vehicles
                    )
                )
                continue

            total_attr = self._get_attr_case_insensitive(timestep, metric)
            totals.append(self._safe_float(total_attr, 0.0))

        if not totals:
            return {"total_emission": 0.0, "ok": False}

        return {
            "total_emission": float(sum(totals)),
            "ok": True,
        }

    def _normalize_metric(self, value: float, ref: float) -> float:
        return float(np.clip(float(value) / float(ref), 0.0, self.metric_clip))

    def _compute_terminal_reward(self) -> tuple[float, dict]:
        output_dir = self._get_output_dir()
        if output_dir is None:
            return 0.0, {
                "parse_ok": False,
                "reason": "missing_output_dir",
            }

        tripinfo_path = os.path.join(output_dir, self.tripinfo_filename)
        emission_path = os.path.join(output_dir, self.emission_filename)
        trip = self._parse_tripinfo_metrics(tripinfo_path)
        co2 = self._parse_emission_metric(emission_path, "CO2")
        nox = self._parse_emission_metric(emission_path, "NOx")
        parse_ok = bool(trip["ok"] and co2["ok"] and nox["ok"])
        if not parse_ok:
            return 0.0, {
                "parse_ok": False,
                "reason": "missing_or_invalid_tripinfo_or_emission",
                "tripinfo_path": tripinfo_path,
                "emission_path": emission_path,
            }

        waiting_penalty = self._normalize_metric(trip["total_waiting_time"], self.terminal_waiting_time_ref)
        co2_penalty = self._normalize_metric(co2["total_emission"], self.terminal_co2_ref)
        nox_penalty = self._normalize_metric(nox["total_emission"], self.terminal_nox_ref)
        emission_penalty = (
            self.emission_co2_mix_weight * co2_penalty
            + self.emission_nox_mix_weight * nox_penalty
        )
        jam_penalty = self._normalize_metric(self._episode_max_jam_len, self.max_jam_len_ref)
        penalty = (
            self.waiting_reward_weight * waiting_penalty
            + self.emission_reward_weight * emission_penalty
            + self.jam_reward_weight * jam_penalty
        )

        use_comparison = bool(self.comparison_reward_enabled and (self.reference_penalty is not None))
        baseline_penalty = float(self.reference_penalty) if use_comparison else 0.0
        reward_raw = baseline_penalty - penalty if use_comparison else -penalty
        reward = float(np.clip(self.terminal_bonus_weight * reward_raw, -self.reward_clip, self.reward_clip))

        components = {
            "parse_ok": True,
            "reward_basis": "baseline_waiting_emission_jam_delta" if use_comparison else "absolute_waiting_emission_jam_penalty",
            "trip_count": int(trip["trip_count"]),
            "avg_waiting_time": float(trip["avg_waiting_time"]),
            "total_waiting_time": float(trip["total_waiting_time"]),
            "total_co2": float(co2["total_emission"]),
            "total_nox": float(nox["total_emission"]),
            "episode_max_jam_len": float(self._episode_max_jam_len),
            "waiting_penalty": float(waiting_penalty),
            "co2_penalty": float(co2_penalty),
            "nox_penalty": float(nox_penalty),
            "emission_penalty": float(emission_penalty),
            "jam_penalty": float(jam_penalty),
            "penalty": float(penalty),
            "baseline_penalty": float(baseline_penalty),
            "delta_penalty": float(baseline_penalty - penalty) if use_comparison else 0.0,
            "reward_raw": float(reward_raw),
            "total": float(reward),
            "tripinfo_path": tripinfo_path,
            "emission_path": emission_path,
        }
        return reward, components

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
        self.last_reward_components = dict(baseline_components)
        self.last_duration_diagnostics = self.get_episode_duration_diagnostics()
        baseline_penalty = float(baseline_components.get("penalty", 0.0))
        return baseline_penalty, baseline_components

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
        self._build_tls_lane_cache()
        for tls in self.tls_list:
            self._sync_phase_duration_memory(tls)
        self._reset_duration_diagnostics()
        self.last_reward_components = {}
        self.last_duration_diagnostics = {}
        self._episode_max_jam_len = 0.0
        self._last_tls_metrics_by_id = {
            tls: self._zero_tls_metrics(taz_id=self.tls_to_taz.get(tls))
            for tls in self.tls_list
        }
        self._last_taz_metrics_by_id = {
            taz: self._zero_taz_metrics()
            for taz in self.taz_ids
        }

        obs = self._collect_observation()
        penalties, _ = self._build_taz_penalties()
        self._prev_taz_penalties = penalties.copy()
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
        action_index = torch.round(action).long().clamp(0, len(self.ACTION_BIN_VALUES) - 1)
        zero_index = torch.full_like(action_index, self.ZERO_ACTION_INDEX)
        action_index = torch.where(self._action_mask, action_index, zero_index)
        selected_action = self.decode_action_indices(action_index)

        applied_action = torch.zeros_like(selected_action)
        applied_duration_delta = torch.zeros_like(selected_action)
        for taz_idx, taz in enumerate(self.taz_ids):
            for slot_idx, tls in enumerate(self.tls_by_taz[taz]):
                action_value = float(selected_action[taz_idx, slot_idx].item())
                applied_action[taz_idx, slot_idx] = action_value
                applied_duration_delta[taz_idx, slot_idx] = float(self._apply_action_to_tls(tls, action_value))

        self.sumo.step(quantity=self.stepSize)
        self.current_step += 1
        sim_running = self.sumo.isRunning()
        time_limit_reached = self.current_step >= self.cooldownSteps

        obs = self._collect_observation()
        penalties, penalty_details = self._build_taz_penalties()

        if self.current_step <= self.warmupSteps:
            dense_reward_vector = np.zeros_like(penalties, dtype=np.float32)
        else:
            dense_reward_vector = self._compute_dense_reward_vector(penalties)

        terminated = (not sim_running) and (not time_limit_reached)
        truncated = bool(time_limit_reached)
        terminal_reward = 0.0
        terminal_components = {}
        reward_vector = dense_reward_vector.copy()
        if terminated or truncated:
            if self.sumo.isLoaded():
                self.sumo.end()
            terminal_reward, terminal_components = self._compute_terminal_reward()
            reward_vector = reward_vector + float(terminal_reward)
            self.last_duration_diagnostics = self.get_episode_duration_diagnostics()
        else:
            self.last_duration_diagnostics = {}

        penalty_by_taz = {
            taz: float(penalties[idx])
            for idx, taz in enumerate(self.taz_ids)
        }
        dense_reward_by_taz = {
            taz: float(dense_reward_vector[idx])
            for idx, taz in enumerate(self.taz_ids)
        }
        reward_by_taz = {
            taz: float(reward_vector[idx])
            for idx, taz in enumerate(self.taz_ids)
        }
        delta_penalty_by_taz = {
            taz: float(self._prev_taz_penalties[idx] - penalties[idx])
            for idx, taz in enumerate(self.taz_ids)
        }

        self.last_reward_components = {
            "reward_basis": "taz_absolute_penalty_plus_delta_with_terminal_baseline_bonus",
            "reward_by_taz": reward_by_taz,
            "dense_reward_by_taz": dense_reward_by_taz,
            "penalty_by_taz": penalty_by_taz,
            "penalty_details_by_taz": penalty_details,
            "delta_penalty_by_taz": delta_penalty_by_taz,
            "reward_mean": float(np.mean(reward_vector)),
            "reward_std": float(np.std(reward_vector)),
            "reward_min": float(np.min(reward_vector)),
            "reward_max": float(np.max(reward_vector)),
            "reward_sum": float(np.sum(reward_vector)),
            "dense_reward_mean": float(np.mean(dense_reward_vector)),
            "dense_reward_sum": float(np.sum(dense_reward_vector)),
            "penalty_mean": float(np.mean(penalties)),
            "penalty_std": float(np.std(penalties)),
            "penalty_min": float(np.min(penalties)),
            "penalty_max": float(np.max(penalties)),
            "penalty_sum": float(np.sum(penalties)),
            "terminal_bonus": float(terminal_reward),
            "terminal_reward": terminal_components,
            "reference_penalty": float(self.reference_penalty) if self.reference_penalty is not None else None,
            "reference_reward_components": dict(self.reference_reward_components),
            "episode_steps": int(self.current_step),
            "num_taz": int(len(self.taz_ids)),
            "num_tls": int(len(self.tls_list)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "warmup": bool(self.current_step <= self.warmupSteps),
        }
        self._prev_taz_penalties = penalties.copy()

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
            device=self.device,
        )
