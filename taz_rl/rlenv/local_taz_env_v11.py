import heapq
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict

from libraries.constants import SUMO_NET_FILE_PATH
from taz_rl.rlenv.local_taz_env_v10 import BoundedSpec, SumoTazEnvV10, UnboundedSpec


class SumoTazEnvV11(SumoTazEnvV10):
    """
    v11 environment:
      - keeps the v10 per-TLS observation backbone and reward design
      - replaces per-TLS actions with coordinated nearby-TLS groups
      - applies one shared delta per group using wider {-10, 0, +10} bins
      - within a group, applies the delta only to TLS currently aligned on the
        same dominant green axis to favor corridor-style progression
    """

    ACTION_BIN_VALUES = (-10.0, 0.0, 10.0)
    ZERO_ACTION_INDEX = 1

    def __init__(
        self,
        *args,
        coordination_distance: float = 350.0,
        coordination_path_length: float = 900.0,
        coordination_group_max_size: int = 3,
        group_axis_align_only: bool = True,
        **kwargs,
    ):
        self.coordination_distance = max(float(coordination_distance), 1.0)
        self.coordination_path_length = max(float(coordination_path_length), 1.0)
        self.coordination_group_max_size = max(int(coordination_group_max_size), 1)
        self.group_axis_align_only = bool(group_axis_align_only)

        super().__init__(*args, **kwargs)

        self.base_agent_obs_dim = int(self.agent_obs_dim)
        self.per_group_feature_dim = 9
        self._tls_nodes = self._build_tls_nodes_from_net(SUMO_NET_FILE_PATH)
        self._tls_positions = self._build_tls_positions_from_net(SUMO_NET_FILE_PATH)
        self._road_graph = self._build_road_graph_from_net(SUMO_NET_FILE_PATH)
        self._tls_topology_distance_by_pair = self._build_tls_topology_distances()
        self._tls_link_axis_by_index = self._build_tls_link_axis_by_index(SUMO_NET_FILE_PATH)
        self.control_groups_by_taz = self._build_control_groups_by_taz()
        self.max_control_groups_per_taz = max(len(groups) for groups in self.control_groups_by_taz.values())

        self.agent_obs_dim = (
            self.max_tls_per_taz * self.per_tls_feature_dim
            + self.per_taz_feature_dim
            + self.max_control_groups_per_taz * self.per_group_feature_dim
        )
        self.observation_spec = UnboundedSpec(
            shape=(len(self.taz_ids), self.agent_obs_dim),
            device=self.device,
        )
        self.action_spec = BoundedSpec(
            low=0,
            high=len(self.ACTION_BIN_VALUES) - 1,
            shape=(len(self.taz_ids), self.max_control_groups_per_taz),
            device=self.device,
        )

        action_mask = []
        for taz in self.taz_ids:
            valid = len(self.control_groups_by_taz[taz])
            row = [True] * valid + [False] * (self.max_control_groups_per_taz - valid)
            action_mask.append(row)
        self._group_action_mask = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
        self._current_action_mask = self._group_action_mask.clone()
        self.action_bins = torch.tensor(self.ACTION_BIN_VALUES, dtype=torch.float32, device=self.device)

        self._prev_action_tls_metrics = None
        self._last_tls_axis_by_id = {
            tls: "mixed"
            for tls in self.tls_list
        }
        self._group_signal_by_taz = {
            taz: [0.0 for _ in groups]
            for taz, groups in self.control_groups_by_taz.items()
        }
        self.last_group_action_details = {}

    @staticmethod
    def _axis_from_delta(dx: float, dy: float) -> str:
        return "EW" if abs(dx) >= abs(dy) else "NS"

    def _build_tls_nodes_from_net(self, net_path: str) -> dict[str, set[str]]:
        tree = ET.parse(net_path)
        root = tree.getroot()

        junction_ids = {
            junction.get("id")
            for junction in root.findall(".//junction")
            if junction.get("id")
        }
        edge_to_to_node = {}
        for edge in root.findall(".//edge"):
            edge_id = edge.get("id")
            to_node = edge.get("to")
            if edge_id and to_node:
                edge_to_to_node[edge_id] = to_node

        tls_nodes = defaultdict(set)
        for connection in root.findall(".//connection"):
            tls_id = connection.get("tl")
            edge_id = connection.get("from")
            if not tls_id:
                continue
            if tls_id in junction_ids:
                tls_nodes[tls_id].add(tls_id)
            if edge_id in edge_to_to_node:
                tls_nodes[tls_id].add(edge_to_to_node[edge_id])

        for tls_id in self.tls_list:
            if tls_id in junction_ids:
                tls_nodes[tls_id].add(tls_id)
            tls_nodes.setdefault(tls_id, set())

        return {
            tls_id: set(node_ids)
            for tls_id, node_ids in tls_nodes.items()
        }

    def _build_tls_positions_from_net(self, net_path: str) -> dict[str, tuple[float, float]]:
        tree = ET.parse(net_path)
        root = tree.getroot()

        junction_positions = {}
        for junction in root.findall(".//junction"):
            jid = junction.get("id")
            x = junction.get("x")
            y = junction.get("y")
            if jid and x and y:
                junction_positions[jid] = (float(x), float(y))

        edge_to_to_node = {}
        for edge in root.findall(".//edge"):
            edge_id = edge.get("id")
            to_node = edge.get("to")
            if edge_id and to_node:
                edge_to_to_node[edge_id] = to_node

        tls_nodes = defaultdict(list)
        for connection in root.findall(".//connection"):
            tls_id = connection.get("tl")
            edge_id = connection.get("from")
            if tls_id and edge_id in edge_to_to_node:
                tls_nodes[tls_id].append(edge_to_to_node[edge_id])

        tls_positions = {}
        for tls_id in self.tls_list:
            points = []
            if tls_id in junction_positions:
                points.append(junction_positions[tls_id])
            for node_id in tls_nodes.get(tls_id, []):
                if node_id in junction_positions:
                    points.append(junction_positions[node_id])
            if not points:
                tls_positions[tls_id] = (0.0, 0.0)
                continue
            tls_positions[tls_id] = (
                float(sum(x for x, _ in points) / len(points)),
                float(sum(y for _, y in points) / len(points)),
            )
        return tls_positions

    def _build_road_graph_from_net(self, net_path: str) -> dict[str, list[tuple[str, float]]]:
        tree = ET.parse(net_path)
        root = tree.getroot()

        junction_positions = {}
        for junction in root.findall(".//junction"):
            jid = junction.get("id")
            x = junction.get("x")
            y = junction.get("y")
            if jid and x and y:
                junction_positions[jid] = (float(x), float(y))

        graph = defaultdict(list)
        for edge in root.findall(".//edge"):
            edge_id = edge.get("id")
            from_node = edge.get("from")
            to_node = edge.get("to")
            function = edge.get("function", "")
            if (
                not edge_id
                or not from_node
                or not to_node
                or function == "internal"
                or edge_id.startswith(":")
            ):
                continue

            length = None
            try:
                length_attr = edge.get("length")
                if length_attr is not None:
                    length = float(length_attr)
            except (TypeError, ValueError):
                length = None

            if length is None:
                lane_lengths = []
                for lane in edge.findall("lane"):
                    lane_length = lane.get("length")
                    if lane_length is None:
                        continue
                    try:
                        lane_lengths.append(float(lane_length))
                    except (TypeError, ValueError):
                        continue
                if lane_lengths:
                    length = float(sum(lane_lengths) / len(lane_lengths))

            if length is None:
                p0 = junction_positions.get(from_node)
                p1 = junction_positions.get(to_node)
                if p0 is not None and p1 is not None:
                    length = self._distance(p0, p1)

            if length is None or not math.isfinite(length):
                length = 1.0
            length = max(float(length), 1.0)

            graph[from_node].append((to_node, length))
            graph[to_node].append((from_node, length))

        return {
            node_id: list(neighbors)
            for node_id, neighbors in graph.items()
        }

    def _build_tls_link_axis_by_index(self, net_path: str) -> dict[str, dict[int, str]]:
        tree = ET.parse(net_path)
        root = tree.getroot()

        junction_positions = {}
        for junction in root.findall(".//junction"):
            jid = junction.get("id")
            x = junction.get("x")
            y = junction.get("y")
            if jid and x and y:
                junction_positions[jid] = (float(x), float(y))

        edge_axis = {}
        for edge in root.findall(".//edge"):
            edge_id = edge.get("id")
            from_node = edge.get("from")
            to_node = edge.get("to")
            if (
                edge_id
                and from_node in junction_positions
                and to_node in junction_positions
            ):
                x0, y0 = junction_positions[from_node]
                x1, y1 = junction_positions[to_node]
                edge_axis[edge_id] = self._axis_from_delta(x1 - x0, y1 - y0)

        link_axis_by_index = defaultdict(dict)
        for connection in root.findall(".//connection"):
            tls_id = connection.get("tl")
            link_index = connection.get("linkIndex")
            if not tls_id or link_index is None:
                continue

            axis = edge_axis.get(connection.get("from", ""))
            if axis is None:
                axis = edge_axis.get(connection.get("to", ""))
            if axis is None:
                continue

            try:
                link_axis_by_index[tls_id][int(link_index)] = axis
            except (TypeError, ValueError):
                continue

        return {
            tls_id: dict(index_map)
            for tls_id, index_map in link_axis_by_index.items()
        }

    @staticmethod
    def _distance(p0: tuple[float, float], p1: tuple[float, float]) -> float:
        return float(math.hypot(p0[0] - p1[0], p0[1] - p1[1]))

    def _multi_source_shortest_paths(self, source_nodes: set[str]) -> dict[str, float]:
        dist = {}
        heap = []
        for node_id in source_nodes:
            if node_id not in self._road_graph:
                continue
            dist[node_id] = 0.0
            heapq.heappush(heap, (0.0, node_id))

        while heap:
            current_dist, node_id = heapq.heappop(heap)
            if current_dist > dist.get(node_id, float("inf")):
                continue
            if current_dist > self.coordination_path_length:
                continue

            for neighbor_id, edge_length in self._road_graph.get(node_id, []):
                next_dist = current_dist + float(edge_length)
                if next_dist >= dist.get(neighbor_id, float("inf")):
                    continue
                if next_dist > self.coordination_path_length:
                    continue
                dist[neighbor_id] = next_dist
                heapq.heappush(heap, (next_dist, neighbor_id))

        return dist

    def _build_tls_topology_distances(self) -> dict[tuple[str, str], float]:
        pair_distances = {}
        for taz in self.taz_ids:
            tls_ids = list(self.tls_by_taz[taz])
            tls_nodes = {
                tls_id: {
                    node_id
                    for node_id in self._tls_nodes.get(tls_id, set())
                    if node_id in self._road_graph
                }
                for tls_id in tls_ids
            }
            for tls_id in tls_ids:
                source_nodes = tls_nodes.get(tls_id, set())
                if not source_nodes:
                    continue
                node_distances = self._multi_source_shortest_paths(source_nodes)
                for other_tls_id in tls_ids:
                    if other_tls_id == tls_id:
                        pair_distances[(tls_id, other_tls_id)] = 0.0
                        continue

                    target_nodes = tls_nodes.get(other_tls_id, set())
                    if not target_nodes:
                        continue

                    best_distance = min(
                        (node_distances.get(node_id, float("inf")) for node_id in target_nodes),
                        default=float("inf"),
                    )
                    if math.isfinite(best_distance):
                        pair_distances[(tls_id, other_tls_id)] = float(best_distance)

        return pair_distances

    def _build_control_groups_by_taz(self) -> dict[str, list[list[str]]]:
        grouped = {}
        for taz in self.taz_ids:
            remaining = set(self.tls_by_taz[taz])
            groups = []
            while remaining:
                anchor = min(
                    remaining,
                    key=lambda tls: (
                        self._tls_positions.get(tls, (0.0, 0.0))[0],
                        self._tls_positions.get(tls, (0.0, 0.0))[1],
                        tls,
                    ),
                )
                remaining.remove(anchor)
                group = [anchor]

                while remaining and len(group) < self.coordination_group_max_size:
                    connected_candidates = []
                    nearby_candidates = []
                    for tls in remaining:
                        tls_pos = self._tls_positions.get(tls, (0.0, 0.0))
                        nearest_member_dist = min(
                            self._distance(tls_pos, self._tls_positions.get(member, (0.0, 0.0)))
                            for member in group
                        )
                        nearest_topology_dist = min(
                            self._tls_topology_distance_by_pair.get((tls, member), float("inf"))
                            for member in group
                        )

                        common_sort = (
                            self._tls_positions.get(tls, (0.0, 0.0))[0],
                            self._tls_positions.get(tls, (0.0, 0.0))[1],
                            tls,
                        )
                        if nearest_topology_dist <= self.coordination_path_length:
                            connected_candidates.append((nearest_topology_dist, nearest_member_dist, *common_sort))
                        elif nearest_member_dist <= self.coordination_distance:
                            nearby_candidates.append((nearest_member_dist, *common_sort))

                    if connected_candidates:
                        _, _, _, _, best_tls = min(connected_candidates)
                    elif nearby_candidates:
                        _, _, _, best_tls = min(nearby_candidates)
                    else:
                        break
                    remaining.remove(best_tls)
                    group.append(best_tls)

                group.sort(
                    key=lambda tls: (
                        self._tls_positions.get(tls, (0.0, 0.0))[0],
                        self._tls_positions.get(tls, (0.0, 0.0))[1],
                        tls,
                    )
                )
                groups.append(group)

            groups.sort(
                key=lambda group: (
                    self._tls_positions.get(group[0], (0.0, 0.0))[0],
                    self._tls_positions.get(group[0], (0.0, 0.0))[1],
                    group[0],
                )
            )
            grouped[taz] = groups
        return grouped

    def get_control_groups_by_taz(self) -> dict[str, list[list[str]]]:
        return {
            taz: [list(group) for group in groups]
            for taz, groups in self.control_groups_by_taz.items()
        }

    def _clone_tls_metrics_snapshot(self) -> dict[str, dict]:
        return {
            tls: {
                key: self._safe_float(value, 0.0)
                for key, value in metrics.items()
            }
            for tls, metrics in self._last_tls_metrics_by_id.items()
        }

    def _infer_tls_current_green_axis(self, tls_id: str) -> str:
        fallback_axis = self._last_tls_axis_by_id.get(tls_id, "mixed")
        link_axis = self._tls_link_axis_by_index.get(tls_id, {})
        if not link_axis:
            return fallback_axis

        try:
            program, _ = self._get_active_program_logic(tls_id)
        except Exception:
            return fallback_axis
        if program is None:
            return fallback_axis

        phases = list(program.phases)
        if not phases:
            return fallback_axis

        current_phase_id = int(getattr(program, "currentPhaseIndex", 0))
        if current_phase_id < 0 or current_phase_id >= len(phases):
            return fallback_axis

        state = str(getattr(phases[current_phase_id], "state", "") or "")
        ew_green = 0
        ns_green = 0
        for link_idx, phase_char in enumerate(state):
            if phase_char not in {"G", "g"}:
                continue
            axis = link_axis.get(link_idx)
            if axis == "EW":
                ew_green += 1
            elif axis == "NS":
                ns_green += 1

        if ew_green > ns_green:
            return "EW"
        if ns_green > ew_green:
            return "NS"
        return fallback_axis

    def _compute_group_signal(self, group_tls_ids: list[str]) -> float:
        if self._prev_action_tls_metrics is None:
            return 0.0

        signals = []
        for tls_id in group_tls_ids:
            current = self._last_tls_metrics_by_id.get(tls_id, self._zero_tls_metrics())
            previous = self._prev_action_tls_metrics.get(tls_id, self._zero_tls_metrics())

            waiting_delta = abs(
                self._safe_float(current.get("total_waiting_time", 0.0), 0.0)
                - self._safe_float(previous.get("total_waiting_time", 0.0), 0.0)
            ) / self.tls_waiting_time_ref
            active_delta = abs(
                self._safe_float(current.get("active_vehicle_count", 0.0), 0.0)
                - self._safe_float(previous.get("active_vehicle_count", 0.0), 0.0)
            ) / self.tls_active_vehicle_ref
            pressure_delta = abs(
                self._safe_float(current.get("pressure", 0.0), 0.0)
                - self._safe_float(previous.get("pressure", 0.0), 0.0)
            ) / self.tls_pressure_ref
            signals.append(waiting_delta + 0.5 * active_delta + 0.25 * pressure_delta)

        if not signals:
            return 0.0
        return float(np.clip(float(sum(signals) / len(signals)), 0.0, 5.0))

    def _build_runtime_action_mask(self) -> torch.Tensor:
        runtime_mask = self._group_action_mask.clone()
        self._group_signal_by_taz = {}
        self._action_signal_by_taz = {}

        for taz_idx, taz in enumerate(self.taz_ids):
            signals = []
            for group_idx, group_tls_ids in enumerate(self.control_groups_by_taz[taz]):
                signal = self._compute_group_signal(group_tls_ids)
                signals.append(signal)
                if self.action_signal_threshold > 0.0 and signal < self.action_signal_threshold:
                    runtime_mask[taz_idx, group_idx] = False
            self._group_signal_by_taz[taz] = list(signals)
            self._action_signal_by_taz[taz] = float(max(signals) if signals else 0.0)

        return runtime_mask

    def _build_group_feature_block(self, taz: str) -> list[float]:
        block = []
        groups = self.control_groups_by_taz[taz]
        for group_idx in range(self.max_control_groups_per_taz):
            if group_idx >= len(groups):
                block.extend([0.0] * self.per_group_feature_dim)
                continue

            group_tls_ids = groups[group_idx]
            tls_metrics = [
                self._last_tls_metrics_by_id.get(tls_id, self._zero_tls_metrics(taz_id=taz))
                for tls_id in group_tls_ids
            ]
            group_size = max(len(group_tls_ids), 1)
            ew_count = 0
            ns_count = 0
            for tls_id in group_tls_ids:
                axis = self._last_tls_axis_by_id.get(tls_id, "mixed")
                if axis == "EW":
                    ew_count += 1
                elif axis == "NS":
                    ns_count += 1

            mean_phase_duration = float(np.mean([
                np.clip(self._safe_float(metrics.get("phase_duration", 0.0), 0.0) / float(self.max_green), 0.0, 2.0)
                for metrics in tls_metrics
            ]))
            mean_active = float(np.mean([
                np.clip(self._safe_float(metrics.get("active_vehicle_count", 0.0), 0.0) / self.tls_active_vehicle_ref, 0.0, 5.0)
                for metrics in tls_metrics
            ]))
            mean_waiting = float(np.mean([
                np.clip(self._safe_float(metrics.get("total_waiting_time", 0.0), 0.0) / self.tls_waiting_time_ref, 0.0, 5.0)
                for metrics in tls_metrics
            ]))
            mean_pressure = float(np.mean([
                np.clip(max(self._safe_float(metrics.get("pressure", 0.0), 0.0), 0.0) / self.tls_pressure_ref, 0.0, 5.0)
                for metrics in tls_metrics
            ]))
            signal = float(self._group_signal_by_taz.get(taz, [0.0] * len(groups))[group_idx])

            block.extend(
                [
                    1.0,
                    float(len(group_tls_ids) / float(self.coordination_group_max_size)),
                    mean_phase_duration,
                    mean_active,
                    mean_waiting,
                    mean_pressure,
                    float(ew_count / group_size),
                    float(ns_count / group_size),
                    signal,
                ]
            )
        return block

    def _collect_observation(self) -> torch.Tensor:
        base_obs = super()._collect_observation()

        if base_obs.ndim != 2 or tuple(base_obs.shape) != (len(self.taz_ids), self.base_agent_obs_dim):
            base_obs = torch.zeros(
                (len(self.taz_ids), self.base_agent_obs_dim),
                dtype=torch.float32,
                device=self.device,
            )

        if not self.sumo.isRunning():
            self._last_tls_axis_by_id = {
                tls: "mixed"
                for tls in self.tls_list
            }
            self._group_signal_by_taz = {
                taz: [0.0 for _ in groups]
                for taz, groups in self.control_groups_by_taz.items()
            }
            self._prev_action_tls_metrics = self._clone_tls_metrics_snapshot()
            group_tensor = torch.zeros(
                (len(self.taz_ids), self.max_control_groups_per_taz * self.per_group_feature_dim),
                dtype=torch.float32,
                device=self.device,
            )
            return torch.cat([base_obs, group_tensor], dim=-1)

        self._last_tls_axis_by_id = {
            tls_id: self._infer_tls_current_green_axis(tls_id)
            for tls_id in self.tls_list
        }

        group_rows = []
        for taz in self.taz_ids:
            group_rows.append(self._build_group_feature_block(taz))
        group_tensor = torch.tensor(group_rows, dtype=torch.float32, device=self.device)

        if not torch.isfinite(group_tensor).all():
            group_tensor = torch.nan_to_num(group_tensor, nan=0.0, posinf=10.0, neginf=-10.0)

        self._prev_action_tls_metrics = self._clone_tls_metrics_snapshot()
        return torch.cat([base_obs, group_tensor], dim=-1)

    def _select_group_members_for_action(self, group_tls_ids: list[str]) -> tuple[list[str], str]:
        if len(group_tls_ids) <= 1 or not self.group_axis_align_only:
            return list(group_tls_ids), "mixed"

        ew_weight = 0.0
        ns_weight = 0.0
        for tls_id in group_tls_ids:
            axis = self._last_tls_axis_by_id.get(tls_id, "mixed")
            weight = 1.0 + self._safe_float(
                self._last_tls_metrics_by_id.get(tls_id, {}).get("total_waiting_time", 0.0),
                0.0,
            ) / self.tls_waiting_time_ref
            if axis == "EW":
                ew_weight += weight
            elif axis == "NS":
                ns_weight += weight

        if ew_weight <= 0.0 and ns_weight <= 0.0:
            return list(group_tls_ids), "mixed"

        dominant_axis = "EW" if ew_weight >= ns_weight else "NS"
        aligned_members = [
            tls_id
            for tls_id in group_tls_ids
            if self._last_tls_axis_by_id.get(tls_id, "mixed") == dominant_axis
        ]
        if not aligned_members:
            return list(group_tls_ids), dominant_axis
        return aligned_members, dominant_axis

    def _step(self, tensordict):
        action = tensordict["action"]
        if action.requires_grad:
            action = action.detach()
        action = action.to(self.device).reshape(len(self.taz_ids), self.max_control_groups_per_taz)
        action_index = torch.round(action).long().clamp(0, len(self.ACTION_BIN_VALUES) - 1)
        zero_index = torch.full_like(action_index, self.ZERO_ACTION_INDEX)
        runtime_action_mask = self.get_action_mask()
        action_index = torch.where(runtime_action_mask, action_index, zero_index)
        selected_action = self.decode_action_indices(action_index)

        applied_action = torch.zeros_like(selected_action)
        applied_duration_delta = torch.zeros_like(selected_action)
        self.last_group_action_details = {}

        for taz_idx, taz in enumerate(self.taz_ids):
            per_taz_details = []
            for group_idx, group_tls_ids in enumerate(self.control_groups_by_taz[taz]):
                action_value = float(selected_action[taz_idx, group_idx].item())
                applied_action[taz_idx, group_idx] = action_value

                if abs(action_value) <= 1e-9:
                    per_taz_details.append(
                        {
                            "group_index": int(group_idx),
                            "tls_ids": list(group_tls_ids),
                            "selected_tls_ids": [],
                            "dominant_axis": "mixed",
                            "requested_delta": float(action_value),
                            "applied_delta_mean": 0.0,
                        }
                    )
                    continue

                selected_tls_ids, dominant_axis = self._select_group_members_for_action(group_tls_ids)
                member_deltas = [
                    float(self._apply_action_to_tls(tls_id, action_value))
                    for tls_id in selected_tls_ids
                ]
                applied_duration_delta[taz_idx, group_idx] = float(
                    sum(member_deltas) / max(len(member_deltas), 1)
                )
                per_taz_details.append(
                    {
                        "group_index": int(group_idx),
                        "tls_ids": list(group_tls_ids),
                        "selected_tls_ids": list(selected_tls_ids),
                        "dominant_axis": dominant_axis,
                        "requested_delta": float(action_value),
                        "applied_delta_mean": float(applied_duration_delta[taz_idx, group_idx].item()),
                    }
                )
            self.last_group_action_details[taz] = per_taz_details

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

        reward_basis = self._describe_step_reward_basis()

        self.last_reward_components = {
            "reward_basis": reward_basis,
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
            "action_signal_by_taz": dict(self._action_signal_by_taz),
            "group_signal_by_taz": {
                taz: list(signals)
                for taz, signals in self._group_signal_by_taz.items()
            },
            "group_action_details": dict(self.last_group_action_details),
            "action_enabled_ratio": float(self._current_action_mask.float().mean().item()),
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
