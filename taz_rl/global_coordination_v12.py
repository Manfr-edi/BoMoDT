from __future__ import annotations

"""
Global coordination helpers for the v12 hierarchical controller.

The global policy does not operate traffic lights directly. Instead, it emits a
per-TAZ "price" that represents how costly it is to push additional traffic into
that TAZ. Local agents receive these prices as extra observation features, and
their rewards are corrected when local actions appear to create downstream
spillover in other TAZs.
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import torch


COORDINATION_PRICE_BINS = (0.0, 0.5, 1.0, 1.5, 2.0)
TAZ_FEATURE_DIM = 14


@dataclass
class CoordinationObservationConfig:
    """Switches controlling which city-level signals enter the global state."""

    per_taz_feature_dim: int = TAZ_FEATURE_DIM
    include_neighbor_mean: bool = True
    include_flow_features: bool = True
    include_previous_prices: bool = True
    include_city_mean: bool = True
    include_city_std: bool = True
    include_city_max: bool = True
    flow_ref: float = 100.0


@dataclass
class CoordinationRewardConfig:
    """Weights for global reward and local reward correction."""

    city_delta_weight: float = 1.00
    terminal_delta_weight: float = 1.00
    imbalance_weight: float = 0.25
    spillover_weight: float = 0.85
    local_externality_weight: float = 0.75
    local_global_share_weight: float = 0.10
    local_bonus_clip: float = 1.5
    global_reward_clip: float = 4.0
    price_effort_weight: float = 0.03
    action_effort_weight: float = 0.25
    flow_ref: float = 100.0


def _safe_std(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] <= 1:
        return torch.zeros_like(values[0])
    return values.std(dim=0, unbiased=False)


def _zscore(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    std = values.std(unbiased=False)
    if float(std.item()) < 1e-6:
        return torch.zeros_like(values)
    return (values - values.mean()) / std


def parse_taz_polygons(taz_additional_path: str) -> dict[str, list[tuple[float, float]]]:
    """Read SUMO TAZ polygons from output_taz.add.xml."""

    root = ET.parse(taz_additional_path).getroot()
    polygons: dict[str, list[tuple[float, float]]] = {}
    for taz in root.findall(".//taz"):
        taz_id = taz.get("id")
        shape = taz.get("shape")
        if not taz_id or not shape:
            continue
        points = []
        for raw_point in shape.split():
            try:
                x_raw, y_raw = raw_point.split(",", maxsplit=1)
                points.append((float(x_raw), float(y_raw)))
            except Exception:
                continue
        if len(points) >= 3:
            polygons[str(taz_id)] = points
    return polygons


def _polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Compute a robust polygon centroid, falling back to the mean point."""

    if not points:
        return 0.0, 0.0
    signed_area = 0.0
    cx = 0.0
    cy = 0.0
    for idx, (x0, y0) in enumerate(points):
        x1, y1 = points[(idx + 1) % len(points)]
        cross = x0 * y1 - x1 * y0
        signed_area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    signed_area *= 0.5
    if abs(signed_area) < 1e-9:
        return (
            float(sum(x for x, _ in points) / len(points)),
            float(sum(y for _, y in points) / len(points)),
        )
    return float(cx / (6.0 * signed_area)), float(cy / (6.0 * signed_area))


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Distance between a point and a segment in SUMO network coordinates."""

    px, py = point
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / denom))
    proj_x = x0 + t * dx
    proj_y = y0 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _segments_distance(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> float:
    """Cheap segment distance used to estimate whether two TAZ boundaries touch."""

    return min(
        _point_segment_distance(a0, b0, b1),
        _point_segment_distance(a1, b0, b1),
        _point_segment_distance(b0, a0, a1),
        _point_segment_distance(b1, a0, a1),
    )


def _simplify_polygon(points: list[tuple[float, float]], max_points: int = 180) -> list[tuple[float, float]]:
    """Downsample detailed TAZ boundaries so adjacency construction stays fast."""

    if len(points) <= max_points:
        return points
    stride = max(int(math.ceil(len(points) / float(max_points))), 1)
    return points[::stride]


def _polygon_boundary_distance(poly_a: list[tuple[float, float]], poly_b: list[tuple[float, float]]) -> float:
    """Approximate the minimum distance between two TAZ boundaries."""

    poly_a = _simplify_polygon(poly_a)
    poly_b = _simplify_polygon(poly_b)
    best = float("inf")
    for idx_a, a0 in enumerate(poly_a):
        a1 = poly_a[(idx_a + 1) % len(poly_a)]
        for idx_b, b0 in enumerate(poly_b):
            b1 = poly_b[(idx_b + 1) % len(poly_b)]
            best = min(best, _segments_distance(a0, a1, b0, b1))
            if best <= 1e-6:
                return 0.0
    return float(best)


def build_taz_adjacency_matrix(
    taz_ids: list[str],
    taz_additional_path: str,
    distance_threshold: float = 80.0,
    fallback_k: int = 3,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Build a symmetric TAZ adjacency matrix from SUMO TAZ polygons.

    TAZs are linked when their polygon boundaries are close enough. If a TAZ has
    no detected neighbor, the nearest polygons by distance are used as fallback
    so every TAZ has at least some coordination context.
    """

    polygons = parse_taz_polygons(taz_additional_path)
    n_taz = len(taz_ids)
    matrix = torch.zeros((n_taz, n_taz), dtype=torch.float32, device=device)
    centroids = {
        taz_id: _polygon_centroid(polygons.get(taz_id, []))
        for taz_id in taz_ids
    }
    distances: dict[str, dict[str, float]] = {taz_id: {} for taz_id in taz_ids}

    for i, taz_i in enumerate(taz_ids):
        poly_i = polygons.get(taz_i, [])
        for j, taz_j in enumerate(taz_ids):
            if i == j:
                continue
            poly_j = polygons.get(taz_j, [])
            if poly_i and poly_j:
                distance = _polygon_boundary_distance(poly_i, poly_j)
            else:
                xi, yi = centroids[taz_i]
                xj, yj = centroids[taz_j]
                distance = math.hypot(xi - xj, yi - yj)
            distances[taz_i][taz_j] = float(distance)
            if distance <= float(distance_threshold):
                matrix[i, j] = 1.0

    for i, taz_id in enumerate(taz_ids):
        if float(matrix[i].sum().item()) > 0.0:
            continue
        nearest = sorted(distances[taz_id].items(), key=lambda item: item[1])[: max(int(fallback_k), 1)]
        for neighbor_id, _ in nearest:
            matrix[i, taz_ids.index(neighbor_id)] = 1.0

    matrix = torch.maximum(matrix, matrix.T)
    metadata = {
        "source": taz_additional_path,
        "distance_threshold": float(distance_threshold),
        "fallback_k": int(fallback_k),
        "neighbors_by_taz": {
            taz_id: [
                taz_ids[j]
                for j in range(n_taz)
                if bool(matrix[i, j].item())
            ]
            for i, taz_id in enumerate(taz_ids)
        },
        "centroids_by_taz": {
            taz_id: [float(centroids[taz_id][0]), float(centroids[taz_id][1])]
            for taz_id in taz_ids
        },
    }
    return matrix, metadata


def extract_taz_features(local_observation: torch.Tensor, env) -> torch.Tensor:
    """Extract the inherited 14 per-TAZ features from a v11/v12 local observation."""

    start_idx = int(env.max_tls_per_taz * env.per_tls_feature_dim)
    end_idx = start_idx + int(env.per_taz_feature_dim)
    return local_observation[:, start_idx:end_idx]


def _dict_values_by_taz(taz_ids: list[str], values: Optional[dict], key: str) -> torch.Tensor:
    if not values:
        return torch.zeros(len(taz_ids), dtype=torch.float32)
    out = []
    for taz_id in taz_ids:
        raw = values.get(taz_id, {})
        if isinstance(raw, dict):
            value = raw.get(key, 0.0)
        else:
            value = 0.0
        try:
            out.append(float(value))
        except Exception:
            out.append(0.0)
    return torch.tensor(out, dtype=torch.float32)


def summarize_step_actions(
    taz_ids: list[str],
    applied_actions: torch.Tensor,
    applied_deltas: torch.Tensor,
    action_mask: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Aggregate local action intensity per TAZ for the global state/reward."""

    actions = applied_actions.detach().cpu()
    deltas = applied_deltas.detach().cpu()
    mask = action_mask.detach().cpu().bool()
    summary = {}
    for taz_idx, taz_id in enumerate(taz_ids):
        valid_actions = actions[taz_idx][mask[taz_idx]]
        valid_deltas = deltas[taz_idx][mask[taz_idx]]
        if valid_actions.numel() == 0:
            summary[taz_id] = {
                "avg_action_selected": 0.0,
                "avg_applied_duration_delta": 0.0,
                "applied_duration_nonzero_ratio": 0.0,
                "aggressive_action_ratio": 0.0,
            }
            continue
        nonzero = (valid_deltas.abs() > 1e-3).float()
        aggressive = (valid_actions.abs() >= 10.0).float()
        summary[taz_id] = {
            "avg_action_selected": float(valid_actions.float().mean().item()),
            "avg_applied_duration_delta": float(valid_deltas.float().mean().item()),
            "applied_duration_nonzero_ratio": float(nonzero.mean().item()),
            "aggressive_action_ratio": float(aggressive.mean().item()),
        }
    return summary


def action_summary_tensor(taz_ids: list[str], action_summary_by_taz: Optional[dict], device: torch.device) -> torch.Tensor:
    """Convert per-TAZ action summaries into fixed-width global-state features."""

    if not action_summary_by_taz:
        return torch.zeros((len(taz_ids), 3), dtype=torch.float32, device=device)
    rows = []
    for taz_id in taz_ids:
        summary = dict(action_summary_by_taz.get(taz_id, {}) or {})
        rows.append(
            [
                float(summary.get("avg_applied_duration_delta", 0.0)) / 10.0,
                float(summary.get("applied_duration_nonzero_ratio", 0.0)),
                float(summary.get("aggressive_action_ratio", 0.0)),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32, device=device)


def build_coordination_observation(
    taz_features: torch.Tensor,
    adjacency_matrix: torch.Tensor,
    previous_flow_matrix: Optional[torch.Tensor] = None,
    previous_prices: Optional[torch.Tensor] = None,
    previous_action_summary: Optional[dict] = None,
    taz_ids: Optional[list[str]] = None,
    extra_context: Optional[torch.Tensor] = None,
    config: Optional[CoordinationObservationConfig] = None,
) -> torch.Tensor:
    """
    Build one global observation row for the coordination policy.

    The observation combines per-TAZ emission/traffic features, neighbor
    aggregates, previous inter-TAZ flows, previous global prices, previous local
    action intensity, city-level statistics, and episode context.
    """

    cfg = config or CoordinationObservationConfig()
    features = taz_features[:, : int(cfg.per_taz_feature_dim)].float()
    device = features.device
    adjacency = adjacency_matrix.to(device=device, dtype=features.dtype)
    chunks = [features.reshape(-1)]

    if cfg.include_neighbor_mean:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbor_mean = adjacency.matmul(features) / degree
        chunks.append(neighbor_mean.reshape(-1))

    if cfg.include_flow_features:
        if previous_flow_matrix is None:
            flow = torch.zeros((features.shape[0], features.shape[0]), dtype=features.dtype, device=device)
        else:
            flow = previous_flow_matrix.to(device=device, dtype=features.dtype)
        flow_ref = max(float(cfg.flow_ref), 1e-6)
        outflow = flow.sum(dim=-1) / flow_ref
        inflow = flow.sum(dim=0) / flow_ref
        neighbor_outflow = (flow * adjacency).sum(dim=-1) / flow_ref
        neighbor_inflow = (flow * adjacency).sum(dim=0) / flow_ref
        chunks.append(torch.stack([outflow, inflow, neighbor_outflow, neighbor_inflow], dim=-1).reshape(-1))

    if cfg.include_previous_prices:
        if previous_prices is None:
            price_values = torch.zeros(features.shape[0], dtype=features.dtype, device=device)
        else:
            price_values = previous_prices.reshape(-1).to(device=device, dtype=features.dtype)
        max_price = max(max(COORDINATION_PRICE_BINS), 1e-6)
        price_norm = price_values / max_price
        neighbor_price = adjacency.matmul(price_norm.unsqueeze(-1)).squeeze(-1) / adjacency.sum(dim=-1).clamp_min(1.0)
        chunks.append(torch.stack([price_norm, neighbor_price, price_norm - neighbor_price], dim=-1).reshape(-1))

    if taz_ids is not None:
        chunks.append(action_summary_tensor(taz_ids, previous_action_summary, device=device).reshape(-1))

    if cfg.include_city_mean:
        chunks.append(features.mean(dim=0))
    if cfg.include_city_std:
        chunks.append(_safe_std(features))
    if cfg.include_city_max:
        chunks.append(features.max(dim=0).values)
    if extra_context is not None:
        chunks.append(extra_context.reshape(-1).to(device=device, dtype=features.dtype))

    return torch.cat(chunks, dim=0).unsqueeze(0)


def augment_local_observation_with_prices(
    local_observation: torch.Tensor,
    price_values: torch.Tensor,
    adjacency_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Append global price context to each local observation row.

    Each local agent receives its own TAZ price, the mean price of neighboring
    TAZs, and the difference between the two. This lets the local policy react
    during rollout instead of only receiving a terminal shaping signal.
    """

    prices = price_values.reshape(-1).to(device=local_observation.device, dtype=local_observation.dtype)
    adjacency = adjacency_matrix.to(device=local_observation.device, dtype=local_observation.dtype)
    max_price = max(max(COORDINATION_PRICE_BINS), 1e-6)
    price_norm = prices / max_price
    degree = adjacency.sum(dim=-1).clamp_min(1.0)
    neighbor_price = adjacency.matmul(price_norm.unsqueeze(-1)).squeeze(-1) / degree
    price_features = torch.stack([price_norm, neighbor_price, price_norm - neighbor_price], dim=-1)
    return torch.cat([local_observation, price_features], dim=-1)


def compute_coordination_step_rewards(
    taz_ids: list[str],
    price_values: torch.Tensor,
    reward_components: dict,
    flow_matrix: torch.Tensor,
    action_summary_by_taz: Optional[dict],
    baseline_penalty: Optional[float] = None,
    terminal_penalty: Optional[float] = None,
    config: Optional[CoordinationRewardConfig] = None,
) -> tuple[torch.Tensor, float, dict]:
    """
    Compute reward shaping for local agents and a scalar reward for the global agent.

    The externality term estimates whether a TAZ sends vehicles toward expensive
    downstream TAZs that are currently worsening. Local agents are penalized for
    that spillover, while the global reward balances total emission improvement,
    terminal baseline improvement, imbalance, price effort, and spillover.
    """

    cfg = config or CoordinationRewardConfig()
    device = price_values.device
    dtype = price_values.dtype
    n_taz = len(taz_ids)

    penalties = torch.tensor(
        [float((reward_components.get("penalty_by_taz", {}) or {}).get(taz_id, 0.0)) for taz_id in taz_ids],
        dtype=dtype,
        device=device,
    )
    delta_penalties = torch.tensor(
        [float((reward_components.get("delta_penalty_by_taz", {}) or {}).get(taz_id, 0.0)) for taz_id in taz_ids],
        dtype=dtype,
        device=device,
    )
    prices = price_values.reshape(-1).to(device=device, dtype=dtype)
    if prices.numel() != n_taz:
        raise ValueError(f"Expected {n_taz} prices, got {prices.numel()}.")

    flow = flow_matrix.to(device=device, dtype=dtype)
    if flow.shape != (n_taz, n_taz):
        flow = torch.zeros((n_taz, n_taz), dtype=dtype, device=device)
    flow_ref = max(float(cfg.flow_ref), 1e-6)
    row_flow = flow.sum(dim=-1, keepdim=True).clamp_min(1.0)
    flow_share = flow / row_flow

    # In the environment, positive delta_penalty means "penalty decreased".
    worsening = torch.clamp(-delta_penalties, min=0.0)
    local_gain = torch.clamp(delta_penalties, min=0.0)
    normalized_price = prices / max(max(COORDINATION_PRICE_BINS), 1e-6)

    action_effort = _dict_values_by_taz(taz_ids, action_summary_by_taz, "aggressive_action_ratio").to(device=device, dtype=dtype)
    flow_intensity = torch.clamp(flow.sum(dim=-1) / flow_ref, min=0.0, max=2.0)
    # Attribute downstream worsening to sources according to observed TAZ->TAZ flows.
    downstream_cost = flow_share.matmul(normalized_price * worsening)
    externality = torch.clamp(
        downstream_cost * (1.0 + cfg.action_effort_weight * action_effort) * (1.0 + 0.25 * flow_intensity)
        - 0.25 * local_gain,
        min=0.0,
    )

    city_delta = float(delta_penalties.sum().item())
    if baseline_penalty is not None and terminal_penalty is not None:
        city_term = float(baseline_penalty - terminal_penalty)
        city_reward_basis = "baseline_terminal_delta"
    else:
        city_term = city_delta
        city_reward_basis = "step_penalty_delta"

    imbalance = float(penalties.std(unbiased=False).item()) if penalties.numel() > 1 else 0.0
    spillover = float(externality.mean().item()) if externality.numel() > 0 else 0.0
    price_effort = float(normalized_price.abs().mean().item()) if normalized_price.numel() > 0 else 0.0
    alignment = float((_zscore(normalized_price) * _zscore(worsening)).mean().item()) if n_taz > 1 else 0.0

    global_reward = (
        cfg.city_delta_weight * city_delta
        + cfg.terminal_delta_weight * city_term
        - cfg.imbalance_weight * imbalance
        - cfg.spillover_weight * spillover
        - cfg.price_effort_weight * price_effort
        + 0.10 * alignment
    )
    global_reward = max(-cfg.global_reward_clip, min(cfg.global_reward_clip, float(global_reward)))

    local_adjustment = (
        -cfg.local_externality_weight * externality
        + cfg.local_global_share_weight * float(city_delta) / max(float(n_taz), 1.0)
    )
    local_adjustment = torch.clamp(local_adjustment, -cfg.local_bonus_clip, cfg.local_bonus_clip)

    diagnostics = {
        "city_delta": float(city_delta),
        "city_reward_basis": city_reward_basis,
        "city_term": float(city_term),
        "imbalance": float(imbalance),
        "spillover": float(spillover),
        "price_effort": float(price_effort),
        "alignment": float(alignment),
        "global_reward": float(global_reward),
        "price_by_taz": {taz_id: float(prices[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "worsening_by_taz": {taz_id: float(worsening[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "local_gain_by_taz": {taz_id: float(local_gain[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "externality_by_taz": {taz_id: float(externality[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "local_adjustment_by_taz": {taz_id: float(local_adjustment[idx].item()) for idx, taz_id in enumerate(taz_ids)},
        "outflow_by_taz": {taz_id: float(flow[idx].sum().item()) for idx, taz_id in enumerate(taz_ids)},
        "inflow_by_taz": {taz_id: float(flow[:, idx].sum().item()) for idx, taz_id in enumerate(taz_ids)},
    }
    return local_adjustment, float(global_reward), diagnostics
