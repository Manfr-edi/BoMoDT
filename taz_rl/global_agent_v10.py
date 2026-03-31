from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


GLOBAL_PRIORITY_BINS = (-1.0, 0.0, 1.0)


@dataclass
class GlobalObservationConfig:
    per_taz_feature_dim: int = 14
    include_city_mean: bool = True
    include_city_std: bool = True
    include_city_max: bool = True
    include_city_min: bool = False
    include_action_density: bool = False


@dataclass
class GlobalRewardConfig:
    local_bonus_scale: float = 0.75
    alignment_reward_weight: float = 0.35
    delta_penalty_weight: float = 0.70
    residual_penalty_weight: float = 0.30
    effort_weight: float = 0.20
    effort_delta_ref: float = 5.0
    local_bonus_clip: float = 1.5
    global_reward_clip: float = 4.0
    city_reward_weight: float = 1.0


def _safe_std(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] <= 1:
        return torch.zeros_like(values[0])
    return values.std(dim=0, unbiased=False)


def _zscore(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    centered = values - values.mean()
    std = values.std(unbiased=False)
    if float(std.item()) < 1e-6:
        return torch.zeros_like(values)
    return centered / std


def _extract_taz_tail_features(
    local_observation: torch.Tensor,
    per_taz_feature_dim: int,
) -> torch.Tensor:
    if local_observation.ndim != 2:
        raise ValueError(
            f"Expected local observation with shape [num_taz, obs_dim], got {tuple(local_observation.shape)}."
        )
    if local_observation.shape[-1] < int(per_taz_feature_dim):
        raise ValueError(
            f"Observation width {local_observation.shape[-1]} is smaller than per_taz_feature_dim={per_taz_feature_dim}."
        )
    return local_observation[:, -int(per_taz_feature_dim):]


def build_global_observation(
    local_observation: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    extra_context: Optional[torch.Tensor] = None,
    config: Optional[GlobalObservationConfig] = None,
) -> torch.Tensor:
    cfg = config or GlobalObservationConfig()
    taz_tail = _extract_taz_tail_features(local_observation, cfg.per_taz_feature_dim)
    chunks = [taz_tail.reshape(-1)]

    if cfg.include_city_mean:
        chunks.append(taz_tail.mean(dim=0))
    if cfg.include_city_std:
        chunks.append(_safe_std(taz_tail))
    if cfg.include_city_max:
        chunks.append(taz_tail.max(dim=0).values)
    if cfg.include_city_min:
        chunks.append(taz_tail.min(dim=0).values)

    if cfg.include_action_density:
        if action_mask is None:
            density = torch.zeros(taz_tail.shape[0], device=taz_tail.device, dtype=taz_tail.dtype)
        else:
            density = action_mask.float().mean(dim=-1).to(device=taz_tail.device, dtype=taz_tail.dtype)
        chunks.append(density)

    if extra_context is not None:
        chunks.append(extra_context.reshape(-1).to(device=taz_tail.device, dtype=taz_tail.dtype))

    global_observation = torch.cat(chunks, dim=0)
    return global_observation.unsqueeze(0)


def build_global_action_mask(num_taz: int, device: torch.device) -> torch.Tensor:
    return torch.ones((1, int(num_taz)), dtype=torch.bool, device=device)


def compute_priority_rewards(
    taz_ids: list[str],
    priority_action_values: torch.Tensor,
    reward_components: dict,
    action_summary_by_taz: dict,
    baseline_penalty: Optional[float],
    terminal_penalty: Optional[float],
    config: Optional[GlobalRewardConfig] = None,
) -> tuple[torch.Tensor, float, dict]:
    cfg = config or GlobalRewardConfig()
    device = priority_action_values.device
    dtype = priority_action_values.dtype

    raw_priority = priority_action_values.reshape(-1).to(device=device, dtype=dtype)
    if raw_priority.numel() != len(taz_ids):
        raise ValueError(
            f"Expected {len(taz_ids)} priority values, got {raw_priority.numel()}."
        )

    centered_priority = raw_priority - raw_priority.mean()
    max_abs_priority = centered_priority.abs().max()
    if float(max_abs_priority.item()) > 1e-6:
        normalized_priority = centered_priority / max_abs_priority
    else:
        normalized_priority = torch.zeros_like(centered_priority)

    penalty_values = torch.tensor(
        [float((reward_components.get("penalty_by_taz", {}) or {}).get(taz, 0.0)) for taz in taz_ids],
        dtype=dtype,
        device=device,
    )
    delta_penalty_values = torch.tensor(
        [float((reward_components.get("delta_penalty_by_taz", {}) or {}).get(taz, 0.0)) for taz in taz_ids],
        dtype=dtype,
        device=device,
    )

    effort_values = []
    for taz in taz_ids:
        summary = dict(action_summary_by_taz.get(taz, {}) or {})
        avg_delta = abs(float(summary.get("avg_applied_duration_delta", 0.0)))
        nonzero_ratio = float(summary.get("applied_duration_nonzero_ratio", 0.0))
        effort_delta_ref = max(float(cfg.effort_delta_ref), 1e-6)
        effort = 0.5 * min(avg_delta / effort_delta_ref, 1.0) + 0.5 * min(max(nonzero_ratio, 0.0), 1.0)
        effort_values.append(effort)
    effort_tensor = torch.tensor(effort_values, dtype=dtype, device=device)

    normalized_delta = _zscore(delta_penalty_values)
    normalized_penalty = _zscore(penalty_values)
    centered_effort = effort_tensor - effort_tensor.mean() if effort_tensor.numel() > 0 else effort_tensor
    priority_signal = (
        cfg.delta_penalty_weight * normalized_delta
        - cfg.residual_penalty_weight * normalized_penalty
        + cfg.effort_weight * centered_effort
    )

    local_bonus = cfg.local_bonus_scale * normalized_priority * priority_signal
    local_bonus = torch.clamp(local_bonus, -cfg.local_bonus_clip, cfg.local_bonus_clip)

    if baseline_penalty is not None and terminal_penalty is not None:
        city_reward = float(baseline_penalty - terminal_penalty)
        city_reward_basis = "baseline_delta"
    else:
        city_reward = -float(penalty_values.mean().item()) if penalty_values.numel() > 0 else 0.0
        city_reward_basis = "negative_mean_penalty"

    alignment_reward = float((normalized_priority * priority_signal).mean().item()) if priority_signal.numel() > 0 else 0.0
    global_reward = cfg.city_reward_weight * city_reward + cfg.alignment_reward_weight * alignment_reward
    global_reward = max(-cfg.global_reward_clip, min(cfg.global_reward_clip, global_reward))

    diagnostics = {
        "city_reward": float(city_reward),
        "city_reward_basis": city_reward_basis,
        "alignment_reward": float(alignment_reward),
        "priority_raw_by_taz": {
            taz: float(raw_priority[idx].item())
            for idx, taz in enumerate(taz_ids)
        },
        "priority_weight_by_taz": {
            taz: float(normalized_priority[idx].item())
            for idx, taz in enumerate(taz_ids)
        },
        "priority_signal_by_taz": {
            taz: float(priority_signal[idx].item())
            for idx, taz in enumerate(taz_ids)
        },
        "effort_by_taz": {
            taz: float(effort_tensor[idx].item())
            for idx, taz in enumerate(taz_ids)
        },
        "local_bonus_by_taz": {
            taz: float(local_bonus[idx].item())
            for idx, taz in enumerate(taz_ids)
        },
        "delta_penalty_by_taz": {
            taz: float(delta_penalty_values[idx].item())
            for idx, taz in enumerate(taz_ids)
        },
        "penalty_by_taz": {
            taz: float(penalty_values[idx].item())
            for idx, taz in enumerate(taz_ids)
        },
    }
    return local_bonus, float(global_reward), diagnostics
