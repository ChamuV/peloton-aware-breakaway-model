# src/strategic_peloton.py
from __future__ import annotations

from dataclasses import replace

from src.backward_method import PelotonParameters


def passive_peloton() -> PelotonParameters:
    return PelotonParameters(
        strategic=False,
        terrain_multiplier=1.0,
        slope_threshold=0.004,
        surge_start=0.8,
    )


def terrain_aware_peloton(
    terrain_multiplier: float = 1.1,
    slope_threshold: float = 0.004,
) -> PelotonParameters:
    return PelotonParameters(
        strategic=True,
        terrain_multiplier=terrain_multiplier,
        slope_threshold=slope_threshold,
        surge_start=1.1,
    )


def strategic_peloton(
    terrain_multiplier: float = 1.1,
    slope_threshold: float = 0.004,
    surge_start: float = 0.8,
) -> PelotonParameters:
    return PelotonParameters(
        strategic=True,
        terrain_multiplier=terrain_multiplier,
        slope_threshold=slope_threshold,
        surge_start=surge_start,
    )


def with_modified_surge(
    params: PelotonParameters,
    surge_start: float | None = None,
) -> PelotonParameters:
    updates = {}
    if surge_start is not None:
        updates["surge_start"] = surge_start
    return replace(params, **updates)


def with_modified_terrain_response(
    params: PelotonParameters,
    terrain_multiplier: float | None = None,
    slope_threshold: float | None = None,
) -> PelotonParameters:
    updates = {}
    if terrain_multiplier is not None:
        updates["terrain_multiplier"] = terrain_multiplier
    if slope_threshold is not None:
        updates["slope_threshold"] = slope_threshold
    return replace(params, **updates)


__all__ = [
    "passive_peloton",
    "terrain_aware_peloton",
    "strategic_peloton",
    "with_modified_surge",
    "with_modified_terrain_response",
]