# src/grid_search.py
from __future__ import annotations

from dataclasses import dataclass

from src.backward_method import (
    BreakawayResult,
    LegacyScaling,
    PelotonParameters,
    SweepResult,
    best_result_from_sweep,
    evaluate_power_sweep,
)
from src.rider_models import RiderArchetype


@dataclass(frozen=True)
class GridSearchConfig:
    n_powers: int = 20
    power_factor_min: float = 1.275
    power_factor_max: float = 2.2
    min_breakaway_location: float = 0.4


def run_grid_search(
    rider: RiderArchetype,
    track_name: str,
    config: GridSearchConfig | None = None,
    x_grid=None,
    phys=None,
    peloton: PelotonParameters | None = None,
    scaling: LegacyScaling | None = None,
) -> SweepResult:
    if config is None:
        config = GridSearchConfig()
    if scaling is None:
        scaling = LegacyScaling()

    return evaluate_power_sweep(
        rider=rider,
        track_name=track_name,
        scaling=scaling,
        peloton=peloton,
        n_powers=config.n_powers,
        power_factor_min=config.power_factor_min,
        power_factor_max=config.power_factor_max,
        min_breakaway_location=config.min_breakaway_location,
    )


def run_grid_search_best(
    rider: RiderArchetype,
    track_name: str,
    config: GridSearchConfig | None = None,
    x_grid=None,
    phys=None,
    peloton: PelotonParameters | None = None,
    scaling: LegacyScaling | None = None,
) -> BreakawayResult:
    sweep = run_grid_search(
        rider=rider,
        track_name=track_name,
        config=config,
        x_grid=x_grid,
        phys=phys,
        peloton=peloton,
        scaling=scaling,
    )
    return best_result_from_sweep(sweep)


__all__ = [
    "GridSearchConfig",
    "run_grid_search",
    "run_grid_search_best",
]