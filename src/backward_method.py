# src/backward_method.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

from src.rider_models import RiderArchetype
from src.track_profiles import compute_theta


@dataclass(frozen=True)
class LegacyScaling:
    mass_scale: float = 70.0
    length_scale: float = 100000.0
    gravity: float = 9.8
    air_density: float = 1.3
    drag_coefficient: float = 0.8
    frontal_area: float = 0.3
    friction_all: float = 3.0
    peloton_power: float = 280.0
    peloton_gamma: float = 0.8
    peloton_dimless_baseline: float = 4.0
    peloton_surge_power: float = 350.0

    @property
    def time_scale(self) -> float:
        return np.sqrt(
            (
                self.air_density
                * self.drag_coefficient
                * self.frontal_area
                * self.length_scale**2
            )
            / (2 * self.mass_scale * self.gravity)
        )

    @property
    def p_star(self) -> float:
        return 1.0 / (self.mass_scale * self.gravity * self.length_scale)

    @property
    def dim_factor(self) -> float:
        return self.peloton_power / self.peloton_dimless_baseline

    @property
    def peloton_power_nd(self) -> float:
        return self.peloton_power * self.time_scale * self.p_star

    @property
    def peloton_surge_power_nd(self) -> float:
        return self.peloton_surge_power * self.time_scale * self.p_star

    @property
    def friction_nd(self) -> float:
        return self.friction_all / (self.mass_scale * self.gravity)


# Backward compatibility alias
PhysicalParameters = LegacyScaling


@dataclass(frozen=True)
class PelotonParameters:
    strategic: bool = False
    terrain_multiplier: float = 1.1
    slope_threshold: float = 0.004
    surge_start: float = 0.8


@dataclass(frozen=True)
class BreakawayResult:
    breakaway_location: float
    breakaway_power_wkg: float
    breakaway_power_watts: float
    total_time_nd: float
    total_time_minutes: float
    feasible: bool


@dataclass(frozen=True)
class SweepResult:
    x_break: np.ndarray
    total_time_minutes: np.ndarray
    total_time_nd: np.ndarray
    breakaway_power_watts: np.ndarray
    breakaway_power_wkg: np.ndarray
    feasible: np.ndarray


def endurance_time_nd(push_dim: float, rider: RiderArchetype, scaling: LegacyScaling) -> float:
    ts = scaling.time_scale

    if rider.name == "sprinter":
        decay = 0.002
        bpo = 5.0
        mpo = 20.0
        if push_dim <= bpo or push_dim >= mpo:
            return np.nan
        return (1.0 / (decay * ts)) * np.log((mpo - bpo) / (push_dim - bpo))

    if rider.name == "climber":
        mpo = 17.0
        h_decay = 0.13
        if push_dim <= 0:
            return np.nan
        return (1.0 / ts) * (mpo / push_dim) ** (1.0 / h_decay)

    # time-trialist / flatter
    mpo = 19.0
    bpo = 4.0
    e_decay = 0.00175
    h_decay = 0.15
    e_weight = 0.3
    h_weight = 0.7
    if push_dim <= bpo or push_dim >= mpo:
        return np.nan
    e_end = (1.0 / (e_decay * ts)) * np.log((mpo - bpo) / (push_dim - bpo))
    h_end = (1.0 / ts) * (mpo / push_dim) ** (1.0 / h_decay)
    return e_weight * e_end + h_weight * h_end


def peloton_velocity_nd(
    x: float,
    track_name: str,
    scaling: LegacyScaling,
    peloton: PelotonParameters,
) -> float:
    slope_term = np.sin(compute_theta(track_name, x))

    if peloton.strategic:
        if x >= peloton.surge_start:
            p_nd = scaling.peloton_surge_power_nd
        elif slope_term > peloton.slope_threshold:
            p_nd = scaling.peloton_power_nd * peloton.terrain_multiplier
        else:
            p_nd = scaling.peloton_power_nd
    else:
        p_nd = scaling.peloton_power_nd

    def f(v: float) -> float:
        return (p_nd / v) - (scaling.peloton_gamma * v**2) - scaling.friction_nd - slope_term

    sol = root_scalar(f, bracket=[1e-6, 100], method="bisect")
    return sol.root if sol.converged else np.nan


def peloton_time_nd(
    x_end: float,
    track_name: str,
    scaling: LegacyScaling,
    peloton: PelotonParameters,
) -> float:
    integrand = lambda xx: 1.0 / peloton_velocity_nd(xx, track_name, scaling, peloton)
    return quad(integrand, 0.0, x_end, epsabs=1e-6, epsrel=1e-6, limit=200)[0]


def breakaway_velocity_nd(
    x: float,
    track_name: str,
    breakaway_power_nd: float,
    rider_gamma: float,
    rider_mass_nd: float,
    scaling: LegacyScaling,
) -> float:
    slope_const = rider_mass_nd * np.sin(compute_theta(track_name, x))

    def f(v: float) -> float:
        return (breakaway_power_nd / v) - (rider_gamma * v**2) - scaling.friction_nd - slope_const

    sol = root_scalar(f, bracket=[1e-6, 100], method="bisect")
    return sol.root if sol.converged else np.nan


def breakaway_travel_time_nd(
    xb: float,
    track_name: str,
    breakaway_power_nd: float,
    rider_gamma: float,
    rider_mass_nd: float,
    scaling: LegacyScaling,
) -> float:
    integrand = lambda xx: 1.0 / breakaway_velocity_nd(
        xx,
        track_name,
        breakaway_power_nd,
        rider_gamma,
        rider_mass_nd,
        scaling,
    )
    return quad(integrand, xb, 1.0, epsabs=1e-8, epsrel=1e-8, limit=200)[0]


def evaluate_power_sweep(
    rider: RiderArchetype,
    track_name: str,
    scaling: LegacyScaling | None = None,
    peloton: PelotonParameters | None = None,
    n_powers: int = 20,
    power_factor_min: float = 1.275,
    power_factor_max: float = 2.2,
    rider_gamma: float = 1.0,
    min_breakaway_location: float = 0.4,
) -> SweepResult:
    if scaling is None:
        scaling = LegacyScaling()
    if peloton is None:
        peloton = PelotonParameters(strategic=False)

    powers_watts = np.linspace(
        power_factor_min * scaling.peloton_power,
        power_factor_max * scaling.peloton_power,
        n_powers,
    )

    rider_mass_nd = rider.mass / scaling.mass_scale

    x_break = np.full(n_powers, np.nan, dtype=float)
    total_time_nd = np.full(n_powers, np.nan, dtype=float)
    feasible = np.zeros(n_powers, dtype=bool)

    for j, push_watts in enumerate(powers_watts):
        push_dim = push_watts / scaling.dim_factor
        t_end_nd = endurance_time_nd(push_dim, rider, scaling)

        if np.isnan(t_end_nd) or t_end_nd <= 0:
            continue

        breakaway_power_nd = push_watts * scaling.time_scale * scaling.p_star * rider_mass_nd

        def f_xb(xb: float) -> float:
            return breakaway_travel_time_nd(
                xb,
                track_name,
                breakaway_power_nd,
                rider_gamma,
                rider_mass_nd,
                scaling,
            ) - t_end_nd

        try:
            if f_xb(0.0) * f_xb(1.0) > 0:
                continue

            left, right = 0.0, 1.0
            for _ in range(50):
                mid = 0.5 * (left + right)
                if f_xb(left) * f_xb(mid) <= 0:
                    right = mid
                else:
                    left = mid
            xb_sol = 0.5 * (left + right)

            if xb_sol < min_breakaway_location:
                continue

            t_pel_nd = peloton_time_nd(xb_sol, track_name, scaling, peloton)
            x_break[j] = xb_sol
            total_time_nd[j] = t_pel_nd + t_end_nd
            feasible[j] = True
        except Exception:
            continue

    total_time_minutes = total_time_nd * scaling.time_scale / 60.0
    breakaway_power_wkg = powers_watts / rider.mass

    return SweepResult(
        x_break=x_break,
        total_time_minutes=total_time_minutes,
        total_time_nd=total_time_nd,
        breakaway_power_watts=powers_watts,
        breakaway_power_wkg=breakaway_power_wkg,
        feasible=feasible,
    )


def best_result_from_sweep(sweep: SweepResult) -> BreakawayResult:
    valid = np.where(sweep.feasible & np.isfinite(sweep.total_time_minutes))[0]
    if len(valid) == 0:
        raise ValueError("No feasible strategies found in sweep.")

    idx = valid[np.argmin(sweep.total_time_minutes[valid])]
    return BreakawayResult(
        breakaway_location=float(sweep.x_break[idx]),
        breakaway_power_wkg=float(sweep.breakaway_power_wkg[idx]),
        breakaway_power_watts=float(sweep.breakaway_power_watts[idx]),
        total_time_nd=float(sweep.total_time_nd[idx]),
        total_time_minutes=float(sweep.total_time_minutes[idx]),
        feasible=True,
    )


__all__ = [
    "LegacyScaling",
    "PhysicalParameters",
    "PelotonParameters",
    "BreakawayResult",
    "SweepResult",
    "endurance_time_nd",
    "peloton_velocity_nd",
    "peloton_time_nd",
    "breakaway_velocity_nd",
    "breakaway_travel_time_nd",
    "evaluate_power_sweep",
    "best_result_from_sweep",
]