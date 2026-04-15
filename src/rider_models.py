# src/rider_models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ArchetypeName = Literal["climber", "sprinter", "time_trialist"]


@dataclass(frozen=True)
class RiderArchetype:
    """
    Parameters for a rider archetype.

    Attributes
    ----------
    name:
        Rider type label.
    mass:
        Rider mass in kg.
    bpo:
        Base power output in W/kg.
    mpo:
        Maximum power output in W/kg.
    model_type:
        One of {'hyperbolic', 'exponential', 'composite'}.
    a, d:
        Hyperbolic model parameters.
    b, M, k:
        Exponential model parameters.
    alpha:
        Weighting parameter for the composite model.
    """

    name: str
    mass: float
    bpo: float
    mpo: float
    model_type: str

    # Hyperbolic parameters
    a: float | None = None
    d: float | None = None

    # Exponential parameters
    b: float | None = None
    M: float | None = None
    k: float | None = None

    # Composite parameter
    alpha: float | None = None


def hyperbolic_power(t: np.ndarray | float, a: float, d: float) -> np.ndarray:
    """
    Hyperbolic-like power model:
        P(t) = a * t^{-d}

    Used for climbers in the report.
    """
    t = np.asarray(t, dtype=float)
    t = np.maximum(t, 1e-8)
    return a * t ** (-d)


def hyperbolic_duration(P: np.ndarray | float, a: float, d: float) -> np.ndarray:
    """
    Inverse hyperbolic model:
        Δt(P) = (a / P)^{1/d}
    """
    P = np.asarray(P, dtype=float)
    P = np.maximum(P, 1e-8)
    return (a / P) ** (1.0 / d)


def exponential_power(
    t: np.ndarray | float,
    b: float,
    M: float,
    k: float,
) -> np.ndarray:
    """
    Exponential decay power model:
        P(t) = b + (M - b) exp(-k t)

    Used for sprinters in the report.
    """
    t = np.asarray(t, dtype=float)
    t = np.maximum(t, 0.0)
    return b + (M - b) * np.exp(-k * t)


def exponential_duration(
    P: np.ndarray | float,
    b: float,
    M: float,
    k: float,
) -> np.ndarray:
    """
    Inverse exponential model:
        Δt(P) = (1/k) log((M - b)/(P - b))

    Only valid for b < P < M.
    """
    P = np.asarray(P, dtype=float)
    eps = 1e-8
    P = np.clip(P, b + eps, M - eps)
    return (1.0 / k) * np.log((M - b) / (P - b))


def composite_duration(
    P: np.ndarray | float,
    a: float,
    d: float,
    b: float,
    M: float,
    k: float,
    alpha: float,
) -> np.ndarray:
    """
    Composite duration model for time-trialists:
        Δt(P) = alpha * (a/P)^{1/d} + (1-alpha) * (1/k) log((M-b)/(P-b))
    """
    P = np.asarray(P, dtype=float)
    return alpha * hyperbolic_duration(P, a, d) + (1.0 - alpha) * exponential_duration(P, b, M, k)


def duration_at_power(P: np.ndarray | float, rider: RiderArchetype) -> np.ndarray:
    """
    Unified interface for Δt(P).
    """
    if rider.model_type == "hyperbolic":
        if rider.a is None or rider.d is None:
            raise ValueError(f"Missing hyperbolic parameters for rider '{rider.name}'.")
        return hyperbolic_duration(P, rider.a, rider.d)

    if rider.model_type == "exponential":
        if rider.b is None or rider.M is None or rider.k is None:
            raise ValueError(f"Missing exponential parameters for rider '{rider.name}'.")
        return exponential_duration(P, rider.b, rider.M, rider.k)

    if rider.model_type == "composite":
        if None in (rider.a, rider.d, rider.b, rider.M, rider.k, rider.alpha):
            raise ValueError(f"Missing composite parameters for rider '{rider.name}'.")
        return composite_duration(
            P=P,
            a=rider.a,
            d=rider.d,
            b=rider.b,
            M=rider.M,
            k=rider.k,
            alpha=rider.alpha,
        )

    raise ValueError(f"Unknown rider model_type '{rider.model_type}'.")


def power_curve(t: np.ndarray | float, rider: RiderArchetype) -> np.ndarray:
    """
    Return a representative power curve P(t) for plotting / inspection.

    For the composite rider, we return a weighted combination of the two underlying
    power curves as a practical visualisation choice.
    """
    if rider.model_type == "hyperbolic":
        if rider.a is None or rider.d is None:
            raise ValueError(f"Missing hyperbolic parameters for rider '{rider.name}'.")
        return hyperbolic_power(t, rider.a, rider.d)

    if rider.model_type == "exponential":
        if rider.b is None or rider.M is None or rider.k is None:
            raise ValueError(f"Missing exponential parameters for rider '{rider.name}'.")
        return exponential_power(t, rider.b, rider.M, rider.k)

    if rider.model_type == "composite":
        if None in (rider.a, rider.d, rider.b, rider.M, rider.k, rider.alpha):
            raise ValueError(f"Missing composite parameters for rider '{rider.name}'.")
        t = np.asarray(t, dtype=float)
        return (
            rider.alpha * hyperbolic_power(t, rider.a, rider.d)
            + (1.0 - rider.alpha) * exponential_power(t, rider.b, rider.M, rider.k)
        )

    raise ValueError(f"Unknown rider model_type '{rider.model_type}'.")


# ---------------------------------------------------------------------
# Default archetypes
#
# These are lightweight reusable parameters guided by the final project
# report.
#
# For the explicit model constants (a, d, b, M, k, alpha), we choose
# practical values consistent with the reported archetype behaviour.
# ---------------------------------------------------------------------

DEFAULT_RIDERS: dict[str, RiderArchetype] = {
    "climber": RiderArchetype(
        name="climber",
        mass=65.0,
        bpo=3.5,
        mpo=17.0,
        model_type="hyperbolic",
        a=17.0,
        d=0.13,
    ),
    "sprinter": RiderArchetype(
        name="sprinter",
        mass=75.0,
        bpo=5.0,
        mpo=20.0,
        model_type="exponential",
        b=5.0,
        M=20.0,
        k=0.002,
    ),
    "time_trialist": RiderArchetype(
        name="time_trialist",
        mass=70.0,
        bpo=4.0,
        mpo=19.0,
        model_type="composite",
        a=19.0,
        d=0.15,
        b=4.0,
        M=19.0,
        k=0.00175,
        alpha=0.7,
    ),
}


def get_rider(name: str) -> RiderArchetype:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in DEFAULT_RIDERS:
        raise ValueError(
            f"Unknown rider '{name}'. Available riders: {sorted(DEFAULT_RIDERS)}"
        )
    return DEFAULT_RIDERS[key]

def customise_rider(
    name: str,
    **overrides: float,
) -> RiderArchetype:
    """
    Return a rider archetype with optional parameter overrides.

    This allows CLI or user-level modification of default parameters
    without mutating the global DEFAULT_RIDERS dictionary.

    Example
    -------
    customise_rider("climber", mass=68.0, a=20.0)
    """
    base = get_rider(name)

    data = base.__dict__.copy()
    data.update(overrides)

    return RiderArchetype(**data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rider model parameter override")

    parser.add_argument("--rider", type=str, default="climber", help="Rider type")
    parser.add_argument("--mass", type=float, help="Override mass")
    parser.add_argument("--bpo", type=float, help="Override base power")
    parser.add_argument("--mpo", type=float, help="Override max power")

    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "rider" and v is not None}

    rider = customise_rider(args.rider, **overrides)

    print("Custom rider:")
    print(rider)