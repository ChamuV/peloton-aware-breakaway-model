# src/track_profiles.py
from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def flat_theta(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.zeros_like(x)


def elevation_theta(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.arcsin(0.01) * np.ones_like(x)


def hill_theta(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.arcsin(-0.04 * (x - 0.5))


def steep_hill_theta(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.arcsin(-0.2 * (x - 0.5))


def sharp_hill_theta(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.arcsin(-0.09 * (x - 0.5))


def double_hill_theta(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    inner = (
        -4.0 * (x - 0.3) * np.exp(-((x - 0.3) ** 2) / 0.01)
        -6.0 * (x - 0.7) * np.exp(-((x - 0.7) ** 2) / 0.01)
    )
    return np.arcsin(inner)


def uphill_1a_theta(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    theta = np.zeros_like(x)
    theta[(x >= 0.2) & (x < 0.4)] = 0.01
    return theta


def uphill_1b_theta(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    theta = np.zeros_like(x)
    theta[(x >= 0.4) & (x < 0.6)] = 0.01
    return theta


def uphill_1c_theta(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    theta = np.zeros_like(x)
    theta[(x >= 0.6) & (x < 0.8)] = 0.01
    return theta


def uphill_1d_theta(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    theta = np.zeros_like(x)
    theta[(x >= 0.8) & (x <= 1.0)] = 0.01
    return theta


def updown_1_theta(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    theta = np.zeros_like(x)
    theta[(x >= 0.3) & (x < 0.4)] = 0.005
    theta[(x >= 0.5) & (x < 0.6)] = -0.005
    return theta


def updown_2_theta(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    theta = np.zeros_like(x)
    theta[(x >= 0.6) & (x < 0.7)] = 0.005
    theta[(x >= 0.8) & (x <= 0.9)] = -0.005
    return theta


def mountainous_theta(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    inner = (
        -2.5 * (x - 0.22) * np.exp(-((x - 0.22) ** 2) / 0.003)
        -3.5 * (x - 0.5) * np.exp(-((x - 0.5) ** 2) / 0.0065)
        -2.8 * (x - 0.76) * np.exp(-((x - 0.76) ** 2) / 0.005)
    )
    return np.arcsin(inner)


TRACK_FUNCTIONS = {
    "flat": flat_theta,
    "elevation": elevation_theta,
    "hill": hill_theta,
    "steep_hill": steep_hill_theta,
    "steep hill": steep_hill_theta,
    "sharp_hill": sharp_hill_theta,
    "sharp hill": sharp_hill_theta,
    "double_hill": double_hill_theta,
    "double-hill": double_hill_theta,
    "uphill_1a": uphill_1a_theta,
    "uphill_1b": uphill_1b_theta,
    "uphill_1c": uphill_1c_theta,
    "uphill_1d": uphill_1d_theta,
    "updown_1": updown_1_theta,
    "updown_2": updown_2_theta,
    "mountainous": mountainous_theta,
    "hills_and_valleys": mountainous_theta,
    "hills and valleys": mountainous_theta,
}


def compute_theta(track_name: str, x: np.ndarray | float) -> np.ndarray:
    key = track_name.strip().lower()
    key = key.replace("-", "_")
    if key not in TRACK_FUNCTIONS:
        raise ValueError(
            f"Unknown track name '{track_name}'. "
            f"Available tracks: {sorted(TRACK_FUNCTIONS)}"
        )
    return TRACK_FUNCTIONS[key](x)


def elevation_profile(
    track_name: str,
    x: np.ndarray,
    length_scale: float = 100000.0,
    make_nonnegative: bool = True,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")

    theta_func = lambda xx: compute_theta(track_name, xx)
    elevation = np.zeros_like(x)

    for i, xi in enumerate(x):
        elevation[i], _ = quad(lambda xx: np.sin(theta_func(xx)), 0.0, xi)

    elevation = elevation * length_scale

    if make_nonnegative:
        elevation = elevation - np.min(elevation)

    return elevation