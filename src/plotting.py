# src/plotting.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.track_profiles import elevation_profile


RIDER_COLORS = {
    "sprinter": "red",
    "time_trialist": "green",
    "climber": "blue",
}

RIDER_LABELS = {
    "sprinter": "Sprinter",
    "time_trialist": "Time - Trialist",
    "climber": "Climber",
}


def apply_plot_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (14, 7),
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 15,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
    })


def plot_finishing_time_vs_breakaway(
    results_by_mode: dict,
    peloton_finish_times: dict,
    title: str = "Finishing Time vs Breakaway Point",
):
    apply_plot_style()
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    for i, mode in enumerate(["passive", "strategic"]):
        ax = axs[i]
        for rider_key in ["sprinter", "time_trialist", "climber"]:
            rider_data = results_by_mode[mode][rider_key]
            x_vals = np.asarray(rider_data["x_break"], dtype=float)
            t_vals = np.asarray(rider_data["T_break"], dtype=float)
            valid = ~np.isnan(x_vals) & np.isfinite(t_vals)

            ax.plot(
                x_vals[valid],
                t_vals[valid],
                marker="o",
                color=RIDER_COLORS[rider_key],
                label=RIDER_LABELS[rider_key],
            )

        peloton_time = peloton_finish_times[mode]
        ax.axhline(peloton_time, color="black", linestyle="--", linewidth=1.5)
        ax.text(
            0.02,
            peloton_time,
            "Peloton Finish",
            fontsize=16,
            color="black",
            ha="left",
            va="bottom",
            transform=ax.get_yaxis_transform(),
        )

        ax.set_title("Passive Peloton" if mode == "passive" else "Strategic Peloton", fontsize=22)
        ax.set_xlabel(r"Breakaway Point $x_b$ (%)", fontsize=17)
        ax.grid(True)

        if i == 0:
            ax.set_ylabel("Finishing Time (minutes)", fontsize=17)

    fig.suptitle(title, fontsize=28, y=0.98)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=18, frameon=False)
    plt.subplots_adjust(top=0.88, bottom=0.15, wspace=0.1)
    return fig, axs


def plot_best_breakaway_points_on_track(
    track_name: str,
    x_grid: np.ndarray,
    best_points_by_mode: dict,
    length_scale: float = 100000.0,
    title: str = "Best Breakaway Points on Track",
):
    apply_plot_style()
    elevation = elevation_profile(track_name, x_grid, length_scale=length_scale)
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    for i, mode in enumerate(["passive", "strategic"]):
        ax = axs[i]
        ax.plot(x_grid, elevation, color="black", linewidth=2)

        used_positions = []
        for rider_key in ["sprinter", "time_trialist", "climber"]:
            rider_point = best_points_by_mode[mode][rider_key]
            xb = float(rider_point["x_break"])
            power_wkg = float(rider_point["power_wkg"])
            yb = np.interp(xb, x_grid, elevation)

            ax.plot(xb, yb, marker="o", color=RIDER_COLORS[rider_key], markersize=10)

            close_count = sum(abs(xb - xp) < 0.035 for xp in used_positions)
            used_positions.append(xb)

            dx = -0.06 + 0.03 * close_count
            dy = (0.04 + 0.03 * close_count) * (np.max(elevation) - np.min(elevation))

            ax.text(
                xb + dx,
                yb + dy,
                f"{power_wkg:.1f} W/kg",
                fontsize=15,
                color=RIDER_COLORS[rider_key],
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title("Passive Peloton" if mode == "passive" else "Strategic Peloton", fontsize=22)
        ax.set_xlabel("Track Position", fontsize=17)
        ax.grid(True)

        if i == 0:
            ax.set_ylabel("Elevation (m)", fontsize=17)

    fig.suptitle(title, fontsize=28, y=0.98)
    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=RIDER_COLORS[r],
            markersize=10,
            label=RIDER_LABELS[r],
        )
        for r in ["sprinter", "time_trialist", "climber"]
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=18, frameon=False)
    plt.subplots_adjust(top=0.88, bottom=0.15, wspace=0.1)
    return fig, axs