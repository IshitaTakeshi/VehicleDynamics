"""Visualization of quarter-car transient response simulation results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from parameters import BUMP_ONSET

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from simulation import SimulationResult

_LINE_COLORS: dict[str, str] = {
    "Set A (Soft/Comfort)": "steelblue",
    "Set B (Stiff/Sport)": "tomato",
}

_LINE_STYLES: dict[str, str] = {
    "Set A (Soft/Comfort)": "-",
    "Set B (Stiff/Sport)": "--",
}

_SUBPLOT_SPECS: list[
    tuple[Callable[[SimulationResult], NDArray[np.float64]], str, str]
] = [
    (
        lambda result: result.sprung_mass_acceleration,
        "Acceleration (m/s\u00b2)",
        "Sprung Mass Acceleration  [Ride Comfort]",
    ),
    (
        lambda result: result.suspension_stroke,
        "Deflection (m)",
        "Suspension Stroke  z\u2082 \u2212 z\u2081  [Working Space]",
    ),
    (
        lambda result: result.tire_deflection,
        "Deflection (m)",
        "Dynamic Tire Deflection  z\u2081 \u2212 z\u2080  [Road Holding]",
    ),
]


def _decorate_axis(axis: Axes, ylabel: str, title: str) -> None:
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=10)
    axis.axvline(
        BUMP_ONSET, color="gray", linestyle=":", linewidth=1, label="Bump onset"
    )
    axis.axhline(0.0, color="black", linewidth=0.5)
    axis.legend(fontsize=8)
    axis.grid(linestyle="--", alpha=0.5)


def _plot_series(
    axis: Axes,
    label: str,
    accessor: Callable[[SimulationResult], NDArray[np.float64]],
    result: SimulationResult,
) -> None:
    axis.plot(
        result.time,
        accessor(result),
        label=label,
        color=_LINE_COLORS[label],
        linestyle=_LINE_STYLES[label],
        linewidth=1.8,
    )


def _configure_subplot(
    axis: Axes,
    accessor: Callable[[SimulationResult], NDArray[np.float64]],
    ylabel: str,
    title: str,
    results: dict[str, SimulationResult],
) -> None:
    for label, result in results.items():
        _plot_series(axis, label, accessor, result)
    _decorate_axis(axis, ylabel, title)


def plot_results(results: dict[str, SimulationResult]) -> None:
    """Create and display the three-subplot transient response figure.

    Saves the figure to ``quarter_car_response.png`` before displaying.

    :param results: Mapping of parameter set label to its simulation result.
    """
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    figure.suptitle("Quarter-Car Suspension: Transient Step Response", fontsize=14)
    for axis, (accessor, ylabel, title) in zip(axes, _SUBPLOT_SPECS, strict=True):
        _configure_subplot(axis, accessor, ylabel, title, results)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("quarter_car_response.png", dpi=150)
    plt.show()
