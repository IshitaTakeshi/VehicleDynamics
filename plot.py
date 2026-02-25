"""Visualization of quarter-car transient response simulation results."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt

from parameters import BUMP_ONSET, PARAMETER_SETS

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from simulation import SimulationResult


class SubplotConfig(TypedDict):
    """Configuration for one subplot in the results figure.

    :param accessor: Extracts the relevant signal array from a SimulationResult.
    :param ylabel: Y-axis label string.
    :param title: Subplot title string.
    """

    accessor: Callable[[SimulationResult], NDArray[np.float64]]
    ylabel: str
    title: str


SUBPLOT_CONFIGS: list[SubplotConfig] = [
    {
        "accessor": lambda result: result.sprung_mass_acceleration,
        "ylabel": "Acceleration (m/s\u00b2)",
        "title": "Sprung Mass Acceleration  [Ride Comfort]",
    },
    {
        "accessor": lambda result: result.suspension_stroke,
        "ylabel": "Deflection (m)",
        "title": "Suspension Stroke  z\u2082 \u2212 z\u2081  [Working Space]",
    },
    {
        "accessor": lambda result: result.tire_deflection,
        "ylabel": "Deflection (m)",
        "title": "Dynamic Tire Deflection  z\u2081 \u2212 z\u2080  [Road Holding]",
    },
]


def _decorate_axis(axis: Axes, configuration: SubplotConfig) -> None:
    """Apply labels, reference lines, legend, and grid to a subplot axis.

    :param axis: The matplotlib Axes object to decorate.
    :param configuration: Subplot configuration providing label and title strings.
    """
    axis.set_ylabel(configuration["ylabel"])
    axis.set_title(configuration["title"], fontsize=10)
    axis.axvline(
        BUMP_ONSET, color="gray", linestyle=":", linewidth=1, label="Bump onset"
    )
    axis.axhline(0.0, color="black", linewidth=0.5)
    axis.legend(fontsize=8)
    axis.grid(linestyle="--", alpha=0.5)


def _configure_subplot(
    axis: Axes,
    configuration: SubplotConfig,
    results: dict[str, SimulationResult],
) -> None:
    """Plot all parameter sets onto a single subplot axis.

    :param axis: The matplotlib Axes object to draw on.
    :param configuration: Subplot configuration including the signal accessor.
    :param results: Mapping of parameter set label to its simulation result.
    """
    for label, result in results.items():
        plot_style = PARAMETER_SETS[label]["style"]
        axis.plot(
            result.time,
            configuration["accessor"](result),
            label=label,
            color=plot_style["color"],
            linestyle=plot_style["linestyle"],
            linewidth=1.8,
        )
    _decorate_axis(axis, configuration)


def plot_results(results: dict[str, SimulationResult]) -> None:
    """Create and display the three-subplot transient response figure.

    Saves the figure to ``quarter_car_response.png`` before displaying.

    :param results: Mapping of parameter set label to its simulation result.
    """
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    figure.suptitle("Quarter-Car Suspension: Transient Step Response", fontsize=14)
    for axis, configuration in zip(axes, SUBPLOT_CONFIGS, strict=True):
        _configure_subplot(axis, configuration, results)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("quarter_car_response.png", dpi=150)
    plt.show()
