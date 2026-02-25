"""Quarter-car suspension transient response simulation.

Simulates and compares the step response of a 2-DOF quarter-car model
for a soft/comfort and a stiff/sport suspension configuration.

Run with::

    uv run python quarter_car_simulation.py
"""

from __future__ import annotations

from parameters import PARAMETER_SETS
from plot import plot_results
from simulation import SimulationResult, run_simulation

if __name__ == "__main__":
    results: dict[str, SimulationResult] = {
        label: run_simulation(parameter_set["suspension"])
        for label, parameter_set in PARAMETER_SETS.items()
    }
    plot_results(results)
