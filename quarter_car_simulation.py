"""Quarter-car suspension transient response simulation.

Simulates and compares the step response of a 2-DOF quarter-car model
for a soft/comfort and a stiff/sport suspension configuration.

Run with::

    uv run python quarter_car_simulation.py
"""

from __future__ import annotations

from parameters import SUSPENSION_PARAMETERS
from plot import plot_results
from simulation import SimulationResult, run_simulation

if __name__ == "__main__":
    results: dict[str, SimulationResult] = {
        label: run_simulation(suspension_parameters)
        for label, suspension_parameters in SUSPENSION_PARAMETERS.items()
    }
    plot_results(results)
