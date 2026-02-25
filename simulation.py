"""Numerical integration of the quarter-car equations of motion.

Uses scipy's Radau implicit solver, which handles the stiffness introduced
by the large tire spring constant (kt = 150,000 N/m).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

from model import quarter_car_ode
from parameters import (
    BUMP_HEIGHT,
    BUMP_ONSET,
    INITIAL_CONDITIONS,
    N_TIME_STEPS,
    T_END,
    T_START,
    SuspensionParameters,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclasses.dataclass(frozen=True)
class SimulationResult:
    """Output signals extracted from the ODE solution.

    :param time: Time vector in seconds.
    :param sprung_mass_acceleration: Body vertical acceleration in m/s².
    :param suspension_stroke: Suspension deflection (z2 - z1) in metres.
    :param tire_deflection: Dynamic tire deflection (z1 - z0) in metres.
    """

    time: NDArray[np.float64]
    sprung_mass_acceleration: NDArray[np.float64]
    suspension_stroke: NDArray[np.float64]
    tire_deflection: NDArray[np.float64]


def _build_result(
    time: NDArray[np.float64],
    states: NDArray[np.float64],
    parameters: SuspensionParameters,
) -> SimulationResult:
    """Derive physical output signals from the raw ODE solution arrays.

    :param time: Time vector from the ODE solver in seconds.
    :param states: State matrix (4 x N) from the ODE solver.
    :param parameters: Physical parameters used during integration.
    :returns: Structured simulation output signals.
    """
    z1, z1_dot, z2, z2_dot = states[0], states[1], states[2], states[3]
    road_profile: NDArray[np.float64] = np.where(time >= BUMP_ONSET, BUMP_HEIGHT, 0.0)
    suspension_force = (
        parameters["ks"] * (z2 - z1) + parameters["cs"] * (z2_dot - z1_dot)
    )
    return SimulationResult(
        time=time,
        sprung_mass_acceleration=-suspension_force / parameters["m2"],
        suspension_stroke=z2 - z1,
        tire_deflection=z1 - road_profile,
    )


def run_simulation(parameters: SuspensionParameters) -> SimulationResult:
    """Integrate the quarter-car ODE and return computed response signals.

    :param parameters: Physical suspension parameters to simulate.
    :returns: Time-domain response signals for the given parameter set.
    """
    time_evaluation = np.linspace(T_START, T_END, N_TIME_STEPS)
    solution = solve_ivp(
        fun=quarter_car_ode,
        t_span=(T_START, T_END),
        y0=INITIAL_CONDITIONS,
        t_eval=time_evaluation,
        args=(parameters,),
        method="Radau",
        rtol=1e-6,
        atol=1e-9,
    )
    return _build_result(solution.t, solution.y, parameters)
