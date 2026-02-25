"""Equations of motion for the 2-DOF quarter-car suspension model.

State vector: [z1, z1_dot, z2, z2_dot]
  - z1: unsprung mass (wheel) vertical displacement [m]
  - z2: sprung mass (body) vertical displacement [m]
  - z0: road surface height [m]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from parameters import BUMP_HEIGHT, BUMP_ONSET, SuspensionParameters

if TYPE_CHECKING:
    from numpy.typing import NDArray


def road_input(time: float) -> float:
    """Return the road surface height at a given simulation time.

    Models a 5 cm step bump that begins at t = 0.5 s.

    :param time: Simulation time in seconds.
    :returns: Road surface height in metres.
    """
    if time >= BUMP_ONSET:
        return BUMP_HEIGHT
    return 0.0


def quarter_car_ode(
    time: float,
    state: NDArray[np.float64],
    parameters: SuspensionParameters,
) -> NDArray[np.float64]:
    """Compute the state-space derivative for the quarter-car model.

    Equations of motion:
      Unsprung: m1*z1_ddot = ks*(z2-z1) + cs*(z2_dot-z1_dot) - kt*(z1-z0)
      Sprung:   m2*z2_ddot = -ks*(z2-z1) - cs*(z2_dot-z1_dot)

    :param time: Current simulation time in seconds.
    :param state: Current state vector [z1, z1_dot, z2, z2_dot].
    :param parameters: Physical parameters of the suspension model.
    :returns: State derivative [z1_dot, z1_ddot, z2_dot, z2_ddot].
    """
    z1, z1_dot, z2, z2_dot = state[0], state[1], state[2], state[3]
    z0 = road_input(time)
    suspension_force = (
        parameters["ks"] * (z2 - z1) + parameters["cs"] * (z2_dot - z1_dot)
    )
    tire_force = parameters["kt"] * (z1 - z0)
    z1_ddot = (suspension_force - tire_force) / parameters["m1"]
    z2_ddot = -suspension_force / parameters["m2"]
    return np.array([z1_dot, z1_ddot, z2_dot, z2_ddot])
