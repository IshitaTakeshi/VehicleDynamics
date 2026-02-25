"""Physical parameters for the quarter-car suspension model.

Defines two suspension configurations — soft/comfort and stiff/sport —
along with shared simulation constants.
"""

from __future__ import annotations

from typing import TypedDict


class SuspensionParameters(TypedDict):
    """Physical parameters of the 2-DOF quarter-car suspension model.

    :param m2: Sprung mass (car body) in kg.
    :param m1: Unsprung mass (wheel + axle) in kg.
    :param ks: Suspension spring stiffness in N/m.
    :param cs: Suspension damping coefficient in N*s/m.
    :param kt: Tire stiffness in N/m.
    """

    m2: float
    m1: float
    ks: float
    cs: float
    kt: float


SUSPENSION_PARAMETERS: dict[str, SuspensionParameters] = {
    "Set A (Soft/Comfort)": {
        "m2": 300.0,
        "m1": 40.0,
        "ks": 15000.0,
        "cs": 1000.0,
        "kt": 150000.0,
    },
    "Set B (Stiff/Sport)": {
        "m2": 300.0,
        "m1": 40.0,
        "ks": 30000.0,
        "cs": 3000.0,
        "kt": 150000.0,
    },
}

T_START: float = 0.0
T_END: float = 3.0
BUMP_ONSET: float = 0.5
BUMP_HEIGHT: float = 0.05
N_TIME_STEPS: int = 3000
INITIAL_CONDITIONS: list[float] = [0.0, 0.0, 0.0, 0.0]
