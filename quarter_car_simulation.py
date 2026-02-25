"""
Quarter-Car Suspension Transient Response Simulation
=====================================================
Simulates and compares the step response of a 2-DOF quarter-car model
for two suspension parameter sets:
  - Set A: Soft / Comfort-oriented
  - Set B: Stiff / Sport-oriented

Closes #1
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameter sets
# ---------------------------------------------------------------------------

PARAM_SETS = {
    "Set A (Soft/Comfort)": {
        "m2": 300.0,     # Sprung mass: car body [kg]
        "m1": 40.0,      # Unsprung mass: wheel + axle [kg]
        "ks": 15000.0,   # Suspension spring stiffness [N/m]
        "cs": 1000.0,    # Suspension damping coefficient [N·s/m]
        "kt": 150000.0,  # Tire stiffness (treated as a spring) [N/m]
        "color": "steelblue",
        "linestyle": "-",
    },
    "Set B (Stiff/Sport)": {
        "m2": 300.0,
        "m1": 40.0,
        "ks": 30000.0,
        "cs": 3000.0,
        "kt": 150000.0,
        "color": "tomato",
        "linestyle": "--",
    },
}


# ---------------------------------------------------------------------------
# Road input
# ---------------------------------------------------------------------------

def road_input(t):
    """Step road disturbance: a 5 cm bump that starts at t = 0.5 s."""
    return 0.05 if t >= 0.5 else 0.0


# ---------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------

def quarter_car_ode(t, y, params):
    """
    State-space derivative for the 2-DOF quarter-car model.

    State vector:
        y = [z1, z1_dot, z2, z2_dot]

    Equations of motion:
        Unsprung mass (wheel):
            m1 * z1_ddot = ks*(z2-z1) + cs*(z2_dot-z1_dot) - kt*(z1-z0)
        Sprung mass (body):
            m2 * z2_ddot = -ks*(z2-z1) - cs*(z2_dot-z1_dot)

    The suspension force (ks*(z2-z1) + cs*(z2_dot-z1_dot)) acts upward on
    the wheel and downward on the body.  The tire force kt*(z1-z0) acts
    downward on the wheel, representing the road pushing up on a compressed
    tire.
    """
    z1, z1_dot, z2, z2_dot = y
    z0 = road_input(t)

    m1 = params["m1"]
    m2 = params["m2"]
    ks = params["ks"]
    cs = params["cs"]
    kt = params["kt"]

    suspension_force = ks * (z2 - z1) + cs * (z2_dot - z1_dot)
    tire_force = kt * (z1 - z0)

    z1_ddot = (suspension_force - tire_force) / m1
    z2_ddot = -suspension_force / m2

    return [z1_dot, z1_ddot, z2_dot, z2_ddot]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

T_START, T_END = 0.0, 3.0
T_EVAL = np.linspace(T_START, T_END, 3000)
Y0 = [0.0, 0.0, 0.0, 0.0]  # All displacements and velocities start at rest


def run_simulation(params):
    """Integrate the ODE and return time and derived output signals."""
    sol = solve_ivp(
        fun=quarter_car_ode,
        t_span=(T_START, T_END),
        y0=Y0,
        args=(params,),
        t_eval=T_EVAL,
        method="Radau",   # Implicit solver — handles the stiff tire spring well
        rtol=1e-6,
        atol=1e-9,
    )

    z1 = sol.y[0]
    z2 = sol.y[2]
    z2_dot = sol.y[3]
    z1_dot = sol.y[1]
    t = sol.t

    # Road profile evaluated at each time step
    z0 = np.vectorize(road_input)(t)

    # Sprung mass acceleration: re-derive from the equation of motion
    # z2_ddot = -ks*(z2-z1)/m2 - cs*(z2_dot-z1_dot)/m2
    z2_ddot = (
        -params["ks"] * (z2 - z1) - params["cs"] * (z2_dot - z1_dot)
    ) / params["m2"]

    suspension_stroke = z2 - z1   # Suspension deflection (working space)
    tire_deflection = z1 - z0     # Dynamic tire deflection (road holding)

    return t, z2_ddot, suspension_stroke, tire_deflection


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Quarter-Car Suspension: Transient Step Response", fontsize=14)

    subplot_cfg = [
        {
            "ax": axes[0],
            "key": "z2_ddot",
            "ylabel": "Acceleration (m/s²)",
            "title": "Sprung Mass Acceleration  [Ride Comfort]",
        },
        {
            "ax": axes[1],
            "key": "suspension_stroke",
            "ylabel": "Deflection (m)",
            "title": "Suspension Stroke  z₂ − z₁  [Working Space]",
        },
        {
            "ax": axes[2],
            "key": "tire_deflection",
            "ylabel": "Deflection (m)",
            "title": "Dynamic Tire Deflection  z₁ − z₀  [Road Holding]",
        },
    ]

    for cfg in subplot_cfg:
        ax = cfg["ax"]
        for label, data in results.items():
            p = PARAM_SETS[label]
            ax.plot(
                data["t"],
                data[cfg["key"]],
                label=label,
                color=p["color"],
                linestyle=p["linestyle"],
                linewidth=1.8,
            )
        ax.set_ylabel(cfg["ylabel"])
        ax.set_title(cfg["title"], fontsize=10)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="Bump onset")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("quarter_car_response.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = {}
    for label, params in PARAM_SETS.items():
        t, z2_ddot, suspension_stroke, tire_deflection = run_simulation(params)
        results[label] = {
            "t": t,
            "z2_ddot": z2_ddot,
            "suspension_stroke": suspension_stroke,
            "tire_deflection": tire_deflection,
        }

    plot_results(results)
