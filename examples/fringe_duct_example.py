"""Flow through a square duct that leaves a magnet: the ANL fringe with an inlet and an outlet.

The duct carries a fixed flow rate from a uniform field ``Ha`` upstream through
the fringe ``B_y = Ha (1 - sin(pi x / 2 x0)) / 2`` into a field-free outlet
(ANL/FPP/TM-228). The inlet profile is LMhdX's own fully developed flow, so the
upstream gradient is the fully developed one; the drop across the fringe is an
output. The solve is the inertialess (Stokes-limit) steady flow, and the drop is
differentiated with respect to the field strength.

Edit the inputs below, then run ``python examples/fringe_duct_example.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

from lmhdx.axial import (
    charge_balance,
    fringe_duct,
    fully_developed_inlet,
    mass_balance,
    pressure_drop,
    solve_open_duct,
    station_flow_rates,
    station_pressure,
)

# Inputs: field, walls, mesh, and outputs. Lengths are in duct half-widths.
OUTPUT_DIR = Path("artifacts/examples/fringe_duct")
HARTMANN_NUMBER = 20.0  # upstream of the magnet edge
WALL_CONDUCTANCE_RATIO = 0.02  # all four walls
HALF_LENGTH = 3.0  # x0: the fringe spans -x0 <= x <= x0
UPSTREAM = 6.0  # uniform-field buffer before the fringe
DOWNSTREAM = 3.0  # field-free buffer after it
AXIAL_SPACING = 0.5
CELLS = 12  # across the duct
CELLS_IN_LAYER = 3
FLOW_RATE = 4.0  # a unit mean velocity
FINITE_DIFFERENCE_STEP = 1e-4
WRITE_PLOTS = True


# Run the solve. The inlet profile is solved first on the duct's own cross-section.
problem = fringe_duct(
    hartmann=HARTMANN_NUMBER,
    wall_conductance=WALL_CONDUCTANCE_RATIO,
    half_length=HALF_LENGTH,
    upstream=UPSTREAM,
    downstream=DOWNSTREAM,
    spacing=AXIAL_SPACING,
    cells=CELLS,
    cells_in_layer=CELLS_IN_LAYER,
    flow_rate=FLOW_RATE,
)
print(
    f"Fringe duct, Ha = {HARTMANN_NUMBER:g}, grid {problem.grid.shape}: fully developed inlet...", flush=True
)
_, inlet_gradient = fully_developed_inlet(problem, FLOW_RATE)
print("  open-duct steady solve...", flush=True)
solution = solve_open_duct(problem)

# Check what the solve guarantees: the flow rate through every station, and mass
# and charge conserved in every cell, all to round-off.
flow_error = float(np.max(np.abs(np.asarray(station_flow_rates(solution.velocity)) - FLOW_RATE)))
checks = {
    "relative_residual": float(solution.residual_norm / solution.initial_residual_norm),
    "cg_iterations": int(solution.iterations),
    "relative_flow_rate_error": flow_error / FLOW_RATE,
    "mass_balance": float(mass_balance(solution.velocity)),
    "charge_balance": float(charge_balance(solution, problem)),
}
if not (checks["relative_residual"] <= 1e-9 and max(list(checks.values())[2:]) < 1e-10):
    raise RuntimeError(f"fringe duct verification failed: {checks}")

# The upstream gradient is the fully developed one, and the drop across the fringe
# exceeds what the local fully developed gradient alone would give.
centres, means = station_pressure(solution.pressure)
means = np.asarray(means)
upstream = (centres > -HALF_LENGTH - UPSTREAM + 1.0) & (centres < -HALF_LENGTH - 2.0)
upstream_gradient = float(np.polyfit(centres[upstream], means[upstream], 1)[0])
drop = float(pressure_drop(solution.pressure, -HALF_LENGTH, HALF_LENGTH))


def fringe_drop(field_scale):
    """Pressure drop across the fringe at a multiple of the field; differentiable."""
    fields = solve_open_duct(problem, field_scale=field_scale)
    return pressure_drop(fields.pressure, -HALF_LENGTH, HALF_LENGTH)


# The derivative of the drop with respect to the field strength is one adjoint
# solve; central differences check it.
print("  adjoint derivative of the drop, then two finite-difference solves...", flush=True)
derivative = float(jax.grad(fringe_drop)(1.0))
step = FINITE_DIFFERENCE_STEP
central = (float(fringe_drop(1.0 + step)) - float(fringe_drop(1.0 - step))) / (2.0 * step)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plots = []
if WRITE_PLOTS:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(centres, means, marker=".")
    ax.axvspan(-HALF_LENGTH, HALF_LENGTH, alpha=0.15, label="fringe")
    ax.set_xlabel("x / half-width")
    ax.set_ylabel("mean pressure")
    ax.legend()
    fig.tight_layout()
    plots.append(OUTPUT_DIR / "fringe_duct_pressure.png")
    fig.savefig(plots[-1], dpi=150)

summary = {
    "hartmann_number": HARTMANN_NUMBER,
    "wall_conductance_ratio": WALL_CONDUCTANCE_RATIO,
    "grid_shape": list(problem.grid.shape),
    "checks": checks,
    "fully_developed_inlet_gradient": inlet_gradient,
    "upstream_gradient": upstream_gradient,
    "upstream_gradient_relative_error": abs(upstream_gradient / inlet_gradient - 1.0),
    "fringe_pressure_drop": drop,
    "drop_derivative_wrt_field_scale": derivative,
    "central_difference": central,
    "derivative_relative_error": abs(derivative - central) / abs(central),
    "plots": [path.name for path in plots],
}
(OUTPUT_DIR / "fringe_duct_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("Results (half-width, density, viscosity and conductivity are 1; pressure in mu U / a):")
print(
    f"  residual {checks['relative_residual']:.1e}, {checks['cg_iterations']} iterations, flow error "
    f"{checks['relative_flow_rate_error']:.1e}, mass {checks['mass_balance']:.1e}, charge {checks['charge_balance']:.1e}"
)
print(f"  upstream gradient {upstream_gradient:.6g} per half-width (fully developed {inlet_gradient:.6g})")
print(
    f"  fringe pressure drop {drop:.6g}; d(drop)/d(field scale) {derivative:.6g} (central difference {central:.6g})"
)
print(f"Wrote {OUTPUT_DIR / 'fringe_duct_summary.json'}")
