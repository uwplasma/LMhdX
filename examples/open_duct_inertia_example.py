"""Check whether inertia matters in a fringing flow with the open-duct solve.

``solve_open_duct`` solves a duct with an inlet and an outlet. Without advection
it is the inertialess (Stokes) flow; with ``advection="central"`` the flow is
marched from that one to its steady state by the Runge-Kutta stepper, at a step
that keeps the transport and the Lorentz damping stable. It is worth using to
check the inertial share: the share below is the change of the fringe pressure
drop when transport is switched on. Keep ``growth=1.0`` (small cell Reynolds
number). This Ha 30 case is cheap; the measured share at Ha 100-200 and Re
10-40 is about 0.045% Re^1.31 Ha^-0.16 of the drop. Check ``status``: only
``CONVERGED`` (0) is a steady flow.

Edit the inputs below, then run ``python examples/open_duct_inertia_example.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from lmhdx.axial import fringe_duct, pressure_drop, solve_open_duct

# Inputs: field, flow, mesh, and outputs. Lengths are in duct half-widths.
OUTPUT_DIR = Path("artifacts/examples/open_duct_inertia")
HARTMANN_NUMBER = 30.0
REYNOLDS_NUMBERS = (5.0, 10.0)  # mean velocity = flow_rate / 4
HALF_LENGTH = 3.0
UPSTREAM = 6.0
DOWNSTREAM = 3.0
AXIAL_SPACING = 0.5
CELLS = 12  # across the duct
CELLS_IN_LAYER = 3


def build(reynolds, advection):
    """The fringe duct at a Reynolds number, with or without transport."""
    return fringe_duct(
        hartmann=HARTMANN_NUMBER,
        half_length=HALF_LENGTH,
        upstream=UPSTREAM,
        downstream=DOWNSTREAM,
        spacing=AXIAL_SPACING,
        cells=CELLS,
        cells_in_layer=CELLS_IN_LAYER,
        flow_rate=4.0 * reynolds,
        advection=advection,
        growth=1.0,
    )


# Run the Stokes-limit solve and the march at each Reynolds number.
rows = []
for reynolds in REYNOLDS_NUMBERS:
    print(f"Ha = {HARTMANN_NUMBER:g}, Re = {reynolds:g}: Stokes, then the march...", flush=True)
    start = time.perf_counter()
    stokes = solve_open_duct(build(reynolds, "off"))
    full = solve_open_duct(build(reynolds, "central"))
    drop_stokes = float(pressure_drop(stokes.pressure, -HALF_LENGTH, HALF_LENGTH))
    drop_full = float(pressure_drop(full.pressure, -HALF_LENGTH, HALF_LENGTH))
    rows.append(
        {
            "reynolds": reynolds,
            "drop_stokes": drop_stokes,
            "drop_with_transport": drop_full,
            "inertial_share_percent": 100.0 * abs(drop_full - drop_stokes) / abs(drop_stokes),
            "march_steps": int(full.iterations),
            "status": int(full.status),
            "relative_residual": float(full.residual_norm / full.initial_residual_norm),
            "seconds": time.perf_counter() - start,
        }
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary = {"hartmann_number": HARTMANN_NUMBER, "rows": rows}
(OUTPUT_DIR / "open_duct_inertia_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("Results (half-width, density, viscosity and conductivity are 1):")
for row in rows:
    print(
        f"  Re {row['reynolds']:g}: drop {row['drop_stokes']:.6g} -> {row['drop_with_transport']:.6g}, "
        f"inertial share {row['inertial_share_percent']:.3f}%, {row['march_steps']} march steps, "
        f"{row['seconds']:.0f} s"
    )
print(f"Wrote {OUTPUT_DIR / 'open_duct_inertia_summary.json'}")
