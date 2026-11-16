"""Solve a Hunt duct, with conducting Hartmann walls and insulating side walls, at a fixed flow rate.

Edit the inputs below, then run ``python examples/hunt_example.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import lmhdx

# Inputs: field, wall conductance ratio c = sigma_w t_w / (sigma a), cells and the target throughput.
OUTPUT_DIR = Path("artifacts/examples/hunt")
HARTMANN_NUMBER = 20.0
WALL_CONDUCTANCE_RATIO = 0.05
CELLS = 24
TARGET_FLOW_RATE = 0.05
DUCT_LENGTH = 2.5  # Fully developed segment length, not a complete blanket.

# Set up the duct: the problem is linear, so a flow rate sets the drive exactly, Q = G * drive.
lmhdx.configure(dtype="float64")
problem = lmhdx.duct(
    HARTMANN_NUMBER, CELLS, wall_conductance=WALL_CONDUCTANCE_RATIO, flow_rate=TARGET_FLOW_RATE
)

# Run: one certified conjugate-gradient solve; the flow check tests that the drive meets the target.
print(f"Solving the Hunt duct at Ha = {HARTMANN_NUMBER:g}, c = {WALL_CONDUCTANCE_RATIO:g}...", flush=True)
solution = lmhdx.solve(problem, output=lmhdx.Output(str(OUTPUT_DIR)))
drive, flow_rate = float(solution.pressure_gradient), float(solution.flow_rate)
conductance = flow_rate / drive
flow_error = abs(flow_rate - TARGET_FLOW_RATE) / TARGET_FLOW_RATE
if not flow_error <= 1e-8:
    raise RuntimeError(f"Hunt throughput verification failed: relative flow error {flow_error:g}")
summary = {
    "case": "hunt",
    "wall_model": "conducting Hartmann walls; insulating side walls",
    "design": {
        "steady_residual": float(solution.residual),
        "target_flow_rate": TARGET_FLOW_RATE,
        "flow_rate": flow_rate,
        "drive": drive,
        "flow_per_unit_drive": conductance,
        "drive_derivative_wrt_flow": 1.0 / conductance,
        "relative_flow_error": flow_error,
        "length": DUCT_LENGTH,
        "hydraulic_power": drive * DUCT_LENGTH * flow_rate,
    },
}
summary_path = OUTPUT_DIR / "hunt_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"  residual {float(solution.residual):.2e}, flow error {flow_error:.1e}")
print("Results, in the duct's units (unit density, viscosity and conductivity; lengths in a):")
print(f"  flow rate Q = {flow_rate:.6g}, drive dp/dx = {drive:.6g}, dQ/d(drive) = {conductance:.6g}")
print(f"  hydraulic power over length {DUCT_LENGTH:g}: {summary['design']['hydraulic_power']:.6g}")
print(f"Wrote {summary_path}")
