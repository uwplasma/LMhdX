"""Solve a Hunt duct with conducting Hartmann and insulating side walls.

Edit the inputs below, then run ``python examples/hunt_example.py``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from lmhdx import make_hunt_case, solve
from lmhdx.fully_developed import hydraulic_power, volumetric_flow_rate
from lmhdx.io import validation_summary, write_case_overview_plots, write_solution_outputs

# Inputs: geometry, wall model, material properties, numerics, and outputs.
OUTPUT_DIR = Path("artifacts/examples/hunt")
HARTMANN_NUMBER = 20.0
WIDTH = 2.0
HEIGHT = 2.0
NY = 24
NZ = 24
WALL_THICKNESS = 0.1
WALL_CONDUCTANCE_RATIO = 0.05
FLUID_CONDUCTIVITY = 1.0
DENSITY = 1.0
VISCOSITY = 1.0
FORCING = 1.0
TARGET_FLOW_RATE = 0.05  # Set to None to prescribe FORCING instead.
DUCT_LENGTH = 2.5  # Fully developed segment length, not a complete blanket.
WRITE_PARAVIEW = True
WRITE_CSV = True
WRITE_NPZ = True
WRITE_PLOTS = True


# Set up the duct. The conducting Hartmann walls are thin walls of conductance
# ratio c = sigma_w t_w / (sigma a) on the staggered core; the side walls insulate.
case = make_hunt_case(
    ha=HARTMANN_NUMBER,
    width=WIDTH,
    height=HEIGHT,
    ny=NY,
    nz=NZ,
    wall_thickness=WALL_THICKNESS,
    fluid_conductivity=FLUID_CONDUCTIVITY,
    wall_conductance_ratio=WALL_CONDUCTANCE_RATIO,
    density=DENSITY,
    viscosity=VISCOSITY,
    output_dir=str(OUTPUT_DIR),
)
case = replace(
    case,
    forcing=FORCING,
    output=replace(
        case.output,
        write_paraview=WRITE_PARAVIEW,
        write_csv_profiles=WRITE_CSV,
        write_npz=WRITE_NPZ,
        write_plots=WRITE_PLOTS,
    ),
)

# The problem is linear, so a unit-drive solve gives the flow per unit drive:
# Q = G * drive. That eliminates the drive from a fixed-throughput design.
print(
    f"Solving the Hunt duct at Ha = {HARTMANN_NUMBER:g}, c = {WALL_CONDUCTANCE_RATIO:g}: unit-drive solve...",
    flush=True,
)
unit = solve(replace(case, forcing=1.0))
conductance = float(volumetric_flow_rate(case, unit.state.u))
if TARGET_FLOW_RATE is not None:
    case = replace(case, forcing=TARGET_FLOW_RATE / conductance)

# Run the design solve at that drive. Each steady solve is one conjugate-gradient
# solve whose residual is certified, and the second reuses the compiled program of
# the first because the drive is an argument; the flow check tests Q = G * drive.
print(f"  {unit.status}, residual {unit.residual:.2e}; design solve at the target flow rate...", flush=True)
solution = solve(case)
flow_rate = float(volumetric_flow_rate(case, solution.state.u))
expected_flow = conductance * case.forcing
flow_error = abs(flow_rate - expected_flow) / max(abs(expected_flow), 1e-30)
if not (unit.converged and solution.converged) or not flow_error <= 1e-8:
    raise RuntimeError(
        f"Hunt throughput verification failed: status={solution.status}, "
        f"residual={solution.residual:g}, relative flow error={flow_error:g}, "
        f"charge balance={float(solution.diagnostics.div_current_max_history[-1]):g}"
    )
generated = write_solution_outputs(solution, case, OUTPUT_DIR)
plots = (
    write_case_overview_plots(
        solution,
        OUTPUT_DIR,
        case_title=f"Hunt duct (Ha={HARTMANN_NUMBER:g})",
    )
    if WRITE_PLOTS
    else []
)
summary = {
    "case": case.name,
    "wall_model": "conducting Hartmann walls; insulating side walls",
    "design": {
        "verification": "cold certified steady solve",
        "status": solution.status,
        "steady_residual": solution.residual,
        "target_flow_rate": TARGET_FLOW_RATE,
        "flow_rate": flow_rate,
        "drive": case.forcing,
        "flow_per_unit_drive": conductance,
        "drive_derivative_wrt_flow": 1.0 / conductance,
        "relative_flow_error": flow_error,
        "length": DUCT_LENGTH,
        "hydraulic_power": float(hydraulic_power(case.forcing, flow_rate, DUCT_LENGTH)),
    },
    "validation": validation_summary(solution, case.name, HARTMANN_NUMBER),
    "generated_files": {
        **{kind: [path.name for path in paths] for kind, paths in generated.items()},
        "plots": [path.name for path in plots],
    },
}
summary_path = OUTPUT_DIR / "hunt_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
design = summary["design"]
print(f"  {solution.status}, residual {solution.residual:.2e}, flow error {flow_error:.1e}")
print("Results, in the case's units (unit density, viscosity and conductivity; lengths in a):")
print(f"  flow rate Q = {flow_rate:.6g}, drive dp/dx = {case.forcing:.6g}, dQ/d(drive) = {conductance:.6g}")
print(f"  hydraulic power over length {DUCT_LENGTH:g}: {design['hydraulic_power']:.6g}")
print(f"Wrote {summary_path}")
