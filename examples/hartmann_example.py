"""Solve an insulating Hartmann duct and compare its midplane profile with the Hartmann profile.

Edit the inputs below, then run ``python examples/hartmann_example.py``.
Outputs are written beneath the ignored ``artifacts/`` directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import lmhdx

# Inputs: the Hartmann number, the cells across the duct and where the files go.
OUTPUT_DIR = Path("artifacts/examples/hartmann")
HARTMANN_NUMBER = 20.0
CELLS = 24

# Set up the duct (half-width one, unit properties, unit pressure gradient, field along y).
lmhdx.configure(dtype="float64")
problem = lmhdx.duct(HARTMANN_NUMBER, CELLS)

# Run: one certified conjugate-gradient solve on the staggered core; the Output writes the record,
# the ParaView file and the summary.
print(f"Solving the Hartmann duct at Ha = {HARTMANN_NUMBER:g} on {CELLS} x {CELLS} cells...", flush=True)
solution = lmhdx.solve(problem, output=lmhdx.Output(str(OUTPUT_DIR)))
print(f"  residual {float(solution.residual):.2e}, flow rate {float(solution.flow_rate):.6g}", flush=True)

# Compare the profile along y at the mid-plane z = 0 with 1 - (cosh(Ha y) - 1) / (cosh Ha - 1).
faces = np.asarray(problem.grid.faces[1])
y = 0.5 * (faces[1:] + faces[:-1])
u = np.asarray(solution.velocity[0].data[0])
profile = u[:, CELLS // 2 - 1 : CELLS // 2 + 1].mean(axis=1)
profile /= profile.max()
reference = 1.0 - (np.cosh(HARTMANN_NUMBER * y) - 1.0) / (np.cosh(HARTMANN_NUMBER) - 1.0)
error = profile - reference
summary = {
    "case": "hartmann",
    "analytical_profile": {
        "l2_error": float(np.sqrt(np.mean(error**2))),
        "linf_error": float(abs(error).max()),
    },
    "residual": float(solution.residual),
    "flow_rate": float(solution.flow_rate),
}
summary_path = OUTPUT_DIR / "hartmann_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(
    f"Analytical profile error (velocity / scale): L2 {summary['analytical_profile']['l2_error']:.3e}, Linf {summary['analytical_profile']['linf_error']:.3e}"
)
print(f"Wrote {summary_path}")
