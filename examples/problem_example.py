"""Solve a Hartmann duct with ``Problem`` and differentiate the pressure gradient with respect to the field.

Edit the inputs below, then run ``python examples/problem_example.py``.
"""

import jax

import lmhdx

# Inputs: the duct (half-width one, unit properties), its drive and the numerics.
HARTMANN_NUMBER = 20.0
CELLS = 48
WALL_CONDUCTANCE = 0.0  # a Hunt duct has conducting Hartmann walls, for example 0.05
FLOW_RATE = 4.0  # the volumetric flux of a duct of unit half-width at unit mean velocity

# Set up the problem. Float64 is chosen once and explicitly; importing LMhdX changes nothing.
lmhdx.configure(dtype="float64")
problem = lmhdx.duct(HARTMANN_NUMBER, CELLS, wall_conductance=WALL_CONDUCTANCE, flow_rate=FLOW_RATE)

# Run: one certified conjugate-gradient solve on the staggered core.
print(f"Solving the duct at Ha = {HARTMANN_NUMBER:g} on {CELLS} x {CELLS} cells...", flush=True)
solution = lmhdx.solve(problem)
print(f"flow rate         = {float(solution.flow_rate):.6f}")
print(f"pressure gradient = {float(solution.pressure_gradient):.6f}")
print(f"residual norm   = {float(solution.residual):.2e}")

# The leaves of a Problem (drive, flow rate, field amplitude) differentiate through the implicit solve.
gradient = jax.grad(lambda p: lmhdx.solve(p).pressure_gradient)(problem)
print(f"d(pressure gradient)/d(field strength) = {float(gradient.B.strength):.6f}")
print(f"d(pressure gradient)/d(flow rate)      = {float(gradient.flow_rate):.6f}")
