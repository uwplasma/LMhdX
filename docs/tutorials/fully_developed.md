# Fully developed duct flows

Far from its ends, the flow in a long straight duct stops changing along the
duct. Such a flow is fully developed. LMhdX then solves a single cross-section
for the axial velocity $u(y,z)$ and the electric potential $\phi(y,z)$. This
page solves a duct with insulating walls and one with conducting walls, sets a
flow rate and measures the pumping power, and fits a measured velocity profile.

```{image} ../_static/hunt_side_layers.webp
:alt: Hunt's duct from Ha 20 to 1000, the jets along the side walls growing with the field
:align: center
```

The figure shows Hunt's duct. As the field grows, jets form along the side walls.

## Solve a duct

The mesh resolves the Hartmann and side layers as {ref}`wall-resolving coordinates <numerics-wall-coordinates>` describes, and the solve finds the steady state as {ref}`a differentiable root <numerics-steady-root>`.

Import LMhdX and work in double precision.

```python
import lmhdx

lmhdx.configure(dtype="float64")
```

`lmhdx.duct(ha, cells)` builds a square duct of unit half-width and unit
properties, with the field along $y$ and a unit pressure gradient as the drive.
Build one with insulating walls and one whose Hartmann walls conduct.

```python
insulating = lmhdx.duct(20, 32)
hunt = lmhdx.duct(20, 32, wall_conductance=0.05)
```

Solve both and print the flow rate and the solve's residual.

```python
for problem in (insulating, hunt):
    result = lmhdx.solve(problem)
    print(f"Q {float(result.flow_rate):.6g}, residual {float(result.residual):.1e}")
```

```text
Q 0.154647, residual 5.1e-09
Q 0.0915996, residual 1.3e-08
```

The conducting walls carry part of the current, so the same drive pushes less
flow.

These are the two ducts, and a third kind of wall:
- **Shercliff's duct.** The field is normal to the insulating $y$ walls, which
  carry the Hartmann layers; the side layers form on the $z$ walls.
- **Hunt's duct.** `wall_conductance` makes the Hartmann walls thin conducting
  walls of conductance ratio `c = sigma_w t_w / (sigma a)`.
- **Resolved walls.** `walls=dict(y=Wall(layers=...))` resolves a wall in cells
  instead (see [walls and fields](walls_and_fields.md)).

The mesh and the solve work as follows:
- The cross-section has `cells` cells along each axis. LMhdX clusters them to
  the Hartmann layer `a/Ha` on the walls normal to the field and to the side
  layer `a/sqrt(Ha)` on the others.
- On 32-64 cells the flow rate is within 1 % of the spectral reference from Ha
  20 to 1000.
- `result.residual` is the residual norm of the certified solve (relative
  tolerance 1e-9, `Solver(tol=)`). A solve that fails raises.
- A sweep of Hartmann numbers, fields or wall conductances on one mesh size
  compiles twice. From the third case on, LMhdX builds the mesh, field and
  factorizations on the host and passes them to the program the second case
  compiled.
- `Problem.replace` changes a visible part of a frozen problem, for example
  `insulating.replace(forcing=(2.0, 0.0, 0.0))`.

For an unsteady run, call
`lmhdx.solve(problem, lmhdx.Solver(dt=..., steps=...))`. A transient duct
settles on the steady state, and `lmhdx.Output` writes its records (see
[restart and output](../how_to/restart_and_output.md)).

## Prescribe throughput and measure pumping work

This uses {ref}`drive elimination <numerics-drive-elimination>`: one solve at unit drive gives the drive for any flow rate.

`python examples/hunt_example.py` runs Hunt's duct at a requested flow rate;
edit `TARGET_FLOW_RATE` and `DUCT_LENGTH` near its top. These are its steps.
Choose the flow rate and the length of the segment.

```python
TARGET_FLOW_RATE = 0.05
DUCT_LENGTH = 2.5  # a fully developed segment, not a whole blanket
```

`duct(..., flow_rate=Q)` sets the drive exactly.

```python
problem = lmhdx.duct(20.0, 24, wall_conductance=0.05, flow_rate=TARGET_FLOW_RATE)
solution = lmhdx.solve(problem)
```

Read the drive, check that the flow meets the target, and compute the
hydraulic power.

```python
drive, flow_rate = float(solution.pressure_gradient), float(solution.flow_rate)
conductance = flow_rate / drive  # G, the flow per unit drive
assert abs(flow_rate - TARGET_FLOW_RATE) <= 1e-8 * TARGET_FLOW_RATE
print(f"drive {drive:.6g}, dQ/d(drive) {conductance:.6g}, power {drive * DUCT_LENGTH * flow_rate:.6g}")
```

```text
drive 0.54214, dQ/d(drive) 0.0922271, power 0.0677675
```

The problem is linear in its drive. One solve gives the flow per unit drive
`G`, the required drive is `Q / G` (`result.pressure_gradient`), and the check
tests `Q = G*drive` to `1e-8`. This checks the linear drive-to-flow relation,
not an independent physical reference. The example also writes the solution
with `lmhdx.Output`, and a JSON summary of the flow, the drive,
`d(drive)/dQ = 1/G` and the hydraulic power `drive*L*Q`. This is the work of a
fully developed segment, excluding entry and exit, manifolds and thermal
effects. It is not a complete blanket pumping budget.

## Cross-section weights on the staggered core

`lmhdx.steady.channel_cross_section_weights(problem)` returns the transverse
`(y, z)` integration weight of every cell, $\Delta y_j \Delta z_k$, for a
`lmhdx.core3d.ChannelProblem`. Axis 0 is the flow axis of every channel this
package builds, since `lmhdx.core3d.duct_problem` and every other constructor
put the periodic axis there. A cell's weight therefore does not depend on the
axial spacing. A channel has no fluid mask. Every transverse cell counts, so
the weights sum to the full cross-section area.

```python
import numpy as np

from lmhdx.core3d import duct_problem
from lmhdx.steady import channel_cross_section_weights

problem = duct_problem(hartmann=20.0, cells=32)
weights = np.asarray(channel_cross_section_weights(problem))
extent = problem.grid.extent
assert weights.shape == problem.grid.shape[1:]
assert np.isclose(weights.sum(), extent[1] * extent[2])
```

This is a mesh-geometry query, not a solve, so it has no convergence or
precision envelope to fail. The new core measures its throughput and
pumping-power metrics against it.

## Throughput and pumping power on the staggered core

`lmhdx.steady.channel_flow_rate`, `channel_flow_response`,
`channel_drive_for_flow_rate` and `channel_fixed_flow_hydraulic_power` measure
the throughput of a `ChannelProblem` and eliminate its drive. They reuse
`channel_cross_section_weights`, `DuctResponse`, `pressure_drop` and
`hydraulic_power`. Find the drive for a target flow, solve at it, and check the
flow.

```python
from lmhdx.steady import channel_drive_for_flow_rate, channel_flow_rate
from lmhdx.steady import solve_steady_state

problem = duct_problem(hartmann=20.0, cells=32)
target = 0.02
drive = channel_drive_for_flow_rate(problem, target)
solution = solve_steady_state(problem, forcing=(float(drive), 0.0, 0.0))
achieved = channel_flow_rate(problem, solution.velocity[0].data[0])
assert np.isclose(float(achieved), target, rtol=1e-8)
```

How they work:
- `channel_flow_rate(problem, velocity)` takes one axial-velocity slice and
  integrates it against `channel_cross_section_weights`:
  $Q = \sum_{j,k} \Delta y_j \Delta z_k\, u_{jk}$. By periodicity, every
  axial station carries the same value.
- `channel_flow_response` solves once at unit axial drive to measure
  $G = Q(f{=}1)$ and returns it as a `DuctResponse`.
- `channel_drive_for_flow_rate` and `channel_fixed_flow_hydraulic_power`
  eliminate the drive the same way, $f = Q_\mathrm{target} / G$, with hydraulic
  power $f L Q_\mathrm{target}$.

These functions work in the Stokes limit only. $Q = Gf$ holds because the steady
residual is affine in the drive when `problem.advection == "off"`.
`channel_flow_response` raises `ValueError` for any other value, since the
residual then carries $-\nabla\cdot(\mathbf u\mathbf u)$ and a single unit-drive
solve stops determining the whole response. Like `hydraulic_power`, these are
isothermal segment quantities, excluding entry and exit losses, manifolds and
thermal effects.

## Fit a measured velocity profile

The gradients come from {ref}`implicit derivatives <numerics-derivatives>` of the steady solve.

This bounded inverse problem infers a positive pressure-gradient drive and a
magnetic-field multiplier from a velocity profile. Replace the synthetic
`target` with axial velocity samples on the same cell-centred mesh, and keep SI
units and geometry consistent. How it works:
- The loss is normalized and area-weighted.
- It uses the production field solve and its implicit SOLVAX derivatives, not
  an iteration-history tape. `jax.jit` compiles the objective and gradient
  together.
- SciPy's bounded optimizer,
  [L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html),
  controls the two design variables on the host.

```python
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from lmhdx.core3d import duct_problem
from lmhdx.steady import channel_cross_section_weights, solve_steady_state

jax.config.update("jax_enable_x64", True)
problem = duct_problem(hartmann=5, cells=12, cells_in_layer=3)
areas = channel_cross_section_weights(problem)

def velocity(parameters):
    return solve_steady_state(
        problem, forcing=(parameters[0], 0.0, 0.0), field_scale=parameters[1]
    ).velocity[0].data[0]

truth = jnp.array([1.4, 1.25])
target = velocity(truth)
normalization = jnp.sum(areas * target**2)
if not float(normalization) > 0:
    raise ValueError("A nonzero fluid velocity target is needed to identify the field.")

def loss(parameters):
    return jnp.sum(areas * (velocity(parameters) - target)**2) / normalization

value_and_gradient = jax.jit(jax.value_and_grad(loss))

def evaluate(parameters):
    value, gradient = value_and_gradient(jnp.asarray(parameters))
    return float(value), np.asarray(gradient, dtype=float)

fit = minimize(evaluate, [1.0, 1.0], jac=True, method="L-BFGS-B",
               bounds=[(0.1, 3.0), (0.5, 2.0)],
               options={"gtol": 1e-12, "ftol": 1e-15, "maxiter": 60})
if not fit.success:
    raise RuntimeError(f"Profile fit did not converge: {fit.message}")
print("drive, field multiplier:", fit.x)
print("relative squared profile error:", fit.fun)
```

```text
drive, field multiplier: [1.4  1.25]
relative squared profile error: 2.791727607454122e-21
```

This synthetic example checks parameter recovery, not experimental validity.
- Velocity alone cannot identify the sign of the field, because magnetic drag
  depends on its squared magnitude. That is why the field bound is positive.
- Weak fields, a bulk-flow observation only, or uncertain viscosity or geometry
  can also make the drive and the field hard to tell apart.
- Keep the full profile, and inspect the residual and the parameter
  sensitivity. For targets outside this two-parameter model, report the
  residual that remains; optimizer success is not a fit-quality certificate.
- Validate inferred parameters with the reporting solver and a held-out finer
  mesh before using them in a design study.

For a prescribed flow rate rather than a profile, don't optimize the drive.
`lmhdx.steady.channel_drive_for_flow_rate(problem, target_flow_rate)` eliminates
it exactly using the linear response. Neither fit is a thermal blanket design;
wall and geometry optimization and heat-transfer validation have separate gates.
