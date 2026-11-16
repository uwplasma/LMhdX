# Fully developed duct flows

A fully developed duct is a periodic duct of one axial cell: the axial velocity $u(y,z)$ and the electric
potential $\phi(y,z)$ on a structured cross-section, from one preconditioned conjugate-gradient solve.
`lmhdx.duct(ha, cells)` is the square duct of unit half-width and unit properties, field along $y$, solved at a
unit pressure gradient:

```python
import lmhdx

insulating = lmhdx.duct(20, 32)
hunt = lmhdx.duct(20, 32, wall_conductance=0.05)

for problem in (insulating, hunt):
    result = lmhdx.solve(problem)
    print(float(result.flow_rate), float(result.residual))
```

The field normal to the insulating $y$ walls gives the Hartmann layers there and the side layers on the
$z$ walls (Shercliff's duct); `wall_conductance` makes the Hartmann walls thin conducting walls of conductance
ratio `c = sigma_w t_w / (sigma a)` (Hunt's duct). `walls=dict(y=Wall(layers=...))` resolves a wall in cells
instead (see [walls and fields](walls_and_fields.md)). The cross-section has `cells` cells along each axis,
clustered to the Hartmann layer `a/Ha` on the walls normal to the field and to the side layer `a/sqrt(Ha)` on
the others. A sweep of Hartmann numbers, fields or wall conductances on one mesh size compiles twice: from the
third case on, the mesh, field and factorizations are built on the host and passed to the program the second
case compiled. On 32-64 cells the flow rate is within 1 % of the spectral reference from Ha 20 to 1000.
`result.residual` is the residual norm of the certified solve (relative tolerance 1e-9, `Solver(tol=)`); a solve
that fails raises. `Problem.replace` changes a visible part of a frozen problem, for example
`insulating.replace(forcing=(2.0, 0.0, 0.0))`.

An unsteady run is `lmhdx.solve(problem, lmhdx.Solver(dt=..., steps=...))`: a transient duct settles on the
steady state, and `lmhdx.Output` writes its records (see [restart and output](../how_to/restart_and_output.md)).

## Prescribe throughput and measure pumping work

Run `python examples/hunt_example.py` for a conducting-wall duct with a requested flow rate. Edit
`TARGET_FLOW_RATE` and `DUCT_LENGTH` near the top. `duct(..., flow_rate=Q)` sets the drive exactly, because the
problem is linear in it: one solve gives the flow per unit drive `G`, the required drive is `Q / G`
(`result.pressure_gradient`), and the flow check tests `Q = G*drive` to `1e-8`. This checks the linear
drive-to-flow relation, not an independent physical reference. The JSON summary records flow, drive,
`d(drive)/dQ = 1/G`, and hydraulic power `drive*L*Q`. This is fully developed segment work, excluding
entry/exit, manifolds and thermal effects. It is not a complete blanket pumping budget.

## Cross-section weights on the staggered core

`lmhdx.steady.channel_cross_section_weights(problem)` returns the transverse `(y, z)` integration weight of
every cell, $\Delta y_j \Delta z_k$, for a `lmhdx.core3d.ChannelProblem`. Axis 0 is the flow axis of every
channel this package builds (`lmhdx.core3d.duct_problem` and every other constructor put the periodic axis
there), so the weight of a cell does not depend on the axial spacing. A channel carries no fluid mask -- every
transverse cell counts, so the weights sum to the full cross-section area:

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
precision envelope to fail; it is the building block the new core's throughput
and pumping-power metrics are measured against.

## Throughput and pumping power on the staggered core

`lmhdx.steady.channel_flow_rate`, `channel_flow_response`, `channel_drive_for_flow_rate` and
`channel_fixed_flow_hydraulic_power` measure the throughput of a `ChannelProblem` and eliminate its drive,
reusing `channel_cross_section_weights`, `DuctResponse`, `pressure_drop` and `hydraulic_power`:

```python
import numpy as np

from lmhdx.core3d import duct_problem
from lmhdx.steady import channel_drive_for_flow_rate, channel_flow_rate
from lmhdx.steady import solve_steady_state

problem = duct_problem(hartmann=20.0, cells=32)
target = 0.02
drive = channel_drive_for_flow_rate(problem, target)
solution = solve_steady_state(problem, forcing=(float(drive), 0.0, 0.0))
achieved = channel_flow_rate(problem, solution.velocity[0].data[0])
assert np.isclose(float(achieved), target, rtol=1e-8)
```

`channel_flow_rate(problem, velocity)` takes one axial-velocity slice --
every axial station carries the same value by periodicity -- and integrates
it against `channel_cross_section_weights`:
$Q = \sum_{j,k} \Delta y_j \Delta z_k\, u_{jk}$. `channel_flow_response` solves once at unit axial
drive to measure $G = Q(f{=}1)$ and returns it as a `DuctResponse`;
`channel_drive_for_flow_rate` and `channel_fixed_flow_hydraulic_power`
eliminate the drive the same way,
$f = Q_\mathrm{target} / G$, with hydraulic power $f L Q_\mathrm{target}$.

Envelope: the Stokes limit only. $Q = Gf$ holds because the steady residual is
affine in the drive when `problem.advection == "off"`; `channel_flow_response`
raises `ValueError` for any other value, since the residual then carries
$-\nabla\cdot(\mathbf u\mathbf u)$ and a single unit-drive solve stops
determining the whole response. Like `hydraulic_power`, these are isothermal
segment quantities, excluding entry/exit losses, manifolds and thermal
effects.

## Fit a measured velocity profile

Use this bounded inverse problem to infer a positive pressure-gradient drive
and magnetic-field multiplier. Replace the synthetic `target` with axial
velocity samples on the same cell-centered mesh; keep SI units and geometry
consistent. The normalized, area-weighted loss
uses the production field solve and its implicit SOLVAX derivatives, not an
iteration-history tape. SciPy controls the two design variables on the host;
the objective and gradient are compiled together. The bounded optimizer is
[SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).

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

This synthetic example checks parameter recovery, not experimental validity.
The sign of the field is not identifiable from velocity alone: magnetic drag
depends on its squared magnitude, hence the positive field bound. Weak fields,
only a bulk-flow observation, or uncertain viscosity/geometry can also make
drive and field poorly distinguishable. Keep the full profile, inspect the
residual and parameter sensitivity, and report an irreducible residual for
targets outside this two-parameter model; optimizer success is not a fit-quality
certificate. Validate inferred parameters with the reporting solver and a
held-out finer mesh before using them in a design study.

For a prescribed flow rate rather than a profile, avoid optimizing drive:
`lmhdx.steady.channel_drive_for_flow_rate(problem, target_flow_rate)` eliminates
it exactly using the linear response. Neither fit is a thermal blanket design;
wall/geometry optimization and heat-transfer validation have separate gates.
