# Fully developed duct flows

Fully developed cases solve the axial velocity $u(y,z)$ and electric potential
$\phi(y,z)$ on a structured cross-section. Start with the named builders:

```python
import lmhdx

hartmann = lmhdx.make_hartmann_case(ha=20, ny=32, nz=32)
shercliff = lmhdx.make_shercliff_case(ha=20, ny=32, nz=32)
hunt = lmhdx.make_hunt_case(ha=20, ny=32, nz=32, wall_cells=4)

for case in (hartmann, shercliff, hunt):
    result = lmhdx.solve(case)
    print(case.name, result.converged, result.residual)
```

Hartmann applies the field normal to insulating Hartmann walls. Shercliff
orients it so side layers control the profile. Hunt has conducting Hartmann
walls and insulating side walls.

`lmhdx.solve` runs a case on the staggered core (`lmhdx.fully_developed`): the
cross-section becomes a `ChannelProblem` with one periodic axial cell, and the
steady state is one preconditioned conjugate-gradient solve, compiled once per
case and reused for any drive. A sweep of Hartmann numbers, fields or wall
conductances on one mesh size compiles twice: from the third case on, the mesh,
field and factorizations are built on the host and passed to the program the
second case compiled, with no trace. `ny` and `nz` are fluid cells, clustered to the
Hartmann layer `a/Ha` on the walls normal to the field and to the side layer
`a/sqrt(Ha)` on the others, exactly as `lmhdx.duct_problem` does. Hunt's
conducting walls are thin walls of conductance ratio `c = sigma_w t_w / (sigma a)`:
the solution covers the fluid, and `wall_cells` do not enter. With
`wall_model="resolved"` on the geometry, the walls of one axis get `wall_cells`
cells of their own instead; that is also the closure for one conducting wall or
two different ones. On 32-64 cells
the flow rate is within 1 % of the spectral reference from Ha 20 to 1000.
`result.residual` is the relative steady residual `||R(u)|| / ||R(0)||`,
certified to 1e-9; a solve that fails raises. The case's pseudo-time controls
(time step, relaxation, potential and coupling iterations) do not enter.
`lmhdx.solve` refuses a `"transient"` case: the pseudo-time loop of the
cell-centred solver is `lmhdx.cases.solve_transient`, and a time history on the
core is `lmhdx.advance` on `lmhdx.fully_developed.channel_problem(case)`.

Use `dataclasses.replace` to change a visible part of a frozen case:

```python
from dataclasses import replace

case = replace(hartmann, forcing=2.0)
```

After solving, check more than the update norm:

```python
from lmhdx.validation import hartmann_validation, validation_summary

comparison = hartmann_validation(result, ha=20)
metrics = validation_summary(result, case.name, ha=20)
print(comparison.l2_error, metrics["div_current_max"])
```

`result.diagnostics` holds the flow rate, the Lorentz and ohmic power and the
largest cell divergence of the current. `lmhdx.solvers.fully_developed_power_balance`
audits the cell-centred solver of `lmhdx.cases`, not this one.
Increase wall and fluid resolution together for high Hartmann number cases;
the mesh-quality helpers report cells across Hartmann and side layers.

## Prescribe throughput and measure pumping work

Run `python examples/hunt_example.py` for a conducting-wall duct with a
requested flow rate. Edit `TARGET_FLOW_RATE` and `DUCT_LENGTH` near the top;
set the target to `None` to prescribe `FORCING` instead. The example solves at
unit drive to measure the flow per unit drive `G`, eliminates the required
drive analytically, and solves again at that drive. Both solves must
report `converged`, and the second must reproduce `G*drive` to a `1e-8`
relative flow check. This checks the linear drive-to-flow relation, not an
independent physical reference. Its JSON summary records flow, drive,
`d(drive)/dQ = 1/G`, and hydraulic power `drive*L*Q`.
This is fully developed segment work, excluding entry/exit, manifolds and
thermal effects. It is not a complete blanket pumping budget.

## Cross-section weights on the staggered core

`lmhdx.design.channel_cross_section_weights(problem)` is the `ChannelProblem`
counterpart of `fluid_cell_areas`: it returns the transverse `(y, z)`
integration weight of every cell, $\Delta y_j \Delta z_k$, for a
`lmhdx.core3d.ChannelProblem`. Axis 0 is the flow axis of every channel this
package builds (`lmhdx.core3d.duct_problem` and every other constructor put the
periodic axis there), so the weight of a cell does not depend on the axial
spacing. A channel carries no fluid mask -- every transverse cell counts, so
the weights sum to the full cross-section area:

```python
import numpy as np

from lmhdx.core3d import duct_problem
from lmhdx.design import channel_cross_section_weights

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

`lmhdx.design.channel_flow_rate`, `channel_flow_response`,
`channel_drive_for_flow_rate` and `channel_fixed_flow_hydraulic_power` are the
`ChannelProblem` counterparts of `volumetric_flow_rate`, `linear_flow_response`,
`drive_for_flow_rate` and `fixed_flow_hydraulic_power` above, reusing
`channel_cross_section_weights`, `DuctResponse`, `pressure_drop` and
`hydraulic_power`:

```python
import numpy as np

from lmhdx.core3d import duct_problem
from lmhdx.design import channel_drive_for_flow_rate, channel_flow_rate
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
$Q = \sum_{j,k} \Delta y_j \Delta z_k\, u_{jk}$, the same total-flux convention
as `volumetric_flow_rate`. `channel_flow_response` solves once at unit axial
drive to measure $G = Q(f{=}1)$ and returns it as a `DuctResponse`;
`channel_drive_for_flow_rate` and `channel_fixed_flow_hydraulic_power`
eliminate the drive the same way the `CaseSpec` route above does,
$f = Q_\mathrm{target} / G$, with hydraulic power $f L Q_\mathrm{target}$.

Envelope: the Stokes limit only. $Q = Gf$ holds because the steady residual is
affine in the drive when `problem.advection == "off"`; `channel_flow_response`
raises `ValueError` for any other value, since the residual then carries
$-\nabla\cdot(\mathbf u\mathbf u)$ and a single unit-drive solve stops
determining the whole response. Like the functions above, these are isothermal
segment quantities, excluding entry/exit losses, manifolds and thermal
effects.

## Fit a measured velocity profile

Use this bounded inverse problem to infer a positive pressure-gradient drive
and magnetic-field multiplier. Replace the synthetic `target` with axial
velocity samples on the same cell-centered mesh; keep SI units and geometry
consistent. Solid cells have zero weight. The normalized, area-weighted loss
uses the production field solve and its implicit SOLVAX derivatives, not an
iteration-history tape. SciPy controls the two design variables on the host;
the objective and gradient are compiled together. The bounded optimizer is
[SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).

```python
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import lmhdx
from lmhdx.design import fluid_cell_areas

jax.config.update("jax_enable_x64", True)
case = lmhdx.make_shercliff_case(ha=5, ny=12, nz=12)
areas = fluid_cell_areas(case)

def velocity(parameters):
    return lmhdx.solve_fully_developed_fields(
        case, forcing=parameters[0], magnetic_field_scale=parameters[1]
    )[0]

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
`lmhdx.design.linear_flow_response(case).drive_for(target_flow_rate)` eliminates
it exactly using the linear response. Neither fit is a thermal blanket design;
wall/geometry optimization and heat-transfer validation have separate gates.
