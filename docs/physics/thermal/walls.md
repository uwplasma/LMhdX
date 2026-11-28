(thermal-walls)=
# Conduction through solid walls

```{image} ../../_static/mistrangelo.webp
:alt: LMhdX with conducting steel walls against the six codes of Mistrangelo et al. 2025
:align: center
```

*LMhdX with conducting steel walls against the six codes of Mistrangelo et al. (2025). The [benchmark record](../../validation/mistrangelo.md) explains each panel.*

Solid walls conduct and store heat around the fluid, as in Mistrangelo et al.'s heated duct (ADR 0009
D36). This page gives the conjugate equation, its discretization and solve, and what the tests check.

## The equation

In the fluid and in each wall the temperature obeys

$$
\rho c_p\,\frac{\partial T}{\partial t} = \nabla\cdot(k\nabla T) + q
$$

with cell-wise $k$ and $\rho c_p$, and the fluid adds its transport and Joule heat from the
[heat transport page](../thermal.md). The walls carry no flow, and **the step omits Joule heat in
the walls.** `joule_heating` deposits the fluid's Joule heat. A solid wall only stores and conducts
heat, as the benchmark's eq. 4 has no source.

## The grid and the operator

- `core3d.conjugate_grid(grid, walls)` adds a `core3d.SolidWall` at either end of the transverse axes
  1 and 2. A `SolidWall` has a thickness, a whole number of equal cells, a `thermal_conductivity`
  and a `heat_capacity` $\rho c_p$. The `ConjugateGrid` holds the extended grid, whose fluid block
  keeps the given faces exactly, the block's slices and the walls.
- `cells(name, fluid_value)` spreads the walls' properties over every cell. A corner cell takes the
  nearer wall's material, and the first axis's when both are equally near to rounding.
- `poisson.variable_helmholtz(grid, conditions, conductivity, shift, coefficient)` is
  $V\,(s\,x - \tau\,\nabla\cdot(k\nabla x))$ for cell-wise $k$ and $s$.
- **Face conductance.** Between two cells with half widths $\delta_1$, $\delta_2$ and conductivities
  $k_1$, $k_2$, the flux per unit area is $(T_2 - T_1)/(\delta_1/k_1 + \delta_2/k_2)$, the half
  cells' series resistance. The same rule holds at a material interface. The face area and the cell
  volume enter once, in the divergence, so the heat that leaves one cell enters the next. A boundary
  face takes its cell's $k$, and you build a heat flux there with that $k$.

(thermal-walls-solve)=
## The solve

One implicit step solves for the increment $\delta = T^{n+1} - T^n$,

$$
\rho c_p\,\delta - \Delta t\,\nabla\cdot(k\nabla\delta)
= \Delta t\,\bigl(q + \nabla\cdot(k\nabla T^n) + b\bigr),
$$

with $b$ the boundary data's `lift`. Because the unknown is the increment, PCG's tolerance applies
to the change.
- The fast-diagonal inverse at the volume-mean $k$ and $s$ preconditions `solvax.pcg_linear_solve`
  ([solvers](../numerics/solvers.md)). The solve sets an insulated axis's null eigenvalue to zero,
  which keeps the shifted operator definite in float32 at a large $\Delta t\,k$.
- The solve differentiates implicitly, and its tangent and adjoint solves start from zero. If it
  does not converge, it returns NaN, and NaN derivatives.
- A wall's conductivity and heat capacity are leaves of the solve.
  `cells(name, fluid_value, walls={(axis, side): value})` puts a wall's value into the cells in the
  trace, and `VariableHelmholtz.with_coefficients(conductivity, shift)` builds the face conductances
  from it, so a new value does not recompile. A wall's thickness, cell count and materials are static.
- The operator matches SOLVAX 0.34.0's `separable.flux_form_operator` to 4e-16.

PCG takes 7–8 iterations for the benchmark's steel and PbLi walls. The count grows with the
contrast, and at a high contrast also as the mesh refines. These counts are measurements at these
contrasts and meshes, not a conditioning bound for others.

| Walls ($k$ and $\rho c_p$ over the fluid's) | PCG iterations, 6 to 24 cells across, $\Delta t$ 0.05 and 100 |
|:--|:--|
| the benchmark's steel and PbLi (0.9, 2.3) | 7–8 |
| copper-like (19, 1.85) | 14–33 |
| 100, 1.85 | 21–63 |

## In the thermal step

`ThermalProblem(..., walls=...)` takes a `SolidWall` or `None` for each end of axes 1 and 2. The
temperature and the heat source then live on `problem.thermal_grid`, `problem.fluid_block` picks out
the fluid's cells, and `problem.on_thermal_grid` pads fluid data onto it.
- The `thermal_conditions`, an inlet profile included, hold on the outer surface. Build a heat flux
  there with the `k` of each boundary cell. The walls' end faces take the axis-0 conditions.
- One `conduct` handles PCG through the walls and the fast-diagonal inverse of one material. A duct
  without walls keeps the fast-diagonal inverse.
- The ledger covers the fluid and its walls, with each cell weighed by its own $\rho c_p$.
  `conduction_iterations` reports the PCG iterations. The Boussinesq measure and the buoyancy
  Courant number cover the fluid only.

Here is a duct walled on four sides with steel-like walls and heated through part of its
outer bottom, as the ledger test runs it:

```python
import jax.numpy as jnp
import numpy as np

import lmhdx
from lmhdx.core3d import ChannelProblem, SolidWall, zero_velocity
from lmhdx.grid import CENTER, DIRICHLET, NEUMANN, BoundaryCondition, Field, Grid, uniform_faces
from lmhdx.thermal import ThermalProblem, advance_thermal, build_thermal_step, initial_thermal_state, thermal_boundary

lmhdx.enable_x64()
cells, axial = 6, 8
fluid = Grid(uniform_faces(axial, 0.0, 1.0), uniform_faces(cells, -1.0, 1.0), uniform_faces(cells, -1.0, 1.0))
opening = BoundaryCondition(DIRICHLET, lower=np.ones((cells, cells)), upper_kind=NEUMANN)
flow = ChannelProblem(fluid, (opening, *(BoundaryCondition(NEUMANN),) * 2), forcing=(0.0, 0.0, 0.0), dt=0.5 / axial)
steel = SolidWall(0.1, 2, 0.18, 2.3)  # 0.1 thick in 2 cells; k 0.9 and rho c_p 2.3 times the fluid's
patch = np.abs(np.asarray(fluid.centers[0]) - 0.5) < 0.25
heated = np.where(patch, -3.0, 0.0)[:, None] * np.ones(cells + 4)  # a flux into the outer bottom, walls included
conditions = (
    thermal_boundary("temperature", 0.0, upper_kind="heat_flux", conductivity=0.2),
    thermal_boundary("heat_flux", conductivity=0.18),  # the outer surface takes the wall's k
    thermal_boundary("heat_flux", heated, 0.0, conductivity=0.18),
)
walls = ((None, None), (steel, steel), (steel, steel))
thermal = ThermalProblem(flow, conditions, 1.0, 0.2, 0.0, joule_heating=False, walls=walls)
rest = zero_velocity(flow)
plug = (rest[0].replace_data(jnp.ones_like(rest[0].data)), *rest[1:])  # held, on the fluid's grid
state = initial_thermal_state(thermal, 0.0, plug)  # the temperature lives on thermal.thermal_grid
heat = Field(jnp.zeros(thermal.thermal_grid.shape), (CENTER,) * 3, thermal.thermal_grid)
step = build_thermal_step(thermal, dtype=jnp.float64, hold_flow=True)
state, history = advance_thermal(step, state, heat, 40)
print(int(history.conduction_iterations.max()), float(jnp.max(jnp.abs(history.heat_balance_residual))))
```

## What the tests check

In `tests/test_ops.py`:
- Two materials in series are exact, with the same flux through every face. Steel brought into
  contact with fluid reaches the exact effusivity-weighted contact temperature at second order. The
  benchmark's steel and PbLi, walled on four sides, stay symmetric and conservative.
- Walls of 1000 times the fluid's $k$, held at 0 outside, surround a uniformly heated square. The
  fluid's mean temperature tends to the Dirichlet Poisson series, 0.140577, raised by the walls' own
  drop $q a t/(2k_w)$. The error falls at about second order, and the Richardson value meets the
  series within 1e-4.
- A wall conductivity's derivative meets the derivative standard, with central differences at two
  steps, a second-order Taylor remainder and the JVP/VJP identity in two walls at once. A new value
  does not recompile, and the derivative is NaN when PCG has not converged.

In `tests/test_timeloop.py`:
- The walled duct above closes every step's ledger, at 300 K, to 1e-10 of its largest term. Each
  region, fluid and walls, stores what its face fluxes bring in, to PCG's tolerance, and the heated
  face's inflow is 3.3 to rounding.
- The conjugate step is backward Euler in conduction and explicit in transport. Against a run at a
  64th of the step it converges at first order as the step halves.
- Walls of the fluid's own material, of unequal cells, reproduce the separable step on the extended
  grid to 1e-10, with upwind and with van Leer transport. A moving walled flow steps exactly as
  `core3d.step`, and a hot wall cell does not change its buoyancy.
- The run differentiates with respect to its heat source and its start. Derivatives in a wall's `k`
  and `ρc_p` through the step are planned. For now `problem.conduction()` binds them.

Mistrangelo et al. is the only conjugate-heat benchmark found so far. The exact two-material series
and contact transient therefore stay the CI gate, and their heated duct is the independent check,
code to code ([record](../../validation/mistrangelo.md)).
