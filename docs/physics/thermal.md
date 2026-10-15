# Heat transport

`lmhdx.thermal` carries a temperature through the duct fluid on the staggered
core: conduction, transport by the flow, Joule heating, a volumetric heat
source, and optional Boussinesq buoyancy (ADR 0008). The fluid is the only
medium. Walls enter through a prescribed temperature or heat flux, not through
their own conduction. The capability is research-stage, and its supported
envelope is below.

## Equations and assumptions

The temperature equation, with constant properties, is

$$
\rho_0 c_p\left[\frac{\partial T}{\partial t}+\nabla\cdot(\mathbf u T)\right]
=\nabla\cdot(k\nabla T)+\frac{|\mathbf J|^2}{\sigma}+q,
$$

alongside the inductionless momentum and charge equations of
[Equations and assumptions](equations.md). With buoyancy, the momentum equation
gains the force density

$$
\mathbf f_b=-\rho_0\beta\,(T-T_r)\,\mathbf g .
$$

| Symbol | Meaning | Where it is set |
|:--|:--|:--|
| $\rho_0$, $\nu$, $\sigma$ | density, kinematic viscosity, electrical conductivity | `ChannelProblem.density`, `.viscosity`, `.conductivity` |
| $c_p$, $k$ | specific heat, thermal conductivity | `ThermalProblem.specific_heat`, `.thermal_conductivity` |
| $\alpha=k/(\rho_0 c_p)$ | thermal diffusivity | `ThermalProblem.diffusivity`, derived |
| $T_r$, $\beta$, $\mathbf g$ | reference temperature, expansion coefficient, gravity (pointing down) | `.reference_temperature`, `.expansion_coefficient`, `.gravity` |
| $q$ | external heat per unit volume and time | the `heat_source` cell field passed to each step |

These are the assumptions:
- The magnetic Reynolds number is small.
- The flow is incompressible.
- Conduction obeys Fourier's law with constant properties.
- Density varies only in the buoyancy force.

The flow should satisfy $\max\beta|T-T_r|\ll 1$. The step reports it as
`boussinesq_measure`; it is not enforced.

The heat equation leaves out:
- viscous heating;
- radiation;
- phase change;
- thermoelectric effects;
- property variation with temperature.

Joule heat is $|\mathbf J|^2/\sigma$, the dissipation in the moving fluid. The
laboratory-frame $\mathbf J\cdot\mathbf E$ also contains the Lorentz work the
momentum equation already carries.

There is no units flag. A run is either consistently SI or consistently
nondimensional. `duct_problem` is nondimensional (half width, density, viscosity
and conductivity one), so its thermal properties must be scaled the same way.

## Boundary conditions

Each axis is periodic, exactly where the flow's is. Otherwise each end
prescribes a wall temperature or an outward conductive heat flux, and the two
ends may differ. Thermal, mechanical and electrical conditions are independent:
an electrically insulating wall need not be adiabatic.

`thermal_boundary` builds the condition from physical data. A positive
`heat_flux` leaves the fluid at both ends. It is stored as the derivative along
the axis that `lmhdx.grid.pad` applies to a Neumann condition, $q/k$ at the
lower end and $-q/k$ at the upper. A condition built for one $k$ therefore does
not hold the same flux under another, and must be rebuilt when $k$ changes.

## Discretization

**Placement.** The temperature lives at the cell centres and the velocity on
the faces.

**Conduction** is implicit. One cell-centred fast-diagonal factorization of
$I-\Delta t\,\alpha L_0$ is built under the homogeneous conditions, and the
boundary data enter as the fixed lift $b=L(0)$. Its unit shift keeps the
operator nonsingular under insulated and periodic walls, so the mean
temperature is kept rather than removed as a gauge.

**Transport** is conservative and first-order upwind:
- Each face carries $u_f T_\mathrm{up}$.
- A wall face carries no advective heat.
- A periodic axis reads its wrap face from its last and first cells.

On a divergence-free flow it makes no new extremum while $\Delta t$ times every
cell's outgoing volume-flux rate stays below one; 0.8 is the documented
envelope. Upwinding adds a numerical diffusivity of about $|u|h/2$, so a
heat-transfer result must be refined until it stops changing.

**Joule heat** is each face current squared, then shared between the half
cells on either side by `ops.face_average_adjoint`:
- A wall face heats only the fluid's half cell.
- The dissipation inside a conducting wall is the wall's, not the fluid's.
- The heat comes from the full face currents, the same ones the Lorentz force
  is built from.
- Its volume integral is `core3d.energy_budget(...).joule`.

**Buoyancy** is averaged onto the velocity faces as the Lorentz force is, and
zeroed on wall faces. It is added before the division by density and the
projection. Without $\beta$ or $\mathbf g$ there is no force term at all.

**The step** is first-order and partitioned, with every source taken from the
old state $n$:

$$
(I-\Delta t\,\alpha L_0)\,T^{n+1}
=T^n-\Delta t\,D\!\left(\mathbf u^n T^n_\mathrm{up}\right)
+\frac{\Delta t}{\rho_0 c_p}\left(q_J^n+q\right)+\Delta t\,\alpha\, b .
$$

The velocity takes the predictor and projection of `core3d.step`, with the
flow's implicit viscous factorizations and the buoyancy of $T^n$. One electric
solve per step supplies both the Lorentz force and the Joule heat.

The fractional step does not hold every hydrostatic state exactly. A
temperature varying only along gravity stays at rest; in general, the claim
is first-order convergence in $\Delta t$.

## Diagnostics and the heat ledger

Each step returns a `ThermalDiagnostics` of scalars. The heat ledger is

$$
\underbrace{\frac{\rho_0 c_p}{\Delta t}\sum_P V_P\,(T^{n+1}_P-T^n_P)}_{\texttt{storage\_rate}}
+\texttt{conductive\_heat\_out}-\texttt{joule\_power}-\texttt{external\_power}
=\texttt{heat\_balance\_residual},
$$

computed from local increments and the same wall gradients the conduction uses.
It closes to its rounding floor, which is about $\varepsilon\,\rho_0 c_p\sum V\,|T|/\Delta t$
for temperatures stored in kelvin. The other diagnostics are:
- `heat_solve_residual`: the largest local residual of the temperature equation;
- `mass_residual` and `charge_residual`;
- `advective_cfl`;
- `boussinesq_measure`.

`finite` only says every value is a number. A run is accepted by checking
these values: the Courant number, the residuals, and the Boussinesq measure. A
failing value is not repaired by clipping.

## Supported envelope

**Geometry**
- Cartesian grids, with periodic axes and closed no-slip walls. Polar grids and
  inflow-outflow axes are refused when the problem is built.
- With momentum transport on (`ChannelProblem.advection` other than `"off"`), a
  periodic axis needs at least two cells.
- Insulating, thin conducting and resolved electrical walls all work, with
  their electrical dissipation kept outside the fluid.

**Precision.** The `dtype` given to `build_thermal_step` must be the core's
compute dtype: float64 after `lmhdx.enable_x64()`, else float32. Set it before
building.

**Derivatives** are supported with respect to anything carried in the state or
the heat source, for example the initial temperature or a source amplitude. They
are not supported with respect to $k$, $c_p$, $\sigma$, $\Delta t$, the geometry
or the boundary data, which are built into the prepared step.

**Memory.** `advance_thermal` keeps no field history; in reverse mode it keeps
one state per step.

**Not included**
- Conjugate wall conduction.
- Open ducts and downstream heat removal.
- Higher-order transport.
- Variable properties.
- A steady thermal solve.
- A thermal restart format.
- Dispatch through `lmhdx.solve`.

A closed domain with net heating has no steady temperature. The state returned
is not a `cases.MHDState` or an archive-compatible `Solution`. The final
electric fields come from `core3d.face_currents(state.velocity, flow)`.

## A worked example

A nondimensional Hartmann duct runs two cases. The first is driven along its
periodic axis, heated by a uniform volumetric source, and cooled through its
top wall. The second is the same duct with buoyancy.

```python
import dataclasses

import jax.numpy as jnp

import lmhdx
from lmhdx.core3d import ChannelProblem
from lmhdx.grid import CENTER, NEUMANN, PERIODIC, BoundaryCondition, Field, Grid, uniform_faces
from lmhdx.thermal import (
    ThermalProblem,
    advance_thermal,
    build_thermal_step,
    initial_thermal_state,
    thermal_boundary,
    validate_thermal_inputs,
)

lmhdx.enable_x64()  # before anything is built: the step's dtype must be the core's

grid = Grid(uniform_faces(1, 0.0, 1.0), uniform_faces(16, -1.0, 1.0), uniform_faces(16, -1.0, 1.0))
periodic, wall = BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN)
flow = ChannelProblem(
    grid, (periodic, wall, wall), magnetic_field=(0.0, 10.0, 0.0), forcing=(1.0, 0.0, 0.0), dt=2.0e-3
)
k = 0.05
walls = (
    periodic,
    thermal_boundary("heat_flux", conductivity=k),  # adiabatic side walls
    thermal_boundary("heat_flux", 0.0, 1.0, conductivity=k),  # the top wall takes out a unit flux
)
thermal = ThermalProblem(flow, walls, specific_heat=1.0, thermal_conductivity=k, reference_temperature=0.0)
heat = Field(jnp.full(grid.shape, 0.5), (CENTER,) * 3, grid)

for problem in (thermal, dataclasses.replace(thermal, expansion_coefficient=0.02, gravity=(0.0, 0.0, -1.0))):
    state = initial_thermal_state(problem, 0.0)
    validate_thermal_inputs(problem, state, heat, divergence_tolerance=1e-10)
    step = build_thermal_step(problem, dtype=jnp.float64)
    state, history = advance_thermal(step, state, heat, 200)
    power = float(history.external_power[-1] + history.joule_power[-1])
    assert bool(history.finite.all()) and float(history.advective_cfl.max()) <= 0.8
    assert abs(float(history.heat_balance_residual[-1])) <= 1e-9 * power
    assert float(history.boussinesq_measure.max()) <= 0.05
    print(
        f"beta {problem.expansion_coefficient}: T max {float(history.temperature_max[-1]):.4f}, "
        f"heat out {float(history.conductive_heat_out[-1]):.4f}, Joule {float(history.joule_power[-1]):.4f}"
    )
```

## Measured cost

The step of `advance_thermal` against `core3d.advance` with implicit viscosity
was measured on an AMD Ryzen 9 5900HS CPU, in one process at float64, with
JAX 0.6.2. "Cold" is the first call, including trace and compile, with the
persistent cache off. "Memory" is XLA's compiled temporary and output size.

| Mesh | Run | Cold (s) | Warm (ms/step) | Forward memory (MiB) | Reverse memory, 10 / 40 steps (MiB) |
|:--|:--|--:|--:|--:|--:|
| 1×48×48 | isothermal | 0.47 | 0.19–0.22 | 0.21 | — |
| 1×48×48 | thermal | 0.66 | 0.36–0.38 | 0.45 | 1.05 / 3.18 |
| 1×48×48 | buoyant | 0.69 | 0.35 | 0.45 | 1.99 / 4.65 |
| 8×32×32 | isothermal | 0.48 | 0.91–0.95 | 1.06 | — |
| 8×32×32 | thermal | 0.89–0.99 | 1.28–1.47 | 1.61 | 3.24 / 9.22 |
| 8×32×32 | buoyant | 0.92–1.05 | 1.34–1.45 | 1.55 | 6.27 / 14.12 |

The forward memory is the same at 10 and 40 steps. The reverse memory grows by
one state per step.
