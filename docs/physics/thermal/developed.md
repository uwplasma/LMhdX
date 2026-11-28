(thermal-developed)=
# Fully developed heat transfer

A duct can be periodic in its flow direction, with the velocity and the heat sources uniform along
it. Its temperature then settles into a fixed cross-section shape, and LMhdX solves for that shape
once on the cross-section, with no time stepping. This page covers the H1 and H2 conditions, the
uniform wall temperature (T) and the pipe. The [evidence tables](../../validation/developed_heat.md)
give every result in full.

(thermal-developed-h1h2)=
## H1 and H2: a uniform axial heat input

Under a uniform axial heat input the temperature is $T=\gamma x+\theta(y,z)$, where $\gamma$ is the
bulk temperature's axial gradient and

$$
k\,\nabla^2\theta=\rho_0 c_p\,u\,\gamma-q-q_J .
$$

`developed_heat(problem, velocity, heat_source, axial_gradient=...)` solves this on the
cross-section with the fast-diagonal Poisson factorization (ADR 0008, D33). The walls set $\gamma$:
- **Heat-flux walls only.** $\gamma$ follows from the heat balance $\rho_0 c_p\gamma Q$ = heat in per
  unit length, and $\theta$ has zero mean. A uniform flux is the H2 condition.
- **Any prescribed-temperature wall.** That wall takes in whatever heat the solve decides, so you
  give $\gamma$. A peripherally uniform wall temperature is the H1 condition.

The duct is one cell long. `validate_developed_inputs` checks on the host that the flow is a single
throughflow, with a flow rate above `flow_rate_tolerance`, the same axial velocity on both axial faces,
and no transverse or reversed velocity, within `velocity_tolerance`. A flow that stops gives NaN for
the bulk temperature and $Nu$.

The result gives the bulk temperature, weighted by the axial flow, the area-mean and peak wall
temperatures, the mean wall heat flux into the fluid, $Nu=q_w D_h/\left(k\,(T_w-T_b)\right)$ with
$D_h=4A/P$, and the heat-balance residual.
- The wall means are over every transverse wall, heated or not. With an internal source, $Nu$ is an
  apparent number for that source, not a coefficient of the duct alone.
- $Nu$ is NaN where $|T_w-T_b|$ is within $64\varepsilon(L/h)^2$ of the largest temperature scale,
  with $L/h$ the section's largest extent over its smallest cell. The temperatures and the heat
  balance stay valid.
- With heat-flux walls only, the physical temperature is $T=T_{b,\mathrm{in}}+\gamma x+\theta-\theta_b$.
  `DevelopedHeat.anchored_peaks(T_in, length)` returns the hottest wall face and cell over the
  length, assuming the profile is developed from the inlet. With a prescribed wall temperature both
  peaks are NaN.
- The result differentiates through the velocity, the heat source and $\gamma$, but not the
  geometry. Through the velocity it reaches the implicit derivative of `steady.solve_steady_state`.

**Result.** On 64 cells the square duct's H1 is within 0.087 % of 3.60795 and its H2 within 0.090 %
of 3.08738, both at second order against exact series. On 32 cells with Joule heat off, a transverse
field raises the square duct's H1 from 3.62 at Ha 0 to 4.56 at Ha 10 and 5.35 at Ha 30, toward the
slug value of 7.09.

```{image} ../../_static/thermal_developed_duct.webp
:alt: Velocity and temperature of an H1 square duct at Ha 0 and 100 in 3-D, Nusselt numbers against aspect ratio and Hartmann number, and mesh convergence
:align: center
```

The figure verifies the solver against exact solutions, not against experiment.
`python scripts/make_showcase_figures.py --only developed` redraws it from `docs/_static/data/`.
Rectangular H1 ducts with aspect ratios from 1 down to 1/8 agree with the exact series to within
0.35 % on 32 cells across the short side. On the meshes of `duct_problem` the square duct's Nu rises
from Ha 0 to 1000 and is still 6 % below the slug-flow value at Ha 1000. Hartmann plates taking in a
uniform flux, in the solver's own steady flow, match their closed form, $Nu=4/\int_0^1F_H^2\,ds$ with $D_h=4a$, to within
0.2 % from Ha 1 to 1000 on 128 layer-resolving cells.

(thermal-developed-t)=
## The uniform wall temperature

**The uniform wall temperature.** The T condition holds every wall at one temperature, and the
excess decays along the duct in a fixed shape, $T-T_w=\theta\,e^{-\lambda x}$. Without axial
conduction or internal heat, $\theta$ is the eigenfunction of $-L(\theta)=\Lambda f\theta$ with the
smallest eigenvalue, where $f=u/\bar u$. `developed_wall_temperature(problem, velocity)` finds it with
`solvax.inverse_iteration` on the same factorization. It returns $\lambda=\Lambda\alpha/\bar u$, the
profile $(T-T_w)/(T_b-T_w)$ and $Nu_T=\Lambda D_hA/P_w$, which is $\Lambda D_h^2/4$ when every wall
is held.

```python
import lmhdx
from lmhdx.core3d import ChannelProblem
from lmhdx.grid import NEUMANN, PERIODIC, BoundaryCondition, Grid, uniform_faces
from lmhdx.steady import solve_steady_state
from lmhdx.thermal import ThermalProblem, developed_wall_temperature, thermal_boundary

lmhdx.enable_x64()
periodic, wall, held = BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN), thermal_boundary("temperature")
grid = Grid(uniform_faces(1, 0.0, 1.0), uniform_faces(32, -1.0, 1.0), uniform_faces(32, -1.0, 1.0))
flow = ChannelProblem(grid, (periodic, wall, wall), forcing=(1.0, 0.0, 0.0))  # one periodic axial cell
thermal = ThermalProblem(
    flow, (periodic, held, held), specific_heat=1.0, thermal_conductivity=1.0, reference_temperature=0.0
)
result = developed_wall_temperature(thermal, solve_steady_state(flow).velocity)
print(f"Nu_T {float(result.nusselt):.4f} in {int(result.iterations)} iterations")  # 2.9823 in 12
```

- The iteration stops once `eigen_residual` is below `tolerance` (1e-10 in float64, 1e-4 in float32)
  or after `max_iterations`. The square duct takes 12 iterations, and a 1:8 duct takes 146.
- The supported derivative is the first derivative of $\Lambda$, the decay rate and $Nu_T$ at a
  converged solve. $\Lambda$ is the Rayleigh quotient of the converged $\theta$, so its derivative
  needs none of the iteration's. Through the steady solve's field scale it meets central differences
  to 1e-5. The profile has a zero gradient, and second derivatives are not supported. This function
  does not implement ADR 0010's 2.0 contract, that an unsupported derivative raises.
- The solve returns NaN, with `converged` false, when it is unconverged, when the flow is reversed in
  any cell or overall, when there is no flow, or when the flow has a transverse component or uneven
  faces. A heat-flux wall must be adiabatic, and sources and Joule heat are left out.

**Result.** The square duct's $Nu_T$ converges at second order to 2.97753, within 0.040 % on 64
cells, and slug flow is exact.

(thermal-developed-pipe)=
## The pipe

**The pipe.** On the polar grid of `lmhdx.pipe`, both developed solves use the polar Poisson
factorization. The wall is the outer radius alone, and $D_h=2R$. The flow comes from `solve_pipe`,
placed on the axial faces. A pipe forms its currents in `lmhdx.pipe`, not in the core, so the
developed solves refuse Joule heat and a pipe's `ThermalProblem` needs `joule_heating=False`:

```python
import dataclasses

import jax.numpy as jnp

import lmhdx
from lmhdx.core3d import ChannelProblem, zero_velocity
from lmhdx.grid import CENTER, NEUMANN, PERIODIC, BoundaryCondition, Field
from lmhdx.pipe import pipe_problem, solve_pipe
from lmhdx.thermal import ThermalProblem, developed_heat, developed_wall_temperature, thermal_boundary

lmhdx.enable_x64()
pipe = pipe_problem(hartmann=0.0, radial=32, azimuthal=16)  # radius, azimuth and one axial cell
periodic, wall = BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN)
flow = ChannelProblem(pipe.grid, (wall, periodic, periodic))
rest = zero_velocity(flow)
axial = jnp.broadcast_to(solve_pipe(pipe)[0].data, rest[2].shape)  # the developed flow on the axial faces
velocity = (*rest[:2], rest[2].replace_data(axial))
heated = thermal_boundary("heat_flux", -1.0, -1.0, conductivity=1.0)  # a unit flux in through the wall
thermal = ThermalProblem(flow, (heated, periodic, periodic), 1.0, 1.0, 0.0, joule_heating=False)
zero = Field(jnp.zeros(pipe.grid.shape), (CENTER,) * 3, pipe.grid)
flux = float(developed_heat(thermal, velocity, zero).nusselt)  # 48/11 to 0.07 %
conditions = (thermal_boundary("temperature"), periodic, periodic)
held = float(developed_wall_temperature(dataclasses.replace(thermal, thermal_conditions=conditions), velocity).nusselt)
print(f"Nu_H {flux:.4f}, Nu_T {held:.4f}")  # 3.65679 to 0.02 %
```

**Result.** The H and T numbers, in Poiseuille and slug flow, converge at second order to their
exact values. The T numbers come from `developed_wall_temperature`, so its derivative contract holds.

The [MHD Nusselt campaign](../../validation/mhd_nusselt.md) compares Shercliff's square duct in H1,
H2 and T against an independent spectral solve, within 0.06 % from Ha 1 to 316, and against the
published curves of Bluck, Wolfendale & Marquis (2015). It does not cover other wall conductances or
aspect ratios against a published $Nu$, for example Hunt's duct (Bluck & Wolfendale 2017).
