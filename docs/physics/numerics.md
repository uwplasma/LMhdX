# Numerical methods

LMhdX discretizes every duct and pipe on one staggered finite-volume grid with a current-conservative electric scheme.
Shared methods, such as the grid, the solvers, the steady solves, the time step and the derivatives, have their own
pages. A scheme that only one physics module uses lives on that module's page. The table lists every scheme and where
it is described.

```{image} ../_static/verification_orders.webp
:alt: The observed order of accuracy of each term of the time step, L-infinity and RMS, near two for every term
:align: center
```

Every term of the time step converges at second order in space ([code verification](../validation/verification.md)).

## Schemes

| Scheme | What it discretizes or solves | Used by | Where |
|---|---|---|---|
| Staggered (marker-and-cell) finite volumes | velocity on faces, pressure and potential at cell centres | every 3-D solver, the pipe, heat transport | {ref}`grid <numerics-grid>` |
| Wall-resolving coordinates | meshes that put cells in the Hartmann and side layers | every 3-D solver | {ref}`grid <numerics-wall-coordinates>` |
| Current-conservative electric scheme (Ni et al.) | the potential equation, face currents and the Lorentz force | every 3-D solver, the pipe, Joule heat | {ref}`grid <numerics-electric>` |
| Thin (Walker) and resolved conducting walls | wall currents, as a sheet node or a solid region | ducts, the pipe | {ref}`grid <numerics-walls>` |
| Divergence-free imposed field | a varying field, such as the ANL fringe | fringe ducts, open ducts | {ref}`grid <numerics-imposed-field>` |
| Fast-diagonal Poisson and Helmholtz solves | pressure, potential, implicit viscosity and conduction | every 3-D solver, heat transport | {ref}`solvers <numerics-poisson>` |
| CG and PCG | the steady Stokes-limit operator | steady ducts, open ducts, the open pipe | {ref}`solvers <numerics-krylov>` |
| Newton-Krylov (flexible GMRES) | the steady open duct with inertia | open ducts | {ref}`solvers <numerics-krylov>` |
| Projection-step preconditioner, varying-field rate (2b.4) | the steady CG solve | steady ducts, open ducts | {ref}`solvers <numerics-preconditioners>` |
| Defect correction | round-off left by the fast solves | steady ducts, open ducts | {ref}`solvers <numerics-defect-correction>` |
| Drive elimination | the drive for a given flow rate, $f=Q/G$ | fully developed ducts | {ref}`steady <numerics-drive-elimination>` |
| Steady state as a differentiable root | steady flows and their derivatives | steady ducts, open ducts | {ref}`steady <numerics-steady-root>` |
| Projection step with an implicit damping shift | one time step of the 3-D core | `lmhdx.core3d`, heat transport | {ref}`time <numerics-projection>` |
| Implicit Euler and IMEX RK3 | time integration | `lmhdx.core3d`, the open-duct march | {ref}`time <numerics-rk3>` |
| Courant and damping step control | the step size | `lmhdx.core3d`, open ducts | {ref}`time <numerics-cfl>` |
| Explicit Lorentz force | transients that the shift would slow | `lmhdx.core3d` runs in time | {ref}`time <numerics-lorentz-shift>` |
| Compiled trajectory (`lax.scan`, checkpointed) | a run in time as one compiled loop | `lmhdx.core3d` | {ref}`time <numerics-trajectory>` |
| Implicit derivatives and adjoints | gradients of converged and transient solves | every solver | {ref}`derivatives <numerics-derivatives>` |
| Inlet lift and outlet face | the inflow and outflow conditions | open ducts | {ref}`open ducts <open-ducts-lift>` |
| March or Newton | the steady flow with inertia | open ducts | {ref}`open ducts <open-ducts-march-newton>` |
| Open polar axis | the pipe with an inlet and an outlet | the open pipe | {ref}`open ducts <open-ducts-pipe>` |
| Variational core-flow discretization | TM-228's core equations | `lmhdx.coreflow` | {ref}`core flow <coreflow-discretization>` |
| Spectral IFRK4 with two-thirds dealiasing | depth-averaged vorticity | `lmhdx.q2d` | {ref}`Q2D <q2d-ifrk4>` |
| Upwind and van Leer heat transport | advection of temperature | `lmhdx.thermal` | {ref}`heat transport <thermal-transport>` |
| Implicit conduction in the fluid | heat conduction | `lmhdx.thermal` | {ref}`heat transport <thermal-conduction>` |
| Conduction through solid walls | conjugate heat transfer | `lmhdx.thermal` | {ref}`solid walls <thermal-walls-solve>` |
| Developed-heat solves | the thermally fully developed duct | `lmhdx.thermal` | {ref}`developed heat <thermal-developed>` |
| Boussinesq buoyancy | the buoyancy force on the velocity faces | `lmhdx.thermal` | {ref}`buoyant convection <thermal-convection>` |

## Run it

The drive elimination is the quickest way to see the numerics at work. One solve measures how much flow a unit drive
gives, and the drive for any flow rate follows exactly. The example takes under half a minute on a laptop CPU.

```python
import jax

from lmhdx.core3d import duct_problem
from lmhdx.steady import channel_flow_rate, channel_flow_response, solve_steady_state

jax.config.update("jax_enable_x64", True)
problem = duct_problem(hartmann=20.0, cells=24)  # square duct, insulating walls, Stokes limit
response = channel_flow_response(problem)  # one solve at unit drive measures G
drive = float(response.drive_for(0.02))  # f = Q / G
solution = solve_steady_state(problem, forcing=(drive, 0.0, 0.0))
flow = float(channel_flow_rate(problem, solution.velocity[0].data[0]))
print(f"G = {float(response.flow_per_unit_drive):.5f}, drive {drive:.4f}, flow rate {flow:.6f}")
# G = 0.15614, drive 0.1281, flow rate 0.020000
```

## How results are checked

Analytical and manufactured tests check the observed order on refined meshes. Production claims also need stable
physical observables, stricter solver tolerances and conservation gates. Timings measure JAX compilation and warm
execution separately.

```{toctree}
:hidden:

numerics/grid
numerics/solvers
numerics/steady
numerics/time
numerics/derivatives
```
