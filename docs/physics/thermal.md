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
- viscous heating. In a developed duct with insulating walls the hydraulic power
  is the viscous plus the Joule dissipation. The Brinkman number
  $\mu U^2/(k\,\Delta T)$ scales the viscous part against conduction: about
  $10^{-8}$ for PbLi at 0.1 m/s and 100 K. It does not bound the Joule part,
  whose scale on the same length is $Ha^2$ times larger, and whose currents
  depend on the walls. Whether either matters is a property of the case; the
  duct design example measures both, under $10^{-5}$ of its duty. Joule heat is
  kept, from the same currents as the Lorentz force, so the ledger matches
  `core3d.energy_budget(...).joule`;
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

An inflow-outflow axis carries an inlet temperature, a number or a profile,
and no conductive flux at its outlet:
`thermal_boundary("temperature", T_in, upper_kind="heat_flux", conductivity=k)`.
Heat leaves by the flow.

`thermal_boundary` builds the condition from physical data. A positive
`heat_flux` leaves the fluid at both ends. It is stored as the derivative along
the axis that `lmhdx.grid.pad` applies to a Neumann condition, $q/k$ at the
lower end and $-q/k$ at the upper.

```{warning}
A heat-flux condition is tied to the conductivity it was built with. After
changing `thermal_conductivity`, rebuild the conditions with `thermal_boundary`
and the new $k$, and rebuild the prepared step. Replacing $k$ alone keeps the
old derivative, and so a different heat flux.
```

## Discretization

**Placement.** The temperature lives at the cell centres and the velocity on
the faces.

**Conduction** is implicit. One cell-centred fast-diagonal factorization of
$I-\Delta t\,\alpha L_0$ is built under the homogeneous conditions, and the
boundary data enter as the fixed lift $b=L(0)$. Its unit shift keeps the
operator nonsingular under insulated and periodic walls, so the mean
temperature is kept rather than removed as a gauge.

**Transport** is conservative, first-order upwind by default:
- Each face carries $u_f T_\mathrm{up}$.
- A wall face carries no advective heat.
- A periodic axis reads its wrap face from its last and first cells.
- An inlet face brings in the inlet temperature. An outlet face takes out the
  last cell's temperature where the flow leaves.
- Where the flow returns through the outlet, the incoming enthalpy is boundary
  data. The fluid comes from outside the model, so its temperature cannot be
  read from the adjacent cell:
  - `outlet_backflow="error"`, the default, refuses such a run.
    `validate_thermal_inputs` rejects any returning outlet face, even when the
    net flow leaves. An unvalidated step reports `backflow_boundary_valid`
    false.
  - `outlet_backflow="reservoir_flux"` brings in $C\,u_n T_\mathrm{res}$, a
    prescribed total (Danckwerts) heat flux. The reservoir temperature
    $T_\mathrm{res}$, a number or a profile over the outlet, is passed to every
    step as `backflow_temperature`. The outlet's conduction operator stays
    homogeneous Neumann, so the fast inverse is unchanged. The condition is
    $C u_n T_b - k\,\partial_n T_b = C u_n T_\mathrm{res}$, not
    $T_b = T_\mathrm{res}$.
  - Recirculation inside the duct needs neither; its upwind donor is another cell.

On a divergence-free flow it makes no new extremum, beyond the range of the
initial, inlet and reservoir temperatures, while $\Delta t$ times every cell's
outgoing volume-flux rate stays below one. `validate_thermal_inputs`
refuses a start whose Courant number, that rate's largest value times
$\Delta t$, exceeds `advective_cfl_limit`, 0.8 by default. Conduction alone
(`advection="off"`) has no such limit. A held flow keeps its Courant number for
the whole run. An evolving flow is checked at its start only, which a start
from rest passes trivially. `advance_thermal` does not check it again, so
compare the `advective_cfl` each step reports with the limit. `lmhdx case.toml`
makes that comparison at every step; see "A run from TOML" below.
Upwinding adds a numerical
diffusivity of about $|u|h/2$, so a heat-transfer result must be refined until
it stops changing.

`advection="van_leer"` gives a second-order face value instead, with the same
conservative flux:
- From a cell of width $w$ the face takes $T+\tfrac{w}{2}s$. Here $s$ is van
  Leer's mean $(a|b|+|a|b)/(|a|+|b|)$ of the face gradients on either side,
  zero at an extremum. The limiter zeroes the slope at a smooth extremum as well
  as at a kink, so the reconstruction there is first order and the scheme's
  order drops below second near smooth extrema. The orders quoted below are for
  a profile without one.
- The face value is held between the two cells it lies between, which keeps it
  bounded on a stretched grid. A linear temperature on the tanh-stretched test
  grid reads exactly.
- The validator's default Courant limit is then 0.5. Upwinding's 0.8 sits
  inside its derived bound of one; van Leer's 0.5 is a tested envelope, not a
  derived bound. Ten steps at 0.5 make no new extremum on the test grids:
  - a tanh-stretched grid;
  - a grid stretched without a ratio limit, whose periodic axis puts widths 27
    to 1 side by side at the wrap.

  One- and two-cell periodic axes carry the same flux through both copies of
  the wrap face. A returning outlet's ledger closes with either scheme. Other
  meshes are not qualified.
- Nothing clips the temperature after a step. Holding the face value between
  its cells is part of the reconstruction.
- Faces at an inlet, an outlet and a wall keep their first-order value.
- The step stays first order in time. The steady convergence below does not
  qualify transient solutions at second order.
- The reconstruction is piecewise smooth, with kinks where a slope vanishes or
  changes sign and where a face value is held between its cells. Away from
  them, the tested gradients agree with central differences: the buoyant duct's
  end-to-end gradients with either scheme, and the face value itself on a field
  with no extremum. At a kink, automatic differentiation follows the
  implementation's derivative conventions, and the result is not guaranteed to
  equal a directional derivative. A finite gradient at a uniform state does not
  establish differentiability there.

On the steady plug flow of the open duct below, the face value converges at
order 1.95 and 1.98, where upwinding converges at order 0.81 and 0.91:

| Cells | Upwind error | Van Leer error |
|--:|--:|--:|
| 16 | 4.6×10⁻² | 4.8×10⁻³ |
| 32 | 2.6×10⁻² | 1.2×10⁻³ |
| 64 | 1.4×10⁻² | 3.1×10⁻⁴ |

**Joule heat** is each face current squared, then shared between the half
cells on either side by `ops.face_average_adjoint`:
- A wall face heats only the fluid's half cell.
- The dissipation inside a conducting wall is the wall's, not the fluid's.
- The heat comes from the full face currents, the same ones the Lorentz force
  is built from.
- Its volume integral is `core3d.energy_budget(...).joule`.

Thin conducting walls take much of the dissipation away from the fluid. In the
developed duct at Ha 20 with $c = 0.05$ the walls dissipate 56 % of it, at
Ha 100 76 %, and at Ha 20 with $c = 0.5$ 43 % (`energy_budget(...).wall` against
`.joule`, unchanged from 24 to 48 cells). The fluid-only model leaves that heat
out until conjugate walls arrive.

**Buoyancy** is averaged onto the velocity faces as the Lorentz force is, and
zeroed on wall faces. It is added before the division by density and the
projection. Without $\beta$ or $\mathbf g$ there is no force term at all.
While $\beta>0$, a component of $\mathbf g$ along a periodic axis is refused
when the problem is built: no periodic pressure balances its mean, and the mean
temperature is not subtracted (ADR 0008, D32). With $\beta=0$ a stored
$\mathbf g$ is inert.

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
is first-order convergence in $\Delta t$. Backward-Euler conduction is first
order in time as well, so any transient heat-transfer number carries an
$O(\Delta t)$ error.

The coupling's time error is set by the step over the buoyancy time scale,
`buoyancy_courant` $=\Delta t\sqrt{\beta|\mathbf g|\,G}$, reported beside
`advective_cfl`, with $G$ the largest absolute face-normal gradient on any face,
one component rather than a colocated gradient magnitude. In the Bénard onset of
#252 the critical Rayleigh number is about 15 % low near 0.2 and under 3 % near
0.025, at first order. That calibrates that case; it is not a general bound.
`validate_thermal_inputs(..., buoyancy_courant_limit=...)` refuses a start
above a chosen limit.

## Diagnostics and the heat ledger

Each step returns a `ThermalDiagnostics` of scalars. The heat ledger is

$$
\underbrace{\frac{\rho_0 c_p}{\Delta t}\sum_P V_P\,(T^{n+1}_P-T^n_P)}_{\texttt{storage\_rate}}
+\texttt{conductive\_heat\_out}+\texttt{advective\_heat\_out}-\texttt{joule\_power}-\texttt{external\_power}
=\texttt{heat\_balance\_residual},
$$

computed from local increments and the same wall gradients the conduction uses.
`advective_heat_out` is the enthalpy the flow carries out through an outlet,
less what it brings in at an inlet or with a returning outlet flow. It is zero
in a closed duct.
At the outlet, `outlet_outflow_rate` and `outlet_backflow_rate` are the volume
rates leaving and returning, and `outlet_backflow_enthalpy_in` is the
enthalpy the return brings in. `backflow_boundary_valid` is separate from
`finite`.
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
- Cartesian grids, with periodic axes and closed no-slip walls. Polar grids are
  refused when the problem is built.
- An inflow-outflow duct carries its heat on a held flow:
  - pass the steady velocity of `axial.solve_open_duct` in the state;
  - build the step with `hold_flow=True`;
  - a held flow takes no buoyancy.
- With momentum transport on (`ChannelProblem.advection` other than `"off"`), a
  periodic axis needs at least two cells.
- Insulating, thin conducting and resolved electrical walls all work, with
  their electrical dissipation kept outside the fluid. Resolved `wall_layers`
  widen only the electric solve: the temperature lives on the fluid cells
  alone, with the fluid's conductivity, and its conditions apply at the fluid's
  wall faces.

**Precision.** The `dtype` given to `build_thermal_step` must be the core's
compute dtype: float64 after `lmhdx.enable_x64()`, else float32. Set it before
building.

**Derivatives** are supported with respect to anything carried in the state or
the heat source, for example the initial temperature or a source amplitude, and
with respect to `backflow_temperature`. They
are not supported with respect to $k$, $c_p$, $\sigma$, $\Delta t$, the geometry
or the boundary conditions, which are built into the prepared step. Upwinding
picks its donor by the sign of each face velocity. A derivative is therefore
one-sided only where a face velocity is exactly zero, or where the perturbation
flips a face's sign; nothing is smoothed to hide it.

**Memory.** `advance_thermal` keeps no field history. In reverse mode, with
`checkpoint=True` (the default), it keeps one state per step and recomputes each
step's intermediates, so the memory grows with the number of steps. A restart
record and the command's chunks are not reverse-mode checkpoints and do not
bound that memory; the command is not differentiated end to end.

`diagnostics=n` keeps the scalars of every `n`-th step only (`steps` must be a
multiple of `n`); the default keeps every step's. With `checkpoint=True` the
block of `n` steps becomes the checkpoint: one state per block is stored, but a
block's reverse pass holds the intermediates of all `n` of its steps. It trades
the first memory for the second, the saving being largest near
$n\approx\sqrt{\text{steps}}$. The steps it skips are neither kept nor checked,
so `lmhdx case.toml`, which accepts a run on every step's diagnostics, stays at
stride 1.

**Checkpointing a long differentiated run.** Split the run into chunks and
checkpoint across them: a `jax.checkpoint`-wrapped chunk stores only its start
state in the forward pass and recomputes its steps in reverse, so the memory is
one state per chunk plus one chunk's steps, not one state per step. Loop over
the chunks with an outer `lax.scan`, so that a jitted objective compiles one
copy of the chunk, not one per chunk.

```python
chunk = jax.checkpoint(lambda state: advance_thermal(thermal_step, state, heat, 100, diagnostics=100)[0])

def run(state):  # 5000 steps
    final, _ = jax.lax.scan(lambda s, _: (chunk(s), None), state, None, length=50)
    return objective(final)

gradient = jax.jit(jax.grad(run))(start)
```

The gradient is the plain run's to round-off
(`test_the_checkpointing_recipe_has_the_plain_runs_gradient`). This is the same
computation as `diagnostics=100` with `checkpoint=True` on the plain call; the
loop spells out the chunks.

Measured, as the compiled temporary memory of `jax.jit(jax.grad(objective))`
(`.lower(start).compile().memory_analysis()`, 64-bit CPU, held flow, upwind),
for 1600 steps on 32 x 16 x 16 cells, where one state is 0.06 MiB for the
temperature alone:

| Run | Compiled memory (MiB) |
|---|---|
| Plain, one scan of 1600 steps | 418 |
| 160 chunks of 10 steps | 46 |
| 40 chunks of 40 steps | 21 |
| 10 chunks of 160 steps | 37 |

The same 1600 steps with `diagnostics=10`, `40` and `160` on the plain call give
the same 46, 21 and 37 MiB: the stride is the chunk. The minimum is near the
square root of the step count, 40; either side of it, the stored states or one
block's intermediates dominate. The figures are for this grid and step; they
scale with the cells, and the ratio is what carries over.

**Not included**
- Conjugate wall conduction.
- A flow that evolves in an open duct with the heat. There the flow is held.
- Transport above second order. The open-duct results below use first-order
  upwinding and quote $Pe_h/2$, its numerical diffusivity over the physical one;
  `advection="van_leer"` removes most of it.
- Variable properties.
- A steady solve of the coupled equations. The developed duct below is the one steady thermal solve, for a given flow.
- An inflow-outflow axis read from TOML. A run from TOML (below) covers closed
  ducts; an open one runs from Python.
- Dispatch through `lmhdx.solve`.

A closed domain with net heating has no steady temperature. The state returned
is not a `cases.MHDState` or an archive-compatible `Solution`. The final
electric fields come from `core3d.face_currents(state.velocity, flow)`.

## The thermally fully developed duct

A duct can be periodic in its flow direction, with the velocity and the heat
sources uniform along it. Under a uniform axial heat input its temperature is
then $T=\gamma x+\theta(y,z)$, where $\gamma$ is the bulk temperature's axial
gradient and

$$
k\,\nabla^2\theta=\rho_0 c_p\,u\,\gamma-q-q_J .
$$

`developed_heat(problem, velocity, heat_source, axial_gradient=...)` solves
this once on the cross-section, with the fast-diagonal Poisson factorization
(ADR 0008, D33). How $\gamma$ is set depends on the walls:
- **Heat-flux walls only.** $\gamma$ follows from the heat balance
  $\rho_0 c_p\gamma Q$ = heat in per unit length, and $\theta$ has zero mean. A
  uniform flux is the H2 condition.
- **Any prescribed-temperature wall.** That wall takes in heat the solve
  decides, so $\gamma$ is given. A peripherally uniform wall temperature with a
  uniform axial heat input is the H1 condition.

The duct is one cell long along its axis. `validate_developed_inputs` checks
on the host that the flow is the reduction's single throughflow, with
tolerances in physical units:
- a flow rate $\sum A u$ above `flow_rate_tolerance`, a volume rate;
- the same axial velocity on both axial faces, and no transverse velocity or
  reversed axial velocity, beyond `velocity_tolerance`, a speed;
- a finite `axial_gradient` where one is given.

A flow that stops gives NaN for the bulk temperature and $Nu$, not a division
by zero.

The result gives:
- the bulk temperature, weighted by the axial flow;
- the area-mean and peak wall temperatures;
- the mean wall heat flux into the fluid;
- $Nu=q_w D_h/\left(k\,(T_w-T_b)\right)$, with $D_h=4A/P$;
- the heat-balance residual.

The wall means are over every transverse wall, heated or not. With an internal
source, $Nu$ is an apparent number for that source, not a coefficient of the
duct alone. It is NaN where $|T_w-T_b|$ is within $64\varepsilon(L/h)^2$ of the
largest temperature scale. Here $L/h$ is the section's largest extent over its
smallest cell, since the solve's rounding grows with its conditioning. The
scales are those of the solution and each heat input's own
($D_h^2|\rho_0c_pu\gamma|/k$, $D_h^2|q+q_J|/k$ and $D_h|q_w|/k$). A
difference cancelled to rounding is therefore undefined, as when heat crosses
the duct from wall to wall. The temperatures and the heat balance stay valid.

With heat-flux walls only, $\theta$ is fixed only up to a constant, its zero
mean. The physical temperature is $T=T_{b,\mathrm{in}}+\gamma x+\theta-\theta_b$,
and `DevelopedHeat.anchored_peaks(T_in, length)` returns the hottest wall face
and hottest cell over the length, at whichever end $\gamma$ rises toward. With a
prescribed wall temperature, whose station values already fix the reference,
both are NaN. It
takes the profile as developed from the inlet. An entrance region needs the
open duct below.

It differentiates through the velocity, and so through the implicit derivative
of `steady.solve_steady_state` in a field scale or a drive. It also
differentiates through the heat source and $\gamma$, but not through the
geometry.

The H1 Nusselt number is checked against its exact double-sine series, which
involves no discrete operator. The flow is the package's own steady solve, or a
uniform slug:

| Duct | Exact | 16 / 32 / 64 cells | Order |
|:--|--:|:--|:--|
| square | 3.60795 | +1.35 % / +0.35 % / +0.087 % | 1.97, 1.99 |
| aspect ratio 0.5 | 4.12330 | +2.7 % / +0.73 % / +0.18 % | 1.92, 1.98 |
| square, slug flow | 7.11354 | −1.5 % / −0.38 % / −0.094 % | 1.98, 1.99 |
| square, H2 (uniform flux on every wall) | 3.091 | +1.3 % / +0.24 % / −0.027 % | 1.97 |

The H2 gate exercises the heat-flux path the design workflow uses. Its 3.091 is
Shah & London's (1978), as cited in the review of #240; Richardson extrapolation
gives 3.0873, 0.12 % below. The H1 numbers' comparison with the book's tables
waits for the book.
The aspect-0.5 duct is gated at 64 cells; the square ducts carry the
refinement study.

A transverse field flattens the core. On 32 cells with Joule heat off, the H1
Nusselt number of the square duct rises from 3.62 at Ha 0 to 4.56 at Ha 10 and
5.35 at Ha 30, toward the slug value of 7.09.

**An MHD reference.** Hartmann flow between plates at $y=\pm a$, each taking
in a uniform heat flux with no source and no Joule heat, has
$Nu=4/\int_0^1F_H^2\,ds$ with $D_h=4a$. Here $F_H$ is the integral of the
mean-normalized Hartmann velocity $f_H=(1-\cosh(Hs)/\cosh H)/(1-\tanh H/H)$.
The reference is computed by quadrature, with no discrete operator. It runs
from $140/17$ at Ha 0 to 12 as Ha grows.

The test gives the solver the cell means of the analytical velocity, across
one cell with insulated ends, and measures the plates' own $Nu$:

| Flow | Exact | 32 / 64 / 128 cells | Order |
|:--|--:|:--|:--|
| Poiseuille, Ha 0 | 8.23529 | −2.6e‑7 / −1.6e‑8 / −1.0e‑9 | 4.0 |
| Ha 10 | 10.25858 | −1.2e‑5 / −7.6e‑7 / −4.7e‑8 | 4.0 |
| slug | 12 | −1.9e‑3 / −4.9e‑4 / −1.2e‑4 | 2.0 |

With the velocity's cell means the discrete fluxes are exact, and the
$O(h^2)$ error of $T_w-T_b$ is $+h^2\theta''_w/6$, half from the wall value and
half from the bulk. Since $k\theta''=\rho_0c_p\gamma u$, no slip removes it;
with point values of the velocity the order falls to 2. Slug flow keeps it, and
its discrete answer is exactly $12/(1+h^2/2)$. The public four-wall
`nusselt` of that fixture averages in the insulated ends, and is a different
number.

**The uniform wall temperature.** With every wall held at one temperature, the
T condition, the excess over the wall decays along the duct in a fixed shape,
$T-T_w=\theta\,e^{-\lambda x}$. Without axial conduction or internal heat,
$\theta$ is the eigenfunction of $-L(\theta)=\Lambda f\theta$ with the smallest
eigenvalue, $f=u/\bar u$. `developed_wall_temperature(problem, velocity)` finds
it by inverse iteration with the same fast-diagonal factorization, and returns:
- the decay rate $\lambda=\Lambda\alpha/\bar u$;
- the profile scaled to a unit bulk excess, $(T-T_w)/(T_b-T_w)$;
- $Nu_T=\Lambda D_hA/P_w$, which is $\Lambda D_h^2/4$ when every wall is held, with
  the wall flux averaged over the held perimeter $P_w$.

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
print(f"Nu_T {float(result.nusselt):.4f} in {int(result.iterations)} iterations")  # 2.9823 in 13
```

The iteration stops once `eigen_residual` is below `tolerance` (1e-10 in
float64, 1e-4 in float32), or after `max_iterations`. Each iteration gains the
ratio of the two smallest eigenvalues of the section's symmetric modes: the
square duct takes 13 iterations, a 1:8 duct about 150.

$\Lambda$ is the Rayleigh quotient of the converged $\theta$. The quotient is
stationary at an eigenfunction, so its derivative in the velocity needs none of
the iteration's; the gradient through the steady solve's field scale meets
central differences to 1e-5. A heat-flux wall must be adiabatic, and sources
and Joule heat are left out, since the T condition has none.

| Duct | Reference | 16 / 32 / 64 cells | Order |
|:--|--:|:--|:--|
| square, slug flow | $\pi^2/2=4.93480$; discretely $(8/h^2)\sin^2(\pi h/4)$ | exact to 3e-16 at each | — |
| square | 2.97753 | +0.63 % / +0.16 % / +0.040 % | 1.97 |
| plates, Poiseuille (32 / 64 / 128) | 7.54070 | −4.6e‑4 / −1.1e‑4 / −2.9e‑5 | 2.0 |
| plates, Ha 10 (32 / 64 / 128) | 8.96255 | −6.5e‑4 / −1.6e‑4 / −4.1e‑5 | 2.0 |

The square duct's reference is a Chebyshev collocation of the same
eigenproblem, extrapolated from 40, 48 and 56 nodes at its observed order 2.0.
The package's own Richardson value, 2.97749, agrees with it to 1.5e-5. Shah &
London (1978) list 2.976, 0.05 % below both. The sampled sines of slug flow are
exact eigenfunctions of the cell-centred Dirichlet Laplacian, which is why that
row is exact. The plates carry the cell means of the analytical velocity, with
the plates' own $D_h=4a$; their references are one-dimensional Chebyshev solves.

Not covered:
- the pipe;
- an MHD duct against a published $Nu$, for example a square insulating duct
  near Ha 100 (Tao & Ni 2013; Smolentsev et al.). That is a documentation
  campaign to come, not a CI gate.

## An open duct

The heated Hunt duct of step 5.3 carries its heat from the inlet to the
outlet. The setup:
- Ha 20, with thin conducting Hartmann walls (c = 0.05) and insulating side
  walls;
- a duct 8 half-widths long with a fully developed inlet at a flow rate of 4;
- the inlet at temperature 0;
- one side wall taking in a unit heat flux;
- the Joule heat of the current.

Over 400 steps the heat balance closes to $2.5\times10^{-14}$ of its largest
term at every step, against the plan's $10^{-6}$. The duct then reaches its
steady state:
- the storage falls to $3\times10^{-11}$;
- the enthalpy leaving, 124.6, is the Joule heat, 110.3, plus the net wall heat,
  14.3. That is the 16 entering through the heated wall, less what conducts
  back out through the inlet.

Its axial cells give $Pe_h/2 = 5$: first-order upwinding carries five times the
physical axial diffusivity. That is why the duct is quoted for its balance,
not for a Nusselt number.

A plug flow carrying a uniform source converges to its exact steady profile at
first order, 0.81 and then 0.91, at $Pe_h/2$ of 0.16, 0.08 and 0.04.

**Crediting a load.** A design that must remove a given duty is judged on what
the duty itself does, not on the gross outlet enthalpy. Joule heat and a
recirculating hot flow also raise the gross figure.
`load_attribution(step, state, background, load, steps)` runs the held flow
twice, with and without the load, from the same start and with the same
velocity, Joule heat, inlet, reservoir and walls. The difference is the load's
own response, since the temperature is affine in the heat source at constant
properties, with upwind or no transport. Van Leer's face value is not affine in
the temperature: its mean has $s(1,3)+s(3,1)=3$ but $s(4,4)=4$. On a held flow
the load's response then depends on the background, so a van Leer step is
refused. It reports the load's power and what the load adds:
- to the enthalpy leaving through the outlet;
- to the conduction out, which is what leaks through the inlet and any
  prescribed-temperature wall, together (a prescribed wall flux cancels);
- to the storage.

A buoyant flow, which the load moves, is not affine in it. The load is a heat
source on the cells: a wall's heat flux is boundary data, present in both runs.
To credit one, pass the source $qA/V$ in the cells along that wall and set the
wall's flux to zero. With upwind or no transport, a face flux enters its cell's
equation exactly as that source, so the cell temperatures are the same. Van
Leer's face value reads the wall's gradient, so there the two can differ.

In a plug flow with $\lambda=\rho_0c_pu/k$, a uniform load leaves through the
outlet in the steady fraction $1-(1-e^{-\lambda})/\lambda$. The rest conducts
back out of the thermostatted inlet. On 32 cells the credited fraction is
0.788 at $\lambda=5$, against 0.801 exactly; the gap is first-order upwind
diffusion, at $Pe_h/2 = 0.08$. A conductive fluid ($\lambda=0.5$) stays coolest only by returning
79 % of its load through the inlet. A field's Joule heat multiplies the gross
outlet enthalpy, from 7.9 to 32.8 as the flow doubles, but adds nothing to the
load's credit.

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

## A run from TOML

A thermal case runs from a TOML file, `lmhdx case.toml`. The keys are the Python
argument names, and `schema_version = 1` (the only version, and the default) opens
the file. Before the run is built, every table, the top level included,
refuses an unknown or missing key and a value of the wrong TOML type: a count is
an integer and a flag a boolean, neither converted.
- `[grid]`: each axis as `{cells, lower, upper}`, with `layer_thickness` and
  `cells_in_layer` for wall-resolving faces;
- `[flow]`: `ChannelProblem`'s values, with `conditions` `"periodic"` or `"wall"`
  per axis and `initial_velocity` `"rest"` or `"steady"`;
- `[thermal]`: `ThermalProblem`'s values, `hold_flow`, and a uniform
  `initial_temperature` and `heat_source`;
- `[thermal.y]` and `[thermal.z]`: each wall axis's `thermal_boundary` (`kind`
  `"temperature"` or `"heat_flux"`, `lower`, `upper`), required for a wall axis
  and refused for a periodic one;
- `[run]`: `steps`, compiled in chunks of `chunk`, and optional limits,
  `advective_cfl_limit` and `buoyancy_courant_limit`;
- `[output]`: the `directory`, `write_paraview` and `write_restart`;
- `[restart]`: an optional record's `path`, and its `mode`.

```toml
schema_version = 1

[case]
name = "heated_duct"
units = "nondimensional"

[solver]
kind = "thermal"

[grid]
x = { cells = 1, lower = 0.0, upper = 1.0 }
y = { cells = 8, lower = -1.0, upper = 1.0 }
z = { cells = 8, lower = -1.0, upper = 1.0 }

[flow]
conditions = ["periodic", "wall", "wall"]
magnetic_field = [0.0, 10.0, 0.0]
forcing = [1.0, 0.0, 0.0]
dt = 2.0e-3

[thermal]
specific_heat = 1.0
thermal_conductivity = 0.05
reference_temperature = 0.0
expansion_coefficient = 0.02
gravity = [0.0, 0.0, -1.0]
heat_source = 0.5

[thermal.y]
kind = "heat_flux"

[thermal.z]
kind = "heat_flux"
upper = 1.0

[run]
steps = 40
chunk = 10

[output]
directory = "out"
```

Each chunk is accepted or rejected after it runs. Every sample in it, not only
the last, must pass these checks:
- it is finite, and the outlet is valid;
- its advective Courant number is within the scheme's limit: 0.8 for upwind,
  0.5 for van Leer, none for conduction alone. `advective_cfl_limit` replaces that limit, and the
  summary records it as an override;
- its buoyancy Courant number is within `buoyancy_courant_limit`, if one is
  given. That limit is a policy calibrated on particular cases, not a stability
  bound.

The start is held to the same limits. An accepted chunk writes two files:
- a record, `heated_duct_000010.npz` and so on;
- a frame, `heated_duct_000010.vtr`, gathered in `heated_duct.pvd`.

The first rejected chunk stops the run. It writes only
`heated_duct_000020_rejected.npz`. That record names its reasons in its run
provenance, so it can be imported but not resumed.

`thermal_summary.json` holds these entries:
- `accepted`;
- the first rejected sample's step, reasons and Courant numbers;
- the limits;
- the last diagnostics and `time`, the time they are at: on a rejection both belong to the rejected chunk
  (its last sample and its end), while `steps` counts the accepted steps only;
- the run's largest heat balance residual, Courant numbers and Boussinesq
  measure. No limit applies to the residual or the Boussinesq measure.

The command exits with 2 if the run was rejected.

To resume, name a record. Its time sets the step count of the frames that
follow, gathered in a series of their own, `heated_duct_from_000020.pvd`, so
that the first series is kept:

```toml
[restart]
path = "out/heated_duct_000020.npz"
```

With `steps = 20`, that run ends on the same record, bit for bit, as the run of
40 steps. Each record keeps the run's heat source and `hold_flow`:
- `mode = "resume"`, the default, refuses a record written with others;
- `mode = "import"` starts another experiment from the record's state.

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
