# Heat transport

LMhdX carries heat through a liquid-metal duct while the metal flows. Heat spreads by conduction and
moves with the flow. The electric currents heat the metal (Joule heating), and you can add a heat
source of your own. Warm metal can rise by buoyancy, and the walls can hold a temperature, pass a
heat flux, or conduct and store heat themselves.

```{image} ../_static/thermal_heated_duct.webp
:alt: Heated wall, floor and outlet temperatures of an insulating and a Hunt duct in 3-D as heat enters from one side wall
:align: center
```

[Run a heated duct](../how_to/heated_duct.md) runs a case. The subpages cover [solid walls](thermal/walls.md),
the [developed duct and pipe](thermal/developed.md) and [convection and open ducts](thermal/convection.md).

(thermal-model)=
## Equations and assumptions

The temperature equation, with constant properties, is

$$
\rho_0 c_p\left[\frac{\partial T}{\partial t}+\nabla\cdot(\mathbf u T)\right]
=\nabla\cdot(k\nabla T)+\frac{|\mathbf J|^2}{\sigma}+q,
$$

alongside the inductionless momentum and charge equations of [Equations and
assumptions](equations.md). With buoyancy, the momentum equation gains the force density

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

The model assumes a small magnetic Reynolds number, an incompressible flow, Fourier conduction with
constant properties, and density that varies only in the buoyancy force. The flow should satisfy
$\max\beta|T-T_r|\ll 1$, which the step reports as `boussinesq_measure` but does not enforce. Joule
heat is $|\mathbf J|^2/\sigma$, the dissipation in the moving fluid, without the Lorentz work that
the laboratory-frame $\mathbf J\cdot\mathbf E$ also holds.

The heat equation leaves out these terms:
- viscous heating. Its Brinkman number $\mu U^2/(k\,\Delta T)$ is about $10^{-8}$ for PbLi at
  0.1 m/s and 100 K. The Joule part of the hydraulic power is $Ha^2$ times larger. The duct design
  example measures both and finds them under $10^{-5}$ of its duty, but this depends on the case;
- radiation, phase change, thermoelectric effects and property variation with temperature.

There is no units flag. A run is SI or nondimensional throughout. `duct_problem` is nondimensional
(half width, density, viscosity and conductivity one), so scale its thermal properties the same way.

(thermal-bcs)=
## Boundary conditions

The temperature is periodic on the same axes as the flow. On any other axis, each end prescribes a
wall temperature or an outward conductive heat flux, and the two ends may differ. The thermal,
mechanical and electrical conditions are independent, so an electrically insulating wall need not be
adiabatic. An inflow-outflow axis takes an inlet temperature, a number or a profile, and no
conductive flux at its outlet, for example
`thermal_boundary("temperature", T_in, upper_kind="heat_flux", conductivity=k)`.

A positive `heat_flux` leaves the fluid at both ends. `thermal_boundary` stores it as the Neumann
derivative that `lmhdx.grid.pad` applies, $q/k$ at the lower end and $-q/k$ at the upper.

```{warning}
A heat-flux condition is tied to the conductivity it was built with. After changing
`thermal_conductivity`, rebuild the conditions with `thermal_boundary` and the new $k$, and rebuild
the prepared step. Replacing $k$ alone keeps the old derivative, and so a different heat flux.
```

## Discretization of heat

The temperature lives at the cell centres and the velocity on the faces of the
[staggered grid](numerics/grid.md), and the velocity takes the flow's [time step](numerics/time.md).

(thermal-transport)=
### Heat transport

The transport term is the discrete divergence $D(\mathbf u_f T_f)$ of face fluxes, so it conserves
heat. `ThermalProblem.advection` picks $T_f$ as `"upwind"` (the default), `"van_leer"` or `"off"`.

**Upwind.** Each face carries $u_f T_\mathrm{up}$. A wall face carries no advective heat, a periodic
axis reads its wrap face from its last and first cells, an inlet face brings in the inlet
temperature, and an outlet face takes out the last cell's. Flow returning through an outlet is
boundary data ([open ducts](thermal/convection.md)). On a divergence-free flow it makes no new
extremum beyond the initial, inlet and reservoir temperatures while the Courant number stays below
one. It adds a numerical diffusivity of about $|u|h/2$, so refine a heat-transfer result until it
stops changing.

**Van Leer.** From a cell of width $w$ the face takes $T+\tfrac{w}{2}s$, with $s$ van Leer's mean
$(a|b|+|a|b)/(|a|+|b|)$ of the face gradients on either side.
- The limiter zeroes the slope at any extremum, smooth or kinked, so the order drops below second
  near smooth extrema. The orders quoted here are for a profile without one.
- The face value stays between the values of its two cells, which keeps it bounded on a stretched
  grid. Nothing clips the temperature after a step.
- Faces at an inlet, an outlet and a wall keep their first-order value. The step stays first order
  in time, so the steady convergence does not qualify transient solutions at second order.
- Away from kinks the tested gradients agree with central differences. At a kink
  a derivative need not equal a directional derivative.

On the steady plug flow of the [open duct](thermal/convection.md), the face value converges at order
1.95 and 1.98 over 16, 32 and 64 cells, while upwinding converges at order 0.81 and 0.91.

(thermal-conduction)=
### Conduction, sources and the coupled step

Conduction is implicit. It uses one cell-centred fast-diagonal factorization of
$I-\Delta t\,\alpha L_0$, built under the homogeneous conditions, and enters the boundary data as the
fixed lift $b=L(0)$. The unit shift keeps the operator nonsingular under insulated and periodic
walls, so the step keeps the mean temperature rather than removing it as a gauge.

Joule heat is each face current squared, which `ops.face_average_adjoint` shares between the half
cells on either side. It uses the same face currents as the Lorentz force, and its volume integral is
`core3d.energy_budget(...).joule`. A wall face heats only the fluid's half cell. The dissipation
inside a conducting wall belongs to the wall, and the model leaves it out. In the developed duct the
walls take 56 % at Ha 20 with $c = 0.05$, 76 % at Ha 100, and 43 % at Ha 20 with $c = 0.5$.

Buoyancy goes onto the velocity faces by the same averaging as the Lorentz force. The step zeroes it
on wall faces and adds it before the division by density and the projection. While $\beta>0$,
building the problem refuses a component of $\mathbf g$ along a periodic axis, because no periodic
pressure balances its mean and the step does not subtract the mean temperature (ADR 0008, D32).

The step is first order and partitioned, with every source taken from the old state $n$:

$$
(I-\Delta t\,\alpha L_0)\,T^{n+1}
=T^n-\Delta t\,D\!\left(\mathbf u^n T^n_f\right)
+\frac{\Delta t}{\rho_0 c_p}\left(q_J^n+q\right)+\Delta t\,\alpha\, b .
$$

The velocity takes the projection step of `core3d.step` with the buoyancy of $T^n$, and one electric
solve gives both the Lorentz force and the Joule heat. A state whose temperature varies only along
gravity stays at rest. In general the step converges at first order in $\Delta t$. A heated run takes
`Solver(method="euler")`, and `method="rk3"`, the flow's IMEX scheme, refuses it.

(thermal-time-step)=
### Time-step limits

- **Advective Courant number.** It is $\Delta t$ times the largest outgoing volume-flux rate of any
  cell. `validate_thermal_inputs` refuses a start above `advective_cfl_limit`, 0.8 for upwind (inside
  its derived bound of one) and 0.5 for van Leer (a tested range, not a derived bound). Conduction
  alone has no limit. For an evolving flow the validator checks only the start, so compare the
  `advective_cfl` that each step reports with the limit, as `lmhdx case.toml` does.
- **Buoyancy Courant number.** The step reports `buoyancy_courant` $=\Delta t\sqrt{\beta|\mathbf g|\,G}$,
  with $G$ the largest absolute face-normal gradient on any face. In the Bénard onset (#252) the critical
  Rayleigh number reads 1675.9 near 0.4 and 1675.6 near 0.025, flat to 0.02 % in the step and 1.9 %
  below 1707.76 on the 16-cell mesh. This calibrates that case only.
  `validate_thermal_inputs(..., buoyancy_courant_limit=...)` refuses a start above a chosen limit.

(thermal-ledger)=
## Diagnostics and the heat ledger

Each step returns a `ThermalDiagnostics` of scalars. The heat ledger is

$$
\underbrace{\frac{\rho_0 c_p}{\Delta t}\sum_P V_P\,(T^{n+1}_P-T^n_P)}_{\texttt{storage\_rate}}
+\texttt{conductive\_heat\_out}+\texttt{advective\_heat\_out}-\texttt{joule\_power}-\texttt{external\_power}
=\texttt{heat\_balance\_residual}.
$$

The step computes it from local increments and the same wall gradients the conduction uses. It
closes to its rounding floor, about $\varepsilon\,\rho_0 c_p\sum V\,|T|/\Delta t$ for temperatures
in kelvin, and with solid walls to PCG's tolerance relative to the step's change.
`advective_heat_out` is the enthalpy carried out through an outlet, less what enters at an inlet or
with a returning flow. `finite` says only that every value is a number. To accept a run, check the
Courant numbers, the residuals (`heat_solve_residual`, `mass_residual`, `charge_residual`) and the
Boussinesq measure.

(thermal-envelope)=
## Supported envelope

- **Geometry.** Cartesian grids, with periodic axes and closed no-slip walls. A polar grid works only
  for the [developed pipe](thermal/developed.md). An inflow-outflow duct carries heat on a held or an
  evolving flow ([heat on an open duct](thermal/convection.md)). With momentum transport on, a
  periodic axis needs at least two cells.
- **Walls.** Insulating, thin conducting, resolved and solid electrical walls work, their dissipation
  outside the fluid. The temperature lives on the fluid cells unless `walls=` adds [solid walls](thermal/walls.md).
- **Precision.** The `dtype` of `build_thermal_step` must be the core's compute dtype.
- **Derivatives** work with respect to anything in the state or the heat source and with respect to
  `backflow_temperature`, not with respect to $k$, $c_p$, $\sigma$, $\Delta t$, the geometry or the
  boundary conditions. Upwinding picks its donor by the sign of each face velocity, so a derivative
  is one-sided where a face velocity is exactly zero or flips sign ([derivatives](numerics/derivatives.md)).
- **Memory.** In reverse mode with `checkpoint=True` (the default), `advance_thermal` keeps one state
  per step. `diagnostics=n` keeps the scalars of every `n`-th step and makes the block of `n` steps
  the checkpoint, with the least memory near $n\approx\sqrt{\text{steps}}$.

A `jax.checkpoint`-wrapped chunk in an outer `lax.scan` does the same, and compiles one copy:

```python
chunk = jax.checkpoint(lambda state: advance_thermal(thermal_step, state, heat, 100, diagnostics=100)[0])

def run(state):  # 5000 steps
    final, _ = jax.lax.scan(lambda s, _: (chunk(s), None), state, None, length=50)
    return objective(final)

gradient = jax.jit(jax.grad(run))(start)
```

The gradient matches the plain run's to round-off (`test_the_checkpointing_recipe_has_the_plain_runs_gradient`).
For 1600 differentiated steps on 32 x 16 x 16 cells, one scan takes 418 MiB and 40 chunks of 40 take 21 MiB.

**Not included.** Thermal cells under a thin electrical wall (ADR 0009 D36); solid walls in the
developed duct; solid walls and inflow-outflow axes from TOML (a run from TOML covers closed ducts);
transport above second order; temperature-dependent properties; a steady solve of the coupled
equations; and a `ThermalProblem` through `lmhdx.solve` (a `Problem` with `thermal=Thermal(...)`
runs the same step). A closed domain with net heating has no steady temperature. The step's measured
cost is in [Run a heated duct](../how_to/heated_duct.md).

```{toctree}
:hidden:

thermal/walls
thermal/developed
thermal/convection
```
