# Heat transport

LMhdX can carry heat through a liquid-metal duct while the metal flows. Heat spreads by conduction
and moves with the flow. The electric currents heat the metal (Joule heating), and you can add a
heat source of your own. Warm metal can also rise by buoyancy, and the walls can hold a temperature,
pass a heat flux, or conduct and store heat themselves.

```{image} ../_static/thermal_heated_duct.webp
:alt: Heated wall, floor and outlet temperatures of an insulating and a Hunt duct in 3-D as heat enters from one side wall
:align: center
```

## Run it

This Python script runs the heated duct of `examples/heated_duct.toml`. The duct is periodic along
its length, in a field of 10 across it. A volume source heats it, its top wall cools it, and
buoyancy is on.

```python
import lmhdx
from lmhdx import Problem, Sides, Solver, Thermal, Uniform, Wall
from lmhdx.grid import Grid, axis

lmhdx.configure(dtype="float64")
problem = Problem(
    Grid(axis(1, 0.0, 1.0), axis(8, -1.0, 1.0), axis(8, -1.0, 1.0)),  # one periodic cell along the duct
    bc=dict(x="periodic", y=Wall(heat_flux=0.0), z=Sides(Wall(heat_flux=0.0), Wall(heat_flux=1.0))),
    B=Uniform(0.0, 10.0, 0.0),
    forcing=(1.0, 0.0, 0.0),
    thermal=Thermal(
        specific_heat=1.0,
        thermal_conductivity=0.05,
        reference_temperature=0.0,
        expansion_coefficient=0.02,  # Boussinesq buoyancy, with gravity along -z
        gravity=(0.0, 0.0, -1.0),
        heat_source=0.5,
    ),
)
solution = lmhdx.solve(problem, Solver(dt=2.0e-3, steps=40))
history = solution.history  # one entry per step: temperatures, heat flows, residuals
print(f"T max {float(history.temperature_max[-1]):.4f}")  # T max 0.0402
```

The same case runs from the command line as `lmhdx examples/heated_duct.toml`
(see "A run from TOML" below). "A worked example" builds the step by hand with
`lmhdx.thermal`, which gives full control. The rest of the page covers the equations, the boundary
conditions, the numerics, the checks, and what LMhdX does not support. The long result tables are
at the end, in {ref}`Reference results <thermal-reference-results>`. ADR 0008 records the design.

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

The assumptions are a small magnetic Reynolds number, an incompressible flow, Fourier conduction
with constant properties, and density varying only in the buoyancy force. The flow should satisfy
$\max\beta|T-T_r|\ll 1$. The step reports this quantity as `boussinesq_measure` but does not
enforce it.

The heat equation leaves out:
- viscous heating. In a developed duct with insulating walls, the hydraulic power is the viscous
  plus the Joule dissipation. The Brinkman number $\mu U^2/(k\,\Delta T)$ compares the viscous part
  with conduction. It is about $10^{-8}$ for PbLi at 0.1 m/s and 100 K. It does not bound the Joule
  part, which is $Ha^2$ times larger on the same length and depends on the walls through the
  currents. Whether either matters depends on the case. The duct design example measures both and
  finds them under $10^{-5}$ of its duty. The heat equation keeps Joule heat, computed from the same
  currents as the Lorentz force, so the ledger matches `core3d.energy_budget(...).joule`;
- radiation, phase change, thermoelectric effects and property variation with temperature.

Joule heat is $|\mathbf J|^2/\sigma$, the dissipation in the moving fluid. The laboratory-frame
$\mathbf J\cdot\mathbf E$ also contains the Lorentz work the momentum equation already carries.

There is no units flag. A run uses SI throughout or is nondimensional throughout. `duct_problem` is
nondimensional (half width, density, viscosity and conductivity one), so you must scale its thermal
properties the same way.

## Boundary conditions

The temperature is periodic on the same axes as the flow. On any other axis, each end prescribes a
wall temperature or an outward conductive heat flux, and the two ends may differ. The thermal,
mechanical and electrical conditions are independent. An electrically insulating wall need not be
adiabatic.

An inflow-outflow axis carries an inlet temperature, a number or a profile, and no conductive flux
at its outlet, for example
`thermal_boundary("temperature", T_in, upper_kind="heat_flux", conductivity=k)`.
Heat leaves with the flow.

`thermal_boundary` builds the condition from physical data. A positive `heat_flux` leaves the fluid
at both ends. `thermal_boundary` stores it as the derivative along the axis that `lmhdx.grid.pad`
applies to a Neumann condition, $q/k$ at the lower end and $-q/k$ at the upper.

```{warning}
A heat-flux condition is tied to the conductivity it was built with. After
changing `thermal_conductivity`, rebuild the conditions with `thermal_boundary`
and the new $k$, and rebuild the prepared step. Replacing $k$ alone keeps the
old derivative, and so a different heat flux.
```

## Discretization

- **Placement.** The temperature lives at the cell centres and the velocity on the faces.
- **Conduction** is implicit. It uses one cell-centred fast-diagonal factorization of
  $I-\Delta t\,\alpha L_0$, built under the homogeneous conditions, and enters the boundary data as
  the fixed lift $b=L(0)$. The unit shift keeps the operator nonsingular under insulated and
  periodic walls, so the step keeps the mean temperature rather than removing it as a gauge.

**Transport** is conservative, first-order upwind by default:
- Each face carries $u_f T_\mathrm{up}$. A wall face carries no advective heat, and a periodic axis
  reads its wrap face from its last and first cells.
- An inlet face brings in the inlet temperature. An outlet face takes out the last cell's
  temperature where the flow leaves.
- Flow returning through the outlet comes from outside the model, so the step cannot read its
  temperature from the adjacent cell. Its enthalpy is boundary data, and `outlet_backflow` chooses
  how to treat it:
  - `outlet_backflow="error"`, the default, refuses such a run. `validate_thermal_inputs` rejects
    any returning outlet face, even when the net flow leaves. A step run without validation reports
    `backflow_boundary_valid` as false.
  - `outlet_backflow="reservoir_flux"` brings in $C\,u_n T_\mathrm{res}$, a prescribed total
    (Danckwerts) heat flux. The condition is $C u_n T_b - k\,\partial_n T_b = C u_n T_\mathrm{res}$,
    not $T_b = T_\mathrm{res}$. Each step receives the reservoir temperature, a number or a profile
    over the outlet, as `backflow_temperature`. The outlet's conduction operator stays homogeneous
    Neumann, so the fast inverse is unchanged.
  - Recirculation inside the duct needs neither, because its upwind donor is another cell.
- On a divergence-free flow, upwind transport makes no new extremum beyond the range of the initial,
  inlet and reservoir temperatures, as long as the Courant number stays below one. The Courant
  number is $\Delta t$ times the largest outgoing volume-flux rate of any cell.
  `validate_thermal_inputs` refuses a start above `advective_cfl_limit`, 0.8 by default. Conduction
  alone (`advection="off"`) has no such limit.
- A held flow keeps its Courant number for the whole run. For an evolving flow the validator checks
  only the start, which a start from rest always passes, and `advance_thermal` does not check it
  again. Compare the `advective_cfl` that each step reports with the limit, as `lmhdx case.toml`
  does at every step ("A run from TOML").
- Upwinding adds a numerical diffusivity of about $|u|h/2$, so refine a heat-transfer result until
  it stops changing.

`advection="van_leer"` gives a second-order face value instead, with the same conservative flux:
- From a cell of width $w$ the face takes $T+\tfrac{w}{2}s$. Here $s$ is van Leer's mean
  $(a|b|+|a|b)/(|a|+|b|)$ of the face gradients on either side, which is zero at an extremum. The
  limiter zeroes the slope at a smooth extremum as well as at a kink, so the order drops below
  second near smooth extrema. The orders quoted here are for a profile without one.
- The scheme holds the face value between the values of its two cells, which keeps it bounded on a
  stretched grid. A linear temperature on the tanh-stretched test grid reads exactly. Nothing clips
  the temperature after a step.
- For van Leer the validator's default Courant limit is 0.5. This is a tested range, not a derived
  bound. Upwinding's 0.8 lies inside its derived bound of one. Ten steps at 0.5 make no new
  extremum on a tanh-stretched grid. They make none either on a grid stretched without a ratio
  limit, whose periodic axis puts widths 27 to 1 side by side at the wrap. One- and two-cell
  periodic axes carry the same flux through both copies of the wrap face, and a returning outlet's
  ledger closes with either scheme. The tests qualify no other meshes.
- Faces at an inlet, an outlet and a wall keep their first-order value. The step stays first order
  in time, so the steady convergence does not qualify transient solutions at second order.
- The reconstruction is piecewise smooth. It has kinks where a slope vanishes or changes sign and
  where the scheme holds a face value between its cells. Away from the kinks, the tested gradients
  agree with central differences. The tests cover the buoyant duct's end-to-end gradients with
  either scheme, and the face value itself on a field with no extremum. At a kink, automatic
  differentiation follows the implementation's derivative conventions, and its result need not
  equal a directional derivative. A finite gradient at a uniform state does not establish
  differentiability there.
- On the steady plug flow of the open duct below, the face value converges at order 1.95 and 1.98,
  while upwinding converges at order 0.81 and 0.91
  ({ref}`Reference results <thermal-reference-results>`, "Van Leer convergence").

**Joule heat** is each face current squared, which `ops.face_average_adjoint` shares between the
half cells on either side. It uses the same full face currents as the Lorentz force, and its volume
integral is `core3d.energy_budget(...).joule`. A wall face heats only the fluid's half cell. The
dissipation inside a conducting wall belongs to the wall, not the fluid. Thin conducting walls take
much of it. In the developed duct the walls dissipate 56 % at Ha 20 with $c = 0.05$, 76 % at Ha
100, and 43 % at Ha 20 with $c = 0.5$. These are `energy_budget(...).wall` against `.joule`, and
they do not change from 24 to 48 cells. The fluid-only model leaves that heat out. Solid walls,
described below, leave it out too, because the benchmark they target has no Joule heat.

**Buoyancy** goes onto the velocity faces by the same averaging as the Lorentz force. The step
zeroes it on wall faces and adds it before the division by density and the projection. Without
$\beta$ or $\mathbf g$ there is no force term at all. While $\beta>0$, building the problem refuses
a component of $\mathbf g$ along a periodic axis. No periodic pressure balances its mean, and the
step does not subtract the mean temperature (ADR 0008, D32). With $\beta=0$ a stored $\mathbf g$
has no effect.

**The step** is first-order and partitioned, with every source taken from the old state $n$:

$$
(I-\Delta t\,\alpha L_0)\,T^{n+1}
=T^n-\Delta t\,D\!\left(\mathbf u^n T^n_\mathrm{up}\right)
+\frac{\Delta t}{\rho_0 c_p}\left(q_J^n+q\right)+\Delta t\,\alpha\, b .
$$

- The velocity takes the predictor and projection of `core3d.step`, with the flow's implicit viscous
  factorizations and the buoyancy of $T^n$. One electric solve per step supplies both the Lorentz
  force and the Joule heat.
- The `ThermalState` carries the projection's pressure, which starts the next step's pressure solve,
  as `core3d.step(pressure=)` does. A `Solution` of an evolving flow gives it, and a held flow's
  state keeps the pressure it was given.
- The fractional step does not hold every hydrostatic state exactly. A state whose temperature
  varies only along gravity stays at rest. In general, the claim is first-order convergence in
  $\Delta t$. Backward-Euler conduction is first order in time as well, so any transient
  heat-transfer number carries an $O(\Delta t)$ error.
- The ratio of the step to the buoyancy time scale sets the coupling's time error. The step reports
  this ratio beside `advective_cfl` as `buoyancy_courant` $=\Delta t\sqrt{\beta|\mathbf g|\,G}$.
  $G$ is the largest absolute face-normal gradient on any face. It is one component, not a
  colocated gradient magnitude. In the Bénard onset (#252) the critical Rayleigh number reads
  1675.9 at `buoyancy_courant` near 0.4 and 1675.6 near 0.025. That is flat to 0.02 % in the step,
  and 1.9 % below 1707.76 on the 16-cell mesh. This calibrates that case only and is not a general
  bound. The earlier first-order momentum stage read 15 % low near 0.2 and under 3 % near 0.025
  ([Discretization, at main's commit](https://github.com/uwplasma/LMhdX/blob/b97ad3a8525a20502279961d5b4201d158d971e9/docs/physics/thermal.md#discretization)).
  `validate_thermal_inputs(..., buoyancy_courant_limit=...)` refuses a start above a chosen limit.

## Diagnostics and the heat ledger

Each step returns a `ThermalDiagnostics` of scalars. The heat ledger is

$$
\underbrace{\frac{\rho_0 c_p}{\Delta t}\sum_P V_P\,(T^{n+1}_P-T^n_P)}_{\texttt{storage\_rate}}
+\texttt{conductive\_heat\_out}+\texttt{advective\_heat\_out}-\texttt{joule\_power}-\texttt{external\_power}
=\texttt{heat\_balance\_residual},
$$

which the step computes from local increments and the same wall gradients the conduction uses. It
closes to its rounding floor, about $\varepsilon\,\rho_0 c_p\sum V\,|T|/\Delta t$ for temperatures
stored in kelvin. With solid walls it closes to PCG's tolerance, relative to the step's change.

`advective_heat_out` is the enthalpy the flow carries out through an outlet, less what it brings in
at an inlet or with a returning outlet flow. It is zero in a closed duct. At the outlet,
`outlet_outflow_rate` and `outlet_backflow_rate` are the volume rates leaving and returning, and
`outlet_backflow_enthalpy_in` is the enthalpy the return brings in. `backflow_boundary_valid` is
separate from `finite`. The other diagnostics are `heat_solve_residual` (the largest local residual
of the temperature equation), `mass_residual`, `charge_residual`, `advective_cfl`,
`conduction_iterations` (with solid walls) and `boussinesq_measure`.

`finite` says only that every value is a number. To accept a run, check the Courant number, the
residuals and the Boussinesq measure. Nothing clips a failing value to repair it.

## Supported envelope

**Geometry**
- Cartesian grids, with periodic axes and closed no-slip walls. A polar grid works only for the
  developed pipe, described below. A time step refuses it.
- An inflow-outflow duct carries its heat on a held flow or an evolving one:
  - For a held flow, pass the steady velocity of `axial.solve_open_duct` in the state and build the
    step with `hold_flow=True`. A held flow takes no buoyancy.
  - For an evolving flow (ADR 0009 D37), build the step without `hold_flow=True`. The flow advances
    through the stage that `axial.solve_open_duct` marches, with its inlet profile carried as a
    lift. The outlet is D26's zero axial gradient. With buoyancy, the outlet holds the hydrostatic
    pressure of its last layer of cells, so a stratified outlet at rest stays at rest. LMhdX refuses
    gravity along the open axis. `lmhdx.solve` starts the run on the steady flow (`start="steady"`)
    or on the inlet profile carried along the duct. The step is first order in time.
    `tests/test_steady.py` measures the outlet's reach upstream, buoyant onset, flow returning
    through the outlet (see `outlet_backflow` above), and a coupled manufactured solution, which
    converges at second order with van Leer heat transport.
- With momentum transport on (`ChannelProblem.advection` other than `"off"`), a periodic axis needs
  at least two cells.
- Insulating, thin conducting, resolved and solid electrical walls all work, and their electrical
  dissipation stays outside the fluid. Resolved `wall_layers` widen only the electric solve. The
  temperature lives on the fluid cells alone, with the fluid's conductivity and its conditions at
  the fluid's wall faces, unless `walls=` adds thermal walls of the same cells (below). A flow's own
  `solid_walls` carry current and heat on one extended grid and become the thermal walls.

**Precision.** The `dtype` given to `build_thermal_step` must be the core's compute dtype, which is
float64 after `lmhdx.enable_x64()` and float32 otherwise. Set it before building.

**Derivatives** work with respect to anything carried in the state or the heat source (for example
the initial temperature or a source amplitude) and with respect to `backflow_temperature`. They do
not work with respect to $k$, $c_p$, $\sigma$, $\Delta t$, the geometry or the boundary conditions,
which the prepared step has built in. Upwinding picks its donor by the sign of each face velocity.
A derivative is therefore one-sided only where a face velocity is exactly zero, or where the
perturbation flips a face's sign. Nothing smooths the donor switch to hide this.

**Memory.**
- `advance_thermal` keeps no field history. In reverse mode, with `checkpoint=True` (the default),
  it keeps one state per step and recomputes each step's intermediates, so the memory grows with the
  number of steps. A restart record and the command's chunks are not reverse-mode checkpoints and do
  not bound that memory. The command itself is not differentiated end to end.
- `diagnostics=n` keeps only the scalars of every `n`-th step, and `steps` must be a multiple of
  `n`. The default keeps every step's. With `checkpoint=True` the block of `n` steps becomes the
  checkpoint. The run stores one state per block, but a block's reverse pass holds the
  intermediates of all `n` of its steps. The saving is largest near $n\approx\sqrt{\text{steps}}$.
  The run neither keeps nor checks the skipped steps, so `lmhdx case.toml`, which accepts a run on
  every step's diagnostics, stays at stride 1.

**Checkpointing a long differentiated run.** A `jax.checkpoint`-wrapped chunk stores only its start
state in the forward pass and recomputes its steps in reverse. The memory is then one state per
chunk plus one chunk's steps, not one state per step. Loop over the chunks with an outer
`lax.scan`, so that a jitted objective compiles one copy of the chunk, not one per chunk.

```python
chunk = jax.checkpoint(lambda state: advance_thermal(thermal_step, state, heat, 100, diagnostics=100)[0])

def run(state):  # 5000 steps
    final, _ = jax.lax.scan(lambda s, _: (chunk(s), None), state, None, length=50)
    return objective(final)

gradient = jax.jit(jax.grad(run))(start)
```

The gradient matches the plain run's to round-off
(`test_the_checkpointing_recipe_has_the_plain_runs_gradient`). It is the same computation as
`diagnostics=100` with `checkpoint=True` on the plain call, with the chunks written out as a loop.
For 1600 differentiated steps on 32 x 16 x 16 cells, one scan takes 418 MiB and 40 chunks of 40
steps take 21 MiB ({ref}`Reference results <thermal-reference-results>`, "Checkpointed memory").

**Not included**
- Thermal cells under a thin electrical wall (ADR 0009 D36).
- Solid walls in the developed duct below. Solid walls and inflow-outflow axes read from TOML. A run
  from TOML (below) covers closed ducts, and an open duct runs from Python.
- Transport above second order. The open-duct results below use first-order upwinding and quote
  $Pe_h/2$, its numerical diffusivity over the physical one. `advection="van_leer"` removes most of
  it.
- Temperature-dependent properties.
- A steady solve of the coupled equations. The developed duct below is the only steady thermal
  solve, and it takes a given flow.
- Dispatch of a `ThermalProblem` through `lmhdx.solve`. A `Problem` with `thermal=Thermal(...)` and
  `Solver(dt=, steps=)` runs the same step through it.

A closed domain with net heating has no steady temperature. The returned state is not a
`cases.MHDState` or an archive-compatible `Solution`. The final electric fields come from
`core3d.face_currents(state.velocity, flow)`.

## Conduction through solid walls

Solid walls conduct and store heat around the fluid, as in Mistrangelo et al.'s heated duct (ADR
0009 D36). This section covers the conduction solve and the thermal step built on it.

- **The grid.** `core3d.conjugate_grid(grid, walls)` adds a `core3d.SolidWall` at either end of the
  transverse axes 1 and 2. A `SolidWall` has a thickness, a whole number of equal cells, a
  `thermal_conductivity` and a `heat_capacity` $\rho c_p$. The `ConjugateGrid` it returns holds the
  extended grid, whose fluid block keeps the given faces exactly, the block's slices, and the walls.
  `cells(name, fluid_value)` spreads the walls' properties over every cell. A corner cell takes the
  nearer wall's material, as the corners of resolved electrical walls do. When both walls are
  equally near to rounding, it takes the first axis's.
- **The operator.** `poisson.variable_helmholtz(grid, conditions, conductivity, shift, coefficient)`
  is $V\,(s\,x - \tau\,\nabla\cdot(k\nabla x))$ for cell-wise $k$ and $s$.
- **The face-conductance contract.** Between two cells with half widths $\delta_1$, $\delta_2$ and
  conductivities $k_1$, $k_2$, the flux per unit area is
  $(T_2 - T_1)/(\delta_1/k_1 + \delta_2/k_2)$, from the half cells' series resistance. The same
  rule holds at a material interface as anywhere else. The face area and the cell volume enter
  once, in the divergence, so the heat that leaves one cell enters the next. A boundary face takes
  its cell's $k$, and you build a heat flux there with that $k$
  (`thermal_boundary(..., conductivity=...)`).
- **The solve.** One implicit step solves for the increment $\delta = T^{n+1} - T^n$,

  $$
  \rho c_p\,\delta - \Delta t\,\nabla\cdot(k\nabla\delta)
  = \Delta t\,\bigl(q + \nabla\cdot(k\nabla T^n) + b\bigr),
  $$

  with $b$ the boundary data's `lift`. Because the unknown is the increment, PCG's tolerance applies
  to the change. The fast-diagonal inverse at the volume-mean $k$ and $s$ preconditions
  `solvax.pcg_linear_solve`. The solve differentiates implicitly, and its tangent and adjoint solves
  start from zero. If it does not converge, it returns NaN, and NaN derivatives. The operator
  matches SOLVAX 0.34.0's `separable.flux_form_operator` to 4e-16. The plan to move to it is in
  [Conduction through solid walls, at main's commit](https://github.com/uwplasma/LMhdX/blob/b97ad3a8525a20502279961d5b4201d158d971e9/docs/physics/thermal.md#conduction-through-solid-walls).
- **What is differentiable.** A wall's conductivity and heat capacity are differentiable.
  `cells(name, fluid_value, walls={(axis, side): value})` puts a wall's value into the cells in the
  trace, `VariableHelmholtz.with_coefficients(conductivity, shift)` builds the face conductances
  from it, and the preconditioner keeps its built means. So a wall's $k$ or $\rho c_p$ is a leaf,
  and a new value does not recompile. A wall's thickness, its cell count and which cell belongs to
  which material are static.
- **PCG iterations** are 7–8 for the benchmark's steel and PbLi walls. They grow with the contrast,
  and at a high contrast they also grow as the mesh refines. A test covers the copper case
  ({ref}`Reference results <thermal-reference-results>`, "PCG iterations"). The solve sets an
  insulated axis's null eigenvalue to zero, reading it from the axis's conditions as the Poisson
  factorization does. This keeps the shifted fast-diagonal operator definite in float32 at a large
  $\Delta t\,k$.
  [Conduction through solid walls, at main's commit](https://github.com/uwplasma/LMhdX/blob/b97ad3a8525a20502279961d5b4201d158d971e9/docs/physics/thermal.md#conduction-through-solid-walls)
  tells how this was found.

**Tested** (`tests/test_ops.py`):
- Two materials in series are exact, with the same flux through every face. Steel brought into
  contact with fluid reaches the exact effusivity-weighted contact temperature at second order. The
  benchmark's steel and PbLi, walled on four sides, stay symmetric and conservative.
- **An independent high-contrast check.** Walls of 1000 times the fluid's $k$, held at 0 outside,
  surround a uniformly heated square. The fluid's mean temperature tends to the Dirichlet Poisson
  series, 0.140577, raised by the walls' own drop $q a t/(2k_w)$. The error falls at about second
  order, and the Richardson value meets the series within 1e-4.
- **A wall conductivity's derivative** meets the derivative standard. The test uses central
  differences at two steps, a Taylor remainder of second order, a tighter tolerance and a finer
  mesh, and the JVP/VJP identity in two walls at once. It also checks that a new value does not
  recompile and that the derivative is NaN when PCG has not converged.

**In the thermal step.**
- `ThermalProblem(..., walls=...)` takes a `SolidWall` or `None` for each end of the wall-bounded
  axes 1 and 2. The temperature and the heat source then live on `problem.thermal_grid`,
  `problem.fluid_block` picks out the fluid's cells, and `problem.on_thermal_grid` pads fluid data
  onto it.
- The `thermal_conditions`, an inlet profile included, hold on the outer surface. Build a heat flux
  there with the `k` of each boundary cell, which is an array where fluid and wall cells meet the
  surface. The walls' end faces take the axis-0 conditions, so an inflow-outflow axis holds them at
  the inlet temperature and insulates them at the outlet.
- The walls carry no flow, and **the step omits Joule heat in the walls.** `joule_heating` deposits
  the fluid's Joule heat. A solid wall only stores and conducts heat. It has no source of its own,
  as the benchmark's eq. 4 has none.
- One `conduct` handles both solves, PCG through the walls or the fast-diagonal inverse of one
  material. A duct without walls keeps the fast-diagonal inverse. The step computes the ledger
  once, for either solve.
- The ledger covers the fluid and its walls. Storage and sensible heat weigh each cell by its own
  $\rho c_p$, and the conduction out is the solve's own flux through the outer surface.
  `conduction_iterations` reports the PCG iterations, which are zero without walls. The extremes
  span every cell, while the Boussinesq measure and the buoyancy Courant number cover the fluid
  only.

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

**Tested** (`tests/test_timeloop.py`):
- **The ledger.** A held flow through a duct walled on four sides and heated through its outer
  bottom closes every step's ledger, at 300 K, to 1e-10 of its largest term. Its sensible heat grows
  by the storage.
- **Fluid and solid energy, region by region.** Each region stores what its face fluxes bring in, to
  PCG's tolerance. The fluid's inflow is the interface flux from its walls, plus what crosses its
  ends. The walls' inflow is the heated face's inflow, 3.3 to rounding, less that same interface
  flux. The interface flux cancels in the sum.
- **Time order.** The conjugate step is backward Euler in conduction and explicit in transport.
  Against a run at a 64th of the step it converges at first order as the step halves.
- **Walls of the fluid's own material,** of unequal cells, reproduce the separable step on the
  extended grid to 1e-10, with upwind and with van Leer transport.
- **A moving walled flow** steps exactly as `core3d.step`, and a hot wall cell does not change its
  buoyancy.
- **Derivatives.** The run differentiates with respect to its heat source and its start. A wall's
  `k` and `ρc_p` are leaves of the conduction solve (`ConjugateGrid.cells(..., walls=...)`).
  Derivatives in them through the step are planned. For now `problem.conduction()` binds them.

**The gate and the independent check.** Mistrangelo et al. is the only conjugate-heat benchmark
found so far. The exact two-material series and contact transient therefore stay the CI gate, and
their heated duct is the independent check, code to code ([record](../validation/mistrangelo.md)).
The check covers Case 1's heated zone on a held flow (step 5.4c), Case 1 gated with buoyancy, the
fringe and inertia (5.4e), and the report on Case 2 (5.4f).

## The thermally fully developed duct

A duct can be periodic in its flow direction, with the velocity and the heat sources uniform along
it. Under a uniform axial heat input its temperature is then $T=\gamma x+\theta(y,z)$, where
$\gamma$ is the bulk temperature's axial gradient and

$$
k\,\nabla^2\theta=\rho_0 c_p\,u\,\gamma-q-q_J .
$$

`developed_heat(problem, velocity, heat_source, axial_gradient=...)` solves this once on the
cross-section, with the fast-diagonal Poisson factorization (ADR 0008, D33). The way to set
$\gamma$ depends on the walls:
- **Heat-flux walls only.** $\gamma$ follows from the heat balance $\rho_0 c_p\gamma Q$ = heat in
  per unit length, and $\theta$ has zero mean. A uniform flux is the H2 condition.
- **Any prescribed-temperature wall.** That wall takes in whatever heat the solve decides, so you
  give $\gamma$. A peripherally uniform wall temperature with a uniform axial heat input is the H1
  condition.

**Inputs.** The duct is one cell long along its axis. `validate_developed_inputs` checks on the host
that the flow is the reduction's single throughflow, with tolerances in physical units. It requires
a flow rate $\sum A u$ above `flow_rate_tolerance`, which is a volume rate. It requires the same
axial velocity on both axial faces and no transverse velocity or reversed axial velocity, all to
within `velocity_tolerance`, which is a speed. Where an `axial_gradient` is given, it must be
finite. A flow that stops gives NaN for the bulk temperature and $Nu$, not a division by zero.

**The result** gives the bulk temperature, weighted by the axial flow; the area-mean and peak wall
temperatures; the mean wall heat flux into the fluid; $Nu=q_w D_h/\left(k\,(T_w-T_b)\right)$, with
$D_h=4A/P$; and the heat-balance residual.
- The wall means are over every transverse wall, heated or not. With an internal source, $Nu$ is an
  apparent number for that source, not a coefficient of the duct alone.
- $Nu$ is NaN where $|T_w-T_b|$ is within $64\varepsilon(L/h)^2$ of the largest temperature scale,
  because the solve's rounding grows with its conditioning. Here $L/h$ is the section's largest
  extent over its smallest cell. The scales are the solution's own and each heat input's
  ($D_h^2|\rho_0c_pu\gamma|/k$, $D_h^2|q+q_J|/k$ and $D_h|q_w|/k$). So when the difference cancels
  to rounding, as when heat crosses the duct from wall to wall, $Nu$ is undefined. The temperatures
  and the heat balance stay valid.
- With heat-flux walls only, the solve fixes $\theta$ only up to a constant and picks the one with
  zero mean. The physical temperature is $T=T_{b,\mathrm{in}}+\gamma x+\theta-\theta_b$.
  `DevelopedHeat.anchored_peaks(T_in, length)` returns the hottest wall face and the hottest cell
  over the length, at whichever end $\gamma$ rises toward. It assumes the profile is developed from
  the inlet, so an entrance region needs the open duct below. With a prescribed wall temperature,
  the station values already fix the reference, and both peaks are NaN.
- The result differentiates through the velocity, the heat source and $\gamma$, but not through the
  geometry. Through the velocity it reaches the implicit derivative of `steady.solve_steady_state`
  in a field scale or a drive.

The tests check the H1 and H2 Nusselt numbers against exact series, which involve no discrete
operator. Both converge at second order. On 64 cells the square duct's H1 is within 0.087 % of
3.60795 and its H2 within 0.090 % of 3.08738
({ref}`Reference results <thermal-reference-results>`, "The developed duct"). A transverse field
flattens the core. On 32 cells with Joule heat off, the H1 Nusselt number of the square duct rises
from 3.62 at Ha 0 to 4.56 at Ha 10 and 5.35 at Ha 30, toward the slug value of 7.09.

```{image} ../_static/thermal_developed_duct.webp
:alt: Velocity and temperature of an H1 square duct at Ha 0 and 100 in 3-D, Nusselt numbers against aspect ratio and Hartmann number, and mesh convergence
:align: center
```

The figure above compares LMhdX with exact results over a wider range than the tests cover. This
figure and the two below verify the solver against exact or independent solutions. None is a
comparison with experiment. `docs/_static/data/` holds their inputs and results, so you can check
or replot each one, and `python scripts/make_showcase_figures.py --only developed` redraws this
one. Every value is a mean Nusselt number, the wall heat flux averaged over the heated perimeter
and made dimensionless with the hydraulic diameter $D_h$. The ducts use the H1 condition. The
plates use their own uniform-flux Nusselt number with $D_h = 4a$.
- Rectangular ducts with aspect ratios from 1 down to 1/8 agree with the exact series to within
  0.35 %, with 32 cells across the short side.
- On the meshes of `duct_problem` (48 cells, with the layers resolved), the square duct's Nu rises
  from Ha 0 to 1000 and is still 6 % below the slug-flow value at Ha 1000.
- The plates described next, driven by the solver's own steady Hartmann flow rather than the exact
  cell means, match the closed form to within 0.2 % from Ha 1 to 1000 on 128 layer-resolving cells,
  and converge at second order on uniform cells.

**An MHD reference.** Hartmann flow between plates at $y=\pm a$, each taking in a uniform heat flux
with no source and no Joule heat, has $Nu=4/\int_0^1F_H^2\,ds$ with $D_h=4a$. Here $F_H$ is the
integral of the mean-normalized Hartmann velocity $f_H=(1-\cosh(Hs)/\cosh H)/(1-\tanh H/H)$.
Quadrature computes the reference with no discrete operator. It runs from $140/17$ at Ha 0 to 12
as Ha grows. Given the cell means of the analytical velocity, across one cell with insulated ends,
the plates' own $Nu$ converges at fourth order with no slip and at second with slug flow
({ref}`Reference results <thermal-reference-results>`, "Hartmann plates").

**The uniform wall temperature.** The T condition holds every wall at one temperature. The excess
over the wall then decays along the duct in a fixed shape, $T-T_w=\theta\,e^{-\lambda x}$. Without
axial conduction or internal heat, $\theta$ is the eigenfunction of $-L(\theta)=\Lambda f\theta$
with the smallest eigenvalue, where $f=u/\bar u$. `developed_wall_temperature(problem, velocity)`
finds it with `solvax.inverse_iteration` on the same fast-diagonal factorization. It returns the
decay rate $\lambda=\Lambda\alpha/\bar u$, the profile scaled to a unit bulk excess,
$(T-T_w)/(T_b-T_w)$, and $Nu_T=\Lambda D_hA/P_w$. $Nu_T$ uses the wall flux averaged over the held
perimeter $P_w$, and is $\Lambda D_h^2/4$ when every wall is held.

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

- The iteration stops once `eigen_residual` is below `tolerance` (1e-10 in float64, 1e-4 in
  float32), or after `max_iterations`. The square duct takes 12 iterations, and a 1:8 duct takes
  146. `tolerance` must be positive and finite, since `inf` would skip the loop, and
  `max_iterations` must be a positive integer. The host checks both.
- **The supported derivative** is the first derivative of $\Lambda$, the decay rate and $Nu_T$ at a
  converged solve. $\Lambda$ is the Rayleigh quotient of the converged $\theta$, which is stationary
  at an eigenfunction, so its derivative in the velocity needs none of the iteration's derivatives.
  The gradient through the steady solve's field scale meets central differences to 1e-5, with a
  Taylor remainder of second order. The profile goes through `stop_gradient` and has a zero gradient,
  since its sensitivity needs an implicit eigenvector solve. Second derivatives are not supported
  either. This function does not implement ADR 0010's 2.0 contract, that an unsupported derivative
  raises.
- **The solve certifies convergence on the raw flow**, in traced arithmetic. It requires a finite,
  positive mean flow, every cell flowing forward, the same axial velocity on both axial faces, and
  no transverse velocity. A fully reversed flow has positive speeds over its mean, so only the raw
  sign rules it out. The solve returns NaN for $\Lambda$ and $Nu_T$, with `converged` false, when
  it is unconverged, when the flow is reversed in any cell or overall, when there is no flow, or
  when the flow has a transverse component or uneven faces.
- A heat-flux wall must be adiabatic, and the solve leaves out sources and Joule heat, since the T
  condition has none. The square duct's $Nu_T$ converges at second order to 2.97753, within 0.040 %
  on 64 cells, and slug flow is exact
  ({ref}`Reference results <thermal-reference-results>`, "The uniform wall temperature").

**The pipe.** On the polar grid of `lmhdx.pipe`, both developed solves use the polar Poisson
factorization. The azimuth wraps and the face on the axis has no area, so the wall is the outer
radius alone, and $D_h=2R$. The flow comes from `solve_pipe`, placed on the axial faces. The
developed solves refuse Joule heat on a pipe, because a pipe forms its currents in `lmhdx.pipe`,
not in the core. A pipe's `ThermalProblem` therefore needs `joule_heating=False`:

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

The H and T numbers, in Poiseuille and slug flow, converge at second order to their exact values
({ref}`Reference results <thermal-reference-results>`, "The pipe"). The T numbers come from
`developed_wall_temperature`, so its derivative contract holds here too.

**An MHD Nusselt campaign.** The [MHD Nusselt campaign](../validation/mhd_nusselt.md) page compares
Shercliff's square duct in H1, H2 and T against an independent spectral solve, within 0.06 % from
Ha 1 to 316, and against the published curves of Bluck, Wolfendale & Marquis (2015). It does not
cover other wall conductances or aspect ratios against a published $Nu$, for example Hunt's duct
(Bluck & Wolfendale 2017).

## An open duct

A heated Hunt duct carries its heat from the inlet to the outlet. The duct is at Ha 20, with thin
conducting Hartmann walls (c = 0.05) and insulating side walls, and is 8 half-widths long. It has a
fully developed inlet at a flow rate of 4 and temperature 0, one side wall taking in a unit heat
flux, and the Joule heat of the current. Over 400 steps the heat balance closes to
$2.5\times10^{-14}$ of its largest term at every step, against a target of $10^{-6}$. At the steady
state the storage falls to $3\times10^{-11}$. The enthalpy leaving, 124.6, is the Joule heat, 110.3,
plus the net wall heat, 14.3. The net wall heat is the 16 entering through the heated wall, less
what conducts back out through the inlet. The duct's axial cells give $Pe_h/2 = 5$, so first-order
upwinding carries five times the physical axial diffusivity. This page therefore quotes the duct
for its balance, not for a Nusselt number. A plug flow carrying a uniform source converges to its
exact steady profile at first order, with observed orders 0.81 and then 0.91, at $Pe_h/2$ of 0.16,
0.08 and 0.04.

**The thermal entrance.** Near a duct's inlet the temperature profile is still developing, and the
local Nusselt number falls along the duct towards its developed value. The test case is a pair of
plates heated with a uniform flux downstream of an inlet at temperature 0. They carry the developed
Hartmann flow, held fixed in an open duct, with van Leer transport. The Graetz eigen-expansion,
which leaves out axial conduction, gives the local Nusselt number. `validation.thermal.graetz`
evaluates it on 4,000 finite volumes, and for slug flow it has the closed form
$Nu=4/\left(1/3-(2/\pi^2)\sum_n e^{-16n^2\pi^2x^+}/n^2\right)$.

At $Pe=400$, with 64 cells across and 135 along, LMhdX stays within 1.3 % of the expansion beyond
$x^+=10^{-3}$, and within 0.3 % beyond $10^{-2}$, at Ha 0, 10 and 50. Closer to the inlet the
remaining difference is axial conduction, which the expansion leaves out. At $Pe=1600$ the
difference at $x^+=10^{-3}$ drops from 1.3 % to 0.14 %, while refining the mesh across or along the
duct leaves it unchanged. The tests check the expansion itself against its slug-flow closed form and
its developed limit, 140/17. `python scripts/make_showcase_figures.py --only entrance` reruns the
sweep.

```{image} ../_static/thermal_entrance.webp
:alt: Temperature above the bulk in 3-D, local Nusselt number along heated Hartmann plates against the Graetz expansion, and its error
:align: center
```

**A side-layer jet.** A square duct at Ha 1000 and $Pe=2000$, 24 half-widths long, takes in a
uniform flux through one side wall downstream of a cold inlet, without Joule heat. With conducting
Hartmann walls ($c=0.05$) the side layers carry jets at 8.4 times the mean velocity along that wall.
The wall's hottest point then rises 25 % less above the inlet temperature than in the insulating
duct. The reduction is 25.0 % on 32 cells across and 64 along, and 25.0 % again on 40 and 80. The
hot strips left along the corners are where the jet meets the Hartmann layers. The figure at the
top of the page shows both ducts. `python scripts/make_showcase_figures.py --only heated-duct` runs
both meshes in about two minutes on a laptop GPU.

The run holds the flow fixed, so this is a heat-transfer result in a given flow, not a coupled flow
and heat result. The 25 % applies at that flow rate, with thin electrical walls and no heat
conduction through solid walls, so it is not a conjugate-wall result either. It does not mean a
lower pumping cost, because at the same flow rate conducting walls raise the MHD pressure drop.

**Crediting a load.** Judge a design that must remove a given duty on what the duty itself does,
not on the gross outlet enthalpy. Joule heat and a recirculating hot flow also raise the gross
figure. `load_attribution(step, state, background, load, steps)` runs the held flow twice, with and
without the load, from the same start and with the same velocity, Joule heat, inlet, reservoir and
walls. The difference is the load's own response, since the temperature is affine in the heat
source at constant properties, with upwind or no transport. Van Leer's face value is not affine in
the temperature. Its mean has $s(1,3)+s(3,1)=3$ but $s(4,4)=4$. On a held flow the load's response
would then depend on the background, so `load_attribution` refuses a van Leer step. It reports the
load's power and what the load adds to the enthalpy leaving through the outlet, to the conduction
out and to the storage. The conduction out is what leaks through the inlet and any
prescribed-temperature wall, together. A prescribed wall flux cancels.

With a buoyant flow, which the load moves, the response is not affine in the load. The load is a
heat source on the cells. A wall's heat flux is boundary data, present in both runs. To credit one,
pass the source $qA/V$ in the cells along that wall and set the wall's flux to zero. With upwind or
no transport, a face flux enters its cell's equation exactly as that source, so the cell
temperatures are the same. Van Leer's face value reads the wall's gradient, so with van Leer the
two can differ.

In a plug flow with $\lambda=\rho_0c_pu/k$, a uniform load leaves through the outlet in the steady
fraction $1-(1-e^{-\lambda})/\lambda$. The rest conducts back out of the thermostatted inlet. On 32
cells the credited fraction is 0.788 at $\lambda=5$, against an exact 0.801. The gap is first-order
upwind diffusion, at $Pe_h/2 = 0.08$. A conductive fluid ($\lambda=0.5$) stays coolest only by
returning 79 % of its load through the inlet. A field's Joule heat multiplies the gross outlet
enthalpy, from 7.9 to 32.8 as the flow doubles, but adds nothing to the load's credit.

## Buoyant convection

A fluid layer heated from below, between rigid plates held at fixed temperatures, becomes unstable
at $Ra_c=1707.76$, and a vertical field delays the onset
(Chandrasekhar 1961, Ch. IV). `python scripts/make_showcase_figures.py --only
onset` measures the growth rate of the critical roll just below and just above Chandrasekhar's
critical Rayleigh number, at his critical wavenumber, for $Q=Ha^2$ from 0 to 1000. It does so at two
time steps, $\Delta t_0=0.05/(Q+Ra_c^{1/2})$ and $\Delta t_0/2$.

The growth rate comes from power iteration on the time stepper. Every half time unit, the script
scales the perturbation of the conducting state back to $10^{-6}$, and the rate settles to five
digits by the third rescaling. A free decay to a smaller amplitude would not work, because
round-off noise sets its rate near onset. [Buoyant convection, at main's commit](https://github.com/uwplasma/LMhdX/blob/b97ad3a8525a20502279961d5b4201d158d971e9/docs/physics/thermal.md#buoyant-convection)
has the details.

The reference is Chandrasekhar's eigenproblem solved by Chebyshev collocation
(`validation.thermal.chandrasekhar_onset`). It reproduces his tabulated values
for $Q\le100$, as quoted by Basak & Kumar (2015), to within 0.01 %, which the tests check. On the
exact momentum stage, LMhdX's onset does not depend on the time step. $\Delta t_0$ and
$\Delta t_0/2$ agree to 0.01 %. The onset lies 0.2 to 0.8 % below the reference across the tenfold
rise of $Ra_c$, which is the error of the 32-cell layer mesh. The tests run the Bénard onset at
$Q=0$ and require it within 4 % of 1707.76. The figure script reruns the full sweep.

```{image} ../_static/thermal_magnetoconvection.webp
:alt: Critical Rayleigh number against the Chandrasekhar number, the critical rolls at Q = 0 and 1000, and the error at two time steps
:align: center
```

**Growth rates.** Beyond the onset, the rate at which a roll grows checks the buoyancy, the
transport and the Lorentz force together. `validation.thermal.convection_growth_rate` solves the
linear problem by Chebyshev collocation at any Prandtl number, in a field along any axis and over a
plane Poiseuille flow. Its checks are 1707.76, Chandrasekhar's 3757 at $Q=100$, and Orr and
Sommerfeld's $Re_c=5772.22$. Power iteration on the thermal step measures the rate for one
wavelength of the critical roll at $Pr=0.02$ (PbLi's) and $Ra=2\times10^4$, with these results:
- Without a field, the rate is 0.35 % low on 8 cells across, then +0.09 % and −0.16 % on 16 and 32,
  where the time and mesh errors have opposite signs.
- In a field of $Q=100$ across the roll's axis or vertical, it is within 0.17 % and 0.10 % on 32
  cells.
- In a field along the roll's axis, which drives no current, the roll grows at the field-free rate
  when the Lorentz force is explicit. Under the default implicit shift it grows too slowly, by the
  factor the numerics page derives, which is 17 % at $\Delta t\,\sigma B^2/\rho=0.27$ (16 cells,
  $Q=10^4$).
- Carried by a plane Poiseuille flow (Re 400 on the half-height, $Ra=10^5$), the rate is within
  +1.7 to +3.3 % with limited momentum and van Leer heat transport on 18 to 48 cells a wavelength.
  The rolls travel 2 to 5 % slow. Upwind heat transport, not in the record, was 13 % slow on 18
  cells.
- With Case 2's ingredients at its $\Delta t\,\sigma B^2/\rho=0.34$, carried by that flow in a field
  along the roll's axis (18 cells, $Q=8100$), the rate is 24.6 % slow under the implicit shift and
  +1.65 % with an explicit force, which is the field-free run's error. In a field across the axis
  ($Q=10^4$, $Ra=4\times10^5$) it is 6.0 % and 2.5 % slow.
- Along the field between insulating Hartmann walls at Ha 235 ($Pr=0.0196$, $Ra=10^5$, with M1's 48
  cells across the walls), the reference is Sommeria and Moreau's model with the exact friction of
  [code verification](../validation/verification.md) ("Braking by Hartmann walls"). That friction
  halves the field-free rate. The rate is +3.5 % on 32 × 48 × 32 cells at
  $\Delta t\,\sigma B^2/\rho=0.34$ and +1.5 % refined to 64 × 96 × 64. Under the implicit shift it
  is 11 % slow (`validation.verification.hartmann_roll_growth`).

A differentially heated square cavity (de Vahl Davis 1983, $Pr=0.71$) checks the steady nonlinear
flow. The peak velocities and the hot wall's mean Nusselt number are within 0.1 % at $Ra=10^3$ on
64 cells, 0.4 % at $10^4$ on 64, 0.1 % at $10^5$ on 128 and 0.6 % at $10^6$ on 128, and they
converge at second order. `python scripts/make_showcase_figures.py --only buoyancy` reruns both in
about 25 minutes on a laptop GPU and writes `docs/_static/data/thermal_buoyancy.json`.

At $Ra=10^4$ and $Pr=0.1$, a periodic $4\times4\times1$ layer convects in cells that keep
rearranging. The run uses 64 × 64 × 32 cells, van Leer heat transport and limited momentum
transport, with a Courant number of at most 0.47. A horizontal field of $Q=1000$ damps every motion
that varies along it and leaves steady rolls with their axes along the field, as in the NaK
experiments of Burr & Müller (2002). The movie colours the side walls and the mid-plane by
temperature. `python scripts/make_showcase_figures.py --only convection` takes about twelve minutes
on a laptop GPU.

This transient run in a field shows how the flow evolves. The page does not compare it with a
reference.

```{image} ../_static/thermal_convection_3d.webp
:alt: A liquid-metal layer heated from below, without a field and in a horizontal field
:align: center
```

## A worked example

This example builds the step by hand with `lmhdx.thermal`, the same step "Run it" reaches through
`lmhdx.solve`. A nondimensional Hartmann duct runs two cases. In the first, a forcing drives the
duct along its periodic axis, a uniform volumetric source heats it, and its top wall cools it. The
second is the same duct with buoyancy.

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
conditions = (
    periodic,
    thermal_boundary("heat_flux", conductivity=k),  # adiabatic side walls
    thermal_boundary("heat_flux", 0.0, 1.0, conductivity=k),  # the top wall takes out a unit flux
)
thermal = ThermalProblem(flow, conditions, specific_heat=1.0, thermal_conductivity=k, reference_temperature=0.0)
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

A thermal case runs from a TOML file with `lmhdx case.toml`. `examples/heated_duct.toml` is the
template (see the [CLI and TOML reference](../reference/cli.md)), and `[problem.thermal]` becomes
`Thermal(**table)`. Add an `[output]` table to write its files:

```toml
[output]
dir = "out"
every = 10
```

The command runs in compiled chunks of `every` steps and accepts or rejects each chunk after it
runs. Every sample in a chunk, not only the last, must pass these checks:
- it is finite, and the outlet is valid;
- its advective Courant number is within the scheme's limit, which is 0.8 for upwind, 0.5 for van
  Leer and none for conduction alone. `[solver] advective_cfl_limit` replaces that limit, and the
  summary records it as an override;
- its buoyancy Courant number is within `[solver] buoyancy_courant_limit`, if one is given. That
  limit is a policy calibrated on particular cases, not a stability bound.

The command holds the start to the same limits and to `divergence_tolerance`. An accepted chunk
writes a record, `heated_duct_000010.npz` and so on, and a frame, `heated_duct_000010.vtr`.
`heated_duct.pvd` gathers the frames.

The first rejected chunk stops the run. It writes only `heated_duct_000020_rejected.npz`. That
record imports but does not resume. `heated_duct_summary.json` holds these entries:
- `accepted`;
- the first rejected sample's step, reasons and Courant numbers;
- the limits;
- the last diagnostics and `time`, the time they belong to. On a rejection both come from the
  rejected chunk, its last sample and its end, while `steps` counts only the accepted steps;
- the run's largest value of each diagnostic. No limit applies to the heat balance residual or the
  Boussinesq measure.

The command exits with code 2 if it rejected the run. To resume, name a record in `[start]`. Its
step numbers the frames that follow, and a series of their own, `heated_duct_from_000020.pvd`,
gathers them, so the first series stays intact:

```toml
[start]
path = "out/heated_duct_000020.npz"
```

With `steps = 20`, that run ends on the same record, bit for bit, as the run of 40 steps.
`mode = "resume"`, the default, refuses a record written for another problem or solver, ignoring
the run's length. `mode = "import"` starts another experiment from the record's state.
`lmhdx upgrade` converts files of schema 1.

## Measured cost

The table compares the step of `advance_thermal` with `core3d.advance` with implicit viscosity. The
measurements ran on an AMD Ryzen 9 5900HS CPU, in one process at float64, with JAX 0.6.2. "Cold" is
the first call, including trace and compile, with the persistent cache off. "Memory" is XLA's
compiled temporary and output size.

| Mesh | Run | Cold (s) | Warm (ms/step) | Forward memory (MiB) | Reverse memory, 10 / 40 steps (MiB) |
|:--|:--|--:|--:|--:|--:|
| 1×48×48 | isothermal | 0.47 | 0.19–0.22 | 0.21 | n/a |
| 1×48×48 | thermal | 0.66 | 0.36–0.38 | 0.45 | 1.05 / 3.18 |
| 1×48×48 | buoyant | 0.69 | 0.35 | 0.45 | 1.99 / 4.65 |
| 8×32×32 | isothermal | 0.48 | 0.91–0.95 | 1.06 | n/a |
| 8×32×32 | thermal | 0.89–0.99 | 1.28–1.47 | 1.61 | 3.24 / 9.22 |
| 8×32×32 | buoyant | 0.92–1.05 | 1.34–1.45 | 1.55 | 6.27 / 14.12 |

The forward memory is the same at 10 and 40 steps. The reverse memory grows by one state per step.

(thermal-reference-results)=
## Reference results

This section gives the tables that the sections above summarize, with the numbers in full.

### Van Leer convergence

On the steady plug flow of the open duct, the face value converges at order 1.95 and 1.98, while
upwinding converges at order 0.81 and 0.91:

| Cells | Upwind error | Van Leer error |
|--:|--:|--:|
| 16 | 4.6×10⁻² | 4.8×10⁻³ |
| 32 | 2.6×10⁻² | 1.2×10⁻³ |
| 64 | 1.4×10⁻² | 3.1×10⁻⁴ |

### PCG iterations

Conduction through solid walls:

| Walls ($k$ and $\rho c_p$ over the fluid's) | PCG iterations, 6 to 24 cells across, $\Delta t$ 0.05 and 100 |
|:--|:--|
| the benchmark's steel and PbLi (0.9, 2.3) | 7–8 |
| copper-like (19, 1.85) | 14–33 |
| 100, 1.85 | 21–63 |

At a high contrast the count grows as the mesh refines. A test covers the copper case. These counts
are measurements at those contrasts and meshes, not a conditioning bound for others.

### Checkpointed memory

The table gives the measured compiled temporary memory of `jax.jit(jax.grad(objective))`
(`.lower(start).compile().memory_analysis()`, 64-bit CPU, held flow, upwind)
for 1600 steps on 32 x 16 x 16 cells. One state is 0.06 MiB for the temperature alone.

| Run | Compiled memory (MiB) |
|---|---|
| Plain, one scan of 1600 steps | 418 |
| 160 chunks of 10 steps | 46 |
| 40 chunks of 40 steps | 21 |
| 10 chunks of 160 steps | 37 |

The same 1600 steps with `diagnostics=10`, `40` and `160` on the plain call give the same 46, 21 and
37 MiB, because the stride is the chunk. The minimum is near the square root of the step count, 40.
On either side of it, either the stored states or one block's intermediates dominate. The figures
hold for this grid and step. They scale with the cells, and the ratio is what carries over.

### The developed duct

H1 and H2 Nusselt numbers. The flow is the package's own steady solve or a uniform slug:

| Duct | Exact | 16 / 32 / 64 cells | Order |
|:--|--:|:--|:--|
| square | 3.60795 | +1.35 % / +0.35 % / +0.087 % | 1.97, 1.99 |
| aspect ratio 0.5 | 4.12330 | +2.7 % / +0.73 % / +0.18 % | 1.92, 1.98 |
| square, slug flow | 7.11354 | −1.5 % / −0.38 % / −0.094 % | 1.98, 1.99 |
| square, H2 (uniform flux on every wall) | 3.08738 | +1.41 % / +0.36 % / +0.090 % | 1.98, 1.99 |

The H2 case tests the heat-flux path that the design workflow uses. Its reference,
`validation.thermal.h2_series`, expands the velocity in sines and the temperature as a lift plus
cosines, and bases Nu on the perimeter mean of the wall temperature. The spectral solve of the MHD
Nusselt campaign
(`validation.shercliff.developed_nusselt`) agrees with it to 1e-8, and the
package's Richardson value, 3.08733, to 2e-5. Shah & London's (1978) 3.091, which the review of
#240 cited and which was the reference until then, is 0.12 % above all three. Bluck et al. (2015)
tabulate 3.0871.

The H1 values lie on Shah & London's polynomial fit for rectangular ducts, with the coefficients
taken from the `ht` library. The comparison uses the fit but never gates on it, and does not
consult the book's own tables. The tests check the aspect-0.5 duct only at 64 cells, and the
refinement study uses the square ducts.

### Hartmann plates

This table uses the plates of "An MHD reference", given the cell means of the analytical velocity
across one cell with insulated ends, and gives the plates' own $Nu$:

| Flow | Exact | 32 / 64 / 128 cells | Order |
|:--|--:|:--|:--|
| Poiseuille, Ha 0 | 8.23529 | −2.6e‑7 / −1.6e‑8 / −1.0e‑9 | 4.0 |
| Ha 10 | 10.25858 | −1.2e‑5 / −7.6e‑7 / −4.7e‑8 | 4.0 |
| slug | 12 | −1.9e‑3 / −4.9e‑4 / −1.2e‑4 | 2.0 |

With the velocity's cell means the discrete fluxes are exact, and the $O(h^2)$ error of $T_w-T_b$ is
$+h^2\theta''_w/6$, half from the wall value and half from the bulk. Since
$k\theta''=\rho_0c_p\gamma u$, no slip removes it. With point values of the velocity the order
falls to 2. Slug flow keeps it, and its discrete answer is exactly $12/(1+h^2/2)$. The public
four-wall `nusselt` of that fixture averages in the insulated ends, and so gives a different
number.

### The uniform wall temperature

$Nu_T$ from `developed_wall_temperature`:

| Duct | Reference | 16 / 32 / 64 cells | Order |
|:--|--:|:--|:--|
| square, slug flow | $\pi^2/2=4.93480$; discretely $(8/h^2)\sin^2(\pi h/4)$ | exact to 3e-16 at each | n/a |
| square | 2.97753 | +0.63 % / +0.16 % / +0.040 % | 1.97 |
| plates, Poiseuille (32 / 64 / 128) | 7.54070 | −4.6e‑4 / −1.1e‑4 / −2.9e‑5 | 2.0 |
| plates, Ha 10 (32 / 64 / 128) | 8.96255 | −6.5e‑4 / −1.6e‑4 / −4.1e‑5 | 2.0 |

The square duct's reference is a Chebyshev collocation of the same eigenproblem, extrapolated from
40, 48 and 56 nodes at its observed order 2.0. The package's own Richardson value, 2.97749, agrees
with it to 1.5e-5. Shah & London (1978) list 2.976, 0.05 % below both. That row for slug flow is
exact because the sampled sines of slug flow are exact eigenfunctions of the cell-centred Dirichlet
Laplacian. The plates carry the cell means of the analytical velocity, with the plates' own
$D_h=4a$. Their references are one-dimensional Chebyshev solves.

### The pipe

| Pipe: mean Nu on the diameter, $D_h=2R$ | Exact | 16 / 32 / 64 radial cells | Order |
|:--|--:|:--|:--|
| H (uniform flux; H1 and H2 coincide in a pipe), Poiseuille | 48/11 = 4.36364 | +0.28 % / +0.071 % / +0.018 % | 2.0 |
| H, slug | 8 | −0.39 % / −0.098 % / −0.024 % | 2.0 |
| T (one wall temperature), Poiseuille | 3.65679 | +0.077 % / +0.019 % / +0.0049 % | 2.0 |
| T, slug | $j_{0,1}^2=5.78319$ | −0.12 % / −0.031 % / −0.0077 % | 2.0 |

The value 3.65679 comes from a radial Chebyshev solve of the same eigenproblem across the diameter.
