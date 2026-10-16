# 0008 — Thermal transport: the first stage, buoyancy and the module budget

Status: proposed 2026-10-14; amends D9 and the release-2.0 non-goals of
`plan.md` Section 1, and restates Phase 5.

Basis: `main` at `0f5d931`; a read of the `core3d`, `ops`, `grid` and `poisson`
symbols the thermal step reuses (listed in the step 5.1 issue); the duct design
law (#216, #221, record #222), whose heat-removal proxy, a lower bound on the
mean velocity, this phase replaces; Mistrangelo et al., Nucl. Fusion 65 (2025)
116006; Shah & London, *Laminar Flow Forced Convection in Ducts* (1978).

**D31. Phase 5 starts with a fluid-only temperature equation on the Cartesian core.**
The equation is `ρ₀c_p [∂_t T + ∇·(uT)] = ∇·(k∇T) + |J|²/σ + q_ext` with
constant properties, the temperature at the cell centres of the MAC grid, and
its own boundary conditions (periodic, prescribed temperature or prescribed
outward heat flux), independent of the electrical and mechanical ones.
Transport is a conservative first-order upwind face flux. Diffusion is implicit
through one cell-centred `fast_diagonal_helmholtz` with shift 1, which keeps the
mean, so stored heat is not removed. Sources are taken at the old state, from
the same face currents as the Lorentz force. Joule heat is the square of the
full face current over σ, deposited with `face_average_adjoint`, so its volume
integral equals `energy_budget(...).joule`; electrical power entering a
conducting wall is not fluid heat. Viscous heating is omitted from the model,
and the documentation says so. A prescribed outward flux `q` enters
`grid.BoundaryCondition` as the coordinate derivative `pad` implements: `q/k`
at the lower end and `−q/k` at the upper. Supported: Cartesian grids with
no-slip or periodic axes. Polar and inflow-outflow problems are refused at setup.

Conjugate walls, inlet-outlet transport, higher-order transport and variable
properties are later steps, each added when a step needs it and with its own
verification, not options carried by the first stage. Step 5.1's exit is
restated: conduction at second order on uniform grids, with wall heat flux and
stretched grids measured separately; upwind transport at first order, measured;
and no heat-transfer number quoted without refinement of that number.

Reason: the full 5.1 bundles four independent numerical problems (conjugate
interfaces, open thermal boundaries, bounded second-order transport and the
coupled step). The coupled step is correct without the other three, and they
build on it.

**D32. Boussinesq buoyancy is an option of the thermal step, amending D9.**
The force density `−ρ₀β(T − T_r)g` is added beside the Lorentz force, before
the division by density and the projection. `β = 0`, the default, is passive
transport and leaves the momentum update unchanged. Buoyancy ships
research-stage and outside the release-2.0 gates. The fractional step does not
hold a general hydrostatic state exactly, so the claim is convergence in the
time step, not a well-balanced scheme.

Reason: heated blanket ducts run in mixed convection. The duct design law
reports `Ri = Gr/Re²` at every optimum without modelling it, and the Phase 5
code-to-code target (Mistrangelo et al. 2025) is a magneto-convection
benchmark. The cost is one force term in the existing step.

**D33. The thermally fully developed duct is the design observable (step 5.2).**
One periodic axial cell carries `T = γx + θ(y, z)`, with `γ` fixed by the heat
balance (heat input per unit length over `ρ₀c_pQ`) and `θ` from one steady
transverse solve with the fully developed velocity. This covers volumetric
heating and wall heat flux that is uniform along the axis; uniform wall
temperature is an eigenvalue problem and waits. The solve returns the bulk
temperature (weighted by enthalpy flux), the Nusselt number and the peak wall
temperature as functions of the cross-section at fixed flow and heat load, with
implicit gradients. Gate: the rectangular-duct Nusselt numbers of Shah & London
(1978), taken from the book, within 0.5 %, and a closed heat balance. These
replace the mean-velocity bound of the duct design problem (3b.3, 5.5). The pipe
values of 5.2 wait for a polar thermal operator.

**D34. One package module: `lmhdx.thermal`.** `package_module_count_max` rises
from 15 to 16. The test-file cap (22) and the curated-example cap (7) stay:
thermal tests go into `tests/test_ops.py`, `tests/test_core3d.py` and
`tests/test_timeloop.py`, and the public workflow is an executable example in
`docs/physics/thermal.md`, exercised by a test. Each pull request raises the
line and checkout-byte ratchets by the measured amount it adds (Section 9). The
first stage adds no root export, no `lmhdx.solve` dispatch, no archive format
and no dependency. `core3d` gains two private helpers and no import of `thermal`.

Reason: the owner's rule of 2026-10-04 (the size target is a direction; keep
adding functionality). A scalar transport placed in `core3d` or `ops` would grow
the two modules that every core change touches.

## Amendments after review (2026-10-15)

From the review of #230–#241. Each adds to its decision, and says so where it narrows the text above.

**D31, the open duct.** An inflow-outflow axis carries heat on a held flow
(step 5.3): the state keeps the steady velocity of `axial.solve_open_duct`, the
step is built with `hold_flow=True`, and a held flow takes no buoyancy. This
narrows the refusal of inflow-outflow problems above to an evolving flow, which
`build_thermal_step` refuses. The inlet prescribes its temperature. The outlet
takes no conductive flux. With `C = rho_0 c_p` and `u_n` the velocity along the
outlet's outward normal, a face where the flow leaves carries out `C u_n T_P`.
Where the flow returns (`u_n < 0`), the incoming enthalpy is boundary data, not
a value inferred from the adjacent cell. By default
(`outlet_backflow="error"`) `validate_thermal_inputs` refuses a returning outlet
face, and a step run without it reports `backflow_boundary_valid` false. With
`outlet_backflow="reservoir_flux"` the face brings in `C u_n T_res`, a
prescribed total (Danckwerts) flux, with the reservoir temperature given to each
step as numerical input. The base outlet operator stays homogeneous Neumann, so
the conduction inverse is unchanged.

**D31, transport** (from the review of #234). Upwinding stays the default.
`advection="van_leer"` adds a van Leer-limited second-order face value, held
between its two cells. Its Courant limit of 0.5 is a tested envelope, not a
derived bound. Open, inlet and wall faces keep the first-order value, and the
step stays first order in time. The face value is not affine in the
temperature, so `load_attribution` accepts upwind or no transport only.

**D31, dissipation and integration** (from the review of #230). Viscous
heating stays out. With the Joule heat it makes up the hydraulic power of a
developed duct with insulating walls. The Brinkman number `mu U^2 / (k dT)`,
about `1e-8` for PbLi at 0.1 m/s and 100 K, scales only the viscous part; the
Joule part is `Ha^2` larger on the same length and depends on the walls. Their
size is measured per case: under `1e-5` of the duty in the design example.
Joule heat is kept so the heat ledger matches `energy_budget(...).joule`.
A thermal run reads TOML input and writes a versioned restart record and
ParaView (VTK XML) output, whose layout is checked by parsing it (plan step
5.6); TOML covers closed ducts.

**D32, gravity along a periodic axis.** Along a periodic axis the mean of
`-rho_0 beta (T - T_r) g` is a uniform force that no periodic pressure can
balance, and with net heat in and no cooled wall `T - T_r` keeps growing. A component of
`g` along a periodic axis is therefore refused while `beta > 0`; the mean
temperature is never subtracted. With `beta = 0` the configuration is exactly
passive, whatever `g` is.

**D33, an MHD gate and what Nu means.** The developed duct is also gated
against Hartmann flow between plates at `y = ±a` heated by equal uniform fluxes,
with no internal source, Joule heat included. With `s = y/a`, `Ha` on `a`, and
`F_H(s)` the integral from 0 of the mean-normalized Hartmann velocity
`(1 - cosh(Ha s)/cosh Ha)/(1 - tanh(Ha)/Ha)`, the plates' own Nusselt number is
`Nu = 4 / integral_0^1 F_H(s)^2 ds` with `D_h = 4a`. It runs from 140/17 at
`Ha = 0` to 12 as `Ha` grows. The gate takes the cell means of the analytical
velocity as input and holds within 0.5 % at `Ha` 0 and 10 and in slug flow.
`DevelopedHeat.nusselt` is `q_w D_h / (k (T_w - T_b))` with `D_h = 4A/P` and
means over all transverse wall area; it is not the plates' number. With an internal source (Joule or external heat) it is an
apparent, source-dependent number, not a forced-convection coefficient. Where
its wall-to-bulk difference is unresolved it is undefined (NaN), while the
temperatures and the heat balance stay valid.
