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
