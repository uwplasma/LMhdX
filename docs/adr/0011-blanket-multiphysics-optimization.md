# 0011 — Blanket multiphysics for shape optimization

Status: proposed 2026-11-12, revised 2026-11-13. It adds steps H4.8, H7.6–H7.8,
H8.4–H8.8 and H9.6 and tests T50–T63 to `plan.md` Program H, and amends:

- ADR 0008 D32: gravity along the developed axis is accepted in the thermally
  developed solve (DB6);
- ADR 0009 D36: one wall type and wall stacks; wall Joule heat behind a switch;
  wall widths on the extended grid are differentiated through DB3, while the
  per-axis wall nodes keep their derivatives by automatic differentiation
  until ADR 0012 DC4's gate holds (DB3, DB4, DB5);
- `plan.md` steps H7.4, H8.1, H8.2, H8.3, H9.2, H9.3 and H9.5, as listed under
  "Amended steps".

ADR 0010 DH5 is not amended: the grid stays static (DB3). The consolidation
this work rests on is ADR 0012.

Basis:

- `main` at `c1a9e9b` (2026-11-13), read module by module against `plan.md`.
  - Steps 5.4b parts 1–3 (#284–#286) are merged: solid walls conduct heat,
    enter the thermal step and carry current.
  - H3 (#302) and H4.4 (#303) are merged; the SOLVAX floor is 0.36.2 (#312).
  - `thermal._joule_heating` divides by the fluid's conductivity, and wall
    cells receive no Joule heat.
  - The facade `Problem` has no route to solid walls; `Thermal` holds a scalar
    heat source.
  - Every thermal property is a Python float closed into the step.
- The note *An extended conjugate heat transfer model for liquid-metal MHD in
  fusion blankets* (2026-11-12): governing equations with thermoelectric terms,
  their energy and entropy identities, discrete forms, the gaps for a blanket
  simulation, cost by fidelity level at reactor parameters, and a modular path
  to shape optimization.
- Smolentsev et al., *Dual-coolant lead–lithium (DCLL) blanket status and R&D
  needs*, Fusion Eng. Des. 100 (2015) 44–54, for the design limits and the
  regime (Ha 10³–10⁴, Re about 10⁴, Gr 10⁹–10¹²).
- Shercliff, *Thermoelectric magnetohydrodynamics*, J. Fluid Mech. 91 (1979)
  231–251; Bond and Howell, arXiv:2407.02470 (2024), for the thermoelectric
  Ohm and Fourier laws.

**DB1. The design target is a liquid breeder blanket; couplings are declared
in one place.** The modules are the field (external), flow and charge, heat
with its thermoelectric terms, species, a 1-D coolant, neutronics (external),
structure (external, or a proxy) and the flow network of H9.2. Flow and charge
are always solved together (DH1, DH3). Every other coupling has a mode:

- `off`;
- `one_way`: the partner field is held. For `advection` that is the flow held
  fixed in time (the kernel's `hold_flow`). A held flow carries no buoyancy,
  so `buoyancy` must then be `off`;
- `lagged`: the partner field of the last step or iterate;
- `monolithic`: one system.

The rules for the declaration:

- **The declaration** is one frozen dataclass, `Coupling`, held as
  `Problem.coupling`. It is static: a mode is structure, not a leaf. It has a
  field only for a coupling the code has: `advection` (flow to heat),
  `buoyancy` (heat to flow) and `electric_heat` (charge to heat) first;
  `seebeck` (heat to charge) with H8.4; property, coolant and species fields
  with the steps that add those couplings.
- **Off** means the kernel does not build the term. The kernel arguments that
  exist today are the mechanism: `joule_heating=False`,
  `expansion_coefficient=0`, `advection="off"`, `hold_flow=True`. There is no
  second code path: a fully coupled run is the same step with every mode
  raised.
- **The default** `Coupling()` builds the kernel calls of today, so every
  existing run is reproduced bit for bit (T61).
- **A mode a solve path does not implement** raises `NotImplementedError` with
  the plan step that adds it. `monolithic` buoyancy exists in the developed
  solve (DB6) and nowhere else.
- **The facade fields this replaces** are removed in the same PR:
  `Thermal.joule_heating`, and `"off"` as a value of `Thermal.advection`,
  which then names the transport scheme only.

Reason: the same physics is wanted at three costs (an optimization loop, a
correction run, a verification run), and a second code path for the coupled
case would be a second implementation (DH15).

**DB2. Thermoelectric terms, in their consistent form.** Ohm's law gains the
Seebeck field, `J = sigma(−grad phi + u × B − S grad T)`; the heat flux gains
the Peltier term `S T J`; the heat source is `J · (J/sigma + S grad T)`. This
is the form whose entropy production is non-negative for every sign of `S`,
`J` and `grad T`, and it closes the Lorentz-work identity of DB5.

- **Where it enters.** Thermoelectric current is driven where materials meet,
  so the terms live on the extended grid of D36 only.
  - `core3d.face_currents` takes `extra_emf`: the thermoelectric field on each
    face of the extended grid, in volts per metre, the face EMF divided by the
    distance between the two cell centres.
  - It passes this to the solid-wall charge solve, where the face conductance
    times the field joins the motional current in the right-hand side and in
    the current.
  - The field is zero on the outer faces of every non-periodic axis, which
    carry no current.
  - With thin or insulating walls an `extra_emf` is refused: a thin wall has no
    temperature, and the insulated boundary faces of the fluid block carry no
    EMF.
  - `core3d` does not import `thermal`. The EMF/Lorentz adjoint pair of D17 and
    DH10 is not touched.
- **The face EMF.** Let `Psi_m(T)` be the integral of `S_m dT`, and `T_fP`,
  `T_fN` the face temperatures on the two sides from conductive flux
  continuity. They are equal unless the face has a thermal contact resistance.
  The EMF across a face is
  `Psi_P(T_fP) − Psi_P(T_P) + Psi_N(T_N) − Psi_N(T_fN)`. Within one material
  this is a discrete gradient that the potential absorbs, so a uniform
  material carries no thermoelectric current to round-off.
- **The heat.** The Peltier flux is one value per face,
  `(S_P T_fP + S_N T_fN)/2 · J_f A_f`, so it moves heat without creating it.
  - In a cell of one material with constant `S`, the flux and the Seebeck work
    cancel by charge conservation.
  - At an interface they release `(S_P T_fP − S_N T_fN) J_f A_f`, half into
    each cell. Without a contact resistance this is the Peltier heat
    `(S_P − S_N) T_f J_f A_f`.
  - The Seebeck work is deposited per half cell: each half cell takes its own
    part of the face EMF, `A_f J_f (Psi_P(T_fP) − Psi_P(T_P))` to `P` and
    `A_f J_f (Psi_N(T_N) − Psi_N(T_fN))` to `N`.
  - Both enter the heat equation as power densities among its sources, not
    through the advective flux, which is multiplied by `rho c_p`.
- **First step.** Constant `S` per material (`Psi = S T`). `S(T)` joins the
  calibrated family of H8.2.
- **Rejected:**
  - an averaged face coefficient `S_f (T_N − T_P)`, which drives spurious
    current inside a material once `S` depends on `T`;
  - the Seebeck work without the Peltier flux, whose entropy production can
    turn negative;
  - the Peltier heat without the Seebeck work, which breaks the Lorentz-work
    identity;
  - folding the EMF into the motional term of the fluid block alone, which
    loses it at the interface faces.

Reason: the Seebeck EMF is 2 to 20 % of the motional EMF in a DCLL channel
(`Delta S` about 10 µV/K, a placeholder until PbLi and EUROFER are measured)
and of order one at the mm/s speeds of slow-flow concepts. The Peltier heat is
under 1 % of the plasma heat flux and is kept for consistency. No
thermoelectric result is claimed before pinned coefficients exist.

**DB3. Shape derivatives through the linearised solve; the grid stays
static.** A shape parameter `s` (half-width, aspect ratio, layer thickness, gap
width) is a leaf of the solve, not of the grid.

- **The residual.** The steady solution `u(s)` is the root of
  `Phi(v; s) = R(P_s v; s) + (v − P_s v)`. Here `R` is the projected steady
  residual and `P_s` the solenoidal projection, both of the problem built for
  `s`.
- **The derivative** solves `Phi_v du/ds = −Phi_s`. `Phi_s` is the centred
  difference in `s` of `Phi` at the frozen solution, evaluated on problems
  built for `s ± eps` with the same cell counts. The frozen fields are rebound
  onto each perturbed grid by their arrays, because a field's grid is static.
- **The split.** With `b = Phi_s` split by the unperturbed projection, the
  solenoidal part `w` solves `A w = −P b`, with `A` the affine operator of the
  steady solve: the steady solve itself at zero drive, with a force field.
  Then `du/ds = w − (b − P b)`.
- **A functional's derivative** is its own difference in `s` at the frozen
  solution plus its JVP along `du/ds`.
- **Cost:** one solve; per parameter, two problem builds on the host (with a
  compile where the problem's programs are not shared), two residual
  evaluations and one linear solve of the same operator. An adjoint form, one
  extra solve per functional, follows when a study has more shape parameters
  than functionals.
- **Layer meshes** follow the shape, because the grid for `s ± eps` is built by
  the same rule. Every shape derivative passes DH14 against centred
  differences of the whole solve.
- **Scope:** the affine problems. These are the steady Stokes duct, and the
  developed cross-section of DB6 with its heat, its wall stacks and films, and
  its monolithic mixed convection at a fixed axial gradient. A functional may
  read the velocity and, in the developed section, the temperature.
- **Not covered:** time-dependent transport, open ducts, topology and
  cell-count changes, curved or mapped blocks (H9.2), second derivatives,
  `jax.jit` over the shape parameters.

Reason: the cross-section of a channel is the first design stage, its
functionals are thermal (interface and wall temperatures, the Nusselt number)
as well as hydraulic, and today the aspect ratio is found by a five-point
sweep and a quartic fit.

Wall widths are differentiated this way on the extended grid. On the per-axis
wall nodes of the thin-wall factorization, a layer's conductivity and
thickness are leaves of `Problem` and are differentiated by automatic
differentiation through `lmhdx.solve`. That stays until ADR 0012 DC4's gate
holds: the extended grid must offer the same derivatives through the same
call.

Rejected for now: face coordinates as traced leaves of `Grid`. `Grid` is the
hash key of the factorization and program caches and the static part of
`Field`, and every metric in `grid`, `ops` and `poisson` is built on the host.
Tracing it edits every numerical core at once (H.10.7) for a derivative the
linearised solve already gives.

**DB4. One wall stack.** A wall is a tuple of `WallLayer`, listed from the
fluid outward, on the extended grid. A layer has:

- a thickness and a number of cells;
- an electrical conductivity, or none (the layer carries heat only);
- a thermal conductivity and a heat capacity;
- a Seebeck coefficient;
- an electrical and a thermal contact resistance at its face toward the fluid.

Rules for layers:

- A layer may be a **stagnant liquid**: the liquid's properties with `u = 0`.
  A flow-channel insert with its gaps is the stack (insert, gap, steel). This
  holds while the gap flow carries little current and heat; flowing gaps and
  pressure-equalization slots need multiblock fluid (H9.2).
- **A heat-only layer** ends the electrical stack. The charge solve's extended
  grid stops at the layer's face toward the fluid, and every layer outside it
  carries heat only. This is exact while those layers have no EMF of their
  own: no `−dA/dt` (H9.5) and no Seebeck contrast among them. The thermal
  extended grid keeps every layer. H7.8 adds this split. Until then a
  coating is a layer of small conductivity.
- **Contact resistances** enter the one series formula,
  `R_f = delta_P/sigma_P + R_c + delta_N/sigma_N`, and the same form for heat.
- The interface physics (series faces, contact and film resistances) is formed
  in LMhdX; the generic variable-coefficient operator takes face resistances
  (ADR 0012 DC5).
- Electrical insulation, thermal insulation and permeation resistance stay
  separate properties of a layer (H9.2).

Rejected for now: one null vector per disconnected conductor. A perfect coating
is a heat-only layer, and no case in the plan needs a floating conductor that
carries current.

**DB5. Wall Joule heat by series resistance, behind a switch.** Each face
dissipates `A_f R_f J_f²`, shared as follows:

- each half cell takes its own resistance's share, which is what dividing a
  cell's half-face sum by that cell's conductivity gives;
- a contact resistance's share `A_f R_c J_f²` goes half to each of the face's
  two cells and is its own ledger column.

`wall_joule_heating` is the documented switch of H7.2. It is off by default,
which reproduces today's numbers and matches the reference equation of the
Mistrangelo et al. case (D35); blanket studies set it on. With it on,

    sum_f A_f R_f J_f² + sum_f A_f J_f e_f^TE = sum_f A_f J_f e_f^m

(electrical heat equals the Lorentz work; `e^m` the motional and `e^TE` the
thermoelectric face EMF) is a gate at every fidelity level, and each term is a
column of the heat ledger: Joule in the fluid, Joule in the walls, contact
Joule and Seebeck work, with the Peltier redistribution summing to zero.

**DB6. A fidelity ladder.**

| Level | Model | Use |
|---|---|---|
| L0 | Thermally developed cross-section: wall stacks, nuclear heating, a film condition, Seebeck forcing, developed species, developed mixed convection | the inner optimization loop, with the shape derivatives of DB3 |
| L0+ | A march of the cross-section energy and species equations along `x` on a locally developed flow, with capacity `rho c_p u` and a 1-D coolant | whole-duct functionals |
| L1 | The inertialess core-flow model; Q2D heat and species only inside an envelope qualified against resolved 3-D (H9.5) | layout |
| L2 | 3-D steady segment | correcting L0 and L1 |
| L3 | 3-D unsteady, every coupling `lagged` | verification; not differentiated (DH3) |
| L4 | Module with manifolds (H9.2) | research |

- **L0** is `thermal.developed_heat` on the extended grid: with walls or a
  film the section solve is the variable-coefficient operator in steady form.
- **Developed mixed convection.** At a fixed axial gradient the coupled
  `(u, theta)` problem is affine.
  - The buoyancy enters the steady solve as a force that depends on the
    velocity.
  - The solve takes the nonsymmetric Newton–Krylov route with its implicit
    adjoint (D19), because the coupled operator is not positive definite for
    one sign of `beta g gamma`.
  - This amends D32: in the developed solve, a component of `g` along the
    developed axis is accepted. D32's two objections do not hold there. The
    mean buoyancy joins the pressure gradient, which is an output at a fixed
    flow rate. The temperature is the bounded deviation `theta` from the axial
    ramp.
  - The reported pressure gradient includes that hydrostatic part, which
    depends on the reference temperature. A hydraulic functional uses the
    gradient with it removed.
  - The time step keeps D32's refusal.
- **L0+** is `thermal.march_section`: implicit in `x`, walls quasi-steady in
  `x`. It replaces "a sequence of developed sections": the flow is locally
  developed, the temperature is not (the thermal development length, about
  `Pe a`, exceeds the duct).

**DB7. Coupling audit.** At the start and end of an optimization:

- each coupling below the highest mode its path implements is raised to the
  next implemented mode;
- the active functionals and their gradients are recomputed;
- a change above the optimizer's tolerance keeps the higher mode.

The package holds one function for this. Stage C (correcting L0 and L1 to L2
values at accepted designs), the aggregation of peak temperatures and robust
optimization over the data ranges are study code outside the package.
Gradients hold neutronics frozen (H9.3).

**DB8. Coolant boundary and heat loads.**

- **A convective wall**, `−k dT/dn = h (T − T_c)`, is a temperature condition
  at `T_c` with a film coefficient `h`, a number or an array over the face.
  The film is a boundary face resistance `1/h` in the series formula of DB4,
  so no boundary kind is added to the grid.
- **The 1-D coolant** is a function that gives `T_c(x)` from the heat absorbed
  upstream; the march of DB6 calls it. The boundary lift is linear in `T_c`,
  so a coolant temperature computed inside the march enters as that linear
  term, not as a new condition.
- **Nuclear heating maps** enter `Thermal.heat_source` as arrays on the
  thermal grid, walls included.
- **The field of a changing applied current**, `−dA/dt`, stays in H9.5.

Rejected: a fourth boundary kind in `grid`, which touches the padding, the
null-space rules and every factorization for one use.

**DB9. Species through `w = c/K`.** With `w = c/K` continuous at interfaces,
the species equation is the heat equation with capacity `K` and conductance
`D K`. So the section solve and the march of DB6 carry it with two arrays
changed. This assumes the same solubility law on both sides of each
interface and no surface-limited permeation; a layer that breaks either
needs an interface resistance or a jump condition, and H9.3 refuses it. H9.3
adds species.

**Amended steps.**

- **H7.4.** Each RK stage uses one set of face currents for the Lorentz work
  and the electrical heat, and transports the temperature with the stage
  velocity.
- **H8.1.** The leaves add the Seebeck coefficients, the contact resistances,
  the film coefficient and the heat-source arrays. New kernels take these as
  arguments and add no host conversion.
- **H8.2.** `S(T)` joins the calibrated family, with `Psi_m(T)` from the same
  fit.
- **H8.3.** The measurement family adds:
  - insert contrasts (`sigma` 10⁴–10⁶);
  - 2 mm stagnant gaps;
  - cell aspect ratios to about 10³.

  D36's iteration argument covers contrasts near 1.2.
- **H9.2.** Flowing gaps and pressure-equalization slots.
- **H9.3.** DB9.
- **H9.5.** The qualified Q2D envelope is the entry condition of L1.

**What this ADR does not change.**

- The Lorentz-damping treatment of DH1 and DH9. Step H8.8 (T50) measures it
  at Ha 10³ and 10⁴, holding and reaching the steady duct over times set by
  the measured slowest decay rate. A failure enters DH13's route with its
  envelope recorded.
- The solver decisions of DH2, DH3 and D36: conjugate gradients
  preconditioned by the separable solve stay the route for a non-separable
  operator.
- The static grid of DH5.

Consequences:

- The source grows by about 1,050 lines over steps H4.8 to H9.6, most of it in
  `thermal`. Each addition is a loan under DH15, with its repayment named in
  `plan.md` H.10.8.
- `thermal.py` passes the largest-module target (about 2,060 lines against
  1,716). That is a loan repaid when H4.7 frees `fully_developed.py` and the
  developed solves move to its successor.
- Open for the owner, with the defaults this ADR takes: `plan.md` H.10.12.
