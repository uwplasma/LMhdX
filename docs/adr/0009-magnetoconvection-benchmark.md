# 0009 — Step 5.4: the heated conducting duct of Mistrangelo et al. (2025)

Status: proposed 2026-10-19. It restates plan step 5.4 and amends:

- ADR 0008 D31: conjugate walls become a step;
- ADR 0008 D31.1: an inflow-outflow duct may evolve;
- plan D26: an evolving duct may take an advective outlet instead of D26's (D37);
- plan D27 and validation row 6: this benchmark moves to row 19 (D35).

Its step 5.4b solves walls that make the solve non-separable by D5's PCG
path.

Basis:

- `main` at `4e1b6b3` (1.11.0), with the plan's Phase 5 status. Main now
  carries Program H and ADR 0010: plan H7.2 and H7.3 are the work order for
  this ADR's wall stack and reference stack;
- a full read of Mistrangelo et al., *A code-to-code benchmark for
  magneto-convection in a horizontal duct*, Nucl. Fusion 65 (2025) 116006,
  covering all pages, Tables 1–5 and Figures 1–38 (page numbers below are the
  journal's);
- a survey of what LMhdX supports for this case;
- plan D27 and validation row 6, which already carry this benchmark's
  uniform-field gradient (−0.0863, 1 %).

## The benchmark

- **The duct** is horizontal and square, of half-width `a` = 23 mm, with 2 mm
  stainless-steel walls that conduct both current and heat (p. 2).
- **The field** is `B_y(x) = B0/2 [1 − tanh((|x| − 0.7315)/0.1)]`, `B0` =
  0.5 T (eq. 1): uniform for `|x| < 0.4` m, negligible beyond 1 m.
- **The flow** is along `+x`, and gravity along `−z`, normal to the field.
- **The heat.** A uniform flux heats the outer surface of the bottom wall
  `z = −a`, a side wall parallel to `B`, for `|x| ≤ 0.3` m. The rest of the
  outer surface is adiabatic and electrically insulating (p. 3).
- **The ends.** The inlet is at 300 °C. The inlet and outlet treatment and the
  domain length (2.0–3.0 m) were left to each code (Table 4).
- **The model** (eqs. 2–4) is inductionless and Boussinesq about 300 °C, with
  constant properties and no Joule, viscous or volumetric heat. The steel
  conducts, and `φ`, `j_n`, `T` and `q''` are continuous at the interface.

| | Ha | Re | Gr | Ri/N | Flow (pp. 7–15) |
|:--|--:|--:|--:|--:|:--|
| Case 1 | 235 | 3220 | 1.42e7 | 0.08 | Steady and laminar in the uniform field and the heated zone; unsteady only at the magnet's entrance and exit. |
| Case 2 | 235 | 1073 | 3.55e7 | 0.6 | Time-dependent: rolls about the field's axis, temperature fluctuations up to 30 K. |

**The data.** Six codes are compared, with no experiment. Every profile is in a
figure only. Case 1's printed scalars are a pressure gradient of −0.0855 (the
analytical developed flow gives −0.0863, p. 8), `Nu → 2.45` toward `x = 0.3` m
and a bulk rise of about 10 K (p. 10). The codes agree well in Case 1, apart
from FLUENT_UK. In Case 2 they disagree widely: CFX predicts a steady flow, and
the averaging windows run from 50 to 450 s (Table 5). The paper puts the
disagreements down to numerical diffusion, axial resolution and outlet
treatment (p. 16).

**What the paper leaves out.** It gives no material properties, only
references. It leaves unstated Case 2's flux and velocity, Pr, g, the length in
`Nu`, the heated width, the wall end-face conditions and the potential's and
pressure's reference levels. 5.4a's specification records each choice: its
source, whether it is printed, inferred or pinned, and which printed number it
reproduces.

## What LMhdX lacks today

1. **Thick walls on all four sides in 3-D.** A resolved wall is extra nodes on
   one axis's 1-D potential operator. Two resolved axes need one axial cell,
   because the corner correction assumes it. A thin wall is not this wall:
   `t_w` = 2 mm exceeds `a/√Ha` = 1.5 mm.
2. **Conduction in a wall.** The temperature lives on fluid cells, with one
   constant-coefficient fast-diagonal inverse (ADR 0008 D31).
3. **A time-dependent open duct.** `core3d.step` zeroes the inlet face of an
   open axis. Only the steady `axial.solve_open_duct` carries the inlet profile,
   and its Newton solve with inertia takes hours (`docs/physics/numerics.md`).
4. **A buoyant, inertial open flow.** The thermal step refuses buoyancy on a
   held flow and refuses an unheld open duct. No test runs buoyancy with
   momentum transport on.

Everything else the case needs exists: an arbitrary `B(x, y, z)`, graded faces,
limited momentum transport, implicit viscosity and damping, Boussinesq buoyancy,
the thermal open axis, and SOLVAX on the 0.34.0 floor.

**D35. The target is Case 1, gated. Case 2 follows as a reported comparison.**

- **Case 1's gate.** Its heated zone is steady and laminar, the codes agree
  there, and the paper prints scalars to check against. The gate is a band
  declared before the run, which combines the reference codes' spread, the
  digitizing uncertainty and LMhdX's discretization error, the last from its
  grid, time-step, tolerance and outlet-length studies. It holds on two meshes
  and two outlet lengths, at each observable the specification names. A known
  failure of the band stays visible.
- **Calibrated and independent.** The wall's conductivity ratio is calibrated
  on the developed pressure gradient (below), so that gradient is in no pass
  list. The independent observables are the velocity (the jets' height and
  position), the temperature, and the potential and currents.
- **Case 2** is run afterwards and reported against the codes' spread, with
  its burn-in, averaging window and statistical uncertainty fixed by a stated
  criterion before the comparison. It has no pass or fail: the references
  themselves differ in kind.
- **What the gate is.** Both are code-to-code evidence, not validation against
  an experiment. Joule heat is off in the fluid and in the walls, as in the
  reference equation; LMhdX's energy verification with Joule heat on stays a
  separate check.
- **Moving-flow claims wait.** The stabilised transient momentum step does not
  keep a pressure-balanced rest state with the field along `y`, so no moving
  thermal flow result is claimed until that step is corrected, or until a
  time-step refinement shows the effect is small for this configuration.
- **Validation row 6.** The paper prints no wall property, so 5.4a uses the
  printed gradient −0.0863 to fix the wall's effective conductivity ratio.
  Row 6's 1 % on that gradient is then met by construction and is not evidence.
  Its Mistrangelo part moves to row 19; its ALEX B1 part stays.

Reason: a gate needs a reference that is itself settled.

**D36. Conducting walls become cells of an extended grid, for the temperature
and the potential.**

- **The grid.** The fluid grid gains wall cells on its two transverse axes,
  with real widths and per-cell `σ`, `k` and `ρc_p`. Velocity and pressure stay
  on the fluid grid. The motional EMF is nonzero only in fluid cells, the
  Lorentz force uses the currents on fluid faces, and the walls carry no
  velocity.
- **The faces** take the distance-weighted harmonic mean of their cells'
  coefficients, the two half-cells' resistances in series. For heat this is the
  shared conductance `G = A/(δ_f/k_f + δ_s/k_s)`, so the two sides' fluxes
  cancel exactly. A contact resistance fits the same formula; the benchmark has
  none.
- **The solves.** Charge conservation and the temperature's capacity-weighted
  `M/dt + K` are both symmetric. Each is solved by `solvax.pcg_linear_solve`,
  which wraps `pcg` in `custom_linear_solve` with its transpose solve (D5). It
  is preconditioned by the fast-diagonal inverse at the volume-mean
  coefficients. The charge operator is singular; its null mode is handled
  inside the operator and preconditioner, so the same call serves.
- **Where the numerics live.** The variable-coefficient operator is
  `solvax.separable.flux_form_operator` (SOLVAX 0.34.0): harmonic-mean,
  series-resistance faces, a cell-wise shift, `pcg_linear_solve` with implicit
  derivatives, NaN on failure, and a singular variant that reports the removed
  defect. LMhdX calls it and keeps only `SolidWall`, `conjugate_grid`, the
  material maps and the physics. It gives each axis one boundary kind, and the
  open duct's axis has a Dirichlet inlet and a Neumann outlet, so until SOLVAX
  takes mixed ends (uwplasma/SOLVAX#172) the same operator stays in
  `poisson.py`, marked to move.
- **What a solid wall needs:** a thickness, a whole number of equal cells, a
  thermal conductivity, a volumetric heat capacity `ρc_p` and, for the charge,
  an electrical conductivity. These are the fields the 2.0 `WallLayer` gains.
- **What is differentiable.** The conductivities, the heat capacity and the
  widths of the wall cells are traced leaves in 2.0, so a gradient with respect
  to them, a wall's conductance and thickness among them, is not ruled out
  here. The number of cells and the walls' topology stay static.
- **The scope.** The extended grid is opt-in. Thin walls, resolved walls and
  the fluid-only thermal inverse keep their separable paths as the default.
  Wall Joule heat is not deposited (the reference has none), and thin walls get
  no thermal cells.
- **The order.** The temperature comes first: 5.4c's held flow is the
  developed section LMhdX solves today. The potential on the extended grid is
  needed first by the evolving duct of 5.4e.

Reason: steel and PbLi differ little here. The ratios are about 1.2 in `σ`
(1.18 measured, 1.25 pinned in 5.4a), 0.9 in `k` and 2.3 in `ρc_p`, so a
constant-coefficient preconditioner should leave few iterations. 5.4b measures
them, including at a high contrast, so that this choice has a documented range.

Rejected: a thin wall with thermal cells only, which is outside the thin wall's
validity; and a 3-D corner correction for the per-axis resolved nodes, which
needs new corner algebra and gives nothing for the heat.

**D37. An inflow-outflow duct may evolve in time.**

- **The inlet:** each step prescribes the inlet face's profile, as a lift, so
  the lift is consistent with the step.
- **The outlet:**
  - by default D26's: a zero axial gradient, with `p = 0`;
  - as an option, an advective condition `∂_t u + U ∂_x u = 0`, with `p = 0`;
  - the outflow corrected to the inflow in either;
  - the thermal outlet's existing policies, with `"error"` the default.

  Both are run in the outlet-length study. The steady open duct keeps D26's.
- **Buoyancy** is allowed once the evolving duct passes its own gates. The
  refusal in `build_thermal_step` stays for a held flow.
- **When the moving thermal flow is done:** an open manufactured solution
  converges at its order; the inlet, outlet and global energy budgets close; the
  backflow treatment is declared; the sensitivity to domain length and the
  temporal refinement are measured; and buoyant onset and decay are checked.
  Conduction on a held flow is not presented as the coupled result.
- **What this is not:** a general energy-stable open boundary. A long outlet
  region and an outlet-length study are part of every comparison, as the paper
  recommends (p. 17).
- **The fallback (D12).** Suppose Case 1's observables move between the two
  outlet lengths by more than the gate.
  - *Hypothesis:* the outlet reflects, so the change shrinks as the outlet
    lengthens, and differs between the two conditions.
  - *Budget:* one more doubling of the outlet region, with both conditions,
    within 5.4e's budget.
  - *Exit:* if the change between the two longer outlets falls inside the gate,
    the gate is taken there. If it does not, Case 1 is reported, not gated,
    with the outlet-length change as a stated uncertainty. The gate is not
    relaxed.

Reason: every one of the paper's codes evolves the flow, and its Case 1 jets
and fringe structures are inertial.

**D38. Where it lives and what it costs.**

- **The campaign** follows the repository's benchmark layout:
  - `src/lmhdx/data/benchmarks/specs/mistrangelo2025.toml` holds the frozen
    specification;
  - `src/lmhdx/data/benchmarks/references/` holds each figure's band and its
    axis calibration, with its figure, page and image hash;
  - `benchmarks/provenance.json` holds the sources and the benchmark's entry;
  - `validation/mistrangelo.py` holds the observables as the paper defines
    them, and the gate;
  - `scripts/digitize_mistrangelo.py` is the digitizer. It runs only on figure
    images extracted from the journal's PDF, which are not in the repository,
    so it stays out of the package and `validation/`.
  - Long runs, records and figures are release assets, not tracked files.
  - In 2.0 the specification, which is metadata, moves to `benchmarks/specs`
    outside the wheel, and a reference stays in the wheel only while a test
    reads it.
- **The references' size.** They ship in the wheel, so they are compact. A
  figure keeps its band, the range of the codes that agree, the curves that
  are not a code's, and its calibration.
  - Each is simplified to within half a pixel of every extracted pixel column,
    with nothing decimated before.
  - That bound costs 90 KB for Case 1 and 77 KB for Case 2, against a first
    budget of 40 KB per case. That budget was met only by decimating the
    extraction without a bound, which put the records up to tens of pixels off
    it at narrow features, as the repository review found.
  - The bound is kept, and the budget is the measured size, justified here as
    plan H.10.8 asks.
  - The wheel cap is 384 KB, and the references leave the wheel in 2.0 unless a
    test reads them.

  The per-code curves are not shipped: they are a release asset, with their
  SHA-256 in the provenance, and the digitizer regenerates them.
- **What is gated from them.**
  - In 5.4a: each figure's calibration, to an axis misfit under 0.6 px; the
    band's shape; and the specification's checks against figures, the pressure
    scale (Figure 8), the developed section's core (Figure 13, within 2 px),
    the bulk rise (Figure 17) and Case 2's inferred inputs (Figures 30 and 34).
  - In 5.4e: Case 1's observables, against the bands.
  - Case 2's comparisons are reported, not gated.
- **The capabilities** go in `thermal`, `core3d` and `poisson`. No module is
  added: the count stays at D34's 16.
- **The validation file cap** rises from 5 to 6, for `validation/mistrangelo.py`.
- **Ratchets:** each pull request raises them by the measured amount it adds.
- **Budget:** 5.4e is a campaign longer than an hour, so its issue records a
  hypothesis, a budget, the expected change in its diagnostics and an exit
  (D12). The estimate is about 400 × 90² cells and 30,000–50,000 steps: hours
  on one GPU in mixed precision, or about a day on the CPU in float64. 5.4b's
  iteration counts and timings replace the estimate.

## Steps

The steps and their exits are plan rows 5.4a–5.4f, written there once. 5.4a
comes first, and nothing new is built before it is frozen. Then 5.4b, followed
by 5.4c, which needs its walls; 5.4d may run in parallel with them. 5.4e needs
5.4b–5.4d, and 5.4f follows 5.4e.
