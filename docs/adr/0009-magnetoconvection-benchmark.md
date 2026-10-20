# 0009 — Step 5.4: the heated conducting duct of Mistrangelo et al. (2025)

Status: proposed 2026-10-19. It restates plan step 5.4 and amends two
decisions of ADR 0008:

- D31: conjugate walls become a step;
- D31.1: an inflow-outflow duct may evolve.

It also implements D5's PCG fallback for walls that make the solve
non-separable.

Basis:

- `main` at `4e1b6b3` (1.11.0), with the Phase 5 status of #276;
- a full read of Mistrangelo et al., *A code-to-code benchmark for
  magneto-convection in a horizontal duct*, Nucl. Fusion 65 (2025) 116006,
  covering all pages, Tables 1–5 and Figures 1–38 (page numbers below are the
  journal's);
- a survey of what LMhdX supports for this case;
- the post-merge thermal review's P2 and P3.

## The benchmark

**The duct.**

- A horizontal square duct of half-width `a` = 23 mm, with 2 mm stainless-steel
  walls that conduct both current and heat (p. 2).
- The field is `B_y(x) = B0/2 [1 − tanh((|x| − 0.7315)/0.1)]`, with `B0` =
  0.5 T, a fit to the magnet of the planned MaPLE experiment (eq. 1). It is
  uniform for `|x| < 0.4` m and negligible beyond `|x|` = 1 m.
- The flow is along `+x` and gravity is along `−z`, so gravity is normal to the
  field. The heated bottom wall `z = −a` is a side wall, parallel to `B`.
- A uniform flux heats the outer surface of the bottom wall for `|x| ≤ 0.3` m:
  4 W/cm² in Case 1.
- The rest of the outer surface is adiabatic and electrically insulating
  (p. 3).
- The inlet is at 300 °C. Inlet and outlet treatment and the domain length
  (2.0–3.0 m) were left to each code (Table 4).

**Its model** (eqs. 2–4):

- inductionless;
- Boussinesq, about `T0` = 300 °C, with constant properties at `T0`;
- an energy equation with no Joule, viscous or volumetric heat;
- conduction in the steel;
- continuity of `φ`, `j_n`, `T` and `q''` at the fluid–wall interface.

**The cases** (Table 1):

| | Ha | Re | Gr | Ri/N | Flow (pp. 7–15) |
|:--|--:|--:|--:|--:|:--|
| Case 1 | 235 | 3220 | 1.42e7 | 0.08 | Steady and laminar in the uniform field and the heated zone. Unsteady only at the magnet's entrance and exit. |
| Case 2 | 235 | 1073 | 3.55e7 | 0.6 | Time-dependent: rolls about the field's axis, temperature fluctuations up to 30 K. |

**The data.**

- Six codes are compared, with no experiment involved (p. 2).
- Every profile appears only in figures.
- The scalars printed in the text, all for Case 1:
  - a pressure gradient of −0.0855 from the codes, against −0.0863 for the
    analytical developed flow (p. 8);
  - `Nu → 2.45` toward `x = 0.3` m (p. 10);
  - a bulk-temperature rise of about 10 K (p. 10).
- The codes agree well in Case 1, apart from FLUENT_UK. In Case 2 they disagree
  widely: CFX predicts a steady flow, and the averaging windows run from 50 to
  450 s (Table 5).
- The paper puts the disagreements down to three causes: numerical diffusion,
  axial resolution, and outlet treatment (p. 16).

**What the paper leaves to the reader:**

- **The material properties.** Only references are given: Martelli et al.
  (2019) for PbLi, and a 1992 steel handbook.
- Martelli's PbLi at 573 K gives `Ha` 233 and `Re` 3158, where the paper prints
  235 and 3220.
- Its `ρc_p` of 1.867e6 J/(m³ K) gives a bulk rise of 10.1 K if the heated
  patch spans the full 50 mm outer width, and 9.3 K if it spans 46 mm. Figure
  17 reads 10.1 K.
- **Case 2's flux and velocity:** 10 W/cm² and 0.01 m/s, inferred from the
  ratios of `Gr` and `Re`.
- **The other unstated quantities:**
  - Pr and g;
  - the length in `Nu`, which is `a` by the figures;
  - the conditions on the wall's end faces;
  - the reference levels of the potential and the pressure.

## What LMhdX lacks today

1. **Thick walls on all four sides in 3-D.** A resolved wall is a set of extra
   nodes on one axis's 1-D potential operator. Two resolved axes need one cell
   along the third, because the corner correction assumes it
   (`core3d.ChannelProblem`, `poisson._corner_fix`).
   - The open-duct path rebuilds its problem without `wall_layers`, silently
     dropping a resolved wall.
   - A thin wall is not this wall: `t_w` = 2 mm exceeds `a/√Ha` = 1.5 mm.
2. **Conduction in a wall.** The temperature lives on fluid cells, with one
   constant-coefficient fast-diagonal inverse (ADR 0008 D31).
3. **A time-dependent open duct.** `core3d.step` zeroes the inlet face of an
   open axis and never applies the inlet profile. Only the steady
   `axial.solve_open_duct` carries it. Its Newton solve with inertia took 3.5 h
   at Ha 200, Re 40, and did not finish in 5.5 h at Ha 100, Re 33
   (`numerics.md`).
4. **A buoyant, inertial open flow.** The thermal step refuses buoyancy on a
   held flow and refuses an unheld open duct. Transient 3-D inertia is tested
   only near Re 5.

**What exists and carries over:**

- an arbitrary `B(x, y, z)` as cell arrays;
- per-axis graded faces;
- explicit central or limited momentum transport;
- implicit viscosity and magnetic damping;
- Boussinesq buoyancy in any direction;
- the thermal open axis with its backflow policies;
- SOLVAX `pcg`, on the 0.19 floor.

**D35. The target is Case 1, gated. Case 2 follows as a reported comparison.**

- **Case 1's gate.** Its heated zone is steady and laminar, the codes agree
  there, and the paper prints scalars against which a number can be checked.
  The gate is declared before the run: agreement with the codes that agree with
  one another, on two meshes and two outlet lengths, at each observable the
  specification names.
- **Case 2** is run afterwards, with its averaging protocol fixed first, and
  reported against the codes' spread. It has no pass or fail: the references
  themselves differ in kind.
- **What the gate is.** Both cases are code-to-code evidence, not validation
  against an experiment.
- **The heat terms.** The comparison switches Joule heat off, as the reference
  equation does. LMhdX's energy verification with Joule heat on stays a
  separate check.

Reason: a gate needs a reference that is itself settled.

**D36. Conducting walls become cells of an extended grid, for the potential and
the temperature.**

*The grid.*

- The fluid grid gains wall cells on its two transverse axes. The cells have
  real widths and carry per-cell `σ`, `k` and `ρc_p`.
- Velocity and pressure stay on the fluid grid, unchanged.
- The motional EMF is nonzero only in fluid cells. The Lorentz force uses the
  currents on fluid faces.
- The temperature is advected only in the fluid; the wall's velocity is zero.

*The face coefficients.*

- A face between two cells takes the harmonic mean of their coefficients. For
  heat that is the shared conductance `G = A/(δ_f/k_f + δ_s/k_s)` across the
  interface, so the two sides' fluxes cancel exactly.
- A contact resistance `R_c` fits into the same formula; the benchmark has
  none.

*The solves.*

- Both are symmetric: charge conservation, and the capacity-weighted
  `M/dt + K` of the temperature.
- Each is solved by SOLVAX `pcg`, preconditioned by the extended grid's
  fast-diagonal inverse with constant coefficients. The solve is wrapped in
  `custom_linear_solve(symmetric=True)` (D5).

*The scope.*

- The extended grid is opt-in.
- Thin walls, one-axis resolved walls and the fluid-only thermal inverse keep
  their separable paths and stay the default.
- Wall Joule heat and the thermal capacity of a thin wall's electrical unknowns
  are not included.

*Reason.* Steel and PbLi differ little here. The ratio of `σ` is about 1.3–1.6,
of `k` about 0.9, and of `ρc_p` about 2.3. So a constant-coefficient
preconditioner should leave few iterations, which 5.4b measures.

*Rejected.*

- **A thin wall with thermal cells only:** it is outside the thin wall's
  validity.
- **A 3-D corner correction for the per-axis resolved nodes:** it needs new
  corner algebra and gives nothing for the heat.

**D37. An inflow-outflow duct may evolve in time.**

- **The inlet:** each step prescribes the inlet face's profile, so the lift is
  consistent with the step.
- **The outlet:**
  - an advective condition, `∂_t u + U ∂_x u = 0`, with `p = 0`;
  - the outflow corrected to the inflow;
  - the thermal outlet's existing policies, with `"error"` the default.
- **Buoyancy:** once the evolving duct passes its own gates, buoyancy is
  allowed on it. The refusal in `build_thermal_step` stays for a held flow.
- **What this is not:** a general energy-stable open boundary. A long outlet
  region and an outlet-length study are part of every comparison, as the paper
  recommends (p. 17).

Reason: every one of the paper's codes evolves the flow, and its Case 1 jets and
fringe structures are inertial.

**D38. Where it lives and what it costs.**

- **The campaign** goes in `validation/`:
  - `validation/mistrangelo.py`: the specification, the observables as the
    paper defines them, and the gate;
  - `validation/data/mistrangelo2025/`: digitized curves as CSV, each with its
    figure, axis calibration and digitizer recorded.
  - Long runs, records and figures are release assets, not tracked files.
- **The capabilities** go in `thermal`, `core3d` and `poisson`. No module is
  added: the count stays at D34's 16.
- **Ratchets:** each pull request raises them by the measured amount it adds.
- **Budget:** 5.4e is a campaign longer than an hour, so its issue records a
  hypothesis, a budget and an exit (D12). The estimate is about 400 × 90² cells
  and 30,000–50,000 steps, which is hours on one GPU in mixed precision or about
  a day on the CPU in float64. 5.4b's iteration counts replace the estimate.

## Steps

| Step | What | Exit |
|:--|:--|:--|
| 5.4a | The specification, frozen before new numerics (listed below) | The specification and data are committed. The 2-D developed section with resolved walls on both axes, which LMhdX supports today, is reported: its pressure gradient against −0.0863, and its jets against Figures 13 and 26. |
| 5.4b | Walls as cells of an extended grid (D36): the potential first, then the temperature | Two-material steady conduction and transient storage against exact solutions. Interface fluxes cancel to rounding. Fluid-plus-wall energy and charge balance close. The 2-D section matches today's resolved-wall path. The thin-wall limit is recovered as `t_w → 0`. PCG iterations and cost are recorded. Directional derivatives are taken in the inputs the step accepts. |
| 5.4c | A reduced Case 1, reported and not gated: the heated zone, −0.4 ≤ x ≤ 0.4 m, with the developed MHD flow held, conjugate walls and no buoyancy | Interface and outer-wall temperatures, bulk temperature and `Nu` are compared against Figures 9, 10, 17, 18 and 38. This measures what buoyancy and the fringe add. |
| 5.4d | The evolving open duct (D37); then buoyancy on it | The steady limit matches `axial.solve_open_duct` (Stokes, and Re 10 at Ha 100). Mass and charge are conserved. A prescribed throughflow is held. The physical pressure drop and the heat ledger close. A coupled manufactured solution converges at its order. Backflow is handled. Outlet-location sensitivity is measured. Zero-buoyancy parity holds. |
| 5.4e | Case 1, gated (D35): the full 3-D duct with the fringe, thick walls, inertia, buoyancy and heat | Every 5.4a observable lies within the gate declared before the run. That holds on two meshes and two outlet lengths. |
| 5.4f | Case 2, reported (D35), with the averaging window and start-up removal fixed before the run | Every 5.4a observable is compared against the codes' spread, with fluctuation statistics kept. |

5.4a's specification records:

- **The properties**, chosen from cited sources: Martelli 2019 for PbLi at
  573 K, and a cited 316-type steel at 573 K. It states which printed
  quantities they reproduce: `Ha`, `Re`, `Gr` and the bulk rise.
- **Every inferred value, flagged as inferred:**
  - Case 2's flux and velocity;
  - the heated width;
  - `L = a` in `Nu`;
  - Pr and g;
  - the wall end-face conditions;
  - the reference levels of the potential and the pressure.
- **Case 1's observables**, as the paper defines them: eqs. 6–8 and Figures
  6–10, 12, 13 and 15–19.
- **Case 2's observables:** Figures 23, 24, 26–29, 34, 35 and 37.
- **The digitized curves**, each with its provenance.

**The order:**

- 5.4b and 5.4c come first, then 5.4d.
- 5.4d may start once 5.4a is frozen.
