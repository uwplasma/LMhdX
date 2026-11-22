# Mistrangelo et al. 2025: a heated conducting duct

Mistrangelo et al., *A code-to-code benchmark for magneto-convection in a
horizontal duct*, Nucl. Fusion 65 (2025) 116006, compares six codes on a heated
PbLi duct with conducting steel walls in a fringing field. There is no
experiment. Plan step 5.4 reproduces it in stages (ADR 0009). This page records
step 5.4a: the frozen specification, the developed section, and both cases'
digitized figures; step 5.4c, Case 1's reduced heated zone; step 5.4e, Case 1
gated, with its edge study; and step 5.4f, Case 2 reported, with a seeded study of its rolls' onset and a
rerun with the Lorentz force explicit.

The specification is `src/lmhdx/data/benchmarks/specs/mistrangelo2025.toml`.
Each value there is marked *printed*, *inferred* or *pinned*, with its source,
and it separates what is calibrated from what is independent. The paper prints
no material properties: the PbLi `μ`, `σ` and `β` are pinned to the printed Ha,
Re and Gr, and the steel's conductivity ratio to the developed gradient
−0.0863 σu₀B₀², which is therefore no acceptance check.

## The inferred values

Each is confirmed by a figure (the checks below):
- the heated width, the full 50 mm outer width (Figure 17);
- Case 2's 10 W/cm² and 0.01 m/s, from the ratios of Gr and Re (Figures 34
  and 30);
- `L = a` in `Nu` (Figures 18 and 38);
- the pressure scale `σ u₀ B₀²` (Figure 8).

## Figure 13: the developed section

`python -m validation.mistrangelo` solves the flow in the uniform field on a
section one cell long, with the steel resolved, at Ha 235.

| Cells (fluid / layer / wall) | −dp/dx in σu₀B₀² | Side jet | Centre |
|:--|--:|:--|--:|
| 48 / 6 / 6 | 0.08599 | 2.826 u₀ at \|z\| = 0.928 a | 0.8469 u₀ |
| 72 / 9 / 8 | 0.08622 | 2.810 u₀ at 0.929 a | 0.8490 u₀ |
| 96 / 12 / 10 | 0.08630 | 2.803 u₀ at 0.929 a | 0.8497 u₀ |

The profile is the independent check, against Figure 13's analytical markers.
On 96/12 the core (\|z\| ≤ 0.7 a) is within 2.09 px of each of the markers' 37
columns, 0.62 % low on average, gated within the record's uncertainty (2.44 px)
and 1 %. The paper puts the jets' maximum at 0.935 a (p. 12); the section has
0.929 a.

## The digitized figures

`src/lmhdx/data/benchmarks/references/mistrangelo2025-case1.json` (90 KB) and
`-case2.json` (77 KB) are written by `python scripts/digitize_mistrangelo.py DIR`
from the paper's figure images; the script's docstring says how a figure is
read. Each record is within 0.53 px of every extracted pixel column, and
carries its digitizing uncertainty, 2.3–2.5 px. A band is the envelope of the
codes chosen to agree, not a confidence interval;
`validation.mistrangelo.band_misfit` compares against it, widened by that
uncertainty. `agree_basis` says who chose the codes: "the paper", or a
"figure reading" where the paper does not say.

Every code's full curve, with the figure images (the paper is CC BY 4.0), is in
the release assets `mistrangelo2025-case1-extraction.zip` and
`-case2-extraction.zip` on
[mistrangelo2025-references](https://github.com/uwplasma/LMhdX/releases/tag/mistrangelo2025-references).
`benchmarks/provenance.json` holds every SHA-256.

### Case 1

| Figure | Quantity | Left out of the band | Why (page) | Chosen by |
|:--|:--|:--|:--|:--|
| 6 | T on the axis | FLUENT_UK; the band within −0.75 < x < 0.8 m | it deviates (p. 7) | the paper |
| 7 | u on the axis | FLUENT_UK; Figure 6's window | its core velocity deviates (p. 8) | figure reading |
| 8 | p on the axis | FLUENT_UK | all agree but FLUENT_UK (p. 8) | the paper |
| 9 | T across, x = −0.3 m | FLUENT_CIEMAT, OF_ORNL | their axial resolution (p. 8) | the paper |
| 10 | T across, x = 0 | FLUENT_UK | it lies lower | figure reading |
| 11 | T across, OF_KIT at five stations | no band | one code (pp. 8–9) | — |
| 12 | u across, x = −0.3 m | FLUENT_UK, CFX | as p. 9 says of Figure 13 | figure reading |
| 13 | u across, x = 0 | FLUENT_UK, CFX | the analytical solution agrees with the other four (p. 9); CFX is not separable from its markers | the paper |
| 15 | φ on the interface | none | most of the data fits well together (p. 10) | the paper |
| 16 | φ across, x = 0 | FLUENT_UK | a slight deviation (p. 10) | the paper |
| 17 | bulk T, eq. 6 | FLUENT_UK | non-physical (p. 10) | the paper |
| 18 | Nu, eq. 7, on L = a | FLUENT_UK | as p. 10 says of Figure 16 | figure reading |
| 19 | E_t, eq. 8 (log) | FLUENT_UK, CFX; COMSOL not shown | as before (p. 10); CFX's diffusion (p. 16) | figure reading |
| 38 | OF_KIT's T_w and T_b, both cases | no band | one code (p. 16) | — |

Checks (`tests/test_steady.py`):
- **Pressure scale.** Figure 8's slope over \|x\| < 0.4 m is −568.6 Pa/m,
  against −564.5 Pa/m for the printed −0.0855 σu₀B₀²: 0.8 %, gated within 1 %.
- **Heated width.** The energy balance with the full outer width heated gives
  310.12 °C, inside Figure 17's band over the outlet plateau to half a pixel; a
  46 mm patch would be 0.8 K off.
- **Length in Nu.** Figure 38's `T_w − T_b` at x = 0.29 m, 19.28 K, gives
  `Nu` = 2.34 on L = a, inside Figure 18's band there (2.23–2.41); L = 2a would
  double it. The printed "Nu → 2.45" (p. 10) matches Figure 18's codes near
  x = 0.3 m.

### Case 2

Case 2 is reported, not gated (ADR 0009 D35): a time-dependent flow whose
figures are time averages, from codes that each chose their own domain, inlet,
outlet and averaging window (Tables 2–5). COMSOL's outlet backflow (pp. 12, 14)
is kept out of the bands of Figures 23, 24 and 30–32, which stop at x = 0.7 m.
LMhdX's averaging protocol, a proposal fixed before any run as D35 asks:
averaging starts after one flow-through and once two consecutive 50 s means
agree within 1 % of the bulk rise, over at least 450 s in five batches whose
independence is checked before any statistical error is quoted. No Case 2
result is claimed before the corrected momentum step. 5.4f's draft (below) makes it exact.

| Figure | Quantity | Left out of the band | Why (page) | Chosen by |
|:--|:--|:--|:--|:--|
| 23 | u on the axis | CFX, OF_ORNL | CFX differs; OF_ORNL fluctuates, numerical problems (p. 12) | the paper |
| 24 | T on the axis | CFX, OF_ORNL | CFX's steeper rise; OF_ORNL fluctuates (p. 12) | the paper |
| 26 | u across, x = −0.3 m | CFX | all but CFX reproduce the analytical solution (p. 13); its markers merge with CFX's line | the paper |
| 27 | u across, x = 0 | CFX | its profile differs, as p. 12 says of Figure 23 | figure reading |
| 28 | T across, x = −0.3 m | all but COMSOL and OF_KIT | the two fit well together (p. 13) | the paper |
| 29 | T across, x = 0 | CFX | all predict the thicker layer (p. 13), but CFX's wall is 20 K hotter | figure reading |
| 30 | p on the axis | none | p. 14 | the paper |
| 31 | φ, z = −0.023 m | FLUENT_CIEMAT, FLUENT_UK | their curves are instantaneous (p. 14) | the paper |
| 32 | φ, z = 0.023 m | FLUENT_CIEMAT, FLUENT_UK | their curves are instantaneous (p. 14) | the paper |
| 33 | φ across, x = 0 | CFX | good agreement (p. 15), but CFX is offset | figure reading |
| 34 | bulk T, eq. 6 | OF_ORNL, COMSOL | unphysical (p. 15, caption; p. 16) | the paper |
| 35 | E_t, eq. 8 (log, seven decades) | CFX, OF_ORNL, COMSOL | CFX differs; OF_ORNL and COMSOL unphysical in the kinetic energy (pp. 15–16) | the paper |
| 37 | Nu, eq. 7, on L = a | CFX, OF_ORNL, COMSOL | CFX differs; the bulk temperatures of OF_ORNL and COMSOL fluctuate (p. 16) | figure reading |

Figure 26 is not compared with the section's jets: what read as CFX's curve
there is the analytical solution's markers, merged in the JPEG.

Checks (`tests/test_steady.py`):
- **Velocity, 0.01 m/s.** Figure 30's slope over \|x\| < 0.4 m is −191.2 Pa/m,
  against −189.9 Pa/m for the developed −0.0863 σu₀B₀² at that velocity: 0.7 %.
- **Flux, 10 W/cm².** Over 0.5 < x < 0.9 m the top of Figure 34's band is
  375.71–375.99 °C, against 375.93 °C for the energy balance: within the
  record's uncertainty (0.45 K).

## Step 5.4c: the heated zone, its flow held

A reduced Case 1, reported and not gated (ADR 0009). It is set up as follows:

- **Domain and flow.** The heated zone, −0.4 ≤ x ≤ 0.4 m. Its flow is the
  developed section's at the pinned wall ratio, scaled to u₀ and a, and the
  same at every station. There is no fringe and no buoyancy.
- **Walls.** The steel conducts and stores heat in 6 cells per wall (plan
  step 5.4b).
- **Thermal conditions.** The PbLi enters at 300 °C and leaves with no
  conductive flux. The outer bottom surface takes 4 W/cm² for |x| ≤ 0.3 m over
  its full outer width, and the rest of the outer surface is insulated.
- **Mesh and transport.** 48 fluid cells across (6 in each layer) and 160
  along, 576,000 cells with the walls. Van Leer transport at Courant 0.5,
  Δt = 0.030 s.
- **The run.** From 300 °C to a steady state, where storage is 1e-4 of the
  1200 W heat input. That took 2,400 steps and 3.5 minutes on an RTX 3060
  Laptop GPU in float64 with JAX 0.12: 0.085 s a step warm, with 16 PCG
  iterations per step at the end.

`python -m validation.mistrangelo heated DIR` writes its record,
`heated_zone.json`. The record of this run is the release asset
`mistrangelo2025-case1-heated-zone.json` of
[LMhdX Research Assets v2](https://github.com/uwplasma/LMhdX/releases/tag/lmhdx-research-assets-v2)
(ADR 0009 D38), SHA-256 `908708d0d4a53e917088f385c28062c9c5d7adceae70f66abe8e0cdc09c21dc5`.

**Against the digitized references:**

| Figure | Quantity | Inside the codes' band | Within half a pixel | Largest outside | Mean offset |
|:--|:--|--:|--:|--:|--:|
| 9 | T(z) at x = −0.3 m, y = 0 | 35 % | 73 % | 0.15 K, in the heated wall | +0.030 K |
| 10 | T(z) at x = 0, y = 0 | 93 % | 100 % | 0.024 K | −0.079 K |
| 17 | bulk T(x) | 86 % | 96 % | 0.025 K | +0.021 K |
| 18 | Nu(x), −0.3 < x < 0.3 m | 98 % | 98 % | 3.3, at the heater's leading edge | +0.12 |
| 38 | OF_KIT's T_w(x) and T_b(x) | — | — | 2.3 K and 0.79 K at the leading edge; 0.41 K and 0.12 K for \|x\| < 0.28 m | −0.079 K and −0.049 K |

**What the numbers show.**
- The bands at x = −0.3 m are narrow: 0.01 K, a pixel of Figure 9, where the
  heat has only begun to enter. So the fractions inside them say little.
- The outlet bulk rise is 10.12 K. The paper says about 10 K, and Figure 17's
  band past x = 0.3 m reads 10.04–10.15 K.
- Nu falls to 2.33 at x = 0.29 m. The paper prints 2.45 toward 0.3 m.
  Figure 18's band is 2.23–2.41 there, and Figure 38's OF_KIT curves give 2.34
  (2.29–2.40 if each curve is a pixel off the other way).
- The largest differences sit at the heater's leading edge, x = −0.3 m, where
  T_w and Nu jump within a cell of the 5 mm axial mesh: half a cell's shift
  there is worth kelvins of T_w. Away from the edges, LMhdX stays within 0.41 K
  of OF_KIT's T_w and 0.12 K of its T_b.
- So in the heated zone, buoyancy (Ri/N = 0.08) and the fringe change Case 1's
  temperatures by a few tenths of a kelvin at most, a few percent of the 10 K
  rise. Its codes' spread is of the same size.

`tests/test_steady.py` runs the same problem on 16 cells across and 16 along.
The coarse run:
- closes its ledger over the fluid and the steel;
- takes T_w as the interface value between the cells either side of it;
- defines Nu over the heated stations only;
- is compared with every figure listed here.

## Step 5.4e: Case 1, gated — the campaign (draft, declared before the run)

Plan step 5.4e gates Case 1 (ADR 0009 D35). This record states the hypothesis, the runs, the rule for each check,
the budget and the exit before any production run (D12). It is a draft until agreed.

**The problem.** `validation.mistrangelo_duct.case1` builds the whole duct: eq. 1's fringe along `y`; steel walls that
carry current at the pinned ratio and conduct and store heat (5.4b); the flow evolving through the open duct of 5.4d,
with limited momentum transport and buoyancy; van Leer heat transport; the PbLi entering at 300 °C on the developed
profile of the inlet section; the heated patch. The axial mesh is fine over |x| ≤ 1 m and grows into buffers out to
±1.5 m, the long end of the codes' 2–3 m domains.

**Checked so far** (coarse mesh, 12 cells across, 5 cm along): the ledger closes to 1e-12 and the throughflow holds at
every station. In the Stokes limit the uniform field's flow at x = 0 tends to the developed section on its transverse
mesh (jets 3.026 against 3.040 u₀, still closing). With inertia, the jets there are 8 % higher: the fringe's
development, with the paper's "almost developed" jets at 0.935 a beyond the section's 0.917.

**Hypothesis.** The time-averaged observables lie within the codes' band, widened by the digitizing uncertainty and by
LMhdX's own uncertainty, on every run below. In the heated zone, buoyancy and the fringe move the temperatures from
5.4c's held flow by a few tenths of a kelvin.

**Gated and reported.**
- Gated: Figures 6 and 7 (T and u on the axis), 9 and 10 (T(z)), 12 and 13 (u(z)), 15 and 16 (the potential),
  17 (the bulk temperature), 18 (Nu) and 19 (the transverse energy).
- Reported: Figure 8, the pressure, whose uniform-field slope is calibrated; Figures 11 and 38, one code's curves
  only.

**Runs.** The time step keeps `dt/Δx` at 3 s/m on every mesh.
- M1L1: 48 cells across (6 in each layer, 6 per wall), 5 mm along the core, outlet at 1.5 m, `dt` 0.015 s.
- M2L1: 64 across (8, 8), 4 mm along, outlet at 1.5 m, `dt` 0.012 s.
- M3L1: 84 across (10, 10), 3.33 mm along, outlet at 1.5 m, `dt` 0.01 s. This third mesh checks that the mesh
  change from M1 to M2 does not understate the discretization error.
- M1L2 and M2L2: the outlet at 2.25 m.
- M1L1 at half the time step.
- M1L1 with every solver's stop 100 times looser: the conduction PCG (1e-11), the charge PCG through the walls
  (1e-13) and the momentum stage's Schur PCG (1e-13).

**Averaging.** After two flow-through times (2 L/u₀, L the duct's length), average over one more, at every step. The
statistical uncertainty is twice the spread of four sub-window means.

**The rule, at each gated point.**
- The excess is `band_misfit`: how far LMhdX lies outside the codes' spread, widened by the digitizing uncertainty in
  both axes.
- LMhdX's uncertainty U is the largest of these at that point:
  - the mesh changes, M1 to M2 and M2 to M3;
  - the outlet-length change, on M1 and on M2;
  - the time-step and tolerance changes;
  - the statistical uncertainty.
- A point passes if its excess is at most U, and U is at most half the band's local width. A larger U means the run
  is not resolved enough to gate there.
- The gate holds if every gated point passes on M1L1, M2L1, M3L1, M1L2 and M2L2. A failure is recorded, not
  dropped.

**Diagnostics expected every step.**
- The heat ledger within 1e-9 of its largest term.
- Mass and charge at their solve floors.
- The throughflow held at every station.
- The advective Courant number at most 0.5.
- No flow returning through the outlet. Case 1 is steady and laminar in the heated zone, and unsteady only at the
  magnet's ends.

**Budget.**
- Measured on one NERSC Perlmutter A100 (40 GB), in float64 with NVIDIA's JAX 26.10 container:
  - M1 (48 across, 5 mm along; 1.62 million cells): 0.038 s a step, peak 0.5 GiB.
  - M2 (64 across, 4 mm along; 3.56 million cells): 0.093 s a step, peak 1.0 GiB.
- For comparison, an RTX 3060 Laptop took 0.53 s a step at 850,000 cells.
- Three flow-through times (300 s) take 20,000 steps on M1 at 0.015 s (about 13 minutes) and 25,000 on M2 at
  0.012 s (about 40 minutes). M3 (about 7 million cells) runs 30,000 steps at 0.01 s. The seven runs take about
  5 GPU hours, a little over a charged node-hour in the shared QOS.
- The time step keeps the advective Courant number under 0.5. In the first 8 s at 0.02 s it rose from 0.25 to
  0.32 as the jets formed.

**Exit (D12).**
- A run whose heated zone is not statistically steady after four flow-through times is reported, not gated.
- A point that fails because of the outlet takes D37's fallback: one more doubling of the outlet region, within this
  budget.
- Otherwise Case 1 is reported as code-to-code agreement, with every failure shown. The band is not relaxed.

## Step 5.4e: Case 1, gated — the results

The seven runs above ran as drafted, before the record's review, on NERSC Perlmutter: one A100 each, float64,
NVIDIA's JAX 26.10 container (image `20cf0aa0e1`), commit `1fc5507`. The observables and the gate are commit
`a83d942`'s; the tag `mistrangelo2025-case1-runs` keeps both. The rule was applied as written.

**The gate holds on six of the eleven gated figures: 6, 7, 10, 15, 16 and 17. It does not hold on 9, 12, 13, 18
and 19.** By the exit, Case 1 is reported as code-to-code agreement, with every failure shown below. The band is not
relaxed, and the rule is not amended after the run.

**The runs.**

| Run | Cells | `dt` (s) | Simulated | s a step | Wall time | Courant max | Ledger max |
|:--|--:|--:|--:|--:|--:|--:|--:|
| M1L1 | 1.62 M | 0.015 | 300 s | 0.039 | 13 min | 0.31 | 4.2e-12 |
| M2L1 | 3.56 M | 0.012 | 300 s | 0.092 | 38 min | 0.32 | 4.8e-12 |
| M3L1 | 7.16 M | 0.010 | 300 s | 0.199 | 100 min | 0.34 | 5.1e-12 |
| M1L2 | 1.68 M | 0.015 | 375 s | 0.040 | 17 min | 0.31 | 5.8e-12 |
| M2L2 | 3.71 M | 0.012 | 375 s | 0.091 | 47 min | 0.32 | 8.3e-12 |
| M1L1, half the time step | 1.62 M | 0.0075 | 300 s | 0.035 | 24 min | 0.16 | 7.5e-12 |
| M1L1, stops 100 times looser | 1.62 M | 0.015 | 300 s | 0.031 | 11 min | 0.31 | 5.2e-10 |

The seven took 4.2 GPU hours, against the 5 budgeted. M3L1's observables were computed on an A100 too; the 7.2
million cells do not fit a laptop's memory.

**Diagnostics against those expected.**
- Every run stayed finite.
- The heat ledger stayed within 1e-11 of its largest term, and 5.2e-10 with the looser stops, inside the 1e-9
  expected.
- The mass residual stayed under 5e-10 (1.8e-9 with the looser stops), and the outlet carried the prescribed
  6.348e-5 m³/s, so the throughflow held at every station.
- The charge residual (the currents' divergence) follows the solver's stop: 0.007, 0.015 and 0.024 on M1, M2 and
  M3, and 0.072 with the stops 100 times looser.
- The advective Courant number peaked at 0.34, under the 0.5 allowed.
- No run had flow returning through the outlet.
- Every run was steady over its window: storage stayed within 3e-7 of the 1200 W heat input. No unsteadiness
  appeared at the magnet's ends either. So no run falls under the exit's "not statistically steady".

**What sets U.** The mesh change, M1 to M2 or M2 to M3, sets U on every figure.
- The outlet, time-step, tolerance and statistical terms are each at most 0.006 of the band's half-width on any
  figure. The runs reach the same steady state whatever the time step, the solver's stops or the outlet's length.
- The largest of them, 0.012 µV on Figures 15 and 16, is the potential's gauge. The longer domain shifts its zero
  mean, uniformly to 7e-6 µV.
- So no point fails because of the outlet, and D37's fallback is not taken.

**Against the bands.** The points are the band's points that every run covers. "Inside the band" is the band as
drawn. A point passes with the digitizing uncertainty added in both axes, since the bands are about a pixel wide, so a
figure can hold with fewer points inside the band as drawn.

| Figure | Quantity | Points | Inside the band, M3L1 | Not passing, M1L1 / M2L1 / M3L1 / M1L2 / M2L2 | Largest U | Gate |
|:--|:--|--:|--:|:--|--:|:--|
| 6 | T on the axis | 168 | 71 % | 0 / 0 / 0 / 0 / 0 | 0.014 K | holds |
| 7 | u on the axis | 60 | 58 % | 0 / 0 / 0 / 0 / 0 | 0.05 mm/s | holds |
| 9 | T(z) at x = −0.3 m | 155 | 58 % | 3 / 3 / 3 / 3 / 3 | 0.047 K | does not hold |
| 10 | T(z) at x = 0 | 244 | 46 % | 0 / 0 / 0 / 0 / 0 | 0.045 K | holds |
| 12 | u(z) at x = −0.3 m | 174 | 70 % | 3 / 2 / 1 / 3 / 2 | 0.93 mm/s | does not hold |
| 13 | u(z) at x = 0 | 235 | 68 % | 4 / 4 / 3 / 4 / 4 | 0.85 mm/s | does not hold |
| 15 | potential on z = −a | 207 | 84 % | 0 / 0 / 0 / 0 / 0 | 0.076 µV | holds |
| 16 | potential across z at x = 0 | 189 | 98 % | 0 / 0 / 0 / 0 / 0 | 0.55 µV | holds |
| 17 | bulk temperature | 169 | 75 % | 0 / 0 / 0 / 0 / 0 | 0.004 K | holds |
| 18 | Nu | 169 | 93 % | 2 / 2 / 2 / 2 / 2 | 0.86 | does not hold |
| 19 | transverse energy E_t | 287 | 41 % | 15 / 16 / 16 / 15 / 16 | 0.072 decades | does not hold |

**Where the five miss.**
- **Figure 9.** Three points are too uncertain to gate. They sit in the heated wall within 0.4 mm of its outer face
  (z = −24.6 to −24.8 mm), at the heater's leading edge.
  - The band there is 0.005–0.023 K wide. With the digitizing uncertainty, its half-width is 0.025–0.034 K.
  - The M1-to-M2 change there is 0.037 K. The M2-to-M3 change is 0.024 K, and M3's excess is zero at all three.
  - *Why:* x = −0.3 m is where the outer heat flux steps from zero to q''. The value there is interpolated between
    the cells either side of the step, so it converges slowly, toward the band: 303.08, 303.04 and 303.02 C at
    z = −24.8 mm from M1 to M3. M3 is 0.09 K above the band as drawn, 3 % of the 3 K local rise, in a gradient
    where 0.1 K is 0.09 mm of z. Numerical.
- **Figure 12.** At the side-layer jet's peak (z = −21.4 mm), the misfit beyond the band falls with the mesh:
  - 0.21 mm/s on M1, 0.18 on M2 and 0.04 on M3, against a U of 0.14 mm/s. So the M1 and M2 runs fail there and M3
    passes.
  - At z = +21.3 mm, M1 fails, by 0.28 mm/s against a U of 0.23 mm/s.
  - One point on the jet's flank (z = −22.7 mm) is too uncertain to gate: the M1-to-M2 change is 0.93 mm/s, above
    the 0.83 mm/s half-width. The M2-to-M3 change is 0.02 mm/s, and no run lies outside there.
  - *Why:* the jets' resolution. M3 is 0.66 mm/s (0.8 %) above the band as drawn and inside the widened band.
- **Figure 13.** The jet's peak at x = 0, on the +z side only (z = 21.2 to 21.5 mm), misses, and refinement does
  not close it.
  - At z = 21.4 and 21.5 mm, M3 lies 0.75 and 0.42 mm/s beyond the widened band, against a U of 0.12 and
    0.14 mm/s.
  - At z = 21.2 mm, M3 lies 0.32 mm/s beyond the band, within its U of 0.38 mm/s. The M1 and M2 runs fail there.
  - The peak moves slowly with the mesh: 83.86, 83.74 and 83.66 mm/s from M1 to M3. Extrapolating that sequence
    leaves about 0.6 mm/s beyond the band, 0.7 % of the peak.
  - The −z side passes.
  - One flank point (z = 22.9 mm) is too uncertain to gate: the M1-to-M2 change is 0.85 mm/s, above the 0.61 mm/s
    half-width. The M2-to-M3 change is 0.28 mm/s.
  - The largest gap is 1.5 mm/s above the band as drawn, 1.8 % of the peak.
  - *Why, the mesh, about a third:* the developed section's jet peak (`section_flow`) on 48, 64, 84 and 128 cells
    across is 84.79, 84.40, 84.18 and 83.98 mm/s. It converges near 83.8 mm/s (order 1.6–1.8), 0.4 mm/s below its
    value on M3's mesh. More side-layer refinement trims the jets by that much, not into the band.
  - *Why, the wall's ratio, an input:* it is pinned to the printed gradient. Matolich's measured ratio lowers the
    developed jet by 0.85 mm/s (1 %) but misses the gradient by 4 %. The jet moves about 2 % per 10 % of the ratio.
    So matching the printed gradient raises the jets about 1 %. This suggests the codes' walls differ slightly
    from LMhdX's resolved steel; it is inferred, not checked against the codes.
  - *Why, buoyancy's asymmetry:* heated from below, the −z jet runs faster than the +z jet: by 2.2 mm/s in LMhdX,
    against 2.9 mm/s between the band's top edges. The weaker asymmetry leaves the +z peak high. It is not traced
    to an input.
- **Figure 18.** No point fails. The first two heated points, at the heater's leading edge, are too uncertain to
  gate.
  - At x = −0.2965 m, Nu is 22.7, 23.1 and 23.1 from M1 to M3. The band narrows there to one code's value, 19.4, so
    its widened half-width is the digitizing 0.09. The M1-to-M2 change, 0.39, exceeds it.
  - At the next point, x = −0.2956 m, M3 moves toward the band (21.2, 21.1 and 20.5 from M1 to M3). That M2-to-M3
    change, 0.57, exceeds the 0.47 half-width.
  - *Why:* Nu is singular at the heater's edge. Every mesh's cell values lie on one curve, Nu ≈ 34 d^−0.37 with d
    the distance from the edge in mm, so LMhdX's Nu is converged point by point. The band's first point is 3.5 mm
    in, between the first two cells, 3.3–5 mm apart. Linear interpolation between them overestimates a curve this
    sharp, and M2 and M3 agree at 23.1 by coincidence.
  - On the curve, Nu is about 21.5 at the first point, 11 % above the one code, and 19.7 at the second, 1 % above
    the band. Past 5 mm from the edge LMhdX lies in the band. 5.4c's 3.3 there is the same interpolation on its
    5 mm cells.
  - Numerical: axial cells of about 0.5 mm within a few centimetres of each edge would resolve it.
- **Figure 19.** Upstream of x = −0.964 m, E_t is converged but far below the band.
  - M2 and M3 agree to within 0.03 decades there.
  - LMhdX's E_t runs from 5e-17 at x = −1.25 m to 1e-7 at −0.964 m.
  - For x < −1.006 m the band is a single value, about 1e-4. Beyond that, the codes' lower edge lies between 1e-7
    and 1e-4.
  - LMhdX's inflow is the inlet section's developed profile, so little transverse motion arrives from upstream. The
    specification leaves the inlet profile and the domain's length to each code (Table 4).
  - *Why:* the setup, not resolution. Upstream of x = −1.006 m the band is one code's, with transverse motion of
    about 1 % of u0 where the field is nearly zero, most likely from its inlet.
  - The two most upstream points are also too uncertain to gate. Their mesh change, 0.06–0.07 decades, exceeds the
    single-valued band's digitizing half-width, 0.02 decades.

**What the misses are.** Numerical, or from the inputs; only part of Figure 13's is not traced.
- *Numerical, at the heater's leading edge, where the heat flux steps:* Figures 9 and 18. LMhdX converges toward
  the band; axial refinement near x = ±0.3 m would resolve them.
- *Numerical, the jets' resolution:* Figure 12's jet peak fails on M1 and M2 and passes on M3.
- *Inputs:*
  - Figure 13's +z jet peak, 0.7 % beyond the widened band after extrapolation: the pinned wall ratio, with a
    weaker buoyant asymmetry not traced to an input;
  - Figure 19 upstream of x = −0.964 m: the inlet, which each code chooses.
- *Too uncertain to gate, though converging:*
  - Figure 9's three points;
  - the jets' flanks in Figures 12 and 13;
  - Figure 18's two points;
  - Figure 19's two most upstream points.

**A property of the rule.** U takes the larger of the two mesh changes. So a point that the M1-to-M2 change leaves too
uncertain to gate stays so, however M3 converges.
- For information only, and not the rule: suppose U dropped the M1-to-M2 change, kept every other term, and only
  M3L1 were judged. Then Figures 9 and 12 would hold, and Figures 13, 18 and 19 would not.
- Any change to the rule would be a new declaration (D12), made in review.

**Figure 38, reported.** M3L1 against OF_KIT:
- T_w lies within 0.40 K for |x| < 0.28 m and 0.44 K at the leading edge, with a mean offset of −0.27 K.
- T_b lies within 0.12 K, and 0.79 K at the leading edge, with a mean offset of −0.05 K.
- 5.4c's held flow had 0.41 K and 0.12 K, with offsets of −0.08 K and −0.05 K. So against OF_KIT, the wall's mean
  offset grows from −0.08 to −0.27 K: it runs cooler than in 5.4c.

**Figure 8, reported: the pressure on the axis.** The runs averaged velocity and temperature, not the pressure.
- *How it was recovered.* The momentum stage returns the pressure and the thermal step drops it. So the pressure is
  the stage's for one more step from each run's last state (`case1_reported.py`, beside the record).
  - That step moved the velocity by at most 5e-14 m/s, against a peak of 0.086 m/s.
  - At a steady state the stage's pressure is the steady pressure, `G p / ρ = R − λ u`, whatever the step.
- *How it is compared.* The paper removes some codes' pressure offsets (p. 14). So `case1_reported` adds the one
  offset that best fits the band.
- *The offset.* It is small against the 790 Pa drop: +2.3, +1.4 and +0.9 Pa on M1, M2 and M3, and −0.2 and
  −1.2 Pa with the outlet at 2.25 m.
- *Against the band.* After the offset, every point on every run lies within the band widened by the digitizing
  uncertainty, and 52–54 % lie inside the band as drawn.
- *The drop.* Over the band's span (x = −1.49 to 1.40 m) the drop is 789.3, 791.0 and 791.8 Pa on M1, M2 and M3,
  against the band's 789.9 Pa: within 0.25 %. The outlet's length and the time step leave it unchanged.
- The core's slope is calibrated, since the wall's ratio is fitted to the printed gradient. So the figure is
  reported, not gated.

**Figure 11, reported: OF_KIT's temperatures across the duct at five stations.**

| Station | Largest difference, M1 / M2 / M3 | Where (M3) | Mean offset, M3 | Within the digitizing 0.156 K, M3 |
|:--|--:|:--|--:|--:|
| x = 0 | 0.25 / 0.21 / 0.19 K | the heated interface, z = −23.0 mm | −0.04 K | 99 % |
| x = 0.15 m | 0.24 / 0.19 / 0.17 K | z = −22.3 mm | −0.04 K | 99 % |
| x = 0.3 m | 0.34 / 0.25 / 0.20 K | the heated interface | −0.03 K | 98 % |
| x = 0.6 m | 0.07 / 0.06 / 0.06 K | z = +6.9 mm | 0.00 K | 100 % |
| x = 1 m | 0.28 / 0.29 / 0.30 K | z = −10.2 mm | +0.05 K | 25 % |

- The largest differences sit at the heated interface and shrink with the mesh.
- At x = 1 m, the magnet's exit, LMhdX stays within 0.30 K but lies mostly just outside the digitizing uncertainty.
- *Excluded at x = 0.15 m:* two adjacent digitized points at z = −13.4 mm.
  - OF_KIT's trace there jumps 2–4 K onto the x = 0.3 and 0.6 m curves, which cross it at 315.8 K, and comes back
    within 0.4 mm.
  - LMhdX's profile is smooth through them, the same on all three meshes. With them, the largest difference is
    4.0 K and the mean −0.10 K.

**The hypothesis.**
- *The observables lie in the widened band on every run:* not met. Five figures miss, as above.
- *Buoyancy and the fringe move the heated zone's temperatures from 5.4c's held flow by a few tenths of a kelvin:*
  met. On 5.4c's mesh (M1L1):
  - T_w moves by at most 0.46 K, with a mean of −0.25 K.
  - T(z) at x = 0 moves by at most 0.28 K.
  - The bulk temperature, which the energy balance fixes, moves by 3e-4 K. The outlet rise is 10.124 K against
    10.123 K.
  - Nu at x = 0.29 m is 2.39 against 2.33. Figure 18's band there is 2.23–2.41.
  - The heated wall runs cooler, consistent with the faster jets of the fringe's development (above).

**The record.** `mistrangelo2025-case1-gated.json` holds the following:
- each run's arguments and metadata;
- its diagnostics every simulated second;
- its observables, with the four sub-windows;
- its reported figures (8, 11 and 38), with the pressure step's velocity change;
- the gate.

Its SHA-256 is `703e31e7385a7c7cb1c3976e2128ca9c158d2336ad74a836bad275909296968c`. It is on the
[research-assets release](https://github.com/uwplasma/LMhdX/releases/tag/lmhdx-research-assets-v2) (D38), with
`case1_reported.py`. Each run's output directory is given under `$SCRATCH`. The averaged fields (2.5 GB) and the
recovered pressures are kept outside the repository.

`tests/test_steady.py` checks the functions that made these results in about 20 s. On 16 cells across, 2 per
wall and 0.05 m along the core:
- a mesh without faces at the heated ends is refused;
- three steps take 1200 W in and pass u0 out, with the ledger closed and no return flow;
- every observable of a known flow, the inlet's profile all along with T and p linear, is as defined.

Without a solve, on curves made from the references:
- the rule holds every figure at the band's middle;
- it leaves unresolved a point the M1 runs move, however M2 and M3 agree;
- it fails a converged miss on every run, and takes U from the statistics when they set it;
- it gates only the points every run covers, and measures U in decades on a log axis;
- Figure 8's offset takes out a constant;
- Figures 11 and 38 report a constant difference as it is.

## Step 5.4e: the heater's edges, reported — the edge study (declared before the run)

The gate leaves Figures 9 and 18 too uncertain to judge at the heater's leading edge (above). The heat flux steps
there, and the campaign's 4–5 mm axial cells interpolate across the step. This study refines the axial mesh at both
ends of the heated zone and reports what it finds. The gate's verdict stands as declared (D12).

**The runs.** M2's and M3's cross-sections, with `case1(edge_spacing=0.001)`: cells of 1 mm either side of both
ends of the heated zone, each 1.1 times the one before up to the core's spacing (`edge_faces`; 31 and 24 more axial
cells). As M2L1 and M3L1, the outlet is at 1.5 m, and each run goes 300 s from the inlet's developed profile,
averaged over the last 100 s in four sub-windows. `dt` is 0.005 s on both, because the 1 mm cells carry the
side-layer jets at 2.8 u₀ and the advective Courant number must stay under 0.5.

**Reported.**
- Figure 9 at x = −0.3 m: the three points in the heated wall that were too uncertain to gate, on both runs beside
  M1L1 to M3L1, with their excess beyond the band and the change between the two runs.
- Figure 18 at the leading edge: Nu at the band's first two points, 3.5 and 4.4 mm in, against the one code there,
  and the curve against the fit Nu ≈ 34 d^−0.37 found above.
- Every other gated figure, for information: its change from M2L1 and M3L1, against its U.
- Figure 8 from the averaged pressure the thermal state now carries, against the first campaign's recovered one.

**Hypothesis.**
- Figure 9's three points converge into the band. M3's excess there was already zero, and the edge cells remove the
  interpolation across the step.
- Figure 18's first point resolves near 21.5, about 11 % above the one code there, and the second near 19.7, within
  1 % of the band. The first would then be a resolved miss, not an unresolved one.
- Elsewhere the figures move by less than their U.

**Budget.** From 5.4e's measured cost per step, with 4–6 % more cells:
- the M2 cross-section: 60,000 steps at about 0.1 s, 1.7 hours;
- the M3 cross-section: 60,000 steps at about 0.21 s, 3.5 hours;
- about 5 GPU hours in all.

**Exit.** Reported, not gated, beside the gate's results. Any change to the rule would be a new declaration.

## Step 5.4e: the heater's edges, reported — the results

The two runs ran as declared, on NERSC Perlmutter: one A100 each, float64, NVIDIA's JAX 26.10 container (image
`20cf0aa0e1`), commit `6863c42`. The comparisons are commit `37841a8`'s; the tag `mistrangelo2025-case2-runs` keeps
both. They are reported, not gated: the gate's verdict above stands.

| Run | Cells | Axial cells | `dt` (s) | Simulated | s a step | Wall time | Courant max | Ledger max |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| M2's cross-section, edges refined | 3.76 M | 587 | 0.005 | 300 s | 0.076 | 77 min | 0.431 | 8.6e-12 |
| M3's cross-section, edges refined | 7.42 M | 686 | 0.005 | 300 s | 0.161 | 162 min | 0.432 | 7.2e-12 |

- They took 4.0 GPU hours, against the 5 budgeted.
- Both stayed finite. Mass stayed under 2e-10, the charge residual at the solver's stop as on M2L1 and M3L1 (0.015
  and 0.025), and no flow returned through the outlet.
- Both were steady over the window: storage within 1e-11 of the 1200 W heat input.

**Figure 9: the hypothesis is met.**
- The gate's three unresolved points, at z = −24.77, −24.69 and −24.61 mm in the heated wall, lie inside the widened
  band on both runs.
  - On M2's cross-section they are 302.92, 302.84 and 302.75 °C. M2L1 had 303.01, 302.92 and 302.83 °C, and M3L1
    302.98, 302.90 and 302.81 °C.
  - The two runs differ there by 0.004 K, against the gate's U of 0.037 K. The points have converged.
- One point lies further out, at the wall's outer face (z = −24.85 mm). It is outside the gate's common points,
  because M1L1 does not reach it. The two runs lie 0.015 and 0.019 K beyond the widened band there, against 0.10
  and 0.07 K on M2L1 and M3L1.
- So Figure 9's misses were the interpolation across the step in the heat flux, as found above. With 1 mm cells
  either side of the step, Figure 9 lies in the band at every point the gate judged.

**Figure 18: the first point is a resolved miss; the numbers sit above the hypothesis.**

| Band point | The one code / band | M1L1 / M2L1 / M3L1 | Edges, M2 / M3 | Beyond the widened band, edges |
|:--|--:|--:|--:|--:|
| 3.48 mm from the edge | 19.43 | 22.70 / 23.10 / 23.11 | 22.12 / 22.06 | 2.55 / 2.49 |
| 4.41 mm | 18.72–19.48 | 21.19 / 21.04 / 20.47 | 19.96 / 19.91 | 0.22 / 0.17 |

- At the first point the two runs differ by 0.06, inside the digitizing half-width of 0.09. So the point is
  resolved, and it misses: 13 % above the one code. The hypothesis put it near 21.5; it lies 3 % above that.
- At the second point the two runs differ by 0.05. They lie 2 % above the band as drawn and 0.17–0.22 beyond the
  widened band. The hypothesis put it at 19.7, within 1 % of the band.
- On both runs the 12 cells within 20 mm of the edge follow Nu = 33.9 d^−0.373, with d in mm, to within −6 % and
  +5 %. That is the curve found on 5.4e's meshes, Nu ≈ 34 d^−0.37. The first cell, 0.5 mm in, has Nu 41.
- The finest of the codes' axial cells is 1.5 mm (0.065 a, Table 3).

**Every other gated figure: the hypothesis is met away from the leading edge.** Each figure's change from M2L1 to
M2's refined cross-section, and from M3L1 to M3's, at the gate's points, against U there:

| Figure | Largest change, M2 / M3 | Where | Points beyond U, M2 / M3 | Change over the widened half-width |
|:--|--:|:--|--:|--:|
| 6 | 2.3e-4 / 1.4e-4 K | x = −0.15 and −0.20 m | 2 / 2 | under 0.01 |
| 7 | 4.7e-6 / 2.1e-6 m/s | x = −0.72 m | 1 / 0 | under 0.01 |
| 9 | 0.12 / 0.086 K | z = −22.7 mm, x = −0.3 m | 63 / 50 | 1.1 / 0.8 |
| 10 | 1.9e-4 / 6.2e-5 K | the heated wall's outer face | 0 / 0 | under 0.01 |
| 12 | 7.1e-6 / 5.0e-6 m/s | the −z jet | 0 / 0 | 0.01 |
| 13 | 5.8e-8 / 5.6e-8 m/s | the jets | 0 / 0 | under 0.01 |
| 15 | 0.022 / 0.010 µV | x = −0.63 m | 1 / 0 | under 0.01 |
| 16 | 1.0e-4 / 3.5e-5 µV | z = −21 mm | 0 / 0 | under 0.01 |
| 17 | 0.009 / 0.006 K | x = −0.2997 m, the leading edge | 7 / 4 | 0.10 / 0.07 |
| 18 | 1.09 / 1.05 | the leading edge | 7 / 2 | 2.3 / 11 |
| 19 | 0.005 / 0.005 decades | x = −0.31 m | 2 / 1 | 0.13 / 0.11 |

- The refinement moves Figures 9, 17 and 18 by more than U at the heater's leading edge, where it acts.
  - Figure 9 is the profile across the duct at x = −0.3 m. Much of it moves, 63 of its 155 points by more than U,
    and it stays in the band except at the wall's outer face, as above.
- Four figures move by more than U at one or two points each: 6, 7, 15 and 19. Their U there is far below the
  band's half-width, and their largest changes stay under 0.13 of it.

**Figure 8: the averaged pressure.**
- The pressure the thermal state now carries, averaged over the window, agrees with 5.4e's recovered one. It differs
  by at most 0.008 Pa on M2's cross-section and 0.003 Pa on M3's, out of a 791 Pa drop.
- The best offsets are 1.36 and 0.85 Pa, as before. Every point lies within the widened band.
- So 5.4e's one-step recovery agrees with the averaged pressure, which now replaces it.

**Figures 11 and 38, for information.**
- At x = 0.3 m, Figure 11's largest difference from OF_KIT falls from 0.25 and 0.20 K (M2L1, M3L1) to 0.15 and
  0.12 K.
- Figure 38 is unchanged: T_w within 0.44 K, T_b within 0.80 K at the leading edge.

**What this changes.**
- The gate is unchanged (D12).
- Figure 9's three unresolved points were numerical, as found above, and converge into the band.
- Figure 18's first point is now a resolved miss, 13 % above the one code. Its second point lies 2 % above the band.

**The record.** `mistrangelo2025-case1-edges.json` holds the following:
- each run's arguments and metadata;
- its diagnostics every simulated second;
- its observables, with the four sub-windows;
- its reported figures (8, 11 and 38);
- the comparisons above, against step 5.4e's released record (`mistrangelo2025-case1-gated.json`, SHA-256
  `703e31e7…`).

Its SHA-256 is `b264e0a44d6dc9b2b2838fa51867d706fce6288972c9094f498ccb2c8a925792`. It goes to the research-assets
release with Case 2's record below (D38).

## Step 5.4f: Case 2, reported — the campaign (draft, declared before the run)

Plan step 5.4f reports Case 2 against the codes' spread (ADR 0009 D35). It has no pass or fail: the references differ
in kind, CFX predicting a steady flow (p. 12). This record fixes the runs, the burn-in, the averaging window and the
statistical uncertainty before any production run (D12). It is a draft until agreed.

**The problem.** `validation.mistrangelo_duct.case2` is `case1`'s duct with Case 2's inferred u₀ = 0.01 m/s and
q'' = 10 W/cm² (Ha 235, Re 1073, Gr 3.55e7, Ri/N 0.60). The energy balance gives a bulk rise of 75.9 K. The paper
finds rolls about the field's axis in and behind the heated zone, and temperature fluctuations of up to 30 K behind
the heater's end (pp. 12–15).

**Checked so far** on a coarse mesh: 16 cells across, 5 cm along, `dt` 0.05 s, 900 s in 9 minutes on a laptop CPU.
- The heat ledger closes to 1.5e-12 of its largest term, and mass to 9e-10.
- Flow returns through the outlet from 129 s to 511 s, up to 40 % of the throughflow, while the heated fluid first
  reaches it. None returns after 511 s.
- The heat stored falls under 1 % of the 3000 W input only at about 560 s, nearly two flow-throughs.
- The flow then settles to a steady state: the probes' temperatures hold to 0.01 K. A mesh this coarse damps the
  rolls. CFX, on axial cells of 0.22 a as M1's, also found a steady flow; FLUENT_UK, on 0.217 a, did not. So each
  run below must show its own time dependence.

**Hypothesis.**
- LMhdX's flow is time-dependent in and behind the heated zone, as every code but CFX finds: the temperature there
  fluctuates by kelvins over the averaging window.
- Its time averages lie within the codes' widened band at most points of the figures on which the codes agree, and
  the bulk temperature's outlet plateau is the energy balance's 375.9 °C.

**Compared and reported.**
- Every Case 2 figure against its band widened by the digitizing uncertainty, with `band_misfit` as in 5.4e:
  Figures 23 and 24 (u and T on the axis), 26–29 (u(z) and T(z) at x = −0.3 and 0 m), 31–33 (the potential on both
  interfaces and across), 34 (the bulk temperature), 35 (E_t) and 37 (Nu). For each: the share of points inside the
  band as drawn and within the widened band, the largest excess and where, and the statistical uncertainty there.
- Figure 30, the pressure on the axis, with one best-fit offset as Figure 8. It is the time average of the
  pressure the thermal state carries, the momentum stage's, as `core3d.step` returns it
  ([option 1 on #327](https://github.com/uwplasma/LMhdX/pull/327#issuecomment-6929038011)).
- Figure 38's Case 2 curves, OF_KIT's T_w and T_b: the largest difference and the mean offset, as in 5.4e.
- The fluctuations: the RMS of T and u over the averaging window on the plane y = 0 and on the heated interface;
  time series every 0.1 s at probes on the heated interface and on the axis at x = 0, 0.15, 0.33 and 0.6 m; and the
  largest temperature swing, against the paper's 30 K.

**Runs.** 5.4e's meshes, with `dt/Δx` at 3 s/m. The codes' axial cells in the heated zone are 0.065 a to 0.71 a
(Table 3); M1's and M2's are 0.22 a and 0.17 a.
- M2 (64 across, 4 mm along), outlet at 1.5 m: the main run.
- M1 (48 across, 5 mm along), outlet at 1.5 m: the mesh's effect, reported beside M2.
- M1 with the outlet at 2.25 m: Case 2's rolls reach further downstream than anything in Case 1, and COMSOL's
  outlet let flow back in (p. 12).

**Burn-in and averaging**, the protocol stated under Case 2 above, made exact.
- Burn-in: at least one flow-through, L/u₀ = 300 s for the 3 m duct. From then on the bulk temperature T_b(x) and
  the heat stored are averaged over consecutive 50 s windows. Averaging starts at the end of the first window whose
  T_b agrees with the one before within 1 % of the bulk rise (0.76 K) at every station, and whose stored heat is
  under 1 % of the input (30 W). A run is advanced in compiled chunks of whole steps, about 1 s, so each window is
  the nearest whole number of chunks (`validation.mistrangelo_duct.case2_settled`).
- The second condition is added because of the coarse run: its probes' 50 s means agreed within 0.76 K by 400 s,
  while 10 % of the input was still being stored.
- Averaging: 450 s, the longest of the codes' windows (OF_KIT's, Table 5), at every step, kept as ten 45 s
  sub-batches.
- Statistical uncertainty: the standard error of the five 90 s batch means (pairs of sub-batches) at each point.
  The batches count as independent if the ten sub-batches give a standard error at least 0.8 times the five
  batches' at the median point of every figure: the median, over the figure's band points, of the ratio of the
  two errors. Points where the batches do not differ, to rounding, are left out (`case2_independence`). If not,
  averaging goes on in 90 s batches up to 900 s; an error that still fails the check is quoted as a lower bound.
- The corrected momentum step that the Case 2 section asks for is 5.4d's (#314, #315).

**Diagnostics expected every step.** As in 5.4e: the heat ledger within 1e-9 of its largest term, mass and charge at
their solve floors, the throughflow held, and the advective Courant number at most 0.5. Flow returning through the
outlet is flagged (`outlet_backflow="error"`), not hidden. It may return during the burn-in, as on the coarse mesh;
none may return during the averaging window.

**Budget.** From 5.4e's measured cost per step on these meshes, to be measured again for Case 2 before the runs.
About 1,100 s simulated: up to 650 s of burn-in and settling, as the coarse run needed, then 450 s of averaging.
- M1 at 0.015 s: 73,000 steps, about 50 minutes.
- M2 at 0.012 s: 92,000 steps, about 2.4 hours.
- M1 with the long outlet: about 55 minutes.
- About 4 GPU hours in all, and 7 if the averaging runs to 900 s.

**Exit (D12).**
- Case 2 is reported, not gated, with every comparison and its statistical uncertainty shown.
- A run whose bulk temperature does not settle by the criterion within 1,200 s is reported with its drift.
- A run with flow returning through its outlet at 1.5 m during the averaging window is reported on the 2.25 m outlet
  (D37's fallback).
- If M2 settles to a steady flow, as the coarse mesh did, it is run again with half its axial spacing (2 mm,
  0.087 a, nearer OF_KIT's 0.065 a), about 10 GPU hours more, before Case 2 is reported steady on LMhdX.

## Step 5.4f: Case 2, reported — the results

The three runs ran as drafted, on NERSC Perlmutter: one A100 each, float64, NVIDIA's JAX 26.10 container (image
`20cf0aa0e1`), commit `6863c42`. The comparisons are commit `37841a8`'s; the tag `mistrangelo2025-case2-runs` keeps
both. The protocol was applied as written. Case 2 is reported, not gated.

**The runs.**

| Run | Cells | `dt` (s) | Settled at | Averaged | Simulated | s a step | Wall time | Courant max | Ledger max |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| M2, outlet at 1.5 m (main) | 3.56 M | 0.012 | 649 s | 450 s | 1,098 s | 0.080 | 125 min | 0.39 | 4.7e-12 |
| M1, outlet at 1.5 m | 1.62 M | 0.015 | 652 s | 450 s | 1,104 s | 0.035 | 45 min | 0.31 | 2.9e-12 |
| M1, outlet at 2.25 m | 1.68 M | 0.015 | 903 s | 900 s | 1,808 s | 0.036 | 74 min | 0.31 | 2.9e-12 |

- The three took 4.1 GPU hours, within the 4 to 7 budgeted.
- The M2 and M1 runs at 1.5 m each began as a 20 s timing job and went on from its checkpoint. A run resumed so is
  the same, bit for bit, as one run straight through; that was checked on a coarse mesh before the runs.

**Diagnostics against those expected.**
- Every run stayed finite. The heat ledger stayed within 5e-12 of its largest term, mass under 1.7e-9, and the
  charge residual at the solver's stop (0.002–0.004). The Courant number peaked at 0.39.
- Flow returned through the outlet during the burn-in, as on the coarse mesh, while the heated fluid first reached
  it: from 120 s to 526 s on M2, up to 49 % of the throughflow. On M1 it lasted to 541 s, and with the long outlet
  from 178 s to 737 s, both up to about 50 %.
- None returned during any averaging window. So no run falls back to the long outlet (D37).
- Settling:
  - On M2, T_b's change from one 50 s window to the next fell 3.9, 2.6, 1.9, 1.4 and 1.2 K, then 0.59 K. The heat
    stored fell from 307 W to 1.2 W over the same windows.
  - M1 settled alike, at 652 s.
  - The long outlet took twelve windows, to 903 s, with more fluid to heat: 1027 W stored over its first window.
  - At the end of every run the heat stored was within 0.12 W of zero, 4e-5 of the 3000 W input.

**Independence of the sub-batches.**
- On M2 and M1 the ten sub-batches pass at 450 s:
  - On M2, every figure's ratio is 0.93 or more. The lowest is Figure 37's, against the 0.8 required.
  - On M1, every ratio is 1.16 or more.
- With the long outlet they fail through 900 s. Figures 23, 24, 30, 34 and 37 give 0.74–0.77, so that run's errors
  are quoted as lower bounds, as declared.
  - *Why:* its 45 s means drift together on a cycle of about 300 s, by ±0.1 K in T_b at x = 0.29 m, a
    lag-1 autocorrelation of 0.6.
  - That is 0.1 % of the bulk rise. It changes no comparison below.
- Figures 26–29, at x = −0.3 m and 0, do not differ between batches to rounding: the flow there is steady. They
  have no ratio, and their error is zero.
- The potential's three figures share one ratio on each run: 2.0, 11.2 and 1.8. The potential is set to zero mean
  over the whole duct, so the unsteady flow downstream shifts it by the same amount everywhere. A ratio above 1
  means the quoted error is, if anything, too large.

**The flow: time-dependent behind the heater, steady over its first half.** The first hypothesis is met behind the
heated zone, not in it.
- The flow is steady to rounding at x = 0: the probes there move by 1e-8 K. At x = 0.15 m they move by under 1e-3 K.
- Fluctuations grow along the heated zone from the rounding level, about fiftyfold every 5 cm, alike on M1 and with
  the long outlet. They saturate at the heater's end, at about half of u₀. The RMS of u on the plane y = 0, on M2:

| x | 0.10 m | 0.15 m | 0.20 m | 0.25 m | 0.30 m |
|:--|--:|--:|--:|--:|--:|
| RMS of u | 1.6e-9 m/s | 6.4e-8 m/s | 3.7e-6 m/s | 1.5e-4 m/s | 4.7e-3 m/s |

- Behind the heater, at x = 0.33 m on the axis:
  - T swings by 27.1 K on M2, 30.2 K on M1 and 22.4 K with the long outlet. The paper finds up to 30 K there.
  - The RMS is 9.4 K. The period is 3.8 s on M2, and 4.5–4.6 s on M1.
- The largest RMS of T is 13.0 K on the plane y = 0 (x = 0.31 m, z = −8 mm), and 1.9 K on the heated interface
  (x = 0.34 m).
- At x = 0.6 m the swing is down to 0.7 K, over slower periods of 26 s and more.

**Against the codes' spread.** M2's run, with M1's and the long outlet's shares within the widened band in brackets.
The error is the standard error of the batch means at the largest excess.

| Figure | Quantity | Points | Inside the band as drawn | Within the widened band | Largest excess, where | Error there |
|:--|:--|--:|--:|--:|:--|--:|
| 23 | u on the axis | 175 | 60 % | 95 % (95, 95) | 0.72 mm/s, x = 0.33 m | 0.05 mm/s |
| 24 | T on the axis | 119 | 31 % | 67 % (71, 71) | 5.8 K, x = 0.26 m | 0.002 K |
| 26 | u(z), x = −0.3 m | 190 | 45 % | 97 % (95, 95) | 0.42 mm/s, the +z jet | steady |
| 27 | u(z), x = 0 | 206 | 69 % | 100 % (100, 100) | — | steady |
| 28 | T(z), x = −0.3 m | 80 | 4 % | 86 % (65, 65) | 0.057 K, z = −20.9 mm | steady |
| 29 | T(z), x = 0 | 175 | 95 % | 100 % (100, 100) | — | steady |
| 31 | φ on z = −a | 122 | 6 % | 74 % (73, 83) | 1.9 µV, x = 0.31 m | 0.001 µV |
| 32 | φ on z = +a | 126 | 15 % | 81 % (81, 92) | 1.4 µV, x = 0.32 m | 0.004 µV |
| 33 | φ(z), x = 0 | 155 | 11 % | 100 % (100, 100) | — | 0.0005 µV |
| 34 | bulk T | 191 | 30 % | 98 % (97, 97) | 1.8 K, x = 0.31 m | 0.025 K |
| 35 | E_t | 426 | 43 % | 79 % (81, 81) | 8.9 decades, x = −1.20 m | 1e-6 decades |
| 37 | Nu | 212 | 29 % | 62 % (63, 63) | 2.1, x = 0.30 m | 0.001 |

- The statistical error is far below every excess. No miss is a matter of averaging.
- M1 and M2 agree: no figure's share within the widened band changes by more than 4 points between them, except
  Figure 28's, which rises from 65 % to 86 % on M2.

**Figure 30, the pressure on the axis.**
- After the one best-fit offset (−0.70 Pa on M2; −0.40 and −0.85 Pa on M1), every point lies within the widened
  band, and 86–88 % inside the band as drawn.
- The drop over the band's span is 258.8 Pa on M2 and 258.3 Pa on M1, against the band's 259.3 Pa: within 0.4 %.
- It is the time average of the pressure the thermal state carries (option 1 on #327).

**Figure 38, OF_KIT's Case 2 curves.**
- T_b lies within 1.5 K of OF_KIT for |x| < 0.28 m, with a mean offset of −0.17 K.
- T_w runs up to 24.7 K above OF_KIT there, and 27.7 K at x = 0.30 m, with a mean offset of +5.7 K. It agrees
  within 1 K up to x ≈ 0, then rises above:

| x | 0 | 0.10 m | 0.20 m | 0.26 m | 0.29 m |
|:--|--:|--:|--:|--:|--:|
| T_w, OF_KIT | 386.3 °C | 397.0 °C | 394.0 °C | 400.1 °C | 402.1 °C |
| T_w, LMhdX on M2 | 385.4 °C | 401.2 °C | 415.3 °C | 424.8 °C | 428.6 °C |

**The hypothesis on the bands.**
- *Within the band at most points of the figures on which the codes agree:* met. Every figure has most of its points
  within the widened band:
  - 95–100 % on six: 26, 27, 29, 33, 34, and 30 after its offset;
  - 74–95 % on five: 23, 28, 31, 32 and 35;
  - about two thirds on Figures 24 and 37.
- The misses gather in the heated zone's second half, below.
- *The bulk temperature's outlet plateau is the energy balance's 375.9 °C:* met. It is 375.93 °C from x = 0.5 m on,
  on every run.

**Where LMhdX leaves the band.**
- **The second half of the heated zone, x ≈ 0.09–0.36 m: Figures 24, 37 and 38's T_w.**
  - Nu stays at 2.1–2.2 from x = 0.1 m to the heater's end. The band rises from 2.5–2.8 at x = 0.1 m to 3.6–5.1 at
    0.2 m: LMhdX's Nu is about half the codes' there.
  - So the wall runs hot (T_w, above), and the axis runs cold: up to 8 K below the band as drawn, and 5.8 K beyond
    the widened band, at x = 0.25 m (353.6 °C against 361.6–363.0 °C). Past the heater's end it runs
    2.7 K beyond the band's top, at 0.34 m.
  - The bulk temperature, which the energy balance fixes, stays in the band: the same heat, carried less from the
    wall into the core.
  - *Why:* in the codes the rolls mix the heated fluid into the core from about x = 0.05 m, where their Nu starts to
    rise. In LMhdX the flow there is steady, and the rolls grow from the rounding level and saturate only at the
    heater's end.
  - What seeds the codes' rolls is not given. At the growth measured above, each tenfold larger seed moves the
    saturation about 3 cm upstream. A seed six or seven decades above the rounding level would bring it to about
    x = 0.1 m, where the codes' Nu has risen.
  - The seeded study below, declared before its runs, checks this. A seed of 1e-4 u₀ just before the heater moves
    the saturation to x = 0.07 m on M1 and brings Nu, the axis temperature and T_w to the codes'.
  - Over x = 0.09–0.28 m, M1 and M2 agree to within 0.13 K in T on the axis and 0.02 in Nu. So the miss there is
    not the axial resolution between 5 and 4 mm. At the heater's end, where the rolls saturate, they differ by up to
    3.5 K.
- **The potential, Figures 31 and 32.**
  - On both interfaces LMhdX lies below the band, by up to 2 µV beyond the widened band, out of 117 µV. Across the
    duct at x = 0, Figure 33, it lies within the widened band but mostly below the band as drawn.
  - On M2, a uniform shift of +2 µV puts every point of all three figures within the widened band, and 93–100 %
    inside the band as drawn.
  - So the miss is in the potential's constant, not its shape. LMhdX sets the potential's mean over the whole duct
    to zero. So the flow downstream, which differs from the codes', can shift it everywhere.
- **At the heater's leading edge, x = −0.3 m, Figure 28:** 0.057 K on M2 and 0.17 K on M1. The misses are at the
  heated wall's outer face and in the fluid 2–5 mm above the heated interface. The band there holds two codes only.
  The miss falls with the mesh, as Case 1's Figure 9 did at the same edge.
- **The jets, Figure 26:** 0.42 mm/s at the +z jet. That is the jets' resolution, as in Case 1's Figure 12.
- **u on the axis, Figure 23:** 0.09 mm/s below the band at x = 0.06 m, and 0.72 mm/s above it at 0.33 m, in the
  rolls.
- **E_t, Figure 35:**
  - *Upstream of x = −1.0 m:* up to 8.9 decades below the band. LMhdX's inflow is the inlet section's developed
    profile, as in Case 1's Figure 19. A disturbance of 1e-2 u₀ at the inlet brings it within 0.3 decades (the
    seeded study below).
  - *At the heater's end (x = 0.29–0.35 m):* up to 0.74 decades above, where LMhdX's rolls saturate.
  - *Downstream of x = 0.8 m, where the field falls off:* up to 1.3 decades above, alike with the long outlet. The
    seeded study below finds that it follows the rolls' onset too.
- **Bulk temperature at the heater's end, Figure 34:** 1.8 K above the band at x = 0.31 m on M2 (3.0 K on M1),
  where the rolls saturate. Then it returns to the plateau.

**The record.** `mistrangelo2025-case2-reported.json` holds the following:
- each run's arguments and metadata;
- its diagnostics every simulated second;
- its settling windows and sub-batches;
- its observables, the window's mean and every sub-batch;
- the comparison (`case2_reported`) and the independence check;
- its fluctuations: the probes' swings, RMS and periods, and the RMS planes' maxima.

`mistrangelo2025-case2-fluctuations.npz` holds each run's probes at the step nearest every 0.1 s, from the start.
It also holds the RMS over the averaging window on the plane y = 0 (T, u and w) and on the heated interface (T),
with their coordinates.

| File | SHA-256 |
|:--|:--|
| `mistrangelo2025-case2-reported.json` | `1795cb16da52cb2b97318716d7e64717940bdc80186bfa3b94b46b86b007f251` |
| `mistrangelo2025-case2-fluctuations.npz` | `662070017f4181eb09d8a5394e13498a3c2ec1ac87038736322a5c8364d0b1c6` |
| `mistrangelo2025-case1-edges.json` | `b264e0a44d6dc9b2b2838fa51867d706fce6288972c9094f498ccb2c8a925792` |
| `mistrangelo2025-case2-seeded.json` | `28b73d09fd10a24a7b188ae2747547548e870916f8803673c0d8715a96512a23` |
| `mistrangelo2025-case2-seeded-fluctuations.npz` | `7f8e916616f4a2a68c0e0fa0356ed3de4404fe2e6ae57a403d02246907345d60` |
| `mistrangelo2025-case2-seeded-calibration.json` | `4874fd87c5e8a3dbb5855b4f8c9f2b7d9e561752fbfd1382db4c6b08d0949574` |
| `case2_reported.py` | `cd7a53ea92d09c1accd2fdb2322f027666059e7aa7eb146017dc453ae1178548` |
| `mistrangelo2025-case2-runs.zip` | `d555bcfd531b1719e05171e71471dd80cb308929108d09f91f7e0273651e29c8` |

- They go to the research-assets release with this step's pull request (D38). The seeded study's three are below.
- `case2_reported.py` writes all six records from the runs' outputs and 5.4e's record. Written again, the first three
  differ from those above only in the commit they name.
- `mistrangelo2025-case2-runs.zip` holds the run harness:
  - the runs (`case2_run.py`, `case1_edge_run.py`), and the seeded study's calibration (`calibrate.py`);
  - the observables of the averaged fields on a GPU (`post.py`);
  - the batch and deploy scripts;
  - the coarse resume checks: `smoke.py`, and `smoke_seeded.py` for the seeded runs;
  - `main-runs/case2_run.py`, the script as the three main runs used it. Without a seed `case2_run.py` runs as it
    does, bit for bit on the coarse mesh (`smoke_seeded.py`).
- Each run's output directory is given under `$SCRATCH`. The averaged fields and sub-batches (5.9 GB, and 3.7 GB for
  the seeded study) are kept outside the repository.

`tests/test_steady.py` checks `case2_reported` without a solve, on curves made from the references:
- at the band's middle every figure lies inside, and Figure 30 after its offset;
- one point off is a figure's only failure and its largest excess;
- the error is the batch means' standard error, in decades on a log axis.

## Step 5.4f: the rolls' onset, seeded (declared before the run)

The results above leave one question. Over the heated zone's second half LMhdX's Nu is about half the codes', its
wall up to 25 K hotter and its axis up to 8 K cooler. Is that because its rolls start late, or because of something in
the flow itself? This study adds a small disturbance and reports what it changes. Case 2's results above stand as
declared (D12).

**What the codes show.** Each code's curves are in the extraction archive (above).
- *The axis temperature, Figure 24.*
  - At x = 0.05 m, OF_KIT, OF_ORNL and COMSOL lie within 1.6 K of LMhdX's 330.0 °C. CFX, whose rise is steeper
    throughout, lies 8 K above.
  - By x = 0.1 m all five codes plotted lie at 340–345 °C, against LMhdX's 335.9 °C. That includes CFX, whose flow
    is steady (p. 12).
  - So in the codes hot fluid reaches the axis between x = 0.05 and 0.1 m. In LMhdX it arrives only between 0.25
    and 0.35 m.
- *Nu, Figure 37.* Five codes' Nu rises from x ≈ 0.1–0.2 m. CFX's stays at 1.5–1.6, below LMhdX's 2.1–2.2, though
  its axis warms as the others' do.
- *LMhdX's fluctuations* grow from the rounding level and saturate at the heater's end (above). Over x = 0.12–0.24 m
  they grow 48-fold every 5 cm on M2 and 63-fold on M1, so a tenfold larger seed would move them about 3 cm.

**Two explanations, and how to tell them apart.**
- *The seed.* The rolls are a convective instability that amplifies whatever reaches it. LMhdX's flow carries only
  rounding, so they saturate late. The codes' solvers carry more, from their inflows or their stopping tolerances,
  so theirs saturate earlier. Then a seed moves the saturation upstream, about 3 cm for each tenfold larger seed at
  the measured growth.
- *The flow.* The rolls are the heater's end's own, a global mode, or the onset is set by the base flow. Then a seed
  that reaches the heated zone leaves the saturation where it is.

**The seed.** A random force on the vertical velocity in a slab, added through the thermal step's `force(t)`, with no
change to the library (`case2_run.py --seed`):
- uniform along the field;
- the three lowest sine modes across z, each with its own random amplitude;
- the amplitudes standard normal at knots every 2 s and linear between them;
- the slab's sin² window along x;
- scaled by ρ₀ times the seed's acceleration.

Two slabs:
- *at the inlet,* x = −1.45 to −1.25 m, in the field's fringe, where a code's inflow would carry its disturbances;
- *before the heater,* x = −0.5 to −0.35 m, in the uniform field.

**The levels.** 1e-4, 1e-3 and 1e-2 u₀, the RMS of the vertical velocity at the slab's downstream end. They are set by
a calibration on M1 from M1L1's last state (`calibrate.py`):
- each slab is seeded at 1e-4 m/s² for 60 s;
- the RMS of w on the rows either side of y = 0 is taken over the last 40 s, the largest across z;
- the amplitudes scale linearly from there.

**The runs.** Six, on M1 with the outlet at 1.5 m: two slabs at three levels each.
- Each starts from M1L1's last state, at t = 1104.5 s, and is seeded throughout.
- Each follows 5.4f's protocol, counted from its start: a 300 s burn-in, 50 s settling windows, then 450 s of
  averaging in ten sub-batches, checked for independence up to 900 s.
- The comparisons are 5.4f's (`case2_reported`).

**Reported.**
- *What the seed reaches.* The RMS of u and w on the plane y = 0 along x, and where the fluctuations saturate: the
  first x where the RMS of u on y = 0 reaches 1e-3 m/s, 0.1 u₀. Unseeded, that is x = 0.266 m on M1 and 0.274 m on M2.
- *The figures.* Nu (Figure 37), T_w (Figure 38) and the axis temperature (Figure 24) between x = 0 and 0.3 m. Every
  figure is compared as above, with the probes' swings and periods.

**Hypotheses.**
- *The seed explains the miss* if, before the heater, each tenfold larger seed moves the saturation upstream by 2–4 cm,
  and Nu and the axis temperature move toward the band with it.
- *The flow's own onset explains it* if a seed that reaches the heated zone leaves the saturation within 1 cm of the
  unseeded run's.
- *The inlet's seeds* arrive damped by the field. How much reaches the heater is measured. It shows whether a code's
  inflow could set its onset.

**Then.**
- If a seeded M1 run brings Nu at x = 0.2 m into the band, M2 repeats it at the same level, from M2L1's last state:
  about 1.5 GPU hours more.
- If none does, the study reports how far the largest seed moves the onset.
- *Added after the M1 runs, before M2's:* two M1 runs bring Nu at x = 0.2 m into the band, the seeds before the
  heater at 1e-4 and 1e-3 u₀. The rule above does not say which M2 repeats. It repeats the smaller, 1e-4 u₀:
  3.158e-5 m/s² from the same calibration on M2.

**Budget.**
- The calibration takes about 0.1 GPU hours.
- Each M1 run simulates 800–1,350 s at 0.035 s a step: 30–50 minutes.
- About 4 GPU hours in all, and 5.5 with M2.

**Exit.** Reported, not gated.
- Case 2's results above stay the record of LMhdX's unseeded flow.
- This study says what sets the rolls' onset.
- A seeded Case 2 protocol would be a new declaration (D12).

## Step 5.4f: the rolls' onset, seeded — the results

The study ran as declared, on NERSC Perlmutter: one A100 each, float64, the image and commit (`6863c42`) of the
main runs. One choice was added after the M1 runs and before M2's: which seed M2 repeats (above). The comparisons are
`case2_reported`'s, unchanged since `37841a8`; the record was written at `5ecf45c`. The study is reported, not gated.
Case 2's results above stay the record of LMhdX's unseeded flow.

**The runs.** On M1 each from M1L1's last state, at t = 1104.5 s, and on M2 from M2L1's, at 1097.6 s, with the outlet
at 1.5 m. The level is the RMS of w at the slab's downstream end; the M1 runs reached it to within 17 %.

| Run | Seed | Amplitude (m/s²) | Settled after | Averaged | Independence, lowest ratio | Wall time | Courant max |
|:--|:--|--:|--:|--:|--:|--:|--:|
| inlet-4 | 1e-4 u₀ at the inlet | 1.332e-5 | 602 s | 450 s | 0.93 | 40 min | 0.29 |
| inlet-3 | 1e-3 u₀ at the inlet | 1.332e-4 | 451 s | 630 s | 0.82 | 43 min | 0.28 |
| inlet-2 | 1e-2 u₀ at the inlet | 1.332e-3 | 401 s | 810 s | 0.80 | 48 min | 0.30 |
| heater-4 | 1e-4 u₀ before the heater | 3.194e-5 | 602 s | 450 s | 0.90 | 42 min | 0.32 |
| heater-3 | 1e-3 u₀ before the heater | 3.194e-4 | 401 s | 450 s | 0.88 | 33 min | 0.31 |
| heater-2 | 1e-2 u₀ before the heater | 3.194e-3 | not settled | — | — | 47 min | 0.31 |
| M2, heater-4 | 1e-4 u₀ before the heater | 3.158e-5 | not settled | — | — | 142 min | 0.37 |

- Every run stayed finite, with the heat ledger within 1.5e-12 of its largest term, and mass under 4e-10 on M1 and
  2.1e-9 on M2.
- Flow returning through the outlet stayed under 0.2 % of the throughflow, and under 0.002 % in every averaging
  window.
- The sub-batches are independent on every averaged run. inlet-3 and inlet-2 needed 630 s and 810 s of their 900.
- The calibrations took 6 minutes on M1 and 13 on M2. The six M1 runs took 4.2 GPU hours and M2's 2.4: 6.9 in all,
  against the 5.5 budgeted, M2's run having gone to the settling limit.
- **heater-2 and M2's run did not settle** (below). They have no averaged fields and so no comparison of figures.

**The settling rule with a seed.** A run settles when T_b's mean over a 50 s window changes by at most 0.759 K, 1 % of
the bulk rise, from the window before at every station, and the heat stored is under 30 W.
- With the rolls saturated over the heater's second half, the windows' means swing by about 1 K about a steady level.
  A seeded run settles only when one window happens to fall under the limit:
  - heater-4 on M1 at its sixth window (0.712 K, after 0.88–1.44 K), heater-3 at its second (0.695 K), the inlet's at
    their second to sixth (0.57–0.73 K, after 0.85–1.09 K);
  - heater-2 not in 18 windows (0.93–2.16 K), nor M2's run in 19 (0.97–2.10 K).
- The heat stored stayed within 14.1 W of zero on every seeded run, so none was drifting. heater-2 swings by 2.4 K
  RMS on the axis at x = 0 and 6.6 K at 0.15 m.
- The rule was declared for the unseeded flow and stands (D12). A seeded protocol would need its own.

**What the seed reaches.** The RMS of w on the plane y = 0, the largest across z, and where the rolls saturate: the
first x where the RMS of u there reaches 1e-3 m/s.

| Run | Slab's end | x = −0.3 m | x = 0 | x = 0.1 m | Saturated at |
|:--|--:|--:|--:|--:|--:|
| unseeded | — | 2.9e-12 m/s | 3.7e-12 m/s | 1.3e-9 m/s | 0.266 m |
| inlet-4 | 1.2e-6 m/s | 1.2e-11 m/s | 2.2e-10 m/s | 1.8e-7 m/s | 0.216 m |
| inlet-3 | 1.2e-5 m/s | 1.3e-10 m/s | 2.9e-9 m/s | 2.4e-6 m/s | 0.185 m |
| inlet-2 | 1.2e-4 m/s | 1.8e-9 m/s | 2.2e-8 m/s | 1.8e-5 m/s | 0.159 m |
| heater-4 | 1.1e-6 m/s | 2.1e-7 m/s | 1.7e-5 m/s | 5.0e-3 m/s | 0.068 m |
| heater-3 | 1.1e-5 m/s | 2.1e-6 m/s | 1.7e-4 m/s | 7.4e-3 m/s | 0.032 m |

- The field damps the inlet's seeds by five decades on their way to the heater's leading edge, x = −0.3 m.
- Over the heater's first half, to x = 0, every seed grows 12- to 82-fold. The unseeded run's rounding does not grow
  there.
- The smallest inlet seed arrives at x = −0.3 m only four times above the rounding, yet moves the saturation 5 cm.
  So what the seed carries matters, not only its size.

**The hypotheses.**
- *The seed explains the miss:* met.
  - Before the heater, the tenfold larger seed moved the saturation 3.6 cm upstream, from 0.068 to 0.032 m. The
    next decade's run did not settle.
  - At the inlet each tenfold larger seed moved it 3.1 cm and then 2.6 cm.
  - Nu and the axis temperature moved into the band with it (below).
- *The flow's own onset explains it:* not met. Every seed moved the saturation, by 5 to 23 cm.
- *The inlet's seeds* arrive five decades weaker. Even the largest, 1e-2 u₀, leaves Nu at x = 0.2 m at 2.34
  against the band's 3.55–5.13. So a disturbance at the inflow alone does not explain the codes' earlier onset.

**The heated zone's second half.** Nu, the axis temperature and T_w at four stations, on M1. Values within the band
widened by the digitizing uncertainty are in bold.

| x | Nu, unseeded | Nu, heater-4 | Nu, heater-3 | Nu, the band |
|:--|--:|--:|--:|:--|
| 0.10 m | 2.26 | 2.33 | **2.73** | 2.52–2.80 |
| 0.15 m | 2.20 | **3.00** | **3.78** | 3.25–4.19 |
| 0.20 m | 2.15 | **4.09** | **4.05** | 3.55–5.13 |
| 0.25 m | 2.12 | **3.96** | 3.63 | 3.87–4.39 |

| x | T on the axis, unseeded | heater-4 | heater-3 | The band |
|:--|--:|--:|--:|:--|
| 0.10 m | 335.9 °C | **338.9 °C** | **342.2 °C** | 340.4–345.3 °C |
| 0.15 m | 341.9 °C | **348.7 °C** | **349.0 °C** | 347.4–349.5 °C |
| 0.20 m | 347.9 °C | **356.0 °C** | **355.4 °C** | 355.4–358.7 °C |
| 0.25 m | 354.0 °C | **361.8 °C** | **361.3 °C** | 362.0–363.2 °C |

| x | 0.10 m | 0.15 m | 0.20 m | 0.25 m |
|:--|--:|--:|--:|--:|
| T_w, OF_KIT | 397.2 °C | 392.4 °C | 394.5 °C | 399.2 °C |
| T_w, unseeded | 400.8 °C | 408.4 °C | 415.8 °C | 423.0 °C |
| T_w, heater-4 | 400.1 °C | 397.0 °C | 391.8 °C | 398.6 °C |
| T_w, heater-3 | 394.6 °C | 388.0 °C | 391.7 °C | 401.4 °C |

**Every figure.** The share within the widened band, on M1:

| Figure | Unseeded | inlet-4 | inlet-3 | inlet-2 | heater-4 | heater-3 |
|:--|--:|--:|--:|--:|--:|--:|
| 23, u on the axis | 95 % | 93 % | 94 % | 93 % | 95 % | 99 % |
| 24, T on the axis | 71 % | 81 % | 82 % | 85 % | 100 % | 100 % |
| 31, φ on z = −a | 73 % | 72 % | 71 % | 69 % | 75 % | 75 % |
| 32, φ on z = +a | 81 % | 80 % | 79 % | 78 % | 80 % | 79 % |
| 33, φ(z), x = 0 | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| 34, bulk T | 97 % | 95 % | 96 % | 99 % | 100 % | 98 % |
| 35, E_t | 81 % | 81 % | 80 % | 79 % | 88 % | 88 % |
| 37, Nu | 63 % | 63 % | 64 % | 70 % | 84 % | 88 % |
| 30, p after its offset | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |

- *Figures 26–29, at x = −0.3 m and 0:* 95, 100, 65 and 100 % on every run, as unseeded.
- *Figure 38, T_w against OF_KIT:* the mean offset falls from +5.5 K unseeded to +0.3 K (heater-4) and −0.8 K
  (heater-3). The largest difference away from the heater's ends falls from 24.5 K to 6.3 K and 4.5 K. T_b stays
  within 2.2 K of OF_KIT with either. With the inlet's seeds it moves up to 3.4 K away, at x = 0.22–0.26 m, where
  their rolls saturate.
- *Figure 30:* the drop is 258.6 Pa with either seed before the heater, against the band's 259.3 Pa, and the offset
  falls from −0.40 Pa to −0.21 and −0.15 Pa.
- *E_t, Figure 35:*
  - Downstream of x = 0.8 m, the unseeded miss of up to 1.2 decades falls to 0.32 decades with heater-4. So this miss,
    not traced above, follows the rolls' onset too.
  - Upstream of x = −1.0 m only the inlet's seeds act: the miss falls from 8.8 decades to 6.3, 3.7 and 0.29. So the
    codes' E_t there is about what a disturbance of 1e-2 u₀ at the inlet gives LMhdX. That disturbance does not, on its
    own, bring Nu into the band.
- *Unchanged by any seed:* the potential's uniform offset (Figures 31 and 32), the heater's leading edge (Figure 28)
  and the jets (Figure 26). These lie upstream of the rolls or do not depend on them.

**What is left with a seed before the heater.**
- Nu at the heater's end, x = 0.28–0.36 m: up to 0.55 below the widened band with heater-4 and 0.74 with heater-3.
  Elsewhere along the heater it is within 0.12 (heater-4, at x = 0.1–0.2 m) and 0.34 (heater-3, at 0.2–0.28 m).
- The swings are larger than unseeded. On the axis at x = 0.33 m T swings by 41 K with heater-4 and 44 K with
  heater-3, against 30 K unseeded, and by 49–55 K at 0.15 m. The paper finds fluctuations of up to 30 K behind the
  heater; it does not say whether that is a swing or an amplitude. The period at 0.33 m is 12.6 s with heater-4 and
  7.0 s with heater-3, against 4.5 s unseeded.

**M2.** The repeat at 1e-4 u₀ before the heater did not settle, so M2 is compared at its probes: their means over its
last 450 s, each station's four together, with the standard error of ten batch means. M1's are over its averaging
window.

| Probe | M2, unseeded | M2, seeded | M1, seeded | M1, unseeded |
|:--|--:|--:|--:|--:|
| Heated interface, x = 0.15 m | 409.01 °C | 397.40 ± 0.57 °C | 397.96 ± 0.53 °C | 408.67 °C |
| Axis, x = 0.15 m | 342.09 °C | 348.96 ± 0.33 °C | 348.57 ± 0.28 °C | 341.81 °C |
| Heated interface, x = 0.33 m | 406.03 ± 0.06 °C | 388.40 ± 0.80 °C | 391.63 ± 1.05 °C | 403.99 ± 0.08 °C |
| Axis, x = 0.33 m | 376.67 ± 0.06 °C | 371.32 ± 0.35 °C | 370.61 ± 0.30 °C | 375.14 ± 0.03 °C |

- At x = 0.15 m the seed moves M2 as it moves M1: the heated interface cools by 11.6 K (10.7 K on M1) and the axis
  warms by 6.9 K (6.8 K). M1 and M2 agree there within one standard error.
- The rolls are saturated at x = 0.15 m on both: u on the axis has an RMS of 5.5 mm/s on M2 and 5.0 mm/s on M1.
- At the heater's end the interface differs by 3.2 K between the meshes, as it does unseeded (2.0 K).
- So M2 confirms M1 where the probes are. The figures, Nu among them, are M1's alone.
- heater-2's probe means lie within 1.2 K of heater-3's on the axis at x = 0.15 and 0.33 m. On the heated interface
  at 0.15 m it lies 4.5 K below them.

**What sets the onset.** The miss over the heated zone's second half is the rolls' late onset. It is set by the
disturbances that reach the heater, not by the flow: LMhdX's flow carries only rounding there, and a seed of 1e-4 u₀
just before the heater brings Nu, the axis temperature and T_w to the codes' on M1, and M2 moves as M1 does. The
paper does not give its codes' disturbances. The codes' E_t upstream is about what a disturbance of 1e-2 u₀ at
LMhdX's inlet gives, and that much does not bring Nu into the band. So in the codes the disturbance that sets the
onset likely arises nearer the heater; from what, this study cannot say. That is inferred, not checked. A seeded
Case 2 protocol, with a settling rule for the seeded flow, would be a new declaration (D12).

**The record.** `mistrangelo2025-case2-seeded.json` holds, for M1L1 and M2L1 unseeded and each seeded run:
- its arguments, metadata and seed;
- its diagnostics every simulated second, settling windows and sub-batches;
- its observables, comparison and independence check, as the main record's;
- its fluctuations and its onset: where the rolls saturate, the RMS of w along x, and Nu, T on the axis and T_w at
  x = 0.1, 0.15, 0.2 and 0.25 m against the band and OF_KIT.

Each run also has its probes' means over 450 s with their batch error. heater-2 and M2's run, which did not settle,
have instead their windows, diagnostics and probes over their last 450 s.
- `mistrangelo2025-case2-seeded-fluctuations.npz` holds the arrays of the main record's npz for each averaged run.
- `mistrangelo2025-case2-seeded-calibration.json` holds both calibrations.

## Step 5.4f: the Lorentz force explicit (declared before the run)

**Why.** The momentum stage takes the magnetic damping, rate λ = σB²/ρ, implicitly as the shift H = 1 + Δt λ, while
the potential's share of the force comes from the old velocity. Where the two balance, as in a roll whose axis lies
along the field, the motion evolves 1 + Δt λ times too slowly; a steady state is unaffected
([numerics](../physics/numerics.md), "What the shift costs a transient"). Against the Chebyshev growth rate, a roll
along the field at Pr 0.02 grows 17 % too slowly at Δt λ = 0.27, and at the field-free rate when the force is
explicit (#337, `ChannelProblem(magnetic_damping="explicit")`). Case 2's runs had Δt λ = 0.34 on M1 (Δt 0.015 s,
λ 22.4 s⁻¹ in the uniform field) and 0.27 on M2, and their rolls arrive later than the codes'. Case 1's gate compares
a steady flow and is unaffected.

**The run.**
- M1 with the outlet at 1.5 m and Δt 0.015 s, as M1L1, with `magnetic_damping="explicit"` (Δt λ = 0.34, under its
  bound of 2), and no seed.
- It starts from M1L1's last state, at t = 1104.5 s, as the seeded runs did, and follows 5.4f's protocol counted from
  there: a 300 s burn-in, 50 s settling windows, 450 s of averaging in ten sub-batches, extendable to 900 s.
- If it has not settled by 1,200 s, it averages from there anyway for 450 s and is reported as unsettled, with its
  windows' drift (`--average-unsettled`). heater-2 and M2's seeded run, which never settled, had nothing to compare.
- The comparisons are `case2_reported`'s and the seeded study's tables: where the rolls saturate, Nu, the axis
  temperature and T_w at x = 0.10–0.25 m, and every figure's share within the widened band, against M1L1 and the
  seeded runs.
- One A100 on Perlmutter, about 45 to 70 minutes, at commit `782e4f3` (the source of #337). The runner is 5.4f's
  `case2_run.py` with `--magnetic-damping` and `--average-unsettled` added.

**Hypotheses.** The explicit force lets the rolls grow at their own rate, so on M1 without a seed:
- the rolls saturate upstream of M1L1's 0.266 m, at or before 0.236 m (3 cm, one decade of the seeds);
- Nu at x = 0.2 m rises from M1L1's 2.15 into the band's 3.55–5.13.

If neither holds, the stage's slowness is not what delays M1's onset, and the seeded study's account stands alone.

**Exit.** Reported, not gated, as the seeded study is. Whether M2 is run again with the explicit force, and whether
Case 2's main results are replaced, is decided after the run, in review.

## Step 5.4f: the Lorentz force explicit — the results

The run ran as declared, on NERSC Perlmutter: one A100, float64, the main runs' image, at commit `782e4f3`, from
M1L1's last state at t = 1104.5 s. A first submission stopped at its start, before any step, on a relative path to
M1L1's checkpoint; it was resubmitted unchanged but for the path. The observables are `post.py`'s at the campaign's
`6863c42`, and the comparisons `case2_reported`'s, as for the seeded runs. The run is reported, not gated.

**The run.**

| Run | Δt λ | Settled after | Averaged | Simulated | s a step | Wall time | Courant max | Ledger max |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| M1, explicit | 0.336 | 401 s | 450 s | 853 s | 0.035 | 33 min | 0.21 | 4.3e-13 |

- It settled at its second window (0.342 K, 0.1 W), so `--average-unsettled` was not used. Its sub-batches are
  independent. Mass stayed under 1.1e-9, the charge residual at the solver's stop (0.0019), and no flow returned
  through the outlet.

**What it found.** On M1 without a seed, beside M1L1 (implicit) and the seeded heater-4 run (implicit):

| | M1L1 | M1, explicit | heater-4 |
|:--|--:|--:|--:|
| Rolls saturated at | 0.266 m | 0.299 m | 0.068 m |
| RMS of w at x = 0 | 3.7e-12 m/s | 4.3e-12 m/s | 1.7e-5 m/s |
| at x = 0.1 m | 1.3e-9 m/s | 3.9e-10 m/s | 5.0e-3 m/s |
| at x = 0.2 m | 6.8e-6 m/s | 7.8e-7 m/s | — |
| Nu at x = 0.2 m (band 3.55–5.13) | 2.15 | 2.15 | **4.09** |
| T on the axis at x = 0.2 m (band 355.4–358.7 °C) | 347.9 °C | 347.9 °C | **356.0 °C** |
| T_w at x = 0.2 m (OF_KIT 394.5 °C) | 415.8 °C | 415.8 °C | 391.8 °C |
| Figure 24, share within the widened band | 71 % | 70 % | 100 % |
| Figure 37 | 63 % | 63 % | 84 % |
| Figure 38, T_w's mean offset from OF_KIT | +5.5 K | +5.5 K | +0.3 K |

- Every other figure's share moved by at most 3 points (Figure 35, 81 % to 78 %). Up to x = 0.15 m the two
  unseeded runs agree to rounding: their probes there are steady and equal.
- The largest temperature swing is 32.5 K, against M1L1's 30.2 K and the paper's 30 K.

**The hypotheses.**
- *The rolls saturate at or before 0.236 m:* not met. They saturate 3.3 cm further downstream, at 0.299 m.
- *Nu at x = 0.2 m rises into the band:* not met. It stays at 2.15.

So, as declared, the stage's slowness is not what delays M1's onset, and the seeded study's account stands alone:
the onset is set by the disturbance that reaches the heater.

**A layer with Case 2's ingredients does not behave as the duct does.** A layer with a through-flow (Re 400, Ra
1e5, Pr 0.02, 18 cells a wavelength) and a field along the rolls' axis at Δt λ = 0.34 (`make_showcase_figures.py
--only buoyancy`):
- under the implicit shift, its rolls grow 25 % too slowly and travel 19 % too slowly;
- with the force explicit, they grow and travel as without the field, within the mesh's 1.6 %.

The duct's rolls, under the explicit force, grow less over the heated zone's second half, not more: from the same
rounding at x = 0, a 1.8e5-fold amplification to x = 0.2 m against M1L1's 1.8e6-fold. So the duct's amplification
from rounding is set by more than a layer's linear growth: the Hartmann walls' braking, the side-layer jets beside the
heated wall and the thermal layer still developing along the heater. Whether M1's onset converges in the time step
would need a run at Δt/4 (Δt λ = 0.084) with either force, about 2.5 GPU hours each. Case 2's results stay those of
the implicit runs above.

## Step 5.4f: why the explicit run grows less, and two more runs (declared before the runs)

**What the two M1 runs show.** Both carry the same rounding upstream of the heater. Over its second half the
explicit run's disturbance grows less per metre:

| x | 0.05–0.10 m | 0.10–0.15 m | 0.15–0.20 m | 0.20–0.25 m |
|:--|--:|--:|--:|--:|
| M1L1 (implicit) | 77 /m | 86 /m | 86 /m | 84 /m |
| explicit | 69 /m | 74 /m | 78 /m | 76 /m |

- The rolls' axial wavelength is the same in both, 3.6–4.0 cm, from the saved velocity at 0.24–0.48 m.
- On the axis probes, the period is 4.52 s implicit and 4.00 s explicit, and the u signal at 0.33 m lags the one at
  0.15 m by four periods and 4.20 s or 3.78 s. So the rolls drift at 0.82 u₀ implicit and 0.92 u₀ explicit.
- Growth per metre times drift is the temporal growth: 85.8 /m × 0.815 u₀ and 76.1 /m × 0.917 u₀, 0.699 and
  0.698 s⁻¹, over 0.10–0.25 m. The shift leaves the rolls' growth in time unchanged; it slows their drift by 11 %, and
  so raises their growth per metre.
- The steel Hartmann walls brake these rolls at about 1.7 s⁻¹ (the numerics page). Their velocity therefore follows
  the buoyancy nearly quasi-statically, and the shift, which slows the velocity's response by 1 + Δt λ, shifts its
  phase more than its amplitude. The layer above has no such braking, which is why it behaves differently.

The explicit run is the time-accurate one (`validation.transient`), so on M1 the implicit runs' earlier onset owes
part of its position to the shift.

**Run A: the implicit force at Δt/4.** M1 at Δt 0.00375 s (Δt λ = 0.084), implicit, from M1L1's last state:
300 s burn-in, then 150 s averaged in five sub-batches, settled or not. A diagnostic, reported, not gated.
- *Hypothesis:* it moves toward the explicit run, with a period of 4.0–4.2 s on the per-step axis probes and a growth
  per metre of 76–80 /m over x = 0.10–0.25 m.
- If it stays at M1L1's 4.5 s and 85 /m, the shift is not what separates the two runs.

**Run B: the axial spacing halved.** At 5 mm, the rolls have 7–8 cells a wavelength along the duct. In the
buoyancy record's through-flow layer, limited momentum and van Leer heat transport at 8 cells a wavelength slow an
advected roll's growth by 23 % and its growth per metre by 13 % (6 % at 10 cells, 3 % at 12). Over Case 2's heater,
though, the base temperature varies along x, which may keep the limiter in its smooth range for so small a
disturbance. M2 (4 mm, 9.5 cells) saturated at 0.274 m against M1's 0.266 m, at a smaller Δt λ.
- The run: M1's cross-section with 2.5 mm along the core (874 axial cells against 450), the explicit force, and
  Δt 0.015 s, from the inlet's developed flow. It follows 5.4f's protocol: 300 s burn-in, 50 s settling windows,
  450 s of averaging in ten sub-batches, extendable to 900 s, and averaged from 1,200 s if unsettled. It is compared
  with the explicit run at 5 mm.
- *Hypotheses (if under-resolved transport delays the onset):*
  - the growth per metre over x = 0.10–0.25 m exceeds the explicit 5 mm run's 76 /m by at least 5 %;
  - the rolls saturate at least 1 cm upstream of its 0.299 m;
  - Nu at x = 0.2 m rises above its 2.15.
- If none holds, the axial spacing is not what delays M1's onset. Reported, not gated.

Both on one A100 on Perlmutter at commit `782e4f3`. The runner is the explicit rerun's `case2_run.py`, which now
re-imposes the inlet profile and face constraints on a loaded state: another Δt rebuilds the inlet profile, equal to
the start's only to its solve's tolerance.

## Step 5.4f: Run A, the implicit force at Δt/4 — the result

Run A (Perlmutter job 61761121) ran 300 s from M1L1's last state and averaged 1,455–1,635 s, unsettled after one
window. Neither declared outcome held:
- *Period 4.0–4.2 s:* met. The axis probes give 4.19 s, against M1L1's 4.52 s and the explicit run's 4.00 s.
- *Growth per metre 76–80 /m over x = 0.10–0.25 m:* not met, and not M1L1's 85 /m either. The disturbance stays at
  its rounding to x = 0.2 m (RMS of w 5e-12 m/s at x = 0, 0.1 and 0.2 m), then grows about 70 /m from 0.2 to 0.4 m. The
  rolls saturate (RMS of u 1e-3 m/s) at 0.88 m, against 0.26 m.

At Δt 0.015 s both forces grow from x ≈ 0, so on M1 the rolls' onset is not converged in the time step, and not only
through the Lorentz split. Buoyancy is taken from the old temperature, a first-order partitioned coupling. Its buoyancy
Courant number at Δt 0.015 s is 0.04–0.05 in both runs, where the calibration of `build_thermal_step` puts the
critical Rayleigh number under 3 % low at 0.025 and about 15 % low at 0.2. Whether that, or Run A's start, accounts
for the change is what Runs C and D test.

## Step 5.4f: Runs C and D (declared before the runs)

**Run C: the explicit force at Δt/4.** M1 at Δt 0.00375 s with the force explicit, from the explicit run's last state
at 1,957.7 s, with Run A's protocol: 300 s burn-in, then 150 s averaged in five sub-batches, settled or not.
- *Hypothesis (a time error other than the Lorentz split removes the growth over the heater):* the RMS of w at
  x = 0.2 m stays within ten times its value at x = 0, and the rolls saturate downstream of 0.5 m.
- If instead it grows at 60 /m or more over x = 0.10–0.25 m and saturates upstream of 0.35 m, the difference between
  Run A and the Δt 0.015 s runs lies in the implicit shift at Δt/4 or in Run A's start.

**Run D: Run A continued.** From Run A's last state at 1,635.2 s, with the same settings: another 300 s burn-in and
150 s averaged.
- *Hypothesis (Run A had settled):* the rolls saturate within 5 cm of Run A's 0.88 m, and the RMS of w at x = 0.3 m
  stays within a factor of three of Run A's 6e-9 m/s.
- If the saturation moves on by more than 5 cm, Run A's window was a transient and is reported as one.

Both reported, not gated, on one A100 at commit `782e4f3` with the explicit rerun's runner.

## Step 5.4f: Runs C and D — the results

Run C (Perlmutter job 61763837) ran 300 s from the explicit run's last state and averaged 2,308–2,488 s. Run D (job
61763839) ran 300 s from Run A's last state and averaged 1,986–2,166 s. Both were unsettled after one window, as Run
A's protocol allows; 72 and 64 minutes on one A100.

**Run D: Run A had not settled.** Neither clause held:
- The rolls died out. Over Run D's window the flow is steady: its probes move by 4e-12 K, and the heat stored
  changes by under 1e-7 W.
- The RMS of w is at its rounding along the whole duct, at most 1.8e-10 m/s (at x = 1.1 m). At x = 0.3 m it is
  5e-12 m/s, against Run A's 6e-9.
- As declared, Run A's window was a transient, and is reported as one. At Δt/4 with the implicit force, Case 2's
  flow on M1 is steady.

**Run C: the explicit force at Δt/4.** Not met as declared: the first clause holds, the second does not.
- *The RMS of w at x = 0.2 m within ten times its value at x = 0:* met. It is 1.4e-11 m/s against 7.3e-12, twice
  that value; at Δt 0.015 s the explicit run grew 1e5-fold there.
- *Saturation downstream of 0.5 m:* not met. Over the window the rolls saturate at 0.47 m. In the six 30 s
  sub-batches they saturate at 0.42 m in the first and at 0.78–0.88 m in the other five: the first sub-batch's burst
  sets the window's figure.
- *The alternative:* neither clause. The growth over x = 0.10–0.25 m is 34 /m, not 60 or more, and the saturation
  lies 12 cm downstream of 0.35 m.
- So at Δt/4 the growth over the heater vanishes with the force explicit, as with it implicit. Neither the shift nor
  Run A's start removed it.

**The time step, not the Lorentz split.**

| Run | Force | Δt (s) | RMS of w, x = 0.2 m over x = 0 | Growth, x = 0.10–0.25 m | Saturated at |
|:--|:--|--:|--:|--:|:--|
| M1L1 | implicit | 0.015 | 1.2e6 | 86 /m | 0.266 m |
| Explicit rerun | explicit | 0.015 | 1.0e5 | 76 /m | 0.299 m |
| Run A | implicit | 0.00375 | 1.2 | 24 /m | 0.88 m |
| Run C | explicit | 0.00375 | 2.0 | 34 /m | 0.47 m (sub-batches: 0.42, then 0.78–0.88 m) |
| Run D | implicit | 0.00375 | 1.2 | 1 /m | none: steady |

- At Δt 0.015 s, with either force, the disturbance grows from x ≈ 0 and the rolls saturate over the heater. At
  Δt/4, with either force, it stays at its rounding to x = 0.2 m. The rolls that remain lie beyond 0.4 m, and with
  the implicit force they die out within 1,000 s. Whether Run C's would too is not known: it was not continued.
- So on M1 Case 2's rolls are not converged in the time step, and the Lorentz split is not the cause. Case 2's
  reported runs (5.4f; Δt 0.015 s on M1, 0.012 s on M2) carry this.
- The suspect is the partitioned buoyancy, taken from the old temperature. Its Courant number is 0.04–0.05 at
  Δt 0.015 s and 0.0095–0.0098 at Δt/4. It is not yet tested on Case 2's rolls.
- Against the codes: their hot fluid reaches the axis between x = 0.05 and 0.1 m, and LMhdX's at Δt 0.015 s between
  0.25 and 0.35 m. A smaller step moves LMhdX's rolls further from theirs, not nearer. One code, CFX, reports a
  steady flow (p. 12).
- Run B, the axial spacing halved, is still queued. It runs at Δt 0.015 s and is compared with the explicit run at
  the same step, so it isolates the axial spacing at that step only.

## Step 5.4f: the step's time error on Case 2's rolls (declared before the runs)

Runs A, C and D leave a time error, outside the Lorentz split, that grows the rolls over the heater at Δt 0.015 s and
not at Δt/4. Its suspect is the partitioned buoyancy. In the field-free Bénard onset the step's critical Rayleigh
number is flat in the step ([thermal](../physics/thermal.md)), but a first-order error vanishes at the onset of a
stationary mode. Case 2's rolls drift past the grid and are braked by their walls, so the test puts them in that
regime, with the step's products matched to Case 2's.

**The layer.** Rolls between Case 2's Hartmann walls, in units of the duct's half-width $a$ and $a^2/\kappa$:
- rigid isothermal plates at z = ±1, heated below, gravity along −z;
- Case 2's steel walls at y = ±1 (Ha 235; M1's 48 cells across, 6 in each Hartmann layer and 6 in each wall);
- Pr 0.0196, and the field along y;
- x periodic over one wavelength at $ka = 3.8$, the rolls' 3.8 cm over $a$ = 2.3 cm, on M1's 8 cells; 32 cells
  along z.
- The walls brake the rolls at $f = 0.065\,\sigma B^2/\rho$ (`vortex_decay_rate`), so $f\Delta t$ = 0.022 at
  $\Delta t\,\sigma B^2/\rho = 0.34$, as in Case 2 on M1 (0.021).

**The runs.**
- Two Rayleigh numbers: 1.1 Ra_c, near the onset (σ = 0.019 f), and 3 Ra_c, growing as Case 2's rolls do at Δt
  0.015 s (σ = 0.31 f). Ra_c is 1.15e6 with the walls' friction.
- Two flows: at rest, and a Poiseuille through-flow (centreline 18.2 κ/a) whose rolls drift 0.024 cells a step at
  0.34, as Case 2's do on M1 (0.028). The flow is driven by a force matched to the walls' braking.
- At each, $\Delta t\,\sigma B^2/\rho$ = 0.34, 0.17, 0.084 and 0.042 (Case 2's Δt, Δt/2, Δt/4 and Δt/8), with the
  force explicit and implicit.
- The growth rate is measured by power iteration on the thermal step, the buoyancy record's method. The buoyancy
  Courant number is 0.012–0.020.
- The reference is each case's rate at Δt → 0, extrapolated at first order from 0.084 and 0.042. Sommeria and
  Moreau's model (`convection_growth_rate` with the walls' friction and the through-flow) is a check.

**The measure** is the fall in the growth rate from 0.34 to 0.084, in units of f, with the force explicit. Case 2's
rolls lose 0.5–0.7 s⁻¹ over the heater between those steps, 0.34–0.48 f at its 1.46 s⁻¹.
- *Hypothesis (the step's time error on the rolls explains Case 2):* the drifting rolls' rate falls by at least
  0.17 f, half of Case 2's smaller figure, at either Rayleigh number.
- *Alternative:* it falls by under 0.034 f, a tenth, at both. Then the step's error on the rolls themselves does not
  explain Case 2. The cause lies in the duct's base flow (the heated wall's layer, the side layers), or in what grows.
- Anything between is reported as a partial account.

**What grows over the heater.** The explicit Δt 0.015 s run's last state, less its window's mean, gives w along x on
the axis over x = 0.05–0.25 m.
- *Hypothesis (the rolls grow from rounding):* its dominant wavelength is 3.2–4.4 cm, the saturated rolls' within
  10 %, so the layer's $ka$ applies.
- *Alternative:* under 2 cm, four cells or fewer. Then the grid scale grows, where the averaging leaves the Lorentz
  braking weak, and the layer at 8 cells is not the model.

Run on the laptop GPU (float64) with the study's script outside the repository, which joins the campaign scripts
when #336 is settled. Reported, not gated. Run B continues on Perlmutter.

## Sources

- C. Mistrangelo et al., Nucl. Fusion 65 (2025) 116006,
  doi:10.1088/1741-4326/ae0800.
- D. Martelli, A. Venturini and M. Utili, Fusion Eng. Des. 138 (2019) 183–195,
  doi:10.1016/j.fusengdes.2018.11.028.
- J. Matolich Jr., *Thermal conductivity and electrical resistivity of Type 316
  stainless steel from 0 to 1800 F*, NASA CR-54151, Battelle Memorial Institute
  (1965).
- ARIES/UCSD material properties, austenitic stainless steel (316),
  <https://qedfusion.org/LIB/PROPS/PANOS/ss.html>.
