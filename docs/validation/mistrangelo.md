# Mistrangelo et al. 2025: a heated conducting duct

Mistrangelo et al., *A code-to-code benchmark for magneto-convection in a
horizontal duct*, Nucl. Fusion 65 (2025) 116006, compares six codes on a heated
PbLi duct with conducting steel walls in a fringing field. There is no
experiment. Plan step 5.4 reproduces it in stages (ADR 0009). This page records
step 5.4a: the frozen specification, the developed section, and both cases'
digitized figures; step 5.4c, Case 1's reduced heated zone; step 5.4e, Case 1
gated, with its edge study; and step 5.4f, Case 2 reported.

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

Its SHA-256 is `a5964eef19a404c3f08ed39e274766199f3a720543155c11e71e6c961c1618d5`. It goes to the research-assets
release with this step's pull request (D38), with `case1_reported.py`. The averaged fields (2.5 GB) and the
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
  - This is inferred, not checked. A run seeded at the inlet would check it; it would be a new declaration (D12).
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
    profile, as in Case 1's Figure 19.
  - *At the heater's end (x = 0.29–0.35 m):* up to 0.74 decades above, where LMhdX's rolls saturate.
  - *Downstream of x = 0.8 m, where the field falls off:* up to 1.3 decades above, alike with the long outlet. Not
    traced.
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
| `case2_reported.py` | `15d1dc5dbfc513a186611e997b240363a007d5a2ed24ca805656f29a00a0e9a2` |
| `mistrangelo2025-case2-runs.zip` | `2346bb672db3bb30fb618c802b906f80b762014a80f4f6fa3140fa52cdff96b9` |

- They go to the research-assets release with this step's pull request (D38).
- `case2_reported.py` writes all three records from the runs' outputs and 5.4e's record.
- `mistrangelo2025-case2-runs.zip` holds the run harness:
  - the runs (`case2_run.py`, `case1_edge_run.py`);
  - the observables of the averaged fields on a GPU (`post.py`);
  - the batch and deploy scripts;
  - the coarse resume check (`smoke.py`).
- Each run's output directory is given under `$SCRATCH`. The averaged fields and sub-batches (5.9 GB) are kept
  outside the repository.

`tests/test_steady.py` checks `case2_reported` without a solve, on curves made from the references:
- at the band's middle every figure lies inside, and Figure 30 after its offset;
- one point off is a figure's only failure and its largest excess;
- the error is the batch means' standard error, in decades on a log axis.

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
