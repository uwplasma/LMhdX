# Mistrangelo et al. 2025: a heated conducting duct

Mistrangelo et al., *A code-to-code benchmark for magneto-convection in a
horizontal duct*, Nucl. Fusion 65 (2025) 116006, compares six codes on a heated
PbLi duct with conducting steel walls in a fringing field. There is no
experiment. Plan step 5.4 reproduces it in stages (ADR 0009). This page records
step 5.4a: the frozen specification, the developed section, and both cases'
digitized figures; and step 5.4c, Case 1's reduced heated zone.

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
result is claimed before the corrected momentum step.

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

**The problem.** `validation.mistrangelo.case1` builds the whole duct: eq. 1's fringe along `y`; steel walls that
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
