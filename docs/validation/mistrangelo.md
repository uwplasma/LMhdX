# Mistrangelo et al. 2025: a heated conducting duct

Mistrangelo et al., *A code-to-code benchmark for magneto-convection in a
horizontal duct*, Nucl. Fusion 65 (2025) 116006, compares six codes on a heated
PbLi duct with conducting steel walls in a fringing field. There is no
experiment. Plan step 5.4 reproduces it in stages, as set out in ADR 0009. This
page records step 5.4a: the frozen specification, the developed section, and
both cases' digitized figures.

## The specification

`src/lmhdx/data/benchmarks/specs/mistrangelo2025.toml` is frozen before any new
numerics. Each value is marked *printed* (quoted, with its page), *inferred*
(deduced, with the reason) or *pinned* (chosen to reproduce a printed number),
with its source. The paper prints no material properties. The file is
metadata, read by `validation.mistrangelo` and the tests, and not a case that
`lmhdx` runs.

**Conventions** (`[conventions]`). `Ha`, `Re` and `Gr` are on the half-width
`a` = 23 mm, as Table 1 defines them; `Nu` is on `L = a` (inferred, below); the
pressure is in `σ u₀ B₀²` (inferred from Figure 8). Joule heat is omitted in the
fluid (printed, eq. 4) and in the walls (inferred: they are given no source).

**The values that are not printed:**
- **PbLi, pinned.** Martelli et al. (2019) at 573 K give Ha 233.2, Re 3157 and
  Gr 1.364e7, where Table 1 prints 235, 3220 and 1.42e7. `μ`, `σ` and `β` are
  pinned to the printed groups (−2.0, −0.4 and +0.1 %), with `ρ`, `c_p` and `k`
  kept.
- **Steel, pinned.** The paper names neither a grade nor a value. Matolich
  (1965) gives a 316 steel at 572 °F a conductivity ratio of 1.183 over the
  pinned PbLi, and 18.3 W/(m K). The developed section reproduces the paper's
  analytical gradient at an effective ratio of 1.246, 5.3 % higher, which is
  pinned (`c = 0.108`). The paper does not say how its analytical solution
  models the wall, so the 5.3 % absorbs that too.
- **Inferred:**
  - the heated width, the full 50 mm outer width (Figure 17 confirms it);
  - Case 2's 10 W/cm² and 0.01 m/s, from the ratios of Gr and Re (Figures 34
    and 30 confirm them);
  - `L = a` in `Nu` (Figures 18 and 38 confirm it);
  - `g = 9.81 m/s²`.
- **Unspecified by the paper:** the wall's end faces, the reference levels of
  the potential and the pressure, the inlet profile and domain length, and the
  analytical solution's wall model.

**What is calibrated, and what checks it** (`[calibration]`, ADR 0009 D35).
- **Calibrated:** the PbLi `μ`, `σ` and `β`, on the printed groups; the wall's
  effective ratio, on the developed gradient −0.0863, which is therefore no
  acceptance check.
- **Independent:** the jets and the axial velocity (Figures 7, 12, 13), the
  temperatures, bulk temperature and `Nu` (Figures 6, 9–11, 17, 18, 38), the
  potential (15, 16) and the transverse energy (19).
- **A known miss:** Matolich's measured ratio gives a gradient 4.1 % below
  −0.0863; nothing is tuned against it.

Case 1's gate is declared before step 5.4e (ADR 0009 D35), so
`tolerances_frozen_before_production` is false.

## Case 1's developed section

`python -m validation.mistrangelo` solves the flow in the uniform field on a
section one cell long, with the steel resolved in cells on both transverse
axes, at Ha 235. The profile is read at `y = 0`, as `a + b y²` through the two
rows nearest the midplane on each side.

| Cells (fluid / layer / wall) | −dp/dx in σu₀B₀² | Side jet | Centre |
|:--|--:|:--|--:|
| 48 / 6 / 6 | 0.08599 | 2.826 u₀ at \|z\| = 0.928 a | 0.8469 u₀ |
| 72 / 9 / 8 | 0.08622 | 2.810 u₀ at 0.929 a | 0.8490 u₀ |
| 96 / 12 / 10 | 0.08630 | 2.803 u₀ at 0.929 a | 0.8497 u₀ |

- **The gradient is not an independent check.** The pinned ratio makes 96/12
  meet 0.0863 by construction; the codes' own value is 0.0855. At Matolich's
  ratio the 96/12 section gives 0.0828, 4.1 % below. Validation row 6's 1 % on
  −0.0863 is therefore met by construction, and its Mistrangelo part moves to
  row 19 (ADR 0009 D35).
- **The core follows from the gradient.** The Hartmann walls' current balance
  gives `K(1 + c + 1/Ha)/(c + 1/Ha) = 0.853`, against the section's 0.850.
- **The profile is the independent check**, against Figure 13's analytical
  markers. On 96/12, the core (\|z\| ≤ 0.7 a) is within 2.09 px of each of the
  markers' 37 columns and 0.62 % low on average, gated within the record's
  uncertainty (2.44 px) and 1 %. On the jets' steep flanks the difference is
  about 10 px. The paper puts the jets' maximum at 0.935 a (p. 12); the section
  has 0.929 a.

## The digitized figures

`src/lmhdx/data/benchmarks/references/mistrangelo2025-case1.json` (90 KB) and
`-case2.json` (77 KB) hold the figures, read by
`python scripts/digitize_mistrangelo.py DIR`. The figures are xmgrace plots
embedded in the PDF as images, extracted with `pymupdf`
(`Document.extract_image(xref)`). They are not in the repository, so the
digitizer is not in the package or in `validation/`. It also writes every
code's curve to `DIR/mistrangelo2025-case1-curves.json` and `-case2-curves.json`.

**How a figure is read.**
- **Calibration.** The frame and the major ticks outside it are detected. Two
  labelled anchors per axis label the ticks, and every tick inside the frame
  must land on the label lattice, at least three of them, or the digitizer
  raises. Every axis fits within 0.54 px.
- **Curves.** Each code is picked out by its colour, one median value per pixel
  column. Figure 13's analytical markers are split from the red by shape, and
  Figure 38's two cases by a branch split.
- **The band** is the spread of the codes that agree, each code's dash gaps up
  to 15 px closed first, within the paper's window where it gives one.
  `agree_basis` records whether the paper chose those codes or my reading did.
- **What is shipped.** Per figure: the image's page, xref and SHA-256, the
  calibration, the band, and the curves that are not a code's (Figure 13's
  markers, Figure 11's stations, Figure 38's branches).
  - Each is simplified to within half a pixel of every extracted pixel column,
    with nothing decimated before. The kept vertices keep their extracted x,
    and the values are rounded to a tenth of a pixel.
  - Against the full extraction, every record is within 0.53 px, log axes in
    decades.
  - The per-code curves, every column of them, are not shipped. With the figure
    images (the paper is CC BY 4.0) they are the release assets
    `mistrangelo2025-case1-extraction.zip` and `-case2-extraction.zip` on
    `mistrangelo2025-references`.
  - The release is a draft until the owner publishes it. Until then the
    archives are reachable only to the repository's maintainers, so the
    extraction is reproducible publicly from the paper's PDF and this script,
    not yet from the archives.
- **Provenance.** `benchmarks/provenance.json` records the paper's DOI and
  SHA-256, and each file's SHA-256 with the extraction date and the
  digitizer's SHA-256, and the release asset's (`extraction`). A test checks
  the file and digitizer hashes against the files.

**The digitizing uncertainty** is in each record (`uncertainty`), per axis in
the axis's units: the read line within 1.5 px, the compact record within 0.6 px
of it (the synthetic figure's bounds), and the axis's calibration misfit, which
each axis also carries in its units (`misfit`). That is 2.3–2.5 px, for example
0.064 K in Figure 6 and 5.5e-4 m/s in Figure 12.

**The band** is the envelope of the codes chosen to agree, not a statistical
confidence interval. A comparison's acceptance band is that envelope widened by
the digitizing uncertainty:
- `validation.mistrangelo.band_misfit(x, y, figure)` takes arrays and numbers,
  and returns the largest excess beyond it, zero inside.
- It interpolates the band as plotted, in decades on a log axis.
- Each edge is taken at its extreme over x ± the x uncertainty, then widened by
  the y uncertainty.

At 5.4e, ADR 0009 D35's band also takes in LMhdX's discretization error.

### Case 1

| Figure | Quantity | Left out of the band | Why (page) | Chosen by |
|:--|:--|:--|:--|:--|
| 6 | T on the axis | FLUENT_UK; the band within −0.75 < x < 0.8 m | it deviates (p. 7) | the paper |
| 7 | u on the axis | FLUENT_UK; Figure 6's window | its core velocity deviates (p. 8) | my reading |
| 8 | p on the axis | FLUENT_UK | all agree but FLUENT_UK (p. 8) | the paper |
| 9 | T across, x = −0.3 m | FLUENT_CIEMAT, OF_ORNL | their axial resolution (p. 8) | the paper |
| 10 | T across, x = 0 | FLUENT_UK | it lies lower | my reading |
| 11 | T across, OF_KIT at five stations | no band | one code (pp. 8–9) | — |
| 12 | u across, x = −0.3 m | FLUENT_UK, CFX | as p. 9 says of Figure 13 | my reading |
| 13 | u across, x = 0 | FLUENT_UK, CFX | the analytical solution agrees with the other four (p. 9); CFX is not separable from its markers | the paper |
| 15 | φ on the interface | none | most of the data fits well together (p. 10) | the paper |
| 16 | φ across, x = 0 | FLUENT_UK | a slight deviation (p. 10) | the paper |
| 17 | bulk T, eq. 6 | FLUENT_UK | non-physical (p. 10) | the paper |
| 18 | Nu, eq. 7, on L = a | FLUENT_UK | as p. 10 says of Figure 16 | my reading |
| 19 | E_t, eq. 8 (log) | FLUENT_UK, CFX; COMSOL not shown | as before (p. 10); CFX's diffusion (p. 16) | my reading |
| 38 | OF_KIT's T_w and T_b, both cases | no band | one code (p. 16) | — |

**Checks of Case 1's figures** (`tests/test_steady.py`):
- **The pressure scale.** Figure 8's slope over \|x\| < 0.4 m is −568.6 Pa/m,
  against −564.5 Pa/m for the printed −0.0855 σu₀B₀² with the pinned PbLi:
  0.8 %, gated within 1 %. It cannot tell the pinned σ from Martelli's, 0.4 %
  away.
- **The heated width.** The energy balance with the full outer width heated
  gives 310.12 °C. Over Figure 17's outlet plateau (310.08–310.15 °C), every
  column read, it lies inside the band to half a pixel either way: at most
  0.08 px below its lower edge and 0.42 px above its upper. A 46 mm patch
  would be 0.8 K off.
- **The length in Nu.** Figure 38's `T_w − T_b` at x = 0.29 m is 19.28 K, so
  `Nu = q'' a / (k (T_w − T_b))` is 2.34 with L = a. That lies inside Figure
  18's band there, 2.23–2.41 (Nu on L = a), and inside its acceptance band
  (`band_misfit`); L = 2a would double it.
- **The printed "Nu → 2.45"** (p. 10, on L = a) matches Figure 18's codes near
  x = 0.3 m, where their band reaches 2.48. It is about 4 % above OF_KIT's
  value, within Figure 38's digitizing uncertainty, about ±0.14 in Nu.

### Case 2

Case 2 is a statistical target, reported and not gated (ADR 0009 D35): a
time-dependent flow whose figures are time averages. COMSOL's outlet backflow
(pp. 12 and 14) is kept out of the bands of Figures 23, 24 and 30–32, which stop
at x = 0.7 m.

**The reference runs differ** (Tables 2–5, section 4), and those differences
are part of the codes' spread. Each code chose its own domain, inlet and outlet:

| Code | Time scheme | Δt (s) | Axial cells | Inlet | Outlet | Length (m) | Averaged over (s) |
|:--|:--|:--|:--|:--|:--|--:|--:|
| FLUENT_CIEMAT | first-order implicit | not given | 141, uniform | developed laminar | p = 0; extra viscosity in the last 0.2 m; backflow allowed | 2.3 | 60 |
| FLUENT_UK | second-order implicit | not given | 287 | ∂ₓu = 0, p = 0 | fixed mass flow | 3 | 50 |
| CFX | backward Euler | 0.015, the adaptive step's final value | 450, uniform | uniform u | opening, backflow allowed; mean p = 0; T from the energy balance | 2.25 | 225 |
| OF_ORNL | forward Euler | not given | 230, uniform | uniform u | ∂ₓu = 0, p = 0; extra viscosity in the last 0.2 m | 2.3 | 100 |
| OF_KIT | second-order backward | 0.01 to 0.004, by mesh | 700 | developed turbulent | convective, ∂ₜu + u∂ₓu = 0 | 3 | 450 |
| COMSOL | second-order backward | not given | 300 | uniform u | ∂ₓu = 0, p = 0 | 2.5 | 50 |

Every inlet is at 300 °C. No code reports a burn-in or a measure of its
averages' statistical convergence, and the windows differ ninefold.

**LMhdX's proposed Case 2 protocol, declared before any run.** It is a proposal for the owner to settle, not yet a decision.
- **The set-up.** Each result records its solver, time step, mesh, domain,
  inlet and outlet: the configuration that 5.4d's outlet-length study settles
  (ADR 0009 D37).
- **Burn-in.** Averaging starts after one flow-through of the domain at u₀, and
  once two consecutive 50 s means move by under 1 % of the bulk rise (0.76 K).
  This applies to both the outlet bulk temperature and the heated wall's
  temperature at x = 0.
- **The window.** At least 450 s, the longest reference window, split into five
  equal batches.
- **The statistical error** is the standard error of the five batch means,
  quoted with every averaged observable.
  - That assumes the batches are independent, which the window does not
    guarantee.
  - Before any statistical precision is quoted, the batch means'
    autocorrelation and the result's sensitivity to the window length are
    inspected, and recorded with the result.
  - If either shows the batches are not independent, the window is
    lengthened, under the rule below.
- **The comparison** is against the codes' spread, widened by the digitizing
  uncertainty and by LMhdX's discretization and statistical errors. The
  spread is the envelope of the codes chosen, not a confidence interval. The
  comparison is reported, not gated.
- **Fixed in advance.** If the burn-in or the window changes after a first
  comparison, the old outcome and the reason stay in the record.
- **Not before the corrected momentum step.** Case 2 is a moving, buoyant flow.
  The stabilised transient step does not keep a pressure-balanced rest state
  with the field along `y`. No Case 2 result is claimed until that step is
  corrected, or until a time-step refinement shows the effect is small here.

| Figure | Quantity | Left out of the band | Why (page) | Chosen by |
|:--|:--|:--|:--|:--|
| 23 | u on the axis | CFX, OF_ORNL | CFX differs; OF_ORNL fluctuates, numerical problems (p. 12) | the paper |
| 24 | T on the axis | CFX, OF_ORNL | CFX's steeper rise; OF_ORNL fluctuates (p. 12) | the paper |
| 26 | u across, x = −0.3 m | CFX | all but CFX reproduce the analytical solution (p. 13); its markers merge with CFX's line | the paper |
| 27 | u across, x = 0 | CFX | its profile differs, as p. 12 says of Figure 23 | my reading |
| 28 | T across, x = −0.3 m | all but COMSOL and OF_KIT | the two fit well together (p. 13) | the paper |
| 29 | T across, x = 0 | CFX | all predict the thicker layer (p. 13), but CFX's wall is 20 K hotter | my reading |
| 30 | p on the axis | none | p. 14 | the paper |
| 31 | φ, z = −0.023 m | FLUENT_CIEMAT, FLUENT_UK | their curves are instantaneous (p. 14) | the paper |
| 32 | φ, z = 0.023 m | FLUENT_CIEMAT, FLUENT_UK | their curves are instantaneous (p. 14) | the paper |
| 33 | φ across, x = 0 | CFX | good agreement (p. 15), but CFX is offset | my reading |
| 34 | bulk T, eq. 6 | OF_ORNL, COMSOL | unphysical (p. 15, caption; p. 16) | the paper |
| 35 | E_t, eq. 8 (log, seven decades) | CFX, OF_ORNL, COMSOL | CFX differs; OF_ORNL and COMSOL unphysical in the kinetic energy (pp. 15–16) | the paper |
| 37 | Nu, eq. 7, on L = a | CFX, OF_ORNL, COMSOL | CFX differs; the bulk temperatures of OF_ORNL and COMSOL fluctuate (p. 16) | my reading |

**Withdrawn:** a comparison of the section's jets with Figure 26. What read as
CFX's curve there is the analytical solution's markers, merged in the JPEG, so
the jets are compared with Figure 13 only.

**Checks of Case 2's figures** (`tests/test_steady.py`):
- **The velocity, 0.01 m/s.** Figure 30's slope over \|x\| < 0.4 m is
  −191.2 Pa/m, against −189.9 Pa/m for the developed −0.0863 σu₀B₀² at that
  velocity: 0.7 %. The slope scales with u₀ alone, so this fixes it.
- **The flux, 10 W/cm².** Over 0.5 < x < 0.9 m, the top of Figure 34's band is
  375.71–375.99 °C, every column read, against 375.93 °C for the energy
  balance: at most 0.22 K off, within the record's uncertainty (0.45 K). The window starts
  past the settling of the bulk temperature after the heated zone, about 0.1 m.
  FLUENT_UK's plateau, 4.5 K lower, sets the band's lower edge.

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
