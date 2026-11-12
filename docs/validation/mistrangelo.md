# Mistrangelo et al. 2025: a heated conducting duct

Mistrangelo et al., *A code-to-code benchmark for magneto-convection in a
horizontal duct*, Nucl. Fusion 65 (2025) 116006, compares six codes on a heated
PbLi duct with conducting steel walls in a fringing field. There is no
experiment. Plan step 5.4 reproduces it in stages (ADR 0009). This page records
step 5.4a: the frozen specification, the developed section, and both cases'
digitized figures.

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
