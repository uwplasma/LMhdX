# Mistrangelo et al. 2025: a heated conducting duct

Mistrangelo et al., *A code-to-code benchmark for magneto-convection in a
horizontal duct*, Nucl. Fusion 65 (2025) 116006, compares six codes on a heated
PbLi duct with conducting steel walls in a fringing field. There is no
experiment. Plan step 5.4 reproduces it in stages, as set out in ADR 0009. This
page records the frozen specification and the first check.

## The specification

`src/lmhdx/data/benchmarks/specs/mistrangelo2025.toml` is frozen before any new
numerics. Each value is marked *printed* (quoted, with its page), *inferred*
(deduced, with the reason) or *pinned* (chosen to reproduce a printed number).
The paper prints no material properties, so two choices carry the comparison.

**PbLi.**
- The properties are Martelli et al. (2019) at 573 K, the paper's reference
  and temperature.
- Those values give Ha 233.2, Re 3157 and Gr 1.364e7, where Table 1 prints 235,
  3220 and 1.42e7.
- The codes' figures carry the printed groups, so `μ`, `σ` and `β` are pinned to
  reproduce them, with `ρ`, `c_p` and `k` kept:
  - `μ` moves −1.95 %;
  - `σ` moves −0.42 %;
  - `β` moves +0.11 %.

  That `β` stays within 0.11 % of Martelli's suggests the codes used these
  properties with a slightly lower viscosity.
- With the heated patch spanning the full 50 mm outer width, the bulk rises
  10.12 K, as Figure 17 reads. A 46 mm patch would give 9.31 K.

**Steel.**
- The paper names neither a grade nor a value. Matolich (1965) measured a
  316 steel at 572 °F (300 °C): a resistivity of 96.0 µΩ cm and a conductivity
  of 18.3 W/(m K). That is 1.042e6 S/m, or 1.18 times the pinned PbLi.
- The developed section below needs a ratio of 1.246 to reproduce the paper's
  analytical gradient. That is 1.097e6 S/m, 5.3 % higher, and it is the
  pinned value. The wall conductance ratio is then
  `c = σ_w t_w / (σ a) = 0.108`.
- The steel's `ρ c_p` matters only in the start-up transient of a steady case.
  It is 7850 × 544 J/(m³ K), from the ARIES table.

**Inferred elsewhere:**
- Case 2's 10 W/cm² and 0.01 m/s, from the ratios of Gr and Re;
- `L = a` in `Nu`;
- `g = 9.81 m/s²`.

**Left unspecified by the paper:** the conditions on the wall's end faces, the
reference levels of the potential and the pressure, and the inlet profile and
domain length, which each code chose for itself.

## The first check: Case 1's developed section

In the uniform field the flow is developed. `validation.mistrangelo` solves it
on a section one cell long, with the steel resolved in cells on both transverse
axes, at Ha 235 (`python -m validation.mistrangelo`):

| Cells (fluid / layer / wall) | −dp/dx in σu₀B₀² | Side jet | Centre |
|:--|--:|:--|--:|
| 48 / 6 / 6 | 0.08599 | 2.798 u₀ at \|z\| = 0.927 a | 0.8469 u₀ |
| 72 / 9 / 8 | 0.08622 | 2.796 u₀ at 0.930 a | 0.8490 u₀ |
| 96 / 12 / 10 | 0.08630 | 2.794 u₀ at 0.931 a | 0.8497 u₀ |

**The pressure gradient.**
- The pinned ratio makes 96/12 meet the printed 0.0863, so the gradient is
  not an independent check here.
- What it does show: the coarser meshes converge to it, and the codes' own
  0.0855 is 0.9 % below.
- At the measured steel ratio of 1.18 the section gives 0.0826, 4.3 % below.
- The gradient is close to linear in the ratio, at about 0.056 per unit.

**The profile is the independent check.** Figure 13 reads a core of about
0.857 u₀ and side jets of about 2.7–2.8 u₀ at |z| ≈ 0.935 a. The section gives
0.850 u₀ and 2.79 u₀ at 0.931 a. A digitized comparison follows with the
reference curves.

`tests/test_steady.py` checks:
- the specification's printed groups and its anchors;
- the 48/6 section's gradient, within 0.5 % of 0.0863.

## Reference curves (Case 1, part 1)

`src/lmhdx/data/benchmarks/references/mistrangelo2025-case1.json` holds
Figures 6–13, read by `validation.mistrangelo.digitize`. The figures are
xmgrace plots, embedded in the PDF as images; each record carries the image's
page, xref and SHA-256.

**How a figure is read.**
- The plot frame, its legend and its major ticks are found in the image.
- Two labelled ticks per axis anchor the scale. The other ticks fit within
  0.67 px.
- Each code is picked out by its colour: one value per pixel column, the
  median of that colour's pixels, sampled to 60 points.
- In Figure 13, the analytical solution's open red circles are split from
  CFX's red line by their shape.

**What each record holds.**
- **A band:** the union of the codes that the paper says agree, with the page
  where it says so. This is what a gate can use.
- **The curves:** each code's own curve, where it is visible. Where the
  curves coincide, the one drawn on top hides the rest, so FLUENT_CIEMAT's
  black is seldom visible. Each curve records its coverage.

**The uncertainty is about 2 px:**

| | x per px | y per px |
|:--|--:|--:|
| Figure 6 | 0.0047 m | 0.028 K |
| Figure 12 | 7.9e-5 m | 2.3e-4 m/s |

Figures 7 and 13 give the same point, the axis at x = 0, 0.4 % apart.

**Two independent checks.**
- **The pressure scale.** Figure 8's slope over |x| < 0.4 m is −567.7 Pa/m.
  The printed −0.0855, taken in σu₀B₀² with the pinned PbLi, gives
  −564.5 Pa/m. They agree to 0.6 %, which confirms both the scale and the
  pinned σ (`tests/test_steady.py`, within 1 %).
- **The developed section against Figure 13's analytical solution.**
  - In the core (|z| < 15 mm) the 96-cell section lies within 1.8 px of the
    markers, 1.1 px (0.9 %) low on average.
  - On the steep flanks of the jets, one value per column is a poor trace,
    and the difference grows to about 10 px.
  - The jets themselves are 0.0838 m/s. The codes' are 0.0854 and 0.0829 at
    x = 0, where buoyancy makes them unequal.
  - At x = −0.3 m (Figure 12), only 0.1 m into the uniform field, the
    codes' jets are 0.081 m/s: the flow there is still developing.

Figures 15–19 and 38, and Case 2's, follow in later parts, along with the
sources in `benchmarks/provenance.json`.

## Sources

- C. Mistrangelo et al., Nucl. Fusion 65 (2025) 116006,
  doi:10.1088/1741-4326/ae0800.
- A. Martelli, A. Venturini and M. Utili, Fusion Eng. Des. 138 (2019) 183–195.
- J. Matolich Jr., *Thermal conductivity and electrical resistivity of Type 316
  stainless steel from 0 to 1800 F*, NASA CR-54151, Battelle Memorial Institute
  (1965).
- ARIES/UCSD material properties, austenitic stainless steel (316),
  <https://qedfusion.org/LIB/PROPS/PANOS/ss.html>.
