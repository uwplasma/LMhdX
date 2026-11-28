# Mistrangelo et al. 2025: a heated conducting duct

Mistrangelo et al. (Nucl. Fusion 65, 2025, 116006) ran six codes on the same problem. PbLi flows through a duct with
2 mm steel walls, heated from below over 60 cm, through a magnet whose field falls off toward both ends. There is no
experiment, so the benchmark is code against code. LMhdX runs the whole duct, with the fringing field, the conducting
walls, inertia, buoyancy and heat conducted through the steel.

```{image} ../_static/mistrangelo.webp
:alt: Nu along the heated zone for Case 1, and Nu and the axis temperature for Case 2, against the six codes' bands
:align: center
```

*LMhdX against the band the six codes span (grey) over the heated zone (shaded). (a) Case 1 on the finest mesh.
(b, c) Case 2 without a perturbation and with one, at two time steps.
`python scripts/make_showcase_figures.py --only mistrangelo` draws it from `docs/_static/data/mistrangelo.json`.*

**Case 1 (steady, 0.03 m/s, 4 W/cm²) agrees with the codes.** We gated it on three meshes and two outlet lengths
against a rule fixed before the runs. The gate holds on 6 of the 11 figures. Three of the five misses are numerical
and converge toward the band with refinement: the heater's sharp edge (Figures 9 and 18) and a side-layer jet
(Figure 12). The other two follow inputs the paper leaves to each code: the steel's conductivity ratio (a jet 0.7 %
high, Figure 13, partly not traced) and the inlet (Figure 19).

**Case 2 (convection rolls, 0.01 m/s, 10 W/cm²) agrees once the flow carries a small disturbance.** Buoyancy makes
rolls aligned with the field grow along the heated zone. How far along they appear depends on how large the
disturbances feeding them are, and the paper does not say what its codes carried.
- With a weak random forcing just upstream of the heater (10⁻⁴ u₀), the rolls start where the codes' do. Nu, the
  axis temperature and the wall temperature then lie in the codes' band. The result is the same at 15 ms and 3.75 ms,
  with the onset moving by 1.5 cm between them.
- Without it, LMhdX's flow is clean to double precision, so the rolls start late (at x ≈ 0.27 m with 15 ms steps) or
  not at all (3.75 ms). Over the heater's second half Nu is then about half the codes'.

We report the perturbed runs as LMhdX's Case 2, on the assumption that the codes carry disturbances of that order. We
have asked the authors what their codes used. Time-dependent buoyant convection stays a research-stage capability
until that is confirmed.

## Set it up

`validation/mistrangelo_duct.py` builds both cases from the frozen specification,
`src/lmhdx/data/benchmarks/specs/mistrangelo2025.toml`. This runs a coarse Case 2 on a laptop CPU:

```python
import jax.numpy as jnp

import lmhdx
from lmhdx.grid import CENTER, Field
from lmhdx.thermal import advance_thermal, build_thermal_step
from validation import mistrangelo_duct as duct

lmhdx.enable_x64()
# 16 fluid cells across (4 in each layer), 2 cells per wall, 5 cm along the core
thermal, state = duct.case2(cells=16, cells_in_layer=4, wall_cells=2, spacing=0.05, dt=0.05)
step = build_thermal_step(thermal, dtype=jnp.float64)
no_source = Field(jnp.zeros(thermal.thermal_grid.shape), (CENTER,) * 3, thermal.thermal_grid)  # q'' is a wall condition
state, history = advance_thermal(step, state, no_source, 20)  # 1 s, with every step's diagnostics
fields = {"u": state.velocity[0].data, "v": state.velocity[1].data, "w": state.velocity[2].data,
          "T": state.temperature.data}
figures = duct.case2_observables(thermal, fields)  # {figure: (x, y)}, as the paper defines them
```

It takes about 15 s. The figures above use the same `case2_observables` on fields averaged over a long run.

The production runs used meshes M1–M3 on one NVIDIA A100 each, in float64:

| Mesh | Fluid cells across | In each layer | Per wall | Along the core | Cells | Δt | Time per step |
|:--|--:|--:|--:|--:|--:|--:|--:|
| M1 | 48 | 6 | 6 | 5 mm | 1.62 M | 15 ms | 0.035–0.039 s |
| M2 | 64 | 8 | 8 | 4 mm | 3.56 M | 12 ms | 0.080–0.092 s |
| M3 | 84 | 10 | 10 | 3.3 mm | 7.16 M | 10 ms | 0.20 s |

What the setup fixes, and what it had to infer:
- **Duct.** The half-width is a = 23 mm along the field, with 2 mm steel walls on all four sides and x from −1.5 to
  1.5 m. The field is the paper's eq. 1, B_y(x) = B₀/2 [1 − tanh((|x| − 0.7315)/0.1)], with Ha = 235.
- **Heating.** The heat enters through the bottom wall's outer surface at |x| ≤ 0.3 m, over its full 50 mm width.
  Every other outer surface is insulated and adiabatic.
- **Inlet and outlet.** The inlet takes the developed profile at u₀ and 300 °C. The outlet has zero-gradient velocity
  and temperature, with a fixed pressure that includes its hydrostatic part.
- **Properties.** The paper prints no material data. We pin PbLi's viscosity, electrical conductivity and expansion to
  the printed Ha, Re and Gr, and take its other properties from Martelli et al. 2019. We pin the steel's conductivity
  ratio to the developed pressure gradient, −0.0863 σu₀B₀².
- **Inferred.** The setup infers Case 2's 10 W/cm² and 0.01 m/s from the ratios of the printed Gr and Re. It also
  infers the heated width, L = a in Nu, and the pressure scale σu₀B₀². A figure confirms each one, as the checks under
  the reference tables below show.
- **Numerics.** The runs use Van Leer heat transport, limited momentum transport and backward-Euler viscosity and
  conduction, and they solve the potential and pressure at every step.
- **Case 2's averaging.** Each run burns in for at least 300 s, until two consecutive 50 s means of the bulk
  temperature agree within 1 % of its rise and the heat stored is under 1 % of the input. It then averages 450 s in
  sub-batches and checks them for independence.
- **Case 2's disturbance.** A random vertical force acts over x = −0.5 to −0.35 m. It is uniform along the field, has
  the three lowest sine modes across the height and draws new random amplitudes every 2 s. Its size makes the vertical
  velocity at the slab's end 10⁻⁴ u₀, which takes 3.19 × 10⁻⁵ m/s² on M1. It enters through the thermal step's
  `force(t)`, so the library is unchanged. The run script is `case2_run.py --seed`, among the run scripts.

## Case 1: results

There were seven runs: M1, M2 and M3 with the outlet at 1.5 m, M1 and M2 with it at 2.25 m, M1 at half the time
step, and M1 with every solver's stopping tolerance 100 times looser. All were steady in the heated zone, with the heat
balance closed to 1e-11 of its largest term (5e-10 with the looser tolerances). A point passes if its distance outside
the codes' band, widened by the digitizing uncertainty, is within LMhdX's own uncertainty U, and U is at most half the
band's width. In practice the mesh change sets U.

| Figure | Quantity | Points | Inside the band, M3 | Points not passing, M1 / M2 / M3 / M1L2 / M2L2 | Gate |
|:--|:--|--:|--:|:--|:--|
| 6 | T on the axis | 168 | 71 % | 0 / 0 / 0 / 0 / 0 | holds |
| 7 | u on the axis | 60 | 58 % | 0 / 0 / 0 / 0 / 0 | holds |
| 9 | T(z) at x = −0.3 m | 155 | 58 % | 3 / 3 / 3 / 3 / 3 | does not hold |
| 10 | T(z) at x = 0 | 244 | 46 % | 0 / 0 / 0 / 0 / 0 | holds |
| 12 | u(z) at x = −0.3 m | 174 | 70 % | 3 / 2 / 1 / 3 / 2 | does not hold |
| 13 | u(z) at x = 0 | 235 | 68 % | 4 / 4 / 3 / 4 / 4 | does not hold |
| 15 | potential on z = −a | 207 | 84 % | 0 / 0 / 0 / 0 / 0 | holds |
| 16 | potential across z at x = 0 | 189 | 98 % | 0 / 0 / 0 / 0 / 0 | holds |
| 17 | bulk temperature | 169 | 75 % | 0 / 0 / 0 / 0 / 0 | holds |
| 18 | Nu | 169 | 93 % | 2 / 2 / 2 / 2 / 2 | does not hold |
| 19 | transverse energy E_t | 287 | 41 % | 15 / 16 / 16 / 15 / 16 | does not hold |

Why the five miss:
- **Figures 9 and 18, the heater's leading edge.** The heat flux steps from zero to q'' at x = −0.3 m, and 4–5 mm
  cells interpolate across the step. With 1 mm cells near both ends (the edge study), Figure 9's points converge into
  the band. Figure 18's first point, 3.5 mm in, is then resolved and lies 13 % above the one code that reaches it.
  Nu there follows Nu ≈ 34 d^−0.37 (d in mm) on every mesh.
- **Figure 12, the side-layer jet.** Its peak passes on M3 and fails on M1 and M2, so the miss is one of resolution.
- **Figure 13, the +z jet at x = 0.** Its peak stays about 0.7 % above the widened band after extrapolation. About a
  third of it comes from the mesh. The rest follows the steel's conductivity ratio, pinned to the printed gradient, and
  a weaker buoyant asymmetry between the two jets than the codes show.
- **Figure 19 upstream of x = −1 m.** LMhdX's inflow is the developed profile, so almost no transverse motion arrives
  from upstream. The band there is one code's, most likely its inlet's. The specification leaves the inlet to each
  code.

Reported, not gated:
- **Figure 8, the pressure.** After one best-fit offset (under 2.5 Pa), every point lies in the widened band. The drop
  over the band's span is 791.8 Pa on M3 against the band's 789.9 Pa.
- **Figures 11 and 38, OF_KIT's curves.** T across the duct agrees within 0.30 K at five stations, T_w within 0.44 K
  and T_b within 0.80 K.
- **Buoyancy and the fringe.** Against the held flow of step 5.4c, they move the heated zone's temperatures by at most
  0.46 K, as expected.

## Case 2: results

Case 2 is reported against the codes' spread, not gated, because the codes each chose their own domain, inlet, outlet
and averaging window (Tables 2–5).

**With the disturbance (M1).** The tables give Nu, the axis temperature and the wall temperature over the heater's
second half.
**Bold** values lie within the band widened by the digitizing uncertainty.

| x | Nu, unseeded | Nu, seeded, 15 ms | Nu, seeded, 3.75 ms | Nu, the band |
|:--|--:|--:|--:|:--|
| 0.10 m | 2.26 | 2.33 | 2.28 | 2.52–2.80 |
| 0.15 m | 2.20 | **3.00** | 2.70 | 3.25–4.19 |
| 0.20 m | 2.15 | **4.09** | **4.17** | 3.55–5.13 |
| 0.25 m | 2.12 | **3.96** | **4.41** | 3.87–4.39 |

| x | T on the axis, unseeded | seeded, 15 ms | seeded, 3.75 ms | The band |
|:--|--:|--:|--:|:--|
| 0.10 m | 335.9 °C | **338.9 °C** | 336.5 °C | 340.4–345.3 °C |
| 0.15 m | 341.9 °C | **348.7 °C** | **348.4 °C** | 347.4–349.5 °C |
| 0.20 m | 347.9 °C | **356.0 °C** | **355.9 °C** | 355.4–358.7 °C |
| 0.25 m | 354.0 °C | **361.8 °C** | **362.2 °C** | 362.0–363.2 °C |

| x | 0.10 m | 0.15 m | 0.20 m | 0.25 m |
|:--|--:|--:|--:|--:|
| T_w, OF_KIT | 397.2 °C | 392.4 °C | 394.5 °C | 399.2 °C |
| T_w, unseeded | 400.8 °C | 408.4 °C | 415.8 °C | 423.0 °C |
| T_w, seeded, 15 ms | 400.1 °C | 397.0 °C | 391.8 °C | 398.6 °C |
| T_w, seeded, 3.75 ms | 400.6 °C | 401.6 °C | 391.5 °C | 395.6 °C |

The next table gives every banded figure as the share of its points within the widened band:

| Figure | Unseeded | Seeded, 15 ms | Seeded, 3.75 ms |
|:--|--:|--:|--:|
| 23, u on the axis | 95 % | 95 % | 94 % |
| 24, T on the axis | 71 % | 100 % | 97 % |
| 26, u(z) at x = −0.3 m | 95 % | 95 % | 95 % |
| 27, u(z) at x = 0 | 100 % | 100 % | 100 % |
| 28, T(z) at x = −0.3 m | 65 % | 65 % | 65 % |
| 29, T(z) at x = 0 | 100 % | 100 % | 100 % |
| 31, φ on z = −a | 73 % | 75 % | 77 % |
| 32, φ on z = +a | 81 % | 80 % | 80 % |
| 33, φ(z) at x = 0 | 100 % | 100 % | 100 % |
| 34, bulk T | 97 % | 100 % | 100 % |
| 35, E_t | 81 % | 88 % | 81 % |
| 37, Nu | 63 % | 84 % | 86 % |

What remains with the disturbance:
- Nu at the heater's end, x = 0.28–0.36 m, up to 0.55 below the widened band.
- The potential (Figures 31 and 32) is about 2 µV low everywhere, out of 117 µV. The offset is a constant, since LMhdX
  fixes the potential's mean over the whole duct. A uniform +2 µV puts every point of Figures 31–33 in the widened band.
- The heater's leading edge (Figure 28) and the jets (Figure 26), as in Case 1, and E_t upstream (Figure 35), where
  only a disturbance at the inlet acts. A disturbance of 10⁻² u₀ there brings E_t within 0.3 decades.
- The swings behind the heater are larger, 41 K on the axis at x = 0.33 m against 30 K unseeded. The paper reports
  fluctuations up to 30 K there without saying whether that is a swing or an amplitude.

**M2 agrees with M1 at its probes.** The seeded M2 run did not meet the settling rule, because its windows' means
swing by about 1 K about a steady level. So we compare it at its probes over its last 450 s:

| Probe | M2, unseeded | M2, seeded | M1, seeded | M1, unseeded |
|:--|--:|--:|--:|--:|
| Heated interface, x = 0.15 m | 409.01 °C | 397.40 ± 0.57 °C | 397.96 ± 0.53 °C | 408.67 °C |
| Axis, x = 0.15 m | 342.09 °C | 348.96 ± 0.33 °C | 348.57 ± 0.28 °C | 341.81 °C |
| Heated interface, x = 0.33 m | 406.03 ± 0.06 °C | 388.40 ± 0.80 °C | 391.63 ± 1.05 °C | 403.99 ± 0.08 °C |
| Axis, x = 0.33 m | 376.67 ± 0.06 °C | 371.32 ± 0.35 °C | 370.61 ± 0.30 °C | 375.14 ± 0.03 °C |

**What sets the onset.** These checks rule out the obvious numerical explanations:
- *The disturbance's size.* Before the heater, a tenfold larger seed moves the onset 3.6 cm upstream. At the inlet the
  field damps a seed by five decades on its way to the heater, so an inflow disturbance alone does not explain the
  codes' early onset.
- *The time step.* A linearised disturbance grows and drifts the same at 15, 7.5 and 3.75 ms (56–58 per metre,
  9.5 mm/s), and with the same seed the onset is 0.068 m at 15 ms and 0.082 m at 3.75 ms. Without a seed the onset is
  not converged in the step. At 15 ms the step itself leaves disturbances of about 10⁻¹¹ K that the rolls amplify, and at
  3.75 ms the flow goes steady.
- *The scheme.* An explicit Lorentz force moved the unseeded onset 3 cm downstream, and halving the axial spacing
  moved it 2 cm upstream. Sub-stepping the transports, conduction or the viscous solve at 15 ms left the growth over
  the heater as it was, and giving the 3.75 ms step the 15 ms step's coupling lag brought none of it back. None of
  these explains the difference between the steps, or the codes' earlier onset.

**Unseeded, for the record.** Three runs (M2 and M1 with the outlet at 1.5 m, M1 at 2.25 m) settled after 650–900 s, and
we averaged them over 450–900 s. Behind the heater the temperature on the axis swings by up to 30 K with a period of
about 4 s, as the paper finds, but the rolls saturate only at the heater's end. 62 % or more of every figure's points
lie within the widened band. The misses gather over the heater's second half.

## Reference tables

### The developed section (Figure 13)

`python -m validation.mistrangelo` solves the flow in the uniform field on a section one cell long, with the steel
resolved.

| Cells (fluid / layer / wall) | −dp/dx in σu₀B₀² | Side jet | Centre |
|:--|--:|:--|--:|
| 48 / 6 / 6 | 0.08599 | 2.826 u₀ at \|z\| = 0.928 a | 0.8469 u₀ |
| 72 / 9 / 8 | 0.08622 | 2.810 u₀ at 0.929 a | 0.8490 u₀ |
| 96 / 12 / 10 | 0.08630 | 2.803 u₀ at 0.929 a | 0.8497 u₀ |

On 96/12 the core (\|z\| ≤ 0.7 a) is within 2.09 px of each of Figure 13's 37 analytical markers, 0.62 % low on
average. The paper puts the jets' maximum at 0.935 a, and the section has 0.929 a.

### The digitized figures and their bands

`python scripts/digitize_mistrangelo.py DIR` writes `src/lmhdx/data/benchmarks/references/mistrangelo2025-case1.json`
(90 KB) and `-case2.json` (77 KB) from the paper's figure images (CC BY 4.0). Each is within 0.53 px of every extracted
pixel column and carries its digitizing uncertainty, 2.3–2.5 px. A band is the envelope of the codes chosen to agree,
not a confidence interval. `validation.mistrangelo.band_misfit` measures the distance outside it, widened by that
uncertainty. Every code's full curve is in the release assets `mistrangelo2025-case1-extraction.zip`
and `-case2-extraction.zip` of
[mistrangelo2025-references](https://github.com/uwplasma/LMhdX/releases/tag/mistrangelo2025-references).
`benchmarks/provenance.json` holds every SHA-256.

| Case 1 figure | Quantity | Left out of the band | Why (page) | Chosen by |
|:--|:--|:--|:--|:--|
| 6 | T on the axis | FLUENT_UK; the band within −0.75 < x < 0.8 m | it deviates (p. 7) | the paper |
| 7 | u on the axis | FLUENT_UK; Figure 6's window | its core velocity deviates (p. 8) | figure reading |
| 8 | p on the axis | FLUENT_UK | all agree but FLUENT_UK (p. 8) | the paper |
| 9 | T across, x = −0.3 m | FLUENT_CIEMAT, OF_ORNL | their axial resolution (p. 8) | the paper |
| 10 | T across, x = 0 | FLUENT_UK | it lies lower | figure reading |
| 11 | T across, OF_KIT at five stations | no band | one code (pp. 8–9) | none |
| 12 | u across, x = −0.3 m | FLUENT_UK, CFX | as p. 9 says of Figure 13 | figure reading |
| 13 | u across, x = 0 | FLUENT_UK, CFX | the analytical solution agrees with the other four (p. 9) | the paper |
| 15 | φ on the interface | none | most of the data fits well together (p. 10) | the paper |
| 16 | φ across, x = 0 | FLUENT_UK | a slight deviation (p. 10) | the paper |
| 17 | bulk T, eq. 6 | FLUENT_UK | non-physical (p. 10) | the paper |
| 18 | Nu, eq. 7, on L = a | FLUENT_UK | as p. 10 says of Figure 16 | figure reading |
| 19 | E_t, eq. 8 (log) | FLUENT_UK, CFX; COMSOL not shown | as before (p. 10); CFX's diffusion (p. 16) | figure reading |
| 38 | OF_KIT's T_w and T_b, both cases | no band | one code (p. 16) | none |

| Case 2 figure | Quantity | Left out of the band | Why (page) | Chosen by |
|:--|:--|:--|:--|:--|
| 23 | u on the axis | CFX, OF_ORNL | CFX differs; OF_ORNL fluctuates (p. 12) | the paper |
| 24 | T on the axis | CFX, OF_ORNL | CFX's steeper rise; OF_ORNL fluctuates (p. 12) | the paper |
| 26 | u across, x = −0.3 m | CFX | all but CFX reproduce the analytical solution (p. 13) | the paper |
| 27 | u across, x = 0 | CFX | its profile differs, as p. 12 says of Figure 23 | figure reading |
| 28 | T across, x = −0.3 m | all but COMSOL and OF_KIT | the two fit well together (p. 13) | the paper |
| 29 | T across, x = 0 | CFX | its wall is 20 K hotter (p. 13) | figure reading |
| 30 | p on the axis | none | p. 14 | the paper |
| 31, 32 | φ on z = ∓0.023 m | FLUENT_CIEMAT, FLUENT_UK | their curves are instantaneous (p. 14) | the paper |
| 33 | φ across, x = 0 | CFX | good agreement (p. 15), but CFX is offset | figure reading |
| 34 | bulk T, eq. 6 | OF_ORNL, COMSOL | unphysical (pp. 15–16) | the paper |
| 35 | E_t, eq. 8 (log) | CFX, OF_ORNL, COMSOL | CFX differs; the others unphysical in E_t (pp. 15–16) | the paper |
| 37 | Nu, eq. 7, on L = a | CFX, OF_ORNL, COMSOL | CFX differs; the others' bulk temperatures fluctuate (p. 16) | figure reading |

COMSOL's outlet backflow (pp. 12, 14) keeps it out of the bands of Figures 23, 24 and 30–32, which stop at x = 0.7 m.

Checks of the inferred values (`tests/test_steady.py`):
- **Pressure scale.** Figure 8's slope over \|x\| < 0.4 m is −568.6 Pa/m, against −564.5 Pa/m for the printed −0.0855
  σu₀B₀², a difference of 0.8 %.
- **Heated width.** The energy balance with the full outer width heated gives 310.12 °C, inside Figure 17's outlet
  plateau to half a pixel. A 46 mm patch would be 0.8 K off.
- **Length in Nu.** Figure 38's T_w − T_b at x = 0.29 m gives Nu = 2.34 on L = a, inside Figure 18's band there
  (2.23–2.41). L = 2a would double it.
- **Case 2's velocity.** Figure 30's slope is −191.2 Pa/m, against −189.9 Pa/m for the developed gradient at 0.01 m/s.
- **Case 2's flux.** Over 0.5 < x < 0.9 m the top of Figure 34's band is 375.71–375.99 °C, against 375.93 °C for the
  energy balance at 10 W/cm².

### Step 5.4c: the heated zone on a held flow

This step solves Case 1's heated zone (−0.4 ≤ x ≤ 0.4 m) on the developed section's flow, with conducting walls and no
fringe or buoyancy. It has 576,000 cells and takes 3.5 minutes on a laptop GPU.
`python -m validation.mistrangelo heated DIR` writes its record.

| Figure | Quantity | Inside the codes' band | Within half a pixel | Largest outside | Mean offset |
|:--|:--|--:|--:|--:|--:|
| 9 | T(z) at x = −0.3 m | 35 % | 73 % | 0.15 K, in the heated wall | +0.030 K |
| 10 | T(z) at x = 0 | 93 % | 100 % | 0.024 K | −0.079 K |
| 17 | bulk T(x) | 86 % | 96 % | 0.025 K | +0.021 K |
| 18 | Nu(x), −0.3 < x < 0.3 m | 98 % | 98 % | 3.3, at the heater's leading edge | +0.12 |

### Records

We declared each campaign before it ran (D12). Its full declaration, run-by-run results and diagnostics stay at the
commit where we wrote them:
- [Case 1, gated, and the edge study](https://github.com/uwplasma/LMhdX/blob/b97ad3a8525a20502279961d5b4201d158d971e9/docs/validation/mistrangelo.md)
  (#327, #331), and Case 2's main and seeded runs on the same page (#331).
- [Case 2's time-step study](https://github.com/uwplasma/LMhdX/blob/50ea1644b97daaf74d36e1b30a732690e175cf05/docs/validation/mistrangelo.md)
  (#338): the explicit Lorentz force, the quarter step, the sub-stepped terms, the linearised disturbance and the seeded
  run at 3.75 ms.

The records are release assets of
[LMhdX Research Assets v2](https://github.com/uwplasma/LMhdX/releases/tag/lmhdx-research-assets-v2) (ADR 0009 D38),
with the run scripts:

| File | SHA-256 |
|:--|:--|
| `mistrangelo2025-case1-heated-zone.json` | `908708d0d4a53e917088f385c28062c9c5d7adceae70f66abe8e0cdc09c21dc5` |
| `mistrangelo2025-case1-gated.json` | `703e31e7385a7c7cb1c3976e2128ca9c158d2336ad74a836bad275909296968c` |
| `mistrangelo2025-case1-edges.json` | `b264e0a44d6dc9b2b2838fa51867d706fce6288972c9094f498ccb2c8a925792` |
| `mistrangelo2025-case2-reported.json` | `1795cb16da52cb2b97318716d7e64717940bdc80186bfa3b94b46b86b007f251` |
| `mistrangelo2025-case2-fluctuations.npz` | `662070017f4181eb09d8a5394e13498a3c2ec1ac87038736322a5c8364d0b1c6` |
| `mistrangelo2025-case2-seeded.json` | `28b73d09fd10a24a7b188ae2747547548e870916f8803673c0d8715a96512a23` |
| `mistrangelo2025-case2-seeded-fluctuations.npz` | `7f8e916616f4a2a68c0e0fa0356ed3de4404fe2e6ae57a403d02246907345d60` |
| `mistrangelo2025-case2-seeded-calibration.json` | `4874fd87c5e8a3dbb5855b4f8c9f2b7d9e561752fbfd1382db4c6b08d0949574` |
| `mistrangelo2025-case2-runs.zip` (the run scripts) | `d555bcfd531b1719e05171e71471dd80cb308929108d09f91f7e0273651e29c8` |

`case2_reported.py` (SHA-256 `cd7a53ea…`), which writes the Case 2 records from the runs' outputs, is not yet
published. It joins the campaign scripts once #336 is settled. The seeded run at 3.75 ms (#338) has no released record
yet either, and its curves are in the figure's data file. The averaged fields (5.9 GB, and 3.7 GB for the seeded runs)
stay outside the repository.

`tests/test_steady.py` checks the functions that made these results, in about 20 s, without a production run: the
coarse duct's heat balance and throughflow, every observable of a known flow, the gate's rule on curves made from the
references, Case 2's settling and independence checks, and `case2_reported`.

## Sources

- C. Mistrangelo et al., Nucl. Fusion 65 (2025) 116006, doi:10.1088/1741-4326/ae0800.
- D. Martelli, A. Venturini and M. Utili, Fusion Eng. Des. 138 (2019) 183–195, doi:10.1016/j.fusengdes.2018.11.028.
- J. Matolich Jr., *Thermal conductivity and electrical resistivity of Type 316 stainless steel from 0 to 1800 F*,
  NASA CR-54151, Battelle Memorial Institute (1965).
- ARIES/UCSD material properties, austenitic stainless steel (316),
  <https://qedfusion.org/LIB/PROPS/PANOS/ss.html>.
