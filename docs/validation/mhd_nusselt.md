# MHD Nusselt campaign

This page checks how well LMhdX predicts heat transfer in a duct with a
magnetic field. The Nusselt number, the wall heat flux against the wall-to-bulk
temperature difference, rises with the field as the flow flattens. LMhdX agrees
with an independent spectral solve within 0.06 %, and with a published analysis
at low and moderate Hartmann numbers.

```{image} ../_static/thermal_developed_duct.webp
:alt: Velocity and temperature of an H1 square duct at Ha 0 and 100 in 3-D, Nusselt numbers against aspect ratio and Hartmann number, and mesh convergence
:align: center
```

The spectral reference takes a few seconds per number. From the repository
root, this computes Ha 10 on 48 points:

```python
from validation.shercliff import developed_nusselt

for heating in ("H1", "H2", "T"):
    print(heating, f"{developed_nusselt(10.0, 48, heating=heating):.4f}")
```

```text
H1 4.5221
H2 3.7985
T 3.6023
```

These are the spectral values in the Ha 10 row of the table below.

The duct is Shercliff's square duct, with insulating walls, a field along $y$
and no Joule heat. Ha is on the half-width $a$, and $D_h=2a$. This page
compares its developed Nusselt numbers against two references:
- an independent spectral solve, `validation.shercliff.developed_nusselt(Ha,
  points, heating=...)`. It uses Chebyshev collocation on the same nodes as that
  module's velocity, and shares no operator with the package;
- a published analysis, Bluck, Wolfendale & Marquis (2015).

LMhdX runs on the 128-cell meshes of `duct_problem`. The thermal conditions
are:
- H1: a uniform axial heat input, the wall at one temperature around the
  section;
- H2: a uniform wall flux, axially and around the section;
- T: the wall at one temperature everywhere, the excess decaying along the duct.

Each Nu is the section's mean Nusselt number, the wall flux averaged over the
whole perimeter against the wall-to-bulk difference, on $D_h=2a$.

**Two kinds of statement.** LMhdX against the spectral solve is verification.
`tests/test_steady.py` gates one point, 48 cells at Ha 10, within 0.3 % of the
spectral solve in each condition. The published curves are a reading from a
figure, reported with its uncertainty. `python -m validation.mhd_nusselt`
regenerates every number on this page in about ten minutes on a laptop CPU. The
compact record,
`src/lmhdx/data/benchmarks/references/bluck2015-shercliff-nusselt.json`, holds
the spectral column of the table below, the definition of Nu and that command,
beside the readings. The test also checks the stored spectral values at Ha 10.

| Ha on $a$ | H1 mean Nu on $D_h$: LMhdX / spectral | H2: LMhdX / spectral | T: LMhdX / spectral |
|--:|:--|:--|:--|
| 1 | 3.6360 / 3.6352 | 3.1098 / 3.1091 | 2.9971 / 2.9968 |
| 10 | 4.5249 / 4.5221 | 3.8006 / 3.7985 | 3.6034 / 3.6023 |
| 31.6 | 5.3983 / 5.3972 | 4.4921 / 4.4912 | 4.1307 / 4.1309 |
| 100 | 6.0633 / 6.0616 | 5.0594 / 5.0577 | 4.4842 / 4.4845 |
| 316 | 6.4877 / 6.4892 | 5.4381 / 5.4392 | 4.6840 / 4.6865 |

**LMhdX against the spectral solve.**
- They agree within 0.06 % in all three conditions.
- The 128-cell value at Ha 100 is converged to 0.02 % (128 against 192 cells).
- At Ha 0 the spectral solve gives the exact series in H1 and H2 (3.60795
  and 3.08738, `validation.thermal`), and 2.97753 in T, where Shah & London
  list 2.976.
- All three numbers climb toward their slug-flow limits, 7.114, 6 and
  $\pi^2/2=4.935$, as the core flattens. The side layers, of thickness
  $\propto Ha^{-1/2}$, slow the approach.

## The published curves

Bluck, Wolfendale & Marquis (2015) is an analytical series for the same duct.
We read their square-duct curves in Figs. 4 (H1) and 6 (H2) from the vector
paths of the accepted manuscript.
- **How we read them.** Each curve is a polyline whose vertices lie at
  Ha $=10^{k/4}$, mapped through the figures' own tick marks. That regular
  spacing suggests the authors' computed points, but the vertices are evidence
  of the drawn curve, not access to their numbers. The readings, the
  calibration and the manuscript's SHA-256 are in
  `src/lmhdx/data/benchmarks/references/bluck2015-shercliff-nusselt.json` and
  `benchmarks/provenance.json`.
- **A check.** The H1 curve's plateau reads 3.60678, against their tabulated
  Ha 0 value of 3.6068. The H2 plateau reads 3.0824, against their 3.0871.

**The Ha convention is inferred, not confirmed.** The extracted curves agree
much more closely with the spectral solve under the following reading:
- **H1**, Figure 4, read on the half-width. It is then within 0.03 % of the
  spectral value at Ha 1 and 0.07 % at Ha 10, falling to 0.19 % low at
  Ha 31.6, 0.49 % at 100 and 1.2 % at 316. Read on $D_h$, as the paper's eq. (8)
  defines Ha, it would lie 3–12 % above.
- **H2**, Figure 6, read on the full width, $D_h=2a$, as eq. (8) defines it. It
  is then 0.17 % low at an equivalent Ha 0.5, and 0.22 %, 0.43 %, 1.0 % and 2.0 %
  low at 5, 15.8, 50 and 158. Read on the half-width it would lie up to 10 %
  below.

So the authors appear to have drawn Figure 4 against the half-width, a factor
of 2 from the paper's own definition. That is our inference from the fit.
Neither the authors nor an independently reproduced analytical value of theirs
at nonzero Ha has confirmed it. The record keeps both readings and the
published definition.

Three uncertainties stay separate:
- the reading's precision, which is the calibration through the figures' tick
  marks;
- the published curve's own error. Figure 4's plateau equals their table to
  1e-5, and Figure 6's lies 0.15 % below theirs. A plateau difference is not a
  bound on the rest of a curve.
- the convention, which is inferred.

Both curves fall below the spectral solve as Ha grows. The authors note that
their series' hyperbolic terms overflow double precision at large Ha, and that
they regroup them through logarithms. The error left there is not quantified.
The agreement of two in-house discretizations, LMhdX and the spectral solve,
does not show that the publication is wrong there. The comparison checks LMhdX
against both references at low and moderate Ha. Above Ha ≈ 30, the reference
used here is the independent spectral solve.
