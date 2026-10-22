# MHD Nusselt campaign

Shercliff's square duct has insulating walls, a field along $y$ and no Joule
heat. Ha is on the half-width $a$, and $D_h=2a$. Its developed Nusselt numbers
are compared against two references:
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

**Two kinds of statement.** LMhdX against the spectral solve is verification:
one point is gated in `tests/test_steady.py`, 48 cells at Ha 10 within 0.3 % of
the spectral solve in each condition. The published curves are a reading from a
figure, reported with its uncertainty. `python -m validation.mhd_nusselt`
regenerates every number on this page on demand, in about ten minutes on a
laptop CPU.

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
- At Ha 0 the spectral solve gives the double-sine series in H1 and H2, and
  2.97753 in T, where Shah & London list 2.976.
- All three numbers climb toward their slug-flow limits, 7.114, 6 and
  $\pi^2/2=4.935$, as the core flattens. The side layers, of thickness
  $\propto Ha^{-1/2}$, keep the approach slow.

## The published curves

Bluck, Wolfendale & Marquis (2015) is an analytical series for the same duct.
Their square-duct curves in Figs. 4 (H1) and 6 (H2) were read from the vector
paths of the accepted manuscript.
- **How they were read.** Each curve's vertices sit on Ha $=10^{k/4}$, so they
  are the authors' computed points, mapped through the figures' own tick marks.
  The readings, the calibration and the manuscript's SHA-256 are in
  `src/lmhdx/data/benchmarks/references/bluck2015-shercliff-nusselt.json` and
  `benchmarks/provenance.json`.
- **A check.** The H1 curve's plateau reads 3.60678, against their tabulated
  Ha 0 value of 3.6068. The H2 plateau reads 3.0824, against their 3.0871.

The two figures use different Ha conventions, by exactly the factor of 2
between the half-width and the full width:
- **H1**, Figure 4, is on the half-width. It is then within 0.03 % of the
  spectral value at Ha 1 and 0.07 % at Ha 10, falling to 0.19 % low at
  Ha 31.6, 0.49 % at 100 and 1.2 % at 316. On $D_h$ it would lie 3–12 % above.
- **H2**, Figure 6, is on the full width, $D_h=2a$, as their eq. (8) defines
  it. It is then
  0.17 % low at an equivalent Ha 0.5, and 0.22 %, 0.43 %, 1.0 % and 2.0 % low at
  5, 15.8, 50 and 158. On the half-width it would lie up to 10 % below.

Read on the other scale, each would be off by 3–12 %, so the factor of 2 is
clear from the fit, though the paper does not say it of Figure 4. **The
reading's own uncertainty** comes from the figures, not from digitizing, since
the vertices are the authors' points. Figure 4's plateau equals their table to
1e-5, and Figure 6's lies 0.15 % below theirs, so the H2 percentages carry
about 0.15 %.

Both curves fall below the reference as Ha grows. The authors note that their
series' hyperbolic terms overflow double precision at large Ha, and they
regroup them through logarithms; the error left there is not quantified. The
comparison anchors LMhdX against both references at low and moderate Ha. Above
Ha ≈ 30, the independent spectral solve, not the published series, is the
reference.
