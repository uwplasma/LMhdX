# Validation

LMhdX separates numerical verification, physics validation, and external-code
comparison. A test passes only its stated claim; finite output alone is never a
validation result.

| Model | Required evidence | Current claim |
|---|---|---|
| Hartmann duct | analytical profile, charge closure, power balance, refinement | validated within documented mesh/tolerance gates |
| Shercliff and Hunt ducts | packaged benchmark values, symmetry, wall/interface current, mesh trends | validated within documented mesh/tolerance gates |
| High-$Ha$ fully developed flow | layer resolution, Richardson trend, integral balances | bounded accepted campaign cases |
| Duct through a fringe, inlet and outlet (`lmhdx.axial`) | flow rate, mass and charge to round-off, upstream fully developed gradient, buffer doubling, adjoint against central differences, ANL fringe on three meshes | inertialess limit only; ANL excess within 1 % of the layer-corrected core-flow model at c 0.1, Ha 2×10⁴ (row 24, met); 1 % of TM-228's uncorrected 0.0178 at c 0.02 is not reachable on the 3-D core, a physical model difference (row 7, closed) |
| Inertialess core-flow model (`lmhdx.coreflow`) | Walker's thin-wall limits at second order, symmetry, adjoint, ANL excess against TM-228 | ANL excess within 0.19 % of TM-228; ALEX B2 against experiment open (plan 4.3) |
| Periodic Q2D | analytical decay, energy identity, spectral incompressibility, spatial refinement, CPU/GPU parity | verified for the documented SM82 model and numerical gates |

For 2-D cases, `validation_summary` reports convergence, current continuity,
gauge, interface, flow, and profile metrics. `hartmann_validation` compares
the computed profile with the analytical Hartmann solution.

For a duct with an inlet and an outlet, `lmhdx.axial` reports the flow rate
through every station, the mass and charge balance of every cell, and the
certified residual of the solve. Mesh and buffer independence remain separate
required checks.

The benchmark specifications and reference arrays shipped under `src/lmhdx/data/benchmarks`
are versioned package data. `benchmarks/provenance.json` records bibliographic
sources and executable test/workflow links.

## Quantitative evidence

The CI-executed `examples/hartmann_example.py` case uses $Ha=20$ on a
$24\times24$ cross-section. Its analytical errors are 0.02276 in $L_2$ and
0.06204 in $L_\infty$, with charge-balance residual
$4.24\times10^{-19}$ and final velocity update $9.48\times10^{-9}$. The
documented profile-error limits are 0.05 and 0.10.

The weekly FreeMHD workflow runs the frozen B2 inputs in the pinned FreeMHD
image and LMhdX's inertialess core-flow model on the same measured field, and
gates each code's own execution. Their transverse pressure difference is
reported, not gated: FreeMHD runs a two-update transient smoke from a uniform
plug, and the core-flow model is the steady inertialess limit. The
[FreeMHD guide](freemhd.md) states what the record contains.

The Q2D Taylor--Green case matches its exact viscous/Hartmann decay, and a
nonlinear three-grid test compares $12^2$ and $24^2$ solutions with a $48^2$
reference. On a $256^2$, 80-step float32 workload run with JAX 0.6.2, the final
CPU and RTX A4000 fields agree to relative $L_2=2.38\times10^{-6}$. This is a
backend-parity result, not external physics validation. Its GPU speed-up is not
quoted. G4 is stated as absolute throughput (ADR 0006, D23): on one RTX A4000
with matmul precision `highest` (plan step 2.1), the $128^3$ 3-D core takes
17.5 ms per step float64-accurate (mixed precision, 8.4 ns per cell per step)
and 4.84 ms in true float32. Against the same JAX code on XLA:CPU on the host's
36 cores in float64 (138 ms, controlled 2026-09-14 rows) that is 7.85x; the 10x
float64 target on an A4000 is withdrawn. The two CPU reports in
`benchmarks/results` are from the uncontrolled 2026-09-07 run and record no
matmul precision; no ratio is quoted from them.

## Smolentsev et al. 2015 Table I on the staggered core (plan step 1.13)

Flow rate $\tilde Q=\int_{-1}^{1}\int_{-1}^{1}\tilde U\,dy\,dz$ for unit
$-dP/dx$ with half-width, density, viscosity and conductivity one, which is the
normalisation of `lmx.duct_problem`; $\tilde Q$ is four times the mean velocity
the steady tests compare. `wall_conductance` in `duct_problem` sets the two
walls normal to the field, so A2 (Hartmann walls $c=0.01$, insulating side
walls) is `duct_problem(..., wall_conductance=0.01)`. The reference is Table I's
analytic column, which has four digits: its rounding is $6\times10^{-5}$ to
$3.6\times10^{-4}$ relative, depending on the row.

All runs are float64 at the default tolerance $10^{-9}$, on JAX 0.6.2 (the CI
floor stack). Meshes are `Ny:layer_y:Nz:layer_z`, built with the fitted
geometric `wall_resolving_faces`: `layer` cells inside $1/Ha$ along the field
($y$) and inside $1/\sqrt{Ha}$ across it. Each series scales all four numbers
by 1.5, so the three meshes are one mapping at three spacings. The order is the
observed order of the three flow rates, and "Richardson" is the extrapolated
flow rate relative to the analytic one.

| Row | Ha | Meshes | Relative error, coarse / medium / fine | Order | Richardson | CG iterations | Verdict |
|---|---|---|---|---|---|---|---|
| A1 | 500 | 32:4:32:4 / 48:6:48:6 / 72:9:72:9 | +2.20 % / +0.98 % / +0.43 % | 2.00 | $-1.5\times10^{-5}$ | 49 / 62 / 66 | gated |
| A2 | 500 | 32:4:32:4 / 48:6:48:6 / 72:9:72:9 | +0.77 % / +0.36 % / +0.18 % | 1.96 | $+2.2\times10^{-4}$ | 53 / 69 / 79 | gated |
| A1 | 5,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +0.91 % / +0.41 % / +0.22 % | 2.44 | $+1.1\times10^{-3}$ | 124 / 180 / 307 | reported |
| A2 | 5,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +0.74 % / +0.34 % / +0.16 % | 2.01 | $+2.0\times10^{-4}$ | 301 / 370 / 432 | reported |
| A1 | 10,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +1.04 % / +0.47 % / +0.23 % | 2.16 | $+6.3\times10^{-4}$ | 132 / 210 / 426 | reported |
| A2 | 10,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +1.03 % / +0.46 % / +0.20 % | 1.88 | $-3.5\times10^{-4}$ | 393 / 574 / 704 | reported |
| A1 | 15,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +1.10 % / +0.49 % / −0.65 % | none | none | 147 / 277 / 522 | **not converging** |
| A2 | 15,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +1.25 % / +0.56 % / +0.24 % | 1.90 | $-3.3\times10^{-4}$ | 511 / 784 / 1,006 | reported |

**Gated.** `tests/test_table_i.py` (channel shard) solves both Ha 500 rows on
their three meshes and requires a monotone error, an observed order between 1.8
and 2.2, and a Richardson value within $10^{-3}$ of Table I (validation row 27).
Both measured Richardson values lie inside the table's own rounding. The error
is in the Hartmann layer: for A1, refining only the field direction
(96:12:48:6) gives +0.27 %, refining only the side direction (48:6:96:12)
gives +0.95 %.

**Reported, not gated.** The Ha 5,000 and 10,000 rows fall monotonically at
close to second order, but A1 at Ha 5,000 has not reached the asymptotic range
(order 2.44), and its Richardson value is just outside row 27's $10^{-3}$. These
rows cost 30–75 s per solve on a loaded laptop, too much for a pull-request
shard. A2 at Ha 15,000 converges on this series, too.

**A1 at Ha 15,000 does not converge with refinement.** On the series above, the
fine-mesh step moves the flow rate by 1.1 %, twice as far as the medium step
did, and it moves it past the analytic value. Refining further along the field
makes it worse: 128:16:32:4 gives +3.15 % and 128:16:64:8 gives +3.16 %, while
64:8:32:4 gives +1.10 %. The evidence points to float64 conditioning of the
fast-diagonal potential solve rather than to layer resolution:

- On 128:16 the smallest cell is $6.3\times10^{-7}$ and the largest
  along-field eigenvalue is $5.4\times10^{12}$.
- The constant (Neumann null) mode's along-field eigenvalue comes out
  $5.5\times10^{-6}$ instead of zero. On 64:8 it is $7\times10^{-11}$.
- One direct solve leaves a relative residual of $5.3\times10^{-3}$. At Ha 500
  on 48:6, the figure is $1.4\times10^{-7}$.
- The core current is a $1/Ha$ cancellation between $u\times B$ and $\nabla\phi$,
  so the potential error reaches the flow rate amplified.

The pressure solve is protected by the double projection in `steady_residual`,
but the potential solve has no such correction. Two float64 refinement sweeps of
the direct solve leave the 64:8:32:4 results unchanged (A1 +1.1003 %, A2
+1.2526 %), so the coarse-mesh error is discretisation. The same sweeps were not
completed on the fine meshes. Why A2 converges at the same Hartmann number while
A1 does not has not been established. At tolerance $10^{-11}$, the A1
128:16:64:8 solve does not converge at all.

The confirming run, two refinement sweeps on 128:16:64:8, was not finished.
The fix is plan step 2b.3: refine the potential solve against the exact face
stencil. Once it lands, re-measure A1 at Ha 10,000 and 15,000 and gate the
remaining rows. The A1 Ha 15,000 row of validation row 27 stays open until then.

## Test gates

The portable suite includes analytical, manufactured, regression, physics, and
validation markers. Combined line/branch coverage must exceed 95%. Structural
3-D changes additionally run the reduced pinned FreeMHD Docker case before
acceptance.
