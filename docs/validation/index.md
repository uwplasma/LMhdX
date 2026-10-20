# Validation

LMhdX separates numerical verification, physics validation, and external-code
comparison. A test passes only its stated claim; finite output alone is never a
validation result.

| Model | Required evidence | Current claim |
|---|---|---|
| Hartmann duct | analytical profile, charge closure, power balance, refinement | validated within documented mesh/tolerance gates |
| Shercliff and Hunt ducts | packaged benchmark values, symmetry, wall/interface current, mesh trends | validated within documented mesh/tolerance gates |
| High-$Ha$ fully developed flow | layer resolution, Richardson trend, integral balances | bounded accepted campaign cases |
| Duct through a fringe, inlet and outlet (`lmhdx.axial`) | flow rate, mass and charge to round-off, upstream fully developed gradient, buffer doubling, adjoint against central differences, ANL fringe on three meshes | inertialess limit only; ANL excess within 1 % of the layer-corrected core-flow model at c 0.1, Ha 2×10⁴ and 4–5 % at c 0.02 (row 24, met at c 0.1; both models are LMhdX's, so internal consistency, not validation); 1 % of TM-228's uncorrected 0.0178 at c 0.02 is not reachable on the 3-D core, a physical model difference (row 7, closed): TM-228's value is an inertialess thin-layer Ha → ∞ asymptote, and LMhdX's Ha → ∞ fits give 0.0183–0.0211 (3–18 % above it), a fit band, not a 1 % check; the ALEX B1 pipe (row 6) is not met station by station (weighted RMS 2.4, L∞ 5.1 against 1 and 2), and the inertia hypothesis for the downstream gap is untested |
| Locally fully developed station sum in a varying field (`validation.open_axis`) | 3-D open duct in a monotone 20 % ramp against the fully developed gradient integrated at the local field, at Ha 50 and 200, refined across and along the duct | square duct within 1 % up to γ√Ha ≈ 0.2, the design-law duct within 0.35 % up to 2; no collapse on γ√Ha across Ha ([duct design law](../tutorials/duct_design_law.md)) |
| Pipe through a fringe, inlet and outlet (`lmhdx.axial.solve_open_pipe`) | operator symmetry, energy identity, mass and charge to round-off, adjoint, ALEX B1 on refined meshes | Stokes limit; ALEX B1 upstream gradient and integrated excess (−3.3 %) met, station-wise tolerance not met ([numerics](../physics/numerics.md)) |
| Inertialess core-flow model (`lmhdx.coreflow`) | Walker's thin-wall limits at second order, symmetry, adjoint, ANL excess against TM-228 | ANL excess within 0.19 % of TM-228; ALEX B2 against experiment open (plan 4.3) |
| Heated conducting duct in a fringe, Mistrangelo et al. 2025 ([page](mistrangelo.md), plan 5.4, ADR 0009) | frozen specification with printed, inferred and pinned values; Case 1's developed section with the steel resolved on both axes | specification frozen; the section meets the printed gradient by construction (the wall's ratio is fitted to it), and its jets are close to Figure 13's (read by eye); the conjugate, buoyant, open-duct comparison is pending (5.4b–5.4e) |
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
$24\times24$ cross-section. Its analytical errors are 0.0064 in $L_2$ and
0.0098 in $L_\infty$, with steady residual and maximum current divergence
below $10^{-9}$ (the iterative residuals vary with platform). The
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
$-dP/dx$ with half-width, density, viscosity and conductivity one, the
normalisation of `lmhdx.core3d.duct_problem`. A1 is Shercliff's insulating
duct; A2 is Hunt's duct with Hartmann walls of conductance ratio $c=0.01$ and
insulating side walls. The reference is Table I's analytic column, which ships
with the package (`lmhdx/data/benchmarks/references/samper-table-i.toml`). It
has four digits, so its rounding is $6.5\times10^{-5}$ to $3.6\times10^{-4}$
relative, depending on the row.

All runs are float64 at the default tolerance $10^{-9}$ on the floor stack
(Python 3.10, JAX 0.6.2, SOLVAX 0.19.0). Meshes are `Ny:layer_y:Nz:layer_z`,
built with the fitted geometric `wall_resolving_faces`: `layer` cells inside
$1/Ha$ along the field ($y$) and inside $1/\sqrt{Ha}$ across it. Each series
scales all four numbers by 1.5, so the three meshes are one mapping at three
spacings. "Order" is the observed order of the three flow rates, and
"Richardson" is the extrapolated flow rate relative to the analytic one.

| Row | Ha | Meshes | Relative error, coarse / medium / fine | Order | Richardson |
|---|---|---|---|---|---|
| A1 | 500 | 32:4:32:4 / 48:6:48:6 / 72:9:72:9 | +2.20 % / +0.98 % / +0.43 % | 2.00 | $-1.5\times10^{-5}$ |
| A2 | 500 | 32:4:32:4 / 48:6:48:6 / 72:9:72:9 | +0.77 % / +0.36 % / +0.18 % | 1.96 | $+2.2\times10^{-4}$ |
| A1 | 5,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +0.91 % / +0.41 % / +0.18 % | 2.00 | $+1.4\times10^{-5}$ |
| A2 | 5,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +0.74 % / +0.34 % / +0.16 % | 2.01 | $+2.0\times10^{-4}$ |
| A1 | 10,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +1.04 % / +0.47 % / +0.22 % | 2.00 | $+1.4\times10^{-4}$ |
| A2 | 10,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +1.03 % / +0.46 % / +0.21 % | 2.02 | $+8.7\times10^{-5}$ |
| A1 | 15,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +1.10 % / +0.49 % / +0.22 % | 2.00 | $-1.5\times10^{-5}$ |
| A2 | 15,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +1.25 % / +0.55 % / +0.25 % | 2.02 | $+4.8\times10^{-5}$ |

**Gated, every row.** `tests/test_table_i.py` solves each row on its three
meshes and requires a monotone error, an observed order between 1.8 and 2.2, and
a Richardson value within validation row 27's tolerance of the analytic column:
$10^{-3}$ up to Ha 5,000 and $5\times10^{-3}$ above. Every row meets
$10^{-3}$, and five of the eight Richardson values lie inside the table's own
rounding. A solve takes 7–10 s warm on a CPU. The Ha 500 rows run on pull
requests (channel shard). The higher rows take 15–35 s each from a cold cache,
so they are marked `slow` and run on every push to `main`.

The error sits in the Hartmann layer. For A1 at Ha 500, refining only the field
direction (96:12:48:6) gives +0.27 %, and refining only the side direction
(48:6:96:12) gives +0.95 %. So the higher rows give the field direction
twice the cells.

**A1 at Ha 15,000 needs the singular mode held at zero.** The potential's
fast-diagonal solve is singular along each Neumann axis, and the constant mode's
eigenvalue there is zero only in exact arithmetic. Computed by `eigh`, it carries
the round-off of the largest eigenvalue: on 144:18 at Ha 15,000 the largest is
$7.3\times10^{12}$ and the null one comes out $7.7\times10^{-5}$. The core
current is a $1/Ha$ cancellation between $u\times B$ and $\nabla\phi$, and it
amplifies the resulting potential error into the flow rate. Since #183, `lmhdx.poisson` sets that eigenvalue to exactly zero.
Restoring the computed one reproduces the earlier failure: A1 at Ha 15,000 reads
−0.65 % on 144:18:72:9 (against +0.22 %) and +3.16 % on 128:16:64:8 (against
+0.27 %). A2 at Ha 15,000 moves only from +0.247 % to +0.242 % without the
fix. Its conducting walls take the thin-wall factorization, and why that route
is insensitive has not been established.

## Observed order and grid-convergence index (plan R2)

`validation/gci.py` computes the observed order, the Richardson value and the
fine-mesh grid-convergence index of Celik et al. 2008 (J. Fluids Eng. 130,
078001; the ASME V&V 20 procedure, safety factor 1.25) from three solutions.
`tests/test_table_i.py` checks it against Celik's own worked example (Table 1:
p 1.53, extrapolated 6.1685, GCI 2.2 %) and gates every Table I row with it.
`python -m validation.gci` reruns the campaign below. These are the existing
three-mesh sequences of the package's tests, run on the floor stack (Python 3.10,
JAX 0.6.2, SOLVAX 0.19.0) on 2026-10-15.

| Row | Case | Fine value | Reference | Fine error | Order | Richardson error | GCI fine |
|---|---|---|---|---|---|---|---|
| 3/4/27 | A1 Shercliff, Ha 500 | 0.0077133549 | 0.00768 | +4.34e-03 | 2.00 | -1.5e-05 | 5.4e-03 |
| 3/4/27 | A2 Hunt c 0.01, Ha 500 | 0.0014074591 | 0.001405 | +1.75e-03 | 1.96 | +2.2e-04 | 1.9e-03 |
| 3/4/27 | A1 Shercliff, Ha 5,000 | 0.00079163316 | 0.0007902 | +1.81e-03 | 2.00 | +1.4e-05 | 2.2e-03 |
| 3/4/27 | A2 Hunt c 0.01, Ha 5,000 | 1.9100796e-05 | 1.907e-05 | +1.61e-03 | 2.01 | +2.0e-04 | 1.8e-03 |
| 3/4/27 | A1 Shercliff, Ha 10,000 | 0.00039736142 | 0.0003965 | +2.17e-03 | 2.00 | +1.4e-04 | 2.5e-03 |
| 3/4/27 | A2 Hunt c 0.01, Ha 10,000 | 5.1796936e-06 | 5.169e-06 | +2.07e-03 | 2.02 | +8.6e-05 | 2.5e-03 |
| 3/4/27 | A1 Shercliff, Ha 15,000 | 0.00026537214 | 0.0002648 | +2.16e-03 | 2.00 | -1.5e-05 | 2.7e-03 |
| 3/4/27 | A2 Hunt c 0.01, Ha 15,000 | 2.4309909e-06 | 2.425e-06 | +2.47e-03 | 2.02 | +4.8e-05 | 3.0e-03 |
| 5 | Conducting pipe, Ha 20, c 0.1 | 0.015296353 | 0.0152869 (spectral) | +6.18e-04 | 2.00 | -9.0e-07 | 7.7e-04 |
| 7 | ANL fringe excess, core-flow model | 0.017809871 | 0.0178 (TM-228) | +5.55e-04 | 1.63 | -1.1e-04 | 8.3e-04 |

Refinement ratios: 1.5 in both cross-section directions for Table I (Ny:layer_y:Nz:layer_z
of the section above); 2 for the pipe (radial, azimuthal and layer cells doubled
together) and for the core-flow model (cross-section 10², 20², 40² and axial
spacing 0.2, 0.1, 0.05 on x ∈ [−10, 6]). In every row the analytic or published
value lies inside the fine mesh's GCI band, and the convergence is monotone.
Table I and the pipe show the formal second order. The core-flow excess shows
1.63, below its trapezoidal second order. The likely cause, not isolated, is the
1/B floor at 1000 that starts between nodes. Its Richardson value is 1.1e-4
from TM-228's three-digit 0.0178, whose own rounding is 2.8e-3.

Three meshes fit the order exactly, so they cannot by themselves show that a
sequence is in the asymptotic range. Here the evidence for that is the observed
order lying at the scheme's formal order. Not in this table:

- row 6 (the ALEX B1 open pipe), whose solve belongs to `lmhdx.axial` (its own
  lane) and takes hours;
- the 3-D fringe of rows 7 and 24, which has two meshes per Hartmann number
  (128² and 192², or 192² and 256²) and so has no observed order;
- row 8's manufactured solution on stretched meshes.

## Test gates

The portable suite includes analytical, manufactured, regression, physics, and
validation markers. Combined line/branch coverage must exceed 95%. Structural
3-D changes additionally run the reduced pinned FreeMHD Docker case before
acceptance.
