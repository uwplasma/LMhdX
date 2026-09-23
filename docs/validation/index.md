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

## Smolentsev et al. 2015 Table I on the staggered core (in progress, plan step 1.13)

Flow rate $\tilde Q=\int_{-1}^{1}\int_{-1}^{1}\tilde U\,dy\,dz$ for unit
$-dP/dx$ with half-width, density, viscosity and conductivity one, which is the
normalisation of `lmx.duct_problem`; $\tilde Q$ is four times the mean velocity
the steady tests compare. `wall_conductance` in `duct_problem` sets the two
walls normal to the field, so A2 (Hartmann walls $c=0.01$, insulating side
walls) is `duct_problem(..., wall_conductance=0.01)`. An exact series (Fourier
in the side-wall direction, closed form along the field) reproduces the
analytic column to all four printed digits and the spectral reference of
`validation/shercliff.py` to $5\times10^{-9}$.

Measured so far, float64, default tolerance, meshes `Ny:layer_y:Nz:layer_z`
(fitted geometric stretching, `layer` cells inside $1/Ha$ along the field and
$1/\sqrt{Ha}$ across it); relative error against the analytic column:

| Row | Ha | Mesh | Relative error | CG iterations |
|---|---|---|---|---|
| A1 | 500 | 32:4:32:4 / 48:6:48:6 / 72:9:72:9 | +2.20 % / +0.98 % / +0.43 % | — |
| A1 | 500 | 96:12:48:6 / 48:6:96:12 | +0.27 % / +0.95 % | — |
| A2 | 500 | 64:8:32:4 / 96:12:48:6 | +0.38 % / +0.18 % | 66 / 95 |
| A1 | 15,000 | 64:8:32:4 / 64:8:64:8 / 128:16:32:4 / 128:16:64:8 | +1.10 % / +1.08 % / +3.15 % / +3.16 % | 122 / 138 / 127 / 142 |
| A2 | 15,000 | 64:8:32:4 / 128:16:64:8 | +1.25 % / +0.33 % | 500 / 632 |

At Ha 500 the A1 series is second order (observed order 2.00 over ratio 1.5)
and its Richardson value is within $1\times10^{-5}$ of the analytic one; the
error lives in the Hartmann layer, not the side layer. At Ha 15,000 refining
the Hartmann direction makes A1 worse: the fast-diagonal potential solve loses
float64 accuracy on these meshes. The along-field eigenvalue of the constant
mode comes out $5.5\times10^{-6}$ instead of zero on 128:16 (smallest cell
$6.3\times10^{-7}$, largest eigenvalue $5.4\times10^{12}$), one direct solve
leaves a relative residual of $5.3\times10^{-3}$, and the core current is a
$1/Ha$ cancellation of $u\times B$ and $\nabla\phi$, so the error reaches the
flow rate amplified. The gate of validation row 27 is not yet assessed.

## Test gates

The portable suite includes analytical, manufactured, regression, physics, and
validation markers. Combined line/branch coverage must exceed 95%. Structural
3-D changes additionally run the reduced pinned FreeMHD Docker case before
acceptance.
