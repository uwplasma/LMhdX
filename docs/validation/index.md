# Validation

LMhdX separates numerical verification, physics validation, and external-code
comparison. A test passes only its stated claim; finite output alone is never a
validation result.

| Model | Required evidence | Current claim |
|---|---|---|
| Hartmann duct | analytical profile, charge closure, power balance, refinement | validated within documented mesh/tolerance gates |
| Shercliff and Hunt ducts | packaged benchmark values, symmetry, wall/interface current, mesh trends | validated within documented mesh/tolerance gates |
| High-$Ha$ fully developed flow | layer resolution, Richardson trend, integral balances | bounded accepted campaign cases |
| Duct through a fringe, inlet and outlet (`lmhdx.axial`) | flow rate, mass and charge to round-off, upstream fully developed gradient, buffer doubling, adjoint against central differences, ANL fringe on three meshes | gated in the Stokes limit; with inertia (Newton) the share at N 1000 is 0.4–2.3 %, reported ([numerics](../physics/numerics.md)); ANL excess within 1 % of the layer-corrected core-flow model at c 0.1, Ha 2×10⁴ (row 24, met); 1 % of TM-228's uncorrected 0.0178 at c 0.02 is not reachable on the 3-D core, a physical model difference (row 7, closed) |
| Locally fully developed station sum in a varying field (`validation.open_axis`) | 3-D open duct in a monotone 20 % ramp against the fully developed gradient integrated at the local field, at Ha 50 and 200, refined across and along the duct | square duct within 1 % up to γ√Ha ≈ 0.2, the design-law duct within 0.35 % up to 2; no collapse on γ√Ha across Ha ([duct design law](../tutorials/duct_design_law.md)) |
| Pipe through a fringe, inlet and outlet (`lmhdx.axial.solve_open_pipe`) | operator symmetry, energy identity, mass and charge to round-off, adjoint, ALEX B1 on refined meshes | Stokes limit; ALEX B1 upstream gradient and integrated excess (−3.3 %) met, station-wise tolerance not met ([numerics](../physics/numerics.md)) |
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
$24\times24$ cross-section. Its analytical errors are 0.0064 in $L_2$ and
0.0098 in $L_\infty$, with steady residual $4.2\times10^{-10}$ and maximum
current divergence $3.9\times10^{-10}$. The
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

## Test gates

The portable suite includes analytical, manufactured, regression, physics, and
validation markers. Combined line/branch coverage must exceed 95%. Structural
3-D changes additionally run the reduced pinned FreeMHD Docker case before
acceptance.
