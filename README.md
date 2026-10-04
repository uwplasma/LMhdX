# LMhdX

[![PyPI version](https://img.shields.io/pypi/v/lmhdx.svg)](https://pypi.org/project/lmhdx/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/github/license/uwplasma/LMhdX)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/LMhdX/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/LMhdX/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/lmhdx/latest?label=docs)](https://lmhdx.readthedocs.io/)

**Differentiable inductionless liquid-metal MHD in JAX.**
Documentation: **<https://lmhdx.readthedocs.io/>** · Install: `pip install lmhdx`

LMhdX solves the flow of liquid metals in strong magnetic fields — the physics of
fusion blanket channels. Ducts and pipes with insulating or thin conducting
walls, three-dimensional ducts with an inlet and an outlet leaving a magnet, and
quasi-two-dimensional vortex dynamics, all differentiable end to end.
Reusable solvers and implicit derivatives come from
[SOLVAX](https://github.com/uwplasma/SOLVAX).

- **Resolve the layers:** meshes chosen from `a/Ha` and `a/√Ha`, not from a cell count.
- **Skip the transient:** the steady state as a differentiable root, not a march.
- **Differentiate the continuous inputs:** drive and field strength on the duct solves, and the field strength across a fringe.
- **Check against something else:** an independent spectral solve that shares no code.
- **Run where you like:** CPU or GPU, one compiled trajectory per run.

![Quasi-2D MHD turbulence](docs/_static/q2d_turbulence_256.webp)

*Decaying quasi-2D MHD turbulence with Hartmann-layer friction — 256², 3,000 steps, about 20 s on a laptop CPU with `python scripts/make_showcase_figures.py --only q2d`.*

Scope: fully developed duct and pipe flows are validated; three-dimensional
convective transport, the fringe in the inertialess limit, the ALEX benchmarks
and multi-device execution are research stage (see [What is validated, what is research](#what-is-validated-what-is-research)).

## Installation

```console
pip install lmhdx
```

LMhdX was called LMX before version 1.5: `import lmx` is now `import lmhdx`.
JAX runs on the CPU by default; install the GPU wheel from the
[JAX guide](https://docs.jax.dev/en/latest/installation.html) and LMhdX uses it.
From source:

```console
git clone https://github.com/uwplasma/LMhdX.git
cd LMhdX
pip install ".[visualization]"
lmhdx examples/hartmann_case.toml
```

The [installation guide](https://lmhdx.readthedocs.io/en/latest/getting_started/install.html)
covers extras and GPUs.

## First run: a duct in three lines

```python
import lmhdx

problem = lmhdx.duct_problem(hartmann=100.0, cells=48, wall_conductance=0.027)
solution = lmhdx.solve(problem)
print(float(solution.velocity[0].data.max()))
```

`duct_problem` picks both transverse meshes from the layers the Hartmann number
implies — `a/Ha` against the walls normal to the field, `a/√Ha` against the
others — so the answer is converged rather than merely computed. `solve` finds
the steady state by preconditioned conjugate gradients, or by matrix-free
Newton–Krylov when advection or a conducting wall makes the problem
nonsymmetric; neither stores more than a restart cycle of vectors. The duct is
solved in float64, which `duct_problem` turns on.
The same case runs from a TOML file with `lmhdx examples/hartmann_case.toml`.

## Documentation

[Install](https://lmhdx.readthedocs.io/en/latest/getting_started/install.html) ·
[First run](https://lmhdx.readthedocs.io/en/latest/getting_started/first_run.html) ·
[Tutorials](https://lmhdx.readthedocs.io/en/latest/tutorials/fully_developed.html) ·
[Equations](https://lmhdx.readthedocs.io/en/latest/physics/equations.html) ·
[Validation](https://lmhdx.readthedocs.io/en/latest/validation/index.html) ·
[API](https://lmhdx.readthedocs.io/en/latest/reference/api.html) ·
[Roadmap](plan.md)

## Fully developed ducts and pipes

### Duct flows against an independent reference

![Hartmann layers, flow-rate error and mesh convergence](docs/_static/validation_ladder.webp)

```console
python scripts/make_showcase_figures.py --only ladder
```

- Hartmann profiles at Ha 20, 100 and 300 **collapse onto `1 − e^{−ξ}`** when
  plotted against the wall distance in layer widths; the points are a spectral
  solve, the lines are LMhdX.
- Flow rate within **0.4 – 2.3 %** on a fixed 48² mesh, across insulating walls
  and wall conductance 0.027 and 0.1.
- Second order in the mesh, so the error at a fixed mesh growing with the field
  is a resolution statement rather than a model one.
- The reference, [`validation/shercliff.py`](validation/shercliff.py), is
  Chebyshev collocation of the governing system converged to eight digits. It
  shares no operator, mesh or solver with the package, and at zero field it
  returns the analytic Poiseuille maximum `0.29468541`.

### Side layers at blanket-scale Hartmann numbers

![Hunt duct side-layer jets from Ha 20 to 1000](docs/_static/hunt_side_layers.webp)

```console
python examples/hunt_example.py
```

- Conducting Hartmann walls drive **jets in the side layers** that carry a
  growing share of the flow as the field rises.
- The jet maximum tracks `Ha^{−1/2}`, the side-layer thickness, over Ha 20 → 1000.
- The same steady solver reaches **Ha 1000** in the insulating duct, 0.5 % from
  the spectral reference on a wall-resolving 64² mesh.

### Pipes

![Pipe profiles, cross-section and flow rate against Hartmann number](docs/_static/pipe_flow.webp)

```console
python scripts/make_showcase_figures.py --only pipe
```

- A polar grid with the metric in `lmhdx.grid`, so the same flux-form operators
  solve a circular pipe. The axis needs no condition: the face at `r = 0` has
  zero area.
- The potential Poisson still factorizes exactly — a Fourier transform in the
  azimuth leaves each mode separable in `(r, z)`.
- `Q/A = 1/8` at zero field, the exact Hagen–Poiseuille value, at second order;
  `Q/A ∝ Ha^{−1}` once the field takes over.
- Within **0.04 – 0.43 %** of [`validation/pipe.py`](validation/pipe.py) at Ha 0
  to 100 — a Fourier–Chebyshev solve on the diameter, which removes the axis
  singularity by construction rather than treating it.

## A duct leaving a magnet

```console
python examples/fringe_duct_example.py
```

- `lmhdx.axial` gives the duct an inlet and an outlet: the inlet carries
  LMhdX's own fully developed profile at the imposed flow rate, the outlet a
  zero gradient and `p = 0`, and `fringe_duct` builds the ANL fringe
  (ANL/FPP/TM-228) with uniform-field and field-free buffers.
- The flow rate through every station, mass and charge hold to round-off; the
  upstream gradient is the fully developed one; the pressure drop across the
  fringe and its derivative with respect to the field strength are outputs.
- Research stage: only the inertialess (Stokes-limit) flow is solved. On the ANL
  case the excess drop falls as Ha^-0.35 from Ha 100 to 3,200 and extrapolates
  6–11 % below the core-flow model's; the 1 % gate needs Ha ≥ 10⁴. The
  [fringe tutorial](https://lmhdx.readthedocs.io/en/latest/tutorials/fringing.html)
  walks through it.

## Design with gradients

```python
import jax, jax.numpy as jnp, lmhdx

lmhdx.enable_x64()
problem = lmhdx.duct_problem(hartmann=20.0, cells=24)

def throughput(drive, field_scale):
    solution = lmhdx.solve_steady_state(problem, forcing=(drive, 0.0, 0.0), field_scale=field_scale)
    return jnp.mean(solution.velocity[0].data)

print(jax.grad(throughput, argnums=(0, 1))(1.0, 1.0))
```

- One adjoint solve at the root, through the implicit function theorem — not a
  tape of the iteration.
- Agrees with central differences to **7e-12** in the drive and **1.2e-10** in
  the field scale on the Ha ≤ 5 test ducts, where the test gate is 1e-6.
- `solve_steady_state` and `solve_fully_developed_fields` differentiate the drive
  and the field scale, and `lmhdx.axial.solve_open_duct` the field scale across
  a fringe (`python examples/fringe_duct_example.py` checks it against central
  differences).
- A solve that stops short raises, rather than returning a plausible field and a
  gradient taken away from a root.

## Quasi-two-dimensional turbulence

![Q2D turbulence snapshots and energy spectrum](docs/_static/q2d_turbulence_poster.webp)

```console
python examples/q2d_turbulence_demo.py
```

- Vortex merging under Hartmann friction. The spectrum panel draws `k^{−3}` as a
  guide line, not a fitted slope.
- Energy and enstrophy budget identities checked on every run.
- The figures above come from `python scripts/make_showcase_figures.py --only q2d`:
  256² for 3,000 steps in about 20 s on a laptop CPU. The demo command runs 64²
  for 160 steps.

## Performance

![Time per step against problem size, CPU and GPU, both precisions](docs/_static/device_scaling.webp)

*The figure is from the uncontrolled 2026-09-07 run and is not yet redrawn from the tables below.*

```console
python scripts/run_benchmarks.py --output benchmarks/results/mine.json
python scripts/make_showcase_figures.py --only scaling
```

G4 is stated as absolute throughput ([ADR 0006](docs/adr/0006-review-2026-09-22.md), D23):
milliseconds per step and nanoseconds per cell per step, with the float64-accurate
mode (mixed precision) and true float32 reported separately. Measured on one idle
RTX A4000 (JAX 0.10.2, matmul precision `highest`, median of 12 timed runs;
[plan](plan.md) step 2.1):

| 3-D core, one A4000 | 64³ | 128³ | 192³ | 256³ |
|---|---|---|---|---|
| float64-accurate (mixed), ms per step | 2.14 | 17.5 | 69.7 | 166 |
| ns per cell per step | 8.2 | 8.4 | 9.8 | 9.9 |
| true float32, ms per step | 0.515 | 4.84 | 19.3 | 48.0 |
| ns per cell per step | 2.0 | 2.3 | 2.7 | 2.9 |

- **256³ fits on one 16 GB card** in every mode. Q2D at 2048² takes 75.9 ms per
  step in float64 and 16.0 ms in true float32.
- **Same-code CPU/GPU ratio:** against this JAX code on XLA:CPU on the host's 36
  cores in float64 (138 ms per step at 128³, controlled 2026-09-14 rows), the GPU's
  mixed mode is 7.85× faster. The baseline is XLA:CPU running this code, not a tuned
  CPU solver. The 10× float64 target on an A4000 is withdrawn: GA10x runs float64
  at 1/64 of its float32 rate, which bounds a fair single-card float64 speed-up
  near the memory-bandwidth ratio.
- **CPU reports:** `benchmarks/results/cpu-*.json` are from the 2026-09-07
  run, taken without load control or a recorded matmul precision; no ratio is
  quoted from them.
- **Trajectory-length scaling:** per step, 80 steps against 20 cost 0.83 in float64
  and 0.92 in float32; this timing ratio alone does not establish absence of host
  synchronization.
- Every number carries an `accepted` flag judged against the precision it was
  computed in; a run that lost its divergence-free constraint is reported, not quoted.

Two GPUs give the **same answer bit for bit** on the Q2D solve, and no speed-up:
the strong-scaling efficiency of an unaided placement is 0.20 in float64 and
0.10 in float32 at 2048², because the transforms all-gather every step across
PCIe. Correct, not yet faster — the numbers are in
[`benchmarks/results`](benchmarks/results) and the next step is in the [plan](plan.md).

## What is validated, what is research

- **Validated:** Hartmann, Shercliff and Hunt ducts against an independent
  spectral solve and against analytical profiles; the pipe against a second,
  independent spectral solve over Ha 0 to 100; implicit adjoints against finite differences;
  the steady mechanical power balance within a 1e-10 relative test gate (measured
  3.6e-14 insulating, 6.3e-14 at wall conductance 0.027); Q2D decay identities.
- **Research stage:** three-dimensional convective transport (`advection="central"`
  or `"limited"`, from `lmhdx.ops`) is tested for conservation, order and
  boundedness but not validated against a reference flow, the fringe is solved
  in the inertialess limit only, the ALEX B1/B2 benchmarks are open, and
  multi-device execution is not yet established. The
  [validation matrix](https://lmhdx.readthedocs.io/en/latest/validation/index.html)
  and the [plan](plan.md) state each gate.

## Comparison with other codes

| Comparison | What it establishes | Status |
|---|---|---|
| [`validation/shercliff.py`](validation/shercliff.py) spectral solve | Duct flow rates, insulating and Hunt walls, Ha 0 → 1000 | independent of the package; 0.4 – 2.3 % on the meshes above |
| [`validation/pipe.py`](validation/pipe.py) spectral solve | Pipe flow rates, insulating and conducting walls, Ha 0 → 100 | independent of the package; 0.04 – 0.43 % |
| Analytic Hartmann, Shercliff, Hunt and Poiseuille | Profiles and flow rates in every limit that has a closed form | `python examples/hartmann_example.py` |
| FreeMHD (OpenFOAM `epotFoam`), pinned [`freemhd_install`](https://github.com/rogeriojorge/freemhd_install) image, B2 case | Both codes run the frozen B2 inputs weekly ([`validation/freemhd.py`](validation/freemhd.py)); LMhdX through the inertialess core-flow model | executions gated; the cross-code pressure comparison is reported, not gated — **not** a production result |
| ALEX B1 pipe and B2 square duct experiments | Fringing-field pressure drop | production acceptance **open**; specs and digitised references are frozen in [`src/lmhdx/data/benchmarks`](src/lmhdx/data/benchmarks) |

The [validation record](https://lmhdx.readthedocs.io/en/latest/validation/index.html)
states each gate and what it does not cover.

## Examples

| Command | Physics |
|---|---|
| `lmhdx examples/hartmann_case.toml` | Hartmann duct from a TOML file, terminal diagnostics |
| `python examples/hartmann_example.py` | analytical error, conservation, mesh convergence |
| `python examples/hunt_example.py` | conducting walls, prescribed throughput and hydraulic power |
| `python examples/li_aln_wall_stack_example.py` | explicit wall material layers and interface currents |
| `python examples/fringe_duct_example.py` | 3-D duct leaving a magnet, drop and its field derivative |
| `python examples/q2d_turbulence_demo.py` | Q2D vorticity evolution, energy decay, movie |

Each example is one editable file that writes to `artifacts/examples/`;
parameters and evidence status are in [`examples/catalog.toml`](examples/catalog.toml).
`python scripts/make_showcase_figures.py` regenerates every figure above.

## Cite and contribute

Cite the commit or release you used; metadata is in [CITATION.cff](CITATION.cff).
Development: `pip install -e ".[dev,docs]"`, then
`python scripts/run_full_test_suite.py --changed-from HEAD`. See
[CONTRIBUTING.md](CONTRIBUTING.md).
