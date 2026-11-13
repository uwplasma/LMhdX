# LMhdX

[![PyPI version](https://img.shields.io/pypi/v/lmhdx.svg)](https://pypi.org/project/lmhdx/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/github/license/uwplasma/LMhdX)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/LMhdX/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/LMhdX/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/lmhdx/latest?label=docs)](https://lmhdx.readthedocs.io/)

**Differentiable inductionless liquid-metal MHD in JAX.** Documentation: <https://lmhdx.readthedocs.io/>

LMhdX solves liquid metal flowing through strong magnetic fields, the physics of fusion blanket channels:
ducts and pipes with insulating or thin conducting walls, three-dimensional ducts leaving a magnet, heat
transfer, and quasi-two-dimensional vortex dynamics. Everything is differentiable end to end; solvers and
implicit derivatives come from [SOLVAX](https://github.com/uwplasma/SOLVAX).

- **Resolve the layers:** meshes chosen from `a/Ha` and `a/√Ha`, not from a cell count.
- **Skip the transient:** the steady state is a differentiable root, not a march.
- **Check against something else:** an independent spectral solve that shares no code.
- **Run where you like:** CPU or GPU, one compiled trajectory per run.

## Install and first run

```console
pip install lmhdx            # JAX runs on the CPU; for a GPU follow the JAX installation guide
lmhdx examples/hartmann_case.toml
```

```python
import lmhdx

solution = lmhdx.solve(lmhdx.duct(100.0, 48, wall_conductance=0.027))
print(float(solution.velocity[0].data.max()))
```

`solve` takes a `Problem` and a `Solver`. A steady problem is solved by conjugate gradients or
Newton–Krylov; an open duct with advection by Newton or by marching the Runge–Kutta step (`method="auto"` picks
the faster); `Solver(method="euler" | "rk3", dt=, steps=)` runs an unsteady problem, and an `Output` writes
records, ParaView files and a summary that a later run restarts from bit for bit. A TOML file builds the
same objects as Python does.

## Validated results

![Hartmann layers, flow-rate error and mesh convergence](docs/_static/validation_ladder.webp)

- **Ducts:** Hartmann profiles at Ha 20, 100 and 300 collapse onto `1 − e^{−ξ}`; flow rate is within 0.4–2.3 %
  of [`validation/shercliff.py`](validation/shercliff.py), a Chebyshev solve sharing no operator, mesh or
  solver with the package, on a 48² mesh, at second order in the mesh.
- **Pipes:** `Q/A = 1/8` at zero field (Hagen–Poiseuille) and within 0.04–0.43 % of
  [`validation/pipe.py`](validation/pipe.py) over Ha 0 to 100.
- **Blanket scale:** Ha 1000 insulating duct 0.5 % from the reference; all eight rows of Smolentsev et al.
  (2015) Table I (Ha 500 to 15,000) at second order with the analytic value inside the GCI band.
- **Gradients:** implicit adjoints agree with central differences to 7e-12 in the drive and 1.2e-10 in the field.

![Hunt duct side-layer jets from Ha 20 to 1000](docs/_static/hunt_side_layers.webp)

Research stage, not validated against a reference flow: three-dimensional convective transport, heat
transfer (though its Nusselt numbers and Chandrasekhar's critical Rayleigh number are matched within 0.35 % and
0.8 %), the fringe in the inertialess limit only, the ALEX B1/B2 benchmarks, and multi-device execution. The
[validation matrix](https://lmhdx.readthedocs.io/en/latest/validation/index.html) states each gate.

## Performance

Measured on one idle RTX A4000 (JAX 0.10.2, median of 12 runs; [ADR 0006](docs/adr/0006-review-2026-09-22.md)):

| 3-D core | 64³ | 128³ | 192³ | 256³ |
|---|---|---|---|---|
| float64-accurate (mixed), ms per step | 2.14 | 17.5 | 69.7 | 166 |
| true float32, ms per step | 0.515 | 4.84 | 19.3 | 48.0 |

256³ fits on one 16 GB card. Against this code on XLA:CPU (36 cores, float64, 128³) the GPU is 7.85× faster.
Two GPUs give the same answer bit for bit on the Q2D solve and no speed-up yet. Reproduce with
`python scripts/run_benchmarks.py`; results are in [`benchmarks/results`](benchmarks/results).

## Examples

| Command | Physics |
|---|---|
| `lmhdx examples/hartmann_case.toml` | Hartmann duct from a TOML file |
| `lmhdx examples/heated_duct.toml` | heated duct with buoyancy |
| `python examples/hunt_example.py` | conducting walls, side-layer jets |
| `python examples/fringe_duct_example.py` | duct leaving a magnet, drop and its field derivative |
| `python examples/open_duct_inertia_example.py` | inertial share of the fringe drop |
| `python examples/q2d_turbulence_demo.py` | Q2D vorticity evolution and energy decay |
| `python examples/duct_design_law.py` | optimum aspect ratio of an insulated duct |

Parameters and evidence status are in [`examples/catalog.toml`](examples/catalog.toml);
`python scripts/make_showcase_figures.py` regenerates the figures.

## Cite and contribute

Cite the commit or release you used ([CITATION.cff](CITATION.cff)). Development: `pip install -e ".[dev,docs]"`,
then `python scripts/run_full_test_suite.py --changed-from HEAD`; see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[plan](plan.md). LMhdX was called LMX before 1.5: `import lmx` is now `import lmhdx`.
