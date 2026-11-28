# LMhdX

[![PyPI version](https://img.shields.io/pypi/v/lmhdx.svg)](https://pypi.org/project/lmhdx/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/github/license/uwplasma/LMhdX)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/LMhdX/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/LMhdX/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/lmhdx/latest?label=docs)](https://lmhdx.readthedocs.io/)

**Differentiable liquid-metal MHD in JAX, for the ducts of fusion blankets.**

LMhdX solves liquid metal flowing through strong magnetic fields. It handles ducts and pipes with insulating or
conducting walls, three-dimensional ducts leaving a magnet, heat transfer through steel walls, and
quasi-two-dimensional vortex dynamics. Every result is differentiable end to end, so a design study can take
gradients through the solve. It runs on a CPU or a GPU, and [SOLVAX](https://github.com/uwplasma/SOLVAX) supplies
the solvers and implicit derivatives.

![Hunt duct side-layer jets from Ha 20 to 1000](docs/_static/hunt_side_layers.webp)

## Install and run

LMhdX needs Python 3.11, 3.12 or 3.13.

```console
pip install lmhdx            # JAX runs on the CPU; for a GPU follow the JAX installation guide
lmhdx examples/hartmann_case.toml
```

The same case from Python:

```python
import lmhdx

solution = lmhdx.solve(lmhdx.duct(100.0, 48, wall_conductance=0.027))
print(float(solution.velocity[0].data.max()))
```

`solve` takes a `Problem` and a `Solver`, and a TOML file builds the same objects. The
[first run](https://lmhdx.readthedocs.io/en/latest/getting_started/first_run.html) walks through it and checks the
answer against the exact profile.

## What it does

- **Fully developed ducts and pipes,** with thin or resolved conducting walls and wall stacks, and meshes chosen from
  the layer thicknesses `a/Ha` and `a/√Ha` rather than from a cell count.
- **Steady states as roots.** The steady solve is a differentiable root found by conjugate gradients or
  Newton-Krylov, not a march in time.
- **Three-dimensional and open ducts,** with an inlet and an outlet through a field that varies along them, with or
  without inertia (`method="auto"` picks Newton or marching the Runge-Kutta step).
- **Runs in time,** by `Solver(method="euler" | "rk3", dt=, steps=)`. An `Output` writes records, ParaView files and
  a summary, and a later run restarts from them bit for bit.
- **Heat transfer,** with conduction through solid walls, Joule heat and Boussinesq buoyancy.
- **Quasi-two-dimensional flow** with Hartmann-layer friction, by a spectral time step.

## Validated results

![Hartmann layers, flow-rate error and mesh convergence](docs/_static/validation_ladder.webp)

- **Ducts.** Hartmann profiles at Ha 20, 100 and 300 collapse onto `1 − e^{−ξ}`. The flow rate is within 0.4–2.3 %
  of [`validation/shercliff.py`](validation/shercliff.py) on a 48² mesh, converging at second order. That reference
  is a Chebyshev solve that shares no operator, mesh or solver with the package.
- **Pipes.** `Q/A = 1/8` at zero field (Hagen–Poiseuille), and within 0.04–0.43 % of
  [`validation/pipe.py`](validation/pipe.py) over Ha 0 to 100.
- **Blanket scale.** The Ha 1000 insulating duct is 0.5 % from the reference. All eight rows of Smolentsev et al.
  (2015) Table I (Ha 500 to 15,000) converge at second order, with the analytic value inside the GCI band.
- **Gradients.** Implicit adjoints agree with central differences to 7e-12 in the drive and 1.2e-10 in the field.
- **Heat transfer.** Nusselt numbers of rectangular ducts and Hartmann plates are within 0.35 %, and
  Chandrasekhar's critical Rayleigh number within 0.8 %. In Mistrangelo et al.'s six-code benchmark of a heated duct
  with conducting steel walls, Case 1 (buoyant, steady) is gated on two meshes and two outlet lengths and holds on
  6 of 11 figures. Case 2 (convection rolls) is reported, and lies in the codes' band once the flow carries a small
  disturbance ([benchmark record](https://lmhdx.readthedocs.io/en/latest/validation/mistrangelo.html)).

Still at the research stage, and not validated against a reference flow: time-dependent buoyant convection
(Mistrangelo's Case 2 is reported, not gated, and its rolls' onset depends on the disturbances present), the ALEX
B1/B2 benchmarks, and multi-device execution. The
[validation matrix](https://lmhdx.readthedocs.io/en/latest/validation/index.html) states each gate.

## Performance

Measured on one idle RTX A4000 (JAX 0.10.2, median of 12 runs; [ADR 0006](docs/adr/0006-review-2026-09-22.md)):

| 3-D core | 64³ | 128³ | 192³ | 256³ |
|---|---|---|---|---|
| float64-accurate (mixed), ms per step | 2.14 | 17.5 | 69.7 | 166 |
| true float32, ms per step | 0.515 | 4.84 | 19.3 | 48.0 |

256³ fits on one 16 GB card. At 128³ in float64 the GPU is 7.85 times faster than this code on XLA:CPU with 36 cores.
Two GPUs give the same answer bit for bit on the Q2D solve, with no speed-up yet.
`python scripts/run_benchmarks.py` reproduces the table, and the results are in
[`benchmarks/results`](benchmarks/results).

## Examples

| Command | Physics |
|---|---|
| `lmhdx examples/hartmann_case.toml` | Hartmann duct from a TOML file |
| `lmhdx examples/heated_duct.toml` | heated duct with buoyancy |
| `python examples/hunt_example.py` | conducting walls, side-layer jets |
| `python examples/fringe_duct_example.py` | duct leaving a magnet, its drop and the drop's field derivative |
| `python examples/open_duct_inertia_example.py` | inertial share of the fringe drop |
| `python examples/q2d_turbulence_demo.py` | Q2D vorticity evolution and energy decay |
| `python examples/duct_design_law.py` | optimum aspect ratio of an insulated duct |

[`examples/catalog.toml`](examples/catalog.toml) lists each example's parameters and evidence status, and
`python scripts/make_showcase_figures.py` regenerates the figures.

## Documentation

The [documentation](https://lmhdx.readthedocs.io/) has
[getting started](https://lmhdx.readthedocs.io/en/latest/getting_started/install.html) guides, step-by-step
[tutorials](https://lmhdx.readthedocs.io/en/latest/tutorials/fully_developed.html), the
[equations and numerical methods](https://lmhdx.readthedocs.io/en/latest/physics/numerics.html), the
[validation](https://lmhdx.readthedocs.io/en/latest/validation/index.html) record and the
[API reference](https://lmhdx.readthedocs.io/en/latest/reference/api.html).

## Citing

If you use LMhdX in published work, cite the release you used and the benchmark sources your case relies on.
[CITATION.cff](CITATION.cff) holds the details, and GitHub's "Cite this repository" button exports them.

```bibtex
@software{lmhdx,
  author = {Jorge, Rogerio},
  title  = {LMhdX: JAX-native inductionless liquid-metal magnetohydrodynamics},
  url    = {https://github.com/uwplasma/LMhdX},
  license = {MIT}
}
```

## Contributing

Install for development with `pip install -e ".[dev,docs]"`, then run the tests for your changes with
`python scripts/run_full_test_suite.py --changed-from HEAD`. [CONTRIBUTING.md](CONTRIBUTING.md) describes the
workflow, and the [plan](plan.md) shows what is in progress. Questions and bug reports go to the
[issue tracker](https://github.com/uwplasma/LMhdX/issues). LMhdX was called LMX before 1.5, and `import lmx` is now
`import lmhdx`.
