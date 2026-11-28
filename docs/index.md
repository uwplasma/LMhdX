# LMhdX

LMhdX simulates liquid metal flowing through strong magnetic fields, the
physics of fusion blanket channels. It solves fully developed ducts and pipes
against analytical references, three-dimensional ducts and pipes, ducts with an
inlet and an outlet in a field that changes along them, heat transfer, and a
depth-averaged (Q2D) model with Hartmann-layer damping. It is written in JAX,
so solutions can be differentiated and run on a CPU or a GPU. LMhdX builds the
physics; [SOLVAX](https://github.com/uwplasma/SOLVAX) supplies the numerical
solvers.

```{image} _static/validation_ladder.webp
:alt: Hartmann-layer collapse, flow-rate error against a spectral reference, and mesh convergence
:align: center
```

After `python -m pip install lmhdx`, a first duct takes three lines:

```python
import lmhdx

solution = lmhdx.solve(lmhdx.duct(20.0, 24))   # Hartmann number 20, 24 x 24 cells
print(float(solution.flow_rate))               # 0.15614...
```

The model is inductionless: the flow does not change the applied field.

## Start here

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Install and run
:link: getting_started/install
:link-type: doc
Install LMhdX, select a JAX backend, and check the command line.
:::

:::{grid-item-card} First duct solve
:link: getting_started/first_run
:link-type: doc
Solve and validate a Hartmann duct from Python or TOML.
:::

:::{grid-item-card} A duct leaving a magnet
:link: tutorials/fringing
:link-type: doc
Solve the flow through the ANL fringe with an inlet and an outlet, and differentiate its pressure drop.
:::

:::{grid-item-card} Q2D vortex dynamics
:link: tutorials/q2d
:link-type: doc
Evolve a depth-averaged strong-field model and reproduce its poster and movie.
:::

:::{grid-item-card} Validation
:link: validation/index
:link-type: doc
See the analytical, numerical, and FreeMHD evidence for every claim.
:::

::::

```{toctree}
:hidden:
:caption: Get started

getting_started/install
getting_started/first_run
getting_started/problem
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/fully_developed
tutorials/duct_design_law
tutorials/fringing
tutorials/walls_and_fields
tutorials/differentiation
tutorials/q2d
```

```{toctree}
:hidden:
:caption: How-to guides

how_to/restart_and_output
```

```{toctree}
:hidden:
:caption: Physics and numerics

physics/equations
physics/numerics
physics/thermal
```

```{toctree}
:hidden:
:caption: Validation

validation/index
validation/verification
validation/q2d
validation/freemhd
validation/mistrangelo
validation/mhd_nusselt
validation/channel_turbulence
```

```{toctree}
:hidden:
:caption: Reference

reference/api
reference/cli
reference/migration
reference/bibliography
```

```{toctree}
:hidden:
:caption: Development

develop/architecture
develop/contributing
adr/0001-plan-adoption
adr/0002-core-discretization
adr/0003-precision-and-derivatives
adr/0004-process
adr/0005-review-2026-09-13
adr/0006-review-2026-09-22
adr/0007-rename-lmhdx
adr/0008-thermal-first-stage
adr/0009-magnetoconvection-benchmark
adr/0010-high-re-and-api-2
adr/0011-blanket-multiphysics-optimization
adr/0012-one-implementation-per-concept
```
