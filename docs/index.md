# LMhdX

LMhdX solves inductionless liquid-metal MHD with JAX. It provides
analytical-reference fully developed flows, three-dimensional ducts and pipes
on a staggered core, ducts with an inlet and an outlet through a field that
varies along them, and periodic Q2D flow
with Hartmann-layer damping. LMhdX builds the physics;
[SOLVAX](https://github.com/uwplasma/SOLVAX) supplies reusable numerical solvers.

```{image} _static/validation_ladder.webp
:alt: Hartmann-layer collapse, flow-rate error against a spectral reference, and mesh convergence
:align: center
```

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
```

```{toctree}
:hidden:
:caption: Validation

validation/index
validation/freemhd
```

```{toctree}
:hidden:
:caption: Reference

reference/api
reference/cli
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
```
