# Python API

`enable_x64` explicitly enables JAX float64 arrays process-wide. Call it before
constructing meshes or tracing functions; importing LMhdX does not change precision.
`enable_x64`, `ChannelProblem` and `Q2DProblem` also set JAX's
`jax_default_matmul_precision` to `'highest'` when it is unset. Float32 matrix
products on Ampere GPUs then run in float32 rather than TensorFloat-32, which
was 3e-4 from float64 on an RTX A4000. A value you set yourself, in
`jax.config` or through `JAX_DEFAULT_MATMUL_PRECISION`, is kept.
`Solver(dtype="float32")` or `"float64"` (default) selects the precision of a solve and calls
`lmhdx.configure`; float32 is qualified only for the tested low-Hartmann cases, not high-Hartmann validation.
See JAX's [dtype and x64 contract](https://docs.jax.dev/en/latest/101/default_dtypes.html).

```python
import lmhdx

result = lmhdx.solve(lmhdx.duct(2, 32), lmhdx.Solver(dtype="float32"))
```

The package root is the small, stable convenience surface. Advanced workflows
live in the module that owns their concepts.

## Root API

| Area | Names |
|---|---|
| Problems | `Problem`, `Wall`, `Sides`, `Open`, `Thermal`, `Uniform`, `Solver`, `Output`, `duct`, `channel`, `fringe_duct`, `configure`, `solve` |
| Staggered core | `ChannelProblem`, `duct_problem`, `solve_steady_state`, `advance` |
| Quasi-2-D | `make_q2d_case`, `evolve_q2d`, `Q2DProblem` |
| Wall models | `WallLayer`, `wall_conductance_ratio`, `effective_pinhole_conductance_ratio`, `tangential_stack_conductance_ratio`, `normal_stack_leakage_ratio`, `equivalent_single_layer`, `nested_wall_layer_resolution_summary` |
| Units | `dynamic_to_kinematic_viscosity`, `kinematic_to_dynamic_viscosity`, `hartmann_number`, `reynolds_number`, `interaction_parameter`, `magnetic_reynolds_number`, `magnetic_field_from_hartmann` |
| Evidence | The energy budget of `lmhdx.core3d` and the analytical profiles of `validation/` |
| Runtime | `enable_compilation_cache` |

`solve(problem)` accepts a `Problem`, a `Q2D`, a `ThermalProblem`, a `ChannelProblem` or a `Q2DProblem`. A duct with
an inlet and an outlet is solved by `lmhdx.axial.solve_open_duct`. A pipe with an inlet and an outlet on the polar grid
(`fringe_pipe` of `validation/open_pipe.py`) is solved in the Stokes limit by `solve_open_pipe` there.

`duct_problem(hartmann=..., cells=..., wall_conductance=...)` builds a square
insulating or Hunt duct with meshes that resolve the layers that exist -- `a/Ha`
against the walls normal to the field, `a/sqrt(Ha)` against the others -- and
`solve_steady_state` finds its steady state as a differentiable root. `advance`
runs the same physics as one compiled trajectory when the transient is what is
wanted. The kernel result types expose `converged`, `status`, `steps`,
`residual`, `fields`, and `diagnostics`; specialized solve functions provide
restart, progress, logging, and timing hooks in their owning modules.

## Problems, fields and the one verb

`lmhdx.solve(problem, solver=None, output=None, start=None)` runs a `Problem` or a `Q2D`;
the guide is [Problems, solvers and one verb](../getting_started/problem.md).
`lmhdx.configure` sets the runtime choices that importing the package leaves alone.

```{eval-rst}
.. automodule:: lmhdx.problem
   :members:
.. automodule:: lmhdx.fields
   :members: Uniform, Fringe, Table, Func
.. automodule:: lmhdx.solver
   :members:
.. automodule:: lmhdx.output
   :members: Output, Start, read_record, write_record, write_vtk, write_pvd
.. automodule:: lmhdx.config
   :members: load, dumps, run, upgrade
.. autofunction:: lmhdx.configure
.. autofunction:: lmhdx.grid.axis
```

## Units and wall models

```{eval-rst}
.. automodule:: lmhdx.units
   :members:
   :show-inheritance:
```

## Ducts with an inlet and an outlet

```{eval-rst}
.. automodule:: lmhdx.axial
   :members:
```

## Heat transport

```{eval-rst}
.. automodule:: lmhdx.thermal
   :members:
```

## Fast-diagonal Poisson and Helmholtz solvers

```{eval-rst}
.. automodule:: lmhdx.poisson
   :members:
   :exclude-members: PipeProblem, pipe_grid, pipe_problem, solve_pipe
```

## Fully developed pipe flow

The pipe names remain importable from `lmhdx.poisson`, where they lived in 1.9.

```{eval-rst}
.. automodule:: lmhdx.pipe
   :members:
```

## Quasi-two-dimensional flow

```{eval-rst}
.. automodule:: lmhdx.q2d
   :members:
```

## The command line

```{eval-rst}
.. automodule:: lmhdx.cli
   :members:
```
