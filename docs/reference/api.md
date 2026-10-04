# Python API

`enable_x64` explicitly enables JAX float64 arrays process-wide. Call it before
constructing meshes or tracing functions; importing LMhdX does not change precision.
`enable_x64`, a case `dtype`, `ChannelProblem` and `Q2DProblem` also set JAX's
`jax_default_matmul_precision` to `'highest'` when it is unset. Float32 matrix
products on Ampere GPUs then run in float32 rather than TensorFloat-32, which
was 3e-4 from float64 on an RTX A4000. A value you set yourself, in
`jax.config` or through `JAX_DEFAULT_MATMUL_PRECISION`, is kept.
Case factories and TOML `[case]` accept `dtype="float32"` or `dtype="float64"`
(default). During the transition, constructing a float64 case with x64 disabled
enables it with a `DeprecationWarning`; explicit activation avoids that warning.
The case dtype controls fully developed mesh, material, initial/restart and output
fields, including compiled derivatives. Float32 is qualified only for the tested
low-Hartmann cases, not high-Hartmann validation. Supplied meshes are cast to the
case dtype; casting cannot recover precision lost during their construction.
See JAX's [dtype and x64 contract](https://docs.jax.dev/en/latest/101/default_dtypes.html).

```python
import lmhdx

lmhdx.enable_x64()
case = lmhdx.make_hartmann_case(ha=2, ny=32, nz=32, dtype="float32")
u, phi, jy, jz, lorentz = lmhdx.solve_fully_developed_fields(case)
```

The package root is the small, stable convenience surface. Advanced workflows
live in the module that owns their concepts.

## Root API

| Area | Names |
|---|---|
| Staggered core | `ChannelProblem`, `duct_problem`, `solve_steady_state`, `advance` |
| Cases and solves | `make_hartmann_case`, `make_shercliff_case`, `make_hunt_case`, `make_q2d_case`, `solve_fully_developed_fields`, `evolve_q2d`, `Q2DProblem`, `solve` |
| Meshes | `generate_rect_duct_mesh_from_faces` (the mesh a case's solution reports) |
| Wall models | `WallLayer`, `wall_conductance_ratio`, `effective_pinhole_conductance_ratio`, `tangential_stack_conductance_ratio`, `normal_stack_leakage_ratio`, `equivalent_single_layer`, `nested_wall_layer_resolution_summary` |
| Units | `dynamic_to_kinematic_viscosity`, `kinematic_to_dynamic_viscosity`, `hartmann_number`, `reynolds_number`, `interaction_parameter`, `magnetic_reynolds_number`, `magnetic_field_from_hartmann` |
| Evidence | The energy budget of `lmhdx.core3d` and the analytical, conservation, and packaged benchmark tools in `lmhdx.validation` |
| Runtime | `enable_compilation_cache` |

`solve(model)` accepts a `ChannelProblem`, `CaseSpec` or `Q2DProblem`. A
steady fully developed `CaseSpec` runs on the staggered core through
`lmhdx.fully_developed`, as does `solve_fully_developed_fields`, and a transient
`CaseSpec` runs implicit Euler steps there too. A duct with
an inlet and an outlet is solved by `lmhdx.axial.solve_open_duct`.

`duct_problem(hartmann=..., cells=..., wall_conductance=...)` builds a square
insulating or Hunt duct with meshes that resolve the layers that exist -- `a/Ha`
against the walls normal to the field, `a/sqrt(Ha)` against the others -- and
`solve_steady_state` finds its steady state as a differentiable root. `advance`
runs the same physics as one compiled trajectory when the transient is what is
wanted. The remaining result types expose `converged`, `status`, `steps`,
`residual`, `fields`, and `diagnostics`; specialized solve functions provide
restart, progress, logging, and timing hooks in their owning modules.

## Case schema

```{eval-rst}
.. automodule:: lmhdx.cases
   :members:
   :show-inheritance:
```

## Inertialess core flow

```{eval-rst}
.. automodule:: lmhdx.coreflow
   :members:
```

## Ducts with an inlet and an outlet

```{eval-rst}
.. automodule:: lmhdx.axial
   :members:
```

## Quasi-two-dimensional flow

```{eval-rst}
.. automodule:: lmhdx.q2d
   :members:
```

## Imposed fields

```{eval-rst}
.. automodule:: lmhdx.cases
   :members:
```

## Fully developed cases on the staggered core

```{eval-rst}
.. automodule:: lmhdx.fully_developed
   :members:
```

## Case builders

```{eval-rst}
.. automodule:: lmhdx.cases
   :members:
```

## Output and restart

```{eval-rst}
.. automodule:: lmhdx.io
   :members:
```

## Validation

```{eval-rst}
.. automodule:: lmhdx.validation
   :members:
```

## Units and walls

```{eval-rst}
.. automodule:: lmhdx.cases
   :members:
```
