# Differentiate physical solves

LMhdX separates traced numerical fields from host-side validation, status, I/O,
and plotting. An accepted field core must carry continuous physical parameters
through the production discretization; mesh counts, iteration limits, and
output policies remain static controls. The accepted field cores are the
steady duct, finite 3-D rectangular/layered-duct and straight-pipe, and
transient Q2D models.

SOLVAX-backed linear systems use implicit differentiation: a VJP solves one
transposed system instead of recording PCG or GMRES iterations. A coupled
steady-flow interface is documented as end-to-end differentiable only when its
production field equations use that contract and pass independent gradient,
residual, runtime, and memory gates.

For `lmhdx.steady.solve_steady_state`, a rejected primal residual or linearized
solve raises eagerly. During autodiff tracing, failure instead produces
nonfinite fields or derivatives without host callbacks: reject any nonfinite
objective **or** gradient before accepting an optimizer step. Drive and field
scale are continuous inputs; close over static problem geometry/materials and
solver controls to compile objectives with `jax.jit(jax.value_and_grad(objective))`.
Host factorizations are assembled once per trace, not differentiated; changing
the static problem requires a new trace. Compiled failures remain nonfinite.

## Through a Problem

A [`Problem`](../getting_started/problem.md) is a pytree: `jax.grad(f)(problem)` returns a `Problem`
of derivatives with respect to its forcing, flow rate, field amplitude, heat source and initial
temperature, and the numbers of its conducting walls. Properties and the grid are static, so a traced one raises.

### Wall conductance and thickness

A conducting `Wall` has leaves `conductance`, or `conductivity` and `thickness` (a thin wall of
`c = sigma_w t_w / sigma`, lengths in the grid's unit), and for resolved walls each `WallLayer`'s
`conductivity` and `thickness`; the cell counts and which walls conduct are static. The Stokes solve of a
periodic duct differentiates all of them, by `jax.grad`, `jax.jvp` and `jax.vmap` (a design sweep over
`c` is one program), and a new value does not recompile:

```python
import jax
import lmhdx
from lmhdx import Wall, duct

lmhdx.enable_x64()


def drive(c):
    # A flow-rate duct: the response is the pressure gradient that carries it.
    return lmhdx.solve(duct(20, 24, flow_rate=4.0, walls=dict(y=Wall(c)))).pressure_gradient


slope = jax.grad(drive)(0.05)  # dp/dx per unit c; central differences agree to 1e-8 on this mesh
```

Under `jax.jit` the primal of `c` is unknown, so declare the wall `Wall(c, conducting=True)`. A
conductance of exactly `0.0` is the insulating wall, which is structure and has no derivative: a traced
zero raises `ValueError` (use a small positive value). The potential of the wall is a fast-diagonal solve
whose eigenbasis is built in the trace; the mirror sheets of a symmetric duct are an exactly degenerate pair,
so the solve is differentiated implicitly, `dx = A^-1 (db - dA x)`, not through `eigh`. Open ducts
(`Open` axis), unsteady and thermal runs, and runs with advection take the primal wall numbers: their
derivative in a wall number is NaN (never a silent zero), and the other leaves are unaffected. Two resolved
walls on two axes with traced parameters raise `NotImplementedError`.

## Steady duct response

```python
import jax
import jax.numpy as jnp
import lmhdx

case = lmhdx.make_shercliff_case(ha=20, ny=48, nz=48)


def objective(forcing, field_scale):
    velocity, potential, jy, jz, lorentz = lmhdx.solve_fully_developed_fields(
        case,
        forcing=forcing,
        magnetic_field_scale=field_scale,
    )
    return jnp.mean(velocity) + 1e-3 * jnp.mean(jy**2 + jz**2)


value, gradient = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(1.0, 1.0)
```

This path is the staggered-core solve `lmhdx.solve` runs on the same case, one
compiled program shared by both. The steady state is a conjugate-gradient solve
differentiated by one more (`lmhdx.steady.solve_steady_state`), so iteration
histories are absent from the reverse tape. The continuous inputs are pressure
forcing and a scalar multiplier on the imposed magnetic field. Rectangular
Hartmann/Shercliff and thin-wall Hunt cases pass central-difference gates; a
prescribed flow rate is met by scaling the unit-drive solution, which keeps it
differentiable.

## Three-dimensional fringe response

`lmhdx.axial.solve_open_duct` solves a duct with an inlet and an outlet through
a field that varies along it, and is differentiable in the field scale. The
derivative is one more conjugate-gradient solve with the same operator, as for
a periodic duct:

```python
import jax
from lmhdx.axial import fringe_duct, pressure_drop, solve_open_duct

problem = fringe_duct(hartmann=20.0, wall_conductance=0.02, upstream=6.0, downstream=3.0,
                      spacing=0.5, cells=12, cells_in_layer=3)


def drop(field_scale):
    return pressure_drop(solve_open_duct(problem, field_scale=field_scale).pressure, -3.0, 3.0)


value, derivative = jax.value_and_grad(drop)(1.0)
```

`python examples/fringe_duct_example.py` checks this derivative against central
differences (relative difference about 1e-9 on that mesh; the test gate is
1e-6). With advection on, the steady root, by Newton or by the march, is differentiated the same way, by the implicit function theorem on one step. The
wall conductance, geometry and flow rate are fixed when the problem is built; see the
[fringe tutorial](fringing.md).

## Transient Q2D response

For a time-dependent field objective, call the field-only core:

```python
import jax
import jax.numpy as jnp
import lmhdx

jax.config.update("jax_enable_x64", True)
case = lmhdx.make_q2d_case(shape=(32, 32), steps=80)


def objective(parameters):
    viscosity, friction = parameters
    vorticity, _, _ = lmhdx.evolve_q2d(
        case.initial_vorticity,
        viscosity=viscosity,
        hartmann_friction=friction,
        dt=case.dt,
        steps=case.steps,
    )
    return jnp.mean(vorticity**2)


value, gradient = jax.value_and_grad(objective)(
    jnp.asarray([case.viscosity, case.hartmann_friction], dtype=jnp.float64)
)
```

The state, forcing and continuous coefficients determine one working dtype
through [JAX type promotion](https://docs.jax.dev/en/latest/101/type_promotion.html),
with at least float32 precision. Explicit float64 parameters promote a float32
initial state when x64 is enabled; weak Python scalar defaults preserve a
float32 state. Both `Q2DProblem` and `evolve_q2d` use this policy and reject
complex physical inputs. No manual cast of the initial state is required.
This derivative is exact for the finite dealiased IFRK4 evolution. Its default
SOLVAX checkpoint schedule stores `O(sqrt(steps))` trajectory states instead of
the full tape. The analytical decay, JVP/VJP identity, and compiled reverse
memory tests in `tests/test_physics.py` are the executable acceptance contract.

The explicit field-level optimization surfaces are
`solve_fully_developed_fields`, `lmhdx.axial.solve_open_duct`,
`lmhdx.axial.solve_open_pipe`, and `evolve_q2d`.
Other result objects are host orchestration unless their API reference
explicitly identifies a traced field core and derivative evidence.
