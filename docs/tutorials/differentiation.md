# Differentiate physical solves

JAX can give the derivative of an LMhdX result, such as a flow rate or a
pressure drop, with respect to an input such as the field strength or a wall's
conductance. LMhdX differentiates a linear solve by solving one more system,
not by recording its iterations. This page shows it for a periodic duct, a duct
leaving a magnet and a Q2D flow.

What JAX can differentiate:
- **Traced and static.** An accepted field core traces continuous physical
  parameters through the production discretization. Mesh counts, iteration
  limits and output policies are static, and validation, status, I/O and
  plotting run on the host.
- **The accepted field cores** are the steady duct, the finite 3-D
  rectangular, layered-duct and straight-pipe models, and the transient Q2D
  model.
- **Implicit differentiation.** A VJP of a SOLVAX-backed linear system solves
  one transposed system instead of recording PCG or GMRES iterations.

How failures show:
- `lmhdx.steady.solve_steady_state` raises eagerly on a rejected primal
  residual or linearized solve.
- Under autodiff tracing or `jax.jit`, a failure gives nonfinite fields or
  derivatives instead. Reject any nonfinite objective or gradient before
  accepting an optimizer step.
- Close over static problem geometry, materials and solver controls to compile
  objectives with `jax.jit(jax.value_and_grad(objective))`.
- LMhdX assembles host factorizations once per trace and does not
  differentiate them. Changing the static problem requires a new trace.

## Through a Problem

{ref}`Derivatives <numerics-derivatives>` explains the policy behind every gradient on this page.

A [`Problem`](../getting_started/problem.md) is a pytree, so
`jax.grad(f)(problem)` returns a `Problem` of derivatives with respect to its
forcing, flow rate, field amplitude, heat source and initial temperature, and
the numbers of its conducting walls. Properties and the grid are static, so a
traced one raises.

### Wall conductance and thickness

Differentiate the pressure gradient a duct needs to carry a flow rate with
respect to its wall conductance `c`.

```python
import jax
import lmhdx
from lmhdx import Wall, duct

lmhdx.enable_x64()


def drive(c):
    # A flow-rate duct: the response is the pressure gradient that carries it.
    return lmhdx.solve(duct(20, 24, flow_rate=4.0, walls=dict(y=Wall(c)))).pressure_gradient


slope = jax.grad(drive)(0.05)  # dp/dx per unit c; central differences agree to 1e-8 on this mesh
print(float(slope))
```

```text
315.3891843920461
```

What is differentiable:
- A conducting `Wall` has leaves `conductance`, or `conductivity` and
  `thickness`. The second pair gives a thin wall of `c = sigma_w t_w / sigma`,
  with lengths in the grid's unit. For resolved walls, each `WallLayer`'s
  `conductivity` and `thickness` are leaves. The cell counts and which walls
  conduct are static.
- The Stokes solve of a periodic duct differentiates all of them by
  `jax.grad`, `jax.jvp` and `jax.vmap`, and a new value does not recompile. A
  sweep over `c` under `jax.vmap` is one program.
- Under `jax.jit` the primal of `c` is unknown, so declare the wall
  `Wall(c, conducting=True)`.
- A conductance of exactly `0.0` is the insulating wall, which is structure and
  has no derivative. A traced zero raises `ValueError`, so use a small positive
  value instead.

What is not:
- Open ducts (`Open` axis), unsteady and thermal runs, and runs with advection
  take the primal wall numbers. Their derivative in a wall number is NaN, never
  a silent zero, and the other leaves are unaffected.
- Two resolved walls on two axes with traced parameters raise
  `NotImplementedError`.

## Steady duct response

Differentiate the mean axial velocity with respect to the whole `Problem`.

```python
import jax.numpy as jnp

problem = lmhdx.duct(20, 48)


def objective(problem):
    return jnp.mean(lmhdx.solve(problem).velocity[0].data[0])


value = objective(problem)
gradient = jax.grad(objective)(problem)
print(float(gradient.forcing[0]), float(gradient.B.strength))
```

```text
0.030703102025215097 -0.0010540563599652397
```

`lmhdx.steady.solve_steady_state` finds the steady state with one
conjugate-gradient solve and differentiates it with one more, so the reverse
tape holds no iteration histories.
- The continuous inputs are the pressure forcing, the field amplitude and the
  wall numbers.
- Rectangular insulating and thin-wall Hunt ducts pass central-difference
  gates.
- The solve meets a prescribed flow rate by scaling the unit-drive solution,
  which keeps it differentiable.
- The kernel-level call `solve_steady_state(problem, forcing=, field_scale=)`
  takes the drive and a scalar multiplier of the field as arguments, so
  `jax.jit` can wrap it.

## Three-dimensional fringe response

The open duct and its solve are described in {ref}`ducts with an inlet and an outlet <open-ducts>`.

`lmhdx.axial.solve_open_duct` solves a duct with an inlet and an outlet through
a field that varies along it, and is differentiable in the field scale. The
derivative is one more conjugate-gradient solve with the same operator, as for
a periodic duct.

```python
from lmhdx.axial import fringe_duct, pressure_drop, solve_open_duct

problem = fringe_duct(hartmann=20.0, wall_conductance=0.02, upstream=6.0, downstream=3.0,
                      spacing=0.5, cells=12, cells_in_layer=3)


def drop(field_scale):
    return pressure_drop(solve_open_duct(problem, field_scale=field_scale).pressure, -3.0, 3.0)


value, derivative = jax.value_and_grad(drop)(1.0)
print(float(value), float(derivative))
```

```text
131.60664371854293 142.12209004153215
```

- `python examples/fringe_duct_example.py` checks this derivative against
  central differences. The relative difference is about 1e-9 on that mesh, and
  the test gate is 1e-6.
- With advection on, LMhdX differentiates the steady root the same way, by the
  implicit function theorem on one step, whether Newton or the march found it.
- `fringe_duct` fixes the wall conductance, geometry and flow rate when it
  builds the problem. See the [fringe tutorial](fringing.md).

## Transient Q2D response

The step is {ref}`the spectral IFRK4 step <q2d-ifrk4>`.

For a time-dependent field objective, call the field-only core.

```python
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
print(float(value), gradient)
```

```text
0.20632671619192336 [-0.66024549 -0.33012275]
```

The working precision:
- The state, forcing and continuous coefficients determine one working dtype
  through [JAX type promotion](https://docs.jax.dev/en/latest/101/type_promotion.html),
  at least float32. The initial state needs no manual cast.
- Explicit float64 parameters promote a float32 initial state when x64 is
  enabled. Weak Python scalar defaults keep a float32 state.
- `Q2DProblem` and `evolve_q2d` reject complex physical inputs.

This derivative is exact for the finite dealiased IFRK4 evolution. Its default
SOLVAX checkpoint schedule stores `O(sqrt(steps))` trajectory states instead of
the full tape. `tests/test_physics.py` checks the analytical decay, the
JVP/VJP identity and the compiled reverse memory.

The field-level entry points for optimization are `lmhdx.solve`,
`lmhdx.axial.solve_open_duct` and `evolve_q2d`. Other result objects run on the
host unless their API reference names a traced field core and its derivative
tests.
