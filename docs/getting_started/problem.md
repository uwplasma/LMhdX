# Problems, solvers and one verb

A run is three nouns and one verb. `Problem` says what is solved, `Solver` how, `Output` where
it goes, and `solve(problem, solver=None, output=None, start=None)` runs it. A preset such as
`duct` only fills a `Problem`; a case no preset covers builds one directly.

```python
import jax
import lmhdx

lmhdx.configure(dtype="float64")           # explicit; importing LMhdX changes no JAX setting
problem = lmhdx.duct(20.0, 48, flow_rate=4.0)       # Hartmann duct, mean velocity one
solution = lmhdx.solve(problem)
print(solution.flow_rate, solution.pressure_gradient)

gradient = jax.grad(lambda p: lmhdx.solve(p).pressure_gradient)(problem)
print(gradient.B.strength)                 # d(pressure gradient) / d(field strength)
```

`python examples/problem_example.py` runs this with the inputs written out. A `Problem` is a
frozen dataclass and a registered pytree, so `jax.grad`, `jax.jvp` and `jax.vjp` take it as an
argument and return a `Problem` of derivatives. The result of `solve` reports its fields by name
(`solution.fields`), the drive of a periodic duct (`pressure_gradient`), the flux (`flow_rate`),
the kernel's `residual`, and for an unsteady run the `history`.

## What a Problem holds

| Field | Meaning |
|---|---|
| `grid` | A `lmhdx.grid.Grid`; an axis is `lmhdx.grid.axis(cells, lower, upper)` with `beta=` (tanh), `ratio=` (geometric), `layer_thickness=` (wall-resolving) or `blend=` (the sine blend of Krasnov, Zikanov and Boeck, `z = a sin(pi xi/2) + (1 - a) xi`). |
| `bc` | One entry per axis: `"periodic"`, `"wall"`, `Wall(conductance=, layers=, temperature=, heat_flux=)`, `Sides(lower, upper)` or `Open(profile=None)` (inflow at the lower x end, free outflow at the upper). |
| `density`, `viscosity`, `conductivity` | Material properties. |
| `B` | `Uniform(bx, by, bz)`, or from `lmhdx.fields` `Fringe(strength, half_length, ...)`, `Table(path, scale)` and `Func(fn, params)`. |
| `forcing`, `flow_rate` | The body force, or the flux that sets the drive of a periodic Stokes duct or the inlet of an `Open` axis; one of them. |
| `advection`, `thermal` | `"off"`, `"central"` or `"limited"` transport; a `Thermal(...)` adds heat transport. |

The presets return the grids and fields of the older constructors: `duct` (`duct_problem`),
`fringe_duct` (`lmhdx.axial.fringe_duct`), `lmhdx.problem.pipe` (`pipe_problem`),
`lmhdx.problem.q2d` (`make_q2d_case`, returning the `Q2D` problem), and `channel`, a plane
channel at constant bulk velocity with half-width one. `channel` takes the bulk Reynolds number
`re`, or Boeck, Krasnov and Zienicke's `R = U0 delta / nu` (then `Re_av = R Ha Uav/U0` with
`Uav/U0 = (cosh Ha - sinh Ha/Ha)/(cosh Ha - 1)`), a wall-normal or spanwise `field`, and takes its
domain from their Table 1 or from Krasnov, Zikanov and Boeck 2011. A Hartmann channel (field normal to
the walls, periodic spanwise) needs a zero net spanwise current, which no kernel imposes yet, so it is
built and refused at `solve`.

## Solvers

| `method` | Runs |
|---|---|
| `stokes` | one certified conjugate-gradient solve (`lmhdx.steady`, `lmhdx.axial.solve_open_duct`, `lmhdx.pipe.solve_pipe`) |
| `newton` | the same with advection on |
| `euler` | the projection step as one compiled scan (`lmhdx.core3d.advance`), or with `thermal` the heat step (`lmhdx.thermal.advance_thermal`) |
| `ifrk4` | the Q2D integrator |

`auto` picks `euler` if `steps` is given and `stokes` or `newton` otherwise. `rk3`, `march` and
`implicit` are named for plan steps H2.2 and H3 and raise until they exist.

## Derivatives

The leaves are the quantities the kernels differentiate: the `forcing`, the `flow_rate`, the field
amplitude (`Uniform.strength`, `Fringe.strength`, `Table.scale`), the Q2D state and coefficients, a
heat source and an initial temperature. They run through the implicit derivative of the steady solve, so
the cost is about one more solve. The grid, the boundary kinds and wall conductances, the material
properties and the scheme names are static for now; they are not in the tree, so a gradient never
returns a silent zero for one, and `replace`-ing one with a traced value raises. A field is
`strength` times a fixed direction, so a derivative with respect to a field is the one through its
amplitude. The factorizations are built on the host from the primal values: differentiate with
`jax.grad` outside `jit`, or close over the problem and `jit` the objective.

## Not yet

`Output` writes `summary.json` and a ParaView frame at the end of the run; chunked records and
restarts, the TOML schema 3 and the `rk3` and `march` steppers follow in later steps of Program H
(ADR 0010). Importing the package root still sets `XLA_FLAGS` for the GPU matrix-product library, as
before; `lmhdx.configure(gpu_gemm=...)` chooses it explicitly, and the import stops doing it in 2.0.
