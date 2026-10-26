# Problems, solvers and one verb

`Problem` says what is solved, `Solver` how, and `solve(problem, solver=None, start=None)` runs it.
A preset such as `duct` only fills a `Problem`; a case no preset covers builds one directly.

```python
import jax
import lmhdx

lmhdx.configure(dtype="float64")                 # explicit; importing LMhdX changes no JAX setting
problem = lmhdx.duct(20.0, 48, flow_rate=4.0)    # Hartmann duct, mean velocity one
solution = lmhdx.solve(problem)
print(solution.flow_rate, solution.pressure_gradient)

gradient = jax.grad(lambda p: lmhdx.solve(p).pressure_gradient)(problem)
print(gradient.B.strength)                       # d(pressure gradient) / d(field strength)
```

`python examples/problem_example.py` runs this with the inputs written out. The result reports its
arrays by name (`solution.fields`, each also an attribute), the drive of a periodic duct
(`pressure_gradient`), the flux (`flow_rate`), the kernel's `residual`, and for an unsteady run the
`history`.

## What a Problem holds

| Field | Meaning |
|---|---|
| `grid` | A `lmhdx.grid.Grid`; an axis is `lmhdx.grid.axis(cells, lower, upper)` with `beta=` (tanh), `ratio=` (geometric), `layer_thickness=` (wall-resolving) or `blend=` (the sine blend of Krasnov, Zikanov and Boeck, `z = a sin(pi xi/2) + (1 - a) xi`). |
| `bc` | One entry per axis: `"periodic"`, `"wall"`, `Wall(conductance=, layers=, temperature=, heat_flux=)`, `Sides(lower, upper)` or `Open(profile=None)` (inflow at the lower x end, free outflow at the upper). |
| `density`, `viscosity`, `conductivity` | Material properties. |
| `B` | `Uniform(bx, by, bz)`, or from `lmhdx.fields` `Fringe`, `Table(path, scale)` and `Func(fn, params)`. |
| `forcing`, `flow_rate` | The body force, or the flux that sets the drive of a periodic Stokes duct or the inlet of an `Open` axis; one of them. |
| `advection`, `thermal` | `"off"`, `"central"` or `"limited"` transport; a `Thermal(...)` adds heat transport. |

The presets return the grids and fields of the older constructors: `duct` (`duct_problem`),
`fringe_duct` (`lmhdx.axial.fringe_duct`), `lmhdx.problem.pipe` (`pipe_problem`), `lmhdx.problem.q2d`
(`make_q2d_case`) and `channel`, a plane channel at constant bulk velocity and half-width one. `channel`
takes the bulk Reynolds number `re`, or `R = U0 delta / nu` of Boeck, Krasnov and Zienicke, a wall-normal
or spanwise `field`, and the domain of their Table 1 or of Krasnov, Zikanov and Boeck 2011. A Hartmann
channel (field normal to the walls) needs a zero net spanwise current, which no kernel imposes yet: it is
built and refused at `solve`.

## Solvers

`method` is `stokes` (one certified conjugate-gradient solve), `newton` (the same with advection on),
`euler` (the projection step as one compiled scan, or with `thermal` the heat step) or `ifrk4` (the Q2D
integrator). `auto` picks `euler` if `steps` is given and `stokes` or `newton` otherwise. `rk3`, `march`
and `implicit` are named for plan steps H2.2 and H3 and raise until they exist.

## Derivatives

The leaves are the `forcing`, the `flow_rate`, the field amplitude (`Uniform.strength`,
`Fringe.strength`, `Table.scale`), the Q2D state and coefficients, a heat source and an initial
temperature. They run through the implicit derivative of the steady solve, at about the cost of one more
solve. The grid, boundary kinds, wall conductances, material properties and scheme names are static: a
traced one raises instead of returning a zero. The factorizations are built on the host from the primal
values, so differentiate with `jax.grad` outside `jit`, or close over the problem and `jit` the objective.
