# Problems, solvers and one verb

Every run in LMhdX is three objects and one function. A `Problem` says what to
solve, a `Solver` says how, and `solve(problem, solver=None, output=None,
start=None)` runs it. An `Output` writes records, ParaView files, a summary and
a log. This page lists what each object holds, so you can build a case no
preset covers.

A preset such as `duct` only fills a `Problem`. The example below solves one at
a set flow rate and differentiates the pressure gradient with respect to the
field strength:

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

```text
4.000000000000002 25.95273925471314
1.0640124579450974
```

`python examples/problem_example.py` runs this with the inputs written out. The
result reports:
- its arrays by name (`solution.fields`, each also an attribute);
- the drive of a periodic duct (`pressure_gradient`) and the flux (`flow_rate`);
- the kernel's `residual`;
- for an unsteady run, the `history`.

## What a Problem holds

| Field | Meaning |
|---|---|
| `grid` | A `lmhdx.grid.Grid`; an axis is `lmhdx.grid.axis(cells, lower, upper)` with `beta=` (tanh), `ratio=` (geometric), `layer_thickness=` (wall-resolving) or `blend=` (the sine blend of Krasnov, Zikanov and Boeck, `z = a sin(pi xi/2) + (1 - a) xi`). |
| `bc` | One entry per axis: `"periodic"`, `"wall"`, `Wall(conductance=, conductivity=, thickness=, layers=, temperature=, heat_flux=)`, `Sides(lower, upper)` or `Open(profile=None)` (inflow at the lower x end, free outflow at the upper). |
| `density`, `viscosity`, `conductivity` | Material properties. |
| `B` | `Uniform(bx, by, bz)`, or from `lmhdx.fields` `Fringe`, `Table(path, scale)` and `Func(fn, params)`. |
| `forcing`, `flow_rate` | The body force, or the flux that sets the drive of a periodic Stokes duct or the inlet of an `Open` axis; one of them. |
| `advection`, `thermal` | `"off"`, `"central"` or `"limited"` transport; a `Thermal(...)` adds heat transport. |

The presets return the grids and fields of the older constructors:

| Preset | Older constructor |
|---|---|
| `duct` | `duct_problem` |
| `fringe_duct` | `lmhdx.axial.fringe_duct` |
| `lmhdx.problem.pipe` | `pipe_problem` |
| `lmhdx.problem.q2d` | `make_q2d_case` |

`channel` is a plane channel at constant bulk velocity and half-width one. It
takes the bulk Reynolds number `re`, or `R = U0 delta / nu` of Boeck, Krasnov
and Zienicke, a wall-normal or spanwise `field`, and the domain of their Table 1
or of Krasnov, Zikanov and Boeck 2011. A Hartmann channel, with the field normal
to the walls, needs a zero net spanwise current, which no kernel imposes yet.
LMhdX builds such a channel, and `solve` refuses it.

## Solvers

`Solver(method=...)` takes one of:

| `method` | What it runs |
|---|---|
| `stokes` | one certified conjugate-gradient solve |
| `newton` | the same with advection on |
| `march` | an open duct with advection on, marched to its steady state instead |
| `euler` | the projection step as one compiled scan, or with `thermal` the heat step |
| `ifrk4` | the Q2D integrator |
| `rk3` | the unsteady step of the three-stage IMEX scheme (a mechanical run; a heated run takes `euler`) |
| `implicit` | named for plan step H2.2; raises until it exists |

`auto` picks `euler` if you give `steps` and `stokes` or `newton` otherwise. On
an open duct with advection it picks `march` while the Courant step of the inlet
flow is at most four damping steps, and `newton` when the Courant step is
longer. `lmhdx.axial.method_for` makes this choice, and
`Solver().resolve(problem)` tells you which method it picked. Both methods
reach the same root and derivative to 1e-6.

## Derivatives

You can differentiate with respect to:
- the `forcing` and the `flow_rate`;
- the field amplitude (`Uniform.strength`, `Fringe.strength`, `Table.scale`);
- the Q2D state and coefficients;
- a heat source and an initial temperature;
- the numbers of a conducting `Wall`: its conductance, or conductivity and
  thickness, and those of its layers (see
  [differentiation](../tutorials/differentiation.md)).

LMhdX computes them through the implicit derivative of the steady solve, at
about the cost of one more solve. The grid, boundary kinds, which walls
conduct, material properties and scheme names are static, and a traced one
raises an error instead of returning a zero. LMhdX builds the factorizations on
the host from the primal values, so differentiate with `jax.grad` outside
`jit`, or close over the problem and `jit` the objective.
