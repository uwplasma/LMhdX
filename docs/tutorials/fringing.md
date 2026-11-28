# A duct leaving a magnet

Where a duct leaves a magnet, the field falls along the duct, and the flow and
the pressure change with it. LMhdX solves such a duct with an inlet and an
outlet along its first axis (`lmhdx.axial`). This page sets up the ANL fringe
of ANL/FPP/TM-228, solves it, checks its balances and differentiates its
pressure drop, as `python examples/fringe_duct_example.py` does.

The boundary conditions follow HIMAG, FreeMHD, GridapMHD and the 2025 six-code
benchmark. The inlet takes LMhdX's own fully developed profile at the inlet
field, solved on the duct's cross-section and scaled to the imposed flow rate.
The pressure drop is an output. At the outlet the
axial gradient of the velocity is zero and `p = 0`. Neither end carries a
normal current.

## Set up and solve

{ref}`The inlet and outlet conditions <open-ducts-bcs>` and {ref}`the lift <open-ducts-lift>` describe the scheme behind this solve.

`fringe_duct` builds the fringe in a square duct. Over `|x| <= x0` the field
falls as `B_y = Ha (1 - sin(pi x / 2 x0)) / 2`. It is uniform upstream and zero
downstream, and the duct has buffers of uniform field before and after the fringe.
Lengths are in duct half-widths.

```python
from lmhdx.axial import fringe_duct, pressure_drop, solve_open_duct, station_flow_rates

problem = fringe_duct(
    hartmann=20.0,
    wall_conductance=0.02,  # all four walls
    half_length=3.0,  # x0
    upstream=6.0,
    downstream=3.0,
    spacing=0.5,  # axial cell size over the fringe
    cells=12,
    cells_in_layer=3,
    flow_rate=4.0,  # a unit mean velocity
)
```

Solve it, then read the flow rate at every station and the pressure drop across
the fringe.

```python
solution = solve_open_duct(problem)
print(station_flow_rates(solution.velocity))  # 4.0 at every station
print(pressure_drop(solution.pressure, -3.0, 3.0))
```

```text
[4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4. 4.
 4. 4. 4. 4. 4.]
131.60664371853954
```

In the Stokes limit the solve is one preconditioned conjugate-gradient solve,
and LMhdX raises if it misses its residual. Check the balances before reading a
pressure.

```python
from lmhdx.axial import charge_balance, mass_balance

print(float(solution.residual_norm / solution.initial_residual_norm))  # <= 1e-9
print(float(mass_balance(solution.velocity)), float(charge_balance(solution, problem)))  # round-off
```

```text
8.451712115357334e-10
1.020443188428928e-16 7.598705794737979e-15
```

`jax.grad` differentiates the drop in the field strength through the same
implicit solve as for a periodic duct.

```python
import jax

drop = jax.grad(lambda scale: pressure_drop(solve_open_duct(problem, field_scale=scale).pressure, -3.0, 3.0))
print(drop(1.0))
```

```text
142.12209004153328
```

The example also checks this derivative against central differences and
compares the upstream gradient with the fully developed one.

`open_duct` turns any `ChannelProblem` with a field that is uniform over the
inlet cross-section, including an `ImposedField` of arrays, into an
inflow-outflow duct. `axial_faces` grades the axial mesh from the fringe into
the buffers.

## Add inertia

{ref}`March or Newton <open-ducts-march-newton>` explains the two inertial solvers and how `method="auto"` chooses.

With advection on (`advection="central"`), `solve_open_duct` adds inertia. It
starts from the Stokes-limit solution and reaches the steady state by
Newton-Krylov or by marching the Runge-Kutta stepper. `method="auto"` picks
whichever it finds faster. `python examples/open_duct_inertia_example.py`
measures the inertial share of the drop this way at Ha 30. Keep `growth=1.0`,
which keeps the cell Reynolds number small.

```python
def build(advection):
    return fringe_duct(hartmann=30.0, half_length=3.0, upstream=6.0, downstream=3.0, spacing=0.5,
                       cells=12, cells_in_layer=3, flow_rate=4.0 * 5.0, advection=advection, growth=1.0)


stokes, full = solve_open_duct(build("off")), solve_open_duct(build("central"))
drops = [float(pressure_drop(fields.pressure, -3.0, 3.0)) for fields in (stokes, full)]
print(f"status {int(full.status)}, inertial share {100 * abs(drops[1] / drops[0] - 1):.3f} % at Re 5")
```

```text
status 0, inertial share 1.131 % at Re 5
```

Check `status`. Only `CONVERGED` (0) is a steady flow, and the other two
statuses are `stagnated` and `unsteady`. The section "Inertia on the open axis"
in {ref}`the open-duct evidence <validation-open-ducts-inertia>` gives the measured inertial share.

## What is and is not established

- The flow rate at every station, mass and charge hold to round-off, and far
  upstream the gradient is the fully developed one (within 0.5 % in the tests).
- Doubling the buffers moves the drop over TM-228's window `[-6, 2]` by
  3.5e-9 at Ha 100.
- On the ANL case the excess drop over the locally fully developed drop agrees
  within 1 % with the inertialess core-flow model (`lmhdx.coreflow`, TM-228
  eqs. 4a–4c) once the model includes the layers' conductance
  (`coreflow.layer_conductances`; c 0.1, Ha 2×10⁴), and within 4–5 % at
  c 0.02. Both models are LMhdX's, so this is a consistency check, not a
  validation.
- TM-228's uncorrected 0.0178 is the inertialess thin-layer asymptote for Ha
  going to infinity, which no 3-D solve reaches. LMhdX's fits for Ha going to
  infinity give 0.0183–0.0211, 3–18 % above it. That is a fit band, not a 1 %
  check.
- LMhdX solves a circular pipe with an open axis (`fringe_pipe`,
  `solve_open_pipe` of `validation/open_pipe.py`) in the Stokes limit only.
  There is no thick or layered wall.

The [FreeMHD guide](../validation/freemhd.md) describes the external comparison.
