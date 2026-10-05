# A duct leaving a magnet

`lmhdx.axial` gives the staggered core an inlet and an outlet along its first
axis, so the flow can pass through a field that changes along the duct. The
conditions follow HIMAG, FreeMHD, GridapMHD and the 2025 six-code benchmark:

- **Inlet:** LMhdX's own fully developed profile at the inlet field, solved on
  the duct's cross-section and scaled to the imposed flow rate. The flow rate is
  exact and the pressure drop is an output.
- **Outlet:** zero axial gradient of the velocity and `p = 0`.
- **Both ends:** no normal current.

`fringe_duct` builds the ANL fringe of ANL/FPP/TM-228 in a square duct: the
field falls as `B_y = Ha (1 - sin(pi x / 2 x0)) / 2` over `|x| <= x0`, uniform
upstream and zero downstream, with buffers of uniform field before and after.

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
solution = solve_open_duct(problem)
print(station_flow_rates(solution.velocity))  # 4.0 at every station
print(pressure_drop(solution.pressure, -3.0, 3.0))
```

The solve is one preconditioned conjugate-gradient solve, certified on its
residual; it raises if the residual is not reached. Check the balances it
guarantees before reading a pressure:

```python
from lmhdx.axial import charge_balance, mass_balance

print(float(solution.residual_norm / solution.initial_residual_norm))  # <= 1e-9
print(float(mass_balance(solution.velocity)), float(charge_balance(solution, problem)))  # round-off
```

The drop is differentiable in the field strength through the same implicit
solve as a periodic duct:

```python
import jax

drop = jax.grad(lambda scale: pressure_drop(solve_open_duct(problem, field_scale=scale).pressure, -3.0, 3.0))
print(drop(1.0))
```

`open_duct` turns any `ChannelProblem` with a field that is uniform over the
inlet cross-section, including an `ImposedField` of arrays, into an
inflow-outflow duct; `axial_faces` grades the axial mesh from the fringe into
the buffers.

What is and is not established:

- The flow rate at every station, mass and charge hold to round-off, and far
  upstream the gradient is the fully developed one (within 0.5 % in the tests).
- Doubling the buffers moves the drop over TM-228's window `[-6, 2]` by
  3.5e-9 at Ha 100.
- Only the inertialess (Stokes-limit) flow is solved: `advection="off"`.
- On the ANL case the excess drop over the locally fully developed drop agrees
  within 1 % with the inertialess core-flow model (`lmhdx.coreflow`, TM-228
  eqs. 4a–4c) once the model carries the layers' conductance
  (`coreflow.layer_conductances`; c 0.1, Ha 2×10⁴). TM-228's uncorrected
  value is the Ha → ∞ limit, which no 3-D solve reaches.
- There is no straight pipe with an open axis yet, and no thick or layered wall.

Run `python examples/fringe_duct_example.py` for the case above with its
conservation checks and a central-difference check of the derivative. The
[FreeMHD guide](../validation/freemhd.md) describes the external comparison.
