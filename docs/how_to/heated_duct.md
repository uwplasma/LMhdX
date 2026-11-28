(how-to-heated-duct)=
# Run a heated duct

This guide runs a heated duct through `lmhdx.solve`, by hand with `lmhdx.thermal`, and from TOML.
The [heat transport page](../physics/thermal.md) gives the equations, limits and diagnostics.

## Run it

This script runs the duct of `examples/heated_duct.toml`, periodic along its length in a field of 10
across it. A volume source heats it, its top wall cools it, and buoyancy is on.

```python
import lmhdx
from lmhdx import Problem, Sides, Solver, Thermal, Uniform, Wall
from lmhdx.grid import Grid, axis

lmhdx.configure(dtype="float64")
problem = Problem(
    Grid(axis(1, 0.0, 1.0), axis(8, -1.0, 1.0), axis(8, -1.0, 1.0)),  # one periodic cell along the duct
    bc=dict(x="periodic", y=Wall(heat_flux=0.0), z=Sides(Wall(heat_flux=0.0), Wall(heat_flux=1.0))),
    B=Uniform(0.0, 10.0, 0.0),
    forcing=(1.0, 0.0, 0.0),
    thermal=Thermal(
        specific_heat=1.0,
        thermal_conductivity=0.05,
        reference_temperature=0.0,
        expansion_coefficient=0.02,  # Boussinesq buoyancy, with gravity along -z
        gravity=(0.0, 0.0, -1.0),
        heat_source=0.5,
    ),
)
solution = lmhdx.solve(problem, Solver(dt=2.0e-3, steps=40))
history = solution.history  # one entry per step: temperatures, heat flows, residuals
print(f"T max {float(history.temperature_max[-1]):.4f}")  # T max 0.0402
```

## A worked example

This example builds by hand with `lmhdx.thermal` the step that "Run it" reaches through
`lmhdx.solve`. It runs a nondimensional Hartmann duct twice. A forcing drives it along its periodic
axis, a uniform volumetric source heats it, and its top wall cools it. The second run adds
buoyancy.

```python
import dataclasses

import jax.numpy as jnp

import lmhdx
from lmhdx.core3d import ChannelProblem
from lmhdx.grid import CENTER, NEUMANN, PERIODIC, BoundaryCondition, Field, Grid, uniform_faces
from lmhdx.thermal import (
    ThermalProblem,
    advance_thermal,
    build_thermal_step,
    initial_thermal_state,
    thermal_boundary,
    validate_thermal_inputs,
)

lmhdx.enable_x64()  # before anything is built: the step's dtype must be the core's

grid = Grid(uniform_faces(1, 0.0, 1.0), uniform_faces(16, -1.0, 1.0), uniform_faces(16, -1.0, 1.0))
periodic, wall = BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN)
flow = ChannelProblem(
    grid, (periodic, wall, wall), magnetic_field=(0.0, 10.0, 0.0), forcing=(1.0, 0.0, 0.0), dt=2.0e-3
)
k = 0.05
conditions = (
    periodic,
    thermal_boundary("heat_flux", conductivity=k),  # adiabatic side walls
    thermal_boundary("heat_flux", 0.0, 1.0, conductivity=k),  # the top wall takes out a unit flux
)
thermal = ThermalProblem(flow, conditions, specific_heat=1.0, thermal_conductivity=k, reference_temperature=0.0)
heat = Field(jnp.full(grid.shape, 0.5), (CENTER,) * 3, grid)

for problem in (thermal, dataclasses.replace(thermal, expansion_coefficient=0.02, gravity=(0.0, 0.0, -1.0))):
    state = initial_thermal_state(problem, 0.0)
    validate_thermal_inputs(problem, state, heat, divergence_tolerance=1e-10)
    step = build_thermal_step(problem, dtype=jnp.float64)
    state, history = advance_thermal(step, state, heat, 200)
    power = float(history.external_power[-1] + history.joule_power[-1])
    assert bool(history.finite.all()) and float(history.advective_cfl.max()) <= 0.8
    assert abs(float(history.heat_balance_residual[-1])) <= 1e-9 * power
    assert float(history.boussinesq_measure.max()) <= 0.05
    print(
        f"beta {problem.expansion_coefficient}: T max {float(history.temperature_max[-1]):.4f}, "
        f"heat out {float(history.conductive_heat_out[-1]):.4f}, Joule {float(history.joule_power[-1]):.4f}"
    )
```

## A run from TOML

The same duct runs as `lmhdx examples/heated_duct.toml` ([CLI and TOML reference](../reference/cli.md)),
where `[problem.thermal]` becomes `Thermal(**table)`. Add an `[output]` table to write its files:

```toml
[output]
dir = "out"
every = 10
```

The command runs in compiled chunks of `every` steps and accepts or rejects each chunk after it
runs. Every sample in a chunk, not only the last, must pass these checks:
- it is finite, and the outlet is valid;
- its advective Courant number is within the scheme's limit, which is 0.8 for upwind, 0.5 for van
  Leer and none for conduction alone. `[solver] advective_cfl_limit` replaces that limit, and the
  summary records it as an override;
- its buoyancy Courant number is within `[solver] buoyancy_courant_limit`, if one is given. That
  limit is a policy calibrated on particular cases, not a stability bound.

The command holds the start to the same limits and to `divergence_tolerance`. Each accepted chunk
writes a record (`heated_duct_000010.npz`) and a frame (`heated_duct_000010.vtr`), gathered by
`heated_duct.pvd`. The first rejected chunk stops the run, writes only
`heated_duct_000020_rejected.npz` (a record that imports but does not resume), and exits with code 2.
`heated_duct_summary.json` holds the verdict, the limits, each diagnostic's largest value, the first
rejected sample's step, reasons and Courant numbers, and the last diagnostics with their `time`. On
a rejection these come from the rejected chunk, while `steps` counts only accepted steps. No limit
applies to the heat balance residual or the Boussinesq measure.

To resume, name a record in `[start]`:

```toml
[start]
path = "out/heated_duct_000020.npz"
```

Its step numbers the frames that follow, and `heated_duct_from_000020.pvd` gathers them, so the
first series stays intact. With `steps = 20`, that run ends on the same record, bit for bit, as the
run of 40 steps. `mode = "resume"`, the default, refuses a record of another problem or solver, and
`mode = "import"` starts another experiment from the record's state. `lmhdx upgrade` converts files of schema 1.

## What a run costs

The table compares `advance_thermal` with `core3d.advance` (implicit viscosity) on an AMD Ryzen 9
5900HS CPU, float64, JAX 0.6.2. "Cold" is the first call, with trace and compile and the persistent
cache off. "Memory" is XLA's compiled temporary and output size.

| Mesh | Run | Cold (s) | Warm (ms/step) | Forward memory (MiB) | Reverse memory, 10 / 40 steps (MiB) |
|:--|:--|--:|--:|--:|--:|
| 1×48×48 | isothermal | 0.47 | 0.19–0.22 | 0.21 | n/a |
| 1×48×48 | thermal | 0.66 | 0.36–0.38 | 0.45 | 1.05 / 3.18 |
| 1×48×48 | buoyant | 0.69 | 0.35 | 0.45 | 1.99 / 4.65 |
| 8×32×32 | isothermal | 0.48 | 0.91–0.95 | 1.06 | n/a |
| 8×32×32 | thermal | 0.89–0.99 | 1.28–1.47 | 1.61 | 3.24 / 9.22 |
| 8×32×32 | buoyant | 0.92–1.05 | 1.34–1.45 | 1.55 | 6.27 / 14.12 |

Forward memory is the same at 10 and 40 steps. Reverse memory grows by one state per step.
