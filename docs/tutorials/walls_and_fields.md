# Walls and imposed fields

Real duct walls are layers of metal and insulator, and real magnets give fields
that vary across the duct. This page builds both: a wall stack whose currents
are solved cell by cell, and an imposed field from a formula or a measured
table. The steps follow `python examples/li_aln_wall_stack_example.py`, which
compares a liquid-lithium duct with an aluminium-nitride (AlN) coated wall
against a bare steel one.

## Wall layers

Import what the solve needs and work in double precision.

```python
import numpy as np

from lmhdx import (
    ChannelProblem,
    WallLayer,
    dynamic_to_kinematic_viscosity,
    enable_x64,
    hartmann_number,
    normal_stack_leakage_ratio,
    solve_steady_state,
    tangential_stack_conductance_ratio,
)
from lmhdx.grid import NEUMANN, PERIODIC, BoundaryCondition, Grid, uniform_faces, wall_resolving_faces

enable_x64()
```

Set the fluid and the field, in SI units. The duct's half-width is 5 cm.

```python
HALF_WIDTH, FIELD = 0.05, 0.05  # m, T
DENSITY, CONDUCTIVITY = 500.0, 3.2e6  # lithium: kg/m^3, S/m
viscosity = dynamic_to_kinematic_viscosity(4.0e-4, DENSITY)  # m^2/s
ha = hartmann_number(
    magnetic_field=FIELD, length_scale=HALF_WIDTH, conductivity=CONDUCTIVITY,
    density=DENSITY, kinematic_viscosity=viscosity,
)
```

State each wall layer's thickness, resolution and conductivity with
`WallLayer`, from the fluid outwards: a thin AlN coating on 316L steel.

```python
layers = (
    WallLayer(name="aln", conductivity=1.0e-8, thickness=2.0e-4, cells=1),
    WallLayer(name="316L", conductivity=1.35e6, thickness=1.0e-3, cells=2),
)
c = tangential_stack_conductance_ratio(layers, fluid_conductivity=CONDUCTIVITY, length_scale=HALF_WIDTH)
leakage = normal_stack_leakage_ratio(layers, fluid_conductivity=CONDUCTIVITY, length_scale=HALF_WIDTH)
```

Give the solver the stack as wall cells: one conductivity ratio (wall over
fluid) and one width per cell.

```python
ratios = tuple(layer.conductivity / CONDUCTIVITY for layer in layers for _ in range(layer.cells))
widths = tuple(layer.thickness / layer.cells for layer in layers for _ in range(layer.cells))
stack = (ratios, widths)
```

Build a cross-section of 24 cells, three of them inside each Hartmann layer,
and one periodic cell along the duct.

```python
faces = wall_resolving_faces(
    24, -HALF_WIDTH, HALF_WIDTH, layer_thickness=HALF_WIDTH / ha, cells_in_layer=3, max_ratio=None
)
grid = Grid(uniform_faces(1, 0.0, HALF_WIDTH), faces, faces)
```

`ChannelProblem(wall_layers=...)` takes one stack per end of each wall-normal
axis. Drive the flow with a pressure gradient of 1 Pa/m.

```python
problem = ChannelProblem(
    grid=grid,
    conditions=(BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN), BoundaryCondition(NEUMANN)),
    density=DENSITY,
    viscosity=viscosity,
    conductivity=CONDUCTIVITY,
    magnetic_field=(0.0, FIELD, 0.0),
    forcing=(1.0, 0.0, 0.0),
    dt=HALF_WIDTH**2 / viscosity,
    wall_layers=(None, (stack, stack), (stack, stack)),
)
```

Solve, then scale the drive to a mean velocity of 1 cm/s; the flow is linear
in it.

```python
solution = solve_steady_state(problem)
areas = np.diff(faces)[:, None] * np.diff(faces)[None, :]
mean_velocity = np.sum(areas * solution.velocity[0].data[0]) / np.sum(areas)
print(f"Ha {ha:.0f}, c {c:.3g}, normal leakage {leakage:.3g}, dp/dx {0.01 / mean_velocity:.4g} Pa/m at 1 cm/s")
```

```text
Ha 224, c 0.00844, normal leakage 7.81e-13, dp/dx 0.3686 Pa/m at 1 cm/s
```

The example prints the same pressure gradient for the coated wall, and 1.14 Pa/m
for the bare one.

The example repeats this for a bare 316L wall of the same thickness, sweeps
pinholes in the coating, checks the residual and the charge balance, and plots
the comparison. Its results assess electrical MHD performance only, not
Li/AlN material compatibility.

How the stack is solved:
- The staggered core resolves the currents in the wall cells, with the
  interface conductance from the adjacent material values and face distances.
- A corner cell takes the nearer wall's layer.
- `wall_conductance_ratio`, `tangential_stack_conductance_ratio` and
  `normal_stack_leakage_ratio` describe the regime of a stack.

## Solid walls

Wall layers solve the potential separably, so they reach two axes only when the
third has one cell. Solid walls lift that limit (ADR 0009, D36).

```python
from lmhdx.core3d import SolidWall

steel = SolidWall(
    2e-3, 4, thermal_conductivity=18.3, heat_capacity=4.27e6, electrical_conductivity=1.04e6
)
# ChannelProblem(..., solid_walls=((None, None), (steel, steel), (steel, steel)))
```

`ChannelProblem(solid_walls=...)` takes a `core3d.SolidWall` or `None` at each
end of the wall-bounded axes 1 and 2. A solid wall is a thickness in equal
cells, with its `electrical_conductivity` (absolute, not over the fluid's)
beside the `thermal_conductivity` and `heat_capacity` a thermal problem uses.

How they are solved:
- The walls become cells of an extended grid (`core3d.conjugate_grid`). The
  velocity and pressure stay on the fluid's grid.
- The charge balance is solved through fluid and walls by
  `solvax.pcg_linear_solve`, preconditioned by the fast-diagonal Poisson
  inverse (`poisson.variable_helmholtz` with no shift).
- The balance is singular, its null mode the constants. A rank-one term on
  them makes it positive definite, and the volume mean is removed after.
- The currents into the walls heat the fluid's half cells and are left out of
  the Lorentz force, as a resolved wall's are. The walls' own Joule heat is not
  deposited.
- A corner cell, in walls of both axes, takes the nearer wall's material, and
  the first axis's where the two are equally near. Walls of equal cells on both
  axes tie exactly there; earlier versions split such ties unevenly
  ([what changed](https://github.com/uwplasma/LMhdX/blob/b97ad3a8525a20502279961d5b4201d158d971e9/docs/tutorials/walls_and_fields.md#solid-walls)).

What is checked (`tests/test_core3d.py`):
- On a section one cell long, solid walls of the same cells give the resolved
  path's potential, currents and powers to 1e-9, corners included, and its
  developed flow rate to 1e-8.
- Thinned at a fixed conductance ratio $c = \sigma_w t_w/(\sigma a)$, Hartmann
  walls alone approach the thin-wall model at second order in $t_w$. The flow
  rate of Hunt's duct at Ha 20 and $c = 0.05$ is 4.3e-3, 1.1e-3 and 2.8e-4 off
  at $t_w = a/10$, $a/20$ and $a/40$.
- On a 12-cell section, the steel of the
  [Mistrangelo benchmark](../validation/mistrangelo.md) (plan step 5.4) at its
  pinned conductivity converges to 1e-13 in 10 PCG iterations.

**The nullspace contract.** The charge balance through insulated walls is
singular, its null mode the constants. The solve adds `a V (V . x)/sum V` and
removes the volume mean after, so the gauge removes the constants and nothing
else.
- A right-hand side whose volume mean exceeds rounding, a net source, is
  refused: the solve returns NaN rather than absorbing it.
- `VariableHelmholtz.defect` reports the mean that would have been removed.
  For the charge balance it is rounding, 1e-15 of the right-hand side.

**Charge, current and work** (`tests/test_core3d.py`):
- The currents' divergence vanishes in every cell, fluid and wall, to the
  solve's 1e-13.
- The walls carry current, and the insulated outer faces carry none.
- The Lorentz work is minus the fluid's and the walls' Joule dissipation, to
  1e-9.

**The walls' Joule heat is omitted, and measured.** The thermal step deposits
the fluid's Joule heat only. On plan step 5.4's developed section (Case 1, 48
fluid cells, the pinned steel), the walls dissipate 84 % of the Joule heat:
0.029 W/m, against the 2000 W/m that heats the duct, or 1.5e-5. Deposited, it
would raise the bulk temperature by under 1e-3 K of its 10 K, far inside
Figure 6's digitizing uncertainty of 0.064 K.

**A published number.** Hunt's duct at Ha 300 has conducting Hartmann walls of
$c = 0.1$ and insulating side walls.
- Ni et al. (2007) Table 3 gives its analytic $Re\,dp/dx$, at a flow rate of 4,
  as 5854.302. The [validation index](../validation/index.md) checks it against
  the spectral solve.
- On 48 cells the thin-wall model is 0.66 % below it, converging from 1.6 % on
  32 cells.
- Solid walls of that $c$, $a/10$, $a/20$ and $a/40$ thick, sit 2.3 %, 0.64 % and
  0.16 % below the thin-wall model, approaching it at second order in the
  thickness. That is the finite thickness's order-1 % effect.

This runs with the scheduled tests.

**Derivatives.** A solid wall's electrical conductivity is a leaf of the charge
solve, put into the cells in the trace. Its derivative is tested to the
derivative standard, with no recompilation for a new value. The steady solve
builds its walls as constants: a gradient of the pressure drop in a wall's
conductivity is planned with the wall-conductance work. A traced conductivity
given to `SolidWall` is refused.

## Add an analytic field

`make_divergence_free_cross_section_field` gives a compact analytic field on
the cross-section. Sample it with `sample_cross_section_field` before a
production run, and check its divergence and its intended extrema.

```python
from lmhdx.cases import make_divergence_free_cross_section_field, sample_cross_section_field

analytic = make_divergence_free_cross_section_field(width=2.0, height=2.0, base_bz=5.0)
y, z, sampled = sample_cross_section_field(analytic, width=2.0, height=2.0, ny=65, nz=65)
divergence = (np.gradient(sampled[..., 1], y, axis=0, edge_order=2)
              + np.gradient(sampled[..., 2], z, axis=1, edge_order=2))  # a finite-difference estimate
print(f"max |div B| {np.abs(divergence).max():.1e}, Bz from {sampled[..., 2].min():.3f} to {sampled[..., 2].max():.3f}")
```

```text
max |div B| 3.5e-05, Bz from 4.250 to 5.750
```

The divergence is the finite differences' error, small against a field of 5.

Pass any such callable to a `Problem` as `lmhdx.fields.Func(fn, params)`, a
function `fn(x, y, z, params)` that returns the three components at the cell
centres. The older `MagneticFieldSpec(kind="analytic", fn=...)` accepts a
callable that returns a three-component field.

```python
import lmhdx
from lmhdx.fields import Func


def field(x, y, z, params):
    return np.moveaxis(np.asarray(analytic(y, z)), -1, 0)


result = lmhdx.solve(lmhdx.duct(20, 24).replace(B=Func(field)))
print(f"flow rate {float(result.flow_rate):.6g}")
```

```text
flow rate 0.400012
```

## Use measured field data

Store Cartesian coordinates (metres) and components (tesla) in an NPZ table.
This synthetic affine example is divergence-free; replace its arrays with your
measured or exported data. Include the conducting walls in the table's domain.

```python
from pathlib import Path

from lmhdx.cases import write_tabulated_field_npz, sample_tabulated_cross_section_field
from lmhdx.fields import Table

y = z = np.linspace(-1.2, 1.2, 17)
yy, zz = np.meshgrid(y, z, indexing="ij")
path = write_tabulated_field_npz(
    Path("artifacts/imposed_field.npz"), y=y, z=z,
    bx=np.zeros_like(yy), by=0.1 * yy, bz=5.0 - 0.1 * zz,
)
sampled = sample_tabulated_cross_section_field(path, y=yy, z=zz)
assert np.allclose(sampled[..., 2], 5.0 - 0.1 * zz)
problem = lmhdx.duct(20, 24).replace(B=Table(str(path)))
```

The rules for a table:
- A file with an `x` axis and components shaped `(nx, ny, nz)` is refused by
  the cross-section sampler. A field that varies along the duct is an
  `lmhdx.core3d.ImposedField` on the 3-D grid (see the
  [fringe tutorial](fringing.md)).
- Axes must be finite, strictly increasing vectors with at least two points,
  and the component shapes must match.
- Out-of-domain or nonfinite queries raise `ValueError`; they are never
  silently extrapolated.
- Loading a file is host-side setup, not a differentiable live coil or geometry
  interface. Keep the source's provenance, the interpolation error and
  independent Maxwell checks with each run.
