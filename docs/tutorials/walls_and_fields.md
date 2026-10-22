# Walls and imposed fields

This tutorial builds explicit wall stacks and imposed magnetic fields, then
uses the same case/result model as the named duct workflows.

## Wall layers

State each layer's thickness, resolution and conductivity with `WallLayer`, then
give the solver the stack as wall cells: one conductivity ratio (wall over fluid)
and one width per cell, from the fluid outwards. `ChannelProblem(wall_layers=...)`
takes one stack per end of each wall-normal axis; the staggered core resolves the
currents in it, with interface conductance from the adjacent material values and
face distances.

```python
from lmhdx import WallLayer

fluid_conductivity = 3.2e6
layers = (
    WallLayer(name="steel", thickness=2e-3, cells=3, conductivity=8e5),
    WallLayer(name="insulator", thickness=5e-4, cells=2, conductivity=1e-8),
)
ratios = tuple(layer.conductivity / fluid_conductivity for layer in layers for _ in range(layer.cells))
widths = tuple(layer.thickness / layer.cells for layer in layers for _ in range(layer.cells))
stack = (ratios, widths)
# ChannelProblem(..., wall_layers=(None, (stack, stack), (stack, stack)))
```

`examples/li_aln_wall_stack_example.py` solves a full Li | AlN | 316L duct this
way. Inspect the regime with `wall_conductance_ratio`,
`tangential_stack_conductance_ratio`, and `normal_stack_leakage_ratio`.

## Solid walls

Wall layers solve the potential separably, so they reach two axes only when the
third has one cell. Solid walls lift that limit (plan step 5.4b, ADR 0009 D36).
`ChannelProblem(solid_walls=...)` takes a `core3d.SolidWall` or `None` at each
end of the wall-bounded axes 1 and 2: a thickness in equal cells, with its
`electrical_conductivity` (absolute, not over the fluid's) beside the
`thermal_conductivity` and `heat_capacity` a thermal problem uses. The walls
become cells of an extended grid (`core3d.conjugate_grid`), and the charge
balance is solved through fluid and walls by `solvax.pcg_linear_solve`,
preconditioned by the fast-diagonal Poisson inverse
(`poisson.variable_helmholtz` with no shift). The balance is singular, its null
mode the constants; a rank-one term on them makes it positive definite, and the
volume mean is removed after. The velocity and pressure stay on
the fluid's grid. The currents into the walls heat the fluid's half cells and
are left out of the Lorentz force, as a resolved wall's are; the walls' own
Joule heat is not deposited.

```python
from lmhdx.core3d import SolidWall

steel = SolidWall(
    2e-3, 4, thermal_conductivity=18.3, heat_capacity=4.27e6, electrical_conductivity=1.04e6
)
# ChannelProblem(..., solid_walls=((None, None), (steel, steel), (steel, steel)))
```

On a section one cell long, solid walls of the same cells give the resolved
path's potential, currents and powers to 1e-9, corners included, and its
developed flow rate to 1e-8 (`tests/test_core3d.py`). Thinned at a fixed
conductance ratio $c = \sigma_w t_w/(\sigma a)$, Hartmann walls alone approach
the thin-wall model at second order in $t_w$: the flow rate of Hunt's duct at
Ha 20 and $c = 0.05$ is 4.3e-3, 1.1e-3 and 2.8e-4 off at $t_w = a/10$, $a/20$
and $a/40$. On a 12-cell section, the steel of plan step 5.4 at its pinned
conductivity converges to 1e-13 in 10 PCG iterations.

A corner cell, in walls of both axes, takes the nearer wall's material, and the
first axis's where the two are equally near. Walls of equal cells on both axes
tie exactly there; before 5.4b the resolved path split those ties by rounding,
unevenly about the duct's middle, which moved a mean velocity by up to 1.4e-4
where the two walls' materials differ.

## Add an analytic field

`MagneticFieldSpec(kind="analytic", fn=...)` accepts a callable that returns a
three-component field. `make_divergence_free_cross_section_field` provides a
compact analytic example. Sample the function with
`sample_cross_section_field` before a production run and verify its divergence
and intended extrema.

## Use measured field data

Store Cartesian coordinates (metres) and components (tesla) in an NPZ table.
This synthetic affine example is divergence-free; replace its arrays with your
measured/exported data. Include the conducting walls in the table's domain.

```python
from pathlib import Path
from dataclasses import replace
import numpy as np
from lmhdx import make_hartmann_case
from lmhdx.cases import write_tabulated_field_npz, sample_tabulated_cross_section_field
from lmhdx.cases import MagneticFieldSpec

y = z = np.linspace(-1.2, 1.2, 17)
yy, zz = np.meshgrid(y, z, indexing="ij")
path = write_tabulated_field_npz(
    Path("artifacts/imposed_field.npz"), y=y, z=z,
    bx=np.zeros_like(yy), by=0.1 * yy, bz=5.0 - 0.1 * zz,
)
sampled = sample_tabulated_cross_section_field(path, y=yy, z=zz)
assert np.allclose(sampled[..., 2], 5.0 - 0.1 * zz)
case = replace(make_hartmann_case(), magnetic_field=MagneticFieldSpec(
    kind="tabulated", table_path=str(path),
))
```

A file with an `x` axis and components shaped `(nx, ny, nz)` is refused by the
cross-section sampler; a field that varies along the duct is an
`lmhdx.core3d.ImposedField` on the 3-D grid (see the
[fringe tutorial](fringing.md)). Axes must be finite, strictly increasing vectors with at
least two points; component shapes must match. Out-of-domain/nonfinite queries
raise `ValueError`, never silently extrapolate. The file-loading interface is
host-side setup, not a differentiable live coil/geometry interface. Keep source
provenance, interpolation error and independent Maxwell checks with each run.

Run `python examples/li_aln_wall_stack_example.py` for explicit conducting and
insulating layers: each wall is a stack of cells of its own conductivity
(`ChannelProblem.wall_layers`), solved on the staggered core with the corner
cells taking the nearer wall's layer.
