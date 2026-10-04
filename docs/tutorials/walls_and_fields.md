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
