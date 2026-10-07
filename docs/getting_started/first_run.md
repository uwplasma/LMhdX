# Run a first duct case

Create one physical case, solve it, and inspect the explicit termination state:

```python
import lmhdx

case = lmhdx.make_hartmann_case(ha=5.0, ny=8, nz=8)
result = lmhdx.solve(case)

assert result.converged, result.status
print(f"status={result.status} residual={result.residual:.3e}")
```

`CaseSpec` groups geometry, regions, magnetic field, boundary conditions,
drive, solver controls, and output policy. All viscosities in a case are
kinematic viscosities in m²/s; the LMhdX helpers convert and form the
dimensionless groups.

The command line uses the same schema:

```console
lmhdx examples/hartmann_case.toml
```

By default a case writes NPZ, JSON, CSV profiles and VTK, and no plots; the
case's `[output]` settings switch each one. A steady command exits nonzero if its convergence gates
are not met.

For an editable complete workflow, run:

```console
python examples/hartmann_example.py
```

The script compares the numerical centerline with the analytical Hartmann
profile and writes the errors beside the solved fields.
