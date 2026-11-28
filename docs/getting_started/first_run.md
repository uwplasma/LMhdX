# Run a first duct case

This page solves a square duct in a magnetic field, the Hartmann duct, and
checks the answer against its exact profile. You do it twice: from Python and
from a TOML file on the command line. Both take a second or two on a laptop.

```{image} ../_static/validation_ladder.webp
:alt: Hartmann-layer collapse, flow-rate error against a spectral reference, and mesh convergence
:align: center
```

## From Python

The steps below are the script `examples/hartmann_example.py`, cut into pieces.

**1. Set up the problem.** `lmhdx.duct(ha, cells)` is the square insulating
duct of unit half-width at Hartmann number `ha`, driven by a unit pressure
gradient, with the field along $y$:

```python
import lmhdx

lmhdx.configure(dtype="float64")
problem = lmhdx.duct(20.0, 24)
```

Properties are nondimensional: density, viscosity and conductivity are one
unless you change them. [Problems, solvers and one verb](problem.md) lists every
field of a `Problem` and the other presets.

**2. Solve it.** One call runs the solve; the `Output` writes a restart record,
a ParaView file, a summary and a log into `artifacts/hartmann`:

```python
solution = lmhdx.solve(problem, output=lmhdx.Output("artifacts/hartmann"))
print(f"residual {float(solution.residual):.2e}, flow rate {float(solution.flow_rate):.6g}")
```

```text
residual 1.25e-08, flow rate 0.156144
```

The residual is that of the certified solve; the iterative residual varies a
little with the platform.

**3. Check it.** Compare the velocity along $y$ at the midplane $z=0$ with the
exact Hartmann profile $1-(\cosh(Ha\,y)-1)/(\cosh Ha-1)$:

```python
import numpy as np

faces = np.asarray(problem.grid.faces[1])
y = 0.5 * (faces[1:] + faces[:-1])                     # cell centres across the field
u = np.asarray(solution.velocity[0].data[0])
profile = u[:, 11:13].mean(axis=1)                     # the two cells beside z = 0
profile /= profile.max()
reference = 1.0 - (np.cosh(20.0 * y) - 1.0) / (np.cosh(20.0) - 1.0)
error = profile - reference
print(f"L2 {np.sqrt(np.mean(error**2)):.4f}, Linf {abs(error).max():.4f}")
```

```text
L2 0.0064, Linf 0.0098
```

These are the errors that [validation](../validation/index.md) quotes for this
case. The full script also writes them beside the solved fields:

```console
python examples/hartmann_example.py
```

## From a TOML file

The command line reads the same objects from a TOML file, in which a table is a
constructor call. `examples/hartmann_case.toml` is the same duct on 32 cells:

```toml
schema_version = 3
name = "hartmann_ha20_toml"
units = "nondimensional"

[problem]
preset = "duct"
ha = 20.0
cells = 32

[output]
dir = "../artifacts/examples/toml_hartmann"
```

Run it:

```console
lmhdx examples/hartmann_case.toml
```

The command writes the files of its `[output]` table (a restart record, a
ParaView file, the summary and the log), prints the summary and exits nonzero
if the solve did not converge. In the summary, `"flow_rate"` reads 0.154647
and `"residual"` is about 5e-09.

Override any key of the file with `table.key=value`, for example a stronger
field:

```console
lmhdx examples/hartmann_case.toml problem.ha=40
```
