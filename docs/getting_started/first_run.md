# Run a first duct case

Create one problem, solve it, and inspect the residual of the certified solve:

```python
import lmhdx

problem = lmhdx.duct(5.0, 8)
result = lmhdx.solve(problem)

assert result.residual < 1e-7, result.residual
print(f"flow rate={float(result.flow_rate):.4f} residual={float(result.residual):.3e}")
```

`lmhdx.duct(ha, cells)` is the square insulating duct of unit half-width at Hartmann number `ha`, driven by a
unit pressure gradient; [Problems, solvers and one verb](problem.md) lists every field of a `Problem` and the
other presets. Properties are nondimensional: density, viscosity and conductivity are one unless you change them.

The command line reads the same objects from a TOML file, in which a table is a constructor call:

```console
lmhdx examples/hartmann_case.toml
lmhdx examples/hartmann_case.toml problem.ha=40
```

The command writes the files of its `[output]` table (a restart record, a ParaView file, the summary and the
log), prints the summary and exits nonzero if the solve did not converge. A key of the file is overridden with
`table.key=value`.

For an editable complete workflow, run:

```console
python examples/hartmann_example.py
```

The script compares the numerical midplane profile with the analytical Hartmann profile and writes the errors
beside the solved fields.
