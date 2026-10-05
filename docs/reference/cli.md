# CLI and TOML reference

LMhdX accepts a TOML case directly or one of three commands. The command is
installed as both `lmhdx` and `lmx`, so `lmx CASE.toml` works the same way:

```console
lmhdx CASE.toml
lmhdx run CASE [OPTIONS]
lmhdx validate CASE [OPTIONS]
lmhdx benchmark [OPTIONS]
```

`run` accepts `hartmann`, `shercliff` and `hunt`. Use `lmhdx run --help` for
geometry, resolution, output, and logging controls.
For rectangular cases, `--ny` and `--nz` set the fluid-cell counts (48 each
by default), including fully developed Hartmann, Shercliff, and Hunt flows;
`--width` and `--height` set the fluid dimensions. Hunt wall cells are additional.

```console
lmhdx run hunt --ha 20 --ny 24 --nz 24 --plots
```

A duct with an inlet and an outlet has no command; it is solved from Python with
`lmhdx.axial` (see the fringe tutorial).

`validate` solves a Hartmann, Shercliff, or Hunt case and writes profiles and
validation metrics. `benchmark` reports cold time, warm median, and warm
coefficient of variation for a bounded Hartmann case; it is a local performance
diagnostic, not a hardware-independent performance claim.

## TOML sections

The schema maps directly to the Python dataclasses:

| Section | Python object | Purpose |
|---|---|---|
| `[case]` | `CaseSpec` | name, forcing, initial state, reference gradient |
| `[geometry]` | `GeometrySpec` | kind, dimensions, mesh resolution, wall resolution |
| `[[regions]]` | `RegionSpec` | fluid/solid density, viscosity, conductivity |
| `[magnetic_field]` | `MagneticFieldSpec` | constant, analytic, or tabulated field |
| `[[boundary_conditions]]` | `BoundaryCondition` | velocity, pressure, current, and wall conditions |
| `[solver]` | `SolverConfig` | model, mode, coupling, SOLVAX selection |
| `[time_stepper]` | `TimeStepperConfig` | step sizes, iteration limits, physical tolerances |
| `[output]` | `OutputSpec` | NPZ, JSON, VTK, CSV, plots, and diagnostic-history stride |

Start from `examples/hartmann_case.toml`. Unknown keys, inconsistent geometry,
nonphysical material values, and unsupported solver combinations fail during
configuration rather than entering the numerical solve.
