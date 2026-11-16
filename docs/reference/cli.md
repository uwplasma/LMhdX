# CLI and TOML reference

LMhdX runs a TOML file, converts an old one, and nothing else. The command is installed as `lmhdx`
(the `lmx` alias is gone):

```console
lmhdx CASE.toml [table.key=value ...]
lmhdx upgrade OLD.toml > CASE.toml
```

The `run`, `validate` and `benchmark` commands of 1.x are gone: write the
duct as a schema 3 file (`preset = "duct"`) and run it. A duct with an inlet and an outlet is a
`fringe_duct` preset in the same file.

## TOML schema 3

A table is a constructor call: `[problem]` is `Problem(**table)`, `[solver]` is `Solver(**table)`, a
nested table is a nested object, and `kind` picks the class where several fit. Python and TOML build
the same objects, so a file and a script give the same result bit for bit. `examples/heated_duct.toml`
is the template:

```{literalinclude} ../../examples/heated_duct.toml
:language: toml
```

| Table | Is | Notes |
|---|---|---|
| `[problem]` | `Problem(**table)`, or `preset = "duct"` with the arguments of `duct`, `channel`, `fringe_duct`, `pipe` or `q2d` | `kind = "q2d"` is the Q2D problem |
| `[problem.grid]` | `Grid(x, y, z)` | each axis is a list of faces or the arguments of `lmhdx.grid.axis` |
| `[problem.bc]` | one entry per axis | `"periodic"`, `"wall"`, or a table: `Wall` keys, `lower`/`upper` for `Sides`, `kind = "open"` |
| `[problem.B]` | `kind` is `uniform` (`value = [bx, by, bz]`), `fringe`, `table` or `func` | a `func` has `fn = "module:function"` or `"file.py:function"` and `params` |
| `[problem.thermal]` | `Thermal(**table)` | |
| `[solver]` | `Solver(**table)` | |
| `[start]` | `Start(**table)`: `state = "rest"` or `"steady"`, or `path` of a record and its `mode` | `"resume"` (default) or `"import"` |
| `[output]` | `Output(**table)`: `dir`, `name`, `every`, `fields`, `paraview`, `restart`, `summary`, `log` | files of a run; see [Restart and write output](../how_to/restart_and_output.md) |

Relative paths (`Table.path`, `file.py:function`, `output.dir`, `start.path`) resolve against the file. An unknown key, a missing
key, a wrong TOML type and an unknown `kind` are refused before anything is solved, with the table path
in the message: an integer stands for a number, a float or a boolean never for an integer. A run with an
`[output]` prints its log and the summary, and exits 2 if a chunk was rejected.

```console
lmhdx examples/heated_duct.toml solver.steps=20 problem.B.value=[0,20,0]
lmhdx upgrade old.toml > new.toml
```

Any `table.key=value` after the file replaces a value (the value is TOML; a bare word is a string).
`lmhdx upgrade` converts a schema 1 (thermal) or schema 2 (case) file; a schema 1 file's `[run]`, `[output]` and
`[restart]` become `[solver]`, `[output]` and `[start]`, what schema 3 cannot hold (a held flow, a schema 2 file's
output, logging and restart) is listed as comments, and a schema 2 case other than a
rectangular duct of one fluid with a constant field and insulating walls is refused with the tables to
write by hand. From Python:

```python
import lmhdx
from lmhdx.config import dumps, load

case = load("examples/heated_duct.toml")      # a Case: problem, solver, start, name, output
solution = lmhdx.solve(case)
print(dumps(case.problem, case.solver))       # the same text back: any problem without a lambda
```

## Schema 1 and 2 files

A file with `[solver] kind = "thermal"` is schema 1, and one with `[case]`, `[geometry]` and `[time_stepper]`
is schema 2. Neither runs under 2.0: `lmhdx upgrade FILE` writes the schema 3
text, and the migration guide maps every other 1.x name to its replacement (see [Heat transport](../physics/thermal.md)
for the heated runs).
