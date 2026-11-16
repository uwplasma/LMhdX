# Restart and write output

`lmhdx.Output` writes the files of a run: restart records, ParaView files, a summary and a log. A steady solve
writes its fields once; an unsteady run writes them every chunk. Prefer the NPZ records plus the JSON summary for
repeatable studies, set `paraview=False` unless a visualization tool needs the files, and keep generated output
outside the repository.

## An unsteady run

`lmhdx.solve(problem, solver, Output("out", every=20))` runs in compiled chunks of 20 steps and writes, for
every kind of run (a projection run, a heated one, Q2D) and, once, for a steady solve:
- the record `run_000020.npz` after each accepted chunk: the state, exactly, with the identity of the problem and
  solver that wrote it;
- the ParaView frame `run_000020.vtr` (`.vts` for a polar grid, at the points $(r\cos\theta, r\sin\theta, z)$),
  gathered in `run.pvd`; `fields=("velocity",)` limits the arrays. Stretched grids need nothing extra: the
  rectilinear file holds the face coordinates;
- `run_summary.json`, whose finite flag, last and largest of each diagnostic are folded chunk by chunk, so that
  it does not grow with the run, and the solver's log (`run.log`: the banner, a block per chunk with its progress
  and the time remaining, the footer) beside a JSON line per chunk in `run.jsonl`. The log also goes to stderr,
  so the command's stdout is the summary alone. Every file carries the run's `name`, so two runs in one
  directory, or a resume, do not overwrite each other.

`Output(name=)` names the files (a case's `name` in a TOML file). A run in chunks is the run in one, bit for bit,
and so is a run restarted from a record, because a step boundary is the whole state:

```python
import lmhdx

solution = lmhdx.solve(problem, lmhdx.Solver(dt=2e-3, steps=40), lmhdx.Output("out", every=10))
first = lmhdx.Solver(dt=2e-3, steps=20)
lmhdx.solve(problem, first, lmhdx.Output("out"))                 # out/run_000020.npz
rest = lmhdx.solve(problem, first, start="out/run_000020.npz")   # the same state as solution
```

A record is a NPZ file of the state's arrays, its step and an identity hash. `start=path` resumes: it refuses a
record written for another problem or solver (the run's length excluded), a leaf of another shape or dtype, and any
leaf that is not finite. `lmhdx.output.Start(path=..., mode="import")` takes the state of any record of the same
fields, for another experiment. A rejected chunk's state is written as `run_000030_rejected.npz` and imports only.

With an `Output` the `Solution.history` is the last chunk's, so that a long run's memory does not grow; the
whole series is in the `.jsonl` file. A history stride must divide `every` and `steps` there; without an `Output`
the kernel's own rule applies. Every step of a heated run is checked against its limits, whatever the
history stride, which only thins the history returned; a rejection reports the first step past a limit.

A `lmhdx.thermal.ThermalProblem`, with solid walls if it has them, goes to `solve` too (Joule heating alone, at
its flow's `dt`, from rest, the `"steady"` flow or a `ThermalState`): its records and ParaView cells are on
`thermal_grid`, the walls' cells included, the velocity zero in a wall, and a record resumes it bit for bit.
A `ThermalState` start at time $t > 0$ numbers its files from step $t/\Delta t$.

Without an `Output` a run does no I/O. `write_vtk(path, grid, **cells)` writes named cell arrays directly,
x varying fastest; `write_pvd` gathers frames into a time series.
