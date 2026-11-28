# Restart and write output

This page shows how to save a run to disk and pick it up again. `lmhdx.Output`
writes restart records, ParaView files, a summary and a log. A steady solve
writes its fields once; an unsteady run writes them every chunk. A restarted
run gives the same answer, bit for bit, as one that never stopped.

Pass an `Output` as the third argument of `solve`, and a record as `start=`:

```python
import lmhdx

lmhdx.configure(dtype="float64")
problem = lmhdx.duct(20.0, 24)

solution = lmhdx.solve(problem, lmhdx.Solver(dt=2e-3, steps=40), lmhdx.Output("out", every=10))
first = lmhdx.Solver(dt=2e-3, steps=20)
lmhdx.solve(problem, first, lmhdx.Output("out"))                 # out/run_000020.npz
rest = lmhdx.solve(problem, first, start="out/run_000020.npz")   # the same state as solution
```

For repeatable studies, prefer the NPZ records plus the JSON summary. Set
`paraview=False` unless a visualization tool needs the files, and keep
generated output outside the repository.

## An unsteady run

`lmhdx.solve(problem, solver, Output("out", every=20))` runs in compiled chunks
of 20 steps. It writes these files for every kind of run (a projection run, a
heated one, Q2D), and once for a steady solve:
- the record `run_000020.npz` after each accepted chunk: the state, exactly,
  with the identity of the problem and solver that wrote it;
- the ParaView frame `run_000020.vtr` (`.vts` for a polar grid, at the points
  $(r\cos\theta, r\sin\theta, z)$), gathered in `run.pvd`.
  `fields=("velocity",)` limits the arrays. Stretched grids need nothing extra:
  the rectilinear file holds the face coordinates;
- `run_summary.json`, whose finite flag, last and largest of each diagnostic
  are folded chunk by chunk, so that it does not grow with the run;
- the solver's log, `run.log` (the banner, a block per chunk with its progress
  and the time remaining, the footer), beside a JSON line per chunk in
  `run.jsonl`. The log also goes to stderr, so the command's stdout is the
  summary alone.

The snippet above leaves `run.jsonl`, `run.log`, `run.pvd`, `run_summary.json`
and a record and a frame at steps 10, 20, 30 and 40 in `out/`.

`Output(name=)` names the files (a case's `name` in a TOML file). Every file
carries the run's `name`, so two runs in one directory, or a resume, do not
overwrite each other. A run in chunks is the run in one, bit for bit, and so is
a run restarted from a record, because a step boundary is the whole state.

## Records and restarts

A record is a NPZ file of the state's arrays, its step and an identity hash.
`start=path` resumes. It refuses:
- a record written for another problem or solver (the run's length excluded);
- a leaf of another shape or dtype;
- any leaf that is not finite.

`lmhdx.output.Start(path=..., mode="import")` takes the state of any record of
the same fields, for another experiment. A rejected chunk's state is written as
`run_000030_rejected.npz` and imports only.

## History

With an `Output` the `Solution.history` is the last chunk's, so that a long
run's memory does not grow; the whole series is in the `.jsonl` file. A history
stride must divide `every` and `steps` there; without an `Output` the kernel's
own rule applies. Every step of a heated run is checked against its limits,
whatever the history stride, which only thins the history returned; a rejection
reports the first step past a limit.

## Heated runs

A `lmhdx.thermal.ThermalProblem`, with solid walls if it has them, goes to
`solve` too: Joule heating alone, at its flow's `dt`, from rest, the `"steady"`
flow or a `ThermalState`.
- Its records and ParaView cells are on `thermal_grid`, the walls' cells
  included. The velocity, and an evolving flow's pressure, are zero in a wall.
- A record resumes it bit for bit. The record holds the state's pressure, which
  starts the next pressure solve, so a thermal record written before the state
  carried it does not resume.
- A `ThermalState` start at time $t > 0$ numbers its files from step
  $t/\Delta t$.

## Without an Output

Without an `Output` a run does no I/O. `write_vtk(path, grid, **cells)` writes
named cell arrays directly, x varying fastest; `write_pvd` gathers frames into a
time series.
