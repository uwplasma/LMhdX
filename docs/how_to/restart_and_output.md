# Restart and write output

This page shows how to save a run to disk and pick it up again. `lmhdx.Output`
writes restart records, ParaView files, a summary and a log.

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
of 20 steps. After each chunk it writes these files, for projection, heated and
Q2D runs alike. A steady solve writes them once.
- the record `run_000020.npz`, after each accepted chunk, which holds the exact
  state and the identity of the problem and solver that wrote it;
- the ParaView frame `run_000020.vtr`, gathered in `run.pvd`. A polar grid
  writes `.vts` instead, at the points $(r\cos\theta, r\sin\theta, z)$.
  `fields=("velocity",)` limits the arrays. Stretched grids need nothing extra,
  because the rectilinear file holds the face coordinates;
- `run_summary.json`, which holds a finite flag and the last and largest value
  of each diagnostic. It does not grow with the run;
- the solver's log, `run.log`, with the banner, a block per chunk with its
  progress and the time remaining, and the footer. Beside it, `run.jsonl` holds
  a JSON line per chunk. The log also goes to stderr, so the command's stdout is
  the summary alone.

The snippet above leaves `run.jsonl`, `run.log`, `run.pvd`, `run_summary.json`
and a record and a frame at steps 10, 20, 30 and 40 in `out/`.

`Output(name=)` names the files. In a TOML file, the case's `name` sets it.
Every file carries the run's `name`, so two runs in one directory, or a resume,
do not overwrite each other. A run in chunks, or one restarted from a record,
matches the same run in one chunk bit for bit, because only the state at a step
boundary carries over to the next step.

## Records and restarts

A record is a NPZ file of the state's arrays, its step and an identity hash.
`start=path` resumes. It refuses:
- a record written for another problem or solver, ignoring the run's length;
- a leaf of another shape or dtype;
- any leaf that is not finite.

`lmhdx.output.Start(path=..., mode="import")` takes the state of any record of
the same fields, for another experiment. LMhdX writes a rejected chunk's state
as `run_000030_rejected.npz`, and you can only import that record.

## History

With an `Output`, `Solution.history` holds only the last chunk, so a long run's
memory does not grow. The whole series is in the `.jsonl` file. There, a
history stride must divide `every` and `steps`. Without an `Output`, the
kernel's own rule applies. LMhdX checks every step of a heated run against its
limits, whatever the stride, and a rejection reports the first step past a
limit.

## Heated runs

`solve` also takes a `lmhdx.thermal.ThermalProblem`, with solid walls if it has
them. It runs unsteady at its flow's `dt` with Joule heating alone, and it
starts from rest, from the `"steady"` flow or from a `ThermalState`.
- Its records and ParaView cells are on `thermal_grid`, including the walls'
  cells. The velocity, and an evolving flow's pressure, are zero in a wall.
- A record resumes it bit for bit. The record holds the state's pressure, which
  starts the next pressure solve. A thermal record written before the state
  carried the pressure does not resume.
- A `ThermalState` start at time $t > 0$ numbers its files from step
  $t/\Delta t$.

## Without an Output

Without an `Output`, a run does no I/O. `write_vtk(path, grid, **cells)` writes
named cell arrays with x varying fastest, and `write_pvd` gathers frames into a
time series.
