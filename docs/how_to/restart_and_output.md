# Restart and write output

A fully developed solve writes a typed restart bundle that records its state,
diagnostics and mesh.

```python
from lmhdx.io import write_restart_npz, load_restart_bundle, validate_restart_bundle

write_restart_npz(result, case, "restart.npz")
restart = load_restart_bundle("restart.npz")
validate_restart_bundle(restart, case=case)
```

The validation rejects a mismatched geometry, mesh shape or case before a
restart enters the solver. A staggered-core trajectory is resumed from its
state with `lmhdx.advance`, which takes a run in chunks.

`write_solution_outputs` honors the
case's `OutputSpec`. Prefer NPZ plus JSON for repeatable studies. VTK is on by default; set
`write_paraview=False` unless a downstream visualization tool needs it, and keep generated output
outside the repository.

Full iteration histories cost memory. `OutputSpec(history_stride=0)` keeps the
terminal sample, which is the default. Set a positive stride to retain the
first sample, every requested interval, and the terminal sample; use `1` only
when every iteration is needed. Positive-stride restart segments preserve
retained samples and add samples from the resumed segment; stride `0` keeps
only the latest terminal. Restart state and compact diagnostics are independent.


## An unsteady run

`lmhdx.solve(problem, solver, Output("out", every=20))` runs in compiled chunks of 20 steps and writes, for
every kind of run (a projection run, a heated one, Q2D) and, once, for a steady solve:
- the record `run_000020.npz` after each accepted chunk: the state, exactly, with the identity of the problem and
  solver that wrote it;
- the ParaView frame `run_000020.vtr` (`.vts` for a polar grid, at the points $(r\cos\theta, r\sin\theta, z)$),
  gathered in `run.pvd`; `fields=("velocity",)` limits the arrays. Stretched grids need nothing extra: the
  rectilinear file holds the face coordinates;
- `summary.json`, and the solver's log (`run.log`: the banner, a block per chunk with its progress and the
  time remaining, the footer) beside a JSON line per chunk in `run.jsonl`.

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

Without an `Output` a run does no I/O. `write_vtk(path, grid, **cells)` writes named cell arrays directly,
x varying fastest; `write_pvd` gathers frames into a time series.
