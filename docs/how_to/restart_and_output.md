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
case's `OutputSpec`. Prefer NPZ plus JSON for repeatable studies. Enable VTK
only when a downstream visualization tool needs it, and keep generated output
outside the repository.

Full iteration histories cost memory. `OutputSpec(history_stride=0)` keeps the
terminal sample, which is the default. Set a positive stride to retain the
first sample, every requested interval, and the terminal sample; use `1` only
when every iteration is needed. Positive-stride restart segments preserve
retained samples and add samples from the resumed segment; stride `0` keeps
only the latest terminal. Restart state and compact diagnostics are independent,


## A thermal run

A thermal state is resumed from a versioned record. The record holds the
staggered velocity, the temperature and the time, exactly:

```python
from lmhdx.thermal import load_thermal_record, write_thermal_record
from lmhdx.io import write_thermal_vtr

path = write_thermal_record("state", problem, state, units="SI")  # state.npz
state = load_thermal_record(path, problem, units="SI")
write_thermal_vtr(problem, state, "frame")  # frame.vtr, for ParaView
```

The record is a NPZ file with this contents:
- its schema (`lmhdx.thermal.state`) and version;
- a units label, which the caller declares and the loader checks;
- the problem it was written with, as plain data.

`load_thermal_record` refuses a record in these cases:
- its schema, version or units differ, or it was written for another problem.
  When the problem differs, the error names the parts that differ;
- a field is missing, mangled or not finite, or the time is not a finite
  scalar of the compute dtype. These are checked before any state is made.

A record restores the prognostic state exactly. The step's other inputs, such
as the heat source and whether the flow is held, are the caller's, so a load
has two meanings:
- **An exact resume.** Pass the same `run` provenance (plain data) to both
  `write_thermal_record` and `load_thermal_record`; a record written by a run
  with other inputs is refused. A run of six steps and a run of three, through a
  record, then three more, agree bit for bit.
- **A state import.** Without `run`, the state becomes the start of another
  experiment.

`write_thermal_vtr` writes a rectilinear grid with two cell arrays:
- the temperature;
- the velocity, as each component's mean over the two faces of its cell.

The cells are in VTK's order, x varying fastest. `write_pvd` gathers frames
into a time series.
