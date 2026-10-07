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

