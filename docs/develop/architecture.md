# Architecture

LMhdX is organized by physical ownership:

```text
ChannelProblem ──> staggered core (grid, ops, em, poisson) ──> steady / timeloop / axial
CaseSpec ── fully developed ──> the core, steady or transient ──> Solution
Q2DProblem ──> vorticity dynamics ──> SOLVAX periodic Poisson ──> Q2DResult
```

The stable package root contains the common 2-D and Q2D workflows. Three-dimensional,
field, differentiation, output, and validation APIs live in their named
modules. Imports remain one-way: specifications and data containers do not
depend on solvers; plotting dependencies load only when an output function
requests them.

`lmhdx.core3d` and `lmhdx.steady` are the 3-D interface; `lmhdx.axial` adds
an inlet and an outlet, and `lmhdx.coreflow` the inertialess core-flow model.
`lmhdx.fully_developed` maps a `CaseSpec` onto the core; `lmhdx.cases` holds
the case builders and `lmhdx.solve`. The cell-centred solver it replaced was
removed in 1.8 (step 4.6).

Package sources live under `src/lmhdx`, so an editable installation and a wheel
resolve the same module tree. The wheel includes `lmhdx/py.typed`; every root API
callable has an explicit signature, and distribution audits require the typing
marker and reject files outside the package and metadata roots.

Reusable algebra belongs in SOLVAX when it is independent of LMhdX geometry,
units, boundaries, and terminology and has its own correctness, gradient,
convergence, documentation, and performance tests. LMhdX retains coefficient
assembly, gauges that express physical constraints, coupling, and physical
acceptance residuals.

FreeMHD execution is repository tooling and is excluded from the runtime wheel.
Only benchmark specifications and compact reference arrays ship as package data.

Structural budgets are enforced by `scripts/audit_architecture.py`: tracked
size, source/module count, test/script/example count, root API size, import
latency, and distribution contents.
