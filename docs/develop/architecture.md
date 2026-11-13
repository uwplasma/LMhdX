# Architecture

LMhdX is organized by physical ownership:

```text
Problem + Solver ──> solve ──> the kernels below ──> Solution ──> Output (records, ParaView, summary, log)
ChannelProblem ──> staggered core (grid, ops, poisson) ──> steady / axial / core3d steps
PipeProblem ──> the developed pipe on its polar grid
Q2DProblem ──> vorticity dynamics ──> SOLVAX periodic Poisson
ThermalProblem ──> the core's stages + conduction ──> advance_thermal, developed_heat
CaseSpec ── fully developed ──> the core (the older entry, removed in 2.0.0)
```

`lmhdx.problem`, `lmhdx.solver`, `lmhdx.fields`, `lmhdx.config` and
`lmhdx.output` are the 2.0 interface (ADR 0010): `Problem` states the physics
with its differentiable numbers as leaves, `Solver` the method, `solve`
dispatches to a kernel, `Output` says where a run goes, and a TOML table is a
constructor call. An unsteady run is a loop of compiled chunks; each accepted
chunk writes a record of the state, exactly, with a digest of the problem and
solver, and a ParaView file, and the summary and log are folded chunk by chunk,
so a run in chunks or restarted from a record is the run in one, bit for bit.
Imports are one-way: the kernels do not import the interface, and plotting
dependencies load only when an output function requests them.

`lmhdx.grid`, `lmhdx.ops` and `lmhdx.poisson` are the staggered grid, its
operators and the direct and preconditioned solves. `lmhdx.core3d` and
`lmhdx.steady` are the 3-D kernel, with the projection and RK3 steps;
`lmhdx.axial` adds an inlet and an outlet, with the steady root by the RK3
march or by Newton, `lmhdx.pipe` the developed pipe and `lmhdx.coreflow` the
inertialess core-flow model. `lmhdx.q2d` is the quasi-two-dimensional model.
`lmhdx._programs` compiles one program per problem shape.
`lmhdx.fully_developed`, `lmhdx.cases` and `lmhdx.io` hold the older case
entry, its builders, reports, figures and commands until 2.0.0.

`lmhdx.thermal` (ADRs 0008, 0009 and 0011) composes two private stages of
`lmhdx.core3d`, one electric solve for the Lorentz force and the Joule heat,
then the momentum stage with an optional buoyancy force, and adds conduction:
the direct fast-diagonal inverse for one material, or with solid walls the
preconditioned conjugate gradients of `poisson.variable_helmholtz` on the
extended grid of `core3d.conjugate_grid`, which also carries the potential.
`lmhdx.core3d` does not import it.

Each concept has one implementation; `plan.md` Section 5 lists the concepts,
their one place and the duplicates still present with the step that removes
each (ADR 0012).

Package sources live under `src/lmhdx`, so an editable installation and a wheel
resolve the same module tree. The wheel includes `lmhdx/py.typed`; every root API
callable has an explicit signature, and distribution audits require the typing
marker and reject files outside the package and metadata roots.

Reusable algebra belongs in SOLVAX when it is independent of LMhdX geometry,
units, boundaries, and terminology and has its own correctness, gradient,
convergence, documentation, and performance tests. LMhdX retains coefficient
assembly, the interface physics of material faces, gauges that express
physical constraints, coupling, and physical acceptance residuals.

FreeMHD execution and per-benchmark solvers are repository tooling under
`validation/` and are excluded from the runtime wheel. Only benchmark
specifications and compact reference arrays ship as package data.

Structural budgets are enforced by `scripts/audit_architecture.py`: tracked
size, source/module count, test/script/example count, root API size, import
latency, and distribution contents. A budget rises only as a loan with a named
repayment step (`plan.md` H.10.8).
