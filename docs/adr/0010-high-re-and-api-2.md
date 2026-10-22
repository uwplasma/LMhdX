# 0010 — High Reynolds number: one stepper, and the 2.0 API

Status: accepted 2026-10-21 by the owner; adds Program H (phases H0–H6) to
`plan.md` Section 6 and amends nothing above it.

Basis: `main` at `d137209` (LMhdX 1.11.0, SOLVAX 0.31.0); a profile and accuracy
audit of both packages; a survey of the source of CaNS, CaNS-EIGEN, Oceananigans,
jax-cfd, JAX-Fluids, PICT, Dedalus and FreeMHD; the papers listed in `plan.md`
H.9, read in full; prototypes of the march, the stepper and the kinetic-energy
fix outside the repository. Numbers are in `plan.md` H.0 and Table H.T.

**DH1. One stepper for steady and unsteady flow.** The step is the IMEX
Runge–Kutta scheme of Spalart, Moser and Rogers (alpha 8/15, 5/12, 3/4; beta 0,
−17/60, −5/12) with Crank–Nicolson viscous and Lorentz-damping terms in the
Helmholtz operator, central explicit transport, and one projection per stage on
`psi = gamma_k dt Phi`. A time-dependent run integrates it to a final time; a
steady run integrates it to a fixed point (`march`), with the residual as the
stopping test and three outcomes: converged, stagnated, unsteady. Newton–Krylov
stays as an optional polish and for a converged tangent. Reasons: the Newton
iteration count is a linear-solver problem that grows faster than Re
(31, 403, 4,969 outer iterations at Re 10, 20, 40), while the march costs 200 to
1,700 cheap steps over the same range; the step is the one a DNS needs; the
imaginary-axis stability limit of the stage map is sqrt 3, so central transport
is stable where forward Euler is not, and the 1e-4 limited-flux bias of the
present march disappears. Restart is at a step boundary (beta_1 = 0), so no
stage state is stored. Euler is the one-stage tableau and stays bit-identical.

**DH2. Solvers: the separable dense eigenbasis stays; generic numerics go to
SOLVAX.** The method is that of CaNS-EIGEN and of the Ilmenau duct codes. On a
CPU the dense contraction beats FFT and tridiagonal solves up to N of order
10³ (128³: 1.1 ms against 1.9 and 11.4 ms). FFT or cosine transforms on uniform
axes are added only after a GPU measurement at N of at least 256. SOLVAX gets
the IMEX tableaux, `march` with its fixed-point adjoint, the separable solver,
the compile-cache infrastructure and the constrained CG; LMhdX keeps the
operators, the walls, the stage loop with its projection, the statistics and the
record I/O.

**DH3. Rejected.** Augmented Lagrangian, PCD, LSC and block preconditioners
(the exact projection already removes the pressure Schur complement); multigrid
on stretched grids (10 to 14 times slower than the direct solve in CaNS-EIGEN);
pseudo-transient continuation with SER on an explicit march (the step is capped
by the CFL; SER diverged once the shift stopped dominating); Krylov recycling
and coarse spaces (marginal, measured in #225); host direct solvers inside the
differentiable loop; approximate factorisation of the implicit operator;
explicit Lorentz force; gradients through long turbulent windows. Evidence and
numbers: `plan.md` H.7.

**DH4. API 2.0: three nouns and one verb.** The public workflow is `Problem`
(what is solved), `Solver` (how) and `Output` (where it goes), with
`solve(problem, solver=None, output=None, start=None)`. Python and TOML are
equally expressive, and one interface serves a five-line beginner script and a
user who replaces the stepper.

- `Problem` is a frozen dataclass and a registered pytree: `grid`, `bc`
  (one entry per axis), `density`, `viscosity`, `conductivity`, `B`, `forcing`,
  `flow_rate`, `ramp`, `advection`, `thermal`, `terms`. Presets `duct`,
  `channel`, `fringe_duct`, `pipe` and `q2d` only fill a `Problem`; a custom case
  never needs one. The pipe is a `Problem` on a polar grid and uses the polar
  kernel; Q2D keeps a small class of its own with the same `Solver`, `Output` and
  `solve`.
- Grids come from one dispatcher, `axis(cells, lower, upper, beta=, ratio=,
  blend=, layer_thickness=, cells_in_layer=)`, over the existing uniform,
  geometric, tanh and wall-resolving constructors plus the KZB blend
  `z = a sin(pi xi/2) + (1 − a) xi`.
- Boundaries per axis: `"periodic"`, `"wall"` or `Wall(conductance, layers,
  temperature, heat_flux)`, `Sides(lower, upper)`, `Open(profile=None)`. Hunt is
  `Wall(conductance=c)`; wall stacks are `Wall(layers=(WallLayer(...), ...))`.
- The field `B` is `Uniform`, `Fringe`, `Table` or `Func(fn, params)`, a traced
  function of position; in TOML `Func` is `{kind = "func", fn = "module:function"}`,
  which is what makes TOML as expressive as Python.
- `Solver(method=...)` is one of `stokes`, `newton`, `march` (steady),
  `rk3`, `euler`, `implicit`, `ifrk4` (unsteady) or `auto`, with `dt`, `cfl`,
  `steps`, `t_end`, `tol`, `tableau`, `linear`, `stats`, `warmup`, `checkpoint`.
  A tableau, a linear solver or the whole method can be a callable.
- Extension points: `Func` fields, `forcing=callable(t)`, `Open(profile=array)`,
  `Problem.terms` (user force densities on the velocity faces), a custom
  tableau, a custom linear solver, and `build(problem, solver)` returning a
  `Model` with `init`, `step`, `advance` and `solve` for a manual loop. The
  operators of `ops`, `core3d` and `poisson` remain the documented advanced layer.
  An induction model and a turbulence closure are reserved names and raise
  today.
- `Output(dir, every, fields, paraview, restart, profiles, summary, log, plots)`
  runs in compiled chunks of `every` steps; each accepted chunk writes a record,
  and ParaView files for every run type. The first rejected chunk stops the run
  and writes only a rejected record. A record carries a fingerprint of problem
  and solver; `start=path` resumes and refuses a different problem, `mode =
  "import"` needs only matching shapes.
- TOML schema 3: a table is a constructor call. `[problem]` is `Problem(**table)`
  or, with `preset = "duct"`, `duct(**rest)`; `[problem.grid]`, `[problem.bc]`,
  `[problem.B]`, `[problem.thermal]`, `[solver]`, `[output]` and `[start]` follow
  the same rule. An unknown key, a missing key or a wrong type is refused before
  anything is built, with the table path in the message. Relative paths resolve
  against the file; `lmhdx.dump(obj)` writes the same text; command-line
  overrides read `problem.B.value=[0,30,0] solver.steps=100`. Schemas 1 (thermal)
  and 2 (cases) are refused with a pointer to `lmhdx upgrade`, which is removed
  in 2.2.
- Names that survive keep their spelling (`viscosity`, `conductivity`,
  `forcing`, `specific_heat`, `thermal_conductivity`, `cells_in_layer`,
  `layer_thickness`, `dt`, `steps`), so a thermal TOML file migrates by moving
  tables, not by renaming keys. Every public name of the 1.x package is mapped to
  its replacement in the migration guide of H4.5.

**DH5. Differentiation.** Continuous numbers are leaves: density, viscosity,
conductivity, the field amplitude, array or parameters, the forcing, the flow
rate, the inlet profile, the ramp, the thermal numbers, the heat source, wall
temperature and flux, and the initial state. Static: the grid, boundary kinds,
scheme names, cell counts, and whether a wall conducts. **Wall conductance and
wall thickness are leaves in 2.0** (owner decision): the thin-wall operator is
assembled with `jnp`, its axis factorisation is traced, the potential solve
differentiates by `eigh` or, if that misses 1e-6 against central differences, by
an implicit `custom_vjp`; c = 0 is the insulating pattern and its gradient is
not defined. Steady solves differentiate implicitly (`stokes`, `newton` by
`root_solve`, `march` by the fixed-point adjoint); unsteady runs by `scan` with
per-step `jax.checkpoint`, in windows of a few eddy turnovers. Reverse mode costs
about the forward march. `jax.vmap` over a leaf is a design sweep; the grid is
resolved for the largest Ha it is built for. Not covered: gradients through cell
counts, two resolved walls with traced parameters, the induction model.

**DH6. Runtime and output.** Importing `lmhdx` imports no module that changes
JAX configuration: no x64 switch, no `XLA_FLAGS`, no compile cache.
`lmhdx.configure(dtype=, cache=, gpu_gemm=)` does it explicitly, and `build()`
calls the safe parts (x64 for `dtype="float64"`, announced once; matmul
precision `highest`; the persistent cache with the Darwin jaxlib exclusion). The
XLA flag that must precede the backend is set only by `configure(gpu_gemm=...)`
and by the command line. Environment variables `LMHDX_*` are replaced by flags.
The logger banner, footer and ETA text are kept and move into `Output.log`
(owner decision); the JSON-lines chunk record is added beside them.

**DH7. Release path.** 1.12.0 adds `Problem` and the equivalence tests; the
following releases add the loader, `Output`, `rk3` and `march` without touching
the old API, so `main` stays green and goldens stay bit-identical on the Euler
and Stokes paths. Warnings for old names switch on only in the PR that migrates
the tests, because pytest treats warnings as errors. Deletion is the last PR and
is 2.0.0. Intentional losses: the `validate` and `benchmark` subcommands, the
`lmx` alias, and x64 as an import side effect.

**DH8. Regime assertions carry tags.** A test of laminarisation, transition or
patterned turbulence states its domain length, resolution and scheme order,
because the threshold moves with them (2 pi box and a sixth-order code: Ha 12.25;
at least 16 pi at 128² and second order: 15; low resolution: 14). Thresholds are
reported, not gated, unless a converged value exists.

Consequences. Source size is estimated to fall from 12.8k to between about 9.5k
and 10.1k lines (the estimates come from function sizes, not written code),
SOLVAX grows by about 1.0k, modules go from 16 to about 19, and the root keeps
about 30 lazy names that are all the user workflow. The 495 tests migrate by one
mechanical codemod; the shipped TOML examples by `lmhdx upgrade`. The risks that
the program must close with measurements: the in-trace eigen denominators (the
differentiable-leaf list assumes them), the traced `eigh` of H4.2, and the
constant-flux control and statistics, which are new numerics specified here only
at the API level.
