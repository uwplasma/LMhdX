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

## Revision of 2026-10-22: decisions from the independent review

Status: proposed 2026-10-22; the owner confirms by merging. DH1–DH8 above are not
edited. Where one of them is changed, the entry below says which and how. Basis:
the external report on `d137209` and SOLVAX 0.31.0 (22 pages, snapshot of 22
October 2026), its probes (quoted as the reviewer's until the PR that fixes each
finding reproduces it), and `plan.md` H.10, which has the steps, tests and
reasons.

**DH9. The momentum stage is constraint-consistent.** The stage solves `H u +
(dt/rho) G p = b`, `D u = 0`, with `H` the implicit viscous and Lorentz-damping
operator, through the matched Schur operator `D H^-1 G` and the correction `u =
w − (dt/rho) H^-1 G p`, `w = H^-1 b`. When Lorentz damping differs between
components, scaling the whole increment component by component turns a gradient
force into a non-gradient, and the ordinary projection cannot restore the rest
state (peak `abs(u)` 4.4e-2 at `B_y = 10`, dt 0.1). This amends DH1: the `psi`
projection of the RK3 stage is the isotropic limit of the stage and its starting
iterate, not a separate path, and the bit-identical `euler` requirement of DH1
and DH7 is withdrawn for the transient momentum step only (steady, Stokes and
held-flow paths stay bit-identical). LMhdX supplies `H`, `D`, `G` and the
boundaries; the generic Schur composition moves to SOLVAX once the interface
is fixed. Steps H0.6, H1.7, H2.2.

**DH10. One geometry helper.** Interpolation weights, limiter slopes and the
carrier weights of the central flux read widths from one boundary-aware
`_ghosted_widths` (wall, open, periodic wrap), and the limited face value uses
physical-distance slopes. The EMF/Lorentz adjoint pair is not touched.
Kinetic-energy conservation of central transport on stretched grids (T23) is
fixed in the same change, which absorbs H2.1. Step H0.5.

**DH11. Physical-time claims need stability, order and budget evidence.** Central
transport with forward Euler is unstable for every Courant number
(`abs(1 − i C sin theta) > 1`); an RK3 stage map has an imaginary-axis limit of
the square root of 3. A claim of time accuracy states a manufactured temporal
order, a transported Fourier mode, shear-mode growth and decay, and a kinetic
energy budget whose terms (buoyancy work, open-boundary flux, numerical
convective work) are included or declared out of scope. Physical time (`rk3`,
`euler`) and pseudo-time (`march`) are distinguished in the docs. Steps H2.7,
H2.8.

**DH12. External solvers: an opt-in SOLVAX adapter, outside the default loop.**
This refines the rejection of host direct solvers in DH3. The default path stays
native and device-resident. One adapter in SOLVAX (PETSc fieldsplit, hypre,
sparse direct including MUMPS and SuperLU_DIST) is admitted when a named
bottleneck is measured at matched physical error and complete cost including
host and device transfers, and when the wrapper actually used passes tests for a
true transpose solve, nullspace, status reporting and coefficient derivatives.
Two uses are sanctioned: a reference solve for accuracy spot checks, and a
fallback for a pressure that is not separable. No hard dependency, no import by
default. Reason: the present MUMPS adapter is sequential, SciPy SuperLU is not
SuperLU_DIST, the differentiable sparse solve is a host callback, and raw
factor-object solves do not accept traced values, so none of them is yet a
distributed or GPU-resident solver. Step H8.3.

**DH13. Oseen-aware block preconditioners are guided research.** This amends
DH3. The argument that the exact projection removes the pressure Schur
complement holds for an `H` that is the same on all components. With anisotropic
Lorentz damping the Schur complement of the stage is real; its first
preconditioner is the separable Poisson solve, with a flexible Krylov method
where inner solves are adaptive. Oseen velocity blocks, augmented-Lagrangian and
current-pressure blocks are entered only when the Schur iteration count or the
time to an accepted objective leaves a declared envelope on a fixed family (mesh
anisotropy, Ha, Re, wall contrast, length), and defaults are chosen by complete
cost. The rejections recorded in `plan.md` H.7 remain measurements of those
configurations and do not show that block smoothing or recycling cannot work.

**DH14. Derivative qualification standard.** A differentiable path is accepted
with a Taylor remainder over a step sweep, the JVP/VJP dot-product identity, a
centred-difference sweep against tighter primal and adjoint tolerances, mesh
convergence of the derivative, independent and mixed directions for each
parameter family, and explicit unsupported behaviour that raises. A derivative is
returned only from a converged residual. The eigenvalue interface is scoped to
the converged first derivative of the smallest simple eigenvalue; eigenvector,
profile and Hessian derivatives raise until an implicit normalised-eigenvector
tangent or adjoint solve exists. Float64 steady gradients keep 1e-6 relative
against central differences unless a test states another. Steps H1.6, H8.1.

**DH15. Code size: one implementation per concept, and loans with repayment.**
A size cap that rises at every PR does not constrain architecture. Each concept
(metric, gauge, reconstruction, solver) has one implementation; plotting,
digitization and movie code stay out of the package; a replaced implementation is
deleted in the PR that adopts its replacement. A source PR states the lines it
removes. A raise of an audit target is a loan with a named repayment step in
`plan.md` H.10.8, and a loan unpaid at a release tag blocks the tag. The byte
target for docs and plan moves with the docs. This replaces the earlier practice
of raising budgets by what a PR adds, and keeps the owner's principle that size
is a direction, not a number.

**DH16. Three test tiers and one performance metric.** The portable PR gate holds
small identities, analytic limits, one representative derivative per changed path
and the public workflow; a scheduled gate holds the 3-D, thermal, contrast and
directional-sensitivity studies with refinement; research campaigns hold large
Ha and Re, external-code comparisons, GPU performance and long statistical runs.
New wall-clock ratios, exact iteration counts and compiler-string expectations
go to diagnostic reports. An existing assertion moves only with the owner's word.
Performance is the synchronised warm time, cold compile and setup, and the total
time and memory to an accepted objective plus gradient, for the same physics,
accuracy and dtype on recorded hardware; milliseconds per step, time to steady
state and time to gradient are not interchangeable.

**DH17. Claims gate and benchmark discipline.** New moving-flow thermal validation
claims wait for the constrained stage and the time-accuracy evidence of DH9 and
DH11; held-flow thermal examples and documentation do not wait for high-Re work.
The Mistrangelo et al. 2025 heated conducting duct is a multi-code statistical
comparison: Case 1 is frozen and run first, Case 2 follows with burn-in,
averaging and uncertainty, agreement is judged inside a band declared before the
run, and a fitted material ratio is labelled calibration and kept apart from the
independent velocity, temperature and current observables. The ALEX B1 tap
location, the FreeMHD two-update smoke and the GPU step-throughput figures are
stated at their real scope (`plan.md` H.10.10).

**DH18. First application: a constrained heated channel in a real stellarator
exterior field.** ESSOS supplies the coil field and VMEX the finite-beta exterior
contribution once its closure and derivatives are qualified; blanket points are
outside the last closed flux surface. The study compares uniform, locally sampled
and full 3-D fields at fixed throughput and heat removal, with plasma, coil,
clearance, duty, pitch and temperature constraints so that no weaker field, lower
heat removal or unconstrained area passes for an optimum, and adds live
coil-parameter derivatives after the field and material gradients (H8.1) work.
Blankets, tritium (OpenMC, LMhdX, FESTIM), wall-material interaction and high
Re and Ha follow as separate studies (`plan.md` H9.2–H9.5), each with its own
entry condition. Step H9.1.

Consequences of the revision. Two repairs, the stage-1 and stage-2 PRs, come
before any RK3 code, and H2.2 starts from the corrected stage. SOLVAX gains the
flux-form operator (issue 158), a certified eigen iteration and the Schur
composition. The thermal chain is finished in a fixed order before moving open
flow and the Mistrangelo comparison. The test matrix grows by T36–T49. The
source size estimates under Consequences above are unchanged; the new loans are listed
in `plan.md` H.10.8.
