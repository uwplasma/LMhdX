# 0012 — One implementation per concept

Status: proposed 2026-11-12, revised 2026-11-13. It applies ADR 0010 DH15, adds the
consolidation steps HC1–HC6 to `plan.md` Program H, and amends:

- ADR 0009 D36: resolved walls keep one discretization, the extended grid; the
  per-axis wall nodes of the thin-wall factorization retire at a gate (DC4);
- ADR 0008 D34 and ADR 0009 D38: their module, test and example counts are
  read as the values of `scripts/audit_architecture.py` under DH15's loan rule
  (DC6);
- `plan.md` H.10.8: the removal table and the loan ledger (DC6); step H7.2's
  remainder moves before the wall-stack steps (DC5).

Basis:

- `main` at `c1a9e9b` (2026-11-13): a read of all twenty-one modules of
  `src/lmhdx` (15,891 lines), of the tests' pins, and of the callers in
  `validation/`, `examples/`, `scripts/` and `docs/`.
- The audit of that commit, in which every line budget equals its target.
- The H3 decision of 2026-10-26 (#302): the march and Newton are both kept,
  chosen by `Solver(method="auto")`.

ADR 0011 builds on the result.

**DC1. The concept register.** `plan.md` Section 5 holds one table: each
concept, the one place it is implemented, and any second implementation still
present with the step that removes it. A PR that adds a second implementation
of a listed concept adds its removal step to that table and to H.10.8 in the
same PR. The register at `c1a9e9b`:

| Concept | The one place | Still duplicated in | Removed by |
|---|---|---|---|
| Problem and setup | `Problem`, `Solver`, `solve` over the kernel problem types | the case objects, schema 1 and 2 loaders and the argparse commands of `cases.py`, `io.py`, `fully_developed.py` | H4.7 |
| Wall type | `WallLayer` | `core3d.SolidWall` | HC5 |
| Resolved-wall discretization | the extended grid (`conjugate_grid`) | the per-axis wall nodes (`wall_layers`, `assemble_thick_wall_operator`, `_corner_fix`) | HC6, at its gate (DC4) |
| Thin wall (a different model) | `assemble_thin_wall_operator` | the oracle `ops.thin_wall_flux` | HC2 |
| Variable-coefficient operator | `solvax.flux_form_operator` | `poisson.VariableHelmholtz` | H7.2r, before H7.8 (DC5) |
| Separable eigen solver, compile cache, constrained CG | SOLVAX | `poisson.py`, `_programs.py` | H4.6 |
| Joule dissipation density | `core3d.joule_density` | `thermal._joule_heating`; the `energy_budget` integral | HC4 |
| Open pipe (evidence for validation row 6) | `validation/open_pipe.py` | `axial.py`, second half | HC3 |
| Time loop | `core3d._trajectory` over `_step` | `thermal.advance_thermal` calling `_advance_momentum` | H7.4 |

**DC2. Evidence code lives outside the package.** H.10.8 keeps per-benchmark
solver branches out of `src/lmhdx`. Two blocks are of that kind and move:

- **The open pipe solver**: `axial.py` from its open-pipe section to the end,
  about 509 lines. It has its own polar discretization and eigen
  preconditioner, is called by `tests/test_pipe.py` alone and is not exported.
  It goes to `validation/open_pipe.py` with `pipe.wall_pressure_gradient`.
- **The thin-wall flux oracle**: 51 lines of `ops.py` with no caller in the
  package. It goes to the tests that use it.

The developed pipe of `pipe.py` and the pipe preset stay.

Reason: together with the hygiene step these remove about 600 lines without
changing a number, which repays the oldest open loans of H.10.8.

**DC3. One Joule density.** `core3d.joule_density` is the cell density
`sum over axes of the half-face average of J² / sigma`, with `sigma` a number
or a cell array. The thermal step calls it. `energy_budget` takes its integral
from it at the next change of its numerics, since the summation order moves
its last digits. The solid-wall route forms its currents from the same
`ops.face_current` at its next change, for the same reason. A contact
resistance's dissipation is added to the density of the face's two cells by
the step that adds contact resistances (ADR 0011 DB5).

**DC4. One wall type, one resolved-wall discretization.** `WallLayer` gains
the fields of `SolidWall`, which is deleted (D36 foresaw this). The `Wall`
pytree carries every field of a layer: the numbers that are leaves as
children, the rest as structure.

The thin wall of Walker stays: it is a different model, and its direct solve
and its derivative without iteration are worth keeping.

A wall resolved in cells has one discretization, the extended grid. It alone
carries heat, a third dimension and the wall stacks of ADR 0011. The per-axis
wall nodes also resolve corners (`_corner_fix`), and the two routes already
agree on currents and powers to 1e-9 with walls on both axes
(`tests/test_core3d.py`). The per-axis wall nodes retire (HC6) when three
gates hold:

1. Every test that pins the per-axis nodes passes on the extended grid at its
   present tolerance, `validation/mistrangelo.py` included.
2. A wall layer's conductivity and thickness stay leaves of `Problem`,
   differentiated by `jax.grad` through `lmhdx.solve` on the extended grid.
   They must meet the tolerances of the tests that pin them on the per-axis
   nodes today. Thickness changes the extended grid, so this needs a
   derivative rule for that leaf, for example a custom JVP built on ADR 0011
   DB3's linearised solve. The public call must not change.
3. A warm solve of a named walled case is at most three times slower, with
   the iteration counts recorded. HC6's PR names the case, the timing
   protocol (median of N calls after the first) and the hardware.

Until then both exist, as D36 left them.

**DC5. The variable-coefficient operator is SOLVAX's; the interface physics is
LMhdX's.** `poisson.VariableHelmholtz` duplicates
`solvax.separable.flux_form_operator` and is removed by H7.2's remainder
(H7.2r).

- **What stays in LMhdX.** `_face_conductances` forms each face's series
  resistance from the materials, the contact resistances and the film
  coefficients.
- **What SOLVAX has.** SOLVAX 0.36.2 has the mixed ends of an axis
  (uwplasma/SOLVAX#200) and the singular rule (no shift and no Dirichlet end).
- **What SOLVAX lacks.** Its operator forms the face conductances from cell
  conductivities alone.
- **The order.** A SOLVAX release adds an optional per-axis face resistance,
  added in the series sum `d_P/k_P + r_f + d_N/k_N`. H7.2r then ports the
  thermal and charge solves to SOLVAX's operator and unifies the lift and flux
  code of the two conduction paths. This comes before H7.8, H7.7a and H8.6a,
  so contact resistances, films and the definite steady solve are built once,
  on SOLVAX's operator, rather than added to the class that H7.2r deletes.
- **What else stays.** The direct fast-diagonal solve stays as the exact case
  of the operator for one material.

**DC6. Loans.** The ledger of H.10.8 is restated with every open loan, its
size and its repayment, as the log rows that drew them give them:

| Loan | Size | Repayment |
|---|---|---|
| H4.1 | +1,065 | H4.7 |
| H2.2 and H3 | 169 outstanding at #302 | the mirror-duct bridge went with SOLVAX 0.36.2 (#312); H4.6 repays the rest. Newton's 64 lines stay by the owner's decision |
| H4.2 | +290 | H4.6, H4.7 |
| H4.3 | +462 | H4.7 |
| H4.4 | +125 at #303 | H4.7 |

HC1–HC3 (−605 lines) repay open loans oldest first. Each PR sets the audit
targets it moves to the measured values; a fall is a repayment and a rise is a
loan with its step. The module, test, example and validation counts quoted in
D34 and D38 are the audit's values of their day, not separate caps. H3.5 is
done (#302): the Newton stack became a branch of the march's function, and
Newton stays (DC7).

**DC7. Not consolidated, by decision.**

- The thin wall and the solid wall (DC4).
- The Newton route of the steady open duct, kept beside the march by the
  owner (2026-10-26) and chosen by `axial.method_for`.
- The inertialess core-flow model, a reduced model with its own unknowns.
- The developed polar pipe, the only developed solver on a polar grid.
- The limited face values of the scalar and of the momentum. They share the
  van Leer mean but differ in lever arm, clipping and boundary stencil, so
  merging them changes numbers. They are merged when transport is next
  changed, with its evidence.
- The mean-removal sites and the polar factorization's own helpers, for the
  same reason.

**DC8. House style for new code.**

- A result is a `NamedTuple`. A class that carries differentiable leaves is a
  dataclass registered with `fields.pytree`.
- `ValueError` for an input that is wrong; `NotImplementedError` naming the
  plan step for behaviour that is planned; `NumericalFailure` (a
  `RuntimeError`) for a solve that did not converge. `NumericalFailure` lives
  in `cases.py` today; H4.7 moves it to `solver.py`, which raises it, before
  deleting `cases.py`. Messages are lowercase clauses without a final period.
- Concrete inputs are validated where a kernel problem is built, never in a
  class that is rebuilt from tracers.
- A number that a step makes a leaf is not converted on the host.
- A new field of a problem or a solver changes the restart identity
  (`output.digest` hashes every field). A field whose default reproduces
  today's run is left out of the digest at its default, so records written
  before it still load.
- Docstrings give the equation or quantity, units, assumptions, the
  differentiable arguments and the failure behaviour, in prose with roles.
  Comments give a numerical reason, not the history of a change.

**Defects found by the read.** Each is fixed in the step that next touches its
code and listed in `plan.md` Section 2:

- `SteadySolution.steps` always holds `max_steps`;
- `duct_problem` raises for a Hartmann number between 0 and 1;
- the `grid` docstring calls Neumann data the outward-normal derivative, where
  `pad` takes the derivative along the axis;
- `face_currents` ignores its `factorization` argument on three of its five
  routes;
- a schema 3 run exits with status 0 when its solve did not converge;
- `upgrade` writes `method = "implicit"` for a transient schema 2 file, which
  `solve` refuses.

Consequences:

- The package falls from 15,891 to about 15,300 lines with HC1–HC4, and to
  about 15,150 with H7.2r.
- It rises to about 16,200 with ADR 0011's steps, and falls to about 12,000
  once H4.6 and H4.7 are done.
- HC6 is not counted until its gate holds.
- `validation/` gains a tenth file.
- No number computed by the package changes before HC6.
