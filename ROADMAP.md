# LMhdX roadmap

[plan.md](plan.md) is authoritative. Issues and PRs record daily evidence; dates
below are planning targets, not validation or release promises.

1. **Integration and hygiene — September 2026.** Complete Phase 0 precision and
   CI qualification (#74, #75). The B1/B2 reference lane is kept at tag `lmx-b2-lane-v1`.
2. **Trustworthy 3-D core and useful inverse design — October–November 2026.**
   Run the 8³ MAC/collocated oracle before choosing the momentum discretization.
   Verify conservation, convergence and implicit derivatives. Ship the fully
   developed fixed-flow design examples without waiting for new 3-D numerics.
3. **Independent validation and research applications — gate-driven, then release.**
   Qualify B2 against the ALEX experiment through the core-flow model and
   FreeMHD evidence. Add measured CPU/GPU scaling, Q2D and passive heat in their
   documented envelopes. Use exterior ESSOS fields and independent blanket
   geometry for device channels; do not extrapolate VMEX interior fields.
4. **High Reynolds number and the 2.0 API — from October 2026.** One IMEX
   Runge–Kutta stepper for steady and unsteady flow, energy-conserving
   transport on stretched grids, Hartmann-channel validation on the GPU, and a
   new `Problem`/`Solver`/`Output` interface with TOML and differentiable wall
   conductance, released as 2.0.0 ([plan.md](plan.md) Program H, ADR 0010).

No release until the plan's acceptance gates pass. No speed, turbulence or heat
transfer claim without its corresponding numerical and physical evidence.
