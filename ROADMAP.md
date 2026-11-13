# LMhdX roadmap

[plan.md](plan.md) is authoritative. Issues and PRs record daily evidence; the
order below is a planning order, not a validation or release promise.

1. **Consolidation and the 2.0 API.** Delivered in 1.12.0: the
   `Problem`/`Solver`/`Output`/`solve` interface with TOML schema 3, restart
   and ParaView output, the RK3 stepper, and the RK3 march beside Newton for
   steady open ducts. Next: one implementation per concept (ADR 0012, steps
   HC1–HC6, with the variable-coefficient operator moved to SOLVAX, H7.2r),
   then the older case objects, loaders and commands removed in 2.0.0
   ([plan.md](plan.md) Program H, ADR 0010).
2. **Conjugate heat transfer for a blanket channel.** The heated conducting
   duct of Mistrangelo et al. (2025) as the code-to-code gate (ADR 0009, step
   5.4); wall stacks with inserts and stagnant gaps, wall Joule heat, a film
   condition on the coolant side, nuclear heating maps and thermoelectric
   terms (ADR 0011, steps H7.6–H8.4).
3. **Shape optimization of a blanket cross-section.** The developed
   cross-section with mixed convection, the march along the duct with a 1-D
   coolant, shape derivatives on a static grid, and the coupling audit (ADR 0011,
   steps H8.5–H8.7 and H9.6). Three-dimensional runs correct and verify the
   reduced models; they are not differentiated.
4. **Independent validation, high Reynolds number and device channels.** ALEX
   B2 through the core-flow model and FreeMHD evidence; Hartmann-channel
   turbulence on the GPU; exterior ESSOS fields for a stellarator channel;
   blanket modules, tritium and wall interaction as research steps
   ([plan.md](plan.md) Phases 4 and 6, H5, H9).

No release until the plan's acceptance gates pass. No speed, turbulence, heat
transfer or thermoelectric claim without its numerical and physical evidence.
