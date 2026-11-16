# Migrating from 1.x to 2.0

Every 1.x entry point below warns (`DeprecationWarning`) and is retired in 2.0. The numbers do not change: each
replacement reaches the same kernel, and `tests/test_problem.py` pins the duct, Hunt and fringe presets against
the older constructors bit for bit.

| 1.x | 2.0 |
|---|---|
| `make_hartmann_case(ha, ny=n, nz=n)`, `make_shercliff_case(...)` | `duct(ha, n)` (one insulating square duct; field along y) |
| `make_hunt_case(ha, ny=n, nz=n, wall_conductance_ratio=c)` | `duct(ha, n, wall_conductance=c)`; resolved walls: `duct(ha, n, walls=dict(y=Wall(layers=...)))` |
| `lmhdx.solve(case)` and `result.state.u`, `.converged`, `.status` | `lmhdx.solve(problem)`: `result.velocity[0].data[0]`, `result.residual`, `result.flow_rate`, `result.pressure_gradient` |
| `solve_fully_developed_fields(case, forcing=f, magnetic_field_scale=s)` | `solve_steady_state(duct_problem(...), forcing=(f, 0, 0), field_scale=s)` (`lmhdx.steady`, jittable), or `problem.replace(forcing=..., B=Uniform(strength=...))` through `lmhdx.solve` |
| a case with a target flow rate | `duct(ha, n, flow_rate=Q)`: the drive is `result.pressure_gradient` |
| `fluid_cell_areas`, `volumetric_flow_rate`, `linear_flow_response`, `drive_for_flow_rate`, `fixed_flow_hydraulic_power` | `lmhdx.steady.channel_cross_section_weights`, `channel_flow_rate`, `channel_flow_response`, `channel_drive_for_flow_rate`, `channel_fixed_flow_hydraulic_power` |
| a transient `CaseSpec` | `lmhdx.solve(problem, lmhdx.Solver(dt=..., steps=...))` |
| `write_solution_outputs`, `write_restart_npz`, `load_restart_bundle` | `lmhdx.Output(dir)` and `solve(..., start=path)` |
| `hartmann_validation`, `validation_summary`, plots | the analytic profiles of `validation/`; ParaView files of `Output` |
| `lmhdx run/validate/benchmark`, schema 1 and 2 TOML files | `lmhdx FILE.toml` over a schema 3 file; `lmhdx upgrade OLD.toml` converts schema 1 and 2 |
| `lmx` command, x64 and `XLA_FLAGS` set on import | `lmhdx`; `lmhdx.configure(dtype="float64")` |
