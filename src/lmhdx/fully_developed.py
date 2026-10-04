"""Fully developed duct cases solved on the staggered core.

A :class:`~lmhdx.specs.CaseSpec` for a rectangular duct becomes a
:class:`~lmhdx.core3d.ChannelProblem` with one periodic axial cell, solved by the
conjugate-gradient steady solve of :mod:`lmhdx.steady` and reported on the
case's cross-section. That is the route :func:`lmhdx.solve` and
:func:`lmhdx.solve_fully_developed_fields` take.

*Mesh.* ``ny`` and ``nz`` are the fluid cells along ``y`` and ``z``. The faces
follow :func:`lmhdx.core3d.duct_problem`: the walls normal to the field are
clustered to the Hartmann layer ``delta = sqrt(rho nu / sigma) / |B|``, the
others to the side layer ``sqrt(a delta)``, with ``a`` the half-width along the
field, six cells in each layer (``hartmann_layer_cells`` overrides) and the
gentlest stretching that spans the duct; an odd count adds one centre cell.
A 2 x 2 duct with unit properties gets exactly the faces of ``duct_problem(hartmann=Ha, cells=n)``.

*Walls.* An insulating wall is the homogeneous Neumann closure. The conducting
walls of a ``layered_duct`` follow ``geometry.wall_model``. ``"thin"`` makes
each a sheet of conductance ``sigma_w t_w / sigma`` (:mod:`lmhdx.poisson`) with no
cells of its own, which needs equal walls on one axis. ``"resolved"`` gives each
wall ``wall_cells`` uniform cells of its own conductivity, insulated outside:
one wall, two different walls, or a layer that no boundary names
(``ChannelProblem.wall_layers``); a corner cell takes the nearer wall's
material, as the retired cell-centred solver assigned it, and the reported fields
cover the fluid. The default ``"auto"`` is thin where that holds and resolved
otherwise. A wall stack of several materials is a ``ChannelProblem`` with
per-cell ratios.

*Field.* A constant field is three numbers; an analytic or tabulated one is
sampled at the cell centres as an :class:`~lmhdx.core3d.ImposedField`, and the
layers follow its peak transverse strength. An axial component is kept: it
adds no electromotive force to the axial flow, and on a varying field it can
drive a secondary flow, which the retired cell-centred solver dropped.

*Drive.* ``forcing`` is the axial force density. With zero forcing and an
``inlet_flow_rate`` boundary, the flow rate is met by scaling the unit-drive
solution, which is exact because the problem is linear in the drive.

The pseudo-time controls of the case (time step, relaxation, potential and
coupling iterations, steady tolerance) do not enter; the steady state is one
preconditioned CG solve to a relative residual of 1e-9. Without
:func:`lmhdx.enable_x64` it runs in float32 to 1e-5, which the solve reaches at
Ha 20 on 32 cells and not at Ha 100 on 48, where it raises; a float32 case
with float64 enabled is solved in float64 and returned in float32.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import solvax

from ._programs import host_array
from .bc import NEUMANN, PERIODIC, BoundaryCondition
from .core3d import ChannelProblem, ImposedField, face_currents, zero_velocity
from .em import lorentz_force, wall_insulated
from .grid import Grid, uniform_faces, wall_resolving_faces
from .mesh import StructuredMesh, generate_rect_duct_mesh_from_faces, sample_tabulated_cross_section_field
from .ops import divergence
from .specs import CaseSpec, Diagnostics, MHDState, Solution, SolverStepRecord, require_finite
from .steady import shared_or_embedded, solve_steady_state

__all__ = [
    "case_mesh",
    "channel_problem",
    "solve_fully_developed",
    "solve_fully_developed_fields",
    "solve_fully_developed_transient",
]

# Relative CG tolerance by the precision JAX computes in: float32 floors near 1e-6.
_TOLERANCE = {"float64": 1.0e-9, "float32": 1.0e-5}
_CELLS_IN_LAYER = 6
_MAX_CG = 12000
_SIDES = {"left": (1, 0), "right": (1, 1), "bottom": (2, 0), "top": (2, 1)}
_PAIRS = {"left_right": ("left", "right"), "top_bottom": ("bottom", "top")}
_IGNORED = {"no_slip", "inlet_velocity", "inlet_flow_rate", "outlet_pressure"}


def channel_problem(case: CaseSpec) -> ChannelProblem:
    """Return the staggered-core problem a fully developed case solves, at unit drive.

    Raise ``ValueError`` for what a case cannot mean: fluids of different
    properties (a ``CaseSpec`` gives regions no geometry), an imposed current
    density, unequal thin walls, a conducting wall without a layered duct. A
    constant field stays three numbers; an analytic or tabulated one is sampled
    at the cell centres as an :class:`ImposedField`.
    """
    _check(case)
    fluid = _fluid(case)
    density, viscosity, conductivity = (
        float(fluid.density or 1.0),
        float(fluid.viscosity or 1.0),
        fluid.conductivity,
    )
    geometry = case.geometry
    halves = (0.5 * geometry.width, 0.5 * geometry.height)
    # The layers follow the strongest transverse field, sampled on a uniform section.
    peak = [
        float(np.max(np.abs(component)))
        for component in _sampled_field(
            case, *(uniform_faces(n, -h, h) for n, h in zip((geometry.ny, geometry.nz), halves))
        )[1:]
    ]
    along = 0 if peak[0] >= peak[1] else 1
    # Ha on the half-width along the field; the layers are a/Ha and a/sqrt(Ha) there.
    hartmann = halves[along] * float(np.hypot(*peak)) * np.sqrt(conductivity / (density * viscosity))
    layers = [np.inf, np.inf]
    if hartmann > 0.0:
        layers = [halves[along] / hartmann] * 2
        layers[1 - along] = halves[along] / np.sqrt(hartmann)
    cells_in_layer = geometry.hartmann_layer_cells or _CELLS_IN_LAYER
    faces = [
        _faces(count, half, layer, cells_in_layer)
        for count, half, layer in zip((geometry.ny, geometry.nz), halves, layers, strict=True)
    ]
    insulating = BoundaryCondition(NEUMANN)
    grid = Grid(uniform_faces(1, 0.0, 1.0), *faces)
    field = _sampled_field(case, *faces)
    if case.magnetic_field.kind == "constant":
        field = tuple(float(component.flat[0]) for component in field)
    else:
        field = ImposedField(grid, tuple(component[None] for component in field))
    return ChannelProblem(
        grid=grid,
        conditions=(BoundaryCondition(PERIODIC), insulating, insulating),
        density=density,
        viscosity=viscosity,
        conductivity=conductivity,
        magnetic_field=field,
        forcing=(1.0, 0.0, 0.0),
        dt=min(halves) ** 2 / viscosity,
        **_walls(case, conductivity),
    )


def case_mesh(case: CaseSpec) -> StructuredMesh:
    """Return the fluid cross-section the core solves ``case`` on, as a :class:`StructuredMesh`."""
    grid = channel_problem(case).grid
    return generate_rect_duct_mesh_from_faces(
        y_faces=jnp.asarray(grid.y_faces, dtype=case.dtype),
        z_faces=jnp.asarray(grid.z_faces, dtype=case.dtype),
        length=case.geometry.length,
    )


def solve_fully_developed_fields(
    case: CaseSpec,
    *,
    forcing: float | jax.Array | None = None,
    magnetic_field_scale: float | jax.Array = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return the steady velocity, potential, currents and Lorentz force on :func:`case_mesh`.

    All five are cell-centred ``(ny, nz)`` arrays in the case's dtype: axial
    velocity, potential (zero volume mean), the ``y`` and ``z`` currents
    averaged from their faces, and the axial Lorentz force density.
    ``forcing`` and ``magnetic_field_scale`` are continuous design inputs,
    differentiable through the implicit solve of :func:`lmhdx.steady.solve_steady_state`;
    the case itself is static, so close over it outside :func:`jax.jit`. A
    solve that fails raises :class:`~lmhdx.specs.NumericalFailure` when called
    with concrete inputs and gives nonfinite fields under tracing.
    """
    problem = channel_problem(case)
    target = _target_flow_rate(case) if forcing is None else None
    drive = target if target is not None else (case.forcing if forcing is None else forcing)
    fields, _, _ = _driven(problem, target is not None, drive, magnetic_field_scale)
    fields = tuple(value.astype(case.dtype) for value in fields)
    if not any(isinstance(value, jax.core.Tracer) for value in (drive, magnetic_field_scale)):
        require_finite("fully developed solve", velocity=fields[0])
    return fields


def solve_fully_developed(case: CaseSpec, *, logger=None, start_time: float = 0.0) -> Solution:
    """Solve a fully developed case to its steady state on the core and report it as a :class:`Solution`.

    ``status`` is ``"converged"`` with ``residual`` the relative steady
    residual ``||R(u)|| / ||R(0)||``; a solve that fails raises. ``steps`` is
    zero: the solve is one CG call with no outer iterations. The diagnostics
    hold one record.
    """
    problem = channel_problem(case)
    mesh = case_mesh(case)
    target = _target_flow_rate(case)
    if logger is not None:
        mean = None if target is None else target / (case.geometry.width * case.geometry.height)
        logger.emit_header(
            case=case,
            mesh=mesh,
            mode="steady",
            potential_solver="staggered core / fast diagonalization / pcg",
            target_mean_velocity=mean,
            reference_mean_velocity=mean,
            restart=None,
        )
    fields, drive, evidence = _driven(
        problem, target is not None, case.forcing if target is None else target, 1.0
    )
    u, phi, jy, jz, lorentz = (value.astype(case.dtype) for value in fields)
    require_finite("fully developed solve", velocity=u, potential=phi, residual=evidence["residual"])
    areas = jnp.asarray(mesh.dy, dtype=u.dtype)[:, None] * jnp.asarray(mesh.dz, dtype=u.dtype)[None, :]
    record = _summary(problem, areas, (u, jy, jz, lorentz), drive)
    record.update(residual_history=evidence["residual"], div_current_max_history=evidence["div_current_max"])
    values = {name: float(value) for name, value in jax.device_get(record).items()}
    diagnostics = Diagnostics(
        time_history=jnp.asarray([start_time]),
        **{name: jnp.asarray([value]) for name, value in values.items()},
    )
    state = MHDState(u, phi, jy, jz, lorentz, float(start_time), values["residual_history"])
    solution = Solution(mesh, state, diagnostics, case.name, converged=True, status="converged", steps=0)
    if logger is not None:
        logger.emit_footer(solution)
    return solution


def solve_fully_developed_transient(
    case: CaseSpec,
    logger=None,
    *,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info=None,
) -> Solution:
    """Run a transient fully developed case on the core, by implicit Euler steps.

    Each step is ``(u - u_n)/dt = A u + b`` for the core's Stokes-limit operator
    ``A`` and drive ``b``, solved by the steady solve's CG with a mass term and its
    preconditioner factorized at the step (see :func:`_transient_programs`). The
    steps run at ``time_stepper.dt`` from ``initial_state`` (or rest plus
    ``initial_velocity``) to ``t_final``, at most ``max_steps``, compiled as one
    scan between kept records. A ramped field scales the Lorentz force step by
    step. With an ``inlet_flow_rate`` and zero forcing each step meets the flow
    rate exactly. The other pseudo-time controls of the retired cell-centred loop
    (relaxation, update limit, coupling and potential iterations) do not enter.
    ``output.history_stride`` keeps every ``stride``-th step and the last (``0``,
    the last alone); ``residual_history`` is the step's largest velocity change,
    ``linear_iterations_history`` its CG iterations, and ``status`` is
    ``"completed"``. A step whose CG fails gives nonfinite fields, which raise.
    """
    stride = case.output.history_stride
    if stride < 0:
        raise ValueError("history_stride must be non-negative")
    controls = case.time_stepper
    if controls.dt <= 0.0:
        raise ValueError("Time-step size dt must be positive")
    if case.solver.time_scheme != "implicit_euler":
        raise NotImplementedError("the transient core run is first order, as implicit_euler")
    problem = dataclasses.replace(channel_problem(case), dt=float(controls.dt), forcing=(0.0, 0.0, 0.0))
    mesh = case_mesh(case)
    start = 0.0 if initial_state is None else float(initial_state.time)
    remaining = max(0.0, float(controls.t_final) - start) / float(controls.dt)
    steps = min(
        int(controls.max_steps), int(np.floor(remaining + 16.0 * np.finfo(float).eps * max(1.0, remaining)))
    )
    steps = max(steps, 0)
    target = _target_flow_rate(case)
    velocity = zero_velocity(problem)
    profile = case.initial_velocity if initial_state is None else jnp.asarray(initial_state.u)
    axial = velocity[0].data + jnp.asarray(profile, dtype=velocity[0].data.dtype)
    velocity = (velocity[0].replace_data(axial), *velocity[1:])
    if logger is not None:
        mean = None if target is None else target / (case.geometry.width * case.geometry.height)
        logger.emit_header(
            case=case,
            mesh=mesh,
            mode="transient",
            potential_solver="staggered core / fast diagonalization",
            target_mean_velocity=mean,
            reference_mean_velocity=mean,
            restart=restart_info,
        )
    times = start + controls.dt * np.arange(1, steps + 1)
    scales = np.asarray([_ramp(case.magnetic_field, time) for time in times], dtype=float)
    kept = [index for index in range(steps) if stride and (index % stride == 0 or index == steps - 1)]
    kept = kept or ([steps - 1] if steps else [])
    run, report = _transient_programs(problem, target is not None)
    drive = jnp.asarray(case.forcing if target is None else target, dtype=jnp.result_type(float))
    records, done = [], 0
    for index in kept:
        velocity, change, forcing, iterations = run(velocity, jnp.asarray(scales[done : index + 1]), drive)
        done = index + 1
        record = report(velocity, jnp.asarray(scales[index]), forcing, change, iterations)
        records.append(record)
        if logger is not None:
            values = {name: float(value) for name, value in jax.device_get(record[1]).items()}
            logger.emit_step(_step_record(done, float(times[index]), values))
    if records:
        fields = tuple(value.astype(case.dtype) for value in records[-1][0])
        histories = {name: np.asarray([float(r[1][name]) for r in records]) for name in records[0][1]}
    else:
        fields, histories = _initial_fields(case, initial_state), {}
    u, phi, jy, jz, lorentz = fields
    histories["time_history"] = times[kept] if steps else np.zeros(0)
    residual = (
        float(histories["residual_history"][-1])
        if records
        else float(getattr(initial_state, "residual", 0.0))
    )
    require_finite("transient core run", velocity=u, potential=phi, residual=residual)
    diagnostics = _transient_diagnostics(
        histories, initial_diagnostics if stride else None, append_diagnostics
    )
    state = MHDState(u, phi, jy, jz, lorentz, float(start + steps * controls.dt), residual)
    solution = Solution(mesh, state, diagnostics, case.name, converged=None, status="completed", steps=steps)
    if logger is not None:
        logger.emit_footer(solution)
    return solution


def _initial_fields(case: CaseSpec, state: MHDState | None) -> tuple[jax.Array, ...]:
    """The fields a run with no steps reports: the restart's, or rest plus the initial velocity."""
    if state is not None:
        return tuple(
            jnp.asarray(value, case.dtype)
            for value in (state.u, state.phi, state.jy, state.jz, state.lorentz_x)
        )
    zeros = jnp.zeros((case.geometry.ny, case.geometry.nz), case.dtype)
    return (zeros + case.initial_velocity, zeros, zeros, zeros, zeros)


def _ramp(spec, time: float) -> float:
    """The clipped affine startup ramp of the field, ``(t - t_start) / (duration + 1e-6)``."""
    if spec.ramp_duration <= 0.0:
        return 1.0
    return float(np.clip((time - spec.ramp_start) / (spec.ramp_duration + 1.0e-6), 0.0, 1.0))


@functools.lru_cache(maxsize=16)
def _transient_programs(problem: ChannelProblem, fixed_flow: bool):
    """Compile the steps between two kept records, and the record of a state, for one problem.

    A step is implicit Euler, ``(u - u_n)/dt = A u + b``: with ``-A`` symmetric
    positive definite on the divergence-free fields, ``(I/dt - A) u = u_n/dt + b``
    is the steady CG system of :mod:`lmhdx.steady` plus a mass term, and the
    steady preconditioner factorized at the time step (one projection step) is
    that system's own approximate inverse. A fixed flow rate adds the solve of a
    unit drive from rest and the multiple of it that meets the rate.
    """
    from .steady import _face_weights, _preconditioner, _projection_solves_at, _rest_residual, steady_residual

    dt = float(problem.dt)
    factorization = problem.factorization()
    potential_factorization = problem.potential_factorization()
    weights = _face_weights(problem)
    precond = _preconditioner(problem, factorization, _projection_solves_at(problem, dt), dt)
    unit = _rest_residual(problem, factorization, (1.0, 0.0, 0.0))
    tolerance = _TOLERANCE[jnp.result_type(float).name]
    dy, dz = (np.asarray(widths) for widths in problem.grid.widths[1:])
    areas = jnp.asarray(dy[:, None] * dz[None, :])

    def implicit(previous, scale, drive):
        def matvec(y):
            velocity = jax.tree.map(jnp.divide, y, weights)
            applied = steady_residual(
                velocity, problem, factorization, forcing=(0.0, 0.0, 0.0), field_scale=scale
            )
            return jax.tree.map(lambda u, a: u / dt - a, velocity, applied)

        rhs = jax.tree.map(lambda u, b: u / dt + drive * b, previous, unit)
        result = solvax.pcg(
            matvec,
            rhs,
            x0=jax.tree.map(jnp.multiply, previous, weights),
            precond=lambda r: jax.tree.map(jnp.multiply, precond(r), weights),
            rtol=tolerance,
            max_steps=_MAX_CG,
        )
        failed = ~(result.converged & jnp.isfinite(result.residual_norm))
        velocity = jax.tree.map(lambda y, w: jnp.where(failed, jnp.nan, y / w), result.x, weights)
        return velocity, result.iterations

    @jax.jit
    def run(velocity, scales, drive):
        def single(state, scale):
            if fixed_flow:
                free, iterations = implicit(state, scale, 0.0)
                response, more = implicit(jax.tree.map(jnp.zeros_like, state), scale, 1.0)
                forcing = (drive - jnp.sum(areas * free[0].data[0])) / jnp.sum(areas * response[0].data[0])
                updated = jax.tree.map(lambda a, b: a + forcing * b, free, response)
                iterations = iterations + more
            else:
                updated, iterations = implicit(state, scale, drive)
                forcing = drive
            change = jnp.max(jnp.abs(updated[0].data - state[0].data))
            return updated, (change, forcing, iterations)

        final, (changes, forcings, iterations) = jax.lax.scan(single, velocity, scales)
        return final, changes[-1], forcings[-1], iterations[-1]

    @jax.jit
    def report(velocity, scale, forcing, change, iterations):
        potential, currents, field = face_currents(velocity, problem, potential_factorization, scale)
        fields, _ = _fields(problem, velocity, potential, currents, field)
        values = _summary(problem, areas, (fields[0], *fields[2:]), forcing)
        values.update(
            residual_history=change,
            courant_like=jnp.max(jnp.abs(fields[0])) * problem.dt / min(float(dy.min()), float(dz.min())),
            div_current_max_history=jnp.max(jnp.abs(divergence(currents).data)),
            linear_iterations_history=iterations,
        )
        return fields, values

    return run, report


def _transient_diagnostics(histories: dict, initial: Diagnostics | None, append: bool) -> Diagnostics:
    def history(name):
        values = jnp.asarray(histories.get(name, np.zeros(0)), dtype=float)
        if initial is None or not append:
            return values
        return jnp.concatenate([jnp.asarray(getattr(initial, name), dtype=float), values])

    names = [entry.name for entry in dataclasses.fields(Diagnostics)]
    return Diagnostics(**{name: history(name) for name in names})


def _summary(problem: ChannelProblem, areas, fields, forcing) -> dict:
    """The section integrals and peaks a record reports, from ``(u, jy, jz, lorentz)``."""
    u, jy, jz, lorentz = fields
    flow_rate, area, magnitude = jnp.sum(areas * u), jnp.sum(areas), jnp.hypot(jy, jz)
    return {
        "courant_like": jnp.zeros(()),
        "ohmic_power": jnp.sum(areas * (jy**2 + jz**2)) / max(problem.conductivity, 1e-300),
        "u_max_history": jnp.max(jnp.abs(u)),
        "mean_velocity_history": flow_rate / area,
        "applied_forcing_history": forcing,
        "current_max_history": jnp.max(magnitude),
        "lorentz_max_history": jnp.max(jnp.abs(lorentz)),
        "volumetric_flow_rate_history": flow_rate,
        "mean_current_magnitude_history": jnp.sum(areas * magnitude) / area,
        "lorentz_power_history": jnp.sum(areas * u * lorentz),
    }


# SolverStepRecord fields read from a transient record; the rest were the retired cell-centred loop's and are zero.
_STEP_FIELDS = {
    "u_max": "u_max_history",
    "mean_velocity": "mean_velocity_history",
    "current_max": "current_max_history",
    "lorentz_max": "lorentz_max_history",
    "residual": "residual_history",
    "linear_iterations": "linear_iterations_history",
    "applied_forcing": "applied_forcing_history",
    "courant_like": "courant_like",
    "ohmic_power": "ohmic_power",
    "volumetric_flow_rate": "volumetric_flow_rate_history",
    "div_current_max": "div_current_max_history",
}


def _step_record(index: int, time: float, values: dict) -> SolverStepRecord:
    names = [entry.name for entry in dataclasses.fields(SolverStepRecord)][2:]
    return SolverStepRecord(
        index, time, **{name: values.get(_STEP_FIELDS.get(name, ""), 0.0) for name in names}
    )


def _driven(problem: ChannelProblem, fixed_flow: bool, drive, field_scale):
    """Scale the unit-drive program's fields by the drive, outside the program.

    ``drive`` is the force density, or the flow rate when ``fixed_flow`` is set.
    The problem is linear in the drive, so the scaling is exact; done outside the
    compiled solve, a derivative in the drive differentiates a product and never
    enters the solve's program, which is then neither linearized nor transposed.
    The relative residual does not depend on the drive, and the current
    divergence scales with it. A field scale that is not traced makes the solve
    a constant of any enclosing trace: it runs then, through the one compiled
    program, so a jitted objective, gradient or tangent in the drive compiles
    only the scaling around it.
    """
    dtype = jnp.result_type(float)
    if isinstance(field_scale, jax.core.Tracer):
        fields, flow_rate, evidence = _compiled(problem)(jnp.asarray(field_scale, dtype=dtype))
    else:
        fields, flow_rate, evidence = _executable(problem, dtype.name)(np.asarray(field_scale, dtype=dtype))
    forcing = jnp.asarray(drive, dtype=dtype)
    if fixed_flow:
        forcing = forcing / flow_rate
    evidence = {**evidence, "div_current_max": jnp.abs(forcing) * evidence["div_current_max"]}
    return tuple(forcing * value for value in fields), forcing, evidence


@functools.lru_cache(maxsize=16)
def _executable(problem: ChannelProblem, dtype: str):
    """The compiled unit-drive program itself, which runs on concrete inputs even inside a trace."""
    return shared_or_embedded(problem, _unit_drive, jax.ShapeDtypeStruct((), dtype))


def _unit_drive(problem: ChannelProblem):
    return _compiled(problem).__wrapped__


@functools.lru_cache(maxsize=16)
def _compiled(problem: ChannelProblem):
    """Compile one steady solve per problem at unit drive, with the field scale as its argument.

    One program replaces the dispatch of every operation from the host, which
    is most of an eager solve's time, cold or warm; :func:`_driven` scales it.
    The fields and the report share the program, so :func:`lmhdx.solve` and
    :func:`solve_fully_developed_fields` agree bit for bit and compile once
    between them, for every case dtype: the fields come back in the precision of
    the solve and the callers cast them.
    """

    def run(field_scale):
        solution = solve_steady_state(
            problem,
            forcing=(1.0, 0.0, 0.0),
            field_scale=field_scale,
            tolerance=_TOLERANCE[jnp.result_type(float).name],
        )
        velocity = solution.velocity
        dy, dz = (host_array(problem.grid, _widths, axis) for axis in (1, 2))
        weights = dy[:, None] * dz[None, :]
        fields, currents = _fields(
            problem, solution.velocity, solution.potential, solution.currents, solution.magnetic_field
        )
        # The solve's own residuals, at rest and at the root: evaluating them again doubled the program.
        scale = solution.initial_residual_norm
        evidence = {
            "residual": solution.residual_norm / jnp.maximum(scale, 1e-300),
            "div_current_max": jnp.max(jnp.abs(divergence(currents).data)),
        }
        return fields, jnp.sum(weights * velocity[0].data[0]), evidence

    return jax.jit(run)


def _widths(grid: Grid, axis: int) -> np.ndarray:
    return grid.widths[axis]


def _fields(problem: ChannelProblem, velocity, potential, currents, field):
    scalar = problem.scalar_conditions
    closed = tuple(wall_insulated(current, axis, scalar[axis]) for axis, current in enumerate(currents))
    force = lorentz_force(closed, field, scalar)
    current_y, current_z = currents[1].data[0], currents[2].data[0]
    jy = 0.5 * (current_y[:-1] + current_y[1:])
    jz = 0.5 * (current_z[:, :-1] + current_z[:, 1:])
    fields = (velocity[0].data[0], potential.data[0], jy, jz, force[0].data[0])
    return fields, currents


def _sampled_field(case: CaseSpec, y_faces: np.ndarray, z_faces: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return ``(B_x, B_y, B_z)`` of the case at the cell centres of a section, each ``(ny, nz)``."""
    y, z = (0.5 * (faces[1:] + faces[:-1]) for faces in (y_faces, z_faces))
    spec = case.magnetic_field
    if spec.kind == "constant":
        return tuple(np.full((y.size, z.size), float(value)) for value in spec.value)
    if spec.kind == "analytic":
        if spec.fn is None:
            raise ValueError("an analytic magnetic field needs fn")
        sampled = spec.fn(*jnp.meshgrid(jnp.asarray(y), jnp.asarray(z), indexing="ij"))
    elif spec.kind == "tabulated":
        if spec.table_path is None:
            raise ValueError("a tabulated magnetic field needs table_path")
        sampled = sample_tabulated_cross_section_field(
            spec.table_path, y=y[:, None] + 0 * z, z=0 * y[:, None] + z
        )
    else:
        raise ValueError(f"unsupported magnetic field kind {spec.kind!r}")
    sampled = np.asarray(sampled, dtype=np.float64)
    return tuple(np.broadcast_to(sampled[..., axis], (y.size, z.size)) for axis in range(3))


def _faces(count: int, half: float, layer: float, cells_in_layer: int) -> np.ndarray:
    """Wall-resolving faces as in ``duct_problem``; fewer layer cells when the mesh is too coarse."""
    if layer < half:
        for cells in range(cells_in_layer, 0, -1):
            try:
                return wall_resolving_faces(
                    count, -half, half, layer_thickness=layer, cells_in_layer=cells, max_ratio=None
                )
            except ValueError:
                pass
    return uniform_faces(count, -half, half)


def _fluid(case: CaseSpec):
    fluids = [region for region in case.regions if region.kind == "fluid"]
    properties = {(region.conductivity, region.density, region.viscosity) for region in fluids}
    if len(properties) != 1:
        # The retired cell-centred solver silently took the first; a region has no geometry to place another.
        raise ValueError("a case fills its duct with one fluid; fluid regions must not differ")
    return fluids[0]


def _check(case: CaseSpec) -> None:
    field = case.magnetic_field
    if case.solver.kind != "fully_developed_inductionless":
        raise ValueError("case must select the fully developed inductionless solver")
    if case.geometry.kind not in {"rect_duct", "layered_duct"}:
        raise ValueError(f"the staggered core does not solve geometry {case.geometry.kind!r}")
    if field.kind == "constant" and field.value is None:
        raise ValueError("a constant magnetic field needs a value")
    if field.ramp_duration > 0.0 and case.solver.mode != "transient":
        raise ValueError("steady fields require an unramped magnetic field")
    for boundary in case.boundary_conditions:
        if boundary.kind not in _IGNORED | {"insulating", "conducting_wall"}:
            raise ValueError(f"the staggered core does not impose {boundary.kind!r} boundaries")


def _walls(case: CaseSpec, conductivity: float) -> dict:
    """Return the ``wall_conductance`` or the ``wall_layers`` of the case's conducting walls.

    A side is conducting when a ``conducting_wall`` boundary names it, or when it
    has wall cells that no boundary names and the first solid region conducts
    (the retired cell-centred solver's fallback); it is insulating when an
    ``insulating`` boundary names it or it has no wall. ``geometry.wall_model``
    chooses the closure: ``"thin"`` makes each wall a sheet of conductance
    ``sigma_w t_w / sigma``, which needs equal walls on an axis, on either or both axes; ``"resolved"``
    gives the walls ``wall_cells`` cells of their own; ``"auto"``
    is thin where that holds and resolved otherwise.
    """
    regions = {region.name: region for region in case.regions}
    geometry, sides = case.geometry, {}
    for boundary in case.boundary_conditions:
        if boundary.kind not in {"insulating", "conducting_wall"}:
            continue
        names = _side_names(boundary)
        if boundary.kind == "insulating":
            for name in names:
                sides.setdefault(name, None)
            continue
        region = regions.get(boundary.region)
        if geometry.kind != "layered_duct" or region is None or not names:
            raise ValueError("a conducting wall needs a layered duct, a solid region and its sides")
        for name in names:
            sides[name] = region
    solids = [region for region in case.regions if region.kind == "solid"]
    walled = geometry.kind == "layered_duct"
    for index, name in enumerate(_SIDES):
        if (
            name not in sides
            and walled
            and geometry.wall_cells[index]
            and solids
            and solids[0].conductivity > 0.0
        ):
            sides[name] = solids[0]
    ends = {}
    for index, name in enumerate(_SIDES):
        region = sides.get(name)
        if region is None or not region.conductivity or not conductivity:
            continue
        thickness, cells = geometry.wall_thickness[index], geometry.wall_cells[index]
        if thickness <= 0.0:
            raise ValueError(f"conducting wall {name!r} has no thickness")
        ends[name] = (region.conductivity / conductivity, thickness, cells)
    axes = {_SIDES[name][0] for name in ends}
    pairs = {axis: [ends.get(name) for name, (index, _) in _SIDES.items() if index == axis] for axis in axes}
    thin = all(
        None not in pair and pair[0][0] * pair[0][1] == pair[1][0] * pair[1][1] for pair in pairs.values()
    )
    model = geometry.wall_model
    if model not in {"auto", "thin", "resolved"}:
        raise ValueError(f"wall_model must be 'auto', 'thin' or 'resolved', got {model!r}")
    if model == "thin" or (model == "auto" and thin):
        if not thin:
            raise ValueError("thin walls need equal conducting walls on both sides of an axis")
        conductance = [0.0, 0.0, 0.0]
        for axis, pair in pairs.items():
            conductance[axis] = pair[0][0] * pair[0][1]
        return {"wall_conductance": tuple(conductance)}
    layers = [None, None, None]
    for axis, pair in pairs.items():
        if any(end is not None and end[2] < 1 for end in pair):
            raise ValueError("a wall resolved in cells needs wall_cells")
        layers[axis] = tuple(None if end is None else (end[0], (end[1] / end[2],) * end[2]) for end in pair)
    return {"wall_layers": tuple(layers)}


def _side_names(boundary) -> tuple[str, ...]:
    """Return the walls a boundary names: ``left_right``, ``top_bottom``, a list, or an axis end."""
    side = (boundary.side or "").lower()
    if side in _PAIRS:
        return _PAIRS[side]
    if side in {"min", "max"} and (boundary.axis or "").lower() in {"y", "z"}:
        return ({"y": ("left", "right"), "z": ("bottom", "top")}[boundary.axis.lower()][side == "max"],)
    names = tuple(part.strip() for part in side.split(",") if part.strip())
    return names if all(name in _SIDES for name in names) else ()


def _target_flow_rate(case: CaseSpec) -> float | None:
    """Return the prescribed flow rate of a case driven by zero forcing."""
    if case.forcing != 0.0:
        return None
    for boundary in case.boundary_conditions:
        if boundary.kind == "inlet_flow_rate" and isinstance(boundary.value, (int, float)):
            return float(boundary.value)
    return None
