"""Ducts with an inlet and an outlet: the non-periodic axial direction (plan 1.9b, D26).

The axial axis is the first. Its conditions follow the published practice of
HIMAG, FreeMHD, GridapMHD and the 2025 six-code benchmark, chosen so that every
solve stays direct and every derivative implicit:

* **Inlet.** The velocity is LMhdX's own fully developed profile at the inlet
  field, solved on the same cross-section and scaled to the imposed flow rate.
  It is array-valued Dirichlet data on the inlet face
  (:class:`lmhdx.grid.BoundaryCondition`). The flow rate is exact and the pressure
  drop is an output; there is no extra unknown.
* **Outlet.** Zero axial gradient of every velocity component and ``p = 0``.
  The pressure operator is then non-singular, and the axial axis stays
  diagonalizable (Neumann at the inlet, Dirichlet at the outlet).
* **No normal current at either end**, ``dphi/dn = (u x B).n``, with the
  potential's gauge fixed by removing its mean.

The inlet enters as a lift. The fully developed profile carried unchanged along
the whole duct is discretely divergence free, so the solution is that lift plus
a correction with a zero inlet face. The correction lives in a linear space on
which the Stokes-limit operator is symmetric in the face-volume inner product,
the two end faces owning half a cell each, so the solve is the same
preconditioned conjugate-gradient solve as a periodic duct's
(:func:`lmhdx.steady.solve_steady_state`) and differentiates the same way. Far
upstream of a field change the lift is the discrete solution, which is why the
uniform region carries its fully developed gradient.

With advection on (plan 1.9d, H3) the steady problem is nonlinear and has two solvers,
``method="march"`` and ``method="newton"``, chosen by ``"auto"`` (:func:`method_for`). The march goes to the fixed
point: the IMEX Runge-Kutta stepper of :mod:`lmhdx.core3d` (central transport, Crank-Nicolson
damping and viscosity, the exact constrained stage) advances the departure from the lift at a
Courant-limited step, in chunks, until the physical steady residual falls to ``tolerance`` of the
lift's. The march is a fixed point of the step, so its root is the root of the steady residual, whatever the
step. The result carries one of three statuses: ``converged``; ``stagnated``, a residual that stopped
falling above the tolerance, which Kelley and Keyes (1998) report as the failure of pseudo-transient
continuation; and ``unsteady``, a residual that grows or whose change keeps turning, the signature of a
flow with no steady state (a limit cycle or bursts). Newton's method takes the same residual from the
same Stokes-limit start, each update a preconditioned :func:`solvax.gcrot` solve, and is the cheaper where damping
sets the step. Either root is differentiated by the implicit function theorem (:func:`solvax.root_solve`) on the
fixed point of one step, its tangent and transposed solves GMRES on that step's linearization.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import solvax
from jax.flatten_util import ravel_pytree

from .core3d import (
    RK3,
    ChannelProblem,
    ImposedField,
    _step,
    duct_problem,
    face_currents,
    fringe_field,
    project,
    zero_velocity,
)
from .grid import CENTER, DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition, Field, Grid, uniform_faces
from .ops import advective_step_limit, face_electromotive_force, neighbours
from .steady import (
    _face_weights,
    _norm,
    _orthogonal_projection,
    _preconditioner,
    _projection_solves,
    _stokes_limit_root,
    _tangent_solve,
    momentum_terms,
    solve_compiled,
    solve_steady_state,
    steady_residual,
    with_inflow,
)

__all__ = [
    "OpenDuctSolution",
    "axial_faces",
    "charge_balance",
    "fringe_duct",
    "fully_developed_inlet",
    "mass_balance",
    "method_for",
    "open_duct",
    "pressure_drop",
    "solve_open_duct",
    "station_flow_rates",
    "station_pressure",
]

_INSULATING = BoundaryCondition(NEUMANN)
# How a march ended: at a steady residual under the tolerance, stalled above it, or with no steady state.
CONVERGED, STAGNATED, UNSTEADY = 0, 1, 2
# A varying field gives every cell the largest damping rate in the implicit part and leaves the rest of its
# own, up to that rate, to the explicit part: the march is stable to ``dt rate`` of 6 and not at 7 (Ha 100, Re 20).
_DAMPING_COURANT = 4.0
_MARCH_RATIO = 4.0  # ``auto`` marches up to this ratio of the Courant step to the damping step


class OpenDuctSolution(NamedTuple):
    """Fields of an inflow-outflow duct, a pytree; ``pressure`` is the physical one, zero at the outlet."""

    velocity: tuple[Field, Field, Field]
    pressure: Field
    potential: Field
    currents: tuple[Field, Field, Field]
    magnetic_field: tuple[Field, Field, Field]
    residual_norm: jnp.ndarray
    initial_residual_norm: jnp.ndarray
    iterations: jnp.ndarray
    status: jnp.ndarray


def axial_faces(
    lower: float, upper: float, core: tuple[float, float], spacing: float, growth: float = 1.1
) -> np.ndarray:
    """Faces of spacing ``spacing`` over ``core``, growing by ``growth`` per cell into the buffers.

    Buffer cells stop growing at ``8 * spacing``; the last cell on each side
    absorbs the remainder, so the ends land exactly on ``lower`` and ``upper``.
    """
    if not lower <= core[0] < core[1] <= upper or spacing <= 0.0 or growth < 1.0:
        raise ValueError("axial faces need lower <= core < upper, a positive spacing and growth >= 1")
    middle = uniform_faces(max(1, round((core[1] - core[0]) / spacing)), *core)
    step = middle[1] - middle[0]

    def buffer(length: float) -> np.ndarray:
        widths, total = [], 0.0
        while total < length - 1e-12:
            width = min(step * growth ** (len(widths) + 1), 8.0 * step, length - total)
            if length - total - width < 0.5 * width:
                width = length - total
            widths.append(width)
            total += width
        return np.cumsum(widths)

    below, above = buffer(core[0] - lower), buffer(upper - core[1])
    return np.concatenate((core[0] - below[::-1], middle, core[1] + above))


def fully_developed_inlet(
    problem: ChannelProblem, flow_rate: float, *, tolerance: float = 1.0e-9, **controls
) -> tuple[np.ndarray, float]:
    """Solve the fully developed flow of ``problem``'s first cross-section at its first cell's field.

    The inlet field must be uniform over the cross-section, as it is upstream of a
    magnet. Returns the axial velocity on that cross-section scaled to ``flow_rate``, and
    the axial pressure gradient that drives it (negative for a positive flow).
    The cross-section, conductances and field are the duct's own, so upstream of
    any field change the three-dimensional solution reproduces it to round-off.
    """
    grid = problem.grid
    section = Grid(uniform_faces(1, 0.0, 1.0), grid.y_faces, grid.z_faces)
    field = problem.magnetic_field
    if isinstance(field, ImposedField):
        slabs = [component[0] for component in field.components]
        if any(np.ptp(slab) > 1e-12 * max(1.0, float(np.max(np.abs(slab)))) for slab in slabs):
            raise ValueError("the inlet field must be uniform over the inlet cross-section")
        field = tuple(float(slab.flat[0]) for slab in slabs)
    periodic = dataclasses.replace(  # the duct's own walls, thin, resolved or solid, with every other field
        problem,
        grid=section,
        conditions=(BoundaryCondition(PERIODIC), *problem.conditions[1:]),
        magnetic_field=field,
        forcing=(1.0, 0.0, 0.0),
        advection="off",
    )
    # A compiled program (shared by problems of one shape), not hundreds of eagerly dispatched operations.
    if tolerance == 1.0e-9 and not controls:
        solution = solve_compiled(periodic)
    else:
        solution = solve_steady_state(periodic, tolerance=tolerance, **controls)
    axial = np.asarray(solution.velocity[0].data[0])
    unit = float(np.sum(axial * section.face_areas(0)[0]))
    return axial * (flow_rate / unit), -flow_rate / unit


def open_duct(problem: ChannelProblem, flow_rate: float, **controls) -> ChannelProblem:
    """Turn the first axis of ``problem`` into an inlet and an outlet at the imposed ``flow_rate``.

    ``problem`` supplies the mesh, walls, field and properties; its first
    condition and forcing are replaced. ``controls`` go to
    :func:`fully_developed_inlet`.
    """
    profile, _ = fully_developed_inlet(problem, flow_rate, **controls)
    return dataclasses.replace(  # every other field, the walls resolved in cells among them, carries over
        problem,
        conditions=(BoundaryCondition(DIRICHLET, lower=profile, upper_kind=NEUMANN), *problem.conditions[1:]),
        forcing=(0.0, 0.0, 0.0),
        wall_conductance=(0.0, *problem.wall_conductance[1:]),
    )


def fringe_duct(
    *,
    hartmann: float,
    wall_conductance: float = 0.0,
    half_length: float = 3.0,
    upstream: float = 15.0,
    downstream: float = 10.0,
    spacing: float = 0.25,
    cells: int = 24,
    cells_in_layer: int = 6,
    flow_rate: float = 4.0,
    solenoidal: bool = False,
    advection: str = "off",
    growth: float = 1.1,
    **controls,
) -> ChannelProblem:
    """The ANL fringe (TM-228) in a square duct with an inlet and an outlet.

    The field falls as ``B_y = Ha (1 - sin(pi x / 2 x0)) / 2`` over
    ``|x| <= half_length`` and is uniform outside; ``upstream`` and
    ``downstream`` half-widths of buffer (D26: 15 and 10) separate the ramp from
    the ends. ``solenoidal=False`` is TM-228's field, ``B_y`` alone, which is
    divergence free. The conductance applies to all four walls, and the default
    flow rate is a unit mean velocity; half-width, density, viscosity and
    conductivity being one, the Reynolds number is the mean velocity,
    ``flow_rate / 4``, and ``N = Ha^2 / Re``. ``growth`` is the buffer cells'
    growth per cell (:func:`axial_faces`); with transport keep it at 1 so the
    buffers stay at ``spacing`` (see :func:`solve_open_duct`).
    """
    base = duct_problem(
        hartmann=hartmann, cells=cells, wall_conductance=wall_conductance, cells_in_layer=cells_in_layer
    )
    # Uniform from 2 x0 upstream of the centre, which holds TM-228's window [-6, 2], through the ramp.
    faces = axial_faces(
        -half_length - upstream,
        half_length + downstream,
        (-half_length - min(half_length, upstream), half_length),
        spacing,
        growth,
    )
    grid = Grid(faces, base.grid.y_faces, base.grid.z_faces)
    field = fringe_field(grid, half_length=half_length, strength=hartmann, solenoidal=solenoidal)
    walled = ChannelProblem(
        grid=grid,
        conditions=(BoundaryCondition(PERIODIC), _INSULATING, _INSULATING),
        conductivity=base.conductivity,
        magnetic_field=field,
        dt=base.dt,
        wall_conductance=(0.0, float(wall_conductance), float(wall_conductance)),
        advection=advection,
    )
    return open_duct(walled, flow_rate, **controls)


def _solve(
    problem: ChannelProblem,
    *,
    field_scale: float | jnp.ndarray = 1.0,
    tolerance: float = 1.0e-9,
    max_iterations: int = 36_000,
    method: str = "auto",
    max_steps: int = 40_000,
    check: int = 50,
    cfl: float = 0.9,
) -> OpenDuctSolution:
    """Solve an inflow-outflow duct; differentiable in ``field_scale``.

    In the Stokes limit, one preconditioned conjugate-gradient solve for the
    correction to the lift, certified on its residual (``tolerance`` relative to
    the lift's residual); a rejected solve raises eagerly and gives nonfinite
    fields under tracing. ``max_iterations`` follows the tolerance rule of #150
    (600 restarts of 60).

    With advection, ``method`` is ``"march"`` (RK3 steps at ``cfl`` times the Courant limit of the
    current state, ``check`` steps between tests of the steady residual, at most ``max_steps``),
    ``"newton"`` or ``"auto"``, which is :func:`method_for`; ``iterations`` counts the steps of the
    march and the Krylov iterations of Newton, and ``status`` is one of
    :data:`CONVERGED`, :data:`STAGNATED` and :data:`UNSTEADY`; only ``CONVERGED`` is a steady
    state, and its derivatives are the only ones to trust.

    Central transport needs the axial cells to keep the cell Reynolds number
    ``U dx / nu`` small: with :func:`fringe_duct`'s default buffer growth the
    downstream cells reach ``8 * spacing`` (cell Reynolds number 66 at Re 33),
    and at Ha 100 the iteration stalls between Re 31 and 33; with
    ``growth=1.0`` it converges (plan 1.9d).
    """
    axis = problem.open_axis
    if axis != 0:
        raise ValueError("solve_open_duct needs the first axis to be the inflow-outflow axis")
    with jax.ensure_compile_time_eval():
        factorization = problem.factorization()
        viscous = _projection_solves(problem, float(problem.dt))
    precond = _preconditioner(problem, factorization, viscous, float(problem.dt))
    stokes = dataclasses.replace(problem, advection="off")
    start = _lift(problem)
    zero = (0.0, 0.0, 0.0)
    rhs = steady_residual(start, stokes, factorization, forcing=zero, field_scale=field_scale, inflow=1.0)
    root, (iterations, _, _) = _stokes_limit_root(
        stokes,
        start,
        factorization,
        precond,
        forcing=zero,
        field_scale=field_scale,
        tolerance=tolerance,
        max_iterations=max_iterations,
        rhs=rhs,
    )
    correction, _ = project(jax.tree.map(jnp.subtract, root, start), problem, factorization)
    velocity = with_inflow(jax.tree.map(jnp.add, start, correction), problem)
    status = jnp.asarray(CONVERGED)
    if problem.advection != "off":
        velocity, iterations, status = _steady_root(
            problem,
            velocity,
            factorization,
            precond,
            field_scale,
            tolerance,
            _norm(rhs),
            method_for(problem, cfl) if method == "auto" else method,
            (check, cfl, max_steps),
        )
    terms = momentum_terms(
        velocity, problem, factorization, forcing=zero, field_scale=field_scale, inflow=1.0
    )
    # Two passes, as in the residual: the second is the defect correction of the first pressure solve.
    once, first = project(terms, problem, factorization)
    residual, second = project(once, problem, factorization)
    pressure = first.replace_data(float(problem.dt) * (first.data + second.data))
    potential, currents, field = face_currents(velocity, problem, factorization, field_scale)
    return OpenDuctSolution(
        velocity, pressure, potential, currents, field, _norm(residual), _norm(rhs), iterations, status
    )


_compiled = jax.jit(
    _solve, static_argnames="problem tolerance max_iterations method max_steps check cfl".split()
)


@functools.wraps(_solve, assigned=("__doc__",), updated=())
def solve_open_duct(*arguments, **controls) -> OpenDuctSolution:
    # Eagerly, one compiled program per problem and controls, whether or not ``field_scale`` is passed;
    # under a trace the solve joins the caller's program. A rejected solve is nonfinite and raises here.
    controls = {"field_scale": 1.0, **controls}
    traced = isinstance(controls["field_scale"], jax.core.Tracer)
    solution = (_solve if traced else _compiled)(*arguments, **controls)
    if not isinstance(solution.residual_norm, jax.core.Tracer) and not np.isfinite(solution.residual_norm):
        raise RuntimeError("the open-duct solve did not converge")
    return solution


def method_for(problem: ChannelProblem, cfl: float = 0.9) -> str:
    """The method ``auto`` picks: ``march`` while the Courant step of the inlet flow is at most
    ``_MARCH_RATIO`` damping steps, ``newton`` where the field binds harder (numerics notes: four measured ducts)."""
    with jax.ensure_compile_time_eval():
        courant = cfl * RK3.limit * advective_step_limit(with_inflow(_lift(problem), problem))
        ratio = float(courant) * max(problem.damping_rates) / _DAMPING_COURANT
    return "march" if ratio <= _MARCH_RATIO else "newton"


def _steady_root(
    problem, velocity, factorization, precond, field_scale, tolerance, reference, method, controls
):
    """The steady advective flow from the Stokes-limit ``velocity`` by ``method``; return it, the work and the status.

    ``march`` repeats chunks of ``check`` RK3 steps at the smaller of ``cfl`` times the Courant limit and the
    damping limit until the steady residual is under ``tolerance`` of ``reference``. ``newton`` updates the
    correction to the lift (at most 12 times) by :func:`solvax.gcrot` preconditioned with a loose solve of the
    Stokes-limit operator. A solve that has not converged is classified by :func:`_watch`. Either root is rebuilt
    by :func:`solvax.root_solve` on the fixed point of one step, which carries the derivatives.
    """
    check, cfl, max_steps = controls
    lift, viscous = with_inflow(_lift(problem), problem), problem.viscous_factorizations(RK3)
    flat, unravel = ravel_pytree(jax.lax.stop_gradient(velocity))
    fixed, zero = jax.lax.stop_gradient(field_scale), (0.0, 0.0, 0.0)
    guess = Field(jnp.zeros(problem.grid.shape, flat.dtype), (CENTER,) * 3, problem.grid)

    def advance(x, dt, scale, pressure):
        state, pressure, _, _ = _step(
            unravel(x), problem, factorization, viscous, RK3, None, 0.0, pressure, (dt, scale, lift)
        )
        return ravel_pytree(state)[0], pressure

    def step_size(x, scale):
        rate = max(problem.damping_rates) * scale**2
        return jnp.minimum(
            cfl * RK3.limit * advective_step_limit(unravel(x)), _DAMPING_COURANT / jnp.maximum(rate, 1e-300)
        )

    def steady(x, scale, of=problem, inflow=1.0):
        found = steady_residual(unravel(x), of, factorization, forcing=zero, field_scale=scale, inflow=inflow)
        return ravel_pytree(found)[0]

    def chunk(x):
        dt = step_size(x, fixed)
        return jax.lax.fori_loop(
            0, check, lambda _, carry: advance(carry[0], dt, fixed, carry[1]), (x, guess)
        )[0]

    def residual(x):
        return jnp.linalg.norm(steady(x, fixed))

    if method == "march":
        found = solvax.fixed_point_iteration(
            chunk,
            flat,
            residual_norm=lambda x: residual(x) / jax.lax.stop_gradient(reference),
            atol=tolerance,
            rtol=0.0,
            max_steps=max_steps // check,
        )
        root, work, converged = found.x, found.iterations * check, found.converged
    else:
        stokes, flat_lift = dataclasses.replace(problem, advection="off"), ravel_pytree(lift)[0]
        weights = ravel_pytree(_face_weights(stokes))[0]

        def inside(w):
            return ravel_pytree(_orthogonal_projection(unravel(w), problem, factorization))[0]

        def function(w):
            # Extended off the divergence-free fields by the identity: invertible on every vector.
            return steady(flat_lift + inside(w), fixed) + (w - inside(w))

        def inverse(r):
            solved = solvax.pcg(
                lambda y: -steady(y / weights, fixed, stokes, None),
                -inside(r),
                precond=lambda r: ravel_pytree(precond(unravel(r)))[0] * weights,
                rtol=0.1,
                max_steps=4_000,
            )
            return solved.x / weights + (r - inside(r))

        def update(carry):
            w, count, total, _ = carry
            value, linear = jax.linearize(function, w)
            krylov = solvax.gcrot(linear, -value, precond=inverse, m=120, k=10, rtol=1e-3, max_restarts=30)
            return w + krylov.x, count + 1, total + krylov.iterations, jnp.linalg.norm(function(w + krylov.x))

        def going(carry):
            return (carry[1] < 12) & (carry[3] > tolerance * reference) & jnp.isfinite(carry[3])

        w = inside(jax.lax.stop_gradient(flat) - flat_lift)
        w, _, work, norm = jax.lax.while_loop(going, update, (w, 0, 0, jnp.linalg.norm(function(w))))
        root, converged = flat_lift + inside(w), norm <= 10.0 * tolerance * reference

    status = jax.lax.cond(
        converged, lambda _: jnp.asarray(CONVERGED), lambda x: _watch(chunk, residual, x), root
    )
    dt = step_size(root, fixed)
    root = solvax.root_solve(
        lambda x: advance(x, dt, field_scale, guess)[0] - x,
        root,
        lambda function, x: x,
        tangent_solve=_tangent_solve,
    )
    return unravel(root), work, status


def _watch(chunk, residual, x):
    """Classify a march that has not converged: sixteen more chunks from ``x`` say how it fails.

    A residual that has not halved over them and either grows by half or changes between chunks
    that turn through more than sixty degrees (an oscillation) means a flow with no steady state,
    ``UNSTEADY``; a residual that is flat, or falling too slowly or still falling, is ``STAGNATED``.

    Under ``vmap`` over ``field_scale`` the ``lax.cond`` predicate is batched, so both branches run
    and every solve pays all its chunks.
    """

    def body(x, _):
        y = chunk(x)
        return y, (y - x, residual(y))

    _, (changes, norms) = jax.lax.scan(body, x, None, length=16)
    unit = changes / jnp.maximum(jnp.linalg.norm(changes, axis=1, keepdims=True), jnp.finfo(x.dtype).tiny)
    turning = jnp.min(unit @ unit.T) < 0.5
    return jnp.where(
        (norms[-1] > 0.5 * norms[0]) & ((norms[-1] > 1.5 * norms[0]) | turning), UNSTEADY, STAGNATED
    )


def _lift(problem: ChannelProblem) -> tuple[Field, Field, Field]:
    """The inlet profile on every axial face but the inlet: divergence free with the inlet added."""
    velocity = zero_velocity(problem)
    profile = jnp.asarray(problem.conditions[0].lower, dtype=velocity[0].dtype)
    axial = velocity[0].replace_data(jnp.broadcast_to(profile, velocity[0].shape).at[0].set(0.0))
    return (axial, *velocity[1:])


def station_flow_rates(velocity: tuple[Field, Field, Field]) -> jnp.ndarray:
    """The flow rate through every axial face."""
    grid = velocity[0].grid
    return jnp.sum(velocity[0].data * jnp.asarray(grid.face_areas(0)), axis=(1, 2))


def station_pressure(pressure: Field) -> tuple[np.ndarray, jnp.ndarray]:
    """The axial cell centres and the area-mean pressure over each cross-section."""
    grid = pressure.grid
    areas = jnp.asarray(grid.face_areas(0)[0])
    return np.asarray(grid.centers[0]), jnp.sum(pressure.data * areas, axis=(1, 2)) / jnp.sum(areas)


def pressure_drop(pressure: Field, start: float, end: float) -> jnp.ndarray:
    """Mean pressure at ``start`` minus that at ``end``, interpolated linearly between centres."""
    centres, means = station_pressure(pressure)
    return jnp.interp(start, centres, means) - jnp.interp(end, centres, means)


def mass_balance(velocity: tuple[Field, Field, Field]) -> jnp.ndarray:
    """Largest net volume flux out of a cell, relative to the largest flux through a cell."""
    return _balance(velocity)


def charge_balance(solution: OpenDuctSolution, problem: ChannelProblem) -> jnp.ndarray:
    """Largest net current out of a cell, relative to the largest motional current through a cell.

    The motional current ``sigma (u x B).n`` is what the potential balances; the
    net current is a small difference of it and the potential gradient, so it is
    the scale the charge equation is solved to.
    """
    scalar = problem.scalar_conditions
    motional = tuple(
        face_electromotive_force(solution.velocity, solution.magnetic_field, axis, scalar)
        for axis in range(3)
    )
    scaled = tuple(m.replace_data(float(problem.conductivity) * m.data) for m in motional)
    return _balance(solution.currents, reference=scaled)


def _balance(
    faces: tuple[Field, Field, Field], reference: tuple[Field, Field, Field] | None = None
) -> jnp.ndarray:
    grid = faces[0].grid

    def sums(fields):
        net, gross = 0.0, 0.0
        for axis, face in enumerate(fields):
            flux = face.data * jnp.asarray(grid.face_areas(axis))
            lower, upper = neighbours(flux, axis)
            net, gross = net + upper - lower, gross + jnp.abs(upper) + jnp.abs(lower)
        return net, gross

    net, gross = sums(faces)
    if reference is not None:
        gross = sums(reference)[1]
    return jnp.max(jnp.abs(net)) / jnp.max(gross)
