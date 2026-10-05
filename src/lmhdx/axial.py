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

With advection on (plan 1.9d) the steady problem is nonlinear. It is solved
by Newton's method from the Stokes-limit solution, each Newton update by
flexible GMRES with recycling (:func:`solvax.gcrot`) right-preconditioned by
the Stokes-limit solve itself, a conjugate-gradient solve run to a loose
tolerance: the Oseen operator differs from the Stokes one by the transport,
which is small against the Lorentz force when ``N = Ha^2 / Re`` is large, so a
few outer iterations suffice. Continuation in the flow rate steps toward the
target Reynolds number. The root is differentiated by the implicit function
theorem (:func:`solvax.root_solve`), its tangent and transposed solves being
the same preconditioned Krylov solves.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import solvax
from jax.flatten_util import ravel_pytree

from .core3d import (
    ChannelProblem,
    ImposedField,
    duct_problem,
    face_currents,
    fringe_field,
    project,
    zero_velocity,
)
from .grid import DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition, Field, Grid, uniform_faces
from .ops import face_electromotive_force
from .steady import (
    _certified,
    _face_weights,
    _norm,
    _orthogonal_projection,
    _preconditioner,
    _projection_solves,
    _stokes_limit_root,
    momentum_terms,
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
    "open_duct",
    "pressure_drop",
    "solve_open_duct",
    "station_flow_rates",
    "station_pressure",
]

_INSULATING = BoundaryCondition(NEUMANN)


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
    periodic = ChannelProblem(
        grid=section,
        conditions=(BoundaryCondition(PERIODIC), *problem.conditions[1:]),
        density=problem.density,
        viscosity=problem.viscosity,
        conductivity=problem.conductivity,
        magnetic_field=field,
        forcing=(1.0, 0.0, 0.0),
        dt=problem.dt,
        wall_conductance=problem.wall_conductance,
        precision=problem.precision,
    )
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
    return ChannelProblem(
        grid=problem.grid,
        conditions=(BoundaryCondition(DIRICHLET, lower=profile, upper_kind=NEUMANN), *problem.conditions[1:]),
        density=problem.density,
        viscosity=problem.viscosity,
        conductivity=problem.conductivity,
        magnetic_field=problem.magnetic_field,
        forcing=(0.0, 0.0, 0.0),
        dt=problem.dt,
        wall_conductance=(0.0, *problem.wall_conductance[1:]),
        advection=problem.advection,
        precision=problem.precision,
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
    ``flow_rate / 4``, and ``N = Ha^2 / Re``.
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


def solve_open_duct(
    problem: ChannelProblem,
    *,
    field_scale: float | jnp.ndarray = 1.0,
    tolerance: float = 1.0e-9,
    max_iterations: int = 36_000,
    continuation: tuple[float, ...] = (1.0,),
    max_newton_steps: int = 12,
    inner_tolerance: float = 1.0e-2,
    inner_iterations: int = 4_000,
) -> OpenDuctSolution:
    """Solve an inflow-outflow duct; differentiable in ``field_scale``.

    In the Stokes limit, one preconditioned conjugate-gradient solve for the
    correction to the lift, certified on its residual (``tolerance`` relative to
    the lift's residual); a rejected solve raises eagerly and gives nonfinite
    fields under tracing. ``max_iterations`` follows the tolerance rule of #150
    (600 restarts of 60).

    With advection, Newton's method from that solution (module docstring):
    ``continuation`` lists the fractions of the flow rate solved in turn, ending
    at 1; each Newton update is a :func:`solvax.gcrot` solve preconditioned by
    the Stokes-limit CG run to ``inner_tolerance`` (at most
    ``inner_iterations``), and ``iterations`` then counts the outer Krylov
    iterations of all Newton steps. The root is certified at ``tolerance``.
    """
    axis = problem.open_axis
    if axis != 0:
        raise ValueError("solve_open_duct needs the first axis to be the inflow-outflow axis")
    if problem.advection != "off" and (not continuation or float(continuation[-1]) != 1.0):
        raise ValueError("the continuation in the flow rate must end at 1")
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
    if problem.advection != "off":
        velocity, iterations = _newton_root(
            problem,
            stokes,
            velocity,
            factorization,
            precond,
            field_scale=field_scale,
            tolerance=tolerance,
            continuation=tuple(float(value) for value in continuation),
            max_steps=max_newton_steps,
            inner_tolerance=inner_tolerance,
            inner_iterations=inner_iterations,
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
        velocity, pressure, potential, currents, field, _norm(residual), _norm(rhs), iterations
    )


def _stokes_inverse(stokes, factorization, precond, field_scale, tolerance, max_steps):
    """``r -> d`` with ``A d = r`` approximately, ``A`` the Stokes-limit operator on the corrections.

    The same weighted conjugate-gradient solve as the Stokes limit
    (:func:`lmhdx.steady._stokes_limit_root`), stopped at ``tolerance``; its
    result is not a linear function of ``r``, which is why the outer Krylov
    method is a flexible one.
    """
    weights = _face_weights(stokes)
    zero = (0.0, 0.0, 0.0)

    def apply(target):
        inside = _orthogonal_projection(target, stokes, factorization)

        def matvec(y):
            velocity = jax.tree.map(jnp.divide, y, weights)
            return steady_residual(velocity, stokes, factorization, forcing=zero, field_scale=field_scale)

        result = solvax.pcg(
            lambda y: jax.tree.map(jnp.negative, matvec(y)),
            jax.tree.map(jnp.negative, inside),
            precond=lambda r: jax.tree.map(jnp.multiply, precond(r), weights),
            rtol=tolerance,
            max_steps=max_steps,
        )
        return jax.tree.map(jnp.divide, result.x, weights)

    return apply


def _newton_root(
    problem,
    stokes,
    velocity,
    factorization,
    precond,
    *,
    field_scale,
    tolerance,
    continuation,
    max_steps,
    inner_tolerance,
    inner_iterations,
):
    """Newton from the Stokes-limit ``velocity`` along the flow-rate ``continuation``; see the module.

    The unknown is the correction ``w`` to the lift, and the residual is extended off
    the constrained divergence-free fields ``V`` by the identity,
    ``F(w) = R(lift + Q w) + (w - Q w)`` with ``Q`` the orthogonal projection onto
    ``V``, as the Stokes-limit derivative is: its Jacobian is invertible on every
    vector, so the transposed solve of an arbitrary cotangent is consistent.
    """
    zero = (0.0, 0.0, 0.0)
    lift_tree = _lift(problem)
    lift, unravel = ravel_pytree(lift_tree)
    fixed = jax.lax.stop_gradient(field_scale)
    inverse = _stokes_inverse(stokes, factorization, precond, fixed, inner_tolerance, inner_iterations)

    def inside(w):
        return ravel_pytree(_orthogonal_projection(unravel(w), problem, factorization))[0]

    def residual_at(fraction, scale):
        def residual(w):
            kept = inside(w)
            state = unravel(fraction * lift + kept)
            value = steady_residual(
                state, problem, factorization, forcing=zero, field_scale=scale, inflow=fraction
            )
            return ravel_pytree(value)[0] + (w - kept)

        return residual

    def preconditioner(r):
        kept = inside(r)
        return ravel_pytree(inverse(unravel(kept)))[0] + (r - kept)

    weights = ravel_pytree(_face_weights(problem))[0]

    def transposed_preconditioner(r):
        # The preconditioner is self-adjoint in the face-volume inner product W, so W M W^-1 is its transpose.
        return weights * preconditioner(r / weights)

    def krylov(matvec, target, rtol, precond=preconditioner):
        result = solvax.gcrot(matvec, target, precond=precond, m=30, k=10, rtol=rtol, max_restarts=30)
        return result.x, result.iterations, result.converged & jnp.isfinite(result.residual_norm)

    def newton(function, guess, fraction):
        # Relative to the residual of the lift, as the Stokes-limit solve is.
        scale = jnp.linalg.norm(function(jnp.zeros_like(guess)))

        def body(carry):
            w, steps, total, _ = carry
            value, linear = jax.linearize(function, w)
            step, count, _ = krylov(linear, -value, 1.0e-3)
            return w + step, steps + 1, total + count, jnp.linalg.norm(function(w + step))

        def going(carry):
            _, steps, _, norm = carry
            return (steps < max_steps) & (norm > tolerance * scale) & jnp.isfinite(norm)

        first = jnp.linalg.norm(function(guess))
        w, _, total, norm = jax.lax.while_loop(going, body, (guess, 0, 0, first))
        return w, total, norm <= 10.0 * tolerance * scale

    def tangent_solve(linear, target):
        def solve(matvec, rhs):
            x, _, accepted = krylov(matvec, rhs, 1.0e-10)
            return _certified(x, accepted, "open-duct tangent solve")

        def transpose_solve(vecmat, rhs):
            x, _, accepted = krylov(vecmat, rhs, 1.0e-10, transposed_preconditioner)
            return _certified(x, accepted, "open-duct adjoint solve")

        return jax.lax.custom_linear_solve(linear, target, solve, transpose_solve)

    total = jnp.asarray(0)
    w = inside(jax.lax.stop_gradient(ravel_pytree(velocity)[0] - lift))
    previous = 1.0
    for fraction in continuation:
        w = w * (fraction / previous)
        w, count, accepted = newton(residual_at(fraction, fixed), w, fraction)
        w = _certified(w, accepted, f"Newton solve at flow-rate fraction {fraction:g}")
        total, previous = total + count, fraction
    root = solvax.root_solve(
        residual_at(1.0, field_scale), w, lambda function, guess: guess, tangent_solve=tangent_solve
    )
    return with_inflow(unravel(lift + inside(root)), problem), total


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
            lower, upper = (
                jax.lax.slice_in_dim(flux, start, start + grid.shape[axis], axis=axis) for start in (0, 1)
            )
            net, gross = net + upper - lower, gross + jnp.abs(upper) + jnp.abs(lower)
        return net, gross

    net, gross = sums(faces)
    if reference is not None:
        gross = sums(reference)[1]
    return jnp.max(jnp.abs(net)) / jnp.max(gross)


# The open pipe: an inlet and an outlet on the polar grid (validation row 6, the ALEX pipe).


@dataclasses.dataclass(frozen=True, eq=False)
class OpenPipe:
    """A circular pipe of unit radius with an inlet and an outlet, in the Stokes limit.

    ``grid`` is polar with the axes ``(r, theta, x)``: the axial direction is the
    third axis, the inlet its lower end. The imposed field is transverse,
    ``B = hartmann * field(x)`` along ``theta = 0``, uniform over each
    cross-section and so divergence free; ``field`` holds its value at the axial
    cell centres. Density, viscosity, conductivity and the radius are one, so
    pressures are in units of ``rho nu U / a`` and divide by ``hartmann**2`` to
    ``sigma U B0^2 a``. ``inlet`` is the axial velocity on the inlet face.
    """

    grid: Grid
    hartmann: float
    wall_conductance: float
    field: np.ndarray
    inlet: np.ndarray

    def __post_init__(self) -> None:
        if not self.grid.is_polar or float(self.grid.x_faces[0]) != 0.0:
            raise ValueError("an open pipe needs a polar grid that starts on the axis")
        field = np.array(self.field, dtype=float)
        if field.shape != (self.grid.shape[2],):
            raise ValueError("the field needs one value per axial cell")
        field.setflags(write=False)
        object.__setattr__(self, "field", field)


class OpenPipeSolution(NamedTuple):
    """Velocity ``(u_r, u_theta, u_x)`` on its faces, pressure and potential at the cells, currents on the faces.

    The azimuthal faces are stored once each, face ``j`` below cell ``j``.
    """

    velocity: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    pressure: jnp.ndarray
    potential: jnp.ndarray
    currents: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    residual_norm: jnp.ndarray
    initial_residual_norm: jnp.ndarray
    iterations: jnp.ndarray


def monotone_interpolant(x: np.ndarray, y: np.ndarray):
    """The shape-preserving cubic (Fritsch-Carlson, as PCHIP) through tabulated points; no extrapolation."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    width, slope = np.diff(x), np.diff(y) / np.diff(x)
    tangent = np.zeros_like(y)
    for i in range(1, len(x) - 1):
        if slope[i - 1] * slope[i] > 0.0:
            w1, w2 = 2.0 * width[i] + width[i - 1], width[i] + 2.0 * width[i - 1]
            tangent[i] = (w1 + w2) / (w1 / slope[i - 1] + w2 / slope[i])
    tangent[0], tangent[-1] = slope[0], slope[-1]

    def evaluate(points):
        points = np.asarray(points, dtype=float)
        if np.any(points < x[0] - 1e-12) or np.any(points > x[-1] + 1e-12):
            raise ValueError("the tabulated field is not extrapolated")
        i = np.clip(np.searchsorted(x, points) - 1, 0, len(x) - 2)
        t, h = (points - x[i]) / width[i], width[i]
        return (
            (2 * t**3 - 3 * t**2 + 1) * y[i]
            + (t**3 - 2 * t**2 + t) * h * tangent[i]
            + (-2 * t**3 + 3 * t**2) * y[i + 1]
            + (t**3 - t**2) * h * tangent[i + 1]
        )

    return evaluate


def fringe_pipe(
    *,
    hartmann: float,
    field,
    wall_conductance: float = 0.0,
    lower: float = -15.0,
    upper: float = 10.0,
    core: tuple[float, float] = (-6.0, 6.0),
    spacing: float = 0.25,
    radial: int = 24,
    azimuthal: int = 32,
    cells_in_layer: int = 6,
    flow_rate: float = np.pi,
) -> OpenPipe:
    """An open pipe from ``lower`` to ``upper`` radii in the field ``hartmann * field(x)``.

    ``field`` is a callable of the axial position (:func:`monotone_interpolant` of
    a table); the axial mesh has ``spacing`` over ``core`` and grows into the
    buffers (:func:`axial_faces`). The radial mesh resolves the ``1/Ha`` layer as
    :func:`lmhdx.poisson.pipe_grid` does. The inlet profile is LMhdX's fully developed pipe
    (:func:`lmhdx.poisson.solve_pipe`) at the inlet field, scaled to ``flow_rate`` (a
    unit mean velocity by default).
    """
    from .poisson import PipeProblem, pipe_grid, solve_pipe

    section = pipe_grid(radial, azimuthal, hartmann, cells_in_layer=cells_in_layer)
    grid = Grid(
        section.x_faces, section.y_faces, axial_faces(lower, upper, core, spacing), geometry=section.geometry
    )
    values = np.asarray(field(np.asarray(grid.centers[2])), dtype=float)
    first = PipeProblem(section, float(hartmann) * float(values[0]), float(wall_conductance))
    profile, _ = solve_pipe(first)
    axial = np.asarray(profile.data)[:, :, 0]
    unit = float(np.sum(axial * section.face_areas(2)[:, :, 0]))
    return OpenPipe(grid, float(hartmann), float(wall_conductance), values, axial * (flow_rate / unit))


def _pipe_metric(pipe: OpenPipe) -> dict:
    """Areas, distances and weights of the polar staggered layout, broadcast to ``(r, theta, x)``."""
    grid = pipe.grid
    rf = np.asarray(grid.x_faces)
    rc, xc = np.asarray(grid.centers[0]), np.asarray(grid.centers[2])
    dr, dx = np.asarray(grid.widths[0]), np.asarray(grid.widths[2])
    dt = float(grid.widths[1][0])
    r, x = (lambda a: a[:, None, None]), (lambda a: a[None, None, :])
    # Distances across faces; the boundary entries are half cells (the wall, the two ends).
    radial = np.concatenate(([1.0], np.diff(rc), [0.5 * dr[-1]]))
    axial = np.concatenate(([0.5 * dx[0]], np.diff(xc), [0.5 * dx[-1]]))
    volume = r(rc * dr) * dt * x(dx)
    ring = np.ones((1, grid.shape[1], 1))
    weights = (r(rf * radial) * dt * x(dx) * ring, volume * ring, r(rc * dr) * dt * x(axial) * ring)
    # Constrained faces (axis, wall, inlet) keep a unit weight: they are masked, never divided by zero.
    weights[0][[0, -1]] = 1.0
    weights[2][..., 0] = 1.0
    return dict(
        rf=rf,
        rc=rc,
        dr=dr,
        dx=dx,
        dt=dt,
        radial=radial,
        axial=axial,
        volume=volume,
        weights=weights,
        area=(r(rf) * dt * x(dx), r(dr) * x(dx), r(rc * dr) * dt),
        sin=np.sin(np.asarray(grid.centers[1]))[None, :, None],
        cos=np.cos(np.asarray(grid.centers[1]))[None, :, None],
    )


def _pipe_mask(velocity, inlet=None):
    """Zero ``u_r`` on the axis and the wall, and set the inlet face of ``u_x`` (zero by default)."""
    ur, ut, ux = velocity
    ux = ux.at[..., 0].set(0.0 if inlet is None else inlet)
    return ur.at[0].set(0.0).at[-1].set(0.0), ut, ux


def _pipe_divergence(velocity, metric):
    ur, ut, ux = velocity
    ar, at, ax = (jnp.asarray(a) for a in metric["area"])
    fr, ft, fx = ar * ur, at * ut, ax * ux
    net = fr[1:] - fr[:-1] + jnp.roll(ft, -1, axis=1) - ft + fx[..., 1:] - fx[..., :-1]
    return net / jnp.asarray(metric["volume"])


def _pipe_gradient(field, metric, outlet: bool):
    """Face gradient of a cell field: zero on the axis, wall and inlet; ``p = 0`` beyond the outlet if ``outlet``."""
    radial = (field[1:] - field[:-1]) / jnp.asarray(metric["radial"][1:-1])[:, None, None]
    edge = jnp.zeros_like(field[:1])
    gr = jnp.concatenate((edge, radial, edge), axis=0)
    gt = (field - jnp.roll(field, 1, axis=1)) / jnp.asarray(metric["rc"] * metric["dt"])[:, None, None]
    axial = jnp.diff(field, axis=2) / jnp.asarray(metric["axial"][1:-1])[None, None, :]
    end = jnp.zeros_like(field[..., :1])
    last = -field[..., -1:] / float(metric["axial"][-1]) if outlet else end
    return gr, gt, jnp.concatenate((end, axial, last), axis=2)


def _pipe_centres(velocity, metric):
    """Velocity components at the cell centres: each the average of its two faces over the cell."""
    ur, ut, ux = velocity
    rf, rc = metric["rf"], metric["rc"]
    lower, upper = (jnp.asarray(w / (2.0 * rc))[:, None, None] for w in (rf[:-1], rf[1:]))
    return (
        lower * ur[:-1] + upper * ur[1:],
        0.5 * (ut + jnp.roll(ut, -1, axis=1)),
        0.5 * (ux[..., :-1] + ux[..., 1:]),
    )


def _pipe_motional(velocity, pipe, metric, scale):
    """``(u x B).n`` on every face, ``B = Ha b(x)`` along ``theta = 0``; zero on the wall, axis and ends."""
    ur, ut, ux = _pipe_centres(velocity, metric)
    strength = scale * float(pipe.hartmann) * jnp.asarray(pipe.field)[None, None, :]
    sin, cos = jnp.asarray(metric["sin"]), jnp.asarray(metric["cos"])
    er, et, ex = ux * strength * sin, ux * strength * cos, -strength * (ur * sin + ut * cos)
    dr, dx = metric["dr"], metric["dx"]
    share = jnp.asarray(dr[:-1] / (dr[:-1] + dr[1:]))[:, None, None]
    radial = share * er[:-1] + (1.0 - share) * er[1:]
    edge = jnp.zeros_like(er[:1])
    along = jnp.asarray(dx[:-1] / (dx[:-1] + dx[1:]))[None, None, :]
    axial = along * ex[..., :-1] + (1.0 - along) * ex[..., 1:]
    end = jnp.zeros_like(ex[..., :1])
    return (
        jnp.concatenate((edge, radial, edge), axis=0),
        0.5 * (jnp.roll(et, 1, axis=1) + et),
        jnp.concatenate((end, axial, end), axis=2),
    )


def _pipe_electric(velocity, pipe, metric, potential_solver, scale):
    """Potential, face currents and the Lorentz force on the velocity faces (minus the adjoint of the EMF)."""
    motional, pullback = jax.vjp(lambda u: _pipe_motional(u, pipe, metric, scale), velocity)
    grid = pipe.grid
    source = Field(_pipe_divergence(motional, metric), (0.5, 0.5, 0.5), grid)
    if pipe.wall_conductance:
        potential, wall = potential_solver.solve_with_wall(source)
        potential, sheet = potential.data, wall.data[-1]
    else:
        potential, sheet = potential_solver.solve(source).data, None
    gradient = _pipe_gradient(potential, metric, outlet=False)
    currents = [m - g for m, g in zip(motional, gradient, strict=True)]
    if sheet is not None:
        currents[0] = currents[0].at[-1].set((potential[-1] - sheet) / (0.5 * float(metric["dr"][-1])))
    face = _pipe_current_weights(metric)
    work = pullback(tuple(w * j for w, j in zip(face, currents, strict=True)))[0]
    force = tuple(-f / jnp.asarray(w) for f, w in zip(work, metric["weights"], strict=True))
    return potential, tuple(currents), force


def _pipe_current_weights(metric):
    """The face measures of the current inner product: area times the distance across the face."""
    ar, at, ax = metric["area"]
    rc, dt = metric["rc"], metric["dt"]
    return (
        jnp.asarray(ar * metric["radial"][:, None, None]),
        jnp.asarray(at * (rc * dt)[:, None, None]),
        jnp.asarray(ax * metric["axial"][None, None, :]),
    )


def _pipe_dissipation(velocity, metric):
    """Half the viscous dissipation, ``(1/2) sum |grad u|^2 dV`` in cylindrical components; no slip at the wall.

    The gradient tensor's azimuthal entries carry the curvature terms,
    ``(d_theta u_r - u_theta)/r`` and ``(d_theta u_theta + u_r)/r``, so the
    variation of this functional is the vector Laplacian; written as a sum of
    squares it is a symmetric, positive operator by construction. Transverse
    velocity vanishes at the inlet; the outlet is natural (zero axial gradient).
    """
    ur, ut, ux = velocity
    rf, rc, dr, dx, dt = (metric[k] for k in ("rf", "rc", "dr", "dx", "dt"))
    axial = metric["axial"]
    r, x = (lambda a: jnp.asarray(a)[:, None, None]), (lambda a: jnp.asarray(a)[None, None, :])
    span = np.diff(metric["rc"])
    xc_gap = axial[1:-1]
    total = 0.0

    def square(difference, weight):
        return jnp.sum(weight * difference**2)

    # u_r on the interior radial faces 1..nr-1.
    inner = ur[1:-1]
    total += square((ur[1:] - ur[:-1]) / r(dr), r(rc * dr) * dt * x(dx))
    shift = (rf[1:-1] - rc[:-1]) / span
    ut_face = ut[:-1] + r(shift) * (ut[1:] - ut[:-1])
    total += square(
        ((inner - jnp.roll(inner, 1, axis=1)) / dt - ut_face) / r(rf[1:-1]), r(span * rf[1:-1]) * dt * x(dx)
    )
    total += square(jnp.diff(inner, axis=2) / x(xc_gap), r(span * rf[1:-1]) * dt * x(xc_gap))
    total += square(inner[..., :1] / (0.5 * dx[0]), r(span * rf[1:-1]) * dt * 0.5 * dx[0])
    # u_theta at the cell centres in r and x.
    ur_c = _pipe_centres(velocity, metric)[0]
    total += square((ut[1:] - ut[:-1]) / r(span), r(rf[1:-1] * span) * dt * x(dx))
    total += square(ut[-1:] / (0.5 * dr[-1]), rf[-1] * dt * x(dx) * 0.5 * dr[-1])
    total += square(((jnp.roll(ut, -1, axis=1) - ut) / dt + ur_c) / r(rc), r(rc * dr) * dt * x(dx))
    total += square(jnp.diff(ut, axis=2) / x(xc_gap), r(rc * dr) * dt * x(xc_gap))
    total += square(ut[..., :1] / (0.5 * dx[0]), r(rc * dr) * dt * 0.5 * dx[0])
    # u_x on the axial faces, the inlet face included as data.
    total += square((ux[1:] - ux[:-1]) / r(span), r(rf[1:-1] * span) * dt * x(axial))
    total += square(ux[-1:] / (0.5 * dr[-1]), rf[-1] * dt * x(axial) * 0.5 * dr[-1])
    total += square((ux - jnp.roll(ux, 1, axis=1)) / (r(rc) * dt), r(dr * rc) * dt * x(axial))
    total += square(jnp.diff(ux, axis=2) / x(dx), r(rc * dr) * dt * x(dx))
    return 0.5 * total


class _ComponentHelmholtz:
    """``(s W + H)^-1 W`` for one velocity component: ``H`` its own viscous part, separable per azimuthal mode.

    ``W = dtheta m_r (x) m_x`` and ``H = dtheta [K_r (x) m_x + kappa_m g_r (x) m_x + m_r (x) K_x]`` in
    the azimuthal Fourier basis, ``kappa_m = 4 sin^2(pi m / n) / dtheta^2``; the curvature terms
    are left out, which only weakens the preconditioner. One generalized radial
    eigendecomposition per mode and one axial.
    """

    def __init__(self, radial, axial, count: int, dtheta: float, shift: float):
        (mass_r, stiff_r, coupling), (mass_x, stiff_x) = radial, axial
        kappa = 4.0 * np.sin(np.pi * np.arange(count) / count) ** 2 / dtheta**2
        root = 1.0 / np.sqrt(mass_r)
        vectors, values = [], []
        for value in kappa:
            matrix = root[:, None] * (stiff_r + np.diag(value * coupling)) * root[None, :]
            lam, vec = np.linalg.eigh(0.5 * (matrix + matrix.T))
            vectors.append(root[:, None] * vec)
            values.append(lam)
        xroot = 1.0 / np.sqrt(mass_x)
        lam_x, vec_x = np.linalg.eigh(xroot[:, None] * stiff_x * xroot[None, :])
        self.radial = np.stack(vectors)  # (mode, node, eigen)
        self.axial = xroot[:, None] * vec_x
        self.denominator = shift + np.stack(values)[:, :, None] + lam_x[None, None, :]
        self.mass = mass_r[:, None, None] * mass_x[None, None, :]

    def solve(self, rhs):
        """The component ``q`` with ``(s W + H) q = W rhs``; ``rhs`` shaped ``(r, theta, x)``."""
        data = jnp.fft.fft(jnp.asarray(self.mass) * rhs, axis=1)
        data = jnp.einsum("mie,imx->emx", jnp.asarray(self.radial), data)
        data = jnp.einsum("emx,xk->emk", data, jnp.asarray(self.axial))
        data = data / jnp.moveaxis(jnp.asarray(self.denominator), 0, 1)
        data = jnp.einsum("emk,xk->emx", data, jnp.asarray(self.axial))
        data = jnp.einsum("mie,emx->imx", jnp.asarray(self.radial), data)
        return jnp.real(jnp.fft.ifft(data, axis=1))


def _tridiagonal(conductances: np.ndarray, ends: tuple[float, float]) -> np.ndarray:
    """The stiffness of ``sum c_k (q_{k+1} - q_k)^2 / 2`` plus Dirichlet ghosts ``ends`` on the two end nodes."""
    size = len(conductances) + 1
    matrix = np.zeros((size, size))
    for k, value in enumerate(conductances):
        matrix[k, k] += value
        matrix[k + 1, k + 1] += value
        matrix[k, k + 1] -= value
        matrix[k + 1, k] -= value
    matrix[0, 0] += ends[0]
    matrix[-1, -1] += ends[1]
    return matrix


def _pipe_helmholtz(pipe: OpenPipe, metric, shift: float):
    """The three component solves of the open pipe's preconditioner (:class:`_ComponentHelmholtz`)."""
    rf, rc, dr, dx = metric["rf"], metric["rc"], metric["dr"], metric["dx"]
    span, axial = np.diff(rc), metric["axial"]
    count, dtheta = pipe.grid.shape[1], metric["dt"]
    cells_r = (rc * dr, _tridiagonal(rf[1:-1] / span, (0.0, rf[-1] / (0.5 * dr[-1]))), dr / rc)
    faces_r = (
        rf[1:-1] * span,
        _tridiagonal(rc[1:-1] / dr[1:-1], (rc[0] / dr[0], rc[-1] / dr[-1])),
        span / rf[1:-1],
    )
    cells_x = (dx, _tridiagonal(1.0 / np.diff(np.asarray(pipe.grid.centers[2])), (1.0 / (0.5 * dx[0]), 0.0)))
    faces_x = (axial[1:], _tridiagonal(1.0 / dx[1:], (1.0 / dx[0], 0.0)))
    return (
        _ComponentHelmholtz(faces_r, cells_x, count, dtheta, shift),
        _ComponentHelmholtz(cells_r, cells_x, count, dtheta, shift),
        _ComponentHelmholtz(cells_r, faces_x, count, dtheta, shift),
    )


def _pipe_solvers(pipe: OpenPipe):
    from .poisson import fast_diagonal_polar_poisson

    wrap = BoundaryCondition(PERIODIC)
    pressure = fast_diagonal_polar_poisson(
        pipe.grid, (_INSULATING, wrap, BoundaryCondition(NEUMANN, upper_kind=DIRICHLET))
    )
    potential = fast_diagonal_polar_poisson(
        pipe.grid, (_INSULATING, wrap, _INSULATING), wall_conductance=float(pipe.wall_conductance)
    )
    return pressure, potential


def _pipe_project(velocity, pipe, metric, pressure_solver):
    """Remove the gradient part: the constrained, divergence-free part and the pressure it took."""
    masked = _pipe_mask(velocity)
    phi = pressure_solver.solve(Field(_pipe_divergence(masked, metric), (0.5, 0.5, 0.5), pipe.grid)).data
    gradient = _pipe_gradient(phi, metric, outlet=True)
    return _pipe_mask(tuple(v - g for v, g in zip(masked, gradient, strict=True))), phi


def _pipe_terms(velocity, pipe, metric, solvers, scale):
    """The steady momentum terms (viscous and Lorentz) on the velocity faces, before projection."""
    viscous = jax.grad(lambda u: _pipe_dissipation(u, metric))(velocity)
    _, _, force = _pipe_electric(velocity, pipe, metric, solvers[1], scale)
    return tuple(f - v / jnp.asarray(w) for v, f, w in zip(viscous, force, metric["weights"], strict=True))


def _pipe_residual(velocity, pipe, metric, solvers, scale):
    once, first = _pipe_project(_pipe_terms(velocity, pipe, metric, solvers, scale), pipe, metric, solvers[0])
    twice, second = _pipe_project(once, pipe, metric, solvers[0])
    return twice, first + second


def solve_open_pipe(
    pipe: OpenPipe,
    *,
    field_scale: float | jnp.ndarray = 1.0,
    tolerance: float = 1.0e-9,
    max_iterations: int = 36_000,
) -> OpenPipeSolution:
    """Solve the open pipe in the Stokes limit by one preconditioned conjugate-gradient solve.

    The same construction as :func:`solve_open_duct`: the inlet profile carried
    along the pipe is a divergence-free lift, the correction has no inlet flux,
    and ``-A`` is symmetric positive definite on the constrained divergence-free
    fields in the face-volume inner product (the viscous part is the Hessian of
    the dissipation, the Lorentz part minus the Gram operator of the EMF under the
    charge balance). CG runs on ``y = W u``, preconditioned by the projected
    component solves of :func:`_pipe_helmholtz` damped at ``sigma (Ha max|b|)^2``;
    differentiable in ``field_scale`` by one more CG solve. A rejected solve
    raises eagerly and gives nonfinite fields under tracing.
    """
    metric = _pipe_metric(pipe)
    with jax.ensure_compile_time_eval():
        solvers = _pipe_solvers(pipe)
    shift = (float(pipe.hartmann) * float(np.max(np.abs(pipe.field)))) ** 2
    helmholtz = _pipe_helmholtz(pipe, metric, shift)
    weights = tuple(jnp.asarray(w) for w in metric["weights"])
    inlet = jnp.asarray(pipe.inlet)[:, :, None]
    lift = (
        jnp.zeros(pipe.grid.face_shape(0)),
        jnp.zeros(pipe.grid.shape),
        jnp.broadcast_to(inlet, pipe.grid.face_shape(2)),
    )

    def operator(correction, scale):
        return _pipe_residual(_pipe_mask(correction), pipe, metric, solvers, scale)[0]

    def precondition(residual):
        solved = tuple(h.solve(r) for h, r in zip(helmholtz, _pipe_unique(residual), strict=True))
        return _pipe_project(_pipe_full(solved), pipe, metric, solvers[0])[0]

    rhs, _ = _pipe_residual(lift, pipe, metric, solvers, field_scale)

    def cg_solve(scale, target):
        result = solvax.pcg(
            lambda y: jax.tree.map(jnp.negative, operator(jax.tree.map(jnp.divide, y, weights), scale)),
            target,
            precond=lambda r: jax.tree.map(jnp.multiply, precondition(r), weights),
            rtol=tolerance,
            max_steps=max_iterations,
        )
        accepted = result.converged & jnp.isfinite(result.residual_norm)
        return _certified(result.x, accepted, "open-pipe CG solve"), result.iterations

    def inside(value):
        return _pipe_project(value, pipe, metric, solvers[0])[0]

    def matvec(scale, y):
        # Extended off the constrained divergence-free fields by the identity, so that it is
        # symmetric on every vector and the transposed solve of any cotangent is this one.
        velocity = jax.tree.map(jnp.divide, y, weights)
        kept = inside(velocity)
        applied = operator(kept, scale)
        return tuple(u - k - a for u, k, a in zip(velocity, kept, applied, strict=True))

    fixed = jax.lax.stop_gradient(field_scale)
    primal, iterations = cg_solve(fixed, jax.lax.stop_gradient(rhs))

    @jax.custom_jvp
    def solved(target, scale, known):
        return known

    @solved.defjvp
    def solved_jvp(primals, tangents):
        target, scale, primal = primals
        target_dot, scale_dot, _ = tangents
        change = jax.jvp(lambda s: matvec(s, primal), (scale,), (scale_dot,))[1]
        right = jax.tree.map(jnp.subtract, target_dot, change)

        def solve(_, value):
            kept = inside(value)
            return tuple(
                y + w * (v - k)
                for y, w, v, k in zip(cg_solve(scale, kept)[0], weights, value, kept, strict=True)
            )

        return primal, jax.lax.custom_linear_solve(lambda y: matvec(scale, y), right, solve, symmetric=True)

    y = solved(rhs, field_scale, primal)
    velocity = tuple(base + c / w for base, c, w in zip(lift, _pipe_mask(y), weights, strict=True))
    velocity = _pipe_mask(velocity, inlet[..., 0])
    residual, pressure = _pipe_residual(velocity, pipe, metric, solvers, field_scale)
    potential, currents, _ = _pipe_electric(velocity, pipe, metric, solvers[1], field_scale)
    return OpenPipeSolution(
        velocity, pressure, potential, currents, _flat_norm(residual), _flat_norm(rhs), iterations
    )


def _pipe_unique(velocity):
    """The unknown nodes of each component: interior radial faces, all azimuthal faces, axial faces past the inlet."""
    ur, ut, ux = velocity
    return ur[1:-1], ut, ux[..., 1:]


def _pipe_full(nodes):
    ur, ut, ux = nodes
    zero_r = jnp.zeros_like(ur[:1])
    return (
        jnp.concatenate((zero_r, ur, zero_r), axis=0),
        ut,
        jnp.concatenate((jnp.zeros_like(ux[..., :1]), ux), axis=2),
    )


def _flat_norm(velocity):
    return jnp.sqrt(sum(jnp.sum(v**2) for v in velocity))


def pipe_station_pressure(pipe: OpenPipe, pressure: jnp.ndarray) -> tuple[np.ndarray, jnp.ndarray]:
    """The axial cell centres and the area-mean pressure over each cross-section."""
    areas = jnp.asarray(pipe.grid.face_areas(2)[:, :, 0])
    return np.asarray(pipe.grid.centers[2]), jnp.sum(pressure * areas[:, :, None], axis=(0, 1)) / jnp.sum(
        areas
    )
