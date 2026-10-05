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
