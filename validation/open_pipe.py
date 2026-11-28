"""The open pipe: an inlet and an outlet on the polar grid, the evidence of validation row 6 (the ALEX pipe).

A pipe section ``(r, theta, x)`` whose axial axis is open, solved in the Stokes limit with the polar
discretization and eigen preconditioner of its own. It is evidence for the row and not part of the package:
only ``tests/test_pipe.py`` calls it.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import solvax

from lmhdx.axial import axial_faces
from lmhdx.grid import DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition, Field, Grid
from lmhdx.steady import _certified

_INSULATING = BoundaryCondition(NEUMANN)


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
    :func:`lmhdx.pipe.pipe_grid` does. The inlet profile is LMhdX's fully developed pipe
    (:func:`lmhdx.pipe.solve_pipe`) at the inlet field, scaled to ``flow_rate`` (a
    unit mean velocity by default).
    """
    from lmhdx.pipe import PipeProblem, pipe_grid, solve_pipe

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
    from lmhdx.poisson import fast_diagonal_polar_poisson

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


def wall_pressure_gradient(grid: Grid, pressure, angle: float | None = None, hartmann: float = 1.0):
    """Return the axial stations and ``-dp/dx / Ha**2`` of the wall pressure of an open pipe.

    ``pressure`` is the cell pressure ``(r, theta, x)`` of :func:`lmhdx.axial.solve_open_pipe`; the wall
    pressure is that of the outermost cell at the azimuth nearest ``angle`` (radians from the field
    direction; ``None`` averages the ring). Behind a fringe the pressure is not uniform across the
    section, so a wall tap and the area mean differ: the tap on the wall parallel to the field
    (``angle = pi/2``) reproduces ALEX B1, the mean does not (docs/validation/open_ducts.md, row 6).
    """
    ring = np.asarray(pressure)[-1]
    if angle is None:
        wall = ring.mean(axis=0)
    else:
        offset = np.angle(np.exp(1j * (np.asarray(grid.centers[1]) - angle)))
        wall = ring[int(np.argmin(np.abs(offset)))]
    x = np.asarray(grid.centers[2])
    return 0.5 * (x[1:] + x[:-1]), -np.diff(wall) / np.diff(x) / float(hartmann) ** 2
