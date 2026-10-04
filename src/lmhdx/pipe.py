"""Fully developed flow along a circular pipe in a transverse magnetic field.

A pipe is the other half of the duct validation, and the geometry the ALEX B1
experiment used. Nothing about the physics changes -- the same inductionless
system, the same insulating or thin conducting wall -- but the coordinates do,
and with them what separates and what does not.

Fully developed means the velocity is axial and depends only on the cross
section, :math:`\\mathbf u = u(r,\\theta)\\hat z`. With :math:`\\mathbf B = B\\hat x`
the Lorentz force is then purely axial as well, so the transverse momentum
equations are satisfied by rest and the whole problem is one scalar equation
coupled to the potential:

.. math::
   \\nabla^2 u + B\\,\\partial_y\\varphi - B^2 u + f = 0,
   \\qquad \\nabla\\cdot\\mathbf J = 0,

with :math:`\\mathbf J = -\\nabla\\varphi + \\mathbf u\\times\\mathbf B`. Both
unknowns are cell centred, which is why this needs none of the staggered vector
machinery of :mod:`lmhdx.core3d`: there is no cross flow to project.

The electromotive force is where the coordinates show. A uniform Cartesian field
is not uniform in polar components, :math:`B_r = B\\cos\\theta` and
:math:`B_\\theta = -B\\sin\\theta`, so :math:`\\mathbf u\\times\\mathbf B` has
:math:`B u\\sin\\theta` through a radial face and :math:`B u\\cos\\theta` through
an azimuthal one. The same face currents then carry the potential equation and
the axial force, which is the consistency the whole package is built on.

The system is linear in the velocity, so it is one preconditioned Krylov solve,
not a Newton iteration. The preconditioner is the damped operator
:math:`(B^2 - \\nabla^2)^{-1}`, factorized exactly by
:func:`lmhdx.poisson.fast_diagonal_polar_poisson`, which is what keeps the
iteration count from growing with the Hartmann number.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import solvax

from .em import thin_wall_current, wall_insulated
from .grid import (
    CENTER,
    DIRICHLET,
    FACE,
    NEUMANN,
    PERIODIC,
    POLAR,
    BoundaryCondition,
    Field,
    Grid,
    uniform_faces,
    wall_resolving_faces,
)
from .ops import divergence, face_average, face_average_adjoint, face_gradient
from .poisson import fast_diagonal_polar_poisson

__all__ = ["PipeProblem", "pipe_grid", "pipe_problem", "solve_pipe"]

_WALL = BoundaryCondition(DIRICHLET)
_INSULATING = BoundaryCondition(NEUMANN)
_WRAP = BoundaryCondition(PERIODIC)


@dataclass(frozen=True)
class PipeProblem:
    """A circular pipe of unit radius in a transverse field of magnitude ``hartmann``."""

    grid: Grid
    hartmann: float
    wall_conductance: float = 0.0
    forcing: float = 1.0

    def __post_init__(self) -> None:
        if not self.grid.is_polar:
            raise ValueError("a pipe needs a polar grid")
        if float(self.hartmann) < 0.0:
            raise ValueError("hartmann must not be negative")
        if float(self.wall_conductance) < 0.0:
            raise ValueError("wall conductance must not be negative")

    @property
    def conditions(self) -> tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]:
        """Conditions on the potential: insulating at the wall, periodic elsewhere."""
        return (_INSULATING, _WRAP, _WRAP)

    def factorization(self):
        """Factorize the potential Laplacian, with the thin wall's sheet node when the wall conducts."""
        return fast_diagonal_polar_poisson(self.grid, self.conditions, wall_conductance=self.wall_conductance)

    def preconditioner(self):
        """Factorize the damped operator that preconditions the velocity solve."""
        return fast_diagonal_polar_poisson(
            self.grid,
            (_WALL, _WRAP, _WRAP),
            shift=max(float(self.hartmann) ** 2, 1.0),
            coefficient=1.0,
        )


def pipe_grid(radial: int, azimuthal: int, hartmann: float, *, cells_in_layer: int = 6) -> Grid:
    """Return a polar grid whose radial cells resolve the ``1/Ha`` wall layer."""
    if hartmann > 1.0:
        faces = wall_resolving_faces(
            2 * radial,
            -1.0,
            1.0,
            layer_thickness=1.0 / hartmann,
            cells_in_layer=cells_in_layer,
            max_ratio=None,
        )
        radial_faces = np.concatenate(([0.0], faces[radial + 1 :]))
    else:
        radial_faces = uniform_faces(radial, 0.0, 1.0)
    return Grid(
        radial_faces,
        uniform_faces(azimuthal, 0.0, 2.0 * np.pi),
        uniform_faces(1, 0.0, 1.0),
        geometry=POLAR,
    )


def pipe_problem(
    *, hartmann: float, radial: int = 48, azimuthal: int = 64, wall_conductance: float = 0.0
) -> PipeProblem:
    """Build a pipe whose mesh resolves the layer its Hartmann number implies."""
    return PipeProblem(pipe_grid(radial, azimuthal, hartmann), float(hartmann), float(wall_conductance))


def _angles(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Return ``sin`` and ``cos`` of the azimuth at the cell centres."""
    angle = np.asarray(grid.centers[1])
    return np.sin(angle)[None, :, None], np.cos(angle)[None, :, None]


def _face_emf(velocity: Field, problem: PipeProblem) -> tuple[Field, Field, Field]:
    """Return ``(u x B).n`` on the radial, azimuthal and axial faces.

    A uniform Cartesian field has ``B_r = B cos(theta)`` and
    ``B_theta = -B sin(theta)``, so an axial velocity drives ``B u sin(theta)``
    through a radial face and ``B u cos(theta)`` through an azimuthal one. The
    motional field is rotated at the cell centre, where the velocity lives, and
    carried to the faces by :func:`lmhdx.ops.face_average`; :func:`_axial_force`
    takes the transpose path back, so the force does exactly minus the work the
    currents dissipate, as in :mod:`lmhdx.em`.
    """
    grid = problem.grid
    sine, cosine = (jnp.asarray(value, dtype=velocity.dtype) for value in _angles(grid))
    motional = float(problem.hartmann) * velocity.data
    axial = Field(jnp.zeros(grid.face_shape(2), dtype=velocity.dtype), (CENTER, CENTER, FACE), grid)
    return (
        face_average(velocity.replace_data(motional * sine), 0, _WALL),
        face_average(velocity.replace_data(motional * cosine), 1, _WRAP),
        axial,
    )


def _face_currents(
    velocity: Field, potential: Field, problem: PipeProblem, wall: Field | None = None
) -> tuple[Field, Field, Field]:
    """Return the face-normal currents, closed at the wall by its own model.

    ``wall`` is the potential of a thin conducting wall on the radial faces, as
    :func:`_potential` returns it; the current into the wall is Ohm's law across
    the half cell against it (:func:`lmhdx.em.thin_wall_current`).
    """
    conditions = problem.conditions
    emf = _face_emf(velocity, problem)
    currents = []
    for axis in range(3):
        gradient = face_gradient(potential, axis, conditions[axis])
        ohmic = wall_insulated(gradient.replace_data(emf[axis].data - gradient.data), axis, conditions[axis])
        if axis == 0 and wall is not None:
            unit = gradient.replace_data(jnp.ones_like(gradient.data))
            ohmic = ohmic.replace_data(ohmic.data + thin_wall_current(potential, wall, unit, 0).data)
        currents.append(ohmic)
    return tuple(currents)


def _axial_force(currents: tuple[Field, Field, Field], problem: PipeProblem) -> jnp.ndarray:
    """Return ``(J x B)_z`` at cell centres: the transpose of :func:`_face_emf`, from the same currents.

    The half-cell current into a thin wall carries no electromotive force, so it exerts no force,
    as in :func:`lmhdx.core3d.electric_state`: with it the Lorentz work missed the dissipation by
    1.2e-4 (c 0.1, Ha 20); the charge balance keeps it.
    """
    sine, cosine = (jnp.asarray(value, dtype=currents[0].dtype) for value in _angles(problem.grid))
    closed = wall_insulated(currents[0], 0, problem.conditions[0])
    radial = face_average_adjoint(closed, 0, _WALL).data
    azimuthal = face_average_adjoint(currents[1], 1, _WRAP).data
    return -float(problem.hartmann) * (radial * sine + azimuthal * cosine)


def _potential(velocity: Field, problem: PipeProblem, factorization) -> tuple[Field, Field | None]:
    """Solve the charge balance directly for the induced potential and the wall's (``None`` if insulating)."""
    conditions = problem.conditions
    motional = tuple(
        wall_insulated(component, axis, conditions[axis])
        for axis, component in enumerate(_face_emf(velocity, problem))
    )
    source = divergence(motional)
    if not problem.wall_conductance:
        return factorization.solve(source), None
    return factorization.solve_with_wall(source)


def pipe_residual(velocity: Field, problem: PipeProblem, factorization) -> Field:
    """Return the steady axial momentum residual of a candidate velocity."""
    from .ops import laplacian

    potential, wall = _potential(velocity, problem, factorization)
    currents = _face_currents(velocity, potential, problem, wall)
    viscous = laplacian(velocity, (_WALL, _WRAP, _WRAP))
    return velocity.replace_data(viscous.data + _axial_force(currents, problem) + float(problem.forcing))


def solve_pipe(problem: PipeProblem, *, tolerance: float = 1.0e-11, max_restarts: int = 40):
    """Return the axial velocity and the potential of a fully developed pipe.

    The system is linear in the velocity, so this is one preconditioned Krylov
    solve. The preconditioner inverts the damped operator exactly, which is the
    stiff part of the problem and the reason the iteration does not lengthen
    with the Hartmann number.
    """
    factorization = problem.factorization()
    damped = problem.preconditioner()
    start = Field(jnp.zeros(problem.grid.shape), (CENTER, CENTER, CENTER), problem.grid)
    drive = pipe_residual(start, problem, factorization)

    def operator(velocity: Field) -> Field:
        residual = pipe_residual(velocity, problem, factorization)
        return residual.replace_data(drive.data - residual.data)

    def solve(matvec, target):
        return solvax.gmres(
            matvec, target, precond=damped.solve, rtol=tolerance, restart=200, max_restarts=max_restarts
        ).x

    velocity = jax.lax.custom_linear_solve(operator, drive, solve, solve)
    residual = pipe_residual(velocity, problem, factorization)
    scale = float(jnp.max(jnp.abs(drive.data)))
    remaining = float(jnp.max(jnp.abs(residual.data)))
    if not np.isfinite(remaining) or remaining > 1.0e-6 * max(scale, 1.0):
        raise RuntimeError(
            f"the pipe solve did not converge: residual {remaining:.3e} against a drive of {scale:.3e}; "
            "raise max_restarts or resolve the wall layer"
        )
    return velocity, _potential(velocity, problem, factorization)[0]


def flow_rate(velocity: Field) -> float:
    """Return the mean axial velocity over the cross-section."""
    volumes = np.asarray(velocity.grid.cell_volumes())
    return float((np.asarray(velocity.data) * volumes).sum() / volumes.sum())
