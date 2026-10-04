"""Steady flow with implicit derivatives, in memory independent of the iteration count.

Potential and pressure are eliminated through their discrete linear solves,
leaving a residual in velocity alone:

.. math:: \\mathbf R(\\mathbf u)
   = \\mathbb P\\left[\\nu\\nabla^2\\mathbf u
     + (\\mathbf F(\\mathbf u) + \\mathbf f)/\\rho
     - \\nabla\\cdot(\\mathbf u\\mathbf u)\\right],

with :math:`\\mathbb P` the discrete projection onto the divergence-free face
fields and :math:`\\mathbf F` the Ni face-form Lorentz force of the potential
that :math:`\\mathbf u` induces. A root of :math:`\\mathbf R` is a steady state
of :func:`lmhdx.core3d.step`, and the projection keeps the iteration inside the
subspace the time stepper never leaves.

Without advection the residual is affine,
:math:`\\mathbf R(\\mathbf u) = A\\mathbf u + \\mathbf b`, and :math:`-A` is
symmetric positive definite on the constrained divergence-free fields in the
face-volume inner product, because the electromotive and force interpolations
are discrete adjoints and the thin-wall closure is a symmetric direct solve.
That case is one preconditioned conjugate-gradient solve, differentiated by
one more. Advection takes matrix-free Newton-Krylov with restarted GMRES, and
the implicit function theorem differentiates its root with tangent and
transpose solves. Neither keeps more than a restart cycle of vectors.

The preconditioner is a projected per-component inverse. A component along a
periodic axis, across a uniform axis-aligned field, is solved exactly along each
field line in the discrete induction form of the insulating duct, and
approximately across the lines. The other components, and all three in a
varying field, take a viscous inverse damped at :math:`\\sigma|B|^2/\\rho` with
the peak :math:`|B|^2` over the cells. On ``duct_problem`` meshes of 48 cells, CG needs
4 / 15 / 44 / 79 iterations at Ha 20 / 100 / 300 / 1000, against 33 / 148 /
403 / 804 with the damped inverse alone, and 68 against 714 on 64 cells at Ha 1000.

Primal residuals and tangent/transpose convergence are certified. Rejection
raises eagerly; during tracing, it produces nonfinite fields and derivatives
without host callbacks. Optimizers must reject nonfinite values and gradients.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import solvax

from . import _programs
from ._programs import attribute, bound, host_array, host_scalar, shape_program
from .advect import momentum_advection
from .core3d import (
    ChannelProblem,
    ImposedField,
    _factorization,
    electric_state,
    enforce_face_constraints,
    face_currents,
    face_lorentz_force,
    project,
    velocity_condition,
    velocity_offset,
    zero_velocity,
)
from .grid import CENTER, FACE, Field
from .ops import face_inner_product, staggered_laplacian
from .poisson import (
    FastDiagonalHelmholtz,
    FastDiagonalPoisson,
    assemble_staggered_axis_operator,
    fast_diagonal_helmholtz,
)

__all__ = ["SteadySolution", "solve_compiled", "solve_steady_state", "steady_residual"]

_MAX_STEPS = 40

# Field lines apply dense velocity-block inverses (lines * n^2 * 8 bytes) up to this size on these
# backends, and the banded solve otherwise: measured faster there, and no slower to compile (2b.11).
_LINE_INVERSE_BYTES = 64 * 2**20
_LINE_INVERSE_BACKENDS = ("gpu", "cuda", "rocm")
# Start derivative solves from the primal solution scaled onto their right-hand side (2b.2).
_REUSE_PRIMAL = True


@dataclass(frozen=True)
class SteadySolution:
    """Steady fields and residual evidence; traced rejection gives nonfinite fields."""

    velocity: tuple[Field, Field, Field]
    pressure: Field
    potential: Field
    residual_norm: jnp.ndarray
    steps: int
    initial_residual_norm: jnp.ndarray | None = None
    currents: tuple[Field, Field, Field] | None = None
    magnetic_field: tuple[Field, Field, Field] | None = None


def steady_residual(
    velocity: tuple[Field, Field, Field],
    problem: ChannelProblem,
    factorization: FastDiagonalPoisson | None = None,
    *,
    forcing: tuple[float, float, float] | None = None,
    field_scale: float | jnp.ndarray = 1.0,
    inflow: float | jnp.ndarray | None = None,
) -> tuple[Field, Field, Field]:
    """Return the projected steady momentum residual of a velocity field.

    Zero exactly when ``velocity`` is a fixed point of :func:`lmhdx.core3d.step`.
    ``forcing`` and ``field_scale`` are the continuous design inputs and may be
    traced; everything ``problem`` carries is static, because the factorizations
    read concrete arrays.

    The projection is applied twice, which is one defect correction of its pressure
    solve. One pass leaves about 1e-13 of what the pressure removes outside the
    divergence-free fields, where the CG preconditioner cannot see it. The gradient part of a
    varying field's force is large, so that leak set the CG floor: 5e-10 of the right-hand
    side at Ha 20 and 6e-7 at Ha 100, against 1e-13 and 3e-12 with the correction (#145).

    ``inflow`` scales the inlet profile of an inflow-outflow axis onto the inlet
    face, which the constraints otherwise hold at zero; left ``None`` the residual
    is linear in ``velocity``, which is what the Krylov solve needs.
    """
    factorization = problem.factorization() if factorization is None else factorization
    terms = momentum_terms(
        velocity, problem, factorization, forcing=forcing, field_scale=field_scale, inflow=inflow
    )
    corrected, _ = project(project(terms, problem, factorization)[0], problem, factorization)
    return corrected


def momentum_terms(
    velocity: tuple[Field, Field, Field],
    problem: ChannelProblem,
    factorization: FastDiagonalPoisson,
    *,
    forcing: tuple[float, float, float] | None = None,
    field_scale: float | jnp.ndarray = 1.0,
    inflow: float | jnp.ndarray | None = None,
) -> tuple[Field, Field, Field]:
    """Return the steady momentum terms before projection: what the pressure gradient balances."""
    velocity = enforce_face_constraints(velocity, problem)
    if inflow is not None:
        velocity = with_inflow(velocity, problem, inflow)
    drive = _drive(problem, forcing)
    _, force = electric_state(velocity, problem, factorization, field_scale)
    body = face_lorentz_force(force, problem)
    conditions = tuple(velocity_condition(problem.conditions, axis) for axis in range(3))
    transport = (
        None
        if problem.advection == "off"
        else momentum_advection(velocity, conditions, limited=problem.advection == "limited")
    )
    viscosity, density = attribute(problem, "viscosity"), attribute(problem, "density")
    terms = []
    for component, field in enumerate(velocity):
        # The damping rate is a preconditioning device, not a term: the conservative
        # face force already carries the whole Lorentz contribution, so subtracting
        # it here as well would count the same physics twice.
        value = (
            viscosity * staggered_laplacian(field, conditions).data
            + (body[component].data + drive[component]) / density
        )
        if transport is not None:
            value = value - transport[component].data
        terms.append(field.replace_data(value))
    return tuple(terms)


def with_inflow(
    velocity: tuple[Field, Field, Field], problem: ChannelProblem, scale: float | jnp.ndarray = 1.0
) -> tuple[Field, Field, Field]:
    """Set the inlet face of an inflow-outflow axis to ``scale`` times its prescribed profile."""
    axis = problem.open_axis
    if axis is None:
        raise ValueError("the problem has no inflow-outflow axis")
    field = velocity[axis]
    profile = host_array(problem, _inlet_profile, axis, dtype=field.dtype)
    data = field.data.at[(slice(None),) * axis + (0,)].set(scale * profile)
    return tuple(field.replace_data(data) if index == axis else v for index, v in enumerate(velocity))


def _inlet_profile(problem: ChannelProblem, axis: int) -> np.ndarray:
    return problem.conditions[axis].lower


def _forcing(problem: ChannelProblem, component: int) -> float:
    return problem.forcing[component]


def _drive(problem: ChannelProblem, forcing) -> tuple:
    """The drive: ``forcing`` as given (fixed by the program), else the problem's, read as its data."""
    if forcing is not None:
        return forcing
    return tuple(host_scalar(problem, _forcing, component) for component in range(3))


def _rest_residual(problem: ChannelProblem, factorization: FastDiagonalPoisson, forcing):
    """:func:`steady_residual` of the fluid at rest: the projected drive, without the operator.

    Every other term is an exact zero there (viscous stress, electromotive force, the potential it
    drives, advection), so this is the same value as the full residual, at a fifth of the program.
    """
    drive, density = _drive(problem, forcing), attribute(problem, "density")
    terms = tuple(
        field.replace_data(jnp.zeros_like(field.data) + drive[component] / density)
        for component, field in enumerate(zero_velocity(problem))
    )
    corrected, _ = project(project(terms, problem, factorization)[0], problem, factorization)
    return corrected


def _preconditioner(
    problem: ChannelProblem,
    factorization: FastDiagonalPoisson,
    viscous: tuple,
    pseudo_step: float,
):
    """One projection step: the conjugate-gradient and the Newton-GMRES preconditioner.

    ``viscous`` holds one solve per component, from :func:`_projection_solves`.
    It ends in :func:`lmhdx.core3d.project`, whose copy of the first periodic face
    completes a solve that returns that duplicate face as zero.

    ``viscous`` has to be factorized at ``pseudo_step`` and not at the problem's
    own step: a preconditioner built at the wrong step is a different operator,
    which is what leaves the Krylov iteration stalling as soon as the damping is
    stiff.
    """

    def apply(direction: tuple[Field, Field, Field]) -> tuple[Field, Field, Field]:
        scaled = tuple(
            viscous[component].solve(field.replace_data(pseudo_step * field.data))
            for component, field in enumerate(direction)
        )
        corrected, _ = project(scaled, problem, factorization)
        return corrected

    return apply


def _isotropic_viscous(
    problem: ChannelProblem, pseudo_step: float, rate: float | None = None
) -> tuple[FastDiagonalHelmholtz, ...]:
    """Viscous factorizations damping every component at one ``rate``, for both routes.

    The rate defaults to the peak ``sigma |B|^2 / rho``. Damping only the components
    normal to the field, as the time step does, leaves a Schur-complement deficit
    once projected: on ``duct_problem(hartmann=300, cells=40)`` the preconditioned
    spectrum spans ``[2.2e-3, 2.0e3]`` and CG takes 3638 iterations; with one shift
    it spans ``[2.2e-3, 8.5]`` and CG takes 399.
    """
    if rate is None:
        rate = float(problem.conductivity) * problem.peak_field_squared / float(problem.density)
    conditions = tuple(velocity_condition(problem.conditions, axis) for axis in range(3))
    return tuple(
        fast_diagonal_helmholtz(
            problem.grid,
            velocity_offset(component),
            conditions,
            shift=1.0 + pseudo_step * rate,
            coefficient=pseudo_step * float(problem.viscosity),
            precision=problem.precision,
        )
        for component in range(3)
    )


def _varying_field_rate(problem: ChannelProblem) -> float:
    """The varying-field damping: ``(nu lambda_1)^(3/4) (sigma |B|^2 / rho)^(1/4)`` (plan 2b.4).

    The peak Joule rate over-damps what the Lorentz force does not brake (a weak or
    absent field, axial odd-even modes its face averages cancel), which left the
    continuum 2b.0 found; damping that varies in space or by component reopens the
    Schur-complement deficit instead. One rate between the slowest viscous rate
    ``nu lambda_1`` and the Hartmann braking rate, their geometric mean, sits in
    the measured optimum: ANL fringe iterations flat within 10 % from 3e-4 to 3e-2
    of the peak at Ha 100 and from 2e-6 to 1e-3 at Ha 1600.
    """
    joule = float(problem.conductivity) * problem.peak_field_squared / float(problem.density)
    conditions = tuple(velocity_condition(problem.conditions, axis) for axis in range(3))
    viscous = fast_diagonal_helmholtz(
        problem.grid, velocity_offset(0), conditions, coefficient=float(problem.viscosity)
    )
    slowest = float(problem.viscosity) * sum(float(np.min(np.abs(values))) for values in viscous.values)
    return min(joule, slowest**0.75 * joule**0.25)


def _factor_lines(bands: np.ndarray) -> np.ndarray:
    """Factorize pentadiagonal lines on the host, in the storage of :func:`solvax.lu_solve_banded`.

    Doolittle without pivoting: weighted by the widths, each block has a positive definite
    symmetric part, so no pivot vanishes. On the host it compiles nothing and runs under
    :func:`jax.ensure_compile_time_eval`, where JAX 0.6.2 cannot evaluate a scan.
    """
    lu, size = np.array(bands, dtype=float), bands.shape[-1]
    for j in range(size - 1):
        for i in range(1, min(3, size - j)):
            lu[:, 2 + i, j] /= lu[:, 2, j]
            for k in range(1, min(3, size - j)):
                lu[:, 2 + i - k, j + k] -= lu[:, 2 + i, j] * lu[:, 2 - k, j + k]
    return lu


def _velocity_inverses(lu: np.ndarray, chunk: int = 512) -> np.ndarray:
    """The velocity-to-velocity block of each factorized line's inverse, ``(lines, n, n)``, on the host.

    The substitutions of :func:`solvax.lu_solve_banded` run once here on the unit
    velocity columns, so the device applies one batched product instead of 2n scan steps.
    """
    size = lu.shape[-1]
    blocks = []
    for start in range(0, lu.shape[0], chunk):
        part = lu[start : start + chunk]
        x = np.zeros((part.shape[0], size, (size + 1) // 2))
        x[:, ::2] = np.eye((size + 1) // 2)
        for j in range(size):
            for i in range(1, min(3, size - j)):
                x[:, j + i] -= part[:, 2 + i, j, None] * x[:, j]
        for j in range(size - 1, -1, -1):
            for k in range(1, min(3, size - j)):
                x[:, j] -= part[:, 2 - k, j + k, None] * x[:, j + k]
            x[:, j] /= part[:, 2, j, None]
        blocks.append(x[:, ::2])
    return np.concatenate(blocks)


class _FieldLine:
    """The projection-step solve of a velocity component across an axis-aligned field.

    With insulating walls and a flow invariant along the component, the
    divergence-free face currents are the discrete curl of a streamfunction on the
    cell edges, so the component's Lorentz operator is ``lambda K* L^-1 K``: ``K``
    is the compact gradient along the field after the average onto the
    electromotive faces, ``L`` the edge Laplacian. Its inverse is the velocity
    block of ``[[S, tau c K*], [-tau c K, S_e]]``, ``S = 1 - tau nu Lap`` at both
    positions and ``c^2 = lambda nu``: Shercliff's induction form. Along the field
    the block is pentadiagonal with velocity and streamfunction interleaved, and
    each line is factorized once. Across it the velocity eigenbasis carries the
    block, each mode's average represented by its norm and the edge Laplacian by
    its Rayleigh quotient, which is exact on uniform or periodic axes. A wide
    collocated difference in place of ``K`` misses the checkerboard modes, and its
    condition number grows with the Hartmann number instead.
    """

    def __init__(self, problem: ChannelProblem, component: int, axis: int, pseudo_step: float):
        grid, coefficient = problem.grid, pseudo_step * float(problem.viscosity)
        conditions = tuple(velocity_condition(problem.conditions, position) for position in range(3))
        offset = velocity_offset(component)
        edge = tuple(CENTER if position == component else FACE for position in range(3))
        velocity = fast_diagonal_helmholtz(grid, offset, conditions, coefficient=coefficient)
        stream = fast_diagonal_helmholtz(grid, edge, conditions, coefficient=coefficient)
        self.velocity, self.axis = velocity, axis
        self.across = [position for position in range(3) if position != axis]
        weights, laplacians = [], []
        for at in self.across:
            average = _average(np.asarray(grid.widths[at]), offset[at], conditions[at])
            unscaled = velocity.vectors[at] / velocity.scales[at][:, None]
            modes = stream.vectors[at].T @ (stream.scales[at][:, None] * (average @ unscaled))
            weights.append(np.sum(modes**2, axis=0))
            laplacians.append(stream.values[at] @ modes**2 / np.maximum(weights[-1], np.finfo(float).tiny))

        def pair(first, second):
            return (first[:, None] + second[None, :]).reshape(-1)

        rate = float(problem.conductivity) * problem.peak_field_squared
        speed = pseudo_step * np.sqrt(rate * float(problem.viscosity) / float(problem.density))
        shift = 1.0 - coefficient * pair(*(velocity.values[at] for at in self.across))
        coefficients = [np.ones_like(shift), shift, 1.0 - coefficient * pair(*laplacians)]
        coefficients.append(speed * np.sqrt(np.outer(*weights).reshape(-1)))
        widths = np.asarray(grid.widths[axis])
        size, distances = 2 * widths.size - 1, 0.5 * (widths[:-1] + widths[1:])
        gradient = np.diff(np.eye(widths.size), axis=0) / distances[:, None]
        u, e = np.arange(0, size, 2), np.arange(1, size, 2)
        blocks = np.zeros((4, size, size))
        for index, at in ((u, offset), (e, edge)):
            operator = assemble_staggered_axis_operator(grid, axis, at, conditions[axis])
            blocks[0][np.ix_(index, index)] = -coefficient * operator
        blocks[1][u, u], blocks[2][e, e] = 1.0, 1.0
        blocks[3][np.ix_(u, e)] = gradient.T * distances[None, :] / widths[:, None]
        blocks[3][np.ix_(e, u)] = -gradient
        # solvax band storage: bands[r, j] = A[j + r - 2, j].
        rows = np.arange(5)[:, None] + np.arange(size)[None, :] - 2
        inside = (rows >= 0) & (rows < size)
        bands = np.where(inside, blocks[:, np.clip(rows, 0, size - 1), np.arange(size)], 0.0)
        lu = _factor_lines(np.einsum("km,krj->mrj", np.stack(coefficients), bands))
        dense = lu.shape[0] * widths.size**2 * 8 <= _LINE_INVERSE_BYTES
        # Host arrays, read in solve through lmhdx._programs.host_array.
        self.dense = dense and jax.default_backend() in _LINE_INVERSE_BACKENDS
        self.inverses = _velocity_inverses(lu) if self.dense else None
        self.lu = None if self.dense else lu

    def solve(self, rhs: Field) -> Field:
        velocity = self.velocity
        data = rhs.data[velocity.slices]
        for at in self.across:
            scale = host_array(self, _line_scale, at, False, dtype=data.dtype)
            data = _modal(data * scale, host_array(self, _line_basis, at, True, dtype=data.dtype), at)
        lines = jnp.moveaxis(data, self.axis, -1)
        flat = lines.reshape(-1, lines.shape[-1])
        if self.dense:
            inverses = host_array(self, _line_array, "inverses")
            flat = jnp.einsum("lij,lj->li", inverses, flat, precision=jax.lax.Precision.HIGHEST)
        else:
            count = self.lu.shape[0]
            factors = solvax.BandedLUFactors(
                host_array(self, _line_array, "lower"),
                host_array(self, _line_array, "upper"),
                jnp.ones(self.lu.shape[::2]),
                jnp.zeros(count, jnp.int32),
            )
            interleaved = jnp.zeros((flat.shape[0], 2 * flat.shape[1] - 1), flat.dtype).at[:, ::2].set(flat)
            flat = jax.vmap(solvax.lu_solve_banded)(factors, interleaved)[:, ::2]
        data = jnp.moveaxis(flat.reshape(lines.shape), -1, self.axis)
        for at in self.across:
            restored = _modal(data, host_array(self, _line_basis, at, False, dtype=data.dtype), at)
            data = restored * host_array(self, _line_scale, at, True, dtype=data.dtype)
        return rhs.replace_data(jnp.zeros_like(rhs.data).at[velocity.slices].set(data))


def _line_array(line: _FieldLine, name: str) -> np.ndarray:
    if name == "inverses":
        return line.inverses
    return line.lu[:, 3:] if name == "lower" else line.lu[:, :3]


def _line_scale(line: _FieldLine, axis: int, inverse: bool) -> np.ndarray:
    scale = line.velocity.scales[axis]
    return (1.0 / scale if inverse else scale).reshape(
        [-1 if position == axis else 1 for position in range(3)]
    )


def _line_basis(line: _FieldLine, axis: int, transpose: bool) -> np.ndarray:
    vectors = line.velocity.vectors[axis]
    return vectors.T if transpose else vectors


def _modal(data: jnp.ndarray, matrix: jnp.ndarray, axis: int) -> jnp.ndarray:
    return jnp.moveaxis(jnp.tensordot(matrix, data, axes=([1], [axis])), 0, axis)


def _average(widths: np.ndarray, position: float, condition) -> np.ndarray:
    """The electromotive average between free entries: centres to faces, or its half-weight transpose."""
    count = widths.size
    if condition.is_periodic:
        faces, left, right = np.arange(count), (np.arange(count) - 1) % count, np.arange(count)
    else:
        faces, left, right = np.arange(count - 1), np.arange(count - 1), np.arange(1, count)
    share = widths[left] / (widths[left] + widths[right])
    matrix = np.zeros((faces.size, count) if position == CENTER else (count, faces.size))
    for cells, weight in ((left, share), (right, 1.0 - share)):
        if position == CENTER:
            np.add.at(matrix, (faces, cells), weight)
        else:
            np.add.at(matrix, (cells, faces), 0.5)
    return matrix


def _projection_solves(problem: ChannelProblem, pseudo_step: float) -> tuple:
    """Field lines for a periodic-axis component across an axis-aligned walled field; damped otherwise.

    The induction form is exact for a component along an axis the flow is
    invariant on, which only a periodic axis can be. The other components keep one
    common damping, since the projection couples them and different shifts
    reopen the Schur-complement deficit. Field lines on those as well gave CG 85
    iterations instead of 44 on ``duct_problem(hartmann=300, cells=48)``, and
    1408 instead of 540 on the Ha 1000 test mesh.
    """
    if isinstance(problem.magnetic_field, ImposedField):
        return _isotropic_viscous(problem, pseudo_step, _varying_field_rate(problem))
    solves = list(_isotropic_viscous(problem, pseudo_step))
    field = np.asarray(problem.magnetic_field, dtype=float) * float(problem.conductivity)
    axis = int(np.argmax(np.abs(field)))
    if np.count_nonzero(field) == 1 and not problem.conditions[axis].is_periodic:
        for component in range(3):
            if component != axis and problem.conditions[component].is_periodic:
                solves[component] = _FieldLine(problem, component, axis, pseudo_step)
    return tuple(solves)


def _pseudo_step(problem: ChannelProblem, pseudo_step: float | None) -> float:
    return float(problem.dt if pseudo_step is None else pseudo_step)


def _projection_solves_at(problem: ChannelProblem, pseudo_step: float | None) -> tuple:
    return _projection_solves(problem, _pseudo_step(problem, pseudo_step))


def _face_weights(problem: ChannelProblem) -> tuple[Field, Field, Field]:
    """Return the diagonal of the face-volume inner product as a velocity."""
    conditions = tuple(velocity_condition(problem.conditions, axis) for axis in range(3))
    ones = jax.tree.map(jnp.ones_like, zero_velocity(problem))
    return jax.grad(
        lambda u: 0.5 * sum(face_inner_product(f, f, axis, conditions[axis]) for axis, f in enumerate(u))
    )(ones)


def _orthogonal_projection(
    velocity: tuple[Field, Field, Field], problem: ChannelProblem, factorization: FastDiagonalPoisson
) -> tuple[Field, Field, Field]:
    """Project onto the constrained divergence-free fields, orthogonally in the face volume.

    :func:`lmhdx.core3d.project` copies the first periodic face onto its duplicate,
    an oblique projection; both copies carry half the weight, so averaging them
    is the orthogonal one.
    """
    constrained = []
    for component, field in enumerate(velocity):
        data = field.data
        selection = (slice(None),) * component
        if problem.conditions[component].is_periodic:
            mean = 0.5 * (data[selection + (0,)] + data[selection + (-1,)])
            data = data.at[selection + (0,)].set(mean).at[selection + (-1,)].set(mean)
        elif problem.conditions[component].is_mixed:
            data = data.at[selection + (0,)].set(0.0)
        else:
            data = data.at[selection + (0,)].set(0.0).at[selection + (-1,)].set(0.0)
        constrained.append(field.replace_data(data))
    return project(tuple(constrained), problem, factorization)[0]


def _stokes_limit_root(
    problem: ChannelProblem,
    start: tuple[Field, Field, Field],
    factorization: FastDiagonalPoisson,
    precond,
    *,
    forcing,
    field_scale,
    tolerance: float,
    max_iterations: int,
    rhs=None,
):
    """Solve the affine Stokes-limit problem with one preconditioned CG solve.

    ``R(u) = A u + b`` with ``-A`` symmetric positive definite on the constrained
    divergence-free fields ``V`` in the face-volume inner product ``W``. CG runs
    on ``y = W u`` with operator ``y -> -A W^-1 y`` and preconditioner
    ``r -> W P r``, so the residual it measures is ``R`` itself. The derivative
    is a symmetric :func:`jax.lax.custom_linear_solve`, whose operator must be
    symmetric on every vector because a cotangent is arbitrary:
    ``K y = -A Q W^-1 y + (I - Q) W^-1 y`` with ``Q`` the orthogonal projection
    onto ``V``. Its inverse projects once per solve, not per iteration.

    The operator being symmetric, the derivative solves start from the primal
    solution scaled onto their projected right-hand side (2b.2). A right-hand
    side parallel to the primal one -- the adjoint of any objective whose weight
    is the drive, such as the flow rate under a uniform drive, and the tangent
    in the drive -- is then solved on entry and CG takes no step; any other
    starts from its component along the primal. One code path, the same CG and
    tolerance. The primal solve sits outside the derivative rule, a
    :func:`jax.custom_jvp`, so an undifferentiated solve compiles one CG loop.
    The rule takes the face weights as arguments rather than closing over them:
    a traced closure would leak when an enclosing ``jit`` is differentiated.
    """
    weights = _face_weights(problem)

    def operator(scale, velocity):
        return steady_residual(velocity, problem, factorization, forcing=(0.0, 0.0, 0.0), field_scale=scale)

    def matvec(weights, scale, state):
        velocity = jax.tree.map(jnp.divide, state, weights)
        inside = _orthogonal_projection(velocity, problem, factorization)
        return jax.tree.map(lambda u, a, q: u - a - q, velocity, operator(scale, inside), inside)

    def staged_cg(weights, scale, inside, start):
        rest = jax.tree.map(jnp.zeros_like, inside) if start is None else start
        rest_leaves = jax.tree.leaves(rest)

        def matvec(y):
            # CG's first product is with a zero start, and the operator is linear: skip it rather
            # than put a whole operator application into the program. Anything else is applied.
            if start is None and all(a is b for a, b in zip(jax.tree.leaves(y), rest_leaves, strict=True)):
                return jax.tree.map(jnp.zeros_like, y)
            return jax.tree.map(jnp.negative, operator(scale, jax.tree.map(jnp.divide, y, weights)))

        result = solvax.pcg(
            matvec,
            inside,
            x0=rest,
            precond=lambda r: jax.tree.map(jnp.multiply, precond(r), weights),
            rtol=tolerance,
            max_steps=max_iterations,
        )
        accepted = result.converged & jnp.isfinite(result.residual_norm)
        kept = _certified(result.x, accepted, "steady CG solve")
        return kept, (result.iterations, result.residual_norm, result.converged)

    traced_cg = {}

    def cg(*arguments):
        """CG traced once per argument signature: the tangent and transposed solves share one trace.

        The trace is replayed with :func:`jax.core.eval_jaxpr` rather than staged as a
        ``jit``, so its constants stay constants of the enclosing program, which
        compiling once per program shape (2b.1) passes as arguments.
        """
        leaves, tree = jax.tree.flatten(arguments)
        key = (tree, tuple((jnp.shape(leaf), jnp.result_type(leaf)) for leaf in leaves))
        if key not in traced_cg:
            traced_cg[key] = jax.make_jaxpr(staged_cg, return_shape=True)(*arguments)
        closed, shapes = traced_cg[key]
        return jax.tree.unflatten(
            jax.tree.structure(shapes), jax.core.eval_jaxpr(closed.jaxpr, closed.consts, *leaves)
        )

    def complete(weights, y, target, inside):
        return jax.tree.map(lambda v, t, q, w: v + w * (t - q), y, target, inside, weights)

    @jax.custom_jvp
    def solved(target, scale, primal_y, primal_inside, weights):
        return complete(weights, primal_y, target, primal_inside)

    @solved.defjvp
    def solved_jvp(primals, tangents):
        target, scale, primal_y, primal_inside, weights = primals
        target_dot, scale_dot = tangents[:2]
        solution = solved(*primals)
        norm2 = _dot(primal_inside, primal_inside)

        def solve(_, value):
            inside = _orthogonal_projection(value, problem, factorization)
            start = None
            if _REUSE_PRIMAL:
                ratio = _dot(inside, primal_inside) / jnp.maximum(norm2, jnp.finfo(norm2.dtype).tiny)
                start = jax.tree.map(lambda v: ratio * v, primal_y)
            return complete(weights, cg(weights, scale, inside, start)[0], value, inside)

        change = jax.jvp(lambda s: matvec(weights, s, solution), (scale,), (scale_dot,))[1]
        rhs = jax.tree.map(jnp.subtract, target_dot, change)
        operator_at = functools.partial(matvec, weights, scale)
        return solution, jax.lax.custom_linear_solve(operator_at, rhs, solve, symmetric=True)

    if rhs is None:
        rhs = steady_residual(start, problem, factorization, forcing=forcing, field_scale=field_scale)
    primal_inside = jax.lax.stop_gradient(_orthogonal_projection(rhs, problem, factorization))
    primal_y, diagnostics = cg(weights, jax.lax.stop_gradient(field_scale), primal_inside, None)
    step = solved(rhs, field_scale, jax.lax.stop_gradient(primal_y), primal_inside, weights)
    return jax.tree.map(lambda u, y, w: u + y / w, start, step, weights), diagnostics


def solve_steady_state(
    problem: ChannelProblem,
    velocity: tuple[Field, Field, Field] | None = None,
    *,
    tolerance: float = 1.0e-9,
    max_steps: int = _MAX_STEPS,
    pseudo_step: float | None = None,
    forcing: tuple[float, float, float] | None = None,
    field_scale: float | jnp.ndarray = 1.0,
    linear_tolerance: float = 1.0e-6,
    linear_restart: int = 60,
    linear_max_restarts: int = 200,
) -> SteadySolution:
    """Find the steady state, differentiably, in memory independent of the iteration count.

    Without advection, with insulating or thin conducting walls, the problem is affine and
    symmetric, and one preconditioned conjugate-gradient solve answers it; its
    budget is ``linear_restart * linear_max_restarts`` iterations, and
    ``linear_tolerance`` does not apply. Otherwise matrix-free Newton-Krylov
    runs restarted GMRES. The drive and ``field_scale`` are differentiable
    through implicit linear solves; close over static ``problem`` and solver
    controls when using :func:`jax.jit`. Factorizations are assembled at trace
    time. Rejected roots raise eagerly or yield nonfinite fields and
    derivatives during tracing.
    """
    step = float(problem.dt if pseudo_step is None else pseudo_step)
    for name, value in (
        ("pseudo_step", step),
        ("tolerance", tolerance),
        ("linear_tolerance", linear_tolerance),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    factorization = bound(problem, _factorization)
    viscous = bound(problem, _projection_solves_at, pseudo_step)
    start = zero_velocity(problem) if velocity is None else enforce_face_constraints(velocity, problem)

    def residual(state):
        return steady_residual(state, problem, factorization, forcing=forcing, field_scale=field_scale)

    # At rest the velocity terms vanish exactly, and so does the potential solve they feed.
    initial = residual(start) if velocity is not None else _rest_residual(problem, factorization, forcing)
    scale = _norm(initial)

    precond = _preconditioner(
        problem, factorization, viscous, host_scalar(problem, _pseudo_step, pseudo_step)
    )

    if problem.advection == "off":
        root, _ = _stokes_limit_root(
            problem,
            jax.lax.stop_gradient(_orthogonal_projection(start, problem, factorization)),
            factorization,
            precond,
            forcing=forcing,
            field_scale=field_scale,
            tolerance=tolerance,
            max_iterations=linear_restart * linear_max_restarts if max_steps > 0 else 0,
            # From rest the projected start is the start, so its residual is already known.
            rhs=initial if velocity is None else None,
        )
        return _finish(root, residual, scale, tolerance, problem, factorization, field_scale, max_steps)

    def solver(function, guess):
        solution = solvax.newton_krylov(
            function,
            guess,
            precond=precond,
            rtol=tolerance,
            max_steps=max_steps,
            linear_rtol=linear_tolerance,
            linear_restart=linear_restart,
            linear_max_restarts=linear_max_restarts,
        )
        return solution.x

    tangent_solve = functools.partial(_tangent_solve, precond=precond)
    root = solvax.root_solve(residual, start, solver, tangent_solve=tangent_solve)
    return _finish(root, residual, scale, tolerance, problem, factorization, field_scale, max_steps)


def solve_compiled(problem: ChannelProblem) -> SteadySolution:
    """Run :func:`solve_steady_state` at its defaults as one compiled program; raise if it fails.

    The program is cached per problem. Compiling the whole solve replaces the
    dispatch of each operation from the host, which is most of the time of an
    eager solve, cold or warm.
    """
    velocity, pressure, potential, residual = _program(problem)()
    # On the host: an eager check compiles a program per component, 0.3 s in a new process.
    if not all(np.isfinite(np.asarray(field.data)).all() for field in velocity):
        raise RuntimeError("the steady solve did not converge")
    return SteadySolution(velocity, pressure, potential, residual, _MAX_STEPS)


@functools.lru_cache(maxsize=16)
def _program(problem: ChannelProblem):
    return shared_or_embedded(problem, _solve_program)


def _solve_program(problem: ChannelProblem):
    def run():
        solution = solve_steady_state(problem, max_steps=_MAX_STEPS)
        return solution.velocity, solution.pressure, solution.potential, solution.residual_norm

    return run


# Per program shape: the keys of its problem arrays, then its shared program; None keeps #175's route.
_SHAPES: dict = {}
_EMBED_AFTER_CALLS = 400


def shape_key(problem: ChannelProblem) -> tuple:
    """What fixes the structure of a problem's program: shapes, conditions and flags, not values."""
    field = problem.magnetic_field
    if isinstance(field, ImposedField):
        pattern = (type(field).__name__, field.faces is None)
    else:
        pattern = tuple(bool(b) for b in field)
    return (
        problem.grid.shape,
        problem.grid.is_polar,
        problem.conditions,
        problem.advection,
        problem.precision,
        pattern,
        tuple(bool(c) for c in problem.wall_conductance),
        bool(problem.conductivity),
        jax.default_backend(),
    )


def _shareable(problem: ChannelProblem) -> bool:
    """The problems whose programs read every array through :mod:`lmhdx._programs` (2b.1 stage 5)."""
    return problem.advection == "off" and not problem.grid.is_polar and problem.open_axis is None


def shared_or_embedded(problem: ChannelProblem, build, *arguments):
    """Compile the first problem of a shape with its constants embedded, later ones shared (2b.1).

    ``build(problem)`` returns the function to compile, of ``arguments``
    (shapes and dtypes). Embedded constants compile faster and run up to a
    third faster warm on a CPU, because XLA folds them, so a single solve takes
    the first. Its trace notes the key of every host array the solve reads; the
    second problem of the shape traces the program once more with those arrays
    as arguments (:class:`lmhdx._programs.ShapeProgram`), and every later one
    builds its arrays on the host and runs that executable, with no trace.
    Programs that do not read every array that way trace each problem and share
    its lowered program (:func:`shape_program`). A shared program runs a warm
    solve slower on a CPU (XLA cannot fold arrays it receives as arguments), so a
    problem solved more than ``_EMBED_AFTER_CALLS`` times compiles its own
    embedded program, bounding that loss by about the compile it saved.
    """
    key = (build, shape_key(problem), tuple((a.shape, str(a.dtype)) for a in arguments))
    function = build(problem)

    def embedded():
        compiled = jax.jit(function).lower(*arguments).compile()
        return lambda *values: compiled(*values)

    if key not in _SHAPES:
        _SHAPES[key] = None
        if not _shareable(problem):
            return embedded()
        # A shape an earlier process solved: its keys, or its program, are on disk.
        _SHAPES[key] = _programs.stored(key)
        if _SHAPES[key] is None:
            with _programs.discovering(problem) as trace:
                lowered = jax.jit(function).lower(*arguments)
            if trace.complete:
                _SHAPES[key] = list(trace.keys)
                _programs.store(key, _SHAPES[key])
            compiled = lowered.compile()
            return lambda *values: compiled(*values)
    shared, calls, program = _shared(key, problem, build, arguments), [0], [None]
    if shared is None:
        shared = shape_program(function, *arguments)

    def run(*values):
        # A problem solved many times earns its own embedded program: its compile (4-5 s on a
        # 48-cell duct) costs what the shared one loses in about 400-600 warm solves (8-10 ms each).
        calls[0] += 1
        if program[0] is None and calls[0] > _EMBED_AFTER_CALLS:
            program[0] = embedded()
        return (program[0] or shared)(*values)

    return run


def _shared(key, problem: ChannelProblem, build, arguments):
    """The shape's program bound to ``problem``, compiling it on the second problem; None if it cannot be."""
    entry = _SHAPES[key]
    if entry is None or getattr(entry, "broken", False):
        _SHAPES[key] = None
        return None
    try:
        if isinstance(entry, list):
            entry = _SHAPES[key] = _programs.ShapeProgram(build, problem, arguments, entry)
            _programs.store(key, entry)
        if isinstance(entry, _programs._StoredProgram):
            return entry.bind(problem, lambda: shape_program(build(problem), *arguments))
        return entry.bind(problem)
    except _programs.Unbound:
        if not isinstance(entry, _programs.ShapeProgram):
            _SHAPES[key] = None
        return None


def _finish(
    root, residual, scale, tolerance, problem, factorization, field_scale, max_steps
) -> SteadySolution:
    """Certify the root on the residual both routes share, then report its fields."""
    final = _norm(residual(root))
    accepted = (
        jnp.isfinite(final) & jnp.isfinite(scale) & (final <= 10.0 * tolerance * jnp.maximum(scale, 1.0))
    )
    root = _certified(root, accepted, "steady solve")
    corrected, pressure = project(root, problem, factorization)
    # The currents and the scaled field come with the potential, so callers need no second potential solve.
    potential, currents, field = face_currents(corrected, problem, factorization, field_scale)
    return SteadySolution(corrected, pressure, potential, final, max_steps, scale, currents, field)


def _krylov(matvec, target, precond=None):
    result = solvax.gmres(matvec, target, precond=precond, rtol=1.0e-10, restart=60, max_restarts=60)
    return _certified(result.x, result.converged & jnp.isfinite(result.residual_norm), "steady linear solve")


def _certified(value, accepted, stage):
    """Reject eagerly; multiply by NaN under tracing so failed gradients fail too."""
    traced = any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves((value, accepted)))
    if not traced and not bool(accepted):
        raise RuntimeError(f"the {stage} did not converge")
    return jax.tree.map(lambda leaf: leaf * jnp.where(accepted, 1.0, jnp.nan), value)


def _tangent_solve(operator, target, precond=None):
    """Solve the linearised system at the root, matrix free, in both directions.

    :func:`jax.lax.custom_linear_solve` makes the Krylov iteration opaque to
    automatic differentiation and asks for the transposed solve explicitly, so
    the adjoint runs GMRES on the transposed operator rather than differentiating
    through the forward iteration -- which cannot be transposed, because a Krylov
    basis is not a linear function of its right-hand side.

    ``precond`` is the primal projection step. The transposed solve uses its
    exact transpose: the step is symmetric only in the face-volume inner product,
    and GMRES measures in the Euclidean one, where a stretched mesh makes the two
    differ by the width ratio. It is the step's pullback, because JAX 0.6.2 cannot
    :func:`jax.linear_transpose` the scans of the banded field-line solve.
    """

    def solve(matvec, rhs):
        return _krylov(matvec, rhs, precond)

    def transpose_solve(vecmat, rhs):
        if precond is None:
            return _krylov(vecmat, rhs)
        _, transposed = jax.vjp(precond, rhs)
        return _krylov(vecmat, rhs, lambda direction: transposed(direction)[0])

    return jax.lax.custom_linear_solve(operator, target, solve, transpose_solve)


def _dot(first: tuple[Field, Field, Field], second: tuple[Field, Field, Field]):
    return sum(jnp.sum(a.data * b.data) for a, b in zip(first, second, strict=True))


def _norm(velocity: tuple[Field, Field, Field]):
    return jnp.sqrt(sum(jnp.sum(field.data**2) for field in velocity))
