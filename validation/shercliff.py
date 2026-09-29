"""A spectral reference for insulating-duct MHD flow, independent of LMhdX.

Fully developed flow along a square duct with a transverse field has no closed
form that is comfortable to transcribe correctly, and a series copied out of a
paper is a reference only as far as the transcription is trusted. This module
solves the governing system directly instead, with Chebyshev collocation and a
dense linear solve, so the reference stands on the equations rather than on a
published expansion. It shares no operator, mesh or solver with the package.

With :math:`\\mathbf u = u(y,z)\\hat x`, :math:`\\mathbf B = B\\hat y` and
constant properties the inductionless system reduces to

.. math::
   \\mu\\nabla^2 u - \\sigma B^2 u + \\sigma B\\,\\partial_z\\varphi + f = 0,
   \\qquad \\nabla^2\\varphi = B\\,\\partial_z u,

with no slip on all four walls. An insulating wall carries no current, so
:math:`\\partial_n\\varphi = 0` there. A thin conducting wall carries the current
it receives along itself, which is Walker's condition: the surface current is
:math:`\\mathbf K = -c\\nabla_\\tau\\varphi` with
:math:`c = \\sigma_w t_w/(\\sigma a)`; charge conservation in the sheet,
:math:`\\nabla_\\tau\\cdot\\mathbf K = \\mathbf J\\cdot\\mathbf n`, then gives
:math:`\\partial_n\\varphi = c\\,\\partial_\\tau^2\\varphi` with
:math:`\\mathbf n` the outward normal. Setting both conductances to zero
recovers Shercliff's insulating duct; a nonzero conductance on the walls normal
to the field is Hunt's case.

The discretization is spectral, so the answer is converged to eight digits by
about forty points per direction and can be treated as exact when a finite-volume
result is compared against it, up to Ha 100. Beyond that the full-domain solve
resolves neither the Hartmann layer nor its own round-off by 96 points (Ha 1000:
0.60 % high); :func:`quadrant_flow_rate` is the reference there. At :math:`B=0` it reproduces the analytic
Poiseuille duct maximum 0.29468541 for unit forcing on ``[-1, 1]^2``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["chebyshev_weights", "duct_flow", "flow_rate", "hartmann_wall_current", "quadrant_flow_rate"]


def _differentiation_matrix(points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the Chebyshev differentiation matrix and its nodes on ``[-1, 1]``."""
    nodes = np.cos(np.pi * np.arange(points + 1) / points)
    scale = np.hstack([2.0, np.ones(points - 1), 2.0]) * (-1.0) ** np.arange(points + 1)
    spread = np.tile(nodes, (points + 1, 1)).T
    difference = spread - spread.T
    matrix = np.outer(scale, 1.0 / scale) / (difference + np.eye(points + 1))
    return matrix - np.diag(matrix.sum(axis=1)), nodes


def duct_flow(
    hartmann: float,
    points: int = 40,
    *,
    forcing: float = 1.0,
    hartmann_wall: float = 0.0,
    side_wall: float = 0.0,
    aspect: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the nodes, velocity and potential of a rectangular duct, square by default.

    Non-dimensionalised so that the half width, density, kinematic viscosity and
    conductivity are one; the field is then ``B = hartmann`` along ``y`` and the
    duct occupies ``[-1, 1]`` along ``y`` and ``[-aspect, aspect]`` along ``z``; the returned
    nodes are the unit ones, so ``z = aspect * nodes``.
    ``hartmann_wall`` and ``side_wall`` are the wall conductance ratios of the
    walls normal to and parallel to the field; both zero is Shercliff's duct.
    """
    if points < 4:
        raise ValueError("the spectral reference needs at least four points per direction")
    derivative, nodes = _differentiation_matrix(points)
    count = points + 1
    identity = np.eye(count)
    second_y = np.kron(derivative @ derivative, identity)
    second_z = np.kron(identity, derivative @ derivative) / aspect**2
    laplacian = second_y + second_z
    along_y = np.kron(derivative, identity)
    along_z = np.kron(identity, derivative) / aspect
    y = np.repeat(nodes, count)
    z = np.tile(nodes, count)
    edge = (np.abs(y) > 1.0 - 1e-12) | (np.abs(z) > 1.0 - 1e-12)

    size = count * count
    field = float(hartmann)
    operator = np.zeros((2 * size, 2 * size))
    source = np.zeros(2 * size)
    operator[:size, :size] = laplacian - field**2 * np.eye(size)
    operator[:size, size:] = field * along_z
    source[:size] = -float(forcing)
    operator[size:, :size] = -field * along_z
    operator[size:, size:] = laplacian
    for row in np.flatnonzero(edge):
        operator[row, :] = 0.0
        operator[row, row] = 1.0
        source[row] = 0.0
        operator[size + row, :] = 0.0
        if abs(y[row]) > 1.0 - 1e-12:
            outward = np.sign(y[row])
            closure = along_y[row] * outward - hartmann_wall * second_z[row]
        else:
            outward = np.sign(z[row])
            closure = along_z[row] * outward - side_wall * second_y[row]
        operator[size + row, size:] = closure
        source[size + row] = 0.0
    anchor = size // 2
    operator[size + anchor, :] = 0.0
    operator[size + anchor, size + anchor] = 1.0
    source[size + anchor] = 0.0

    solution = np.linalg.solve(operator, source)
    return nodes, solution[:size].reshape(count, count), solution[size:].reshape(count, count)


def chebyshev_weights(points: int) -> np.ndarray:
    """Return Clenshaw-Curtis quadrature weights for the Chebyshev nodes."""
    angles = np.pi * np.arange(points + 1) / points
    weights = np.zeros(points + 1)
    interior = np.ones(points - 1)
    for order in range(2, points, 2):
        interior -= 2.0 * np.cos(order * angles[1:points]) / (order**2 - 1)
    if points % 2 == 0:
        weights[0] = weights[points] = 1.0 / (points**2 - 1)
        interior -= np.cos(points * angles[1:points]) / (points**2 - 1)
    else:
        weights[0] = weights[points] = 1.0 / points**2
    weights[1:points] = 2.0 * interior / points
    return weights


def flow_rate(hartmann: float, points: int = 40, *, forcing: float = 1.0, **walls: float) -> float:
    """Return the mean velocity of the cross-section, ``Q / A`` (``Q = 4 * aspect * (Q / A)``)."""
    _, velocity, _ = duct_flow(hartmann, points, forcing=forcing, **walls)
    weights = chebyshev_weights(points)
    return float(weights @ velocity @ weights) / 4.0


def hartmann_wall_current(hartmann: float, points: int = 64, *, hartmann_wall: float) -> float:
    """Return the current a Hartmann wall carries across its midplane.

    The fluid drives current into the wall at ``y = 1`` on one side of ``z = 0``
    and takes it back on the other; the sheet carries the difference along
    itself, and the corners, which meet insulating side walls, carry none. The
    current crossing the midplane is therefore half the integral of the normal
    current density ``|J_y| = |d phi / dy|`` (the velocity vanishes at the wall)
    over the wall, which reads the exchange rather than one point of the sheet.
    """
    derivative, _ = _differentiation_matrix(points)
    _, _, potential = duct_flow(hartmann, points, hartmann_wall=hartmann_wall)
    normal = (derivative @ potential)[0]
    return 0.5 * float(chebyshev_weights(points) @ np.abs(normal))


def quadrant_flow_rate(
    hartmann: float, points: int = 48, *, beta: float = 5.0, hartmann_wall: float = 0.0, aspect: float = 1.0
) -> float:
    """Return ``Q / A`` from one quadrant, with the collocation points mapped onto the walls.

    ``aspect`` is the half-width ratio along ``z``, as in :func:`duct_flow`; it cancels in ``Q / A``.

    The same equations and wall closures as :func:`duct_flow`, on ``[0, 1]^2``
    using the parity of the solution: ``u`` is even in ``y`` and ``z``, the
    potential even in ``y`` and odd in ``z``, so the symmetry lines carry
    ``du/dn = 0``, ``dphi/dy = 0`` (``y = 0``) and ``phi = 0`` (``z = 0``). The
    collocation variable ``t`` is mapped to ``x = 1 - sinh(beta (1 - t)) / sinh(beta)``,
    which refines the wall by ``beta / sinh(beta)`` and keeps spectral
    convergence. The dense system (1-norm condition about 5e16 at Ha 1000) is
    row-equilibrated, factorized once and refined twice, which removes the
    round-off scatter of a plain solve. Shercliff Ha 1000 is 0.00097210343 at
    48 points and 0.00097210342 at 64; Hunt Ha 300 converges monotonically from
    below at ``beta = 5`` (``beta = 7`` returns garbage at 64-80 points).
    """
    import scipy.linalg

    derivative, nodes = _differentiation_matrix(points)
    t = 0.5 * (1.0 + nodes)
    x = 1.0 - np.sinh(beta * (1.0 - t)) / np.sinh(beta)
    first = 2.0 * derivative / (beta * np.cosh(beta * (1.0 - t)) / np.sinh(beta))[:, None]
    count = points + 1
    identity = np.eye(count)
    along_y, along_z = np.kron(first, identity), np.kron(identity, first) / aspect
    second_z = np.kron(identity, first @ first) / aspect**2
    size = count * count
    field = float(hartmann)
    operator = np.zeros((2 * size, 2 * size))
    source = np.zeros(2 * size)
    operator[:size, :size] = np.kron(first @ first, identity) + second_z - field**2 * np.eye(size)
    operator[:size, size:] = field * along_z
    source[:size] = -1.0
    operator[size:, :size] = -field * along_z
    operator[size:, size:] = operator[:size, :size] + field**2 * np.eye(size)
    y, z = np.repeat(x, count), np.tile(x, count)
    wall_y, wall_z, line_y, line_z = (
        np.isclose(v, edge) for v, edge in ((y, 1.0), (z, 1.0), (y, 0.0), (z, 0.0))
    )
    for row in np.flatnonzero(wall_y | wall_z | line_y | line_z):
        operator[row, :], source[row] = 0.0, 0.0
        if wall_y[row] or wall_z[row]:
            operator[row, row] = 1.0
        else:
            operator[row, :size] = along_y[row] if line_y[row] else along_z[row]
        charge = size + row
        operator[charge, :] = 0.0
        if line_z[row]:
            operator[charge, charge] = 1.0
        elif wall_y[row]:
            operator[charge, size:] = along_y[row] - hartmann_wall * second_z[row]
        else:
            operator[charge, size:] = along_z[row] if wall_z[row] else along_y[row]
    scale = 1.0 / np.abs(operator).max(axis=1)
    factors = scipy.linalg.lu_factor(operator * scale[:, None], check_finite=False)
    solution = scipy.linalg.lu_solve(factors, source * scale, check_finite=False)
    for _ in range(2):
        solution += scipy.linalg.lu_solve(factors, (source - operator @ solution) * scale, check_finite=False)
    weights = 0.5 * chebyshev_weights(points) * beta * np.cosh(beta * (1.0 - t)) / np.sinh(beta)
    return float(weights @ solution[:size].reshape(count, count) @ weights)
