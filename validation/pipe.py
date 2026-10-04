"""A spectral reference for pipe flow in a transverse field, independent of LMhdX.

The pipe has the same difficulty the duct had -- no closed form worth
transcribing -- and one the duct did not: a coordinate singularity at the axis.
The representation here removes it rather than treating it. A function on the
disk is written on the *diameter*, with Chebyshev points on ``[-1, 1]`` and an
odd point count so that none of them lands on the centre, and the identification
:math:`f(-r, \\theta) = f(r, \\theta + \\pi)` is built into the operators. The axis
is then an interior point of the representation and needs no condition at all;
this is Trefethen's construction for the Poisson equation on a disk.

The system is the same one :mod:`validation.shercliff` solves, in the geometry
:mod:`lmhdx.poisson` solves it in. With :math:`\\mathbf u = u(r,\\theta)\\hat z` and
:math:`\\mathbf B = B\\hat x`,

.. math::
   \\nabla^2 u + B\\,\\partial_y\\varphi - B^2 u + f = 0,
   \\qquad \\nabla^2\\varphi = B\\,\\partial_y u,

with :math:`\\partial_y = \\sin\\theta\\,\\partial_r + r^{-1}\\cos\\theta\\,\\partial_\\theta`,
no slip at :math:`r=1`, and either an insulating wall,
:math:`\\partial_r\\varphi = 0`, or Walker's thin conducting one,
:math:`\\partial_r\\varphi = c\\,\\partial_\\theta^2\\varphi`.

At :math:`B=0` it returns the exact Hagen-Poiseuille mean velocity, one eighth
for a unit radius under unit forcing, which is what fixes every normalisation.
"""

from __future__ import annotations

import numpy as np

from .shercliff import _differentiation_matrix

__all__ = ["flow_rate", "pipe_flow"]


def _disk_operators(radial: int, azimuthal: int):
    """Return the radial half-grid and the operators the double cover implies.

    ``radial`` must be odd so that no Chebyshev point sits on the axis, and
    ``azimuthal`` even so that the half-turn that identifies ``-r`` with
    ``theta + pi`` is a permutation of the azimuthal points.
    """
    if radial % 2 == 0:
        raise ValueError("the radial point count must be odd so no point lands on the axis")
    if azimuthal % 2:
        raise ValueError("the azimuthal point count must be even for the half-turn identification")
    derivative, nodes = _differentiation_matrix(radial)
    half = (radial + 1) // 2
    radius = nodes[:half]
    second = derivative @ derivative
    turn = np.roll(np.eye(azimuthal), azimuthal // 2, axis=0)
    identity = np.eye(azimuthal)

    def cover(matrix: np.ndarray) -> np.ndarray:
        """Fold a full-diameter operator onto the half-grid, through the identification."""
        near = matrix[:half, :half]
        far = matrix[:half, radial : half - 1 : -1]
        return np.kron(near, identity) + np.kron(far, turn)

    angle = 2.0 * np.pi * np.arange(azimuthal) / azimuthal
    modes = np.fft.fftfreq(azimuthal, d=1.0 / azimuthal)
    transform = np.exp(1j * np.outer(angle, modes)) / azimuthal
    inverse = np.exp(-1j * np.outer(modes, angle))
    azimuthal_first = np.real(transform @ np.diag(1j * modes) @ inverse)
    azimuthal_second = np.real(transform @ np.diag(-(modes**2)) @ inverse)
    return radius, angle, cover(derivative), cover(second), azimuthal_first, azimuthal_second


def pipe_flow(
    hartmann: float,
    radial: int = 31,
    azimuthal: int = 32,
    *,
    forcing: float = 1.0,
    wall_conductance: float = 0.0,
):
    """Return the radius, azimuth, velocity and potential of a circular pipe."""
    radius, angle, first, second, azimuthal_first, azimuthal_second = _disk_operators(radial, azimuthal)
    half = radius.size
    size = half * azimuthal
    inverse_radius = np.repeat(1.0 / radius, azimuthal)
    laplacian = (
        second
        + inverse_radius[:, None] * first
        + (inverse_radius**2)[:, None] * np.kron(np.eye(half), azimuthal_second)
    )
    along_theta = np.kron(np.eye(half), azimuthal_first)
    sine = np.tile(np.sin(angle), half)
    cosine = np.tile(np.cos(angle), half)
    along_y = sine[:, None] * first + (cosine * inverse_radius)[:, None] * along_theta

    field = float(hartmann)
    operator = np.zeros((2 * size, 2 * size))
    source = np.zeros(2 * size)
    operator[:size, :size] = laplacian - field**2 * np.eye(size)
    operator[:size, size:] = field * along_y
    source[:size] = -float(forcing)
    operator[size:, :size] = -field * along_y
    operator[size:, size:] = laplacian

    # The wall is the first radial point, r = 1; every azimuthal point on it.
    for column in range(azimuthal):
        row = column
        operator[row, :] = 0.0
        operator[row, row] = 1.0
        source[row] = 0.0
        operator[size + row, :] = 0.0
        operator[size + row, size:] = (
            first[row] - float(wall_conductance) * np.kron(np.eye(half), azimuthal_second)[row]
        )
        source[size + row] = 0.0
    anchor = size - 1
    operator[size + anchor, :] = 0.0
    operator[size + anchor, size + anchor] = 1.0
    source[size + anchor] = 0.0

    solution = np.linalg.solve(operator, source)
    return radius, angle, solution[:size].reshape(half, azimuthal), solution[size:].reshape(half, azimuthal)


def flow_rate(hartmann: float, radial: int = 31, azimuthal: int = 32, **options) -> float:
    """Return the mean axial velocity of the cross-section, ``Q / A``.

    The azimuthal average is a plain mean, which is spectrally accurate on
    equispaced points. The radial integral is Gauss-Legendre on ``[0, 1]``, not
    Clenshaw-Curtis on the diameter: the area element carries ``|r|``, whose
    kink at the axis costs a Chebyshev rule its accuracy and leaves the flow
    rate converging at first order while the field it integrates is exact to
    1e-13.
    """
    radius, _, velocity, _ = pipe_flow(hartmann, radial, azimuthal, **options)
    profile = velocity.mean(axis=1)
    nodes, weights = np.polynomial.legendre.leggauss(radial + 2)
    points = 0.5 * (nodes + 1.0)
    interpolated = _barycentric(radius, profile, points)
    return float(2.0 * np.sum(0.5 * weights * points * interpolated))


def _barycentric(nodes: np.ndarray, values: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate the Chebyshev interpolant of ``values`` at ``points``.

    ``nodes`` are the positive half of a Chebyshev diameter grid and the
    profile is even, so the interpolant of the whole diameter is recovered by
    mirroring before interpolating.
    """
    full = np.concatenate([nodes, -nodes[::-1]])
    mirrored = np.concatenate([values, values[::-1]])
    count = full.size - 1
    weights = np.array([0.5 if index in (0, count) else 1.0 for index in range(count + 1)])
    weights = weights * (-1.0) ** np.arange(count + 1)
    difference = points[:, None] - full[None, :]
    exact = np.isclose(difference, 0.0)
    difference = np.where(exact, 1.0, difference)
    terms = weights[None, :] / difference
    result = (terms * mirrored[None, :]).sum(axis=1) / terms.sum(axis=1)
    hits = exact.any(axis=1)
    if hits.any():
        result[hits] = mirrored[np.argmax(exact[hits], axis=1)]
    return result
