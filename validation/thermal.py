"""Heat-transfer references independent of LMhdX: the figures of ``scripts/make_showcase_figures.py``.

Each is an exact series, a closed form or a dense spectral solve of the governing equations, and shares no
operator, mesh or solver with the package:

- :func:`h1_series`, the H1 rectangular duct by double sine series, and :func:`shah_london_h1`, Shah &
  London's (1978) fit of it;
- :func:`plate_nusselt`, uniformly heated Hartmann plates in closed form;
- :func:`graetz`, the plates' thermal entrance without axial conduction, by eigen-expansion;
- :func:`chandrasekhar_onset`, the onset of convection between rigid plates in a vertical field
  (Chandrasekhar 1961, Ch. IV), by Chebyshev collocation, against :data:`CHANDRASEKHAR_TABLE`.

``tests/test_steady.py`` checks each against a tabulated value or a limit.
"""

from __future__ import annotations

import numpy as np

from validation.shercliff import _differentiation_matrix

__all__ = [
    "CHANDRASEKHAR_TABLE",
    "chandrasekhar_onset",
    "graetz",
    "h1_series",
    "plate_nusselt",
    "shah_london_h1",
]

# Chandrasekhar (1961), rigid plates, as tabulated by Basak & Kumar (2015), Table I: (Q, Ra_c).
CHANDRASEKHAR_TABLE = ((0.0, 1707.8), (10.0, 1945.9), (50.0, 2802.1), (100.0, 3757.4))


def h1_series(a: float, b: float, *, slug: bool = False, terms: int = 801) -> float:
    """Nu of the H1 duct ``[-a, a] x [-b, b]`` from double sine series: no discrete operator involved.

    Velocity and temperature both solve Dirichlet Poisson problems, so with the
    eigenvalues ``lambda_mn`` and ``S_k = sum 256 ab / (m^2 n^2 pi^4 lambda^k)``
    over odd ``m, n``, ``Nu = S_1^2 D_h / (P S_3)``; slug flow gives ``16 a^2 b^2 D_h / (P S_1)``.
    """
    m, n = np.arange(1, terms, 2)[:, None], np.arange(1, terms, 2)[None, :]
    eigen = (m * np.pi / (2.0 * a)) ** 2 + (n * np.pi / (2.0 * b)) ** 2
    weight = 256.0 * a * b / (m**2 * n**2 * np.pi**4)
    first, perimeter, diameter = np.sum(weight / eigen), 4.0 * (a + b), 4.0 * a * b / (a + b)
    if slug:
        return 16.0 * a**2 * b**2 * diameter / (perimeter * first)
    return first**2 * diameter / (perimeter * np.sum(weight / eigen**3))


def shah_london_h1(alpha):
    """Shah & London's (1978) fit of the rectangular-duct H1 Nusselt number, short over long side ``alpha``.

    The coefficients are those of the ``ht`` library's ``Nu_laminar_rectangular_Shan_London``.
    """
    return 8.235 * np.polyval([-0.1861, 1.0578, -2.4765, 3.0853, -2.0421, 1.0], alpha)


def plate_nusselt(hartmann):
    """Uniformly heated Hartmann plates, ``D_h = 4a``: the closed form, 140/17 at Ha 0 and 12 as Ha grows."""
    ha = np.maximum(np.asarray(hartmann, dtype=float), 1e-3)
    tanh, sech = np.tanh(ha), 1.0 / np.cosh(np.minimum(ha, 300.0))
    value = 4.0 * (1.0 - tanh / ha) ** 2 / (1 / 3 - 2 / ha**2 - sech**2 / (2 * ha**2) + 2.5 * tanh / ha**3)
    return np.where(ha > 1e-2, value, 140.0 / 17.0)


def graetz(hartmann: float, xplus, cells: int = 4000, modes: int = 120) -> np.ndarray:
    """Local Nu of plates each taking in a uniform flux from ``x = 0``, without axial conduction.

    ``f theta_xi = theta_eta,eta`` with ``xi = alpha x/(U a^2) = 16 x+`` and ``f`` the mean-normalized
    velocity: Poiseuille at Ha 0, Hartmann, or slug at Ha inf, where the series is the closed form
    ``4 / (1/3 - (2/pi^2) sum exp(-16 n^2 pi^2 x+)/n^2)``. The developed part ``xi + phi(eta)`` and the
    modes of ``Y'' + lambda f Y = 0``, ``Y' = 0`` at the plates, a symmetric tridiagonal eigenproblem on fine
    finite volumes. ``Nu = 4/(theta_w - theta_b)`` with ``D_h = 4a``.
    """
    from scipy.linalg import eigh_tridiagonal

    def profile(eta):
        if hartmann == 0.0:
            return 1.5 * (1.0 - eta**2)
        if np.isinf(hartmann):
            return np.ones_like(eta)
        return (1.0 - np.cosh(hartmann * eta) / np.cosh(hartmann)) / (1.0 - np.tanh(hartmann) / hartmann)

    width = 2.0 / cells
    eta = -1.0 + width * (np.arange(cells) + 0.5)
    diagonal, scale = np.full(cells, 2.0 / width**2), 1.0 / np.sqrt(profile(eta))
    diagonal[[0, -1]] = 1.0 / width**2
    off = -scale[:-1] * scale[1:] / width**2
    rates, vectors = eigh_tridiagonal(diagonal * scale**2, off, select="i", select_range=(1, modes))
    shapes = vectors * scale[:, None]
    fine = np.linspace(-1.0, 1.0, 200001)
    weight = profile(fine)
    slope = -1.0 + np.concatenate(([0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(fine))))
    phi = np.concatenate(([0.0], np.cumsum(0.5 * (slope[1:] + slope[:-1]) * np.diff(fine))))
    phi -= np.trapezoid(weight * phi, fine) / np.trapezoid(weight, fine)
    weighted = profile(eta)[:, None] * shapes
    start = np.interp(eta, fine, phi)[:, None]
    amplitudes = -np.sum(weighted * start, axis=0) / np.sum(weighted * shapes, axis=0)
    decay = np.exp(-rates * 16.0 * np.asarray(xplus)[:, None])
    return 4.0 / (phi[-1] + np.sum(amplitudes * shapes[-1] * decay, axis=1))


def marginal_rayleigh(wavenumber: float, chandrasekhar: float, count: int = 48) -> float:
    """Stationary onset between rigid isothermal plates in a vertical field (Chandrasekhar 1961, Ch. IV).

    ``[(D^2 - a^2)^2 - Q D^2] W = Ra a^2 Theta`` and ``(D^2 - a^2) Theta = -W`` with ``W = DW = Theta = 0`` at
    ``z = +-1/2``, by Chebyshev collocation: the smallest real Ra of the generalized eigenproblem.
    """
    from scipy.linalg import eig

    derivative = 2.0 * _differentiation_matrix(count)[0]  # on [-1/2, 1/2]
    identity, second = np.eye(count + 1), derivative @ derivative
    helmholtz = second - wavenumber**2 * identity
    dirichlet, source = helmholtz.copy(), -identity
    dirichlet[[0, -1]], source[[0, -1]] = identity[[0, -1]], 0.0
    left = helmholtz @ helmholtz - chandrasekhar * second
    right = wavenumber**2 * np.linalg.solve(dirichlet, source)
    ends = ((0, identity[0]), (1, derivative[0]), (count - 1, derivative[-1]), (count, identity[-1]))
    for row, condition in ends:
        left[row], right[row] = condition, 0.0
    values = eig(left, right, right=False)
    values = values[np.isfinite(values)]
    real = values[(np.abs(values.imag) <= 1e-6 * np.abs(values.real)) & (values.real > 0)]
    return float(np.min(real.real))


def chandrasekhar_onset(chandrasekhar: float) -> tuple[float, float]:
    """The critical Rayleigh number and wavenumber, the marginal curve's minimum."""
    from scipy.optimize import minimize_scalar

    guess = (np.pi**4 * chandrasekhar / 2.0) ** (1.0 / 6.0) if chandrasekhar > 100.0 else 3.1
    bounds = (0.6 * guess, 1.8 * guess)
    result = minimize_scalar(marginal_rayleigh, bounds=bounds, args=(chandrasekhar,), method="bounded")
    return float(result.fun), float(result.x)
