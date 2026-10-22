"""Q2D literature campaigns: validation rows 13 (Sommeria 1986) and 15 (Tithof et al. 2017).

``python -m validation.q2d_campaigns tithof`` and ``... sommeria`` rerun the tables in
``docs/validation/q2d.md``. Both use only ``lmhdx.q2d.evolve_q2d``.

Row 15. Tithof, Suri, Pallantla, Grigoriev and Schatz (J. Fluid Mech. 828, 837, 2017,
Section 4.2) linearise the Kolmogorov-like flow ``u_s ~ sin(kappa y)`` of
``du/dt + beta u.grad u = -grad p + nu lap u - alpha u + f`` and give the neutral curve
(their eq. 4.1, three Fourier modes, unbounded domain)

.. math::
   Re_n(q) = \\frac{\\pi}{\\beta q}\\sqrt{\\frac{1+q^2}{1-q^2}
   \\Bigl(q^2+\\frac{\\alpha}{\\nu\\kappa^2}\\Bigr)\\Bigl(1+q^2+\\frac{\\alpha}{\\nu\\kappa^2}\\Bigr)},

whose minimum for beta = 1 is ``Re_c = 7.60`` at ``q_c = 0.465`` for the experiment's
``alpha = 0.064 1/s``, ``nu = 3.26e-6 m^2/s`` and magnet width ``w = 1.27 cm`` (so
``kappa = pi/w``). ``Re = U w / nu`` with ``U`` the rms speed. The model with
``beta = 1`` is the one ``lmhdx.q2d`` solves (``beta`` rescales the velocity and the
forcing), so the campaign steps the exact steady flow plus a small perturbation of
wavenumber ``q kappa`` and reads the growth rate off the perturbation norm.

Row 13. Sommeria (J. Fluid Mech. 170, 139, 1986) drives a lattice of vortices
electrically and measures the inverse-cascade spectrum ``E(k) = C eps^(2/3) k^(-5/3)``
with ``C`` between 3 and 7 over his friction Reynolds numbers. Here the lattice is
``f_omega = F sin(k_f x) sin(k_f y)`` with ``k_f = 6`` periods per side of a
``2 pi`` box, steady, with friction ``alpha = 1``; ``eps`` is the local spectral energy
flux ``|Pi(k)|`` from the nonlinear transfer, so no assumption about what friction removes is
needed. The friction Reynolds number is ``Re_alpha = u_rms k_f / alpha``.
"""

from __future__ import annotations

import sys

import numpy as np

__all__ = ["lattice_spectrum", "tithof_critical", "tithof_growth", "tithof_neutral"]

TITHOF_FRICTION = 0.064 / (3.26e-6 * (np.pi / 0.0127) ** 2)  # alpha / (nu kappa^2) = 0.3208
TITHOF_PUBLISHED = (7.60, 0.465)  # Re_c and q_c for beta = 1


def tithof_formula(q, friction=TITHOF_FRICTION):
    """Eq. (4.1) of Tithof et al. 2017 for beta = 1."""
    return np.pi / q * np.sqrt((1 + q * q) / (1 - q * q) * (q * q + friction) * (1 + q * q + friction))


def tithof_growth(re, q, friction=TITHOF_FRICTION, *, cells=16, dt=0.02, window=(40.0, 120.0), eps=1e-8):
    """Growth rate of the wavenumber-``q`` perturbation of the sinusoidal flow at ``Re`` (nu = kappa = 1)."""
    import jax.numpy as jnp

    from lmhdx import enable_x64
    from lmhdx.q2d import evolve_q2d

    enable_x64()
    length = (2 * np.pi / q, 2 * np.pi)
    x, y = (np.arange(cells) * s / cells for s in length)
    base = (
        -np.sqrt(2) * re / np.pi * np.cos(y)[None, :] * np.ones((cells, 1))
    )  # u = U0 sin(y), U0 = sqrt(2) Re / pi
    forcing = (1 + friction) * base
    rng = np.random.default_rng(0)
    modes = [(m, *rng.standard_normal(2)) for m in (-1, 0, 1)]
    noise = sum(c * np.cos(q * x[:, None] + m * y) + s * np.sin(q * x[:, None] + m * y) for m, c, s in modes)
    start = jnp.asarray(base + eps * noise / np.abs(noise).max())

    def size(time):
        steps = round(time / dt)
        final = evolve_q2d(
            start,
            forcing=jnp.asarray(forcing),
            length=length,
            viscosity=1.0,
            hartmann_friction=friction,
            dt=dt,
            steps=steps,
        )[0]
        return float(np.log(np.sqrt(np.mean((np.asarray(final) - base) ** 2))))

    return (size(window[1]) - size(window[0])) / (window[1] - window[0])


def tithof_neutral(q, friction=TITHOF_FRICTION, **kwargs):
    """Reynolds number at which the growth rate of the wavenumber-``q`` perturbation vanishes (secant)."""
    low, high = tithof_formula(q, friction), 1.02 * tithof_formula(q, friction)
    g_low, g_high = (tithof_growth(r, q, friction, **kwargs) for r in (low, high))
    for _ in range(6):
        low, g_low, high = high, g_high, high - g_high * (high - low) / (g_high - g_low)
        g_high = tithof_growth(high, q, friction, **kwargs)
        if abs(high - low) < 2e-4 * high:
            break
    return high


def tithof_critical(wavenumbers=(0.40, 0.44, 0.465, 0.50, 0.54), **kwargs):
    """Return ``(Re_c, q_c, rows)`` from a parabola through the neutral curve; rows are ``(q, formula, measured)``."""
    measured = [tithof_neutral(q, **kwargs) for q in wavenumbers]
    a, b, c = np.polyfit(wavenumbers, measured, 2)
    q_c = -b / (2 * a)
    rows = [(q, float(tithof_formula(q)), r) for q, r in zip(wavenumbers, measured, strict=True)]
    return float(a * q_c**2 + b * q_c + c), float(q_c), rows


def lattice_spectrum(
    forcing_amplitude,
    *,
    seed=1,
    cells=128,
    friction=1.0,
    viscosity=2e-3,
    k_f=6,
    dt=None,
    duration=150.0,
    average=90.0,
):
    """Run the steady vortex lattice and return ``(E(k), Pi(k), u_rms, eps_in)`` averaged over the last ``average``."""
    import jax.numpy as jnp

    from lmhdx import enable_x64
    from lmhdx.q2d import evolve_q2d

    enable_x64()
    n = cells
    if dt is None:  # the Courant number grows with the speed, which grows with F
        dt = 2e-3 * min(1.0, 600.0 / forcing_amplitude)
    x = np.arange(n) * 2 * np.pi / n
    forcing = forcing_amplitude * np.sin(k_f * x)[:, None] * np.sin(k_f * x)[None, :]
    omega = jnp.asarray(1e-2 * np.random.default_rng(seed).standard_normal((n, n)))
    k1 = np.fft.fftfreq(n, 1 / n)
    kx, ky = np.meshgrid(k1, k1, indexing="ij")
    k2 = np.where((kx == 0) & (ky == 0), 1.0, kx**2 + ky**2)
    shell = np.rint(np.sqrt(k2)).astype(int).ravel()
    kmax = n // 3
    forcing_hat = np.fft.fft2(forcing) / n**2

    def measure(field):
        w_hat = np.fft.fft2(field) / n**2
        psi = w_hat / k2
        ux, uy = (np.real(np.fft.ifft2(s * 1j * k * psi * n**2)) for s, k in ((1, ky), (-1, kx)))
        advection = (
            np.fft.fft2(
                ux * np.real(np.fft.ifft2(1j * kx * w_hat * n**2))
                + uy * np.real(np.fft.ifft2(1j * ky * w_hat * n**2))
            )
            / n**2
        )
        energy = np.bincount(shell, (0.5 * k2 * np.abs(psi) ** 2).ravel())[:kmax]
        transfer = np.bincount(shell, np.real(np.conj(psi) * -advection).ravel())[:kmax]
        return (
            energy,
            -np.cumsum(transfer),
            np.sqrt(np.mean(ux**2 + uy**2)),
            np.real(np.sum(np.conj(psi) * forcing_hat)),
        )

    samples, time, chunk = [], 0.0, 5.0
    while time < duration - 1e-9:
        omega = evolve_q2d(
            omega,
            forcing=jnp.asarray(forcing),
            viscosity=viscosity,
            hartmann_friction=friction,
            dt=dt,
            steps=round(chunk / dt),
        )[0]
        time += chunk
        if time > duration - average:
            samples.append(measure(np.asarray(omega)))
    return tuple(np.mean([s[i] for s in samples], axis=0) for i in range(4))


def inverse_cascade_constant(energy, flux, band=(3, 6)):
    """Mean of ``E(k) k^(5/3) / |Pi(k)|^(2/3)`` over the shells in ``band`` (inclusive)."""
    k = np.arange(band[0], band[1] + 1)
    return float(np.mean(energy[k] * k ** (5 / 3) / np.abs(flux[k]) ** (2 / 3)))


def _tithof_table():
    print("| q | Eq. (4.1) Re_n | lmhdx.q2d Re_n | difference |\n|---|---|---|---|")
    re_c, q_c, rows = tithof_critical()
    for q, formula, measured in rows:
        print(f"| {q:.3f} | {formula:.4f} | {measured:.4f} | {measured / formula - 1:+.2%} |")
    published = TITHOF_PUBLISHED
    print(
        f"\nRe_c = {re_c:.3f} ({re_c / published[0] - 1:+.2%} from {published[0]}), q_c = {q_c:.3f} ({q_c / published[1] - 1:+.2%} from {published[1]})"
    )


def _sommeria_table():
    print(
        "| F | seed | Re_alpha | E(1)/E | eps_in | |Pi| at k = 4 | C (k 3-6) |\n|---|---|---|---|---|---|---|"
    )
    for amplitude in map(float, sys.argv[2:] or (30, 100, 300, 600, 1200)):
        for seed in (1, 2):
            energy, flux, u_rms, eps_in = lattice_spectrum(amplitude, seed=seed)
            print(
                f"| {amplitude:g} | {seed} | {u_rms * 6:.1f} | {energy[1] / energy.sum():.3f} | {eps_in:.2f} | {abs(flux[4]):.2f} "
                f"| {inverse_cascade_constant(energy, flux):.2f} |",
                flush=True,
            )


if __name__ == "__main__":
    {"tithof": _tithof_table, "sommeria": _sommeria_table}[sys.argv[1]]()
