"""Exact identities for the quasi-two-dimensional model (plan validation row 12).

Two properties pin the nonlinear evolution without any reference data.

The first is linear. The Taylor--Green vortex is an exact steady solution of the
two-dimensional Euler equation, so its advection term vanishes identically and
the model reduces to ``d omega / dt = -(nu k^2 + gamma) omega``. Any error in the
linear part therefore shows up directly, undiluted by advection.

The second is nonlinear and is the reason this file exists. Without viscosity,
Hartmann friction can be removed from the equation entirely by a change of
variables: with ``omega = exp(-gamma t) W`` and the rescaled time
``tau = (1 - exp(-gamma t)) / gamma`` the frictional equation becomes the
frictionless one. Velocity is linear in vorticity, so the advection term picks up
exactly the factor the rescaling needs. A frictional run and a frictionless run
must then agree after that mapping, which tests the coupling between advection
and friction rather than either alone. The agreement is checked by refinement,
because each run carries its own time-integration error.

Friction dominating an eddy turnover is the blanket regime rather than an
academic corner: for lead-lithium at ``h = 0.1 m`` and ``B = 5 T`` the Hartmann
braking time is a few seconds against a turnover of about one.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lmhdx.q2d import evolve_q2d
from validation.q2d_campaigns import TITHOF_PUBLISHED, tithof_formula, tithof_neutral

pytestmark = pytest.mark.numerical

SHAPE = (64, 64)
LENGTH = (2.0 * np.pi, 2.0 * np.pi)


def _coordinates() -> tuple[np.ndarray, np.ndarray]:
    """Build the grid in NumPy so the test states are unambiguously double precision."""
    x = np.arange(SHAPE[0]) * LENGTH[0] / SHAPE[0]
    y = np.arange(SHAPE[1]) * LENGTH[1] / SHAPE[1]
    return x[:, None], y[None, :]


def _taylor_green(amplitude: float = 1.0) -> jnp.ndarray:
    x, y = _coordinates()
    return jnp.asarray(amplitude * np.sin(x) * np.sin(y))


def _multi_scale() -> jnp.ndarray:
    x, y = _coordinates()
    field = (
        np.sin(x) * np.sin(y)
        + 0.6 * np.sin(2.0 * x + 0.3) * np.cos(y)
        + 0.4 * np.cos(3.0 * x) * np.sin(2.0 * y - 0.7)
    )
    return jnp.asarray(field / np.sqrt(np.mean(field**2)))


@pytest.mark.parametrize("friction", [0.0, 0.35, 2.0])
def test_taylor_green_decay_matches_its_analytic_rate(friction):
    """Advection vanishes for this field, so the decay is purely linear."""
    viscosity, dt, steps = 5.0e-3, 2.0e-3, 200
    initial = _taylor_green()
    final, _, _ = evolve_q2d(
        initial, length=LENGTH, viscosity=viscosity, hartmann_friction=friction, dt=dt, steps=steps
    )
    # The Taylor-Green mode has |k|^2 = 2 on this domain.
    expected = np.asarray(initial) * np.exp(-(viscosity * 2.0 + friction) * dt * steps)
    scale = float(np.max(np.abs(expected)))
    assert np.max(np.abs(np.asarray(final) - expected)) < 1e-10 * scale


def _frictional_mismatch(dt: float, *, friction: float, total_time: float) -> float:
    """Compare a frictional run with the rescaled frictionless run it must equal."""
    steps = int(round(total_time / dt))
    initial = _multi_scale()
    damped, _, _ = evolve_q2d(
        initial, length=LENGTH, viscosity=0.0, hartmann_friction=friction, dt=dt, steps=steps
    )
    # tau = (1 - exp(-gamma t)) / gamma is the time the frictionless flow must run.
    rescaled_time = (1.0 - np.exp(-friction * total_time)) / friction
    inviscid_steps = int(round(rescaled_time / dt))
    inviscid_dt = rescaled_time / inviscid_steps
    inviscid, _, _ = evolve_q2d(
        initial, length=LENGTH, viscosity=0.0, hartmann_friction=0.0, dt=inviscid_dt, steps=inviscid_steps
    )
    expected = np.exp(-friction * total_time) * np.asarray(inviscid)
    return float(np.max(np.abs(np.asarray(damped) - expected)) / np.max(np.abs(expected)))


def test_friction_maps_onto_the_frictionless_flow_under_time_rescaling():
    """The mismatch is time-integration error only, so refinement must remove it."""
    friction, total_time = 0.8, 0.4
    coarse = _frictional_mismatch(4.0e-3, friction=friction, total_time=total_time)
    fine = _frictional_mismatch(2.0e-3, friction=friction, total_time=total_time)
    assert coarse < 5.0e-3, coarse
    assert fine < coarse
    # A fourth-order scheme should gain far more than a factor of two per halving.
    assert coarse / fine > 6.0, (coarse, fine)


def test_frictionless_inviscid_evolution_conserves_its_invariants():
    """With no viscosity and no friction, energy and enstrophy must not drift."""
    initial = _multi_scale()
    dt, steps = 2.0e-3, 200
    final, ux, uy = evolve_q2d(
        initial, length=LENGTH, viscosity=0.0, hartmann_friction=0.0, dt=dt, steps=steps
    )
    kx = 2.0 * np.pi * np.fft.fftfreq(SHAPE[0], d=LENGTH[0] / SHAPE[0])
    ky = 2.0 * np.pi * np.fft.fftfreq(SHAPE[1], d=LENGTH[1] / SHAPE[1])
    squared = kx[:, None] ** 2 + ky[None, :] ** 2

    def energy(vorticity):
        transformed = np.fft.fftn(np.asarray(vorticity))
        stream = np.divide(transformed, squared, out=np.zeros_like(transformed), where=squared > 0)
        return float(0.5 * np.sum(np.abs(stream) ** 2 * squared) / np.prod(SHAPE) ** 2)

    def enstrophy(vorticity):
        return float(0.5 * np.mean(np.asarray(vorticity) ** 2))

    assert energy(final) == pytest.approx(energy(initial), rel=2e-6)
    assert enstrophy(final) == pytest.approx(enstrophy(initial), rel=2e-6)
    assert np.all(np.isfinite(np.asarray(ux))) and np.all(np.isfinite(np.asarray(uy)))


@pytest.mark.unit
def test_the_working_dtype_follows_the_state_it_is_given():
    """Precision is chosen by the state, with float32 as a floor rather than a cap.

    The states above are built in NumPy on purpose. A grid written as
    ``jnp.arange(n) * step`` is weakly typed, and the promotion rules can then
    hand the solver a single-precision state even with x64 enabled, which would
    quietly cap the identities in this module at about 1e-7 instead of 1e-10.
    """
    import jax

    if not jax.config.x64_enabled:  # pragma: no cover - depends on session config
        pytest.skip("double precision is not enabled in this session")
    double = jnp.asarray(_taylor_green(), dtype=jnp.float64)
    final, ux, uy = evolve_q2d(
        double, length=LENGTH, viscosity=1.0e-3, hartmann_friction=0.25, dt=2.0e-3, steps=5
    )
    assert final.dtype == jnp.float64
    assert ux.dtype == jnp.float64 and uy.dtype == jnp.float64
    assert _taylor_green().dtype == jnp.float64

    single = jnp.asarray(_taylor_green(), dtype=jnp.float32)
    assert (
        evolve_q2d(single, length=LENGTH, viscosity=1.0e-3, hartmann_friction=0.25, dt=2.0e-3, steps=5)[
            0
        ].dtype
        == jnp.float32
    )


@pytest.mark.validation
def test_the_neutral_reynolds_number_at_the_critical_wavenumber_is_within_2_percent_of_tithof_et_al():
    """Section 4.2 of J. Fluid Mech. 828, 837: Re_c 7.60 at q_c 0.465 for beta 1, alpha/(nu kappa^2) 0.3208.

    The published value is the minimum of their three-mode eq. (4.1), 7.646 here at q_c (7.60 as printed);
    the periodic-box growth rates of ``lmhdx.q2d`` give 7.694 at q_c, +1.2 % from 7.60 and +0.6 % from
    the formula, whose truncation to three modes is the expected sign of the difference.
    """
    published, q_c = TITHOF_PUBLISHED
    measured = tithof_neutral(q_c)
    assert measured == pytest.approx(published, rel=0.02)
    assert measured == pytest.approx(float(tithof_formula(q_c)), rel=0.01)
    assert measured > float(tithof_formula(q_c))
