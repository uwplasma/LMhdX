"""The TM-228 inertialess core-flow model: Walker's limits, the ANL fringe, symmetry and the adjoint."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmhdx.core3d import fringe_field
from lmhdx.coreflow import (
    CoreFlow,
    fully_developed_gradient,
    layer_conductances,
    midplane_field,
    side_layer_coefficient,
)
from lmhdx.grid import Grid, uniform_faces

pytestmark = pytest.mark.unit


def _anl(x, half_length=3.0):
    """TM-228 eq. (16): ``B_y = (1 - sin(pi x / 2 x0)) / 2`` inside the fringe, 1 upstream, 0 downstream."""
    inside = np.clip(x, -half_length, half_length)
    return 0.5 * (1.0 - np.sin(np.pi * inside / (2.0 * half_length)))


def _anl_square_integral(lower, upper=2.0):
    """Closed form of the integral of ``B_y^2`` from ``lower <= -3`` to ``upper`` in ``[-3, 3]`` (x0 = 3)."""

    def antiderivative(u):
        return 1.5 * u + 2.0 * np.cos(u) - np.sin(2.0 * u) / 4.0

    fringe = (antiderivative(np.pi * upper / 6.0) - antiderivative(-np.pi / 2.0)) * 6.0 / (4.0 * np.pi)
    return (-3.0 - lower) + fringe


@pytest.mark.physics
@pytest.mark.parametrize(
    ("aspect", "c_t", "c_s", "scale"),
    [(1.0, 0.02, 0.02, 1.0), (0.5, 0.05, 0.1, 2.0), (2.0, 0.1, 0.01, 0.7)],
)
def test_a_uniform_field_gives_walkers_fully_developed_gradient_at_second_order(aspect, c_t, c_s, scale):
    exact = scale**2 * fully_developed_gradient(c_t, c_s, aspect)
    errors = []
    for cells in (8, 16):
        model = CoreFlow(np.linspace(0.0, 2.0, 9), aspect=aspect, nz=cells, ny=cells)
        result = model.solve(np.ones(9), c_t=c_t, c_s=c_s, field_scale=scale)
        errors.append(float(result.pressure_drop) / 2.0 / exact - 1.0)
        assert float(jnp.ptp(result.pressure[4])) < 1e-12
        assert float(jnp.ptp(result.axial_flux)) < 1e-12
        assert float(jnp.mean(result.axial_flux)) == pytest.approx(aspect, rel=1e-12)
    # The only error is the trapezoidal side-wall integral of a quadratic potential.
    assert abs(errors[1]) < 1e-3
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.02)


@pytest.mark.physics
def test_perfectly_conducting_side_walls_give_the_hartmann_wall_limit():
    model = CoreFlow(np.linspace(0.0, 2.0, 9), nz=8, ny=8)
    result = model.solve(np.ones(9), c_t=0.02, c_s=1e4)
    assert float(result.pressure_drop) / 2.0 == pytest.approx(0.02 / 1.02, rel=2e-6)


def test_the_coupled_operator_is_symmetric_and_indefinite():
    x = np.linspace(-6.0, 2.0, 17)
    model = CoreFlow(x, aspect=0.8, nz=4, ny=5)
    matrix = model.operator(_anl(x), c_t=0.03, c_s=0.05, field_scale=1.3)
    asymmetry = abs(matrix - matrix.T).max() / abs(matrix).max()
    assert asymmetry < 1e-12
    eigenvalues = np.linalg.eigvalsh(matrix.toarray())
    assert eigenvalues.min() < 0.0 < eigenvalues.max()
    assert np.min(np.abs(eigenvalues)) > 1e-10 * np.max(np.abs(eigenvalues))


@pytest.mark.validation
def test_the_anl_fringe_reproduces_tm228():
    """ANL/FPP/TM-228 section 4.1 (x0 = 3, c_t = c_s = 0.02, a = 1): row 7 of the validation ladder.

    The gate is the excess of the drop from x = -6 to 2 over the locally fully
    developed drop, both computed by LMhdX on the same mesh, against TM-228's
    0.0932 - 0.0754 = 0.0178. The absolute drop (0.0954 here, 0.0951 on exactly
    [-6, 2]) is reported, not gated: TM-228's own fully developed gradient
    integrates to 0.0776 over [-6, 2], not the quoted 0.0754 (plan, D27a).
    """
    gradient = fully_developed_gradient(0.02, 0.02)
    assert gradient * _anl_square_integral(-6.0) == pytest.approx(0.0775728, rel=1e-6)
    # The domain runs past the fringe into the zero field, as TM-228's cap on 1/B implies.
    x = np.linspace(-10.0, 6.0, 161)
    model = CoreFlow(x, nz=20, ny=20)
    result = model.solve(_anl(x), c_t=0.02, c_s=0.02)
    # 1/B is capped at 1000 from x = 2.9 (B = 6.9e-4) to the end of the domain.
    assert int(result.floored_nodes) == 32
    assert float(jnp.ptp(result.axial_flux)) < 1e-10
    pressure = np.asarray(result.pressure).mean(axis=1)
    drop = np.interp(-6.0, x, pressure) - np.interp(2.0, x, pressure)
    # LMhdX's own fully developed gradient on the same cross-section mesh, integrated over the same stations.
    uniform = CoreFlow(np.linspace(0.0, 2.0, 9), nz=20, ny=20).solve(np.ones(9), c_t=0.02, c_s=0.02)
    inside = (x > -6.0 - 1e-9) & (x < 2.0 + 1e-9)
    square = _anl(x[inside]) ** 2
    local = (
        float(uniform.pressure_drop)
        / 2.0
        * float(np.sum(0.5 * (square[1:] + square[:-1]) * np.diff(x[inside])))
    )
    assert drop - local == pytest.approx(0.0932 - 0.0754, rel=0.01)
    # Figure 10: the three-dimensional excess is k c^(1/2) with k = 0.126 for x0 = 3.
    assert drop - local == pytest.approx(0.126 * 0.02**0.5, rel=0.01)


def test_the_fringe_field_provider_feeds_the_model():
    grid = Grid(uniform_faces(40, -6.0, 2.0), uniform_faces(4, -1.0, 1.0), uniform_faces(2, -1.0, 1.0))
    x, field = midplane_field(fringe_field(grid, solenoidal=False))
    # Each cell holds the mean of its two faces, a difference of the vector potential.
    np.testing.assert_allclose(field, _anl(x), atol=1e-3)
    result = CoreFlow(x, nz=4, ny=4).solve(field, c_t=0.02, c_s=0.02)
    assert 0.08 < float(result.pressure_drop) < 0.1


@pytest.mark.numerical
def test_the_adjoint_matches_central_differences():
    x = np.linspace(-4.0, 2.0, 25)
    model = CoreFlow(x, aspect=0.9, nz=6, ny=5)
    field = _anl(x)

    def drop(parameters):
        c_t, c_s, scale = parameters
        return model.solve(field, c_t=c_t, c_s=c_s, field_scale=scale).pressure_drop

    def flux(parameters):
        c_t, c_s, drive = parameters
        return jnp.mean(model.solve(field, c_t=c_t, c_s=c_s, drive=drive, mean_velocity=None).axial_flux)

    derivatives = {}
    for objective, point in ((drop, (0.03, 0.05, 1.2)), (flux, (0.03, 0.05, 0.7))):
        point = jnp.asarray(point)
        derivatives[objective] = jax.jit(jax.grad(objective))
        gradient = np.asarray(derivatives[objective](point))
        tangent = np.asarray(jax.jit(jax.jacfwd(objective))(point))
        objective = jax.jit(objective)
        steps = 1e-5 * np.asarray(point)
        central = np.array(
            [
                (objective(point.at[k].add(steps[k])) - objective(point.at[k].add(-steps[k])))
                / (2 * steps[k])
                for k in range(3)
            ]
        )
        np.testing.assert_allclose(gradient, central, rtol=1e-6)
        np.testing.assert_allclose(tangent, gradient, rtol=1e-10)
    # The raw flux is linear in the drive, and the drop scales as the field squared.
    point = jnp.asarray((0.03, 0.05, 1.2))
    assert float(derivatives[drop](point)[2]) == pytest.approx(2.0 * float(drop(point)) / 1.2, rel=1e-8)


def test_invalid_meshes_and_fields_are_refused():
    with pytest.raises(ValueError, match="uniformly spaced"):
        CoreFlow(np.array([0.0, 1.0, 3.0, 4.0]))
    with pytest.raises(ValueError, match="aspect"):
        CoreFlow(np.linspace(0.0, 1.0, 5), nz=1)
    with pytest.raises(ValueError, match="stations"):
        CoreFlow(np.linspace(0.0, 1.0, 5)).solve(np.ones(4), c_t=0.1, c_s=0.1)
    with pytest.raises(ValueError, match="each of the 5 stations"):
        CoreFlow(np.linspace(0.0, 1.0, 5)).solve(np.ones(5), c_t=np.full(4, 0.1), c_s=0.1)


def test_conductances_that_vary_along_the_duct_keep_the_operator_symmetric_and_the_adjoint_exact():
    x = np.linspace(-4.0, 2.0, 25)
    model = CoreFlow(x, aspect=0.9, nz=6, ny=5)
    field = _anl(x)
    # A constant array, a callable and a number are the same conductance.
    reference = float(model.solve(field, c_t=0.03, c_s=0.05).pressure_drop)
    as_array = model.solve(field, c_t=np.full(x.size, 0.03), c_s=lambda stations: 0.05 + 0 * stations)
    assert float(as_array.pressure_drop) == pytest.approx(reference, rel=1e-13)
    c_t = 0.03 + 0.02 * field
    c_s = 0.05 * (1.0 + 0.5 * np.cos(x))
    matrix = model.operator(field, c_t=c_t, c_s=c_s)
    assert abs(matrix - matrix.T).max() / abs(matrix).max() < 1e-12

    def drop(conductances):
        return model.solve(field, c_t=conductances[0], c_s=conductances[1]).pressure_drop

    point = jnp.asarray(np.stack([c_t, c_s]))
    gradient = np.asarray(jax.jit(jax.grad(drop))(point))
    direction = np.random.default_rng(1).standard_normal(point.shape) * np.asarray(point)
    step = 1e-5
    central = (float(drop(point + step * direction)) - float(drop(point - step * direction))) / (2 * step)
    assert float(np.sum(gradient * direction)) == pytest.approx(central, rel=1e-6)
    # A Hartmann wall conducting only where the field is strong costs less than everywhere.
    assert float(drop(point)) < float(drop(jnp.full_like(point, 0.0) + jnp.asarray([[0.05], [0.075]])))


# #204, 70 x 192^2 and 70 x 256^2 on the A4000: excess of the 3-D core over [-6, 2] at c 0.1, Ha 2e4,
# and its locally fully developed drop (the 2-D sections, 0.37066).
_CORE_3D_EXCESS = (0.042087, 0.042063)
_CORE_3D_FULLY_DEVELOPED = 0.37066
# k of the side layers at c 0.1 from side_layer_coefficient (#204, 48-64 cells).
_K_TABLE = (
    (100.0, 400.0, 800.0, 1600.0, 3200.0, 1e4, 2e4),
    (2.585, 1.397, 1.242, 1.138, 1.041, 0.832, 0.636),
)


@pytest.mark.validation
def test_the_layer_corrected_model_matches_the_3d_core_at_c_01_and_ha_2e4():
    """Rows 7/24 (#204): with the layers' conductance at the local field, within 1 % of the 3-D core.

    Measured: the closure's excess is 0.042093 (frozen below Ha B = 1), 0.042000 (100) and 0.041840
    (1000) against 0.042087 and 0.042063; its fully developed drop 0.370505 against 0.37066.
    """
    hartmann, c = 2e4, 0.1
    x = np.linspace(-10.0, 6.0, 161)
    model, section = CoreFlow(x, nz=20, ny=20), CoreFlow(np.linspace(0.0, 2.0, 9), nz=20, ny=20)
    c_t, c_s = layer_conductances(c, c, hartmann, _anl(x), _K_TABLE)
    pressure = np.asarray(model.solve(_anl(x), c_t=c_t, c_s=c_s).pressure).mean(axis=1)
    drop = np.interp(-6.0, x, pressure) - np.interp(2.0, x, pressure)

    def gradient(field):
        local_t, local_s = layer_conductances(c, c, hartmann, field, _K_TABLE)
        result = section.solve(np.ones(9), c_t=float(local_t), c_s=float(local_s))
        return float(result.pressure_drop) / 2.0 * field**2

    nodes, weights = np.polynomial.legendre.leggauss(24)
    fringe = sum(
        w * gradient(_anl(np.array([s]))[0]) for w, s in zip(weights, 2.5 * nodes - 0.5, strict=True)
    )
    fully_developed = 3.0 * gradient(1.0) + 2.5 * fringe
    assert fully_developed == pytest.approx(_CORE_3D_FULLY_DEVELOPED, rel=1e-3)
    for core in _CORE_3D_EXCESS:
        assert drop - fully_developed == pytest.approx(core, rel=0.01)


@pytest.mark.physics
def test_the_side_layer_coefficient_comes_from_the_fully_developed_core():
    """#204 measured k = 1.397 at c 0.1, Ha 400 on 48 cells."""
    k = side_layer_coefficient(0.1, 400.0, cells=48)
    assert k == pytest.approx(1.397, rel=0.01)
    c_t, c_s = layer_conductances(0.1, 0.1, 400.0, 1.0, k)
    assert float(c_t) == pytest.approx(0.1025) and float(c_s) == pytest.approx(0.1 + k / 20.0)
