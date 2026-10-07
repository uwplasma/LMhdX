"""Fully developed pipe flow: the other geometry of the validation ladder."""

import jax.numpy as jnp
import numpy as np
import pytest

from lmhdx.grid import CENTER, FACE, Field, Grid, uniform_faces
from lmhdx.ops import divergence
from lmhdx.pipe import (
    PipeProblem,
    _face_currents,
    _potential,
    flow_rate,
    pipe_grid,
    pipe_problem,
    solve_pipe,
)

pytestmark = pytest.mark.numerical

# Hagen-Poiseuille in a unit pipe with unit forcing and unit viscosity: the
# profile is (1 - r^2)/4 and its mean over the section is exactly 1/8.
POISEUILLE = 0.125


def test_the_pipe_reproduces_poiseuille_without_a_field():
    """The one flow rate that is known exactly, and it fixes every normalisation at once."""
    errors = []
    for radial in (16, 32):
        velocity, _ = solve_pipe(pipe_problem(hartmann=0.0, radial=radial, azimuthal=16))
        errors.append(abs(flow_rate(velocity) - POISEUILLE) / POISEUILLE)
    assert errors[1] < 1.5e-3
    assert np.log2(errors[0] / errors[1]) > 1.9


def test_the_azimuthal_discretization_is_second_order():
    """The azimuth is uniform and periodic, so it should be clean second order."""
    rates = [
        flow_rate(solve_pipe(pipe_problem(hartmann=20.0, radial=32, azimuthal=count))[0])
        for count in (16, 32, 64)
    ]
    differences = [abs(rates[1] - rates[0]), abs(rates[2] - rates[1])]
    assert np.log2(differences[0] / differences[1]) > 1.8, differences


def test_the_field_slows_the_pipe_and_the_mesh_agrees_with_itself():
    coarse = flow_rate(solve_pipe(pipe_problem(hartmann=20.0, radial=32, azimuthal=64))[0])
    fine = flow_rate(solve_pipe(pipe_problem(hartmann=20.0, radial=48, azimuthal=64))[0])
    assert fine < 0.4 * POISEUILLE
    assert abs(coarse - fine) / fine < 0.01


def test_a_conducting_wall_short_circuits_the_pipe():
    """More wall conductance is more current returned through the wall, and less flow."""
    rates = [
        flow_rate(solve_pipe(pipe_problem(hartmann=20.0, radial=32, azimuthal=32, wall_conductance=c))[0])
        for c in (0.0, 0.027, 0.1)
    ]
    assert rates[0] > rates[1] > rates[2]


def test_the_face_currents_conserve_charge():
    """The potential equation is the divergence of the same currents the force uses."""
    problem = pipe_problem(hartmann=20.0, radial=24, azimuthal=32)
    factorization = problem.factorization()
    velocity, potential = solve_pipe(problem)
    currents = _face_currents(velocity, potential, problem)
    residual = float(jnp.max(jnp.abs(divergence(currents).data)))
    assert residual < 1e-8 * float(jnp.max(jnp.abs(velocity.data)))
    # The potential the solve reports is the one the residual was built on.
    assert (
        float(jnp.max(jnp.abs(_potential(velocity, problem, factorization)[0].data - potential.data))) < 1e-12
    )


def test_the_pipe_states_what_it_needs():
    cartesian = Grid(uniform_faces(4, 0.0, 1.0), uniform_faces(4, 0.0, 1.0), uniform_faces(4, 0.0, 1.0))
    with pytest.raises(ValueError, match="a pipe needs a polar grid"):
        PipeProblem(cartesian, 1.0)
    grid = pipe_grid(8, 8, 0.0)
    with pytest.raises(ValueError, match="hartmann must not be negative"):
        PipeProblem(grid, -1.0)
    with pytest.raises(ValueError, match="wall conductance must not be negative"):
        PipeProblem(grid, 1.0, -0.5)


def test_the_mesh_resolves_the_layer_the_field_implies():
    coarse = np.diff(pipe_grid(32, 32, 0.0).x_faces)
    layered = np.diff(pipe_grid(32, 32, 200.0).x_faces)
    assert np.allclose(coarse, coarse[0])
    assert layered.min() < 1.0 / 200.0
    assert layered[-1] < layered[0]


# Mean velocity of a unit-forced pipe from validation.pipe, a Fourier-Chebyshev
# solve on the diameter that shares no operator with the package, at the
# resolution where 47 and 63 radial points agree to better than 1e-5 relative.
SPECTRAL_FLOW_RATE = {
    (0.0, 0.0): 0.125000000,
    (5.0, 0.0): 0.089620450,
    (20.0, 0.0): 0.035346710,
    (100.0, 0.0): 0.008151880,
    (20.0, 0.1): 0.015286900,
}


@pytest.mark.parametrize(("hartmann", "radial", "azimuthal"), [(0.0, 48, 32), (5.0, 48, 64), (20.0, 48, 64)])
def test_the_pipe_matches_an_independent_spectral_solve(hartmann, radial, azimuthal):
    """The insulating pipe, against a solve that removes the axis by construction."""
    problem = pipe_problem(hartmann=hartmann, radial=radial, azimuthal=azimuthal)
    rate = flow_rate(solve_pipe(problem)[0])
    exact = SPECTRAL_FLOW_RATE[(hartmann, 0.0)]
    assert abs(rate - exact) / exact < 5e-3


def test_the_conducting_pipe_converges_to_the_reference():
    """Second order under a refinement of every length the solution has.

    Refining the radius alone at a fixed azimuth measured the thin-wall closure
    only while that closure was first order: its error dominated and shrank. The
    closure is second order now, and ``pipe_grid`` keeps a fixed number of cells
    in the ``1/Ha`` layer, so a radial-only sequence stalls on the azimuthal
    truncation and the unrefined layer -- the insulating pipe, which never sees
    the wall, stalls the same way (0.44, 0.32, 0.31 % at 24, 48, 96 radial
    cells). Radial cells, azimuthal cells and cells in the layer are therefore
    doubled together, and the bound is the pre-asymptotic one of the azimuthal
    gate. Measured: 0.99, 0.25, 0.062 % (orders 2.00, 2.00); with the first-order
    wall the same sequence gave 8.7, 3.0, 1.2 % (orders 1.51, 1.39).
    """
    exact = SPECTRAL_FLOW_RATE[(20.0, 0.1)]
    errors = []
    for radial, azimuthal, layer in ((24, 32, 3), (48, 64, 6), (96, 128, 12)):
        grid = pipe_grid(radial, azimuthal, 20.0, cells_in_layer=layer)
        errors.append(abs(flow_rate(solve_pipe(PipeProblem(grid, 20.0, 0.1))[0]) - exact) / exact)
    # The default 48-cell mesh is the middle level: within 0.5 % of the reference.
    assert errors[1] < 5e-3, errors
    assert np.log2(errors[1] / errors[2]) > 1.8, errors


def test_the_pipe_force_is_minus_the_adjoint_of_its_electromotive_force():
    """``<u, F(J)> = -<J, E(u)>`` for currents that vanish on the wall: the force does the work the currents dissipate."""
    from lmhdx.ops import cell_inner_product, face_inner_product
    from lmhdx.pipe import _WALL, _WRAP, _axial_force, _face_emf

    problem = pipe_problem(hartmann=7.0, radial=10, azimuthal=12)
    grid = problem.grid
    generator = np.random.default_rng(3)
    velocity = Field(jnp.asarray(generator.standard_normal(grid.shape)), (CENTER, CENTER, CENTER), grid)
    radial, azimuthal = (generator.standard_normal(grid.face_shape(axis)) for axis in (0, 1))
    radial[[0, -1]], azimuthal[:, -1] = 0.0, azimuthal[:, 0]
    currents = [Field(jnp.asarray(radial), (FACE, CENTER, CENTER), grid)]
    currents.append(Field(jnp.asarray(azimuthal), (CENTER, FACE, CENTER), grid))
    emf = _face_emf(velocity, problem)
    force = _axial_force((*currents, emf[2]), problem)
    work = float(cell_inner_product(velocity, velocity.replace_data(force)))
    dissipation = sum(
        float(face_inner_product(current, emf[axis], axis, condition))
        for axis, (current, condition) in enumerate(zip(currents, (_WALL, _WRAP), strict=True))
    )
    assert abs(work + dissipation) < 1e-12 * abs(dissipation)


def test_the_current_into_a_conducting_pipe_wall_exerts_no_force():
    """Both identities with the currents a conducting wall really takes, which do not vanish on the wall.

    The half-cell current into the sheet carries no electromotive force; while it exerted a force,
    the adjoint and the ohmic identity both missed by 5.1e-4 here. The wall power is taken at the
    sheet potential, as in ``core3d._wall_power``. An insulated wall is closed already: bit for bit.
    """
    from lmhdx.ops import cell_inner_product, face_average_adjoint, face_inner_product
    from lmhdx.pipe import _WALL, _WRAP, _angles, _axial_force, _face_emf

    values = np.random.default_rng(5).standard_normal((24, 32, 1))
    for conductance in (0.1, 0.0):
        problem = pipe_problem(hartmann=20.0, radial=24, azimuthal=32, wall_conductance=conductance)
        grid, conditions = problem.grid, problem.conditions
        velocity = Field(jnp.asarray(values), (CENTER,) * 3, grid)
        potential, sheets = _potential(velocity, problem, problem.factorization())
        currents = _face_currents(velocity, potential, problem, sheets)
        force = _axial_force(currents, problem)
        if not conductance:
            sine, cosine = _angles(grid)
            adjoint = [face_average_adjoint(currents[a], a, c).data for a, c in ((0, _WALL), (1, _WRAP))]
            assert bool(jnp.all(force == -20.0 * (adjoint[0] * sine + adjoint[1] * cosine)))
            continue
        emf, outward = _face_emf(velocity, problem), currents[0].data[-1]
        assert float(jnp.max(jnp.abs(outward))) > 0.1 * float(jnp.max(jnp.abs(currents[0].data)))
        work = float(cell_inner_product(velocity, velocity.replace_data(force)))
        pairing = sum(float(face_inner_product(currents[a], emf[a], a, conditions[a])) for a in range(3))
        half, area = 0.5 * float(grid.widths[0][-1]), jnp.asarray(grid.face_areas(0)[-1])
        wall = float(jnp.sum((potential.data[-1] - outward * half) * outward * area))
        joule = sum(float(face_inner_product(c, c, a, conditions[a])) for a, c in enumerate(currents))
        joule -= float(jnp.sum(outward**2 * area)) * half
        assert abs(work + pairing) < 1e-12 * abs(pairing)
        assert abs(joule + wall + work) < 1e-12 * joule


@pytest.mark.slow
def test_the_pipe_holds_at_hartmann_100():
    problem = pipe_problem(hartmann=100.0, radial=64, azimuthal=128)
    exact = SPECTRAL_FLOW_RATE[(100.0, 0.0)]
    assert abs(flow_rate(solve_pipe(problem)[0]) - exact) / exact < 5e-3


@pytest.mark.slow
def test_the_spectral_reference_returns_what_is_cached():
    """Poiseuille exactly at zero field, and the cached values it was compared against."""
    from validation.pipe import flow_rate as spectral

    assert spectral(0.0, 31, 32) == pytest.approx(0.125, rel=1e-12)
    for (hartmann, conductance), cached in SPECTRAL_FLOW_RATE.items():
        value = spectral(hartmann, 47, 48, wall_conductance=conductance)
        assert value == pytest.approx(cached, rel=1e-5), (hartmann, conductance)


def _open_pipe(field, wall_conductance=0.05):
    from lmhdx.axial import fringe_pipe

    return fringe_pipe(
        hartmann=20.0,
        field=field,
        wall_conductance=wall_conductance,
        lower=-4.0,
        upper=4.0,
        core=(-2.0, 2.0),
        spacing=0.5,
        radial=10,
        azimuthal=12,
        cells_in_layer=3,
    )


def _fringe(x):
    return 0.5 * (1.0 - np.tanh(x))


def test_the_open_pipe_operator_is_symmetric_and_conserves_charge_and_energy():
    """Row 6 machinery: -A symmetric definite in the face volume; the Lorentz work is minus the Joule heat."""
    from lmhdx import axial

    pipe = _open_pipe(_fringe)
    metric, solvers = axial._pipe_metric(pipe), axial._pipe_solvers(pipe)
    rng = np.random.default_rng(0)
    shapes = (pipe.grid.face_shape(0), pipe.grid.shape, pipe.grid.face_shape(2))

    def sample():
        velocity = tuple(jnp.asarray(rng.standard_normal(shape)) for shape in shapes)
        return axial._pipe_project(velocity, pipe, metric, solvers[0])[0]

    def dot(first, second, weights=metric["weights"]):
        return sum(float(jnp.sum(w * a * b)) for w, a, b in zip(weights, first, second, strict=True))

    first, second = sample(), sample()
    assert float(jnp.max(jnp.abs(axial._pipe_divergence(first, metric)))) < 1e-10
    applied = [axial._pipe_residual(u, pipe, metric, solvers, 1.0)[0] for u in (first, second)]
    assert abs(dot(first, applied[1]) - dot(applied[0], second)) < 1e-12 * abs(dot(first, applied[1]))
    assert dot(first, applied[0]) < 0.0
    _, currents, force = axial._pipe_electric(first, pipe, metric, solvers[1], 1.0)
    net = axial._pipe_divergence(currents, metric) * metric["volume"]
    assert float(jnp.max(jnp.abs(net))) < 1e-12 * float(jnp.max(jnp.abs(currents[0] * metric["area"][0])))
    motional = axial._pipe_motional(first, pipe, metric, 1.0)
    joule = -dot(currents, motional, axial._pipe_current_weights(metric))
    assert dot(first, force) == pytest.approx(joule, rel=1e-12)


def test_the_uniform_open_pipe_is_the_fully_developed_pipe():
    """Under a uniform field the 3-D open pipe keeps the 2-D pipe's profile and gradient to round-off."""
    from lmhdx.axial import pipe_station_pressure, solve_open_pipe

    pipe = _open_pipe(np.ones_like)
    solution = solve_open_pipe(pipe)
    np.testing.assert_allclose(
        np.asarray(solution.velocity[2]),
        np.broadcast_to(pipe.inlet[:, :, None], pipe.grid.face_shape(2)),
        atol=1e-10,
    )
    section = Grid(
        pipe.grid.x_faces, pipe.grid.y_faces, uniform_faces(1, 0.0, 1.0), geometry=pipe.grid.geometry
    )
    profile, _ = solve_pipe(PipeProblem(section, 20.0, 0.05))
    unit = float(np.sum(np.asarray(profile.data)[:, :, 0] * section.face_areas(2)[:, :, 0]))
    centres, means = pipe_station_pressure(pipe, solution.pressure)
    assert np.polyfit(centres, np.asarray(means), 1)[0] == pytest.approx(-np.pi / unit, rel=1e-10)


def test_the_open_pipe_balances_mass_and_its_drop_adjoint_matches_central_differences():
    import jax

    from lmhdx.axial import _pipe_divergence, _pipe_metric, pipe_station_pressure, solve_open_pipe

    pipe = _open_pipe(_fringe)
    metric = _pipe_metric(pipe)
    solution = solve_open_pipe(pipe)
    flows = jnp.sum(solution.velocity[2] * metric["area"][2], axis=(0, 1))
    assert float(jnp.max(jnp.abs(flows - np.pi))) < 1e-12 * np.pi
    assert float(jnp.max(jnp.abs(_pipe_divergence(solution.velocity, metric) * metric["volume"]))) < 1e-12

    def drop(scale):
        centres, means = pipe_station_pressure(pipe, solve_open_pipe(pipe, field_scale=scale).pressure)
        return jnp.interp(-3.0, centres, means) - jnp.interp(3.0, centres, means)

    gradient = float(jax.grad(drop)(1.0))
    central = (float(drop(1.0 + 1e-4)) - float(drop(1.0 - 1e-4))) / 2e-4
    assert gradient == pytest.approx(central, rel=1e-6)


def test_the_tabulated_field_is_monotone_and_not_extrapolated():
    from lmhdx.axial import monotone_interpolant

    table = np.array([-2.0, 0.0, 1.0, 3.0]), np.array([1.0, 0.8, 0.1, 0.0])
    field = monotone_interpolant(*table)
    points = np.linspace(-2.0, 3.0, 201)
    values = field(points)
    np.testing.assert_allclose(field(table[0]), table[1], atol=1e-14)
    assert np.all(np.diff(values) <= 1e-14) and values.min() >= 0.0
    with pytest.raises(ValueError, match="not extrapolated"):
        field([3.5])
