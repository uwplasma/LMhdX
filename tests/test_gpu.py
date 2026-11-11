"""Small-grid smoke checks for an accelerator; each runs in seconds and passes on CPU too."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmhdx.core3d import RK3, ChannelProblem, advance, kinetic_energy, zero_velocity
from lmhdx.grid import (
    CENTER,
    DIRICHLET,
    PERIODIC,
    BoundaryCondition,
    Field,
    Grid,
    uniform_faces,
)
from lmhdx.ops import laplacian
from lmhdx.poisson import fast_diagonal_poisson

pytestmark = pytest.mark.gpu

FIXED = BoundaryCondition(DIRICHLET)
WRAPPED = BoundaryCondition(PERIODIC)
GRID = Grid(*(uniform_faces(8, 0.0, 1.0) for _ in range(3)))


def _random_cells(seed: int) -> Field:
    data = jax.random.normal(jax.random.PRNGKey(seed), GRID.shape, dtype=jnp.float64)
    return Field(data, (CENTER,) * 3, GRID)


def test_arrays_live_on_the_default_backend():
    value = jnp.ones((4,)) * 2.0
    assert value.devices().pop().platform == jax.default_backend()


@pytest.mark.parametrize(
    "conditions", [(FIXED,) * 3, (WRAPPED, WRAPPED, FIXED)], ids=["dirichlet", "periodic"]
)
def test_direct_poisson_solve_inverts_the_operator(conditions):
    exact = _random_cells(3)
    rhs = laplacian(exact, conditions)
    factorization = fast_diagonal_poisson(GRID, conditions)
    solution = factorization.solve(rhs)
    assert float(jnp.max(jnp.abs(solution.data - exact.data))) < 1e-10


def test_poisson_gradient_matches_a_central_difference_under_jit():
    conditions = (FIXED,) * 3
    factorization = fast_diagonal_poisson(GRID, conditions)

    def energy(values):
        return jnp.sum(factorization.solve(Field(values, (CENTER,) * 3, GRID)).data ** 2)

    values = _random_cells(17).data
    direction = _random_cells(19).data
    step = 1.0e-6
    gradient = jax.jit(jax.grad(energy))(values)
    difference = (energy(values + step * direction) - energy(values - step * direction)) / (2.0 * step)
    assert abs(float(jnp.vdot(gradient, direction)) - float(difference)) < 1e-6 * abs(float(difference))


def test_rk3_steps_of_a_viscous_periodic_box_dissipate_energy():
    faces = uniform_faces(8, 0.0, 2.0 * np.pi)
    grid = Grid(faces, faces, uniform_faces(4, 0.0, 1.0))
    problem = ChannelProblem(grid, (WRAPPED,) * 3, viscosity=0.05, advection="central", dt=0.02)
    xf = np.asarray(faces)[:, None]
    yf = np.asarray(faces)[None, :]
    corner = np.sin(xf) * np.sin(2.0 * yf)
    u = np.diff(corner, axis=1) / np.diff(yf, axis=1)
    v = -np.diff(corner, axis=0) / np.diff(xf, axis=0)
    zero = zero_velocity(problem)
    start = (
        zero[0].replace_data(jnp.asarray(u[:, :, None] * np.ones(zero[0].data.shape))),
        zero[1].replace_data(jnp.asarray(v[:, :, None] * np.ones(zero[1].data.shape))),
        zero[2],
    )
    before = float(kinetic_energy(start, problem))
    run = advance(problem, 5, start, viscous=problem.viscous_factorizations(RK3), scheme=RK3)
    assert 0.0 < float(kinetic_energy(run.velocity, problem)) < before
    assert float(jnp.max(jnp.abs(run.divergence_residual))) < 1e-8
