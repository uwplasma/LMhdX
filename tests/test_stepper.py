"""The IMEX Runge-Kutta stepper (plan H2.2, H2.8): stability, temporal order, budget, growth, derivative.

The constrained stage it is built from is tested in ``test_core3d.py``; this file asks what the three
stages add. Every check has a closed form or an independent reference: the amplification polynomial of
the stage map, a manufactured flow, an exactly damped mode, the Orr-Sommerfeld eigenvalue, the energy
ledger, and a Taylor remainder.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmhdx.core3d import (
    EULER,
    RK3,
    ChannelProblem,
    advance,
    courant_step,
    energy_budget,
    enforce_face_constraints,
    kinetic_energy,
    step,
    velocity_offset,
    zero_velocity,
)
from lmhdx.grid import NEUMANN, PERIODIC, BoundaryCondition, Field, Grid, tanh_faces, uniform_faces

pytestmark = pytest.mark.numerical

PERIODIC_X = BoundaryCondition(PERIODIC)
WALL = BoundaryCondition(NEUMANN)


def _rk_box(cells: int = 16, **overrides) -> ChannelProblem:
    """A periodic box with central transport; the transported modes below live on ``x`` and ``y``."""
    faces = uniform_faces(cells, 0.0, 2.0 * np.pi)
    grid = Grid(faces, faces, uniform_faces(4, 0.0, 1.0))
    return ChannelProblem(
        grid, (PERIODIC_X,) * 3, **({"viscosity": 0.05, "advection": "central"} | overrides)
    )


def _smooth_vortex(problem: ChannelProblem):
    """A solenoidal field ``psi = sin x sin 2y + cos(2x + y) / 2``, exact on the staggered grid through its corners."""
    xf, yf = (
        np.asarray(f)[:, None] if a == 0 else np.asarray(f)[None, :]
        for a, f in enumerate(problem.grid.faces[:2])
    )
    corner = np.sin(xf) * np.sin(2.0 * yf) + 0.5 * np.cos(2.0 * xf + yf)
    u = np.diff(corner, axis=1) / np.diff(yf, axis=1)
    v = -np.diff(corner, axis=0) / np.diff(xf, axis=0)
    zero = zero_velocity(problem)
    return (
        zero[0].replace_data(jnp.asarray(u[:, :, None] * np.ones(zero[0].data.shape))),
        zero[1].replace_data(jnp.asarray(v[:, :, None] * np.ones(zero[1].data.shape))),
        zero[2],
    )


def _orders(values) -> np.ndarray:
    values = np.abs(np.asarray(values))
    return np.log2(values[:-1] / values[1:])


def test_central_transport_is_stable_under_rk3_up_to_its_stage_map_limit():
    """A shear mode carried by a uniform stream has ``|g|^2 = 1 - C^4/12 + C^6/36`` per RK3 step.

    The mode sits at ``k h = pi/2`` and the stream speed is the Courant number ``C`` (the step is one cell
    width), so one compiled run serves them all. The polynomial is the three-stage stage map at ``z = -iC``:
    below one up to ``C = sqrt 3`` (0.9976, 0.9718, 0.9591 at 0.5, 1, 1.6) and above it past (1.105 at 1.9).
    Forward Euler has ``|g|^2 = 1 + C^2``, growth at every Courant number (plan H.10.1, item 4; T41); it is run
    to ``C = 1``, past which the mode overflows within the run.
    """
    steps, cells = 40, 16
    problem = _rk_box(viscosity=1e-12, dt=2.0 * np.pi / cells)
    x = np.asarray(problem.grid.centers[0])

    def growth(scheme, stream):
        velocity = zero_velocity(problem)
        mode = 1e-6 * np.cos(4 * x[:, None, None]) * np.ones(velocity[1].data.shape)
        start = (
            velocity[0].replace_data(stream * jnp.ones_like(velocity[0].data)),
            velocity[1].replace_data(jnp.asarray(mode)),
            velocity[2],
        )
        run = advance(problem, steps, start, viscous=problem.viscous_factorizations(scheme), scheme=scheme)
        return jnp.abs(jnp.mean(run.velocity[1].data[:, 0, 0] * jnp.exp(-4j * x)))

    for scheme, courants in ((RK3, (0.5, 1.0, 1.6, 1.9)), (EULER, (0.5, 1.0))):
        measure = jax.jit(lambda stream, scheme=scheme: growth(scheme, stream))
        for courant in courants:
            per_step = (2.0 * float(measure(courant)) / 1e-6) ** (1.0 / steps)
            square = (1.0 - courant**4 / 12.0 + courant**6 / 36.0) if scheme is RK3 else 1.0 + courant**2
            assert per_step == pytest.approx(np.sqrt(square), rel=1e-9)
            assert (per_step < 1.0) == (scheme is RK3 and courant < RK3.limit)


@pytest.mark.parametrize(("scheme", "steps"), [(EULER, 32), (RK3, 8)], ids=["euler", "rk3"])
def test_limited_transport_never_gains_energy(scheme, steps):
    """The van Leer blend dissipates where the central flux under Euler gains energy (same flow as below).

    The unsteady inviscid box loses 37 % of its energy to the limiter on 16 cells, under Euler at Courant
    0.2 and under RK3 at 0.9, and gains none.
    """
    problem = _rk_box(viscosity=1e-12, advection="limited", dt=2.0 / steps)
    start = _smooth_vortex(problem)
    run = advance(problem, steps, start, viscous=problem.viscous_factorizations(scheme), scheme=scheme)
    before = kinetic_energy(start, problem)
    assert -0.5 < float((kinetic_energy(run.velocity, problem) - before) / before) < 0.0


def test_the_courant_step_is_the_stage_map_limit_over_the_peak_rate():
    problem = _rk_box()
    velocity = zero_velocity(problem)
    velocity = (velocity[0].replace_data(jnp.ones_like(velocity[0].data)), *velocity[1:])
    assert courant_step(velocity, RK3, 1.0) == pytest.approx(np.sqrt(3.0) * 2.0 * np.pi / 16)
    with pytest.raises(ValueError, match="unstable"):
        courant_step(velocity, EULER)


def _shatrov_velocity(point, t):
    """Shatrov and Gerbeth's divergence-free, no-slip manufactured flow (``Re`` 100, ``Omega`` 10), written once."""
    x, y, z = point
    pulse, pi = jnp.sin(10.0 * t), jnp.pi
    return jnp.array(
        [
            pi * jnp.sin(x) * jnp.sin(2 * pi * y) * jnp.sin(2 * pi * z) * pulse,
            -0.5 * jnp.cos(x) * jnp.sin(pi * y) ** 2 * jnp.sin(2 * pi * z) * pulse,
            -0.5 * jnp.cos(x) * jnp.sin(2 * pi * y) * jnp.sin(pi * z) ** 2 * pulse,
        ]
    )


def _on_faces(grid: Grid, function) -> tuple[Field, Field, Field]:
    """``function(point)`` sampled where each velocity component lives (faces along its axis, centres across)."""
    fields = []
    for component in range(3):
        axes = [np.asarray(grid.faces[a] if a == component else grid.centers[a]) for a in range(3)]
        points = jnp.asarray(np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3))
        values = jax.vmap(lambda p, c=component: function(p)[c])(points).reshape([len(a) for a in axes])
        fields.append(Field(values, velocity_offset(component), grid))
    return tuple(fields)


def _shatrov_run(cells: int, steps: int, scheme):
    """The flow is ``V(x) sin(Omega t)``, so its force ``dv/dt + v . grad v - nu lap v`` is three fixed fields
    weighted by ``cos``, ``sin^2`` and ``sin``: sampled once, outside the compiled run. The pressure gradient is
    left to the projection."""
    stretched = tanh_faces(cells, -1.0, 1.0, 2.0)
    grid = Grid(uniform_faces(cells, 0.0, 2.0 * np.pi), stretched, stretched)
    problem = ChannelProblem(
        grid, (PERIODIC_X, WALL, WALL), viscosity=0.01, advection="central", dt=0.1 / steps
    )
    shape = lambda p: _shatrov_velocity(p, jnp.pi / 20.0)  # noqa: E731 -- sin(Omega t) = 1 here
    inertia = _on_faces(grid, lambda p: 10.0 * shape(p))
    transport = _on_faces(grid, lambda p: jax.jacfwd(shape)(p) @ shape(p))
    friction = _on_faces(grid, lambda p: -0.01 * jnp.einsum("iaa->i", jax.hessian(shape)(p)))

    def force(t):
        weights = (jnp.cos(10.0 * t), jnp.sin(10.0 * t) ** 2, jnp.sin(10.0 * t))
        return tuple(
            Field(
                sum(w * f[c].data for w, f in zip(weights, (inertia, transport, friction))),
                f_off.offset,
                grid,
            )
            for c, f_off in enumerate(inertia)
        )

    return advance(problem, steps, viscous=problem.viscous_factorizations(scheme), scheme=scheme, force=force)


@pytest.mark.parametrize(
    ("scheme", "steps", "low", "high"),
    [(RK3, (4, 8, 16), 2.0, 3.5), (EULER, (16, 32, 64), 0.9, 1.15)],
    ids=["rk3", "euler"],
)
def test_the_stepper_converges_in_time_on_the_shatrov_manufactured_flow(scheme, steps, low, high):
    """Plan T28: the 3-D no-slip flow on tanh-stretched 12^3 cells, ``Re`` 100, ``t = 0.1``, forcing from the flow.

    The spatial error is common to the three runs, so the differences between them are the time error: RK3 with
    Crank-Nicolson viscous terms is second order, 2.32 (transport alone would be third), Euler first, 0.97
    (coarser steps approach it from below: 0.86, 0.94). The RK3 solution at 16 steps stays within the spatial
    error of the analytic flow, 0.06.
    """
    runs = [_shatrov_run(12, count, scheme) for count in steps]
    differences = [
        max(float(jnp.max(jnp.abs(a.data - b.data))) for a, b in zip(coarse.velocity, fine.velocity))
        for coarse, fine in zip(runs[:-1], runs[1:])
    ]
    order = float(_orders(differences)[0])
    assert low < order < high, (differences, order)
    if scheme is RK3:
        exact = _on_faces(runs[-1].velocity[0].grid, lambda p: _shatrov_velocity(p, 0.1))
        assert max(float(jnp.max(jnp.abs(f.data - e.data))) for f, e in zip(runs[-1].velocity, exact)) < 0.08
        assert float(jnp.max(runs[-1].divergence_residual)) < 1e-13


@pytest.mark.parametrize(
    ("scheme", "low", "high"), [(RK3, 1.95, 2.05), (EULER, 1.0, 1.2)], ids=["rk3", "euler"]
)
def test_a_damped_shear_mode_decays_at_the_discrete_rate_to_the_schemes_order(scheme, low, high):
    """``u_x = cos(2 y)`` across a field along ``y``: ``exp(-(nu k_h^2 + sigma B^2/rho) t)``, ``k_h`` the grid's.

    Crank-Nicolson viscous terms and the implicit damping give RK3 second order here (the stage weights
    reproduce ``e^z`` to ``z^2`` for a linear operator); Euler is first order.
    """
    cells, wavenumber, viscosity, field = 16, 2, 0.05, 2.0
    h = 2.0 * np.pi / cells
    rate = viscosity * (2.0 - 2.0 * np.cos(wavenumber * h)) / h**2 + field**2
    errors = []
    for steps in (4, 8, 16):
        faces = uniform_faces(cells, 0.0, 2.0 * np.pi)
        grid = Grid(uniform_faces(4, 0.0, 1.0), faces, uniform_faces(4, 0.0, 1.0))
        problem = ChannelProblem(
            grid, (PERIODIC_X,) * 3, viscosity=viscosity, magnetic_field=(0.0, field, 0.0), dt=1.0 / steps,
            advection="central",
        )  # fmt: skip
        y = np.asarray(grid.centers[1])[None, :, None]
        start = zero_velocity(problem)
        start = (
            start[0].replace_data(jnp.asarray(np.cos(wavenumber * y) * np.ones(start[0].data.shape))),
            *start[1:],
        )
        run = advance(problem, steps, start, viscous=problem.viscous_factorizations(scheme), scheme=scheme)
        exact = np.exp(-rate) * np.cos(wavenumber * y[0, :, 0])
        errors.append(float(jnp.max(jnp.abs(run.velocity[0].data[0, :, 0] - exact))))
    orders = _orders(errors)
    assert np.all((low < orders) & (orders < high)), (errors, orders)


def test_central_transport_conserves_kinetic_energy_to_the_third_order_under_rk3():
    """Plan T24: an unsteady inviscid periodic box. The energy change falls as ``dt^3`` (2.95, 2.99).

    Forward Euler with the same central flux gains energy at first order: +8.7e-3 of the energy at 16 steps.
    """

    def change(scheme, steps):
        problem = _rk_box(viscosity=1e-12, dt=2.0 / steps)
        start = _smooth_vortex(problem)
        run = advance(problem, steps, start, viscous=problem.viscous_factorizations(scheme), scheme=scheme)
        before = kinetic_energy(start, problem)
        return float((kinetic_energy(run.velocity, problem) - before) / before)

    rk3 = [change(RK3, steps) for steps in (16, 32, 64)]
    assert all(c < 0.0 for c in rk3) and np.all(np.abs(_orders(rk3) - 3.0) < 0.2), (rk3, _orders(rk3))
    assert change(EULER, 16) > 1e-3


@pytest.mark.parametrize(
    ("scheme", "steps", "low", "high"),
    [(RK3, (8, 16, 32), 1.9, 2.1), (EULER, (8, 16), 0.9, 1.2)],
    ids=["rk3", "euler"],
)
def test_the_kinetic_energy_budget_closes_to_the_schemes_order(scheme, steps, low, high):
    """Plan T34: drive, Lorentz damping and viscosity on, central transport (which does no net work).

    ``K(t) - K(0) - integral of (drive + Lorentz - viscous)``, the trapezoid rule over the step ends, falls as
    ``dt^2`` for RK3 (1.998, 1.9996) and ``dt`` for Euler (1.04); ``energy_budget`` is the existing ledger.
    """
    errors = []
    for count in steps:
        problem = _rk_box(dt=1.0 / count, magnetic_field=(0.0, 1.5, 0.0), forcing=(0.3, 0.0, 0.0))
        factorization, viscous = problem.factorization(), problem.viscous_factorizations(scheme)

        @jax.jit
        def advanced(velocity):
            moved = step(velocity, problem, factorization, viscous, scheme=scheme)[0]
            return moved, kinetic_energy(moved, problem), energy_budget(moved, problem, factorization).defect

        velocity = _smooth_vortex(problem)
        start = kinetic_energy(velocity, problem)
        energy, power = start, [energy_budget(velocity, problem, factorization).defect]
        for _ in range(count):
            velocity, energy, defect = advanced(velocity)
            power.append(defect)
        errors.append(abs(float(energy - start) - np.trapezoid(np.asarray(power), dx=1.0 / count)))
    orders = _orders(errors)
    assert np.all((low < orders) & (orders < high)), (errors, orders)


def _orr_sommerfeld_mode(reynolds: float, alpha: float, nodes: int = 80):
    """Least-damped, here unstable, Orr-Sommerfeld mode of plane Poiseuille flow: its ``c`` and ``phi(y)`` as a Chebyshev series.

    Chebyshev collocation with clamped rows, ``(U - c)(phi'' - alpha^2 phi) - U'' phi = (phi'''' - 2 alpha^2 phi''
    + alpha^4 phi)/(i alpha Re)``; the eigenvalue nearest the tabulated ``c`` is taken.
    """
    from numpy.polynomial import chebyshev as cheb
    from scipy.linalg import eig

    k = np.arange(nodes + 1)
    y = np.cos(np.pi * k / nodes)
    weight = np.where((k == 0) | (k == nodes), 2.0, 1.0) * (-1.0) ** k
    spacing = y[:, None] - y[None, :]
    d1 = np.outer(weight, 1.0 / weight) / (spacing + np.eye(nodes + 1))
    d1 -= np.diag(d1.sum(axis=1))
    d2, eye = d1 @ d1, np.eye(nodes + 1)
    laplace = d2 - alpha**2 * eye
    operator = (
        (1 - y**2)[:, None] * laplace
        + 2 * eye
        - (d2 @ d2 - 2 * alpha**2 * d2 + alpha**4 * eye) / (1j * alpha * reynolds)
    )
    b = laplace.astype(complex)
    for row, equation in ((0, eye[0]), (1, d1[0]), (nodes - 1, d1[nodes]), (nodes, eye[nodes])):
        operator[row], b[row] = equation, 0.0
    values, vectors = eig(operator, b)
    mode = np.argmin(np.abs(values - (0.247075 + 0.00266441j)))
    return values[mode], cheb.chebfit(y[::-1], vectors[::-1, mode], nodes)


def _orr_sommerfeld_rate(cells: tuple[int, int], dt: float, horizon: float = 60.0) -> float:
    """Growth rate of the amplitude of ``v`` over the last third of ``horizon``, for the Re 8000, ``alpha`` 1 mode seeded at 1e-5."""
    from numpy.polynomial import chebyshev as cheb

    reynolds, nu = 8000.0, 1.0 / 8000.0
    c, series = _orr_sommerfeld_mode(reynolds, 1.0)
    grid = Grid(
        uniform_faces(cells[0], 0.0, 2.0 * np.pi),
        tanh_faces(cells[1], -1.0, 1.0, 2.75),
        uniform_faces(2, 0.0, 1.0),
    )
    problem = ChannelProblem(
        grid, (PERIODIC_X, WALL, PERIODIC_X), viscosity=nu, advection="central", forcing=(2.0 * nu, 0.0, 0.0), dt=dt
    )  # fmt: skip
    xf, yf = np.asarray(grid.faces[0]), np.asarray(grid.faces[1])
    # The discrete Poiseuille profile (nu times the discrete second difference of U equals minus the drive).
    widths, spacing = np.asarray(grid.widths[1]), np.diff(np.asarray(grid.centers[1]))
    lower, upper = np.append(0.0, 1.0 / spacing), np.append(1.0 / spacing, 0.0)
    lower[0], upper[-1] = 2.0 / widths[0], 2.0 / widths[-1]
    second = (np.diag(-(lower + upper)) + np.diag(upper[:-1], 1) + np.diag(lower[1:], -1)) / widths[:, None]
    base = np.linalg.solve(nu * second, -2.0 * nu * np.ones(cells[1]))
    corner = np.real(cheb.chebval(yf, series)[None, :] * np.exp(1j * xf)[:, None])
    u = np.diff(corner, axis=1) / np.diff(yf)[None, :]
    v = -np.diff(corner, axis=0) / np.diff(xf)[:, None]
    zero = zero_velocity(problem)
    state = (
        zero[0].replace_data(
            jnp.asarray(base[None, :, None] + 1e-5 * u[:, :, None] * np.ones(zero[0].data.shape))
        ),
        zero[1].replace_data(jnp.asarray(1e-5 * v[:, :, None] * np.ones(zero[1].data.shape))),
        zero[2],
    )
    per_chunk, amplitudes = round(horizon / dt / 3), []
    viscous = problem.viscous_factorizations(RK3)
    chunk = jax.jit(lambda start: advance(problem, per_chunk, start, viscous=viscous, scheme=RK3).velocity)
    for _ in range(3):
        state = chunk(state)
        amplitudes.append(float(jnp.linalg.norm(state[1].data)))
    return float(np.log(amplitudes[2] / amplitudes[1]) / (per_chunk * dt))


def test_rk3_grows_the_orr_sommerfeld_mode_at_its_rate_and_not_at_the_step():
    """Plan T25, the linear-growth gate that stands in for T29's duct: plane Poiseuille, Re 8000, ``alpha`` 1.

    The tabulated amplitude growth is 2.66441e-3. On 32 x 48 tanh cells (``gamma`` 2.75) around the
    discrete Poiseuille profile, read over t = 40 to 60, the stepper gives 2.6626e-3 at ``dt`` = 0.125
    (-0.07 %) and 2.6634e-3 at 0.25: the step moves it by 8.7e-7. Other meshes (``x`` by ``y`` cells, % from the
    table): 16 x 48 -4.9; 32 x 96 -0.67; 32 x 192 -1.06; 64 x 96 +0.29; 64 x 192 -0.13; 64 x 384 -0.25
    (second order in each direction; 16 x 24 is under-resolved, +110). At ``dt`` = 0.5 (Courant 2.6,
    past the stage map's ``sqrt 3``) the run does not stay finite.
    """
    reference = _orr_sommerfeld_mode(8000.0, 1.0)[0].imag
    fine, coarse = (_orr_sommerfeld_rate((32, 48), dt) for dt in (0.125, 0.25))
    assert fine == pytest.approx(reference, rel=0.005)
    assert abs(coarse - fine) < 5e-6


def test_rk3_gives_the_derivative_standard_through_a_short_window():
    """Plan H.10.6 on the stepper: JVP/VJP dot-product identity and a Taylor remainder of slope two.

    Three RK3 steps with implicit viscosity, walls across ``y`` and ``z``, an oblique field and central
    transport; the leaves are the initial velocity and the drive. The identity holds to round-off
    (``<J d, w> = <d, J^T w>``), the remainder of the tangent falls with the square of the step.
    """
    grid = Grid(uniform_faces(4, 0.0, 1.0), tanh_faces(6, -1.0, 1.0, 1.3), uniform_faces(5, -1.0, 1.0))
    problem = ChannelProblem(
        grid, (PERIODIC_X, WALL, WALL), viscosity=0.2, magnetic_field=(0.3, 1.0, 0.5), advection="central", dt=0.02
    )  # fmt: skip
    viscous, factorization = problem.viscous_factorizations(RK3), problem.factorization()
    rng = np.random.default_rng(5)
    shapes = [grid.offset_shape(velocity_offset(c)) for c in range(3)]
    leaves = [rng.normal(size=shape) for shape in shapes] + [1.0]
    directions = [rng.normal(size=shape) for shape in shapes] + [0.7]

    def final(*parameters):
        *data, drive = parameters
        start = enforce_face_constraints(
            tuple(Field(d, velocity_offset(c), grid) for c, d in enumerate(data)), problem
        )
        driven = dataclasses.replace(problem, forcing=(drive, 0.0, 0.0))
        run = advance(driven, 3, start, factorization=factorization, viscous=viscous, scheme=RK3)
        return tuple(component.data for component in run.velocity)

    leaves, directions = [jnp.asarray(x) for x in leaves], [jnp.asarray(x) for x in directions]
    final = jax.jit(final)
    values, tangent = jax.jvp(final, leaves, directions)
    _, pullback = jax.vjp(final, *leaves)
    weights = [jnp.asarray(rng.normal(size=v.shape)) for v in values]
    forward = sum(float(jnp.vdot(w, t)) for w, t in zip(weights, tangent))
    backward = sum(float(jnp.vdot(d, p)) for d, p in zip(directions, pullback(tuple(weights))))
    assert forward == pytest.approx(backward, rel=1e-11)
    remainders = []
    for h in (1e-2, 5e-3, 2.5e-3):
        moved = final(*[x + h * d for x, d in zip(leaves, directions)])
        remainders.append(
            max(float(jnp.max(jnp.abs(m - v - h * t))) for m, v, t in zip(moved, values, tangent))
        )
    slopes = np.log2(np.asarray(remainders[:-1]) / np.asarray(remainders[1:]))
    assert np.all(np.abs(slopes - 2.0) < 0.1), (remainders, slopes)
