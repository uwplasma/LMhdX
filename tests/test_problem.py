"""The 2.0 facade (plan H4.1): Problem, the presets and solve agree with the kernels they wrap.

Every equivalence test builds the case twice, once through the older entry point and once
through ``Problem``, and compares the arrays bit for bit. The derivative tests differentiate
the leaves of a ``Problem`` with Taylor remainders, central differences and the JVP/VJP pairing.
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lmhdx
from lmhdx import Open, Problem, Sides, Solver, Thermal, Uniform, Wall, duct
from lmhdx.axial import fringe_duct as old_fringe_duct
from lmhdx.axial import pressure_drop, solve_open_duct
from lmhdx.cases import WallLayer
from lmhdx.core3d import ChannelProblem, advance, duct_problem
from lmhdx.fields import Fringe, Func, Table
from lmhdx.grid import CENTER, NEUMANN, PERIODIC, BoundaryCondition, Field, Grid, axis, uniform_faces
from lmhdx.pipe import pipe_problem, solve_pipe
from lmhdx.problem import _BKZ_DOMAINS, bulk_ratio, channel, fringe_duct, pipe, q2d
from lmhdx.q2d import evolve_q2d, make_q2d_case, solve_q2d
from lmhdx.thermal import (
    ThermalProblem,
    advance_thermal,
    build_thermal_step,
    initial_thermal_state,
    thermal_boundary,
)

pytestmark = pytest.mark.numerical


def _same(first, second) -> bool:
    return all(
        np.array_equal(a, b) for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True)
    )


def test_importing_the_facade_changes_no_jax_configuration():
    """The new modules flip no x64, pin no precision and open no cache; ``configure`` does it, explicitly."""
    code = """
import jax
import lmhdx, lmhdx.fields, lmhdx.problem, lmhdx.solver
assert not jax.config.jax_enable_x64
assert jax.config.jax_default_matmul_precision is None
assert jax.config.jax_compilation_cache_dir is None
lmhdx.configure(dtype="float64")
assert jax.config.jax_enable_x64 and jax.config.jax_default_matmul_precision == "highest"
try:
    lmhdx.configure(dtype="float16")
except ValueError:
    pass
else:
    raise AssertionError
"""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION")
        and not key.startswith(("JAX_COMPILATION_CACHE", "JAX_PERSISTENT_CACHE"))
    }
    environment["LMHDX_COMPILATION_CACHE"] = "0"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120, env=environment)


def test_axis_is_one_dispatcher_over_the_face_constructors():
    from lmhdx.grid import geometric_faces, tanh_faces, wall_resolving_faces

    assert np.array_equal(axis(8, -1, 1), uniform_faces(8, -1, 1))
    assert np.array_equal(axis(8, -1, 1, beta=2.0), tanh_faces(8, -1, 1, 2.0))
    assert np.array_equal(axis(8, 0, 1, ratio=1.2), geometric_faces(8, 0, 1, 1.2))
    assert np.array_equal(
        axis(24, -1, 1, layer_thickness=0.1, cells_in_layer=4, max_ratio=None),
        wall_resolving_faces(24, -1, 1, layer_thickness=0.1, cells_in_layer=4, max_ratio=None),
    )
    with pytest.raises(ValueError, match="one of"):
        axis(8, -1, 1, beta=2.0, ratio=1.2)
    with pytest.raises(ValueError, match="blend"):
        axis(8, -1, 1, blend=1.5)


def test_the_kzb_blend_reproduces_the_published_wall_cells():
    """Krasnov, Zikanov and Boeck 2012: z = 0.96 sin(pi xi / 2) + 0.04 xi on 768 cells (plan T7)."""
    widths = np.diff(axis(768, -1.0, 1.0, blend=0.96))
    assert widths.min() == pytest.approx(1.12e-4, rel=1e-2)
    assert widths.max() == pytest.approx(4.03e-3, rel=1e-2)
    faces = axis(768, -1.0, 1.0, blend=0.96)
    assert faces[0] == -1.0 and faces[-1] == 1.0 and np.allclose(faces, -faces[::-1], atol=1e-15)
    assert np.allclose(axis(8, 0.0, 2.0, blend=1e-12), uniform_faces(8, 0.0, 2.0), atol=1e-9)


@pytest.mark.parametrize("conductance", [0.0, 0.05])
def test_the_duct_preset_is_the_older_duct_bit_for_bit(conductance):
    new = duct(20, 32, wall_conductance=conductance)
    old = duct_problem(hartmann=20.0, cells=32, wall_conductance=conductance)
    assert all(np.array_equal(a, b) for a, b in zip(new.grid.faces, old.grid.faces))
    found, expected = lmhdx.solve(new), lmhdx.solve(old)
    assert _same(
        (found.velocity, found.pressure, found.potential),
        (expected.velocity, expected.pressure, expected.potential),
    )
    assert found.pressure_gradient == 1.0
    assert float(found.residual) == float(expected.residual_norm)


def test_resolved_walls_are_the_older_wall_layers():
    new = duct(
        10,
        32,
        walls=dict(
            y=Wall(layers=(WallLayer("lithium", 0.5, 0.1, 4),)),
            z=Wall(layers=(WallLayer("steel", 1e-12, 0.1, 4),)),
        ),
    )
    base = duct_problem(hartmann=10.0, cells=32)
    conducting, insulating = ((0.5,) * 4, (0.025,) * 4), ((1e-12,) * 4, (0.025,) * 4)
    old = ChannelProblem(
        base.grid,
        base.conditions,
        magnetic_field=base.magnetic_field,
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
        wall_layers=(None, (conducting, conducting), (insulating, insulating)),
    )
    found, expected = lmhdx.solve(new), lmhdx.solve(old)
    assert _same(found.velocity, expected.velocity) and _same(found.potential, expected.potential)


def test_a_flow_rate_sets_the_drive_of_the_stokes_duct_exactly():
    unit, driven = lmhdx.solve(duct(20, 32)), lmhdx.solve(duct(20, 32, flow_rate=4.0))
    assert float(driven.flow_rate) == pytest.approx(4.0, rel=1e-13)
    assert float(driven.pressure_gradient) == pytest.approx(4.0 / float(unit.flow_rate), rel=1e-13)
    scaled = jax.tree.map(lambda a: a * driven.pressure_gradient, unit.velocity)
    assert _same(scaled, driven.velocity)
    with pytest.raises(ValueError, match="not both"):
        lmhdx.solve(duct(20, 32).replace(flow_rate=4.0))
    with pytest.raises(NotImplementedError, match="constant-flux"):
        lmhdx.solve(duct(20, 32, flow_rate=4.0, advection="central"))


_SMALL_FRINGE = dict(
    wall_conductance=0.02,
    half_length=1.0,
    upstream=2.0,
    downstream=2.0,
    spacing=1.0,
    cells=8,
    cells_in_layer=2,
)


def test_the_fringe_duct_preset_is_the_older_open_duct_bit_for_bit():
    found = lmhdx.solve(fringe_duct(10.0, **_SMALL_FRINGE))
    expected = solve_open_duct(old_fringe_duct(hartmann=10.0, **_SMALL_FRINGE))
    assert _same(
        (found.velocity, found.pressure, found.potential, found.fields["currents"]),
        (expected.velocity, expected.pressure, expected.potential, expected.currents),
    )
    assert float(found.pressure_drop(-1.0, 1.0)) == float(pressure_drop(expected.pressure, -1.0, 1.0))
    assert float(found.flow_rate) == pytest.approx(4.0, rel=1e-8)


def test_the_fringe_duct_gradient_is_the_older_field_scale_derivative():
    new, old = fringe_duct(10.0, **_SMALL_FRINGE), old_fringe_duct(hartmann=10.0, **_SMALL_FRINGE)

    def old_drop(scale):
        return pressure_drop(solve_open_duct(old, field_scale=scale).pressure, -1.0, 1.0)

    # The leaf is the amplitude: d drop / d strength is the kernel's d / d field_scale over the strength.
    gradient = jax.grad(lambda q: lmhdx.solve(q).pressure_drop(-1.0, 1.0))(new)
    assert float(gradient.B.strength) == pytest.approx(float(jax.grad(old_drop)(1.0)) / 10.0, rel=1e-10)
    # The Stokes solution is linear in the flow rate, so its derivative is the drop over the rate.
    drop = float(lmhdx.solve(new).pressure_drop(-1.0, 1.0))
    assert float(gradient.flow_rate) == pytest.approx(drop / 4.0, rel=1e-8)
    with pytest.raises(NotImplementedError, match="flow rate"):
        jax.grad(lambda q: lmhdx.solve(q).pressure_drop(-1.0, 1.0))(new.replace(advection="central"))


def test_the_open_axis_takes_an_inlet_profile_and_a_thermal_wall_its_temperature():
    from lmhdx.problem import channel_kernel, thermal_kernel

    problem = fringe_duct(10.0, upstream=2.0, downstream=2.0, spacing=1.0, cells=8, cells_in_layer=2)
    profile = np.linspace(0.0, 1.0, 64).reshape(8, 8)
    kernel, forcing, scale, rate = channel_kernel(problem.replace(bc=(Open(profile), Wall(0.02), Wall(0.02))))
    assert (
        kernel.open_axis == 0
        and kernel.conditions[0].is_mixed
        and np.array_equal(kernel.conditions[0].lower, profile)
    )
    assert forcing is None and scale == 1.0 and rate == 1.0
    closed = duct(10, 24).replace(
        bc=("periodic", Sides(Wall(temperature=1.0), Wall(heat_flux=2.0)), Wall(heat_flux=0.0)),
        thermal=Thermal(1.0, 0.5, 0.0),
    )
    heated = thermal_kernel(closed, channel_kernel(closed)[0])
    assert heated.thermal_conditions[1].kinds == ("dirichlet", "neumann")
    assert heated.thermal_conditions[1].lower == 1.0 and heated.thermal_conditions[1].upper == -2.0 / 0.5
    with pytest.raises(NotImplementedError, match="H7.4"):
        thermal_kernel(problem.replace(thermal=Thermal(1.0, 0.5, 0.0)), kernel)


def test_the_pipe_preset_is_the_older_pipe_bit_for_bit():
    found, (velocity, potential) = (
        lmhdx.solve(pipe(10, 16, 24)),
        solve_pipe(pipe_problem(hartmann=10, radial=16, azimuthal=24)),
    )
    assert np.array_equal(found.velocity[0].data, velocity.data) and np.array_equal(
        found.potential.data, potential.data
    )
    assert 0.0 < found.flow_rate < 1.0
    with pytest.raises(NotImplementedError, match="Stokes"):
        lmhdx.solve(
            pipe(10, 16, 24, wall_conductance=0.02).replace(advection="central"), Solver(method="newton")
        )


def test_the_q2d_preset_is_the_older_q2d_bit_for_bit_and_differentiates_in_its_coefficients():
    problem = q2d(shape=(16, 16))
    found = lmhdx.solve(problem, Solver(dt=0.01, steps=20))
    expected = solve_q2d(make_q2d_case(shape=(16, 16), dt=0.01, steps=20))
    assert np.array_equal(found.fields["vorticity"], expected.vorticity)
    assert np.array_equal(found.fields["velocity"][0], expected.velocity_x)

    def energy(p):
        return jnp.sum(lmhdx.solve(p, Solver(dt=0.01, steps=20)).fields["vorticity"] ** 2)

    gradient = jax.grad(energy)(problem)
    old = jax.grad(
        lambda nu, friction: jnp.sum(
            evolve_q2d(
                problem.initial_vorticity,
                viscosity=nu,
                hartmann_friction=friction,
                dt=0.01,
                steps=20,
            )[0]
            ** 2
        ),
        argnums=(0, 1),
    )(problem.viscosity, problem.hartmann_friction)
    assert float(gradient.viscosity) == pytest.approx(float(old[0]), rel=1e-12)
    assert float(gradient.hartmann_friction) == pytest.approx(float(old[1]), rel=1e-12)
    with pytest.raises(ValueError, match="dt and steps"):
        lmhdx.solve(problem)


def _heated_duct():
    grid = Grid(uniform_faces(1, 0.0, 1.0), uniform_faces(8, -1.0, 1.0), uniform_faces(8, -1.0, 1.0))
    new = Problem(
        grid,
        bc=("periodic", Wall(heat_flux=0.0), Sides(Wall(heat_flux=0.0), Wall(heat_flux=1.0))),
        B=Uniform(0.0, 10.0, 0.0),
        forcing=(1.0, 0.0, 0.0),
        thermal=Thermal(
            1.0, 0.05, 0.0, expansion_coefficient=0.02, gravity=(0.0, 0.0, -1.0), heat_source=0.5
        ),
    )
    wall = BoundaryCondition(NEUMANN)
    flow = ChannelProblem(
        grid,
        (BoundaryCondition(PERIODIC), wall, wall),
        magnetic_field=(0.0, 10.0, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=2.0e-3,
    )
    conditions = (
        BoundaryCondition(PERIODIC),
        thermal_boundary("heat_flux", conductivity=0.05),
        thermal_boundary("heat_flux", 0.0, 1.0, conductivity=0.05),
    )
    heated = ThermalProblem(
        flow,
        conditions,
        specific_heat=1.0,
        thermal_conductivity=0.05,
        reference_temperature=0.0,
        expansion_coefficient=0.02,
        gravity=(0.0, 0.0, -1.0),
    )
    return new, flow, heated


def test_the_heated_duct_run_is_the_older_thermal_run_bit_for_bit():
    new, flow, heated = _heated_duct()
    found = lmhdx.solve(new, Solver(dt=2.0e-3, steps=40))
    state = initial_thermal_state(heated, 0.0)
    source = Field(jnp.full(flow.grid.shape, 0.5), (CENTER,) * 3, flow.grid)
    step = build_thermal_step(heated, dtype=jnp.result_type(float))
    final, history = jax.jit(lambda s: advance_thermal(step, s, source, 40))(state)
    assert _same((found.temperature, found.velocity), (final.temperature, final.velocity))
    assert _same(found.history, history)
    assert bool(jnp.all(found.history.finite)) and found.history.heat_balance_residual.shape == (40,)
    # Restarting from the end state is the same as going on: 20 + 20 steps equal 40 steps.
    half = lmhdx.solve(new, Solver(dt=2.0e-3, steps=20), start=None)
    from lmhdx.thermal import ThermalState

    resumed = lmhdx.solve(
        new,
        Solver(dt=2.0e-3, steps=20),
        start=ThermalState(half.velocity, half.temperature, jnp.asarray(20 * 2.0e-3)),
    )
    assert np.max(np.abs(np.asarray(resumed.temperature.data) - np.asarray(final.temperature.data))) < 1e-12
    with pytest.raises(ValueError, match="no start and no heat"):
        lmhdx.solve(new)
    # A heat source and an initial temperature are leaves, differentiated through the scan.
    solver = Solver(dt=2.0e-3, steps=5)

    def heat(source, start):
        thermal = dataclasses.replace(new.thermal, heat_source=source, initial_temperature=start)
        return jnp.sum(lmhdx.solve(new.replace(thermal=thermal), solver).temperature.data)

    slope = jax.grad(heat, argnums=(0, 1))(0.5, 0.0)
    step = 1.0e-4
    central = [
        (heat(0.5 + step, 0.0) - heat(0.5 - step, 0.0)) / (2 * step),
        (heat(0.5, step) - heat(0.5, -step)) / (2 * step),
    ]
    assert [float(g) for g in slope] == pytest.approx([float(c) for c in central], rel=1e-6)
    with pytest.raises(NotImplementedError, match="concrete drive"):
        jax.grad(lambda q: jnp.sum(lmhdx.solve(q, solver).temperature.data))(new)


def test_the_unsteady_duct_is_the_older_projection_run_bit_for_bit():
    problem = duct(5, 12, cells_in_layer=3)
    run = lmhdx.solve(problem, Solver(dt=0.01, steps=8))
    kernel = ChannelProblem(
        problem.grid,
        duct_problem(hartmann=5.0, cells=12, cells_in_layer=3).conditions,
        magnetic_field=(0.0, 5.0, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=0.01,
    )
    expected = advance(kernel, 8, viscous=kernel.viscous_factorizations())
    assert _same(run.velocity, expected.velocity) and _same(run.pressure, expected.pressure)
    assert np.array_equal(run.history["kinetic_energy"], expected.kinetic_energy)


def test_the_four_fields_are_pytrees_and_agree_on_a_uniform_field(tmp_path):
    base = duct(20, 32)
    reference = lmhdx.solve(base).velocity[0].data
    path = tmp_path / "table.npz"
    axes = np.linspace(-1.0, 1.0, 5)
    zeros = np.zeros((5, 5))
    np.savez(path, y=axes, z=axes, bx=zeros, by=20.0 * np.ones((5, 5)), bz=zeros)
    uniform = lambda x, y, z, a: (0 * x, a[0] + 0 * y, 0 * z)  # noqa: E731
    for field in (Func(uniform, jnp.array([20.0])), Table(str(path))):
        found = lmhdx.solve(base.replace(B=field)).velocity[0].data
        assert float(jnp.max(jnp.abs(found - reference))) < 1e-10 * float(jnp.max(jnp.abs(reference)))
    for field in (Uniform(0.0, 3.0, 4.0), Fringe(2.0, 3.0), Table("t.npz", 2.0), Func(uniform, (1.0, 2.0))):
        leaves, tree = jax.tree.flatten(field)
        assert jax.tree.unflatten(tree, leaves).__class__ is field.__class__
    assert jax.tree.leaves(Uniform(0.0, 3.0, 4.0)) == [5.0]
    assert Uniform(0.0, 3.0, 4.0).direction == (0.0, 0.6, 0.8)
    assert len(jax.tree.leaves(base)) == 4  # forcing (3) and the field strength; no unsupported leaf
    with pytest.raises(NotImplementedError, match="Func parameter"):
        jax.grad(lambda a: lmhdx.solve(base.replace(B=Func(uniform, a))).flow_rate)(jnp.array([20.0]))


def test_ha_derivative_has_a_second_order_taylor_remainder_and_matches_central_differences():
    """dp/dHa of a flow-rate-driven Stokes duct, the grid frozen at Ha 20 (plan 5.5)."""
    base = duct(20, 32, flow_rate=4.0)

    def pressure_gradient(ha):
        return lmhdx.solve(base.replace(B=Uniform(0.0, ha, 0.0))).pressure_gradient

    value, slope = jax.value_and_grad(pressure_gradient)(20.0)
    assert float(value) == pytest.approx(25.865404698155118, rel=1e-8)
    step = 1.0e-3
    central = (pressure_gradient(20.0 + step) - pressure_gradient(20.0 - step)) / (2 * step)
    assert float(slope) == pytest.approx(float(central), rel=1e-7)
    remainders = [
        abs(float(pressure_gradient(20.0 + eps)) - float(value) - eps * float(slope))
        for eps in (0.8, 0.4, 0.2)
    ]
    assert 3.6 < remainders[0] / remainders[1] < 4.4 and 3.6 < remainders[1] / remainders[2] < 4.4
    assert remainders[2] > 1e-6  # well above the solver's noise: the ratio is the curvature, not round-off


def test_the_problem_gradient_pairs_forward_and_reverse_mode():
    base = duct(20, 32)

    def response(problem):
        found = lmhdx.solve(problem)
        return jnp.concatenate([found.velocity[0].data[0, ::5, ::5].ravel(), found.flow_rate[None]])

    leaves, tree = jax.tree.flatten(base)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    tangent = jax.tree.unflatten(tree, [jax.random.normal(keys[0], jnp.shape(leaf)) for leaf in leaves])
    primal, forward = jax.jvp(response, (base,), (tangent,))
    _, pullback = jax.vjp(response, base)
    weight = jax.random.normal(keys[1], primal.shape)
    reverse = pullback(weight)[0]
    left = float(jnp.vdot(forward, weight))
    right = sum(
        float(jnp.vdot(a, b)) for a, b in zip(jax.tree.leaves(tangent), jax.tree.leaves(reverse), strict=True)
    )
    assert left == pytest.approx(right, rel=1e-9)


def test_the_problem_gradient_follows_the_physics():
    base = duct(20, 32)
    # The flux is linear in the drive and falls with the field: both leaves of the gradient follow.
    gradient = jax.grad(lambda p: lmhdx.solve(p).flow_rate)(base)
    flux = float(lmhdx.solve(base).flow_rate)
    assert float(gradient.forcing[0]) == pytest.approx(flux, rel=1e-8)
    assert (
        float(gradient.forcing[1]) == pytest.approx(0.0, abs=1e-9 * flux) and float(gradient.B.strength) < 0.0
    )
    step = 1.0e-3 * 20.0
    central = (
        lmhdx.solve(base.replace(B=Uniform(0.0, 20.0 + step, 0.0))).flow_rate
        - lmhdx.solve(base.replace(B=Uniform(0.0, 20.0 - step, 0.0))).flow_rate
    ) / (2 * step)
    assert float(gradient.B.strength) == pytest.approx(float(central), rel=1e-6)


def test_what_cannot_be_differentiated_yet_is_refused_not_silently_zero():
    base = duct(20, 32)
    with pytest.raises(NotImplementedError, match="viscosity"):
        jax.grad(lambda nu: lmhdx.solve(base.replace(viscosity=nu)).flow_rate)(1.0)
    with pytest.raises(NotImplementedError, match="wall conductance"):
        jax.grad(lambda c: lmhdx.solve(base.replace(bc=("periodic", Wall(c), "wall"))).flow_rate)(0.05)
    with pytest.raises(TypeError, match="abstract"):
        jax.jit(lambda ha: lmhdx.solve(base.replace(B=Uniform(0.0, ha, 0.0))).flow_rate)(20.0)
    with pytest.raises(ValueError, match="nonzero primal"):
        jax.grad(lambda b: lmhdx.solve(base.replace(B=Uniform(b, 0.0 * b, 0.0))).flow_rate)(0.0)
    # The closed-over problem jits, compiling the objective once.
    objective = jax.jit(jax.grad(lambda f: lmhdx.solve(base.replace(forcing=(f, 0.0, 0.0))).flow_rate))
    assert float(objective(1.0)) == pytest.approx(float(lmhdx.solve(base).flow_rate), rel=1e-8)


def test_the_solver_resolves_its_method_and_refuses_what_the_kernels_cannot_do():
    stokes, inertial = duct(20, 32), duct(20, 32, advection="central")
    assert Solver().resolve(stokes) == "stokes" and Solver().resolve(inertial) == "newton"
    assert Solver(steps=3, dt=0.1).resolve(stokes) == "euler"
    assert Solver(dt=0.01, steps=2).resolve(q2d(shape=(8, 8))) == "ifrk4"
    with pytest.raises(ValueError, match="dt and steps"):
        Solver().resolve(q2d(shape=(8, 8)))
    for method, plan in (("rk3", "H2.2"), ("march", "H3"), ("implicit", "H2.2")):
        with pytest.raises(NotImplementedError, match=plan):
            Solver(method=method).resolve(stokes)
    with pytest.raises(ValueError, match="one of"):
        Solver(method="rk4").resolve(stokes)
    with pytest.raises(ValueError, match="does not fit"):
        Solver(method="newton").resolve(stokes)
    with pytest.raises(ValueError, match="does not fit"):
        Solver(method="stokes").resolve(inertial)
    with pytest.raises(ValueError, match="dt and steps"):
        Solver(method="euler").resolve(stokes)
    with pytest.raises(ValueError, match="no start"):
        lmhdx.solve(stokes, start="rest")
    with pytest.raises(ValueError, match="bc"):
        Problem(stokes.grid, bc=("periodic", "slip", "wall")).ends(1)
    with pytest.raises(NotImplementedError, match="Open axis is x"):
        lmhdx.solve(Problem(stokes.grid, bc=("periodic", Open(), "wall"), flow_rate=1.0))
    with pytest.raises(NotImplementedError, match="one conductance"):
        lmhdx.solve(stokes.replace(bc=("periodic", Sides(Wall(0.1), Wall(0.2)), "wall")))
    with pytest.raises(ValueError, match="not both"):
        Wall(temperature=1.0, heat_flux=1.0)


def test_the_root_solve_still_takes_the_older_models():
    legacy = duct_problem(hartmann=10.0, cells=16)
    assert np.array_equal(lmhdx.solve(legacy).velocity[0].data, lmhdx.cases.solve(legacy).velocity[0].data)
    with pytest.raises(TypeError, match="no solver"):
        lmhdx.solve(legacy, Solver())


def test_the_channel_preset_follows_the_boeck_krasnov_zienicke_conventions():
    ratio = bulk_ratio(20.0)
    assert (bulk_ratio(10.0), ratio) == (pytest.approx(0.90008, abs=1e-5), pytest.approx(0.95000, abs=1e-5))
    problem = channel(R=700, ha=20.0, cells=(4, 4, 32))
    reynolds = 700 * 20.0 * ratio
    assert reynolds == pytest.approx(13300.0, rel=1e-5) and problem.viscosity == pytest.approx(1.0 / reynolds)
    assert problem.B.strength == pytest.approx(20.0 * np.sqrt(problem.viscosity)) and problem.B.direction == (
        0.0,
        0.0,
        1.0,
    )
    assert problem.grid.extent[:2] == pytest.approx(
        (2 * np.pi, 0.8 * np.pi)
    ) and problem.flow_rate == pytest.approx(1.6 * np.pi)
    assert np.array_equal(problem.grid.z_faces, axis(32, -1.0, 1.0, blend=0.96))
    assert np.array_equal(
        channel(re=reynolds, ha=20.0, cells=(4, 4, 32)).grid.y_faces, problem.grid.y_faces
    )  # the same row, by Re
    spanwise = channel(re=6667, ha=30.0, field="spanwise", cells=(4, 4, 16), stretch=dict(beta=2.0))
    assert spanwise.grid.extent[:2] == pytest.approx((2 * np.pi, np.pi)) and spanwise.B.direction == (
        0.0,
        1.0,
        0.0,
    )
    assert np.array_equal(spanwise.grid.z_faces, axis(16, -1.0, 1.0, beta=2.0))
    for bad in (
        dict(re=1000.0, ha=20.0),
        dict(R=700.0, ha=21.0),
        dict(R=700.0, re=1.0, ha=20.0),
        dict(R=700.0),
    ):
        with pytest.raises(ValueError):
            channel(**bad)
    with pytest.raises(ValueError, match="wall_normal"):
        channel(re=1.0, field="up")


# Boeck, Krasnov and Zienicke 2007, Table 1: (R, Ha) -> R_av = Uav delta / nu, Re_tau, c_f.
_BKZ_ROWS = {
    (450, 10): (405.0, 240.7, 7.06e-3),
    (450, 15): (420.0, 359.2, 6.50e-3),
    (450, 20): (427.0, 475.8, 6.21e-3),
    (450, 30): (435.0, 710.8, 5.93e-3),
    (500, 10): (450.0, 268.2, 7.11e-3),
    (500, 15): (466.7, 398.8, 6.49e-3),
    (500, 20): (475.0, 528.4, 6.18e-3),
    (500, 30): (483.3, 785.9, 5.87e-3),
    (700, 10): (630.055, 365.0, 6.72e-3),
    (700, 15): (653.33, 538.0, 6.02e-3),
    (700, 20): (665.0, 708.6, 5.67e-3),
    (700, 30): (676.0, 1047.3, 5.33e-3),
    (900, 10): (810.1, 456.0, 6.34e-3),
    (900, 15): (840.0, 681.3, 5.84e-3),
    (900, 20): (855.0, 891.9, 5.44e-3),
    (900, 25): (864.0, 1101.0, 5.20e-3),
}


def test_the_boeck_krasnov_zienicke_table_is_self_consistent():
    """Re_tau = Re_av sqrt(c_f / 2) holds to 6.4e-4 on all 16 rows (plan T27); the laminar floors follow."""
    assert set(_BKZ_ROWS) == set(_BKZ_DOMAINS)
    for (R, ha), (r_av, re_tau, cf) in _BKZ_ROWS.items():
        assert re_tau == pytest.approx(ha * r_av * np.sqrt(cf / 2.0), rel=6.4e-4)
        assert r_av == pytest.approx(
            R * bulk_ratio(float(ha)), rel=1.2e-3
        )  # row (450, 20) prints 427 for 427.5
    for (R, ha), floor in (
        ((450, 10), 5.486e-3),
        ((450, 30), 4.756e-3),
        ((500, 30), 4.281e-3),
        ((700, 20), 3.166e-3),
        ((900, 25), 2.411e-3),
    ):
        ratio, reynolds = bulk_ratio(float(ha)), R * ha * bulk_ratio(float(ha))
        wall_shear = ha * np.sinh(ha) / (np.cosh(ha) - 1.0) / ratio  # in U0 / d, U0 = Uav / ratio
        assert 2.0 * wall_shear / reynolds == pytest.approx(floor, rel=1e-3)


def test_a_spanwise_field_leaves_the_laminar_channel_a_parabola_and_a_wall_normal_one_is_refused():
    """A field parallel to the walls drives no current in laminar flow: Poiseuille, G Re_av = 3 (c_f = 6 / Re_av)."""
    from validation.gci import grid_convergence

    def drive(cells):
        problem = channel(re=6667, ha=30.0, field="spanwise", cells=(1, 1, cells), stretch=dict(ratio=1.0))
        found = lmhdx.solve(problem)
        assert float(found.flow_rate) == pytest.approx(problem.flow_rate, rel=1e-12)
        return float(found.pressure_gradient) * 6667

    result = grid_convergence(drive(16), drive(32), drive(64), 2.0)
    assert 1.95 < result.order < 2.05 and result.extrapolated == pytest.approx(3.0, rel=1e-5)
    with pytest.raises(NotImplementedError, match="net current"):
        lmhdx.solve(channel(R=700, ha=20.0, cells=(1, 1, 33)))


def test_the_beginner_example_solves_and_differentiates_the_duct(tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples/problem_example.py"
    done = subprocess.run(
        [sys.executable, script], cwd=tmp_path, timeout=250, check=True, capture_output=True, text=True
    )
    values = {
        line.split("=")[0].strip(): float(line.split("=")[1])
        for line in done.stdout.splitlines()
        if "=" in line and "Solving" not in line
    }
    assert values["flow rate"] == pytest.approx(4.0, rel=1e-12)
    assert values["pressure gradient"] == pytest.approx(25.9527, rel=1e-5)
    # The flux is fixed, so the field raises the drive, and the drive is linear in the flux.
    assert 1.0 < values["d(pressure gradient)/d(field strength)"] < 1.2
    drive = values["d(pressure gradient)/d(flow rate)"]
    assert drive == pytest.approx(values["pressure gradient"] / 4.0, rel=1e-6)
