"""The 2.0 facade (plan H4.1): Problem, the presets and solve agree with the kernels they wrap.

Every equivalence test builds the case twice, once through the older entry point and once
through ``Problem``, and compares the arrays bit for bit. The derivative tests differentiate
the leaves of a ``Problem`` with Taylor remainders, central differences and the JVP/VJP pairing.
"""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lmhdx
from lmhdx import Open, Problem, Sides, Solver, Thermal, Uniform, Wall, duct
from lmhdx.axial import fringe_duct as old_fringe_duct
from lmhdx.axial import pressure_drop, solve_open_duct
from lmhdx.cases import NumericalFailure, WallLayer
from lmhdx.core3d import ChannelProblem, advance, duct_problem
from lmhdx.fields import Fringe, Func, Table
from lmhdx.grid import CENTER, NEUMANN, PERIODIC, BoundaryCondition, Field, Grid, axis, uniform_faces
from lmhdx.output import Output, Start, read_record
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
    kernel, forcing, scale, rate, _ = channel_kernel(
        problem.replace(bc=(Open(profile), Wall(0.02), Wall(0.02)))
    )
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
        jax.grad(
            lambda c: lmhdx.solve(pipe(20, 16, 16).replace(bc=(Wall(c), "periodic", "periodic"))).flow_rate
        )(0.05)
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


# Differentiable walls (plan H4.2, T35). The drive of a flow-rate duct is the response: G(c) = dp/dx.


def _drive(problem):
    return lmhdx.solve(problem).pressure_gradient


def _hunt(wall, cells=24, ha=20.0):
    return duct(ha, cells, flow_rate=4.0, walls=dict(y=wall))


def _central(function, x, step):
    return (function(x * (1 + step)) - function(x * (1 - step))) / (2 * x * step)


def test_a_walls_numbers_are_leaves_and_the_insulating_wall_is_structure():
    assert jax.tree.leaves(Wall()) == [] and jax.tree.leaves(Wall(0.0)) == []
    assert jax.tree.leaves(Wall(0.05)) == [0.05]
    assert jax.tree.leaves(Wall(conductivity=2.0, thickness=0.025)) == [2.0, 0.025]
    layered = Wall(layers=(WallLayer("a", 2.0, 0.1, 3), WallLayer("b", 0.5, 0.2, 2)), temperature=1.0)
    leaves, tree = jax.tree.flatten(layered)
    assert leaves == [2.0, 0.1, 0.5, 0.2]
    assert (
        jax.tree.unflatten(tree, leaves) == layered
    )  # the cell counts and the thermal condition are structure
    problem = duct(20, 24, walls=dict(y=Wall(0.05), z=Sides(Wall(), Wall(conductivity=3.0, thickness=0.1))))
    flat, tree = jax.tree.flatten(problem)
    assert flat[-3:] == [0.05, 3.0, 0.1] and len(flat) == 7
    assert jax.tree.unflatten(tree, flat).bc == problem.bc
    with pytest.raises(ValueError, match="not both"):
        lmhdx.solve(_hunt(Wall(0.05, conductivity=1.0, thickness=0.05)))
    with pytest.raises(ValueError, match="together"):
        lmhdx.solve(_hunt(Wall(conductivity=1.0)))
    with pytest.raises(ValueError, match="conducting fluid"):
        lmhdx.solve(duct(0, 24, flow_rate=4.0, walls=dict(y=Wall(conductivity=1.0, thickness=0.05))))


def test_the_thin_wall_forms_its_conductance_from_conductivity_and_thickness():
    by_conductance = lmhdx.solve(_hunt(Wall(0.05)))
    by_material = lmhdx.solve(_hunt(Wall(conductivity=2.0, thickness=0.025)))
    assert _same(
        (by_material.velocity, by_material.potential), (by_conductance.velocity, by_conductance.potential)
    )
    # c = sigma_w t_w / sigma: a wall as conducting as a fluid of twice the conductivity is half as conducting.
    heavy = _hunt(Wall(conductivity=2.0, thickness=0.05)).replace(conductivity=2.0)
    assert _same(
        lmhdx.solve(heavy).velocity, lmhdx.solve(heavy.replace(bc=("periodic", Wall(0.05), "wall"))).velocity
    )


def test_the_wall_conductance_derivative_meets_the_derivative_standard():
    """T35: central differences with the plateau, Taylor order 2, JVP against VJP, and one program for every value."""
    lmhdx.enable_x64()

    def drive(c):
        return _drive(_hunt(Wall(c, conducting=True)))

    value, slope = jax.value_and_grad(drive)(0.05)
    assert float(value) == pytest.approx(float(_drive(_hunt(Wall(0.05)))), rel=1e-9)
    for step in (1e-3, 1e-4):
        assert float(slope) == pytest.approx(float(_central(drive, 0.05, step)), rel=1e-6)
    assert float(jax.jvp(drive, (0.05,), (1.0,))[1]) == pytest.approx(float(slope), rel=1e-9)
    remainders = [
        abs(float(drive(0.05 + eps)) - float(value) - eps * float(slope)) for eps in (0.016, 0.008, 0.004)
    ]
    assert 3.6 < remainders[0] / remainders[1] < 4.4 and 3.6 < remainders[1] / remainders[2] < 4.4
    # A new conductance is a new argument of the same compiled program: no retrace, no recompile.
    from lmhdx.solver import _program

    objective, before = jax.jit(jax.grad(drive)), _program.cache_info()
    values = [float(objective(c)) for c in (0.03, 0.05, 0.07)]
    assert objective._cache_size() == 1 and _program.cache_info().misses <= before.misses + 1
    assert values[1] == pytest.approx(float(slope), rel=1e-9) and values[0] > values[1] > values[2] > 0.0
    sweep = jax.vmap(jax.grad(drive))(jnp.array([0.03, 0.05, 0.07]))
    np.testing.assert_allclose(sweep, values, rtol=1e-9)


def test_conductivity_and_thickness_are_distinct_directions_of_the_conductance():
    sigma, thickness = 2.0, 0.025

    def drive(k, t):
        return _drive(_hunt(Wall(conductivity=k, thickness=t)))

    c_slope = jax.grad(lambda c: _drive(_hunt(Wall(c))))(sigma * thickness)
    d_k, d_t = jax.grad(drive, argnums=(0, 1))(sigma, thickness)
    assert float(d_k) == pytest.approx(
        thickness * float(c_slope), rel=1e-9
    )  # the chain rule, each through its own leaf
    assert float(d_t) == pytest.approx(sigma * float(c_slope), rel=1e-9)
    assert float(d_k) == pytest.approx(float(_central(lambda k: drive(k, thickness), sigma, 1e-3)), rel=1e-6)
    assert float(d_t) == pytest.approx(float(_central(lambda t: drive(sigma, t), thickness, 1e-3)), rel=1e-6)
    mixed = jax.jvp(drive, (sigma, thickness), (0.3, -0.7))[1]
    assert float(mixed) == pytest.approx(0.3 * float(d_k) - 0.7 * float(d_t), rel=1e-9)
    assert abs(float(d_k) / float(d_t) - thickness / sigma) < 1e-9 and float(d_t) > 0.0


def test_the_wall_leaves_of_a_problem_pair_forward_and_reverse_mode():
    base = _hunt(Wall(0.05)).replace(bc=("periodic", Wall(0.05), Wall(conductivity=1.5, thickness=0.04)))

    def response(problem):
        found = lmhdx.solve(problem)
        return jnp.concatenate([found.velocity[0].data[0, ::5, ::5].ravel(), found.pressure_gradient[None]])

    leaves, tree = jax.tree.flatten(base)
    assert len(leaves) == 8
    keys = jax.random.split(jax.random.PRNGKey(3), 2)
    tangent = jax.tree.unflatten(tree, [jax.random.normal(keys[0], jnp.shape(leaf)) for leaf in leaves])
    primal, forward = jax.jvp(response, (base,), (tangent,))
    reverse = jax.vjp(response, base)[1](weight := jax.random.normal(keys[1], primal.shape))[0]
    right = sum(
        float(jnp.vdot(a, b)) for a, b in zip(jax.tree.leaves(tangent), jax.tree.leaves(reverse), strict=True)
    )
    assert float(jnp.vdot(forward, weight)) == pytest.approx(right, rel=1e-8)
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(reverse))


def test_resolved_layers_differentiate_in_conductivity_and_thickness_as_the_thin_wall_does_to_a_percent():
    def drive(k, t):
        return _drive(_hunt(Wall(layers=(WallLayer("steel", k, t, 6),)), cells=32))

    k, t = 1.0, 0.05
    d_k, d_t = jax.grad(drive, argnums=(0, 1))(k, t)
    assert float(d_k) == pytest.approx(float(_central(lambda x: drive(x, t), k, 1e-3)), rel=1e-6)
    assert float(d_t) == pytest.approx(float(_central(lambda x: drive(k, x), t, 1e-3)), rel=1e-6)
    thin = jax.grad(lambda x: _drive(_hunt(Wall(conductivity=k, thickness=x), cells=32)))(t)
    assert float(d_t) == pytest.approx(float(thin), rel=1e-2) and float(d_t) != float(thin)
    # Measured 0.76 % and 0.55 %: Hunt's wall depends on the product, to the percent by which a resolved wall differs from a thin one.
    assert float(d_k * k / (d_t * t)) == pytest.approx(1.0, rel=1e-2)


def test_two_conducting_axes_differentiate_through_the_corner_correction():
    both = duct(20, 24, flow_rate=4.0, walls=dict(y=Wall(0.05), z=Wall(0.1)))
    for index, c in ((1, 0.05), (2, 0.1)):

        def drive(x, index=index):
            walls = [Wall(0.05), Wall(0.1)]
            walls[index - 1] = Wall(x)
            return _drive(both.replace(bc=("periodic", *walls)))

        assert float(jax.grad(drive)(c)) == pytest.approx(float(_central(drive, c, 1e-3)), rel=1e-6)


def test_the_wall_derivative_is_refused_where_it_is_not_defined_or_not_yet_available():
    lmhdx.enable_x64()
    with pytest.raises(ValueError, match="insulating"):
        jax.grad(lambda c: _drive(_hunt(Wall(c))))(0.0)
    with pytest.raises(TypeError, match="conducting=True"):
        jax.jit(lambda c: _drive(_hunt(Wall(c))))(0.05)
    with pytest.raises(NotImplementedError, match="one conductance"):
        jax.grad(
            lambda c: _drive(_hunt(Wall(c)).replace(bc=("periodic", Sides(Wall(c), Wall(2.0 * c)), "wall")))
        )(0.05)
    # An open duct and an unsteady run take the primal; their derivative in a wall number is NaN, not a silent zero.
    small = dict(_SMALL_FRINGE) | dict(wall_conductance=0.02)
    fringe = fringe_duct(10.0, **small)
    gradient = jax.grad(lambda q: lmhdx.solve(q).pressure_drop(-1.0, 1.0))(fringe)
    assert np.isnan(float(gradient.bc[1].conductance)) and np.isfinite(float(gradient.flow_rate))
    exact = lmhdx.solve(fringe.replace(bc=(Open(), Wall(jnp.asarray(0.02)), Wall(0.02))))
    assert float(exact.pressure_drop(-1.0, 1.0)) == float(lmhdx.solve(fringe).pressure_drop(-1.0, 1.0))


# Output (plan H4.4): restart, ParaView files, summary and log for every run type.


def _vtk(path):
    """The cell arrays of a ParaView file, by name, and its type and extent."""
    root = ElementTree.parse(path).getroot()
    arrays = {i.get("Name"): np.array(i.text.split(), dtype=float) for i in root.iter("DataArray")}
    return root.get("type"), root[0].get("WholeExtent"), arrays


_RUNS = {
    "euler": (lambda: duct(5, 12, cells_in_layer=3), Solver(dt=0.01, steps=8), 4),
    "thermal": (lambda: _heated_duct()[0], Solver(dt=2.0e-3, steps=40), 10),
    "q2d": (lambda: q2d(shape=(16, 16)), Solver(dt=0.01, steps=20, history=5), 5),
}


@pytest.mark.parametrize("name", _RUNS)
def test_a_run_in_chunks_or_from_its_record_is_the_run_in_one_bit_for_bit(tmp_path, name):
    build, solver, every = _RUNS[name]
    problem, half = build(), dataclasses.replace(solver, steps=solver.steps // 2)
    whole = lmhdx.solve(problem, solver)
    for chunk in (0, every):  # an Output changes no number, whether the run is one chunk or several
        found = lmhdx.solve(problem, solver, Output(str(tmp_path / f"c{chunk}"), every=chunk))
        assert _same(found.fields, whole.fields) and _same(found.history, whole.history)
    # 20 + 20 steps are 40: the second half starts from the first half's record.
    lmhdx.solve(problem, half, Output(str(tmp_path / "half")))
    record = tmp_path / "half" / f"run_{half.steps:06d}.npz"
    again = lmhdx.solve(problem, half, Output(str(tmp_path / "again")), start=record)
    assert _same(again.fields, whole.fields)
    assert (tmp_path / "again" / f"run_{solver.steps:06d}.npz").exists()
    assert {p.name for p in (tmp_path / "again").glob("*.pvd")} == {f"run_from_{half.steps:06d}.pvd"}
    for file in ("run_%06d.npz" % solver.steps,):
        with np.load(tmp_path / "again" / file) as resumed, np.load(tmp_path / f"c{every}" / file) as chunked:
            assert all(np.array_equal(resumed[k], chunked[k]) for k in chunked.files if k != "notes")


def test_a_record_resumes_only_the_run_that_wrote_it(tmp_path):
    problem, solver = duct(5, 12, cells_in_layer=3), Solver(dt=0.01, steps=4)
    lmhdx.solve(problem, solver, Output(str(tmp_path)))
    record = tmp_path / "run_000004.npz"
    with pytest.raises(ValueError, match="another problem or solver"):
        lmhdx.solve(duct(6, 12, cells_in_layer=3), solver, start=record)
    with pytest.raises(ValueError, match="another problem or solver"):
        lmhdx.solve(problem, dataclasses.replace(solver, dt=0.02), start=record)
    # An import takes the state of any record of the same fields.
    imported = lmhdx.solve(
        duct(6, 12, cells_in_layer=3), solver, start=Start(path=str(record), mode="import")
    )
    assert imported.velocity[0].shape == (2, 12, 12)
    with np.load(record) as arrays:
        arrays = dict(arrays)
    hot = arrays["leaf0"].copy()
    hot[0, 0, 0] = np.nan
    bad, shape = "is not a lmhdx.record record of version 1", arrays["leaf0"].shape
    for change, message in (
        (dict(version=2), bad),
        (dict(leaf0=arrays["leaf0"][:, :3]), rf"leaf 0 is float64\(2, 3, {shape[2]}\)"),
        (dict(leaf0=hot), "leaf 0 is not finite"),
        (dict(leaf1=arrays["leaf1"].astype(np.float32)), "leaf 1 is float32"),
        (dict(leaf9=arrays["leaf1"]), bad),
    ):
        np.savez(tmp_path / "bad.npz", **{**arrays, **change})
        with pytest.raises(ValueError, match=message):
            lmhdx.solve(problem, solver, start=tmp_path / "bad.npz")
    np.savez(tmp_path / "bad.npz", **{k: v for k, v in arrays.items() if k != "leaf3"})
    with pytest.raises(ValueError, match=bad):
        lmhdx.solve(problem, solver, start=tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="multiples of output.every"):
        lmhdx.solve(problem, solver, Output(str(tmp_path / "x"), every=3))
    with pytest.raises(ValueError, match="concrete"):
        jax.grad(
            lambda b: (
                lmhdx.solve(problem.replace(B=Uniform(0.0, b, 0.0)), solver, Output(str(tmp_path / "y")))
                .velocity[0]
                .data.sum()
            )
        )(5.0)
    with pytest.raises(ValueError, match="no start"):
        lmhdx.solve(problem, start=record)


def test_a_steady_solve_writes_its_fields_once(tmp_path):
    problem = duct(20, 24)
    found = lmhdx.solve(problem, output=Output(str(tmp_path), name="hartmann"))
    assert _same(found.fields, lmhdx.solve(problem).fields)
    assert {p.name for p in tmp_path.iterdir()} == {
        "hartmann_000000.npz",
        "hartmann_000000.vtr",
        "hartmann.pvd",
        "hartmann.log",
        "hartmann.jsonl",
        "summary.json",
    }
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["accepted"] and summary["steps"] == 0 and summary["method"] == "stokes"
    assert summary["flow_rate"] == float(found.flow_rate) and summary["residual"] == float(found.residual)
    assert found.summary == summary
    state, step = read_record(tmp_path / "hartmann_000000.npz", found.fields)
    assert step == 0 and _same(state, found.fields)
    # ParaView reads the stretched grid as it is, and the arrays as they are, x fastest.
    kind, extent, arrays = _vtk(tmp_path / "hartmann_000000.vtr")
    grid = problem.grid
    assert kind == "RectilinearGrid" and extent == "0 1 0 24 0 24"
    assert np.allclose(arrays["Y"], grid.y, rtol=1e-12) and np.allclose(arrays["Z"], grid.z, rtol=1e-12)
    assert np.allclose(arrays["pressure"], np.transpose(found.pressure.data).ravel(), rtol=1e-12)
    centred = [np.moveaxis(np.asarray(v.data), a, 0) for a, v in enumerate(found.velocity)]
    centred = [np.moveaxis(0.5 * (c[:-1] + c[1:]), 0, a) for a, c in enumerate(centred)]
    expected = np.stack([np.transpose(c) for c in centred], axis=-1).ravel()
    assert np.allclose(arrays["velocity"], expected, rtol=1e-12, atol=1e-14)
    text = (tmp_path / "hartmann.log").read_text()
    assert text.startswith(
        "LMhdX solver\ncase=hartmann mode=steady solver=stokes geometry=cartesian cells=(1,24,24)"
    )
    assert "completed case=hartmann" in text


def test_a_pipe_is_written_as_a_structured_grid_at_its_true_points(tmp_path):
    problem = pipe(5, 6, 8, cells_in_layer=2)
    found = lmhdx.solve(problem, output=Output(str(tmp_path), fields=("velocity",)))
    kind, extent, arrays = _vtk(tmp_path / "run_000000.vts")
    assert (
        kind == "StructuredGrid"
        and extent == f"0 6 0 8 0 {problem.grid.shape[2]}"
        and "pressure" not in arrays
    )
    points = arrays["Points"].reshape(-1, 3)
    assert np.allclose(np.max(np.hypot(points[:, 0], points[:, 1])), problem.grid.x[-1], rtol=1e-12)
    r, theta = (
        problem.grid.x,
        problem.grid.y,
    )  # x runs fastest, then the angle: the first two points are on one ray
    assert np.allclose(points[1], [r[1] * np.cos(theta[0]), r[1] * np.sin(theta[0]), problem.grid.z[0]])
    assert np.allclose(points[len(r)], [r[0] * np.cos(theta[1]), r[0] * np.sin(theta[1]), problem.grid.z[0]])
    assert np.allclose(arrays["velocity"], np.transpose(found.velocity[0].data).ravel(), rtol=1e-12)


def test_a_rejected_chunk_stops_the_run_and_keeps_the_chunk_before(tmp_path):
    problem, solver = _heated_duct()[0], Solver(dt=2.0e-3, steps=40, advective_cfl_limit=4.0e-5)
    with pytest.raises(NumericalFailure, match="rejected at step") as caught:
        lmhdx.solve(problem, solver, Output(str(tmp_path), every=10))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert caught.value.summary == summary and not summary["accepted"] and summary["steps"] == 10
    assert summary["rejected"]["reasons"] == ["advective_cfl over 4e-05"] and summary["rejected"]["step"] > 10
    written = {p.name for p in tmp_path.iterdir()}
    assert {"run_000010.npz", "run_000020_rejected.npz", "run.pvd"} <= written
    assert not written & {"run_000020.npz", "run_000020.vtr"}
    # The rejected state imports but does not resume.
    with pytest.raises(ValueError, match="another problem or solver"):
        lmhdx.solve(problem, solver, start=tmp_path / "run_000020_rejected.npz")
    assert lmhdx.solve(
        problem,
        Solver(dt=2.0e-3, steps=10),
        start=Start(path=str(tmp_path / "run_000020_rejected.npz"), mode="import"),
    )
    with pytest.raises(NumericalFailure, match="rejected at step"):
        lmhdx.solve(problem, solver)  # without an Output the run raises just the same
    with pytest.raises(ValueError, match="exceeds buoyancy_courant_limit"):  # the start is held to it too
        lmhdx.solve(problem, Solver(dt=2.0e-3, steps=10, buoyancy_courant_limit=1.0e-3))
    # Conduction alone is not held to an advective limit.
    quiet = problem.replace(thermal=dataclasses.replace(problem.thermal, advection="off"))
    assert lmhdx.solve(
        quiet, Solver(dt=2.0e-3, steps=10, advective_cfl_limit=1.0e-9), Output(str(tmp_path / "q"))
    ).summary["accepted"]


def test_the_log_keeps_the_solvers_text_and_adds_a_json_line_per_chunk(tmp_path):
    lmhdx.solve(
        _heated_duct()[0], Solver(dt=2.0e-3, steps=40), Output(str(tmp_path), every=20, name="heated")
    )
    lines = (tmp_path / "heated.log").read_text().splitlines()
    assert lines[0] == "LMhdX solver"
    assert (
        lines[2]
        == "domain=(1.000000e+00,2.000000e+00,2.000000e+00) dt=2.000000e-03 end=8.000000e-02 max_steps=40"
    )
    assert [line.split()[0] for line in lines if line.startswith("step=")] == ["step=20", "step=40"]
    assert sum(line.startswith("progress=50.0%") for line in lines) == 1 and lines[-1].startswith(
        "completed case=heated"
    )
    records = [json.loads(line) for line in (tmp_path / "heated.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [20, 40] and records[1]["time"] == pytest.approx(0.08, rel=1e-12)
    assert records[1]["finite"] == 1.0 and "heat_balance_residual" in records[1]
