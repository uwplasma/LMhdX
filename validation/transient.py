"""Transient checks of the thermal step with the Lorentz force on, against references that share none of its operators.

Each runs :func:`lmhdx.thermal.build_thermal_step`, the path of a run in time (Mistrangelo's Case 2):

- :func:`manufactured_run` marches a manufactured solution in time with every term on: limited momentum transport,
  viscosity, the Lorentz force with insulating or conducting walls, buoyancy, van Leer heat transport, conduction and
  Joule heat. The field is along ``y``. The velocity ``a(t) u_B + b(t) u_D`` has a part invariant along the field,
  ``u_B = (d_z psi, 0, -d_x psi)``, whose electromotive force is ``B grad psi`` and drives no current (the potential
  is ``B a psi``), and a part ``u_D`` with no vorticity along the field, whose current is ``sigma b u_D x B``. Both
  vanish with their wall-normal derivatives at the walls ``z = -1, 1``, so a conducting wall there carries no current
  and the potential and current are exact.
- :func:`vortex_decay` measures the decay of a vortex invariant along the field between Hartmann walls, insulating or
  Case 2's steel resolved in cells, against :func:`validation.shercliff.vortex_decay_rate`.
- :func:`hartmann_roll_growth` measures the growth of Benard rolls along the field between insulating Hartmann walls
  against Sommeria and Moreau's model, :func:`validation.thermal.convection_growth_rate` with the friction of
  :func:`validation.shercliff.vortex_decay_rate`.

``tests/test_steady.py`` checks the first two; ``scripts/make_showcase_figures.py --only transient`` records all three.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from lmhdx import enable_x64
from lmhdx.core3d import ChannelProblem, SolidWall, zero_velocity
from lmhdx.grid import (
    CENTER,
    NEUMANN,
    PERIODIC,
    BoundaryCondition,
    Field,
    Grid,
    uniform_faces,
    wall_resolving_faces,
)
from lmhdx.thermal import (
    ThermalProblem,
    advance_thermal,
    build_thermal_step,
    initial_thermal_state,
    thermal_boundary,
)
from validation.shercliff import vortex_decay_rate
from validation.thermal import convection_growth_rate

__all__ = ["CASE2_WALL", "hartmann_roll_growth", "manufactured_run", "vortex_decay"]

CASE2_WALL = (1.24571, 0.002 / 0.023)  # Case 2's steel: the pinned conductivity ratio and t_w / a
_PERIODIC, _WALL = BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN)
_K, _KY, _NU, _KAPPA, _BETA_G, _T_REF = (
    1.0,
    2.0,
    0.05,
    0.05,
    1.0,
    0.5,
)  # the manufactured solution's constants


def _amplitudes(t):
    """``a, a', b, b', c, c'`` of ``u = a u_B + b u_D`` and ``T = T_0 + c theta``."""
    return (
        1.0 + 0.5 * jnp.sin(3.0 * t),
        1.5 * jnp.cos(3.0 * t),
        0.5 * jnp.cos(2.0 * t),
        -jnp.sin(2.0 * t),
        0.3 + 0.2 * jnp.sin(4.0 * t),
        0.8 * jnp.cos(4.0 * t),
    )


def _balanced(p):
    gradient = jax.grad(lambda q: jnp.sin(_K * q[0]) * (1 - q[2] ** 2) ** 2)(p)
    return jnp.array([gradient[2], 0.0, -gradient[0]])


def _damped(p):
    shape = lambda q: jnp.cos(_K * q[0]) * (1 - q[2] ** 2) ** 3  # noqa: E731
    gradient, hessian = jax.grad(shape)(p), jax.hessian(shape)(p)
    along = jnp.cos(_KY * p[1])
    return jnp.array(
        [
            along * gradient[0],
            -jnp.sin(_KY * p[1]) * (hessian[0, 0] + hessian[2, 2]) / _KY,
            along * gradient[2],
        ]
    )


def _theta(p):
    return jnp.cos(_K * p[0] + 0.4) * jnp.cos(_KY * p[1] + 0.2) * (1 - p[2] ** 2) ** 2


def _base(p):
    return 0.5 * (1.0 - p[2])


def _momentum_terms(p, component):
    """The spatial factors of the momentum source, in the order of the weights of ``_momentum_source``."""
    balanced, damped = _balanced(p), _damped(p)
    jb, jd = jax.jacfwd(_balanced)(p), jax.jacfwd(_damped)(p)
    vertical = jnp.where(component == 2, 1.0, 0.0)
    return jnp.array([
        balanced[component],
        damped[component],
        (jb @ balanced)[component],
        (jd @ balanced + jb @ damped)[component],
        (jd @ damped)[component],
        jnp.trace(jax.hessian(lambda q: _balanced(q)[component])(p)),
        jnp.trace(jax.hessian(lambda q: _damped(q)[component])(p)),
        jnp.where(component == 1, 0.0, damped[component]),
        vertical * (_base(p) - _T_REF),
        vertical * _theta(p),
    ])  # fmt: skip


def _momentum_source(terms, t, rate):
    """``du/dt + (u.grad) u - nu lap u - j x B / rho - beta g (T - T_r) z`` at ``t``, ``rate = sigma B^2 / rho``."""
    a, da, b, db, c, _ = _amplitudes(t)
    weights = jnp.array([da, db, a * a, a * b, b * b, -_NU * a, -_NU * b, rate * b, -_BETA_G, -_BETA_G * c])
    return jnp.tensordot(weights, terms, axes=1)


def _heat_terms(p):
    balanced, damped = _balanced(p), _damped(p)
    base, theta = jax.grad(_base)(p), jax.grad(_theta)(p)
    laplacian = jnp.trace(jax.hessian(_theta)(p))
    return jnp.array([_theta(p), balanced @ base, damped @ base, balanced @ theta, damped @ theta, laplacian,
                      damped[0] ** 2 + damped[2] ** 2])  # fmt: skip


def _heat_source(terms, t, rate):
    """``rho c_p (dT/dt + u.grad T) - k lap T - j^2/sigma`` at ``t``, with ``rho c_p = 1`` and ``k = kappa``."""
    a, _, b, _, c, dc = _amplitudes(t)
    weights = jnp.array([dc, a, b, a * c, b * c, -_KAPPA * c, -rate * b * b])
    return jnp.tensordot(weights, terms, axes=1)


def _velocity(p, t):
    a, _, b, _, _, _ = _amplitudes(t)
    return a * _balanced(p) + b * _damped(p)


def _temperature(p, t):
    return _base(p) + _amplitudes(t)[4] * _theta(p)


def _points(grid: Grid, offset) -> tuple[jnp.ndarray, tuple[int, ...]]:
    axes = [np.asarray(grid.centers[a] if offset[a] == CENTER else grid.faces[a]) for a in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return jnp.asarray(np.stack([m.ravel() for m in mesh], axis=1)), mesh[0].shape


@functools.lru_cache(maxsize=4)
def _sampled(grid: Grid, offsets: tuple) -> tuple:
    """The source's spatial factors on each velocity face and cell, once per mesh, in one compiled pass."""
    places = [_points(grid, offset) for offset in offsets]
    points = jnp.concatenate([where for where, _ in places])
    components = jnp.concatenate([jnp.full(len(where), c) for c, (where, _) in enumerate(places)])
    sampled = jax.jit(jax.vmap(_momentum_terms))(points, components)
    ends = np.cumsum([0] + [len(where) for where, _ in places])
    terms = [sampled[a:b].T.reshape((-1, *shape)) for a, b, (_, shape) in zip(ends, ends[1:], places)]
    cells_at, shape = _points(grid, (CENTER,) * 3)
    heat_terms = jax.jit(jax.vmap(_heat_terms))(cells_at).T.reshape((-1, *shape))
    return places, terms, cells_at, shape, heat_terms


def manufactured_run(
    cells: int = 16,
    *,
    rate: float = 10.0,
    damping: str = "explicit",
    dt: float = 0.01,
    final: float = 1.0,
    wall: tuple[int, float] | None = None,
) -> dict:
    """March the manufactured solution to ``final``; return each field's max and RMS errors and its final arrays.

    ``cells`` along ``x`` (one wavelength ``2 pi``) and ``z`` (``-1..1``), half as many along ``y`` (``pi``);
    ``rate`` is ``sigma B^2 / rho`` (zero for no field), with ``nu = kappa = 0.05`` and ``beta g = 1``. ``wall`` is
    ``(cells, sigma_w / sigma)`` for conducting solid walls 0.2 thick at ``z = -1, 1``, with the fluid's ``k`` and
    ``rho c_p``. Every source is taken at the step's old time, as the step takes its force. Errors are in the order
    ``u_x, u_y, u_z, T``, on every velocity face and every fluid cell.
    """
    enable_x64()
    grid = Grid(
        uniform_faces(cells, 0.0, 2 * np.pi / _K),
        uniform_faces(cells // 2, 0.0, 2 * np.pi / _KY),
        uniform_faces(cells, -1.0, 1.0),
    )
    solid, thickness = ((None, None),) * 3, 0.0
    if wall is not None:
        thickness = 0.2
        solid = ((None, None), (None, None), (SolidWall(thickness, wall[0], _KAPPA, 1.0, wall[1]),) * 2)
    flow = ChannelProblem(
        grid, (_PERIODIC, _PERIODIC, _WALL), viscosity=_NU, conductivity=1.0 if rate else 0.0,
        magnetic_field=(0.0, math.sqrt(rate), 0.0), dt=float(dt), advection="limited", solid_walls=solid,
        magnetic_damping=damping,
    )  # fmt: skip
    ends = thermal_boundary("temperature", 1.0 + 0.5 * thickness, -0.5 * thickness)  # T_0 at the outer faces
    thermal = ThermalProblem(
        flow, (_PERIODIC, _PERIODIC, ends), 1.0, _KAPPA, _T_REF, joule_heating=True, expansion_coefficient=_BETA_G,
        gravity=(0.0, 0.0, -1.0), advection="van_leer", walls=solid if wall else None,
    )  # fmt: skip
    rest = zero_velocity(flow)
    places, terms, cells_at, shape, heat_terms = _sampled(grid, tuple(v.offset for v in rest))
    blank = Field(jnp.zeros(thermal.thermal_grid.shape), (CENTER,) * 3, thermal.thermal_grid)

    def force(t):
        return tuple(rest[c].replace_data(_momentum_source(terms[c], t, rate)) for c in range(3))

    step = build_thermal_step(thermal, dtype=jnp.float64, force=force)

    def exact_velocity(t):
        return [
            jax.vmap(lambda p, c=c: _velocity(p, t)[c])(points).reshape(s)
            for c, (points, s) in enumerate(places)
        ]

    temperature = np.asarray(jax.vmap(lambda p: _temperature(p, 0.0))(cells_at)).reshape(shape)
    if wall is not None:  # T_0 through the walls, the fluid's own inside
        full = 0.5 * (1.0 - np.asarray(thermal.thermal_grid.centers[2]))[None, None, :] * np.ones(blank.shape)
        full[thermal.fluid_block] = temperature
        temperature = full
    start = tuple(rest[c].replace_data(v) for c, v in enumerate(exact_velocity(0.0)))
    state = initial_thermal_state(thermal, temperature, start)
    steps = round(final / dt)

    @jax.jit
    def march(state):
        def one(_, s):
            heat = blank.replace_data(thermal.on_thermal_grid(_heat_source(heat_terms, s.time, rate)))
            return step(s, heat)[0]

        return jax.lax.fori_loop(0, steps, one, state)

    state = march(state)
    t = float(state.time)
    fields = [np.asarray(v.data) for v in state.velocity]
    fields.append(np.asarray(state.temperature.data)[thermal.fluid_block])
    exact = [np.asarray(v) for v in exact_velocity(t)]
    exact.append(np.asarray(jax.vmap(lambda p: _temperature(p, t))(cells_at)).reshape(shape))
    differences = [f - e for f, e in zip(fields, exact, strict=True)]
    return {
        "time": t,
        "max": [float(np.max(np.abs(d))) for d in differences],
        "rms": [float(np.sqrt(np.mean(d**2))) for d in differences],
        "fields": fields,
    }


def vortex_decay(
    cells_along: int = 16,
    *,
    step_rate: float = 0.02,
    damping: str = "explicit",
    steel: bool = True,
    hartmann: float = 235.0,
    cells: int = 48,
    cells_in_layer: int = 6,
    wall_cells: int = 6,
    wavenumber: float = np.pi / 2,
) -> tuple[float, float]:
    """Return LMhdX's decay rate of ``u_z = U(y) cos(k x)`` between Hartmann walls and the exact one, in ``sigma B^2/rho``.

    Units ``a = rho = sigma = B = 1``, so ``nu = 1/Ha^2`` and ``dt = step_rate``. Walls at ``y = -1, 1`` on
    :func:`lmhdx.grid.wall_resolving_faces` (Case 2's M1: 48 cells, 6 in each Hartmann layer), Case 2's steel in
    ``wall_cells`` cells or insulating; periodic ``x`` over one wavelength in ``cells_along`` cells, periodic ``z`` on
    two. Advection is exactly zero for this flow. The rate is fitted to ``ln E`` over the last half of the decay after
    ``t = 10``, past the Hartmann layers' transient, with ``E = sum u_z^2 dy``.
    """
    enable_x64()
    ratio, thickness = CASE2_WALL if steel else (0.0, 0.0)
    across = wall_resolving_faces(
        cells, -1.0, 1.0, layer_thickness=1.0 / hartmann, cells_in_layer=cells_in_layer, max_ratio=None
    )
    grid = Grid(uniform_faces(cells_along, 0.0, 2 * np.pi / wavenumber), across, uniform_faces(2, 0.0, 1.0))
    walls = (
        (None, None),
        (SolidWall(thickness, wall_cells, 1.0, 1.0, ratio),) * 2 if steel else (None, None),
    )
    walls = (walls[0], walls[1], (None, None))
    flow = ChannelProblem(
        grid, (_PERIODIC, _WALL, _PERIODIC), viscosity=1.0 / hartmann**2, magnetic_field=(0.0, 1.0, 0.0),
        dt=float(step_rate), advection="limited", solid_walls=walls, magnetic_damping=damping,
    )  # fmt: skip
    thermal = ThermalProblem(
        flow, (_PERIODIC, thermal_boundary("heat_flux", conductivity=1.0), _PERIODIC), 1.0, 1.0, 0.0,
        joule_heating=False, walls=walls if steel else None,
    )  # fmt: skip
    exact = vortex_decay_rate(wavenumber, hartmann, wall_ratio=ratio, wall_thickness=thickness)
    horizon, intervals = 10.0 + 3.0 / abs(exact), 40
    steps = max(1, round(horizon / intervals / step_rate))
    rest = zero_velocity(flow)
    x, y = np.asarray(grid.centers[0])[:, None, None], np.asarray(grid.centers[1])[None, :, None]
    profile = (1.0 - np.exp(-(1.0 - np.abs(y)) * hartmann)) * np.cos(wavenumber * x)
    vortex = rest[2].replace_data(jnp.asarray(profile * np.ones(rest[2].shape)))
    state = initial_thermal_state(thermal, 0.0, (rest[0], rest[1], vortex))
    heat = Field(jnp.zeros(thermal.thermal_grid.shape), (CENTER,) * 3, thermal.thermal_grid)
    step = build_thermal_step(thermal, dtype=jnp.float64)
    advance = jax.jit(lambda s: advance_thermal(step, s, heat, steps, checkpoint=False)[0])
    widths = np.asarray(grid.widths[1])[None, :, None]
    times, logs = [], []
    for _ in range(intervals + 1):
        times.append(float(state.time))
        logs.append(math.log(float(np.sum(np.asarray(state.velocity[2].data) ** 2 * widths))))
        state = advance(state)
    times, logs = np.array(times), np.array(logs)
    late = times >= 0.5 * (times[-1] + 10.0)
    return 0.5 * float(np.polyfit(times[late], logs[late], 1)[0]), exact


def hartmann_roll_growth(
    cells_along: int = 32,
    *,
    cells: int = 48,
    cells_in_layer: int = 6,
    layer: int = 32,
    rayleigh: float = 1e5,
    prandtl: float = 0.0196,
    hartmann: float = 235.0,
    step_rate: float = 0.34,
    damping: str = "explicit",
    intervals: int = 40,
) -> tuple[float, float]:
    """Return the growth rate of Benard rolls along the field between insulating Hartmann walls, and the model's.

    ``kappa = 1``, ``nu = Pr``; the layer ``z = -1..1`` in ``layer`` cells (rigid, ``T = 1`` below, 0 above), the
    walls ``y = -1, 1`` as in :func:`vortex_decay`, periodic ``x`` over one wavelength at ``k d = 3.117``; the field
    along ``y`` with ``sigma B^2/rho = Pr Ha^2`` and ``dt`` times it ``step_rate``. Measured by power iteration on
    the thermal step, as the buoyancy record's rolls; the model is Sommeria and Moreau's friction at the roll's ``k``.
    """
    enable_x64()
    k, rate = 3.117 / 2.0, prandtl * hartmann**2
    friction = -rate * vortex_decay_rate(k, hartmann) - prandtl * k**2
    model = convection_growth_rate(k, rayleigh, prandtl, friction=friction).real
    across = wall_resolving_faces(
        cells, -1.0, 1.0, layer_thickness=1.0 / hartmann, cells_in_layer=cells_in_layer, max_ratio=None
    )
    grid = Grid(uniform_faces(cells_along, 0.0, 2 * np.pi / k), across, uniform_faces(layer, -1.0, 1.0))
    dt = step_rate / rate
    flow = ChannelProblem(
        grid, (_PERIODIC, _WALL, _WALL), viscosity=prandtl, magnetic_field=(0.0, math.sqrt(rate), 0.0),
        dt=float(dt), advection="limited", magnetic_damping=damping,
    )  # fmt: skip
    thermal = ThermalProblem(
        flow, (_PERIODIC, thermal_boundary("heat_flux", conductivity=1.0), thermal_boundary("temperature", 1.0, 0.0)),
        1.0, 1.0, 0.5, joule_heating=False, expansion_coefficient=rayleigh * prandtl / 8.0, gravity=(0.0, 0.0, -1.0),
        advection="van_leer",
    )  # fmt: skip
    x, z = np.asarray(grid.centers[0])[:, None, None], np.asarray(grid.centers[2])[None, None, :]
    conduction = np.broadcast_to(0.5 - 0.5 * z, grid.shape)
    rest = zero_velocity(flow)
    base = initial_thermal_state(thermal, conduction, rest)
    disturbed = initial_thermal_state(
        thermal, conduction + 1e-7 * np.cos(k * x) * np.cos(np.pi * z / 2), rest
    )
    step, heat = (
        build_thermal_step(thermal, dtype=jnp.float64),
        Field(jnp.zeros(grid.shape), (CENTER,) * 3, grid),
    )
    steps = max(4, round(0.2 / max(abs(model), 0.05) / dt))
    run = jax.jit(
        lambda a, b: tuple(advance_thermal(step, s, heat, steps, checkpoint=False)[0] for s in (a, b))
    )

    def energy(a, b):
        return float(
            sum(jnp.sum((v.data - u.data) ** 2) for u, v in zip(a.velocity, b.velocity, strict=True))
        )

    rescale = jax.jit(lambda a, b, f: jax.tree_util.tree_map(lambda u, v: u + (v - u) * f, a, b))
    base, disturbed = run(base, disturbed)
    disturbed = rescale(base, disturbed, 1e-7 / math.sqrt(energy(base, disturbed)))
    before, rates = energy(base, disturbed), []
    for _ in range(intervals):
        base, disturbed = run(base, disturbed)
        after = energy(base, disturbed)
        rates.append(0.5 * math.log(after / before) / (steps * dt))
        disturbed = rescale(base, disturbed, math.sqrt(before / after))
        before = energy(base, disturbed)
        if len(rates) > 8 and all(abs(rates[-j] - rates[-j - 1]) < 2e-5 * abs(rates[-1]) for j in (1, 2, 3)):
            break
    return rates[-1], model
