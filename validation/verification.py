"""Code verification of the thermal step: manufactured solutions term by term and all at once, and exact solutions with
the Lorentz force, each against references that share none of the step's operators.

Each runs :func:`lmhdx.thermal.build_thermal_step`, the path of a run in time (Mistrangelo's Case 2):

- :data:`TERMS` holds one manufactured solution per term of the step, with only that term's source besides the time
  derivative and the incompressibility the step always enforces, and ``"coupled"``, every term at once. Units
  ``rho = c_p = sigma = 1``; the field lies along ``y``.
- :func:`manufactured_run` marches one to its final time, ``dt`` falling as ``h^2`` from 0.01 at eight cells, and returns
  each field's max and RMS errors, the final fields, and the error along the term's line.
- :func:`vortex_eigenmode` and :func:`vortex_decay` follow a vortex invariant along the field between Hartmann walls,
  insulating or Case 2's steel resolved in cells, against :func:`validation.shercliff.vortex_decay_rate`.
- :func:`hartmann_roll_growth` measures Benard rolls along the field between insulating Hartmann walls against Sommeria
  and Moreau's model, :func:`validation.thermal.convection_growth_rate` with the friction of the decay reference.

A velocity invariant along the field, ``(d_z psi, 0, -d_x psi)``, has the electromotive force ``B grad psi``, so the
potential ``B psi`` and no current: the balanced motion. One with no vorticity along the field has the current
``sigma u x B``: the damped motion. Both vanish with their wall-normal derivatives at ``z = -1, 1``, so a conducting wall
there carries no current and the potential and current are exact. ``tests/test_steady.py`` checks the coupled case and
the decay; ``scripts/make_showcase_figures.py --only verification`` records them all (docs/validation/verification.md).
"""

from __future__ import annotations

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

__all__ = [
    "CASE2_WALL",
    "TERMS",
    "hartmann_roll_growth",
    "manufactured_run",
    "vortex_decay",
    "vortex_eigenmode",
]

CASE2_WALL = (1.24571, 0.002 / 0.023)  # Case 2's steel: the pinned conductivity ratio and t_w / a
_PERIODIC, _WALL = BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN)
_RATE = 30.0  # sigma B^2 / rho of the Lorentz terms on their own
_FIELDS = ("u_x", "u_y", "u_z")
_SOURCES: dict = {}  # compiled manufactured sources by term, cells, rate and wall


# --------------------------------------------------------------------------------------------- exact fields
def _a(t):
    return 1.0 + 0.5 * jnp.sin(3.0 * t)


def _b(t):
    return 0.5 * jnp.cos(2.0 * t) + 0.2


def _b0(t):  # the coupled case's
    return 0.5 * jnp.cos(2.0 * t)


def _c(t):
    return 0.3 + 0.2 * jnp.sin(4.0 * t)


def _inside(p):
    return jnp.abs(p[2]) <= 1.0


def _curl(p):
    """Divergence free and periodic over 2 pi along x and z and pi along y."""

    def potential(q):
        x, y, z = q
        return jnp.array([
            jnp.sin(2 * y + 0.3) * jnp.cos(z), jnp.cos(z + 0.5) * jnp.sin(x), jnp.sin(x + 0.2) * jnp.cos(2 * y)
        ])  # fmt: skip

    j = jax.jacfwd(potential)(p)
    return jnp.array([j[2, 1] - j[1, 2], j[0, 2] - j[2, 0], j[1, 0] - j[0, 1]])


def _stream(p, along_field=False):
    """``(d_z psi, 0, -d_x psi)`` with ``psi = sin x (1 - z^2)^2``, times ``1 + cos(2y)/2`` unless invariant along y."""

    def psi(q):
        return jnp.sin(q[0]) * (1 - q[2] ** 2) ** 2 * (1.0 if along_field else 1 + 0.5 * jnp.cos(2 * q[1]))

    g = jax.grad(psi)(p)
    return jnp.array([g[2], 0.0, -g[0]])


def _damped(p):
    """No vorticity along y; zero, with its normal derivative, at z = -1, 1, and outside."""
    shape = lambda q: jnp.cos(q[0]) * (1 - q[2] ** 2) ** 3  # noqa: E731
    g, h = jax.grad(shape)(p), jax.hessian(shape)(p)
    return jnp.where(_inside(p), 1.0, 0.0) * jnp.array([
        jnp.cos(2 * p[1]) * g[0], -jnp.sin(2 * p[1]) * (h[0, 0] + h[2, 2]) / 2.0, jnp.cos(2 * p[1]) * g[2]
    ])  # fmt: skip


def _perpendicular(v):
    return v * jnp.array([1.0, 0.0, 1.0])


def _theta(p):
    return jnp.cos(p[0] + 0.4) * (1 + 0.5 * jnp.cos(2 * p[1])) * (1 - p[2] ** 2) ** 2


def _theta0(p):  # the coupled case's, zero in its walls
    return jnp.where(_inside(p), jnp.cos(p[0] + 0.4) * jnp.cos(2 * p[1] + 0.2) * (1 - p[2] ** 2) ** 2, 0.0)


def _box_theta(p):
    x, y, z = p
    return jnp.sin(x + 0.4) * jnp.cos(2 * y + 0.1) * jnp.cos(z + 0.2) + 0.5 * jnp.cos(2 * x) * jnp.sin(z)


def _base(p):
    return 0.5 * (1.0 - p[2])


# Conjugate conduction: T = a(t) cos x F(z) in the fluid (k 0.1, rho c_p 1), a(t) cos x G(z) in walls 0.25 thick
# (k 0.4, rho c_p 2), G linear and zero outside; T and k dT/dz continuous at z = -1, 1.
_KF, _KW, _CW, _TW = 0.1, 0.4, 2.0, 0.25
_ALPHA = 1.0 / (1.0 + 2.0 * _KF * _TW / _KW)


def _conjugate(p, t):
    inside, outside = 1.0 - _ALPHA * p[2] ** 2, 2.0 * _ALPHA * _KF / _KW * (1.0 + _TW - jnp.abs(p[2]))
    return _a(t) * jnp.cos(p[0]) * jnp.where(_inside(p), inside, outside)


def _box(n, ny=None, walls=True):
    z = uniform_faces(n, -1.0, 1.0) if walls else uniform_faces(n, 0.0, 2 * np.pi)
    return Grid(uniform_faces(n, 0.0, 2 * np.pi), uniform_faces(n // 2 if ny is None else ny, 0.0, np.pi), z)


def _walls(n, thickness, conductivity, capacity, electric):
    return (
        (None, None),
        (None, None),
        (SolidWall(thickness, max(1, n // 8), conductivity, capacity, electric),) * 2,
    )


# --------------------------------------------------------------------------------------------- the terms
TERMS = {
    "advection": dict(
        title="Momentum advection", math=r"(\mathbf{u}\cdot\nabla)\mathbf{u}", show=("u_x", 0, (None, 0.6, 1.0)),
        grid=lambda n: _box(n, walls=False), periodic=True, nu=1e-4, advection="limited",
        u=lambda p, t: _a(t) * _curl(p)),
    "viscosity": dict(
        title="Viscous diffusion", math=r"\nu\nabla^{2}\mathbf{u}", show=("u_x", 0, (None, 0.6, 0.5)),
        u=lambda p, t: _a(t) * _stream(p)),
    "pressure": dict(
        title="Pressure gradient", math=r"-\nabla p/\rho", show=("p", 0, (None, 0.6, 0.5)),
        u=lambda p, t: _a(t) * _stream(p),
        p=lambda p, t: _a(t) * jnp.cos(p[0] + 0.3) * (p[2] ** 3 / 3 - p[2]) * (1 + 0.3 * jnp.sin(2 * p[1]))),
    "lorentz_balanced": dict(
        title="Lorentz force, balanced part",
        math=r"\mathbf{j}=\sigma(-\nabla\phi+\mathbf{u}\times\mathbf{B})=\mathbf{0}", show=("u_x", 0, (None, 0.0, 0.5)),
        grid=lambda n: _box(n, ny=2), rate=_RATE, sizes=(8, 16, 32, 64, 128),
        u=lambda p, t: _a(t) * _stream(p, along_field=True), lorentz=lambda p, t: jnp.zeros(3)),
    "lorentz_damped": dict(
        title="Lorentz force, damped part",
        math=r"\mathbf{j}\times\mathbf{B}/\rho=-(\sigma B^{2}/\rho)\,\mathbf{u}_{\perp}", show=("u_x", 0, (None, 0.3, 0.5)),
        rate=_RATE, u=lambda p, t: _b(t) * _damped(p),
        lorentz=lambda p, t: -_RATE * _b(t) * _perpendicular(_damped(p))),
    "lorentz_conducting": dict(
        title="Lorentz force with conducting walls",
        math=r"\nabla\cdot(\sigma\nabla\phi)=\nabla\cdot(\sigma\,\mathbf{u}\times\mathbf{B})",
        show=("u_x", 0, (None, 0.3, 0.5)), rate=_RATE, walls=lambda n: _walls(n, 0.25, 0.1, 1.0, 2.0),
        thermal_z=thermal_boundary("heat_flux", conductivity=0.1),
        u=lambda p, t: _a(t) * _stream(p, along_field=True) + _b(t) * _damped(p),
        lorentz=lambda p, t: -_RATE * _b(t) * _perpendicular(_damped(p))),
    "buoyancy": dict(
        title="Buoyancy", math=r"-\beta\,(T-T_{\mathrm{r}})\,\mathbf{g}", show=("u_z", 0, (None, 0.6, 0.0)),
        beta_g=1.0, u=lambda p, t: jnp.zeros(3), T=lambda p, t: _base(p) + _c(t) * _theta(p)),
    "conduction": dict(
        title="Heat conduction", math=r"\kappa\nabla^{2}T", show=("T", 0, (None, 0.6, 0.5)), hold=True,
        u=lambda p, t: jnp.zeros(3), T=lambda p, t: _base(p) + _c(t) * _theta(p)),
    "heat_advection": dict(
        title="Heat advection", math=r"\mathbf{u}\cdot\nabla T", show=("T", 0, (None, 0.6, 1.0)),
        grid=lambda n: _box(n, walls=False), periodic=True, hold=True, kappa=1e-4, heat_advection="van_leer",
        u=lambda p, t: _curl(p), T=lambda p, t: (1.0 + 0.5 * jnp.sin(4.0 * t)) * _box_theta(p)),
    "joule": dict(
        title="Joule heating", math=r"|\mathbf{j}|^{2}/(\sigma\rho c_{p})", show=("T", 0, (None, 0.3, 0.5)), hold=True,
        rate=_RATE, joule=lambda p, t: _RATE * jnp.sum(_perpendicular(_damped(p)) ** 2),
        u=lambda p, t: _damped(p), T=lambda p, t: _base(p) + _c(t) * _theta(p)),
    "conjugate": dict(
        title="Conjugate conduction", math=r"\nabla\cdot(k\nabla T),\ [T]=[k\,\partial_{z}T]=0",
        show=("T", 2, (0.0, 0.0, None)), grid=lambda n: _box(n, ny=2), hold=True, kappa=_KF, walls_in_norm=True,
        walls=lambda n: _walls(n, _TW, _KW, _CW, 1.0), thermal_z=thermal_boundary("temperature", 0.0, 0.0),
        k_of=lambda p: jnp.where(_inside(p), _KF, _KW), c_of=lambda p: jnp.where(_inside(p), 1.0, _CW),
        sizes=(8, 16, 32, 64, 128), u=lambda p, t: jnp.zeros(3), T=_conjugate),
    "coupled": dict(
        title="All terms together, conducting walls", math="", show=("u_x", 0, (None, 0.3, 0.5)),
        nu=0.05, kappa=0.05, rate=10.0, advection="limited", heat_advection="van_leer", beta_g=1.0, coupled=True,
        u=lambda p, t: _a(t) * _stream(p, along_field=True) * jnp.where(_inside(p), 1.0, 0.0) + _b0(t) * _damped(p),
        T=lambda p, t: _base(p) + _c(t) * _theta0(p)),
}  # fmt: skip


def _points(grid: Grid, offset) -> tuple[jnp.ndarray, tuple[int, ...], list]:
    axes = [np.asarray(grid.centers[a] if offset[a] == CENTER else grid.faces[a]) for a in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return jnp.asarray(np.stack([m.ravel() for m in mesh], axis=1)), mesh[0].shape, axes


def _line(error, axes, along, at):
    index = [slice(None)] * 3
    for a in range(3):
        if a != along:
            index[a] = int(np.argmin(np.abs(axes[a] - at[a])))
    return np.asarray(axes[along]), np.asarray(error[tuple(index)])


def manufactured_run(
    term: str,
    cells: int,
    *,
    dt: float | None = None,
    final: float = 0.2,
    rate: float | None = None,
    damping: str = "explicit",
    wall: tuple[int, float] | None = None,
) -> dict:
    """March one of :data:`TERMS` on ``cells`` cells to ``final``; return its errors, fields and line.

    ``cells`` along ``x`` (one period, ``2 pi``) and ``z``, half as many along ``y`` (``pi``) unless the term says
    otherwise; ``dt`` defaults to ``0.01 (8/cells)^2``. Every source is taken at the step's old time, as the step takes
    its force. ``rate`` (``sigma B^2/rho``), ``damping`` and ``wall`` (``(cells, sigma_w/sigma)`` for conducting walls
    0.2 thick at ``z = -1, 1``, with the fluid's ``k`` and ``rho c_p``) apply to ``"coupled"``. The errors (exact less
    computed) are on every velocity face, every fluid cell (and wall cell for ``"conjugate"``), and the pressure less its
    mean where the term has one. ``fields`` are the computed velocity components and the fluid's temperature.
    """
    enable_x64()
    spec = TERMS[term]
    coupled = spec.get("coupled", False)
    rate = rate if coupled and rate is not None else spec.get("rate", 0.0)
    grid = spec.get("grid", _box)(cells)
    dt = 0.01 * (8.0 / cells) ** 2 if dt is None else dt
    nu, kappa = spec.get("nu", 0.1), spec.get("kappa", 0.1)
    walls, thickness = spec.get("walls", lambda n: ((None, None),) * 3)(cells), 0.0
    if coupled and wall is not None:
        thickness = 0.2
        walls = ((None, None), (None, None), (SolidWall(thickness, wall[0], kappa, 1.0, wall[1]),) * 2)
    has_walls = any(w is not None for pair in walls for w in pair)
    periodic = spec.get("periodic", False)
    flow = ChannelProblem(
        grid, (_PERIODIC, _PERIODIC, _PERIODIC if periodic else _WALL), viscosity=nu,
        conductivity=1.0 if (rate or has_walls) else 0.0, magnetic_field=(0.0, math.sqrt(rate), 0.0), dt=float(dt),
        advection=spec.get("advection", "off"), solid_walls=walls, magnetic_damping=damping,
    )  # fmt: skip
    ends = spec.get("thermal_z", thermal_boundary("temperature", 1.0 + 0.5 * thickness, -0.5 * thickness))
    thermal = ThermalProblem(
        flow, (_PERIODIC, _PERIODIC, _PERIODIC if periodic else ends), 1.0, kappa, 0.5,
        joule_heating="joule" in spec or coupled, expansion_coefficient=spec.get("beta_g", 0.0),
        gravity=(0.0, 0.0, -1.0), advection=spec.get("heat_advection", "off"), walls=walls if has_walls else None,
    )  # fmt: skip
    u_fn, t_fn = spec["u"], spec.get("T", lambda p, t: 0.0 * p[0])
    lorentz, joule = spec.get("lorentz"), spec.get("joule")
    if coupled:
        lorentz = lambda p, t: -rate * _b0(t) * _perpendicular(_damped(p))  # noqa: E731
        joule = lambda p, t: rate * _b0(t) ** 2 * jnp.sum(_perpendicular(_damped(p)) ** 2)  # noqa: E731
    beta_g, transport = spec.get("beta_g", 0.0), spec.get("advection", "off") != "off"

    def momentum(p, t, c):
        s = jax.jacfwd(u_fn, argnums=1)(p, t)[c] - nu * jnp.trace(jax.hessian(lambda q: u_fn(q, t)[c])(p))
        if transport:
            s = s + (jax.jacfwd(u_fn)(p, t) @ u_fn(p, t))[c]
        if "p" in spec:
            s = s + jax.grad(spec["p"])(p, t)[c]
        if lorentz is not None:
            s = s - lorentz(p, t)[c]
        if beta_g and c == 2:
            s = s - beta_g * (t_fn(p, t) - 0.5)
        return s

    k_of, c_of = spec.get("k_of", lambda p: kappa), spec.get("c_of", lambda p: 1.0)
    carried = spec.get("heat_advection", "off") != "off"

    def heat(p, t):
        q = c_of(p) * jax.jacfwd(t_fn, argnums=1)(p, t) - k_of(p) * jnp.trace(
            jax.hessian(lambda r: t_fn(r, t))(p)
        )
        if carried:
            q = q + c_of(p) * jax.grad(t_fn)(p, t) @ u_fn(p, t)
        return q if joule is None else q - joule(p, t)

    rest = zero_velocity(flow)
    places = [_points(grid, v.offset) for v in rest]
    cells_at, cshape, caxes = _points(thermal.thermal_grid, (CENTER,) * 3)
    blank = Field(jnp.zeros(cshape), (CENTER,) * 3, thermal.thermal_grid)

    # The sources compile once per term, mesh, field and walls, and serve every time step and dt: the march only
    # passes their arrays to the step, which compiles light.
    key = (term, cells, rate, wall)
    if key not in _SOURCES:
        while len(_SOURCES) >= 8:
            _SOURCES.pop(next(iter(_SOURCES)))
        _SOURCES[key] = (
            jax.jit(lambda t: tuple(
                jax.vmap(lambda p, c=c: momentum(p, t, c))(where).reshape(shape)
                for c, (where, shape, _) in enumerate(places)
            )),
            jax.jit(lambda t: jax.vmap(lambda p: heat(p, t))(cells_at).reshape(cshape)),
        )  # fmt: skip
    momentum_at, heat_at = _SOURCES[key]
    hold, current = spec.get("hold", False), {}
    step = build_thermal_step(
        thermal, dtype=jnp.float64, hold_flow=hold, force=None if hold else (lambda t: current["force"])
    )

    @jax.jit
    def advance(state, forcing, heating):
        if not hold:  # the force the step asks for in this trace
            current["force"] = tuple(rest[c].replace_data(f) for c, f in enumerate(forcing))
        return step(state, blank.replace_data(heating))[0]

    def exact_velocity(t):
        return [
            np.asarray(jax.vmap(lambda p, c=c: u_fn(p, t)[c])(where)).reshape(shape)
            for c, (where, shape, _) in enumerate(places)
        ]

    start = tuple(rest[c].replace_data(jnp.asarray(v)) for c, v in enumerate(exact_velocity(0.0)))
    state = initial_thermal_state(
        thermal, np.asarray(jax.vmap(lambda p: t_fn(p, 0.0))(cells_at)).reshape(cshape), start
    )
    steps = round(final / dt)

    for index in range(steps):
        time = index * dt
        state = advance(state, () if hold else momentum_at(time), heat_at(time))
    t = float(state.time)
    errors, axes, fields = {}, {}, []
    for c, (name, exact) in enumerate(zip(_FIELDS, exact_velocity(t), strict=True)):
        fields.append(np.asarray(state.velocity[c].data))
        errors[name], axes[name] = exact - fields[-1], places[c][2]
    temperature = np.asarray(state.temperature.data)
    fields.append(temperature[thermal.fluid_block])
    block = (slice(None),) * 3 if spec.get("walls_in_norm") else thermal.fluid_block
    exact_t = np.asarray(jax.vmap(lambda p: t_fn(p, t))(cells_at)).reshape(cshape)
    errors["T"], axes["T"] = (
        (exact_t - temperature)[block],
        [axis[b] for axis, b in zip(caxes, block, strict=True)],
    )
    if "p" in spec:
        where, shape, axes["p"] = _points(grid, (CENTER,) * 3)
        exact_p, computed = (
            np.asarray(jax.vmap(lambda p: spec["p"](p, t))(where)).reshape(shape),
            np.asarray(state.pressure.data),
        )
        errors["p"] = (exact_p - exact_p.mean()) - (computed - computed.mean())
    name, along, at = spec["show"]
    return {
        "time": t, "dt": dt, "steps": steps, "fields": fields,
        "max": {k: float(np.max(np.abs(e))) for k, e in errors.items()},
        "rms": {k: float(np.sqrt(np.mean(e**2))) for k, e in errors.items()},
        "line": _line(errors[name], axes[name], along, at),
    }  # fmt: skip


def _hartmann_duct(
    cells_along, hartmann, cells, cells_in_layer, wall_cells, wavenumber, step_rate, damping, steel
):
    ratio, thickness = CASE2_WALL if steel else (0.0, 0.0)
    across = wall_resolving_faces(
        cells, -1.0, 1.0, layer_thickness=1.0 / hartmann, cells_in_layer=cells_in_layer, max_ratio=None
    )
    grid = Grid(uniform_faces(cells_along, 0.0, 2 * np.pi / wavenumber), across, uniform_faces(2, 0.0, 1.0))
    walls = (
        ((None, None), (SolidWall(thickness, wall_cells, 1.0, 1.0, ratio),) * 2, (None, None))
        if steel
        else None
    )
    flow = ChannelProblem(
        grid, (_PERIODIC, _WALL, _PERIODIC), viscosity=1.0 / hartmann**2, magnetic_field=(0.0, 1.0, 0.0),
        dt=float(step_rate), advection="limited", solid_walls=walls or ((None, None),) * 3, magnetic_damping=damping,
    )  # fmt: skip
    thermal = ThermalProblem(
        flow, (_PERIODIC, thermal_boundary("heat_flux", conductivity=1.0), _PERIODIC), 1.0, 1.0, 0.0,
        joule_heating=False, walls=walls,
    )  # fmt: skip
    heat = Field(jnp.zeros(thermal.thermal_grid.shape), (CENTER,) * 3, thermal.thermal_grid)
    return grid, flow, thermal, heat, ratio, thickness


def vortex_eigenmode(cells_along: int = 16, *, hartmann: float = 235.0, time: float = 5.0) -> dict:
    """Start from the exact eigenmode ``u_z = U(y) cos(k x)`` between Case 2's steel Hartmann walls (``k = pi/2``,
    M1's 48 cells across, 6 in each layer and wall, the explicit force at ``dt`` 0.02 in ``rho/(sigma B^2)``) and
    return its errors against ``U(y) cos(k x) e^{s t}`` at ``time``, with the line along ``x`` at ``y = 0``."""
    from scipy.interpolate import BarycentricInterpolator

    enable_x64()
    k = np.pi / 2
    grid, flow, thermal, heat, ratio, thickness = _hartmann_duct(
        cells_along, hartmann, 48, 6, 6, k, 0.02, "explicit", True
    )
    rate, nodes, shape = vortex_decay_rate(
        k, hartmann, 160, wall_ratio=ratio, wall_thickness=thickness, mode=True
    )
    x, y = np.asarray(grid.centers[0]), np.asarray(grid.centers[1])
    profile = BarycentricInterpolator(nodes, shape)(y)[None, :, None] * np.cos(k * x)[:, None, None]
    rest = zero_velocity(flow)
    vortex = rest[2].replace_data(jnp.asarray(profile * np.ones(rest[2].shape)))
    steps = round(time / 0.02)
    step = build_thermal_step(thermal, dtype=jnp.float64)
    state = initial_thermal_state(thermal, 0.0, (rest[0], rest[1], vortex))
    state = jax.jit(lambda s: advance_thermal(step, s, heat, steps, checkpoint=False)[0])(state)
    error = profile * np.exp(rate * float(state.time)) - np.asarray(state.velocity[2].data)
    return {
        "time": float(state.time), "dt": 0.02, "steps": steps,
        "max": {"u_z": float(np.max(np.abs(error)))}, "rms": {"u_z": float(np.sqrt(np.mean(error**2)))},
        "line": (x, error[:, int(np.argmin(np.abs(y))), 0]),
    }  # fmt: skip


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
    grid, flow, thermal, heat, ratio, thickness = _hartmann_duct(
        cells_along, hartmann, cells, cells_in_layer, wall_cells, wavenumber, step_rate, damping, steel
    )
    exact = vortex_decay_rate(wavenumber, hartmann, wall_ratio=ratio, wall_thickness=thickness)
    horizon, intervals = 10.0 + 3.0 / abs(exact), 40
    steps = max(1, round(horizon / intervals / step_rate))
    rest = zero_velocity(flow)
    x, y = np.asarray(grid.centers[0])[:, None, None], np.asarray(grid.centers[1])[None, :, None]
    profile = (1.0 - np.exp(-(1.0 - np.abs(y)) * hartmann)) * np.cos(wavenumber * x)
    vortex = rest[2].replace_data(jnp.asarray(profile * np.ones(rest[2].shape)))
    state = initial_thermal_state(thermal, 0.0, (rest[0], rest[1], vortex))
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
