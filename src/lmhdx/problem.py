"""What is solved: :class:`Problem`, its boundaries, and the presets that fill one.

A :class:`Problem` is a frozen dataclass and a registered pytree. Its leaves are the inputs the
kernels differentiate: the ``forcing``, the ``flow_rate``, the field amplitude (:mod:`lmhdx.fields`),
the Q2D state and coefficients, a heat source and the initial temperature. The grid, boundary kinds,
walls, material properties and scheme names are static, so ``jax.grad`` never returns a silent zero
for one; properties and wall conductances become leaves when the kernels take them traced (plan H4.2).

:func:`lmhdx.solve` translates a problem to :class:`~lmhdx.core3d.ChannelProblem`,
:class:`~lmhdx.thermal.ThermalProblem`, :class:`~lmhdx.pipe.PipeProblem` or
:class:`~lmhdx.q2d.Q2DProblem`; the presets return the grids and fields of the older constructors.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from math import cosh, pi, sinh, sqrt

import numpy as np

from .axial import axial_faces, open_duct
from .cases import WallLayer
from .core3d import ChannelProblem
from .fields import Fringe, Uniform, host, is_traced, primal, pytree
from .grid import DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition, Grid, axis
from .pipe import pipe_grid
from .q2d import make_q2d_case
from .thermal import ThermalProblem, thermal_boundary

__all__ = [
    "Open",
    "Problem",
    "Q2D",
    "Sides",
    "Thermal",
    "Wall",
    "channel",
    "duct",
    "fringe_duct",
    "pipe",
    "q2d",
]


def _frozen(value):
    """Return an array-like as nested tuples, so a boundary stays hashable."""
    return value if value is None or not np.ndim(value) else tuple(map(_frozen, value))


@dataclass(frozen=True)
class Wall:
    """A no-slip wall at one end of an axis, or at both.

    ``conductance`` is the thin-wall ``c = sigma_w t_w / (sigma L)``; ``layers``
    resolve the wall in cells instead, a tuple of :class:`lmhdx.cases.WallLayer`
    from the fluid outward (absolute conductivity, divided by the fluid's here).
    ``temperature`` or ``heat_flux`` (outward conductive flux, positive cools; a
    number or an array over the two other axes) set the thermal condition; neither
    is adiabatic. Both are static until the thermal kernels take traced data.
    """

    conductance: float = 0.0
    layers: tuple[WallLayer, ...] = ()
    temperature: float | None = None
    heat_flux: float | None = None

    def __post_init__(self) -> None:
        for name in ("temperature", "heat_flux"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        if self.temperature is not None and self.heat_flux is not None:
            raise ValueError("a wall takes a temperature or a heat flux, not both")


@dataclass(frozen=True)
class Sides:
    """Different walls at the two ends of an axis."""

    lower: Wall | str = "wall"
    upper: Wall | str = "wall"


@dataclass(frozen=True)
class Open:
    """The inflow-outflow axis (x): inflow at the lower end, free outflow at the upper.

    ``profile=None`` takes the fully developed profile at ``Problem.flow_rate``; an
    array over the cross-section is an inlet of its own.
    """

    profile: tuple | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", _frozen(self.profile))


@pytree(
    ("heat_source", "initial_temperature"),
    (
        "specific_heat",
        "thermal_conductivity",
        "reference_temperature",
        "expansion_coefficient",
        "gravity",
        "joule_heating",
        "advection",
        "outlet_backflow",
    ),
)
@dataclass(frozen=True, eq=False)
class Thermal:
    """Heat capacity, conductivity, Boussinesq buoyancy and heating, as :class:`lmhdx.thermal.ThermalProblem`."""

    specific_heat: float
    thermal_conductivity: float
    reference_temperature: float
    expansion_coefficient: float = 0.0
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joule_heating: bool = True
    advection: str = "upwind"
    outlet_backflow: str = "error"
    heat_source: float = 0.0
    initial_temperature: float = 0.0


@pytree(
    ("forcing", "flow_rate", "B", "thermal"),
    ("grid", "bc", "density", "viscosity", "conductivity", "advection"),
)
@dataclass(frozen=True, eq=False)
class Problem:
    """A duct, channel or pipe: grid, boundaries, material, field and drive.

    ``bc`` has one entry per axis, a 3-tuple or ``dict(x=, y=, z=)``, each
    ``"periodic"``, ``"wall"``, a :class:`Wall`, :class:`Sides` or :class:`Open`.
    ``forcing`` is the body force (a pressure gradient, positive drives the flow)
    and ``flow_rate`` the volumetric flux that sets the drive instead (a periodic
    Stokes duct) or the inlet (an :class:`Open` axis); one of them. A polar grid is a pipe: axes
    ``(r, theta, z)``, ``"wall"`` at ``r``, unit material, the field's strength the
    transverse one. ``thermal`` adds heat transport.
    """

    grid: Grid
    bc: tuple = ("periodic", "wall", "wall")
    density: float = 1.0
    viscosity: float = 1.0
    conductivity: float = 1.0
    B: object = Uniform(0.0, 0.0, 0.0)
    forcing: tuple[float, float, float] = (0.0, 0.0, 0.0)
    flow_rate: float | None = None
    advection: str = "off"
    thermal: Thermal | None = None

    def __post_init__(self) -> None:
        if isinstance(self.bc, dict):
            object.__setattr__(self, "bc", tuple(self.bc[name] for name in "xyz"))

    def replace(self, **changes) -> Problem:
        return dataclasses.replace(self, **changes)

    def ends(self, index: int):
        """Return axis ``index`` as ``None`` (periodic), an :class:`Open`, or its two :class:`Wall` ends."""
        entry = self.bc[index]
        if entry == "periodic":
            return None
        if isinstance(entry, Open):
            return entry
        if isinstance(entry, Sides):
            return tuple(Wall() if end == "wall" else end for end in (entry.lower, entry.upper))
        if entry == "wall" or isinstance(entry, Wall):
            return (Wall() if entry == "wall" else entry,) * 2
        raise ValueError(f"bc[{index}] must be 'periodic', 'wall', a Wall, Sides or Open, got {entry!r}")


@pytree(("initial_vorticity", "forcing", "viscosity", "hartmann_friction"), ("length",))
@dataclass(frozen=True, eq=False)
class Q2D:
    """The periodic Sommeria-Moreau vorticity problem: state, source, viscosity and Hartmann friction.

    All four are leaves, differentiable through :func:`lmhdx.q2d.evolve_q2d`. ``dt`` and
    ``steps`` belong to the :class:`lmhdx.solver.Solver`.
    """

    initial_vorticity: object
    forcing: object = None
    viscosity: float = 1.0e-2
    hartmann_friction: float = 0.1
    length: tuple[float, float] = (2.0 * pi, 2.0 * pi)


def _layers(wall: Wall, fluid: float):
    """Return a resolved wall as per-cell ``(ratios, widths)`` from the fluid outward, or ``None``."""
    if not wall.layers:
        return None
    ratios = tuple(layer.conductivity / fluid for layer in wall.layers for _ in range(layer.cells))
    widths = tuple(layer.thickness / layer.cells for layer in wall.layers for _ in range(layer.cells))
    return ratios, widths


def channel_kernel(problem: Problem, dt: float = 1.0, precision: str = "state"):
    """Return ``(kernel, forcing, scale, rate)``: the :class:`ChannelProblem` and what rides beside it.

    ``forcing`` is ``None`` when the body force is a number, which the kernel then carries; a traced one
    is an argument. ``scale`` multiplies the imposed field. A periodic duct driven by ``flow_rate`` is the
    unit-drive kernel, which :func:`lmhdx.solver.solve` rescales; an open duct whose Stokes solution is linear in a
    traced ``flow_rate`` is built at its primal, and ``rate`` is the factor that restores the derivative.
    """
    if problem.grid.is_polar:
        raise ValueError("a polar grid is a pipe, which has no ChannelProblem")
    conditions, conductance, layers, opened = [], [], [], None
    for index in range(3):
        ends = problem.ends(index)
        wall = ends is not None and not isinstance(ends, Open)
        if isinstance(ends, Open):
            if index:
                raise NotImplementedError("the Open axis is x")
            opened = ends
        conditions.append(BoundaryCondition(NEUMANN if wall else PERIODIC))
        conductance.append(0.0)
        layers.append(None)
        if wall:
            first, second = (float(host(w.conductance, "a wall conductance")) for w in ends)
            if first != second:
                raise NotImplementedError("a thin wall takes one conductance on both ends of an axis")
            conductance[-1] = first
            if any(w.layers for w in ends):
                layers[-1] = tuple(_layers(w, float(problem.conductivity)) for w in ends)
    field, scale = problem.B.sample(problem.grid)
    peak = np.max(np.abs(field.components), axis=(1, 2, 3)) if hasattr(field, "components") else np.abs(field)
    # u_x B_across drives a current along a periodic axis that nothing closes: the Hartmann channel.
    if any(conditions[axis].is_periodic and peak[1 + (axis == 1)] for axis in (1, 2)):
        raise NotImplementedError(
            "a field across a periodic axis (a Hartmann channel) drives a net current along it; "
            "the zero-mean-current condition arrives with plan step H2.5"
        )
    drive, traced = tuple(problem.forcing), is_traced(problem.forcing)
    if problem.flow_rate is not None and not traced and any(drive):
        raise ValueError("give forcing or flow_rate, not both")
    unit = problem.flow_rate is not None and opened is None
    kernel = ChannelProblem(
        problem.grid,
        tuple(conditions),
        density=float(host(problem.density, "the density")),
        viscosity=float(host(problem.viscosity, "the viscosity")),
        conductivity=float(host(problem.conductivity, "the conductivity")),
        magnetic_field=field,
        forcing=(1.0, 0.0, 0.0) if unit else (0.0, 0.0, 0.0) if traced else tuple(float(f) for f in drive),
        dt=dt,
        advection=problem.advection,
        wall_conductance=tuple(conductance),
        precision=precision,
        wall_layers=tuple(layers),
    )
    rate = 1.0
    if opened is not None and opened.profile is None:
        reference = primal(problem.flow_rate, "the flow rate")
        if problem.advection != "off":
            host(problem.flow_rate, "the flow rate")
        kernel = open_duct(kernel, reference)
        rate = problem.flow_rate / reference if is_traced(problem.flow_rate) else 1.0
    elif opened is not None:
        inlet = BoundaryCondition(DIRICHLET, lower=np.asarray(opened.profile), upper_kind=NEUMANN)
        kernel = dataclasses.replace(kernel, conditions=(inlet, *kernel.conditions[1:]))
    return kernel, (drive if traced and not unit else None), scale, rate


def thermal_kernel(problem: Problem, flow: ChannelProblem) -> ThermalProblem:
    """Return the :class:`ThermalProblem` of ``problem.thermal`` on ``flow``."""
    thermal = problem.thermal
    conductivity = float(host(thermal.thermal_conductivity, "thermal_conductivity"))
    conditions = []
    for index in range(3):
        ends = problem.ends(index)
        if isinstance(ends, Open):
            raise NotImplementedError("heat transport in an open duct arrives with plan H7.4")
        if ends is None:
            conditions.append(BoundaryCondition(PERIODIC))
            continue
        (kind, lower), (upper_kind, upper) = (
            ("temperature", w.temperature) if w.temperature is not None else ("heat_flux", w.heat_flux or 0.0)
            for w in ends
        )
        conditions.append(
            thermal_boundary(
                kind,
                lower,
                upper,
                upper_kind=upper_kind if upper_kind != kind else None,
                conductivity=conductivity,
            )
        )
    numbers = {
        n: float(host(getattr(thermal, n), n))
        for n in ("specific_heat", "reference_temperature", "expansion_coefficient")
    }
    return ThermalProblem(
        flow,
        tuple(conditions),
        thermal_conductivity=conductivity,
        joule_heating=thermal.joule_heating,
        gravity=tuple(float(g) for g in thermal.gravity),
        advection=thermal.advection,
        outlet_backflow=thermal.outlet_backflow,
        **numbers,
    )


def _cross_section(ha: float, cells: int, cells_in_layer: int):
    """The y and z faces of the unit-half-width section, resolving the Hartmann (``1/Ha``) and side (``1/sqrt(Ha)``) layers."""
    layers = (1.0 / ha, 1.0 / sqrt(ha)) if ha else (np.inf, np.inf)
    wall = dict(cells_in_layer=cells_in_layer, max_ratio=None)
    return [axis(cells, -1.0, 1.0, **(wall | {"layer_thickness": t} if t < 1.0 else {})) for t in layers]


def duct(
    ha: float,
    cells: int = 48,
    *,
    wall_conductance: float = 0.0,
    cells_in_layer: int = 6,
    walls: dict | None = None,
    flow_rate: float | None = None,
    **fields: object,
) -> Problem:
    """A square duct of unit half-width at Hartmann number ``ha``, field along y, periodic in x, as :func:`duct_problem`.

    The section resolves the layers that exist, ``1/Ha`` at the Hartmann walls (y) and ``1/sqrt(Ha)``
    at the others, in ``cells_in_layer`` cells each. ``wall_conductance`` is the thin conductance of the
    Hartmann walls; ``walls=dict(y=Wall(...), z=...)`` replaces ``bc`` entries (resolved Hunt walls). The
    drive is a unit pressure gradient unless ``flow_rate`` (the volumetric flux) is given. Any other
    :class:`Problem` field is a keyword.
    """
    ha = float(ha)
    if ha < 0.0:
        raise ValueError("ha must not be negative")
    y, z = _cross_section(ha, cells, cells_in_layer)
    nu = fields.get("density", 1.0) * fields.get("viscosity", 1.0) / fields.get("conductivity", 1.0)
    bc = {"y": Wall(wall_conductance), "z": "wall"} | (walls or {})
    return Problem(
        Grid(axis(1, 0.0, 1.0), y, z),
        bc=("periodic", bc["y"], bc["z"]),
        B=Uniform(0.0, ha * sqrt(nu), 0.0),
        flow_rate=flow_rate,
        **{
            "conductivity": 1.0 if ha else 0.0,
            "forcing": (0.0 if flow_rate is not None else 1.0, 0.0, 0.0),
            **fields,
        },
    )


# Domains (Lx, Ly) / pi in half-widths: Boeck, Krasnov and Zienicke 2007 (JFM 572, 179), Table 1, keyed
# by (R, Ha), and Krasnov, Zikanov and Boeck 2011, Table 1, spanwise field, keyed by (Re_av, Ha).
_BKZ_DOMAINS = {
    (450, 10): (5, 2),
    (450, 15): (10 / 3, 4 / 3),
    (450, 20): (3, 6 / 5),
    (450, 30): (10 / 3, 16 / 15),
    (500, 10): (4, 2),
    (500, 15): (10 / 3, 4 / 3),
    (500, 20): (8 / 3, 16 / 15),
    (500, 30): (2, 4 / 5),
    (700, 10): (10 / 3, 4 / 3),
    (700, 15): (8 / 3, 16 / 15),
    (700, 20): (2, 4 / 5),
    (700, 30): (2, 4 / 5),
    (900, 10): (3, 6 / 5),
    (900, 15): (12 / 5, 1),
    (900, 20): (2, 4 / 5),
    (900, 25): (2, 4 / 5),
}
_KZB_DOMAINS = {(6667, 30): (2, 1)}


def bulk_ratio(ha: float) -> float:
    """The exact laminar Hartmann ratio of the bulk to the centreline velocity, ``(cosh Ha - sinh Ha / Ha) / (cosh Ha - 1)``."""
    return (cosh(ha) - sinh(ha) / ha) / (cosh(ha) - 1.0)


def channel(
    re: float | None = None,
    ha: float = 0.0,
    *,
    R: float | None = None,
    field: str = "wall_normal",
    cells: tuple[int, int, int] = (64, 64, 64),
    lengths: tuple[float, float] | None = None,
    stretch: dict | None = None,
    **fields: object,
) -> Problem:
    """A plane channel at constant bulk velocity, periodic in x and y, walls normal to z, half-width one.

    The unit system has density, conductivity and bulk velocity one, so ``viscosity =
    1 / Re_av`` and the field is ``Ha sqrt(viscosity)``. Give the bulk Reynolds number
    ``re = Re_av`` or, for a Hartmann channel, Boeck, Krasnov and Zienicke's ``R = U0 delta / nu``
    (``U0`` the laminar centreline velocity, ``delta = 1/Ha``), which gives ``Re_av = R Ha Uav/U0`` with
    :func:`bulk_ratio`. ``field`` is ``"wall_normal"`` (a Hartmann channel, B along z) or
    ``"spanwise"`` (B along y). ``lengths = (Lx, Ly)`` default to the row of the BKZ 2007 Table 1 or
    of Krasnov, Zikanov and Boeck 2011 (spanwise, Re 6667, Ha 30); other cases give them. The wall
    axis uses ``stretch`` (default the KZB blend ``dict(blend=0.96)``; ``dict(beta=2.0)`` is their
    tanh grid). ``flow_rate`` is the bulk velocity times the section, ``2 Ly``.
    """
    if field not in ("wall_normal", "spanwise"):
        raise ValueError("field is 'wall_normal' or 'spanwise'")
    if (re is None) == (R is None):
        raise ValueError("give re or R, not both")
    if R is not None:
        if not ha > 0.0:
            raise ValueError("R needs ha > 0")
        re = R * ha * bulk_ratio(ha)
    elif ha > 0.0 and field == "wall_normal":
        R = re / (ha * bulk_ratio(ha))
    if lengths is None:
        table, target = (_BKZ_DOMAINS, R) if field == "wall_normal" else (_KZB_DOMAINS, re)
        row = (
            table.get((round(target), ha)) if target and abs(target - round(target)) < 1e-3 * target else None
        )
        if row is None:
            raise ValueError(
                f"no tabulated domain for re={re}, ha={ha}, field={field!r}: give lengths=(Lx, Ly)"
            )
        lengths = tuple(pi * length for length in row)
    viscosity = 1.0 / re
    strength = ha * sqrt(viscosity)
    direction = (0.0, 0.0, 1.0) if field == "wall_normal" else (0.0, 1.0, 0.0)
    nx, ny, nz = cells
    grid = Grid(
        axis(nx, 0.0, lengths[0]),
        axis(ny, 0.0, lengths[1]),
        axis(nz, -1.0, 1.0, **(stretch or dict(blend=0.96))),
    )
    fields = {
        "B": Uniform(strength=strength, direction=direction),
        "flow_rate": 2.0 * lengths[1],
        "viscosity": viscosity,
        **fields,
    }
    return Problem(grid, bc=("periodic", "periodic", "wall"), **fields)


def fringe_duct(
    ha: float,
    wall_conductance: float = 0.0,
    half_length: float = 3.0,
    upstream: float = 15.0,
    downstream: float = 10.0,
    spacing: float = 0.25,
    cells: int = 24,
    cells_in_layer: int = 6,
    flow_rate: float = 4.0,
    solenoidal: bool = False,
    advection: str = "off",
    growth: float = 1.1,
    **fields: object,
) -> Problem:
    """The ANL fringe (TM-228) in a square duct with an inlet and an outlet, as :func:`lmhdx.axial.fringe_duct`.

    The field falls as ``B_y = Ha (1 - sin(pi x / 2 x0)) / 2`` over ``|x| <= half_length``
    and is uniform outside, with ``upstream`` and ``downstream`` half-widths of buffer; the
    conductance is on all four walls; ``flow_rate`` 4 is a unit mean velocity.
    """
    y, z = _cross_section(float(ha), cells, cells_in_layer)
    window = (-half_length - min(half_length, upstream), half_length)
    x = axial_faces(-half_length - upstream, half_length + downstream, window, spacing, growth)
    return Problem(
        Grid(x, y, z),
        bc=(Open(), Wall(wall_conductance), Wall(wall_conductance)),
        conductivity=1.0 if ha else 0.0,
        B=Fringe(float(ha), half_length, 0.0, solenoidal),
        flow_rate=flow_rate,
        advection=advection,
        **fields,
    )


def pipe(
    ha: float,
    radial: int = 48,
    azimuthal: int = 64,
    *,
    wall_conductance: float = 0.0,
    cells_in_layer: int = 6,
    forcing: float = 1.0,
) -> Problem:
    """A circular pipe of unit radius in a transverse field of strength ``ha``, as :func:`lmhdx.pipe.pipe_problem`."""
    return Problem(
        pipe_grid(radial, azimuthal, float(ha), cells_in_layer=cells_in_layer),
        bc=(Wall(wall_conductance), "periodic", "periodic"),
        B=Uniform(strength=float(ha)),
        forcing=(0.0, 0.0, forcing),
    )


def q2d(
    shape: tuple[int, int] = (64, 64),
    length: tuple[float, float] = (2.0 * pi, 2.0 * pi),
    mode: tuple[int, int] = (1, 1),
    amplitude: float = 1.0,
    viscosity: float = 1.0e-2,
    hartmann_friction: float = 0.1,
) -> Q2D:
    """A Taylor-Green decay case with an exponential rate, the state of :func:`lmhdx.q2d.make_q2d_case`."""
    case = make_q2d_case(
        shape=shape,
        length=length,
        mode=mode,
        amplitude=amplitude,
        viscosity=viscosity,
        hartmann_friction=hartmann_friction,
    )
    return Q2D(case.initial_vorticity, None, viscosity, hartmann_friction, length)
