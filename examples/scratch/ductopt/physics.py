"""Core evaluator: dimensionless duct meshes, cached and jitted, and the physical
pumping-power objective built on top of them.

Derivation (stage_2_plan.txt Section 4.1, re-derived and verified here against
LMX's own convention). A fully developed Stokes-limit duct of half-width ``a``
along the field, aspect ``beta = b/a``, dynamic viscosity ``mu``, conductivity
``sigma`` and field ``B`` obeys

    0 = mu * lap(u) + Lorentz(u, B) + F,      F = -dp/dx  (Pa/m)

Nondimensionalising lengths by ``a`` and the velocity by the Poiseuille scale
``F a^2 / mu`` turns this into LMX's own dimensionless duct problem: half-width
1, density = viscosity = conductivity = 1, field magnitude = Ha = B a
sqrt(sigma/mu), forcing = 1. Its solution gives ``q(Ha, beta) =
channel_flow_response(...).flow_per_unit_drive``, the dimensionless flow at
unit drive. Matching the two integrals over the cross-section gives, exactly,

    Q_phys = F * a^4 / mu * q(Ha, beta)   =>   F = mu * Q_phys / (a^4 * q)

which is what every station-pressure-gradient call below computes. This was
cross-checked against ``examples/scratch/duct_opt_probes/probe_design_law.py``
and ``probe_rect.py`` (both dimensionless, mu = a = 1 there) and reproduces
their numbers to full precision (see stage_2_plan.txt P6/P9).

The mesh a station needs depends on the aspect ratio ``beta`` ALONE, not on the
physical size ``a`` -- the whole size dependence is absorbed into a traced
``field_scale`` (LMX's own mechanism), so one mesh per ``beta`` serves every
area a design search visits, as long as it is built for the LARGEST Hartmann
number that area range can reach (``ha_mesh``, with a small safety margin).
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass

import jax
import jax.monitoring
import numpy as np

from lmhdx.core3d import ChannelProblem
from lmhdx.fully_developed import channel_flow_rate, channel_flow_response
from lmhdx.grid import PERIODIC, BoundaryCondition, Grid, uniform_faces, wall_resolving_faces
from lmhdx.steady import solve_steady_state

_WALL = BoundaryCondition("neumann")
_MESH_MARGIN = 1.02  # ha_mesh built 2% above the largest Ha a design box can reach


def _round_key(value: float, digits: int) -> float:
    return float(round(float(value), digits))


@functools.lru_cache(maxsize=256)
def _build_problem(
    beta: float, ha_mesh: float, cells: int, cells_in_layer: int, centre: float = 0.0, tilt: float = 0.0
) -> ChannelProblem:
    """One insulating rectangular duct, half-width 1 along B, aspect ``beta``.

    With ``centre`` > 0 the mesh is the one built for aspect ``centre``, with its z faces scaled to
    ``beta`` (O12): the discrete W is then a smooth function of ``beta``, which a mesh rebuilt at
    every ``beta`` (its own clustering and CG floor) is not.

    Field magnitude is fixed at ``ha_mesh``; every physical Hartmann number the
    design box can reach is then a ``magnetic_field_scale <= 1`` of it. The
    z-clustering length follows the probes: the side layer, capped so a very
    slender duct (small ``beta``) still gets several cells across its own
    half-width. A tilted field (``tilt`` = B_p / B_T) puts a normal component on the long walls, whose layer
    is ``1 / (ha_mesh * tilt)``: the z clustering length is the thinner of that and the side layer.
    """
    y = wall_resolving_faces(
        cells, -1.0, 1.0, layer_thickness=1.0 / ha_mesh, cells_in_layer=cells_in_layer, max_ratio=None
    )
    side_layer = min(1.0 / np.sqrt(ha_mesh), 0.25 * (centre or beta), 1.0 / (ha_mesh * tilt) if tilt else np.inf)
    z = wall_resolving_faces(
        cells, -(centre or beta), centre or beta, layer_thickness=side_layer, cells_in_layer=cells_in_layer,
        max_ratio=None,
    )
    if centre:
        z = z * (beta / centre)
    grid = Grid(uniform_faces(1, 0.0, 1.0), y, z)
    return ChannelProblem(
        grid=grid,
        conditions=(BoundaryCondition(PERIODIC), _WALL, _WALL),
        conductivity=1.0,
        magnetic_field=(0.0, ha_mesh, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
    )


@functools.lru_cache(maxsize=256)
def _compiled_value_and_grad(beta: float, ha_mesh: float, cells: int, cells_in_layer: int, centre: float = 0.0):
    """Return a jitted ``field_scale -> (q, dq/d field_scale)`` for one mesh."""
    problem = _build_problem(beta, ha_mesh, cells, cells_in_layer, centre)

    def q(field_scale):
        return channel_flow_response(problem, magnetic_field_scale=field_scale).flow_per_unit_drive

    return jax.jit(jax.value_and_grad(q)), jax.jit(q)


def mesh_key(beta: float, ha_mesh: float, centre: float = 0.0) -> tuple[float, float]:
    """Round (beta, ha_mesh) to cache-friendly keys. The scaled-mesh family (``centre`` > 0) keeps beta to
    13 digits: the 6-digit key perturbs the geometry by 3e-6 and W(w) by 5e-6, which hides dW/dw."""
    return _round_key(beta, 13 if centre else 6), _round_key(ha_mesh, 3)


def q_and_grad_ha(
    beta: float, ha_target: float, ha_mesh: float, cells: int, cells_in_layer: int, centre: float = 0.0
) -> tuple[float, float]:
    """Return ``(q(Ha_target, beta), dq/dHa)`` on the mesh built for ``ha_mesh``."""
    beta_k, ha_mesh_k = mesh_key(beta, ha_mesh, centre)
    value_and_grad_fn, _ = _compiled_value_and_grad(beta_k, ha_mesh_k, cells, cells_in_layer, centre)
    scale = ha_target / ha_mesh_k
    q, dq_ds = value_and_grad_fn(scale)
    return float(q), float(dq_ds) / ha_mesh_k


def q_only(
    beta: float, ha_target: float, ha_mesh: float, cells: int, cells_in_layer: int, centre: float = 0.0
) -> float:
    beta_k, ha_mesh_k = mesh_key(beta, ha_mesh, centre)
    _, q_fn = _compiled_value_and_grad(beta_k, ha_mesh_k, cells, cells_in_layer, centre)
    return float(q_fn(ha_target / ha_mesh_k))


_COMPILES = {"count": 0, "seconds": 0.0}


def _count_compile(event: str, duration: float, **_) -> None:
    if event == "/jax/core/compile/backend_compile_duration":
        _COMPILES["count"] += 1
        _COMPILES["seconds"] += duration


jax.monitoring.register_event_duration_secs_listener(_count_compile)


def cache_stats() -> dict:
    """Meshes built (cache misses) and XLA compiles in this process so far (R7)."""
    info = _compiled_value_and_grad.cache_info()
    return {"hits": info.hits, "meshes_built": info.misses, "compiles": _COMPILES["count"],
            "compile_s": _COMPILES["seconds"]}


@dataclass(frozen=True)
class Station:
    """One length of duct at (locally) uniform field ``B`` (T)."""

    B: float
    L: float
    label: str = ""
    R: float = 0.0  # major radius (m) of a radial run's station, for gamma = a/R; 0 = no axial variation


@dataclass(frozen=True)
class DesignBox:
    """The optimization's fixed physical parameters and bounds."""

    Q: float  # m^3/s
    mu: float  # Pa s (dynamic viscosity)
    sigma: float  # S/m
    V_min: float  # m/s
    V_max: float  # m/s
    beta_lo: float
    beta_hi: float
    stations: tuple[Station, ...]

    @property
    def u_lo(self) -> float:
        return float(np.log(self.Q / self.V_max))

    @property
    def u_hi(self) -> float:
        return float(np.log(self.Q / self.V_min))

    def a_of(self, u: float, w: float) -> float:
        beta = np.exp(w)
        return float(np.sqrt(np.exp(u) / (4.0 * beta)))

    def ha_mesh_for_beta(self, beta: float) -> float:
        """Mesh field magnitude: the largest Ha the whole box can reach at this beta."""
        a_max = float(np.sqrt(np.exp(self.u_hi) / (4.0 * beta)))
        b_max = max(station.B for station in self.stations)
        return _MESH_MARGIN * b_max * a_max * float(np.sqrt(self.sigma / self.mu))


def station_pressure_gradient(
    box: DesignBox, station: Station, a: float, beta: float, cells: int, cells_in_layer: int, centre: float = 0.0
) -> dict:
    """Return ``F_k = -dp/dx`` (Pa/m) at one station and ``d ln F_k / d ln a`` at fixed beta.

    ``F_k = mu Q / (a^4 q_k)``, so ``d ln F_k/d ln a = -4 - d ln q_k / d ln a``,
    and ``d ln q_k / d ln a = (dq_k/dHa_k) Ha_k / q_k`` since ``Ha_k`` is
    exactly proportional to ``a`` at fixed field and beta.
    """
    ha_mesh = box.ha_mesh_for_beta(centre or beta)
    ha_k = station.B * a * np.sqrt(box.sigma / box.mu)
    q_k, dq_dha = q_and_grad_ha(beta, ha_k, ha_mesh, cells, cells_in_layer, centre)
    F_k = box.mu * box.Q / (a**4 * q_k)
    dlnq_dlna = dq_dha * ha_k / q_k
    dlnF_dlna = -4.0 - dlnq_dlna
    return {
        "F": F_k,
        "dlnF_dlna": dlnF_dlna,
        "Ha": ha_k,
        "q": q_k,
        "ha_mesh": ha_mesh,
    }


def objective(box: DesignBox, u: float, w: float, cells: int, cells_in_layer: int, centre: float = 0.0) -> dict:
    """Return W(u, w), its exact gradient in u (at fixed w), and per-station data.

    ``W = sum_k L_k Q F_k`` is the total pumping power; ``dW/du = sum_k L_k Q
    F_k dlnF_dlna_k * 0.5`` because ``d ln a / du = 1/2`` (``a = sqrt(A/(4
    beta))``, ``A = e^u``).
    """
    beta = float(np.exp(w))
    a = box.a_of(u, w)
    W = 0.0
    dW_du = 0.0
    delta_p = 0.0
    per_station = []
    for station in box.stations:
        info = station_pressure_gradient(box, station, a, beta, cells, cells_in_layer, centre)
        Wk = station.L * box.Q * info["F"]
        dWk_du = Wk * info["dlnF_dlna"] * 0.5
        W += Wk
        dW_du += dWk_du
        delta_p += station.L * info["F"]
        per_station.append({"label": station.label, "B": station.B, "L": station.L, **info})
    V = box.Q / np.exp(u)
    return {
        "u": u,
        "w": w,
        "beta": beta,
        "a": a,
        "b": beta * a,
        "A": float(np.exp(u)),
        "V": V,
        "W": W,
        "dW_du": dW_du,
        "delta_p": delta_p,
        "stations": per_station,
    }


def objective_value(box: DesignBox, u: float, w: float, cells: int, cells_in_layer: int, centre: float = 0.0) -> float:
    return objective(box, u, w, cells, cells_in_layer, centre)["W"]


def dW_dw_central(
    box: DesignBox, u: float, w: float, cells: int, cells_in_layer: int, h: float = 1.0e-3
) -> tuple[float, float, float]:
    """Central-difference ``dW/dw`` and ``d^2W/dw^2`` from three neighbouring meshes."""
    Wp = objective_value(box, u, w + h, cells, cells_in_layer)
    W0 = objective_value(box, u, w, cells, cells_in_layer)
    Wm = objective_value(box, u, w - h, cells, cells_in_layer)
    dW_dw = (Wp - Wm) / (2.0 * h)
    d2W_dw2 = (Wp - 2.0 * W0 + Wm) / (h * h)
    return dW_dw, d2W_dw2, W0


def spectral_value(box: DesignBox, u: float, w: float, points: int = 48) -> float:
    """``W(u, w)`` with each station's flow from the independent spectral reference (5A.C4, exit (e))."""
    from validation.shercliff import quadrant_flow_rate

    beta, a = float(np.exp(w)), box.a_of(u, w)
    return sum(
        s.L * box.Q * box.mu * box.Q
        / (a**4 * 4.0 * beta * quadrant_flow_rate(s.B * a * np.sqrt(box.sigma / box.mu), points, aspect=beta))
        for s in box.stations
    )


def tilted_flow(
    beta: float, ha: float, tilt: float, cells: int = 48, cells_in_layer: int = 6, centre: float = 0.0,
    ha_mesh: float = 0.0,
) -> float:
    """Flow per unit drive of a duct whose field of magnitude ``ha`` is tilted by ``arctan(tilt)`` in the
    cross-section (C8, the tilt law): the damped preconditioner of a field off the mesh axes. With ``centre``
    the mesh is the O12 scaled family built for ``ha_mesh`` (default ``ha``)."""
    problem = _build_problem(*mesh_key(beta, ha_mesh or ha, centre), cells, cells_in_layer, centre, tilt)
    theta = float(np.arctan(tilt))
    problem = dataclasses.replace(problem, magnetic_field=(0.0, ha * np.cos(theta), ha * np.sin(theta)))
    solution = solve_steady_state(problem, forcing=(1.0, 0.0, 0.0))
    return float(channel_flow_rate(problem, solution.velocity[0].data[0]))
