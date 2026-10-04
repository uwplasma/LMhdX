"""Fully developed duct cases: their schema, materials, meshes and the solve entry point.

A ``CaseSpec`` (geometry, regions, field, conditions, solver and output choices)
is solved on the staggered core by :mod:`lmhdx.fully_developed`: steady cases by
:func:`~lmhdx.fully_developed.solve_fully_developed`, transient ones by
:func:`~lmhdx.fully_developed.solve_fully_developed_transient`. This module also
holds the material and nondimensional relations (Hartmann, Reynolds and
interaction numbers, wall conductance ratios and wall stacks), the structured
mesh a solution reports, tabulated imposed fields, TOML run configurations and
the streaming solver log.
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, TextIO

import jax.numpy as jnp
import numpy as np
from jax import config as jax_config
from jax.scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from .core3d import ChannelProblem
    from .q2d import Q2DProblem, Q2DResult
    from .steady import SteadySolution

MU0 = 4.0e-7 * math.pi


def _require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def dynamic_to_kinematic_viscosity(dynamic_viscosity: float, density: float) -> float:
    """Return kinematic viscosity ``nu = mu / rho`` in ``m^2/s``."""

    _require_positive("density", density)
    if dynamic_viscosity < 0.0:
        raise ValueError("dynamic_viscosity must be non-negative")
    return float(dynamic_viscosity) / float(density)


def kinematic_to_dynamic_viscosity(kinematic_viscosity: float, density: float) -> float:
    """Return dynamic viscosity ``mu = rho nu`` in ``Pa s``."""

    _require_positive("density", density)
    if kinematic_viscosity < 0.0:
        raise ValueError("kinematic_viscosity must be non-negative")
    return float(kinematic_viscosity) * float(density)


def hartmann_number(
    *,
    magnetic_field: float,
    length_scale: float,
    conductivity: float,
    density: float,
    kinematic_viscosity: float,
) -> float:
    """Return ``Ha = B a sqrt(sigma / (rho nu))``."""

    for name, value in (
        ("length_scale", length_scale),
        ("conductivity", conductivity),
        ("density", density),
        ("kinematic_viscosity", kinematic_viscosity),
    ):
        _require_positive(name, value)
    return (
        abs(float(magnetic_field))
        * float(length_scale)
        * math.sqrt(float(conductivity) / (float(density) * float(kinematic_viscosity)))
    )


def magnetic_field_from_hartmann(
    *,
    hartmann: float,
    length_scale: float,
    conductivity: float,
    density: float,
    kinematic_viscosity: float,
) -> float:
    """Return ``B`` from a target Hartmann number using kinematic viscosity."""

    for name, value in (
        ("length_scale", length_scale),
        ("conductivity", conductivity),
        ("density", density),
        ("kinematic_viscosity", kinematic_viscosity),
    ):
        _require_positive(name, value)
    return float(hartmann) / (
        float(length_scale) * math.sqrt(float(conductivity) / (float(density) * float(kinematic_viscosity)))
    )


def reynolds_number(*, velocity: float, length_scale: float, kinematic_viscosity: float) -> float:
    """Return ``Re = U a / nu``."""

    _require_positive("length_scale", length_scale)
    _require_positive("kinematic_viscosity", kinematic_viscosity)
    return abs(float(velocity)) * float(length_scale) / float(kinematic_viscosity)


def interaction_parameter(
    *,
    magnetic_field: float,
    length_scale: float,
    conductivity: float,
    density: float,
    velocity: float,
) -> float:
    """Return ``N = sigma B^2 a / (rho U)``."""

    for name, value in (
        ("length_scale", length_scale),
        ("conductivity", conductivity),
        ("density", density),
        ("velocity", abs(velocity)),
    ):
        _require_positive(name, value)
    return (
        float(conductivity)
        * float(magnetic_field) ** 2
        * float(length_scale)
        / (float(density) * abs(float(velocity)))
    )


def magnetic_reynolds_number(
    *,
    velocity: float,
    length_scale: float,
    conductivity: float,
    magnetic_permeability: float = MU0,
) -> float:
    """Return ``Rm = mu0 sigma U a``."""

    for name, value in (
        ("length_scale", length_scale),
        ("conductivity", conductivity),
        ("magnetic_permeability", magnetic_permeability),
    ):
        _require_positive(name, value)
    return float(magnetic_permeability) * float(conductivity) * abs(float(velocity)) * float(length_scale)


def wall_conductance_ratio(
    *,
    wall_conductivity: float,
    wall_thickness: float,
    fluid_conductivity: float,
    length_scale: float,
) -> float:
    """Return thin-wall tangential conductance ratio ``c``."""

    for name, value in (
        ("wall_thickness", wall_thickness),
        ("fluid_conductivity", fluid_conductivity),
        ("length_scale", length_scale),
    ):
        _require_positive(name, value)
    if wall_conductivity < 0.0:
        raise ValueError("wall_conductivity must be non-negative")
    return (
        float(wall_conductivity) * float(wall_thickness) / (float(fluid_conductivity) * float(length_scale))
    )


def normal_leakage_ratio(
    *,
    coating_conductivity: float,
    coating_thickness: float,
    fluid_conductivity: float,
    length_scale: float,
) -> float:
    """Return normal shunt ratio ``g_perp``."""

    for name, value in (
        ("coating_thickness", coating_thickness),
        ("fluid_conductivity", fluid_conductivity),
        ("length_scale", length_scale),
    ):
        _require_positive(name, value)
    if coating_conductivity < 0.0:
        raise ValueError("coating_conductivity must be non-negative")
    return (
        float(coating_conductivity)
        * float(length_scale)
        / (float(fluid_conductivity) * float(coating_thickness))
    )


@dataclass(frozen=True)
class WallLayer:
    """One solid layer in a fluid-facing electrical wall stack."""

    name: str
    conductivity: float
    thickness: float
    cells: int = 1


def _validate_layers(layers: Sequence[WallLayer]) -> None:
    if not layers:
        raise ValueError("at least one wall layer is required")
    for layer in layers:
        if layer.thickness <= 0.0:
            raise ValueError(f"wall layer {layer.name!r} has non-positive thickness")
        if layer.conductivity < 0.0:
            raise ValueError(f"wall layer {layer.name!r} has negative conductivity")
        if layer.cells < 0:
            raise ValueError(f"wall layer {layer.name!r} has negative cells")


def tangential_stack_conductance_ratio(
    layers: Sequence[WallLayer], *, fluid_conductivity: float, length_scale: float
) -> float:
    """Return the thin-wall tangential ratio for layers in parallel."""

    _validate_layers(layers)
    _require_positive("fluid_conductivity", fluid_conductivity)
    _require_positive("length_scale", length_scale)
    surface_conductance = sum(float(layer.conductivity) * float(layer.thickness) for layer in layers)
    return surface_conductance / (float(fluid_conductivity) * float(length_scale))


def normal_stack_leakage_ratio(
    layers: Sequence[WallLayer], *, fluid_conductivity: float, length_scale: float
) -> float:
    """Return the normal leakage ratio for layers in series."""

    _validate_layers(layers)
    _require_positive("fluid_conductivity", fluid_conductivity)
    _require_positive("length_scale", length_scale)
    if any(layer.conductivity <= 0.0 for layer in layers):
        return 0.0
    resistance = sum(float(layer.thickness) / float(layer.conductivity) for layer in layers)
    return float(length_scale) / (float(fluid_conductivity) * resistance)


def effective_pinhole_conductance_ratio(
    *, intact_conductance_ratio: float, metal_conductance_ratio: float, pinhole_fraction: float
) -> float:
    """Return the area-weighted conductance ratio for a pinholed coating."""

    if not 0.0 <= pinhole_fraction <= 1.0:
        raise ValueError("pinhole_fraction must be between 0 and 1")
    if intact_conductance_ratio < 0.0 or metal_conductance_ratio < 0.0:
        raise ValueError("conductance ratios must be non-negative")
    return (1.0 - float(pinhole_fraction)) * float(intact_conductance_ratio) + float(
        pinhole_fraction
    ) * float(metal_conductance_ratio)


def equivalent_single_layer(layers: Sequence[WallLayer], *, name: str = "equivalent_wall") -> WallLayer:
    """Return one layer with the same tangential surface conductance."""

    _validate_layers(layers)
    total_thickness = sum(float(layer.thickness) for layer in layers)
    surface_conductance = sum(float(layer.conductivity) * float(layer.thickness) for layer in layers)
    return WallLayer(
        name=name,
        conductivity=surface_conductance / total_thickness,
        thickness=total_thickness,
        cells=sum(max(int(layer.cells), 0) for layer in layers),
    )


def nested_wall_layer_resolution_summary(
    layers: Sequence[WallLayer], *, minimum_cells_per_layer: int = 3
) -> dict[str, object]:
    """Return mesh-resolution metrics for a wall-layer stack."""

    _validate_layers(layers)
    if minimum_cells_per_layer < 1:
        raise ValueError("minimum_cells_per_layer must be positive")
    rows = [
        {
            "name": layer.name,
            "thickness": float(layer.thickness),
            "conductivity": float(layer.conductivity),
            "cells": int(layer.cells),
            "cell_width": float(layer.thickness) / max(int(layer.cells), 1),
            "cell_count_pass": int(layer.cells) >= minimum_cells_per_layer,
        }
        for layer in layers
    ]
    minimum = min(int(layer.cells) for layer in layers)
    return {
        "layer_count": len(layers),
        "total_thickness": sum(float(layer.thickness) for layer in layers),
        "total_cells": sum(int(layer.cells) for layer in layers),
        "minimum_cells_per_layer": minimum,
        "minimum_required_cells_per_layer": int(minimum_cells_per_layer),
        "resolution_pass": minimum >= minimum_cells_per_layer,
        "layers": rows,
    }


# The structured mesh a solution reports, and tabulated imposed fields (formerly ``lmhdx.cases``).


@dataclass(frozen=True)
class StructuredMesh:
    x_faces: jnp.ndarray
    y_faces: jnp.ndarray
    z_faces: jnp.ndarray
    geometry: str = "rect_duct"
    point_coordinates: jnp.ndarray | None = None
    fluid_mask: jnp.ndarray | None = None
    sigma: jnp.ndarray | None = None
    region_ids: jnp.ndarray | None = None
    region_names: tuple[str, ...] = ()

    @property
    def nx(self) -> int:
        return int(self.x_faces.size - 1)

    @property
    def ny(self) -> int:
        return int(self.y_faces.size - 1)

    @property
    def nz(self) -> int:
        return int(self.z_faces.size - 1)

    @property
    def x_centers(self) -> jnp.ndarray:
        return 0.5 * (self.x_faces[:-1] + self.x_faces[1:])

    @property
    def y_centers(self) -> jnp.ndarray:
        return 0.5 * (self.y_faces[:-1] + self.y_faces[1:])

    @property
    def z_centers(self) -> jnp.ndarray:
        return 0.5 * (self.z_faces[:-1] + self.z_faces[1:])

    @property
    def dx(self) -> jnp.ndarray:
        return jnp.diff(self.x_faces)

    @property
    def dy(self) -> jnp.ndarray:
        return jnp.diff(self.y_faces)

    @property
    def dz(self) -> jnp.ndarray:
        return jnp.diff(self.z_faces)

    @property
    def yz_shape(self) -> tuple[int, int]:
        return (self.ny, self.nz)


def _validated_faces(faces: jnp.ndarray, *, name: str) -> jnp.ndarray:
    values = jnp.asarray(faces, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.size < 2:
        raise ValueError(f"{name} must contain at least two faces")
    if bool(jnp.any(jnp.diff(values) <= 0.0)):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def generate_rect_duct_mesh_from_faces(
    *,
    y_faces: jnp.ndarray,
    z_faces: jnp.ndarray,
    length: float = 1.0,
    nx: int = 1,
) -> StructuredMesh:
    """Build a rectangular duct mesh from explicit cross-section faces."""

    y = _validated_faces(y_faces, name="y_faces")
    z = _validated_faces(z_faces, name="z_faces")
    x_faces = jnp.linspace(0.0, length, nx + 1)
    return StructuredMesh(x_faces=x_faces, y_faces=y, z_faces=z, geometry="rect_duct")


def make_divergence_free_cross_section_field(
    *,
    width: float,
    height: float,
    base_bz: float,
    perturbation: float = 0.15,
):
    """Return an analytic cross-sectional field with dBy/dy + dBz/dz = 0."""

    def field(y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        y_hat = 2.0 * y / width
        z_hat = 2.0 * z / height
        phase_y = 0.5 * jnp.pi * y_hat
        phase_z = 0.5 * jnp.pi * z_hat
        by = perturbation * base_bz * jnp.sin(phase_y) * jnp.cos(phase_z)
        bz = base_bz * (1.0 - perturbation * (height / width) * jnp.cos(phase_y) * jnp.sin(phase_z))
        bx = jnp.zeros_like(by)
        return jnp.stack([bx, by, bz], axis=-1)

    return field


def sample_cross_section_field(
    field_fn,
    *,
    width: float,
    height: float,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.linspace(-0.5 * width, 0.5 * width, ny)
    z = np.linspace(-0.5 * height, 0.5 * height, nz)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    field = np.asarray(field_fn(jnp.asarray(yy), jnp.asarray(zz)), dtype=float)
    return y, z, field


def write_tabulated_field_npz(
    path: str | Path,
    *,
    x: np.ndarray | None = None,
    y: np.ndarray,
    z: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
) -> Path:
    payload: dict[str, np.ndarray] = {
        "y": np.asarray(y, dtype=float),
        "z": np.asarray(z, dtype=float),
        "bx": np.asarray(bx, dtype=float),
        "by": np.asarray(by, dtype=float),
        "bz": np.asarray(bz, dtype=float),
    }
    if x is not None:
        payload["x"] = np.asarray(x, dtype=float)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    return out


def load_tabulated_field(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    if source.suffix.lower() != ".npz":
        raise ValueError("Tabulated magnetic fields currently use NPZ files with x/y/z and bx/by/bz arrays")
    with np.load(source) as payload:
        data = {key: np.asarray(payload[key], dtype=float) for key in payload.files}
    required_2d = {"y", "z", "bx", "by", "bz"}
    if not required_2d.issubset(data):
        raise ValueError(
            "Tabulated magnetic-field NPZ must contain either y/z/bx/by/bz or x/y/z/bx/by/bz arrays"
        )
    axes = tuple(data[name] for name in (("x", "y", "z") if "x" in data else ("y", "z")))
    if any(a.ndim != 1 or a.size < 2 or not np.isfinite(a).all() or not (np.diff(a) > 0).all() for a in axes):
        raise ValueError(
            "Field-table axes must be finite, strictly increasing vectors with at least two points"
        )
    if any(
        data[key].shape != tuple(a.size for a in axes) or not np.isfinite(data[key]).all()
        for key in ("bx", "by", "bz")
    ):
        raise ValueError("Field-table components must be finite and match the coordinate-grid shape")
    return data


def sample_tabulated_cross_section_field(
    path: str | Path,
    *,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    data = load_tabulated_field(path)
    if "x" in data:
        raise ValueError("3D tabulated field needs an x coordinate; use sample_tabulated_field_volume(...)")
    return _interpolate_tabulated_field(data, y=y, z=z)


def _interpolate_tabulated_field(data: dict[str, np.ndarray], **coordinates: np.ndarray) -> np.ndarray:
    axes = tuple(np.asarray(data[name], dtype=float) for name in coordinates)
    values = tuple(np.asarray(value, dtype=float) for value in coordinates.values())
    points = np.stack([value.reshape(-1) for value in values], axis=-1)
    if any(not np.isfinite(q).all() or np.any(q < a[0]) or np.any(q > a[-1]) for a, q in zip(axes, values)):
        raise ValueError("Field-table queries must be finite and inside the tabulated domain")
    field = np.stack([data[key] for key in ("bx", "by", "bz")], axis=-1)
    sampled = RegularGridInterpolator(axes, field, bounds_error=False, fill_value=jnp.nan)(points)
    return np.asarray(sampled).reshape((*values[0].shape, 3))


# The case schema, solution containers, run configuration and solver log (formerly ``lmhdx.cases``).

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only used on Python < 3.11
    import tomli as tomllib


RegionKind = Literal["fluid", "solid"]
GeometryKind = Literal["rect_duct", "layered_duct"]
WallModel = Literal["auto", "thin", "resolved"]
SolverKind = Literal["fully_developed_inductionless"]
SolveMode = Literal["steady", "transient"]
BoundaryKind = Literal[
    "no_slip",
    "insulating",
    "conducting_wall",
    "inlet_velocity",
    "inlet_flow_rate",
    "outlet_pressure",
    "imposed_current_density",
]
MagneticFieldKind = Literal["constant", "analytic", "tabulated"]
PotentialSolverKind = Literal["auto", "jacobi", "cg", "cg_volume"]
PreconditionerKind = Literal["none", "jacobi"]
TimeSchemeKind = Literal["implicit_euler", "crank_nicolson"]
CouplingAccelerationKind = Literal["none", "aitken", "anderson"]


@dataclass(frozen=True)
class RegionSpec:
    """Material-region specification.

    ``viscosity`` is kinematic viscosity ``nu`` in ``m^2/s``. Dynamic
    viscosity ``mu`` should be converted before constructing a case.
    """

    name: str
    kind: RegionKind
    conductivity: float
    density: float | None = None
    viscosity: float | None = None
    wall_thickness: float | None = None


@dataclass(frozen=True)
class BoundaryCondition:
    name: str
    kind: BoundaryKind
    value: float | tuple[float, float, float] | None = None
    region: str | None = None
    axis: str | None = None
    side: str | None = None


@dataclass(frozen=True)
class MagneticFieldSpec:
    kind: MagneticFieldKind
    value: tuple[float, float, float] | None = None
    fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None
    table_path: str | None = None
    ramp_start: float = 0.0
    ramp_duration: float = 0.0


@dataclass(frozen=True)
class SolverConfig:
    kind: SolverKind = "fully_developed_inductionless"
    mode: SolveMode = "steady"
    preconditioner: PreconditionerKind = "jacobi"
    time_scheme: TimeSchemeKind = "implicit_euler"
    coupling_iterations: int = 12
    coupling_tolerance: float = 1e-8
    coupling_acceleration: CouplingAccelerationKind = "none"
    coupling_min_relaxation: float = 0.05
    coupling_max_relaxation: float = 100.0
    coupling_history_depth: int = 6
    coupling_regularization: float = 1.0e-8
    coupling_damping: float = 1.0


@dataclass(frozen=True)
class TimeStepperConfig:
    dt: float
    t_final: float
    max_steps: int
    potential_iterations: int = 400
    potential_tolerance: float | None = None
    potential_relaxation: float = 1.0
    potential_solver: PotentialSolverKind = "auto"
    steady_tolerance: float = 1e-8
    steady_potential_tolerance: float | None = None
    relaxation: float = 0.35
    velocity_update_limit: float = 1e-3


@dataclass(frozen=True)
class OutputSpec:
    directory: str | None = None
    write_paraview: bool = True
    write_csv_profiles: bool = True
    write_npz: bool = True
    write_json_summary: bool = True
    write_plots: bool = False
    copy_input_file: bool = True
    write_stride: int = 1
    history_stride: int = 0


@dataclass(frozen=True)
class GeometrySpec:
    kind: GeometryKind
    width: float
    height: float
    length: float = 1.0
    nx: int = 1
    ny: int = 64
    nz: int = 64
    wall_thickness: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    wall_cells: tuple[int, int, int, int] = (0, 0, 0, 0)
    target_ha: float | None = None
    target_side_layer: float | None = None
    hartmann_layer_cells: int | None = None
    wall_model: WallModel = "auto"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    geometry: GeometrySpec
    regions: tuple[RegionSpec, ...]
    magnetic_field: MagneticFieldSpec
    boundary_conditions: tuple[BoundaryCondition, ...]
    time_stepper: TimeStepperConfig
    solver: SolverConfig = field(default_factory=SolverConfig)
    output: OutputSpec = field(default_factory=OutputSpec)
    forcing: float = 1.0
    initial_velocity: float = 0.0
    reference_pressure_gradient: float = -1.0
    reference_phi_cell: tuple[int, int] = (0, 0)
    notes: str = ""
    dtype: str = "float64"

    def __post_init__(self):
        if self.dtype not in ("float32", "float64"):
            raise ValueError("case dtype must be 'float32' or 'float64'")
        if self.dtype == "float64" and not jax_config.x64_enabled:
            warnings.warn(
                "Call lmhdx.enable_x64() before constructing float64 cases or meshes; "
                "case construction is activating float64 implicitly.",
                DeprecationWarning,
                stacklevel=2,
            )
            from . import enable_x64

            enable_x64()
        # The case dtype is where LMhdX sets precision, float32 included; see lmhdx.enable_x64.
        from . import _pin_matmul_precision

        _pin_matmul_precision()

    @property
    def output_dir(self) -> Path | None:
        if self.output.directory is None:
            return None
        return Path(self.output.directory)


class NumericalFailure(RuntimeError):
    """Raised when a solver produces nonfinite numerical state."""


def require_finite(stage: str, **values) -> None:
    """Raise with field names when numerical output is nonfinite."""

    failed = [name for name, value in values.items() if not bool(jnp.all(jnp.isfinite(jnp.asarray(value))))]
    if failed:
        raise NumericalFailure(f"{stage} produced nonfinite {', '.join(failed)}")


@dataclass(frozen=True)
class MHDState:
    u: jnp.ndarray
    phi: jnp.ndarray
    jy: jnp.ndarray
    jz: jnp.ndarray
    lorentz_x: jnp.ndarray
    time: float
    residual: float


@dataclass(frozen=True)
class Diagnostics:
    residual_history: jnp.ndarray
    courant_like: jnp.ndarray
    ohmic_power: jnp.ndarray
    time_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    u_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    mean_velocity_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    applied_forcing_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    current_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    face_current_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    emf_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    lorentz_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    face_lorentz_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    potential_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    potential_iterations_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    linear_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    linear_iterations_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    volumetric_flow_rate_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    mean_current_magnitude_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    lorentz_power_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    div_current_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    charge_balance_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    gauge_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    interface_current_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))


@dataclass(frozen=True)
class Solution:
    mesh: StructuredMesh
    state: MHDState
    diagnostics: Diagnostics
    case_name: str
    converged: bool | None = None
    status: str = "not_recorded"
    steps: int = 0

    @property
    def residual(self) -> float:
        """Terminal normalized solver residual."""

        return float(self.state.residual)

    @property
    def fields(self) -> MHDState:
        """Final fully developed MHD state."""

        return self.state


@dataclass(frozen=True)
class LoggingSpec:
    enabled: bool = True
    banner: bool = True
    print_footer: bool = True
    flush: bool = True
    step_stride: int = 1

    def is_enabled(self) -> bool:
        return bool(self.enabled)


@dataclass(frozen=True)
class RestartSpec:
    enabled: bool = False
    path: Path | None = None
    reset_histories: bool = True
    write_restart: bool = False
    restart_filename: str | None = None


@dataclass(frozen=True)
class RunConfig:
    case: CaseSpec
    logging: LoggingSpec = field(default_factory=LoggingSpec)
    restart: RestartSpec = field(default_factory=RestartSpec)
    input_path: Path | None = None


def _load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required TOML key '{key}'")
    return mapping[key]


def _optional_tuple(
    mapping: dict[str, Any], key: str, *, length: int | None = None, cast=float
) -> tuple[Any, ...] | None:
    if key not in mapping:
        return None
    values = tuple(cast(value) for value in mapping[key])
    if length is not None and len(values) != length:
        raise ValueError(f"TOML key '{key}' must have length {length}")
    return values


def _parse_regions(entries: list[dict[str, Any]]) -> tuple[RegionSpec, ...]:
    return tuple(
        RegionSpec(
            name=str(_require(entry, "name")),
            kind=str(_require(entry, "kind")),
            conductivity=float(_require(entry, "conductivity")),
            density=None if entry.get("density") is None else float(entry["density"]),
            viscosity=None if entry.get("viscosity") is None else float(entry["viscosity"]),
            wall_thickness=None if entry.get("wall_thickness") is None else float(entry["wall_thickness"]),
        )
        for entry in entries
    )


def _parse_boundary_value(value: Any) -> float | tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        values = tuple(float(component) for component in value)
        if len(values) != 3:
            raise ValueError("Boundary-condition vector values must have length 3")
        return values
    raise ValueError(f"Unsupported boundary-condition value {value!r}")


def _parse_boundaries(entries: list[dict[str, Any]]) -> tuple[BoundaryCondition, ...]:
    return tuple(
        BoundaryCondition(
            name=str(_require(entry, "name")),
            kind=str(_require(entry, "kind")),
            value=_parse_boundary_value(entry.get("value")),
            region=None if entry.get("region") is None else str(entry["region"]),
            axis=None if entry.get("axis") is None else str(entry["axis"]),
            side=None if entry.get("side") is None else str(entry["side"]),
        )
        for entry in entries
    )


def load_run_config(path: str | Path) -> RunConfig:
    input_path = Path(path).resolve()
    root = _load_toml(input_path)

    case_table = root.get("case", {})
    geometry_table = root.get("geometry", {})
    field_table = root.get("magnetic_field", {})
    solver_table = root.get("solver", {})
    time_table = root.get("time_stepper", {})
    output_table = root.get("output", {})
    logging_table = root.get("logging", {})
    restart_table = root.get("restart", {})
    regions_table = root.get("regions", [])
    boundaries_table = root.get("boundary_conditions", [])

    if field_table.get("kind") == "analytic":
        raise ValueError(
            "TOML input does not support analytic magnetic-field callables; use the Python API for that case"
        )

    geometry = GeometrySpec(
        kind=str(_require(geometry_table, "kind")),
        width=float(_require(geometry_table, "width")),
        height=float(_require(geometry_table, "height")),
        length=float(geometry_table.get("length", 1.0)),
        nx=int(geometry_table.get("nx", 1)),
        ny=int(_require(geometry_table, "ny")),
        nz=int(_require(geometry_table, "nz")),
        wall_thickness=_optional_tuple(geometry_table, "wall_thickness", length=4, cast=float)
        or (0.0, 0.0, 0.0, 0.0),
        wall_cells=_optional_tuple(geometry_table, "wall_cells", length=4, cast=int) or (0, 0, 0, 0),
        target_ha=None if geometry_table.get("target_ha") is None else float(geometry_table["target_ha"]),
        target_side_layer=None
        if geometry_table.get("target_side_layer") is None
        else float(geometry_table["target_side_layer"]),
        wall_model=str(geometry_table.get("wall_model", "auto")),
    )

    magnetic_field = MagneticFieldSpec(
        kind=str(_require(field_table, "kind")),
        value=_optional_tuple(field_table, "value", length=3, cast=float),
        fn=None,
        table_path=None
        if field_table.get("table_path") is None
        else str((input_path.parent / str(field_table["table_path"])).resolve()),
        ramp_start=float(field_table.get("ramp_start", 0.0)),
        ramp_duration=float(field_table.get("ramp_duration", 0.0)),
    )

    time_stepper = TimeStepperConfig(
        dt=float(_require(time_table, "dt")),
        t_final=float(_require(time_table, "t_final")),
        max_steps=int(_require(time_table, "max_steps")),
        potential_iterations=int(time_table.get("potential_iterations", 400)),
        potential_tolerance=None
        if time_table.get("potential_tolerance") is None
        else float(time_table["potential_tolerance"]),
        potential_relaxation=float(time_table.get("potential_relaxation", 1.0)),
        potential_solver=str(time_table.get("potential_solver", "auto")),
        steady_tolerance=float(time_table.get("steady_tolerance", 1e-8)),
        steady_potential_tolerance=None
        if time_table.get("steady_potential_tolerance") is None
        else float(time_table["steady_potential_tolerance"]),
        relaxation=float(time_table.get("relaxation", 0.35)),
        velocity_update_limit=float(time_table.get("velocity_update_limit", 1e-3)),
    )

    solver_mode = str(solver_table.get("mode", "steady"))
    if solver_mode not in {"steady", "transient"}:
        raise ValueError(f"Unsupported solve mode {solver_mode!r}")
    solver_kind = str(solver_table.get("kind", "fully_developed_inductionless"))
    if solver_kind != "fully_developed_inductionless":
        raise ValueError(f"Unsupported solver kind {solver_kind!r}")
    solver = SolverConfig(
        kind=solver_kind,
        mode=solver_mode,
        preconditioner=str(solver_table.get("preconditioner", "jacobi")),
        time_scheme=str(solver_table.get("time_scheme", "implicit_euler")),
        coupling_iterations=int(solver_table.get("coupling_iterations", 12)),
        coupling_tolerance=float(solver_table.get("coupling_tolerance", 1e-8)),
        coupling_acceleration=str(solver_table.get("coupling_acceleration", "none")),
        coupling_min_relaxation=float(solver_table.get("coupling_min_relaxation", 0.05)),
        coupling_max_relaxation=float(solver_table.get("coupling_max_relaxation", 100.0)),
        coupling_history_depth=int(solver_table.get("coupling_history_depth", 6)),
        coupling_regularization=float(solver_table.get("coupling_regularization", 1.0e-8)),
        coupling_damping=float(solver_table.get("coupling_damping", 1.0)),
    )
    output_dir = output_table.get("directory")
    if output_dir is not None:
        output_dir = str((input_path.parent / str(output_dir)).resolve())

    output = OutputSpec(
        directory=output_dir,
        write_paraview=bool(output_table.get("write_paraview", True)),
        write_csv_profiles=bool(output_table.get("write_csv_profiles", True)),
        write_npz=bool(output_table.get("write_npz", True)),
        write_json_summary=bool(output_table.get("write_json_summary", True)),
        write_plots=bool(output_table.get("write_plots", False)),
        copy_input_file=bool(output_table.get("copy_input_file", True)),
        write_stride=int(output_table.get("write_stride", 1)),
        history_stride=int(output_table.get("history_stride", 0)),
    )

    case = CaseSpec(
        name=str(_require(case_table, "name")),
        geometry=geometry,
        regions=_parse_regions(regions_table),
        magnetic_field=magnetic_field,
        boundary_conditions=_parse_boundaries(boundaries_table),
        time_stepper=time_stepper,
        solver=solver,
        output=output,
        forcing=float(case_table.get("forcing", 1.0)),
        initial_velocity=float(case_table.get("initial_velocity", 0.0)),
        reference_pressure_gradient=float(case_table.get("reference_pressure_gradient", -1.0)),
        reference_phi_cell=_optional_tuple(case_table, "reference_phi_cell", length=2, cast=int) or (0, 0),
        notes=str(case_table.get("notes", "")),
        dtype=str(case_table.get("dtype", "float64")),
    )

    logging = LoggingSpec(
        enabled=bool(logging_table.get("enabled", True)),
        banner=bool(logging_table.get("banner", True)),
        print_footer=bool(logging_table.get("print_footer", True)),
        flush=bool(logging_table.get("flush", True)),
        step_stride=int(logging_table.get("step_stride", 1)),
    )
    restart_enabled = bool(restart_table.get("enabled", False))
    restart_path = restart_table.get("path")
    restart = RestartSpec(
        enabled=restart_enabled,
        path=None if restart_path is None else (input_path.parent / str(restart_path)).resolve(),
        reset_histories=bool(restart_table.get("reset_histories", True)),
        write_restart=bool(restart_table.get("write_restart", restart_enabled)),
        restart_filename=None
        if restart_table.get("restart_filename") is None
        else str(restart_table["restart_filename"]),
    )
    return RunConfig(
        case=case,
        logging=logging,
        restart=restart,
        input_path=input_path,
    )


@dataclass(frozen=True)
class RestartLogInfo:
    enabled: bool = False
    path: str | None = None
    start_time: float = 0.0
    reset_histories: bool = True


@dataclass(frozen=True)
class SolverStepRecord:
    step_index: int
    time: float
    u_max: float
    mean_velocity: float
    current_max: float
    lorentz_max: float
    residual: float
    potential_residual: float
    potential_iterations: float
    linear_residual: float
    linear_iterations: float
    applied_forcing: float
    courant_like: float
    ohmic_power: float
    volumetric_flow_rate: float
    div_current_max: float
    gauge_residual: float
    interface_current_residual: float
    charge_balance_residual: float = 0.0
    potential_initial_residual: float = 0.0
    linear_initial_residual: float = 0.0


class StreamingSolverLogger:
    def __init__(self, config: LoggingSpec | None = None, *, stream: TextIO | None = None) -> None:
        self.config = config or LoggingSpec()
        self.streams: list[TextIO] = [stream or sys.stdout]
        self._start_time = time.perf_counter()
        self._max_steps: int | None = None
        self._target_final_time: float | None = None

    def add_stream(self, stream: TextIO) -> None:
        self.streams.append(stream)

    def _write(self, line: str = "") -> None:
        for stream in self.streams:
            print(line, file=stream, flush=self.config.flush)

    def emit_header(
        self,
        *,
        case: CaseSpec,
        mesh,
        mode: str,
        potential_solver: str,
        target_mean_velocity: float | None,
        reference_mean_velocity: float | None,
        restart: RestartLogInfo | None = None,
    ) -> None:
        if not self.config.is_enabled():
            return
        self._max_steps = int(case.time_stepper.max_steps)
        self._target_final_time = float(case.time_stepper.t_final)
        if self.config.banner:
            self._write("LMhdX solver")
        solver = getattr(case, "solver", None)
        solver_kind = getattr(solver, "kind", "fully_developed_inductionless")
        self._write(
            f"case={case.name} mode={mode} solver={solver_kind} geometry={case.geometry.kind} "
            f"cells=({mesh.nx},{mesh.ny},{mesh.nz})"
        )
        self._write(
            f"domain=({case.geometry.length:.6e},{case.geometry.width:.6e},{case.geometry.height:.6e}) "
            f"dt={case.time_stepper.dt:.6e} end={case.time_stepper.t_final:.6e} max_steps={self._max_steps}"
        )
        if solver is not None:
            self._write(
                f"linear=solvax_pcg preconditioner={solver.preconditioner} "
                f"coupling_steps={solver.coupling_iterations} coupling_tolerance={solver.coupling_tolerance:.6e}"
            )
        self._write(
            f"potential={potential_solver} max_steps={case.time_stepper.potential_iterations} "
            f"tolerance={case.time_stepper.potential_tolerance}"
        )
        self._write(
            f"field={case.magnetic_field.kind} value={case.magnetic_field.value} forcing={case.forcing:.6e} "
            f"target_mean_velocity={target_mean_velocity} reference_mean_velocity={reference_mean_velocity}"
        )
        if restart is not None and restart.enabled:
            self._write(
                f"restart={restart.path} start={restart.start_time:.6e} reset_histories={restart.reset_histories}"
            )

    def _progress_fraction(self, record: SolverStepRecord) -> float | None:
        candidates: list[float] = []
        if self._max_steps is not None and self._max_steps > 0:
            candidates.append(record.step_index / float(self._max_steps))
        if self._target_final_time is not None and self._target_final_time > 0.0:
            candidates.append(record.time / float(self._target_final_time))
        if not candidates:
            return None
        return min(max(max(candidates), 0.0), 1.0)

    @staticmethod
    def _format_seconds(seconds: float | None) -> str:
        if seconds is None or not math.isfinite(float(seconds)):
            return "unknown"
        total = max(float(seconds), 0.0)
        minutes, secs = divmod(total, 60.0)
        hours, minutes = divmod(minutes, 60.0)
        if hours >= 1.0:
            return f"{int(hours):02d}:{int(minutes):02d}:{secs:04.1f}"
        return f"{int(minutes):02d}:{secs:04.1f}"

    def emit_step(self, record: SolverStepRecord) -> None:
        if not self.config.is_enabled():
            return
        if record.step_index > 1 and (record.step_index - 1) % max(self.config.step_stride, 1) != 0:
            return
        elapsed = time.perf_counter() - self._start_time
        progress = self._progress_fraction(record)
        average_step = elapsed / max(record.step_index, 1)
        estimated_total = elapsed / progress if progress is not None and progress > 0.0 else None
        remaining = estimated_total - elapsed if estimated_total is not None else None
        self._write(
            f"step={record.step_index} time={record.time:.6e} residual={record.residual:.6e} "
            f"max_u={record.u_max:.6e} mean_u={record.mean_velocity:.6e} Q={record.volumetric_flow_rate:.6e}"
        )
        self._write(
            f"potential_residual={record.potential_residual:.6e} potential_steps={int(record.potential_iterations)} "
            f"linear_residual={record.linear_residual:.6e} linear_steps={int(record.linear_iterations)}"
        )
        self._write(
            f"max_j={record.current_max:.6e} max_lorentz={record.lorentz_max:.6e} "
            f"div_j={record.div_current_max:.6e} charge={record.charge_balance_residual:.6e} "
            f"gauge={record.gauge_residual:.6e} interface_current={record.interface_current_residual:.6e}"
        )
        self._write(
            f"forcing={record.applied_forcing:.6e} courant={record.courant_like:.6e} "
            f"ohmic_power={record.ohmic_power:.6e}"
        )
        if progress is None:
            self._write(f"elapsed={elapsed:.3f}s average_step={average_step:.3f}s")
        else:
            self._write(
                f"progress={100.0 * progress:.1f}% elapsed={elapsed:.3f}s average_step={average_step:.3f}s "
                f"remaining={self._format_seconds(remaining)} total={self._format_seconds(estimated_total)}"
            )
        self._write("")

    def emit_footer(self, solution: Solution) -> None:
        if not self.config.is_enabled() or not self.config.print_footer:
            return
        elapsed = time.perf_counter() - self._start_time
        self._write(
            f"completed case={solution.case_name} time={solution.state.time:.6e} "
            f"residual={solution.state.residual:.6e} elapsed={elapsed:.3f}s"
        )


def default_log_path(out_dir: str | Path, case_name: str) -> Path:
    out_dir = Path(out_dir)
    return out_dir / f"{case_name}.log"


# The named duct cases and the solve entry point.


def _ha_to_b(ha: float, length_scale: float, conductivity: float, density: float, viscosity: float) -> float:
    """Return ``B`` for target ``Ha`` with ``viscosity`` interpreted as ``nu``."""

    return magnetic_field_from_hartmann(
        hartmann=ha,
        length_scale=length_scale,
        conductivity=conductivity,
        density=density,
        kinematic_viscosity=viscosity,
    )


def _wall_conductivity_from_conductance_ratio(
    *,
    wall_conductance_ratio: float,
    fluid_conductivity: float,
    wall_thickness: float,
    hartmann_half_spacing: float,
) -> float:
    if wall_thickness <= 0.0:
        raise ValueError(
            "wall_thickness must be positive when deriving wall conductivity from conductance ratio"
        )
    if hartmann_half_spacing <= 0.0:
        raise ValueError(
            "hartmann_half_spacing must be positive when deriving wall conductivity from conductance ratio"
        )
    return wall_conductance_ratio * fluid_conductivity * hartmann_half_spacing / wall_thickness


def _hunt_short_transient_controls(ha: float) -> TimeStepperConfig:
    if ha <= 20.0:
        return TimeStepperConfig(
            dt=0.002,
            t_final=1.0,
            max_steps=500,
            potential_iterations=400,
            relaxation=0.08,
            velocity_update_limit=2e-3,
        )
    if ha <= 100.0:
        return TimeStepperConfig(
            dt=0.002,
            t_final=1.0,
            max_steps=500,
            potential_iterations=400,
            relaxation=0.1,
            velocity_update_limit=1e-3,
        )
    return TimeStepperConfig(
        dt=0.002,
        t_final=1.0,
        max_steps=500,
        potential_iterations=400,
        relaxation=0.1,
        velocity_update_limit=1e-3,
    )


def _fully_developed_solver(mode: str = "steady") -> SolverConfig:
    return SolverConfig(
        kind="fully_developed_inductionless",
        mode=mode,
        preconditioner="jacobi",
        time_scheme="implicit_euler",
        coupling_iterations=16,
        coupling_tolerance=1e-8,
    )


def make_hartmann_case(
    ha: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 96,
    nz: int = 96,
    conductivity: float = 1.0,
    density: float = 1.0,
    viscosity: float = 1.0,
    output_dir: str | None = None,
    dtype: str = "float64",
) -> CaseSpec:
    """Build an insulating rectangular Hartmann-duct reference case."""

    bmag = _ha_to_b(ha, 0.5 * height, conductivity, density, viscosity)
    anchor = (ny // 2, nz // 2)
    return CaseSpec(
        name=f"hartmann_ha{int(ha)}",
        dtype=dtype,
        geometry=GeometrySpec(kind="rect_duct", width=width, height=height, ny=ny, nz=nz, target_ha=ha),
        regions=(RegionSpec("fluid", "fluid", conductivity, density, viscosity),),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 1.0 * bmag, 0.0)),
        boundary_conditions=(
            BoundaryCondition("walls", "no_slip"),
            BoundaryCondition("electric", "insulating"),
        ),
        time_stepper=TimeStepperConfig(
            dt=0.001, t_final=1.0, max_steps=400, potential_iterations=200, relaxation=0.1
        ),
        solver=_fully_developed_solver(),
        output=OutputSpec(directory=output_dir),
        forcing=1.0,
        reference_pressure_gradient=-1.0,
        reference_phi_cell=anchor,
        notes="Planar Hartmann-like reference configuration for solver smoke tests.",
    )


def make_shercliff_case(
    ha: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 96,
    nz: int = 96,
    conductivity: float = 1.0,
    density: float = 1.0,
    viscosity: float = 1.0,
    output_dir: str | None = None,
    dtype: str = "float64",
) -> CaseSpec:
    """Build an all-insulating rectangular Shercliff-duct case."""

    bmag = _ha_to_b(ha, 0.5 * width, conductivity, density, viscosity)
    anchor = (ny // 2, nz // 2)
    return CaseSpec(
        name=f"shercliff_ha{int(ha)}",
        dtype=dtype,
        geometry=GeometrySpec(kind="rect_duct", width=width, height=height, ny=ny, nz=nz, target_ha=ha),
        regions=(RegionSpec("fluid", "fluid", conductivity, density, viscosity),),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 1.0 * bmag, 0.0)),
        boundary_conditions=(
            BoundaryCondition("walls", "no_slip"),
            BoundaryCondition("electric", "insulating"),
        ),
        time_stepper=TimeStepperConfig(
            dt=0.001, t_final=1.5, max_steps=400, potential_iterations=225, relaxation=0.1
        ),
        solver=_fully_developed_solver(),
        output=OutputSpec(directory=output_dir),
        forcing=1.0,
        reference_pressure_gradient=-1.0,
        reference_phi_cell=anchor,
        notes="All-insulating Shercliff-style duct. Analytical validation hooks are staged through the benchmark and validation utilities.",
    )


def make_hunt_case(
    ha: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 72,
    nz: int = 72,
    wall_cells: int = 8,
    wall_thickness: float = 0.1,
    insulator_cells: int | None = None,
    insulator_thickness: float | None = None,
    fluid_conductivity: float = 1.0,
    wall_conductance_ratio: float = 0.05,
    wall_conductivity: float | None = None,
    insulator_conductivity: float | None = None,
    insulator_conductivity_ratio: float = 1e-12,
    density: float = 1.0,
    viscosity: float = 1.0,
    output_dir: str | None = None,
    dtype: str = "float64",
) -> CaseSpec:
    """Build a Hunt duct with conducting Hartmann and insulating side walls."""

    bmag = _ha_to_b(ha, 0.5 * width, fluid_conductivity, density, viscosity)
    if wall_conductivity is None:
        wall_conductivity = _wall_conductivity_from_conductance_ratio(
            wall_conductance_ratio=wall_conductance_ratio,
            fluid_conductivity=fluid_conductivity,
            wall_thickness=wall_thickness,
            hartmann_half_spacing=0.5 * height,
        )
    if insulator_cells is None:
        insulator_cells = wall_cells
    if insulator_thickness is None:
        insulator_thickness = wall_thickness
    if insulator_conductivity is None:
        insulator_conductivity = fluid_conductivity * insulator_conductivity_ratio
    anchor = ((ny + 2 * insulator_cells) // 2, (nz + 2 * wall_cells) // 2)
    controls = _hunt_short_transient_controls(ha)
    return CaseSpec(
        name=f"hunt_ha{int(ha)}",
        dtype=dtype,
        geometry=GeometrySpec(
            kind="layered_duct",
            width=width,
            height=height,
            ny=ny,
            nz=nz,
            wall_thickness=(insulator_thickness, insulator_thickness, wall_thickness, wall_thickness),
            wall_cells=(insulator_cells, insulator_cells, wall_cells, wall_cells),
            target_ha=ha,
        ),
        regions=(
            RegionSpec("fluid", "fluid", fluid_conductivity, density, viscosity),
            RegionSpec("conducting_wall", "solid", wall_conductivity, density, viscosity, wall_thickness),
            RegionSpec(
                "insulating_wall", "solid", insulator_conductivity, density, viscosity, insulator_thickness
            ),
        ),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 1.0 * bmag, 0.0)),
        boundary_conditions=(
            BoundaryCondition("walls", "no_slip"),
            BoundaryCondition(
                "conducting_hartmann_walls", "conducting_wall", region="conducting_wall", side="left_right"
            ),
            BoundaryCondition(
                "insulating_side_walls", "insulating", region="insulating_wall", side="top_bottom"
            ),
        ),
        time_stepper=controls,
        solver=_fully_developed_solver(),
        output=OutputSpec(directory=output_dir),
        forcing=1.0,
        reference_pressure_gradient=-1.0,
        reference_phi_cell=anchor,
        notes=(
            "Hunt-style duct with explicit conducting Hartmann-wall layers and insulating side-wall layers. "
            f"Default wall conductance ratio c={wall_conductance_ratio:g}."
        ),
    )


def solve(
    model: "ChannelProblem | CaseSpec | Q2DProblem",
) -> "SteadySolution | Solution | Q2DResult":
    """Solve a duct, a fully developed case, or a Q2D problem.

    A :class:`lmhdx.core3d.ChannelProblem` goes to the staggered core's steady
    solve, :func:`lmhdx.steady.solve_steady_state`, and so does a steady
    ``CaseSpec``, through :func:`lmhdx.fully_developed.solve_fully_developed`,
    which reports it on the case's cross-section; each is compiled once per
    problem. A transient ``CaseSpec`` runs implicit Euler steps on the core,
    :func:`lmhdx.fully_developed.solve_fully_developed_transient`. A duct with an
    inlet and an outlet is solved by :func:`lmhdx.axial.solve_open_duct`.
    """

    from .core3d import ChannelProblem

    if isinstance(model, ChannelProblem):
        from .steady import solve_compiled

        return solve_compiled(model)
    if isinstance(model, CaseSpec):
        from .fully_developed import solve_fully_developed, solve_fully_developed_transient

        if model.solver.mode == "transient":
            return solve_fully_developed_transient(model)
        return solve_fully_developed(model)
    from .q2d import Q2DProblem, solve_q2d

    if isinstance(model, Q2DProblem):
        return solve_q2d(model)
    raise TypeError(f"solve expects ChannelProblem, CaseSpec, or Q2DProblem, got {type(model).__name__}")
