"""Materials, magnetic fields, nondimensional groups, and wall physics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

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
