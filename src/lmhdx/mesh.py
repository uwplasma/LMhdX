from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import jax.numpy as jnp
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from .physics import WallLayer


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


def generate_layered_duct_mesh_from_fluid_faces(
    *,
    fluid_y_faces: jnp.ndarray,
    fluid_z_faces: jnp.ndarray,
    width: float,
    height: float,
    length: float = 1.0,
    nx: int = 1,
    wall_thickness: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    wall_cells: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> StructuredMesh:
    """Build a layered duct around explicit fluid-region faces."""

    fluid_y = _validated_faces(fluid_y_faces, name="fluid_y_faces")
    fluid_z = _validated_faces(fluid_z_faces, name="fluid_z_faces")
    left_t, right_t, bottom_t, top_t = wall_thickness
    left_c, right_c, bottom_c, top_c = wall_cells
    if left_c:
        left_faces = jnp.linspace(float(fluid_y[0]) - left_t, float(fluid_y[0]), left_c + 1)
        fluid_y = jnp.concatenate([left_faces[:-1], fluid_y])
    if right_c:
        right_faces = jnp.linspace(float(fluid_y[-1]), float(fluid_y[-1]) + right_t, right_c + 1)
        fluid_y = jnp.concatenate([fluid_y, right_faces[1:]])
    if bottom_c:
        bottom_faces = jnp.linspace(float(fluid_z[0]) - bottom_t, float(fluid_z[0]), bottom_c + 1)
        fluid_z = jnp.concatenate([bottom_faces[:-1], fluid_z])
    if top_c:
        top_faces = jnp.linspace(float(fluid_z[-1]), float(fluid_z[-1]) + top_t, top_c + 1)
        fluid_z = jnp.concatenate([fluid_z, top_faces[1:]])

    x_faces = jnp.linspace(0.0, length, nx + 1)
    yc, zc = jnp.meshgrid(
        0.5 * (fluid_y[:-1] + fluid_y[1:]),
        0.5 * (fluid_z[:-1] + fluid_z[1:]),
        indexing="ij",
    )
    half_width = 0.5 * width
    half_height = 0.5 * height
    fluid_mask = (jnp.abs(yc) <= half_width) & (jnp.abs(zc) <= half_height)
    return StructuredMesh(
        x_faces=x_faces,
        y_faces=fluid_y,
        z_faces=fluid_z,
        geometry="layered_duct",
        fluid_mask=fluid_mask,
    )


def _clustered_segment(start: float, stop: float, count: int, beta: float = 2.5) -> jnp.ndarray:
    if count <= 1:
        return jnp.asarray([start, stop], dtype=float)
    s = jnp.linspace(-1.0, 1.0, count + 1)
    mapped = jnp.tanh(beta * s) / jnp.tanh(beta)
    scaled = 0.5 * (mapped + 1.0)
    return start + (stop - start) * scaled


def _geometric_widths(total_width: float, cells: int, ratio: float) -> jnp.ndarray:
    if cells <= 0:
        return jnp.asarray([], dtype=float)
    if cells == 1 or abs(ratio - 1.0) < 1e-12:
        return jnp.full((cells,), total_width / max(cells, 1), dtype=float)
    scale = total_width * (ratio - 1.0) / (ratio**cells - 1.0)
    indices = jnp.arange(cells, dtype=float)
    return scale * ratio**indices


def _symmetric_boundary_layer_segment(
    start: float,
    stop: float,
    count: int,
    *,
    layer_thickness: float,
    layer_cells: int,
    growth_ratio: float = 1.35,
) -> jnp.ndarray:
    length = stop - start
    if count <= 1 or layer_cells <= 0 or layer_thickness <= 0.0:
        return jnp.linspace(start, stop, count + 1)
    capped_layer_thickness = min(layer_thickness, 0.24 * length)
    capped_layer_cells = min(layer_cells, max(1, (count - 2) // 2))
    if capped_layer_cells <= 0:
        return jnp.linspace(start, stop, count + 1)
    core_cells = count - 2 * capped_layer_cells
    if core_cells <= 0:
        return _clustered_segment(start, stop, count, beta=1.5)
    left_widths = _geometric_widths(capped_layer_thickness, capped_layer_cells, growth_ratio)
    right_widths = left_widths[::-1]
    core_width = (length - 2.0 * capped_layer_thickness) / core_cells
    core_widths = jnp.full((core_cells,), core_width, dtype=float)
    widths = jnp.concatenate([left_widths, core_widths, right_widths])
    faces = start + jnp.concatenate([jnp.asarray([0.0], dtype=float), jnp.cumsum(widths)])
    return faces


def _smooth_boundary_layer_segment(
    start: float,
    stop: float,
    count: int,
    *,
    layer_thickness: float,
    layer_cells: int,
) -> jnp.ndarray:
    """Build a symmetric geometric mesh with an exact wall-layer cell count.

    Unlike a piecewise wall/expansion/core construction, every adjacent cell
    width changes by one geometric ratio. This avoids the large spacing jump
    immediately outside thin high-Ha layers while preserving the requested
    number of cells inside the physical layer thickness.
    """

    length = float(stop - start)
    if count <= 1 or layer_thickness <= 0.0 or layer_cells <= 0:
        return jnp.linspace(start, stop, count + 1)
    half_cells = count // 2
    cells_in_layer = min(int(layer_cells), half_cells)
    half_length = 0.5 * length
    if cells_in_layer <= 0 or layer_thickness >= half_length:
        return jnp.linspace(start, stop, count + 1)
    has_center = count % 2

    def geometric_sum(ratio: float, cells: int) -> float:
        if abs(ratio - 1.0) < 1.0e-12:
            return float(cells)
        return (ratio**cells - 1.0) / (ratio - 1.0)

    target_ratio = half_length / float(layer_thickness)

    def span_ratio(ratio: float) -> float:
        half_span = geometric_sum(ratio, half_cells)
        if has_center:
            half_span += 0.5 * ratio**half_cells
        return half_span / geometric_sum(ratio, cells_in_layer)

    if target_ratio <= span_ratio(1.0):
        return jnp.linspace(start, stop, count + 1)
    lower = 1.0
    upper = 1.1
    while span_ratio(upper) < target_ratio:
        upper *= 1.25
        if upper > 10.0:
            return jnp.linspace(start, stop, count + 1)
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if span_ratio(middle) < target_ratio:
            lower = middle
        else:
            upper = middle
    ratio = 0.5 * (lower + upper)
    first_width = float(layer_thickness) / geometric_sum(ratio, cells_in_layer)
    left = first_width * ratio ** jnp.arange(half_cells, dtype=float)
    pieces = [left]
    if has_center:
        pieces.append(jnp.asarray([first_width * ratio**half_cells], dtype=float))
    pieces.append(left[::-1])
    widths = jnp.concatenate(pieces)
    # Remove only roundoff in the nonlinear ratio solve; relative spacing and
    # the exact layer allocation are otherwise unchanged.
    widths = widths * (length / jnp.sum(widths))
    return start + jnp.concatenate([jnp.asarray([0.0]), jnp.cumsum(widths)])


def generate_rect_duct_mesh(
    width: float,
    height: float,
    length: float = 1.0,
    nx: int = 1,
    ny: int = 64,
    nz: int = 64,
    target_ha: float | None = None,
    magnetic_axis: str | None = None,
) -> StructuredMesh:
    """Generate a rectangular duct mesh with optional MHD layer clustering."""

    x_faces = jnp.linspace(0.0, length, nx + 1)
    if target_ha and target_ha > 0.0:
        side_y = 0.5 * width / jnp.sqrt(target_ha)
        side_z = 0.5 * height / jnp.sqrt(target_ha)
        hartmann_y = 0.5 * width / target_ha
        hartmann_z = 0.5 * height / target_ha
        if magnetic_axis == "y":
            y_layer_thickness = hartmann_y
            z_layer_thickness = side_z
        elif magnetic_axis == "z":
            y_layer_thickness = side_y
            z_layer_thickness = hartmann_z
        else:
            y_layer_thickness = hartmann_y
            z_layer_thickness = hartmann_z
        if target_ha >= 100.0:
            y_faces = _smooth_boundary_layer_segment(
                -0.5 * width,
                0.5 * width,
                ny,
                layer_thickness=float(y_layer_thickness),
                layer_cells=max(5, int(round(0.10 * ny))),
            )
            z_faces = _smooth_boundary_layer_segment(
                -0.5 * height,
                0.5 * height,
                nz,
                layer_thickness=float(z_layer_thickness),
                layer_cells=max(5, int(round(0.10 * nz))),
            )
        else:
            y_layer_cells = max(5, int(round((0.20 if magnetic_axis == "y" else 0.16) * ny)))
            z_layer_cells = max(5, int(round((0.20 if magnetic_axis == "z" else 0.16) * nz)))
            y_faces = _symmetric_boundary_layer_segment(
                -0.5 * width,
                0.5 * width,
                ny,
                layer_thickness=float(y_layer_thickness),
                layer_cells=y_layer_cells,
                growth_ratio=1.35,
            )
            z_faces = _symmetric_boundary_layer_segment(
                -0.5 * height,
                0.5 * height,
                nz,
                layer_thickness=float(z_layer_thickness),
                layer_cells=z_layer_cells,
                growth_ratio=1.35,
            )
    else:
        y_faces = jnp.linspace(-0.5 * width, 0.5 * width, ny + 1)
        z_faces = jnp.linspace(-0.5 * height, 0.5 * height, nz + 1)
    return StructuredMesh(x_faces=x_faces, y_faces=y_faces, z_faces=z_faces, geometry="rect_duct")


def generate_layered_duct_mesh(
    width: float,
    height: float,
    length: float = 1.0,
    nx: int = 1,
    ny: int = 64,
    nz: int = 64,
    wall_thickness: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    wall_cells: tuple[int, int, int, int] = (0, 0, 0, 0),
    target_ha: float | None = None,
    magnetic_axis: str | None = None,
) -> StructuredMesh:
    """Generate a rectangular fluid mesh plus explicit surrounding wall cells."""

    left_t, right_t, bottom_t, top_t = wall_thickness
    left_c, right_c, bottom_c, top_c = wall_cells

    if target_ha and target_ha > 0.0:
        fluid_mesh = generate_rect_duct_mesh(
            width=width,
            height=height,
            length=length,
            nx=max(nx, 1),
            ny=ny,
            nz=nz,
            target_ha=target_ha,
            magnetic_axis=magnetic_axis,
        )
        fluid_y = fluid_mesh.y_faces
        fluid_z = fluid_mesh.z_faces
    else:
        fluid_y = _clustered_segment(-0.5 * width, 0.5 * width, ny, beta=2.0)
        fluid_z = _clustered_segment(-0.5 * height, 0.5 * height, nz, beta=2.0)

    if left_c:
        left_faces = jnp.linspace(-0.5 * width - left_t, -0.5 * width, left_c + 1)
        fluid_y = jnp.concatenate([left_faces[:-1], fluid_y])
    if right_c:
        right_faces = jnp.linspace(0.5 * width, 0.5 * width + right_t, right_c + 1)
        fluid_y = jnp.concatenate([fluid_y, right_faces[1:]])
    if bottom_c:
        bottom_faces = jnp.linspace(-0.5 * height - bottom_t, -0.5 * height, bottom_c + 1)
        fluid_z = jnp.concatenate([bottom_faces[:-1], fluid_z])
    if top_c:
        top_faces = jnp.linspace(0.5 * height, 0.5 * height + top_t, top_c + 1)
        fluid_z = jnp.concatenate([fluid_z, top_faces[1:]])

    x_faces = jnp.linspace(0.0, length, nx + 1)
    y_faces = jnp.asarray(fluid_y)
    z_faces = jnp.asarray(fluid_z)

    yc, zc = jnp.meshgrid(
        0.5 * (y_faces[:-1] + y_faces[1:]),
        0.5 * (z_faces[:-1] + z_faces[1:]),
        indexing="ij",
    )
    fluid_mask = (jnp.abs(yc) <= 0.5 * width) & (jnp.abs(zc) <= 0.5 * height)
    return StructuredMesh(
        x_faces=x_faces,
        y_faces=y_faces,
        z_faces=z_faces,
        geometry="layered_duct",
        fluid_mask=fluid_mask,
    )


_WALL_SIDES = ("left", "right", "bottom", "top")


def generate_multilayer_duct_mesh(
    *,
    width: float,
    height: float,
    length: float = 1.0,
    nx: int = 1,
    ny: int = 64,
    nz: int = 64,
    wall_layers: dict[str, Sequence[WallLayer]] | None = None,
    fluid_conductivity: float = 1.0,
    target_ha: float | None = None,
    magnetic_axis: str | None = None,
) -> StructuredMesh:
    """Build a rectangular duct with explicit nested wall-layer cells.

    ``wall_layers`` maps ``left``, ``right``, ``bottom``, and/or ``top`` to
    ``WallLayer`` sequences ordered from the fluid outward. Faces are inserted
    at every layer interface, and ``mesh.sigma``/``mesh.region_ids`` are filled
    so downstream QA and material-field construction can use the explicit
    multilayer geometry directly.
    """

    if width <= 0.0 or height <= 0.0 or length <= 0.0:
        raise ValueError("width, height, and length must be positive")
    if ny <= 0 or nz <= 0 or nx <= 0:
        raise ValueError("nx, ny, and nz must be positive")
    if fluid_conductivity <= 0.0:
        raise ValueError("fluid_conductivity must be positive")
    layers_by_side = _normalized_wall_layers(wall_layers or {})
    if target_ha and target_ha > 0.0:
        fluid_mesh = generate_rect_duct_mesh(
            width=width,
            height=height,
            length=length,
            nx=nx,
            ny=ny,
            nz=nz,
            target_ha=target_ha,
            magnetic_axis=magnetic_axis,
        )
        y_faces = fluid_mesh.y_faces
        z_faces = fluid_mesh.z_faces
    else:
        y_faces = _clustered_segment(-0.5 * width, 0.5 * width, ny, beta=2.0)
        z_faces = _clustered_segment(-0.5 * height, 0.5 * height, nz, beta=2.0)

    left_faces = _wall_faces(-0.5 * width, layers_by_side["left"], positive=False)
    right_faces = _wall_faces(0.5 * width, layers_by_side["right"], positive=True)
    bottom_faces = _wall_faces(-0.5 * height, layers_by_side["bottom"], positive=False)
    top_faces = _wall_faces(0.5 * height, layers_by_side["top"], positive=True)
    if left_faces.size:
        y_faces = jnp.concatenate([left_faces[:-1], y_faces])
    if right_faces.size:
        y_faces = jnp.concatenate([y_faces, right_faces[1:]])
    if bottom_faces.size:
        z_faces = jnp.concatenate([bottom_faces[:-1], z_faces])
    if top_faces.size:
        z_faces = jnp.concatenate([z_faces, top_faces[1:]])

    x_faces = jnp.linspace(0.0, length, nx + 1)
    yc, zc = np.meshgrid(
        np.asarray(0.5 * (y_faces[:-1] + y_faces[1:]), dtype=float),
        np.asarray(0.5 * (z_faces[:-1] + z_faces[1:]), dtype=float),
        indexing="ij",
    )
    fluid_mask = (np.abs(yc) <= 0.5 * width) & (np.abs(zc) <= 0.5 * height)
    region_names, region_sigmas, region_ids = _multilayer_region_assignment(
        yc,
        zc,
        width=width,
        height=height,
        fluid_mask=fluid_mask,
        fluid_conductivity=fluid_conductivity,
        wall_layers=layers_by_side,
    )
    sigma = np.asarray(region_sigmas, dtype=float)[region_ids]
    return StructuredMesh(
        x_faces=x_faces,
        y_faces=jnp.asarray(y_faces),
        z_faces=jnp.asarray(z_faces),
        geometry="layered_duct",
        fluid_mask=jnp.asarray(fluid_mask),
        sigma=jnp.asarray(sigma, dtype=float),
        region_ids=jnp.asarray(region_ids, dtype=int),
        region_names=tuple(region_names),
    )


def _normalized_wall_layers(
    wall_layers: dict[str, Sequence[WallLayer]],
) -> dict[str, tuple[WallLayer, ...]]:
    normalized: dict[str, tuple[WallLayer, ...]] = {}
    for side in _WALL_SIDES:
        layers = tuple(wall_layers.get(side, ()))
        for layer in layers:
            if layer.thickness <= 0.0:
                raise ValueError(f"{side} wall layer {layer.name!r} has non-positive thickness")
            if layer.cells <= 0:
                raise ValueError(f"{side} wall layer {layer.name!r} must have at least one cell")
            if layer.conductivity < 0.0:
                raise ValueError(f"{side} wall layer {layer.name!r} has negative conductivity")
        normalized[side] = layers
    unknown = set(wall_layers) - set(_WALL_SIDES)
    if unknown:
        raise ValueError(f"unsupported wall side(s): {sorted(unknown)}")
    return normalized


def _wall_faces(
    inner_boundary: float,
    layers: Sequence[WallLayer],
    *,
    positive: bool,
) -> jnp.ndarray:
    if not layers:
        return jnp.asarray([], dtype=float)
    cursor = float(inner_boundary)
    ordered_layers = layers
    if not positive:
        cursor -= sum(float(layer.thickness) for layer in layers)
        ordered_layers = tuple(reversed(layers))
    segments = []
    for layer in ordered_layers:
        stop = cursor + float(layer.thickness)
        segment = jnp.linspace(cursor, stop, int(layer.cells) + 1)
        segments.append(segment if not segments else segment[1:])
        cursor = stop
    return jnp.concatenate(segments)


def _multilayer_region_assignment(
    yc: np.ndarray,
    zc: np.ndarray,
    *,
    width: float,
    height: float,
    fluid_mask: np.ndarray,
    fluid_conductivity: float,
    wall_layers: dict[str, tuple[WallLayer, ...]],
) -> tuple[list[str], list[float], np.ndarray]:
    region_names = ["fluid"]
    region_sigmas = [float(fluid_conductivity)]
    region_for_side_layer: dict[tuple[str, int], int] = {}
    for side in _WALL_SIDES:
        for index, layer in enumerate(wall_layers[side]):
            region_for_side_layer[(side, index)] = len(region_names)
            region_names.append(f"{side}:{layer.name}")
            region_sigmas.append(float(layer.conductivity))

    region_ids = np.zeros(yc.shape, dtype=int)
    distances = {
        "left": np.where(yc < -0.5 * width, -0.5 * width - yc, np.inf),
        "right": np.where(yc > 0.5 * width, yc - 0.5 * width, np.inf),
        "bottom": np.where(zc < -0.5 * height, -0.5 * height - zc, np.inf),
        "top": np.where(zc > 0.5 * height, zc - 0.5 * height, np.inf),
    }
    distance_stack = np.stack([distances[side] for side in _WALL_SIDES], axis=0)
    side_index = np.argmin(distance_stack, axis=0)
    for side_number, side in enumerate(_WALL_SIDES):
        side_mask = (~fluid_mask) & (side_index == side_number)
        if not np.any(side_mask) or not wall_layers[side]:
            continue
        distance = distances[side]
        lower = 0.0
        for index, layer in enumerate(wall_layers[side]):
            upper = lower + float(layer.thickness)
            layer_mask = side_mask & (distance > lower - 1.0e-12) & (distance <= upper + 1.0e-12)
            region_ids[layer_mask] = region_for_side_layer[(side, index)]
            lower = upper
        region_ids[side_mask & (distance > lower)] = region_for_side_layer[(side, len(wall_layers[side]) - 1)]
    return region_names, region_sigmas, region_ids


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
