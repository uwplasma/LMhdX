"""Grid geometry, staggered containers and wall-resolving coordinate families."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lmhdx.grid import (
    CENTER,
    FACE,
    POLAR,
    Field,
    Grid,
    geometric_faces,
    tanh_faces,
    uniform_faces,
    wall_resolving_faces,
)

pytestmark = pytest.mark.unit


def _grid(nx=3, ny=4, nz=5) -> Grid:
    return Grid(
        uniform_faces(nx, 0.0, 3.0),
        geometric_faces(ny, -1.0, 1.0, 1.2),
        tanh_faces(nz, -0.5, 0.5, 1.4),
    )


def test_grid_reports_consistent_cells_widths_and_centers():
    grid = _grid()
    assert grid.shape == (3, 4, 5)
    for faces, width, center, extent in zip(grid.faces, grid.widths, grid.centers, grid.extent, strict=True):
        assert width.size == faces.size - 1
        assert np.all(width > 0.0)
        assert float(np.sum(width)) == pytest.approx(extent, abs=1e-14)
        assert np.allclose(center, 0.5 * (faces[:-1] + faces[1:]), atol=1e-15)


def test_cell_volumes_and_face_areas_sum_to_the_domain_measure():
    grid = _grid()
    volumes = grid.cell_volumes()
    assert volumes.shape == grid.shape
    assert float(np.sum(volumes)) == pytest.approx(float(np.prod(grid.extent)), rel=1e-14)
    for index, axis in enumerate("xyz"):
        areas = grid.face_areas(axis)
        assert areas.shape == grid.face_shape(axis)
        transverse = float(np.prod([e for i, e in enumerate(grid.extent) if i != index]))
        assert float(np.sum(areas[(slice(None),) * index + (0,)])) == pytest.approx(transverse, rel=1e-14)


def test_offset_and_face_shapes_follow_the_staggered_convention():
    grid = _grid()
    assert grid.offset_shape((CENTER, CENTER, CENTER)) == (3, 4, 5)
    assert grid.offset_shape((FACE, CENTER, CENTER)) == (4, 4, 5)
    assert grid.offset_shape((CENTER, FACE, CENTER)) == (3, 5, 5)
    assert grid.face_shape("z") == (3, 4, 6)
    assert grid.axis_index("y") == grid.axis_index(1) == 1


def test_grid_is_hashable_static_metadata():
    first, second = _grid(), _grid()
    assert first == second and hash(first) == hash(second)
    assert first != _grid(nx=4)
    assert len({first, second}) == 1


@pytest.mark.parametrize(
    ("faces", "message"),
    [
        (np.array([0.0]), "at least two faces"),
        (np.array([[0.0, 1.0]]), "one-dimensional"),
        (np.array([0.0, 0.0, 1.0]), "strictly increasing"),
        (np.array([0.0, np.inf]), "finite"),
    ],
)
def test_grid_rejects_degenerate_faces(faces, message):
    with pytest.raises(ValueError, match=message):
        Grid(faces, uniform_faces(2, 0.0, 1.0), uniform_faces(2, 0.0, 1.0))


def test_field_is_a_pytree_carrying_static_position():
    grid = _grid()
    field = Field(jnp.ones(grid.offset_shape((FACE, CENTER, CENTER))), (FACE, CENTER, CENTER), grid)
    leaves, treedef = jax.tree_util.tree_flatten(field)
    assert len(leaves) == 1 and leaves[0].shape == (4, 4, 5)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.offset == field.offset and restored.grid == grid

    doubled = jax.jit(lambda item: item.replace_data(2.0 * item.data))(field)
    assert doubled.offset == field.offset
    assert doubled.grid == grid
    assert np.allclose(np.asarray(doubled.data), 2.0)


def test_field_preserves_requested_precision():
    grid = _grid()
    single = Field(jnp.ones(grid.shape, dtype=jnp.float32), (CENTER,) * 3, grid)
    assert single.dtype == jnp.float32
    assert single.shape == grid.shape


def test_uniform_faces_are_equally_spaced_and_span_the_domain():
    faces = uniform_faces(6, -1.0, 2.0)
    widths = np.diff(faces)
    assert faces[0] == -1.0 and faces[-1] == 2.0
    assert np.allclose(widths, widths[0], atol=1e-15)


def test_geometric_faces_grow_by_the_requested_ratio():
    faces = geometric_faces(5, 0.0, 1.0, 1.3, both_ends=False)
    widths = np.diff(faces)
    assert np.allclose(widths[1:] / widths[:-1], 1.3, rtol=1e-12)
    assert faces[0] == 0.0 and faces[-1] == pytest.approx(1.0, abs=1e-15)


def test_geometric_faces_cluster_symmetrically_at_both_walls():
    faces = geometric_faces(8, -1.0, 1.0, 1.4)
    widths = np.diff(faces)
    assert np.allclose(widths, widths[::-1], atol=1e-14)
    assert widths[0] < widths[len(widths) // 2]
    assert np.allclose(faces, -faces[::-1], atol=1e-14)


def test_tanh_faces_cluster_at_both_walls_and_stay_monotone():
    faces = tanh_faces(10, -1.0, 1.0, 2.0)
    widths = np.diff(faces)
    assert np.all(widths > 0.0)
    assert np.allclose(widths, widths[::-1], atol=1e-14)
    assert widths[0] < np.mean(widths)
    assert faces[0] == pytest.approx(-1.0) and faces[-1] == pytest.approx(1.0)


def test_stronger_tanh_stretching_thins_the_wall_cell():
    mild = np.diff(tanh_faces(10, -1.0, 1.0, 1.0))[0]
    strong = np.diff(tanh_faces(10, -1.0, 1.0, 3.0))[0]
    assert strong < mild


@pytest.mark.parametrize("hartmann", [100.0, 1000.0, 10000.0])
def test_wall_resolving_faces_place_the_requested_cells_inside_the_layer(hartmann):
    half_width, cells = 1.0, 8
    layer = half_width / hartmann
    faces = wall_resolving_faces(
        160, -half_width, half_width, layer_thickness=layer, cells_in_layer=cells, max_ratio=1.15
    )
    widths = np.diff(faces)
    assert np.all(widths > 0.0)
    assert np.allclose(widths, widths[::-1], atol=1e-12)
    assert float(np.sum(widths[:cells])) <= layer * (1.0 + 1.0e-9)
    assert np.max(widths[1:] / widths[:-1]) <= 1.15 + 1.0e-9
    assert faces[0] == pytest.approx(-half_width) and faces[-1] == pytest.approx(half_width)


def test_wall_resolving_faces_refuse_an_unreachable_request():
    with pytest.raises(ValueError, match="increase count"):
        wall_resolving_faces(8, -1.0, 1.0, layer_thickness=1.0e-4, cells_in_layer=6, max_ratio=1.05)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"layer_thickness": 0.0}, "layer_thickness must be positive"),
        ({"layer_thickness": 0.1, "cells_in_layer": 0}, "cells_in_layer must be positive"),
        ({"layer_thickness": 0.1, "max_ratio": 0.9}, "max_ratio must be at least one"),
        ({"layer_thickness": 5.0}, "must not exceed the half-width"),
    ],
)
def test_wall_resolving_faces_validate_their_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        wall_resolving_faces(20, -1.0, 1.0, **kwargs)


def test_an_odd_cell_count_puts_one_geometric_cell_on_the_midplane():
    faces = wall_resolving_faces(9, -1.0, 1.0, layer_thickness=0.1, cells_in_layer=2, max_ratio=None)
    widths = np.diff(faces)
    assert faces.shape == (10,) and faces[0] == -1.0 and faces[-1] == 1.0
    assert np.allclose(widths, widths[::-1], rtol=1e-12)
    ratios = widths[1:5] / widths[:4]
    assert np.allclose(ratios, ratios[0], rtol=1e-9) and ratios[0] > 1.0
    assert float(np.sum(widths[:2])) <= 0.1 * (1.0 + 1e-9)
    even = wall_resolving_faces(8, -1.0, 1.0, layer_thickness=0.1, cells_in_layer=2, max_ratio=None)
    assert even.shape == (9,) and even[4] == 0.0


def test_geometric_faces_require_an_even_cell_count():
    with pytest.raises(ValueError, match="even"):
        geometric_faces(9, -1.0, 1.0, 1.2)


@pytest.mark.parametrize("factory", [uniform_faces, tanh_faces, geometric_faces])
def test_coordinate_families_reject_degenerate_extents(factory):
    extra = () if factory is uniform_faces else (1.2,)
    with pytest.raises(ValueError, match="count must be positive"):
        factory(0, 0.0, 1.0, *extra)
    with pytest.raises(ValueError, match="upper must exceed lower"):
        factory(4, 1.0, 1.0, *extra)


def test_grid_equality_defers_to_other_types():
    assert _grid().__eq__(object()) is NotImplemented
    assert _grid() != object()


@pytest.mark.parametrize(("axis", "message"), [(7, "out of range"), ("w", "unknown axis")])
def test_axis_index_rejects_unknown_axes(axis, message):
    with pytest.raises(ValueError, match=message):
        _grid().axis_index(axis)


def test_stretching_families_reject_degenerate_parameters():
    with pytest.raises(ValueError, match="ratio must be positive"):
        geometric_faces(4, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="beta must be positive"):
        tanh_faces(4, 0.0, 1.0, 0.0)


def test_unit_ratio_reproduces_uniform_spacing():
    assert np.allclose(
        geometric_faces(6, 0.0, 1.0, 1.0, both_ends=False), uniform_faces(6, 0.0, 1.0), atol=1e-15
    )
    assert np.allclose(geometric_faces(6, 0.0, 1.0, 1.0), uniform_faces(6, 0.0, 1.0), atol=1e-15)


def test_wall_resolving_faces_accept_a_uniform_growth_ratio():
    faces = wall_resolving_faces(100, -1.0, 1.0, layer_thickness=0.5, cells_in_layer=8, max_ratio=1.0)
    widths = np.diff(faces)
    assert np.allclose(widths, widths[0], atol=1e-14)
    assert float(np.sum(widths[:8])) <= 0.5 + 1.0e-9


def test_the_polar_metric_integrates_a_cylinder():
    """Volumes and areas are the metric; every stencil reads the geometry through them."""
    grid = Grid(
        uniform_faces(8, 0.0, 1.0),
        uniform_faces(16, 0.0, 2.0 * np.pi),
        uniform_faces(2, 0.0, 1.0),
        geometry=POLAR,
    )
    assert grid.is_polar
    assert grid.cell_volumes().sum() == pytest.approx(np.pi)
    # The face on the axis has no area, which is what makes it need no condition.
    assert np.max(np.abs(grid.face_areas(0)[0])) == 0.0
    assert grid.face_areas(0)[-1].sum() == pytest.approx(2.0 * np.pi)
    assert grid.face_areas(2)[:, :, 0].sum() == pytest.approx(np.pi)
    # An azimuthal face is dr*dz, with no radius in it.
    assert grid.face_areas(1)[0, 0, 0] == pytest.approx(0.125 * 0.5)


def test_a_polar_grid_checks_what_its_coordinates_mean():
    with pytest.raises(ValueError, match="geometry must be one of"):
        Grid(
            uniform_faces(2, 0.0, 1.0),
            uniform_faces(2, 0.0, 1.0),
            uniform_faces(2, 0.0, 1.0),
            geometry="toroidal",
        )
    with pytest.raises(ValueError, match="span 2\\*pi in the azimuth"):
        Grid(
            uniform_faces(2, 0.0, 1.0), uniform_faces(2, 0.0, 3.0), uniform_faces(2, 0.0, 1.0), geometry=POLAR
        )
    with pytest.raises(ValueError, match="non-negative radius"):
        Grid(
            uniform_faces(2, -1.0, 1.0),
            uniform_faces(2, 0.0, 2.0 * np.pi),
            uniform_faces(2, 0.0, 1.0),
            geometry=POLAR,
        )
    cartesian = Grid(uniform_faces(2, 0.0, 1.0), uniform_faces(2, 0.0, 1.0), uniform_faces(2, 0.0, 1.0))
    polar = Grid(
        uniform_faces(2, 0.0, 1.0),
        uniform_faces(2, 0.0, 2.0 * np.pi),
        uniform_faces(2, 0.0, 1.0),
        geometry=POLAR,
    )
    assert cartesian != polar
    assert hash(cartesian) != hash(polar)
