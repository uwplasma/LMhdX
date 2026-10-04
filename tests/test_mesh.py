import jax.numpy as jnp
import numpy as np
import pytest

from lmhdx.mesh import (
    _smooth_boundary_layer_segment,
    generate_layered_duct_mesh,
    generate_layered_duct_mesh_from_fluid_faces,
    generate_multilayer_duct_mesh,
    generate_rect_duct_mesh,
    generate_rect_duct_mesh_from_faces,
    load_tabulated_field,
    make_divergence_free_cross_section_field,
    sample_cross_section_field,
    sample_tabulated_cross_section_field,
    write_tabulated_field_npz,
)
from lmhdx.physics import WallLayer

pytestmark = pytest.mark.unit


def test_rect_duct_mesh_shape():
    mesh = generate_rect_duct_mesh(width=2.0, height=1.0, nx=2, ny=8, nz=6)
    assert mesh.nx == 2
    assert mesh.ny == 8
    assert mesh.nz == 6


def test_smooth_boundary_layer_segment_allocates_layer_without_spacing_jumps():
    faces = _smooth_boundary_layer_segment(-1.0, 1.0, 99, layer_thickness=0.002, layer_cells=10)
    widths = jnp.diff(faces)
    assert faces.shape == (100,)
    assert float(jnp.sum(widths[:10])) == pytest.approx(0.002, abs=5.0e-8)
    assert float(jnp.max(jnp.maximum(widths[1:] / widths[:-1], widths[:-1] / widths[1:]))) < 1.3
    assert widths.tolist() == pytest.approx(widths[::-1].tolist(), abs=2.0e-7)


@pytest.mark.parametrize(
    "count, layer_thickness, layer_cells",
    [(1, 0.1, 1), (8, 0.0, 2), (8, 0.1, 0), (8, 0.6, 2)],
)
def test_smooth_boundary_layer_segment_falls_back_to_uniform_for_degenerate_requests(
    count, layer_thickness, layer_cells
):
    faces = _smooth_boundary_layer_segment(
        0.0,
        1.0,
        count,
        layer_thickness=layer_thickness,
        layer_cells=layer_cells,
    )
    assert faces.tolist() == pytest.approx(jnp.linspace(0.0, 1.0, count + 1).tolist())


def test_layered_duct_mesh_has_solid_cells():
    mesh = generate_layered_duct_mesh(
        width=2.0,
        height=1.0,
        ny=8,
        nz=8,
        wall_thickness=(0.0, 0.0, 0.1, 0.1),
        wall_cells=(0, 0, 2, 2),
        target_ha=100.0,
    )
    assert mesh.fluid_mask is not None
    assert (~mesh.fluid_mask).sum() > 0


def test_moderate_ha_rect_mesh_clusters_boundary_layers():
    mesh = generate_rect_duct_mesh(width=0.2, height=0.2, ny=32, nz=32, target_ha=20.0, magnetic_axis="z")
    dy = mesh.dy
    dz = mesh.dz
    uniform_spacing = 0.2 / 32.0
    assert float(dy.min()) < float(dy.max())
    assert float(dz.min()) < float(dz.max())
    assert float(dy.min()) < 0.4 * uniform_spacing
    assert float(dz.min()) < 0.4 * uniform_spacing


def test_generate_rect_duct_mesh_from_faces_preserves_explicit_faces():
    mesh = generate_rect_duct_mesh_from_faces(
        y_faces=jnp.asarray([-0.1, -0.05, 0.0, 0.1]),
        z_faces=jnp.asarray([-0.2, 0.0, 0.2]),
        length=3.0,
        nx=3,
    )

    assert mesh.geometry == "rect_duct"
    assert mesh.nx == 3
    assert mesh.ny == 3
    assert mesh.nz == 2
    assert mesh.y_centers.tolist() == pytest.approx([-0.075, -0.025, 0.05])
    assert mesh.z_centers.tolist() == pytest.approx([-0.1, 0.1])


def test_generate_rect_duct_mesh_from_faces_rejects_invalid_faces():
    with pytest.raises(ValueError, match="strictly increasing"):
        generate_rect_duct_mesh_from_faces(y_faces=jnp.asarray([0.0, 0.0]), z_faces=jnp.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        generate_rect_duct_mesh_from_faces(y_faces=jnp.ones((2, 2)), z_faces=jnp.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="at least two"):
        generate_rect_duct_mesh_from_faces(y_faces=jnp.asarray([0.0]), z_faces=jnp.asarray([0.0, 1.0]))


def test_generate_layered_duct_mesh_from_fluid_faces_adds_wall_regions():
    mesh = generate_layered_duct_mesh_from_fluid_faces(
        fluid_y_faces=jnp.asarray([-0.1, 0.0, 0.1]),
        fluid_z_faces=jnp.asarray([-0.2, 0.0, 0.2]),
        width=0.2,
        height=0.4,
        wall_thickness=(0.02, 0.04, 0.03, 0.05),
        wall_cells=(1, 2, 1, 1),
    )

    assert mesh.geometry == "layered_duct"
    assert mesh.y_faces.tolist() == pytest.approx([-0.12, -0.1, 0.0, 0.1, 0.12, 0.14])
    assert mesh.z_faces.tolist() == pytest.approx([-0.23, -0.2, 0.0, 0.2, 0.25])
    assert mesh.fluid_mask.shape == mesh.yz_shape
    assert bool(mesh.fluid_mask[1, 1])
    assert not bool(mesh.fluid_mask[0, 1])
    assert not bool(mesh.fluid_mask[1, 0])


def test_layered_meshes_support_fluid_only_and_hartmann_targeting():
    explicit = generate_layered_duct_mesh_from_fluid_faces(
        fluid_y_faces=jnp.asarray([-0.5, 0.0, 0.5]),
        fluid_z_faces=jnp.asarray([-0.5, 0.0, 0.5]),
        width=1.0,
        height=1.0,
    )
    clustered = generate_layered_duct_mesh(width=1.0, height=1.0, ny=4, nz=4)
    targeted = generate_multilayer_duct_mesh(width=1.0, height=1.0, ny=12, nz=12, target_ha=20.0)
    assert bool(jnp.all(explicit.fluid_mask))
    assert bool(jnp.all(clustered.fluid_mask))
    assert bool(jnp.all(targeted.fluid_mask))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0.0},
        {"nx": 0},
        {"fluid_conductivity": 0.0},
        {"wall_layers": {"left": [WallLayer("bad", 1.0, 0.0)]}},
        {"wall_layers": {"left": [WallLayer("bad", 1.0, 0.1, 0)]}},
        {"wall_layers": {"left": [WallLayer("bad", -1.0, 0.1)]}},
        {"wall_layers": {"front": [WallLayer("bad", 1.0, 0.1)]}},
    ],
)
def test_multilayer_duct_mesh_rejects_invalid_inputs(kwargs):
    request = {"width": 1.0, "height": 1.0, "ny": 4, "nz": 4} | kwargs
    with pytest.raises(ValueError):
        generate_multilayer_duct_mesh(**request)


def test_generate_multilayer_duct_mesh_aligns_interfaces_and_sigma():
    wall_layers = {
        side: (
            WallLayer("aln", conductivity=1.0e-8, thickness=0.01, cells=2),
            WallLayer("metal", conductivity=1.0e6, thickness=0.02, cells=2),
        )
        for side in ("left", "right", "bottom", "top")
    }

    mesh = generate_multilayer_duct_mesh(
        width=1.0,
        height=1.0,
        ny=8,
        nz=8,
        wall_layers=wall_layers,
        fluid_conductivity=2.0,
    )

    assert mesh.fluid_mask is not None
    assert mesh.sigma is not None
    assert mesh.region_ids is not None
    assert mesh.region_names[0] == "fluid"
    assert "left:aln" in mesh.region_names
    assert float(mesh.sigma[mesh.region_ids == 0][0]) == pytest.approx(2.0)
    assert float(mesh.sigma[mesh.region_ids == mesh.region_names.index("left:aln")][0]) == pytest.approx(
        1.0e-8
    )
    assert float(mesh.sigma[mesh.region_ids == mesh.region_names.index("left:metal")][0]) == pytest.approx(
        1.0e6
    )
    y_faces = [float(value) for value in mesh.y_faces]
    z_faces = [float(value) for value in mesh.z_faces]
    assert any(abs(value + 0.5) < 1.0e-6 for value in y_faces)
    assert any(abs(value + 0.51) < 1.0e-6 for value in y_faces)
    assert any(abs(value - 0.51) < 1.0e-6 for value in y_faces)
    assert any(abs(value + 0.51) < 1.0e-6 for value in z_faces)
    assert any(abs(value - 0.51) < 1.0e-6 for value in z_faces)


def test_divergence_free_cross_section_field_has_small_discrete_divergence():
    field_fn = make_divergence_free_cross_section_field(width=2.0, height=1.5, base_bz=10.0, perturbation=0.1)
    y, z, field = sample_cross_section_field(field_fn, width=2.0, height=1.5, ny=61, nz=61)
    divergence = np.gradient(field[..., 1], y, axis=0) + np.gradient(field[..., 2], z, axis=1)
    assert np.max(np.abs(divergence)) < 0.2
    assert np.sqrt(np.mean(divergence**2)) < 0.05


def test_sample_cross_section_field_returns_expected_shape():
    field_fn = make_divergence_free_cross_section_field(width=2.0, height=1.0, base_bz=8.0, perturbation=0.1)
    y, z, field = sample_cross_section_field(field_fn, width=2.0, height=1.0, ny=21, nz=25)
    assert y.shape == (21,)
    assert z.shape == (25,)
    assert field.shape == (21, 25, 3)


def test_tabulated_field_npz_round_trip_and_sampling(tmp_path):
    field_fn = make_divergence_free_cross_section_field(width=2.0, height=1.0, base_bz=8.0, perturbation=0.1)
    y, z, field = sample_cross_section_field(field_fn, width=2.0, height=1.0, ny=21, nz=25)
    path = write_tabulated_field_npz(
        tmp_path / "field.npz",
        y=y,
        z=z,
        bx=field[..., 0],
        by=field[..., 1],
        bz=field[..., 2],
    )
    payload = load_tabulated_field(path)
    assert set(payload) == {"y", "z", "bx", "by", "bz"}
    sampled = sample_tabulated_cross_section_field(
        path, y=field[..., 0] * 0.0 + y[:, None], z=field[..., 0] * 0.0 + z[None, :]
    )
    assert sampled.shape == field.shape
    assert abs(float(sampled[..., 2].mean()) - float(field[..., 2].mean())) < 1.0e-8


def test_tabulated_field_validation_and_dimension_mismatch_paths(tmp_path):
    text_path = tmp_path / "field.txt"
    text_path.write_text("not npz")
    with pytest.raises(ValueError, match="NPZ"):
        load_tabulated_field(text_path)

    incomplete = tmp_path / "incomplete.npz"
    np.savez(incomplete, y=[0.0], z=[0.0], bx=[[0.0]])
    with pytest.raises(ValueError, match="must contain"):
        load_tabulated_field(incomplete)

    x = np.asarray([0.0, 1.0])
    y = np.asarray([0.0, 1.0])
    z = np.asarray([0.0, 1.0])
    zeros = np.zeros((2, 2, 2))
    field3d = write_tabulated_field_npz(tmp_path / "field3d.npz", x=x, y=y, z=z, bx=zeros, by=zeros, bz=zeros)
    with pytest.raises(ValueError, match="needs an x coordinate"):
        sample_tabulated_cross_section_field(field3d, y=np.asarray([[0.0]]), z=np.asarray([[0.0]]))

    field2d = write_tabulated_field_npz(
        tmp_path / "field2d.npz", y=y, z=z, bx=zeros[0], by=zeros[0], bz=zeros[0]
    )
    sampled = sample_tabulated_cross_section_field(field2d, y=np.asarray([[0.5]]), z=np.asarray([[0.5]]))
    assert sampled.shape == (1, 1, 3)
    for axis in ([0.0], [0.0, 0.0], [1.0, 0.0], [0.0, np.nan], [[0.0, 1.0]]):
        np.savez(incomplete, y=axis, z=z, bx=zeros[0], by=zeros[0], bz=zeros[0])
        with pytest.raises(ValueError, match="axes"):
            load_tabulated_field(incomplete)
    for component in (np.zeros((2, 3)), np.full((2, 2), np.nan)):
        np.savez(incomplete, y=y, z=z, bx=component, by=zeros[0], bz=zeros[0])
        with pytest.raises(ValueError, match="components"):
            load_tabulated_field(incomplete)
    with pytest.raises(ValueError, match="inside the tabulated domain"):
        sample_tabulated_cross_section_field(field2d, y=np.asarray([1.01]), z=np.asarray([0.5]))
