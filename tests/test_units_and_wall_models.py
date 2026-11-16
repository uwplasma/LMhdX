import math

import numpy as np
import pytest

from lmhdx import (
    WallLayer,
    dynamic_to_kinematic_viscosity,
    effective_pinhole_conductance_ratio,
    equivalent_single_layer,
    hartmann_number,
    interaction_parameter,
    kinematic_to_dynamic_viscosity,
    magnetic_field_from_hartmann,
    magnetic_reynolds_number,
    nested_wall_layer_resolution_summary,
    normal_stack_leakage_ratio,
    reynolds_number,
    tangential_stack_conductance_ratio,
    wall_conductance_ratio,
)
from lmhdx.fields import Table, _interpolate_tabulated_field, load_tabulated_field, write_tabulated_field_npz
from lmhdx.grid import Grid, uniform_faces
from lmhdx.units import normal_leakage_ratio

pytestmark = pytest.mark.unit


def test_viscosity_conversion_round_trip_and_hartmann_convention():
    rho = 500.0
    mu = 2.5e-3
    nu = dynamic_to_kinematic_viscosity(mu, rho)

    assert nu == pytest.approx(5.0e-6)
    assert kinematic_to_dynamic_viscosity(nu, rho) == pytest.approx(mu)

    b = magnetic_field_from_hartmann(
        hartmann=20.0,
        length_scale=0.1,
        conductivity=2.0e6,
        density=rho,
        kinematic_viscosity=nu,
    )
    recovered = hartmann_number(
        magnetic_field=b,
        length_scale=0.1,
        conductivity=2.0e6,
        density=rho,
        kinematic_viscosity=nu,
    )
    assert recovered == pytest.approx(20.0)


def test_nondimensional_numbers_are_hand_calculable():
    assert reynolds_number(velocity=0.2, length_scale=0.05, kinematic_viscosity=1.0e-6) == pytest.approx(
        10000.0
    )
    assert interaction_parameter(
        magnetic_field=2.0,
        length_scale=0.1,
        conductivity=1.0e6,
        density=1000.0,
        velocity=0.5,
    ) == pytest.approx(800.0)
    assert magnetic_reynolds_number(velocity=0.5, length_scale=0.1, conductivity=1.0e6) == pytest.approx(
        4.0e-7 * math.pi * 5.0e4
    )


def test_thin_wall_and_normal_leakage_ratios_are_distinct():
    c_parallel = wall_conductance_ratio(
        wall_conductivity=5.0,
        wall_thickness=1.0e-3,
        fluid_conductivity=1.0,
        length_scale=0.1,
    )
    g_perp = normal_leakage_ratio(
        coating_conductivity=5.0,
        coating_thickness=1.0e-3,
        fluid_conductivity=1.0,
        length_scale=0.1,
    )

    assert c_parallel == pytest.approx(0.05)
    assert g_perp == pytest.approx(500.0)
    assert g_perp != c_parallel


def test_nested_wall_stack_reduces_to_equivalent_single_layer_for_tangential_conduction():
    layers = (
        WallLayer("aln", conductivity=1.0e-8, thickness=2.0e-4, cells=4),
        WallLayer("metal", conductivity=1.0e6, thickness=1.0e-3, cells=8),
    )
    stack_c = tangential_stack_conductance_ratio(layers, fluid_conductivity=1.0e6, length_scale=0.01)
    equivalent = equivalent_single_layer(layers)
    equivalent_c = wall_conductance_ratio(
        wall_conductivity=equivalent.conductivity,
        wall_thickness=equivalent.thickness,
        fluid_conductivity=1.0e6,
        length_scale=0.01,
    )

    assert stack_c == pytest.approx(equivalent_c)
    assert equivalent.cells == 12


def test_normal_stack_leakage_is_limited_by_insulating_layer():
    layers = (
        WallLayer("aln", conductivity=1.0e-8, thickness=2.0e-4, cells=4),
        WallLayer("metal", conductivity=1.0e6, thickness=1.0e-3, cells=8),
    )
    leakage = normal_stack_leakage_ratio(layers, fluid_conductivity=1.0e6, length_scale=0.01)
    single_aln = normal_leakage_ratio(
        coating_conductivity=1.0e-8,
        coating_thickness=2.0e-4,
        fluid_conductivity=1.0e6,
        length_scale=0.01,
    )

    assert leakage == pytest.approx(single_aln, rel=1.0e-8)


def test_effective_pinhole_model_interpolates_between_intact_and_metal_limits():
    assert effective_pinhole_conductance_ratio(
        intact_conductance_ratio=1.0e-8,
        metal_conductance_ratio=10.0,
        pinhole_fraction=0.0,
    ) == pytest.approx(1.0e-8)
    assert effective_pinhole_conductance_ratio(
        intact_conductance_ratio=1.0e-8,
        metal_conductance_ratio=10.0,
        pinhole_fraction=1.0,
    ) == pytest.approx(10.0)
    assert effective_pinhole_conductance_ratio(
        intact_conductance_ratio=1.0e-8,
        metal_conductance_ratio=10.0,
        pinhole_fraction=0.25,
    ) == pytest.approx(2.5000000075)


def test_nested_wall_layer_resolution_summary_marks_underresolved_layers():
    summary = nested_wall_layer_resolution_summary(
        (
            WallLayer("aln", conductivity=1.0e-8, thickness=2.0e-4, cells=2),
            WallLayer("metal", conductivity=1.0e6, thickness=1.0e-3, cells=8),
        ),
        minimum_cells_per_layer=3,
    )

    assert summary["resolution_pass"] is False
    assert summary["minimum_cells_per_layer"] == 2


@pytest.mark.parametrize(
    ("function", "kwargs", "message"),
    [
        (
            dynamic_to_kinematic_viscosity,
            {"dynamic_viscosity": 1.0, "density": 0.0},
            "density",
        ),
        (
            dynamic_to_kinematic_viscosity,
            {"dynamic_viscosity": -1.0, "density": 1.0},
            "dynamic_viscosity",
        ),
        (
            kinematic_to_dynamic_viscosity,
            {"kinematic_viscosity": 1.0, "density": 0.0},
            "density",
        ),
        (
            kinematic_to_dynamic_viscosity,
            {"kinematic_viscosity": -1.0, "density": 1.0},
            "kinematic_viscosity",
        ),
        (
            reynolds_number,
            {"velocity": 1.0, "length_scale": 0.0, "kinematic_viscosity": 1.0},
            "length_scale",
        ),
        (
            reynolds_number,
            {"velocity": 1.0, "length_scale": 1.0, "kinematic_viscosity": 0.0},
            "kinematic_viscosity",
        ),
        (
            interaction_parameter,
            {
                "magnetic_field": 1.0,
                "length_scale": 1.0,
                "conductivity": 1.0,
                "density": 1.0,
                "velocity": 0.0,
            },
            "velocity",
        ),
        (
            magnetic_reynolds_number,
            {
                "velocity": 1.0,
                "length_scale": 1.0,
                "conductivity": 1.0,
                "magnetic_permeability": 0.0,
            },
            "magnetic_permeability",
        ),
        (
            wall_conductance_ratio,
            {
                "wall_conductivity": -1.0,
                "wall_thickness": 1.0,
                "fluid_conductivity": 1.0,
                "length_scale": 1.0,
            },
            "wall_conductivity",
        ),
        (
            normal_leakage_ratio,
            {
                "coating_conductivity": -1.0,
                "coating_thickness": 1.0,
                "fluid_conductivity": 1.0,
                "length_scale": 1.0,
            },
            "coating_conductivity",
        ),
    ],
)
def test_unit_helpers_reject_invalid_physical_inputs(function, kwargs, message):
    with pytest.raises(ValueError, match=message):
        function(**kwargs)


@pytest.mark.parametrize("bad_name", ["length_scale", "conductivity", "density", "kinematic_viscosity"])
def test_hartmann_helpers_reject_nonpositive_scales(bad_name):
    kwargs = dict(length_scale=1.0, conductivity=1.0, density=1.0, kinematic_viscosity=1.0)
    kwargs[bad_name] = 0.0
    with pytest.raises(ValueError, match=bad_name):
        hartmann_number(magnetic_field=1.0, **kwargs)
    with pytest.raises(ValueError, match=bad_name):
        magnetic_field_from_hartmann(hartmann=1.0, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "intact_conductance_ratio": 0.0,
            "metal_conductance_ratio": 1.0,
            "pinhole_fraction": -0.1,
        },
        {
            "intact_conductance_ratio": -1.0,
            "metal_conductance_ratio": 1.0,
            "pinhole_fraction": 0.1,
        },
        {
            "intact_conductance_ratio": 0.0,
            "metal_conductance_ratio": -1.0,
            "pinhole_fraction": 0.1,
        },
    ],
)
def test_pinhole_model_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        effective_pinhole_conductance_ratio(**kwargs)


@pytest.mark.parametrize(
    "layers",
    [
        (),
        (WallLayer("bad", 1.0, 0.0),),
        (WallLayer("bad", -1.0, 1.0),),
        (WallLayer("bad", 1.0, 1.0, cells=-1),),
    ],
)
def test_wall_stack_rejects_invalid_layers(layers):
    with pytest.raises(ValueError):
        equivalent_single_layer(layers)


def test_wall_stack_rejects_invalid_scales_and_handles_perfect_insulator():
    layers = (WallLayer("wall", 1.0, 1.0),)
    for function in (tangential_stack_conductance_ratio, normal_stack_leakage_ratio):
        with pytest.raises(ValueError, match="fluid_conductivity"):
            function(layers, fluid_conductivity=0.0, length_scale=1.0)
        with pytest.raises(ValueError, match="length_scale"):
            function(layers, fluid_conductivity=1.0, length_scale=0.0)
    assert (
        normal_stack_leakage_ratio(
            (WallLayer("insulator", 0.0, 1.0),),
            fluid_conductivity=1.0,
            length_scale=1.0,
        )
        == 0.0
    )


def test_wall_resolution_summary_validates_controls():
    layers = (WallLayer("wall", 1.0, 1.0),)
    with pytest.raises(ValueError, match="minimum_cells"):
        nested_wall_layer_resolution_summary(layers, minimum_cells_per_layer=0)


# Tabulated fields.


def _table(tmp_path, ny=21, nz=25):
    """An affine, divergence-free field on a rectangle: ``By = 0.1 y``, ``Bz = 8 - 0.1 z``."""
    y, z = np.linspace(-1.0, 1.0, ny), np.linspace(-0.5, 0.5, nz)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    path = write_tabulated_field_npz(
        tmp_path / "field.npz", y=y, z=z, bx=0 * yy, by=0.1 * yy, bz=8.0 - 0.1 * zz
    )
    return path, yy, zz


def test_tabulated_field_npz_round_trip_and_sampling(tmp_path):
    path, yy, zz = _table(tmp_path)
    payload = load_tabulated_field(path)
    assert set(payload) == {"y", "z", "bx", "by", "bz"}
    sampled = _interpolate_tabulated_field(payload, y=yy, z=zz)
    assert sampled.shape == (*yy.shape, 3)
    np.testing.assert_allclose(sampled[..., 1], 0.1 * yy, atol=1e-12)
    np.testing.assert_allclose(sampled[..., 2], 8.0 - 0.1 * zz, atol=1e-12)
    grid = Grid(uniform_faces(1, 0.0, 1.0), uniform_faces(8, -0.9, 0.9), uniform_faces(8, -0.4, 0.4))
    imposed, scale = Table(str(path), scale=2.0).sample(grid)
    assert scale == 2.0 and imposed.components[1].shape == grid.shape


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
    assert set(load_tabulated_field(field3d)) == {"x", "y", "z", "bx", "by", "bz"}
    field2d = write_tabulated_field_npz(
        tmp_path / "field2d.npz", y=y, z=z, bx=zeros[0], by=zeros[0], bz=zeros[0]
    )
    data = load_tabulated_field(field2d)
    sampled = _interpolate_tabulated_field(data, y=np.asarray([[0.5]]), z=np.asarray([[0.5]]))
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
        _interpolate_tabulated_field(data, y=np.asarray([1.01]), z=np.asarray([0.5]))
