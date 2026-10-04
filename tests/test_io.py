from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from lmhdx.cases import make_hartmann_case
from lmhdx.io import (
    _prepare_plot_output,
    _save_figure_pair,
    load_restart_bundle,
    validate_restart_bundle,
    write_case_overview_plots,
    write_paraview,
    write_solution_npz,
    write_solution_outputs,
)
from lmhdx.solvers import _build_mesh
from lmhdx.specs import Diagnostics, MHDState, Solution

pytestmark = pytest.mark.unit


def _sample_solution(case) -> Solution:
    mesh = _build_mesh(case)
    shape = mesh.yz_shape
    base = jnp.ones(shape)
    state = MHDState(
        u=base,
        phi=2.0 * base,
        jy=3.0 * base,
        jz=4.0 * base,
        lorentz_x=5.0 * base,
        time=0.25,
        residual=1.0e-6,
    )
    diagnostics = Diagnostics(
        residual_history=jnp.asarray([1.0e-2, 1.0e-4, 1.0e-6]),
        courant_like=jnp.asarray([0.05, 0.04, 0.03]),
        ohmic_power=jnp.asarray([0.2, 0.15, 0.1]),
        time_history=jnp.asarray([0.0, 0.125, 0.25]),
        u_max_history=jnp.asarray([1.0, 1.0, 1.0]),
        mean_velocity_history=jnp.asarray([0.8, 0.85, 0.9]),
        applied_forcing_history=jnp.asarray([1.2, 1.1, 1.0]),
        current_max_history=jnp.asarray([1.5, 1.4, 1.3]),
        face_current_max_history=jnp.asarray([1.6, 1.5, 1.4]),
        emf_max_history=jnp.asarray([0.9, 0.8, 0.7]),
        lorentz_max_history=jnp.asarray([0.7, 0.6, 0.5]),
        face_lorentz_max_history=jnp.asarray([0.75, 0.65, 0.55]),
        potential_residual_history=jnp.asarray([1.0e-3, 1.0e-4, 1.0e-5]),
        potential_iterations_history=jnp.asarray([10.0, 8.0, 6.0]),
        linear_residual_history=jnp.asarray([1.0e-2, 1.0e-4, 1.0e-6]),
        linear_iterations_history=jnp.asarray([12.0, 10.0, 8.0]),
        volumetric_flow_rate_history=jnp.asarray([0.9, 0.95, 1.0]),
        mean_current_magnitude_history=jnp.asarray([0.5, 0.45, 0.4]),
        lorentz_power_history=jnp.asarray([0.3, 0.25, 0.2]),
        div_current_max_history=jnp.asarray([1.0e-6, 5.0e-7, 2.5e-7]),
        charge_balance_residual_history=jnp.asarray([1.0e-7, 8.0e-8, 6.0e-8]),
        gauge_residual_history=jnp.asarray([1.0e-8, 5.0e-9, 2.5e-9]),
        interface_current_residual_history=jnp.asarray([1.0e-6, 8.0e-7, 6.0e-7]),
    )
    return Solution(mesh=mesh, state=state, diagnostics=diagnostics, case_name=case.name)


def test_paraview_writer(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=16, nz=16)
    solution = _sample_solution(case)
    paths = write_paraview(solution, tmp_path)
    assert all(path.exists() for path in paths)


def test_write_solution_npz(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    solution = _sample_solution(case)

    path = write_solution_npz(solution, case, tmp_path / "hartmann_results.npz")

    assert path.exists()
    with np.load(path, allow_pickle=False) as data:
        assert "u" in data
        assert "phi" in data
        assert "state_time" in data
        assert "state_residual" in data
        assert {item.name for item in fields(Diagnostics)} <= set(data.files)
        assert data["u"].shape == solution.state.u.shape


def test_load_restart_bundle_round_trips_solution_npz(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    solution = _sample_solution(case)
    path = write_solution_npz(solution, case, tmp_path / "hartmann_results.npz")

    bundle = load_restart_bundle(path)

    validate_restart_bundle(
        bundle,
        mesh=_build_mesh(case),
        geometry_kind=case.geometry.kind,
        case_name=case.name,
    )
    assert bundle.path == path.resolve()
    assert bundle.geometry_kind == case.geometry.kind
    assert bundle.state.u.shape == solution.state.u.shape
    assert float(bundle.state.time) == pytest.approx(float(solution.state.time))
    assert float(bundle.state.residual) == pytest.approx(float(solution.state.residual))
    for item in fields(Diagnostics):
        np.testing.assert_allclose(
            getattr(bundle.diagnostics, item.name),
            getattr(solution.diagnostics, item.name),
        )


def test_load_restart_bundle_falls_back_to_metadata_and_residual_history(
    tmp_path: Path,
):
    path = tmp_path / "restart_minimal.npz"
    np.savez_compressed(
        path,
        metadata_json='{"case": "demo", "time": 0.75, "geometry_kind": "rect_duct"}',
        y_faces=np.array([-1.0, 1.0]),
        z_faces=np.array([-1.0, 1.0]),
        u=np.array([[1.0]]),
        phi=np.array([[0.0]]),
        jy=np.array([[0.0]]),
        jz=np.array([[0.0]]),
        lorentz_x=np.array([[0.0]]),
        residual_history=np.array([1.0e-2, 1.0e-3]),
    )

    bundle = load_restart_bundle(path)

    assert float(bundle.state.time) == pytest.approx(0.75)
    assert float(bundle.state.residual) == pytest.approx(1.0e-3)
    assert bundle.diagnostics.time_history.shape == (0,)


def test_validate_restart_bundle_rejects_geometry_shape_faces_and_case_mismatch(
    tmp_path: Path,
):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    solution = _sample_solution(case)
    path = write_solution_npz(solution, case, tmp_path / "hartmann_results.npz")
    bundle = load_restart_bundle(path)
    mesh = _build_mesh(case)

    with pytest.raises(ValueError, match="geometry_kind"):
        validate_restart_bundle(bundle, mesh=mesh, geometry_kind="pipe", case_name=case.name)

    wrong_shape_bundle = bundle.__class__(
        **{
            **bundle.__dict__,
            "state": bundle.state.__class__(**{**bundle.state.__dict__, "u": jnp.zeros((1, 1))}),
        }
    )
    with pytest.raises(ValueError, match="field shape"):
        validate_restart_bundle(
            wrong_shape_bundle,
            mesh=mesh,
            geometry_kind=case.geometry.kind,
            case_name=case.name,
        )

    wrong_y_faces = bundle.__class__(**{**bundle.__dict__, "y_faces": np.array([0.0, 1.0])})
    with pytest.raises(ValueError, match="y_faces"):
        validate_restart_bundle(
            wrong_y_faces,
            mesh=mesh,
            geometry_kind=case.geometry.kind,
            case_name=case.name,
        )

    wrong_z_faces = bundle.__class__(**{**bundle.__dict__, "z_faces": np.array([0.0, 1.0])})
    with pytest.raises(ValueError, match="z_faces"):
        validate_restart_bundle(
            wrong_z_faces,
            mesh=mesh,
            geometry_kind=case.geometry.kind,
            case_name=case.name,
        )

    wrong_case = bundle.__class__(
        **{**bundle.__dict__, "metadata": {**bundle.metadata, "case": "other_case"}}
    )
    with pytest.raises(ValueError, match="Restart case"):
        validate_restart_bundle(wrong_case, mesh=mesh, geometry_kind=case.geometry.kind, case_name=case.name)


def test_write_solution_outputs_respects_output_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    case = case.__class__(
        **{
            **case.__dict__,
            "output": case.output.__class__(
                **{
                    **case.output.__dict__,
                    "write_paraview": False,
                    "write_csv_profiles": False,
                    "write_npz": False,
                    "write_plots": False,
                }
            ),
        }
    )
    solution = _sample_solution(case)

    monkeypatch.setattr(
        "lmhdx.io.write_paraview",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected paraview")),
    )
    monkeypatch.setattr(
        "lmhdx.io.write_solution_npz",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected npz")),
    )

    outputs = write_solution_outputs(solution, case, tmp_path, write_npz=True, write_plots=True)

    assert outputs == {"paraview": [], "csv": [], "npz": [], "plots": []}


def _plot_solution(case) -> Solution:
    mesh = _build_mesh(case)
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    u = 1.0 - 0.2 * y**2 - 0.3 * z**2
    phi = y - z
    zeros = jnp.zeros_like(u)
    diagnostics = Diagnostics(
        residual_history=jnp.asarray([1.0e-2, 1.0e-4]),
        courant_like=jnp.asarray([0.1, 0.05]),
        ohmic_power=jnp.asarray([0.2, 0.1]),
        time_history=jnp.asarray([0.0, 0.1]),
        u_max_history=jnp.asarray([0.8, 0.9]),
        mean_velocity_history=jnp.asarray([0.5, 0.55]),
        applied_forcing_history=jnp.asarray([1.0, 1.0]),
        current_max_history=jnp.asarray([0.4, 0.35]),
        face_current_max_history=jnp.asarray([0.38, 0.33]),
        emf_max_history=jnp.asarray([0.25, 0.2]),
        lorentz_max_history=jnp.asarray([0.18, 0.16]),
        face_lorentz_max_history=jnp.asarray([0.16, 0.14]),
        potential_residual_history=jnp.asarray([1.0e-3, 1.0e-4]),
        potential_iterations_history=jnp.asarray([8.0, 6.0]),
        linear_residual_history=jnp.asarray([1.0e-2, 1.0e-5]),
        linear_iterations_history=jnp.asarray([12.0, 8.0]),
        volumetric_flow_rate_history=jnp.asarray([0.9, 1.0]),
        mean_current_magnitude_history=jnp.asarray([0.2, 0.18]),
        lorentz_power_history=jnp.asarray([0.1, 0.09]),
        div_current_max_history=jnp.asarray([1.0e-6, 5.0e-7]),
        charge_balance_residual_history=jnp.asarray([1.0e-7, 8.0e-8]),
        gauge_residual_history=jnp.asarray([1.0e-8, 5.0e-9]),
        interface_current_residual_history=jnp.asarray([1.0e-6, 8.0e-7]),
    )
    return Solution(
        mesh=mesh,
        state=MHDState(
            u=u,
            phi=phi,
            jy=zeros,
            jz=zeros,
            lorentz_x=zeros,
            time=0.1,
            residual=1.0e-5,
        ),
        diagnostics=diagnostics,
        case_name=case.name,
    )


def _assert_figure_pair(outputs: list[Path], out_dir: Path, stem: str) -> None:
    expected = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    assert outputs == expected
    assert all(path.exists() for path in expected)


@pytest.fixture(autouse=True)
def _stub_repeated_figure_pair_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    def save(
        fig,
        out_dir: Path,
        stem: str,
        *,
        dpi: int | None = None,
        tight: bool = True,
    ) -> list[Path]:
        paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
        for path in paths:
            path.write_bytes(b"plot")
        plt.close(fig)
        return paths

    monkeypatch.setattr("lmhdx.io._save_figure_pair", save)


def test_figure_pair_owner_writes_real_png_and_pdf(tmp_path: Path):
    out_dir = _prepare_plot_output(tmp_path)
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    png, pdf = _save_figure_pair(fig, out_dir, "formats")
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.read_bytes().startswith(b"%PDF")


def test_figure_pair_owner_saves_tight_png_and_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = []
    fig = SimpleNamespace(savefig=lambda path, **kwargs: calls.append((path, kwargs)))
    monkeypatch.setattr(plt, "close", lambda figure: None)

    paths = _save_figure_pair(fig, tmp_path, "options")

    assert paths == [tmp_path / "options.png", tmp_path / "options.pdf"]
    assert calls == [(path, {"bbox_inches": "tight"}) for path in paths]


def test_write_case_overview_plots_writes_overview_and_diagnostics(tmp_path: Path):
    solution = _plot_solution(make_hartmann_case(ha=5.0, ny=8, nz=8))
    outputs = write_case_overview_plots(
        solution,
        tmp_path,
        case_title="Hartmann demo",
        y_reference_coordinate=jnp.asarray([-1.0, 0.0, 1.0]),
        y_reference_values=jnp.asarray([0.0, 1.0, 0.0]),
        z_reference_coordinate=jnp.asarray([-1.0, 0.0, 1.0]),
        z_reference_values=jnp.asarray([0.0, 1.0, 0.0]),
    )
    _assert_figure_pair(outputs[:2], tmp_path, "overview")
    _assert_figure_pair(outputs[2:], tmp_path, "diagnostics")


def test_write_case_overview_plots_skips_diagnostics_when_no_time_history(
    tmp_path: Path,
):
    solution = _plot_solution(make_hartmann_case(ha=5.0, ny=8, nz=8))
    solution = Solution(
        mesh=solution.mesh,
        state=solution.state,
        diagnostics=Diagnostics(
            residual_history=jnp.asarray([]),
            courant_like=jnp.asarray([]),
            ohmic_power=jnp.asarray([]),
            time_history=jnp.asarray([]),
        ),
        case_name=solution.case_name,
    )
    outputs = write_case_overview_plots(solution, tmp_path, case_title="No diagnostics")
    _assert_figure_pair(outputs, tmp_path, "overview")
    assert not (tmp_path / "diagnostics.png").exists()
