import dataclasses
import json
import os
import runpy
import subprocess
import sys
from dataclasses import fields
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from test_problem import _heated_duct, _same

import lmhdx
from lmhdx import Solver, Uniform
from lmhdx import io as cli
from lmhdx.cases import (
    Diagnostics,
    LoggingSpec,
    MHDState,
    NumericalFailure,
    RestartSpec,
    RunConfig,
    Solution,
    make_hartmann_case,
)
from lmhdx.fully_developed import _header
from lmhdx.fully_developed import case_mesh as _build_mesh
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
from lmhdx.output import Output, RestartLogInfo, SolverStepRecord, Start, StreamingSolverLogger, read_record
from lmhdx.problem import duct, pipe, q2d

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]


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


# Meshes and tabulated fields.


def _stub_validation_cli(
    monkeypatch: pytest.MonkeyPatch, *, case_name: str, output_dir: Path
) -> dict[str, object]:
    case = SimpleNamespace(name=case_name, output=SimpleNamespace(directory=str(output_dir)))
    solution = SimpleNamespace(state=SimpleNamespace(time=1.25, residual=0.01), mesh=SimpleNamespace())
    recorded: dict[str, object] = {}
    monkeypatch.setattr(cli, "_build_case", lambda args: case)
    monkeypatch.setattr(cli, "solve_fully_developed", lambda built_case, **_: solution)
    monkeypatch.setattr(cli, "write_paraview", lambda solved, out_dir: [])
    monkeypatch.setattr(cli, "write_profile_csv", lambda path, profile: path)
    monkeypatch.setattr(cli, "extract_centerline", lambda solved: {"y": [0.0], "u": [1.0]})
    monkeypatch.setattr(cli, "extract_midplane_profile", lambda solved, axis: {"z": [0.0], "u": [1.0]})
    monkeypatch.setattr(
        cli,
        "validation_summary",
        lambda solved, name, ha: {"case": name, "residual": 0.01, "u_max": 1.0},
    )
    monkeypatch.setattr(
        cli,
        "write_metrics_json",
        lambda payload, path: recorded.update(metrics=payload, metrics_path=path) or path,
    )
    return recorded


@pytest.mark.parametrize(
    ("arguments", "expected"),
    (
        (["--help"], ("Run and validate", "Run a named", "CASE.toml")),
        (["run", "--help"], ("solver family", "geometry and resolution", "Hartmann")),
        (["benchmark", "--help"], ("cold and warm", "Timed repetitions", "JSON")),
        (["validate", "--help"], ("duct case", "Hartmann", "L2 gate")),
    ),
)
def test_cli_help_describes_each_user_workflow(arguments, expected, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(arguments)

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert all(fragment in output for fragment in expected)


def test_cli_benchmark_branch_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    report_path = tmp_path / "benchmark.json"
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "benchmark_solver",
        lambda repeats, ha, ny, nz: {
            "case": "hartmann_ha5",
            "cold_seconds": 1.0,
            "warm_seconds": 0.5,
            "mean_seconds": 0.75,
            "repeats": float(repeats),
            "backend": "cpu",
            "device_kind": "cpu",
            "jax_version": "0",
            "python_version": "3",
        },
    )
    monkeypatch.setattr(
        cli,
        "write_benchmark_report",
        lambda payload, path: recorded.update(payload=payload, path=path) or Path(path),
    )

    exit_code = cli.main(
        [
            "benchmark",
            "--repeats",
            "2",
            "--ha",
            "5",
            "--ny",
            "8",
            "--nz",
            "8",
            "--output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert recorded["path"] == str(report_path)
    assert recorded["payload"]["case"] == "hartmann_ha5"
    assert '"case": "hartmann_ha5"' in capsys.readouterr().out


def test_cli_benchmark_branch_skips_writer_when_output_empty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(
        cli,
        "benchmark_solver",
        lambda repeats, ha, ny, nz: {
            "case": "hartmann_ha5",
            "cold_seconds": 1.0,
            "warm_seconds": 0.5,
            "mean_seconds": 0.75,
        },
    )
    monkeypatch.setattr(
        cli,
        "write_benchmark_report",
        lambda payload, path: (_ for _ in ()).throw(AssertionError("unexpected write")),
    )

    exit_code = cli.main(["benchmark"])

    assert exit_code == 0
    assert '"case": "hartmann_ha5"' in capsys.readouterr().out


def test_cli_run_branch_uses_case_builder_and_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    output_dir = tmp_path / "run"
    case = cli._build_case(SimpleNamespace(case="hartmann", ha=5.0, output=str(output_dir)))
    solution = SimpleNamespace(state=SimpleNamespace(time=1.25, residual=0.01), mesh=SimpleNamespace())
    recorded: list[tuple[str, object]] = []

    monkeypatch.setattr(cli, "_build_case", lambda args: case)
    monkeypatch.setattr(
        cli,
        "solve_fully_developed",
        lambda built_case, **_: recorded.append(("solve", built_case)) or solution,
    )
    monkeypatch.setattr(
        cli,
        "write_solution_outputs",
        lambda solved, built_case, out_dir, write_npz, write_plots: (
            recorded.append(("outputs", out_dir)) or {"paraview": [], "csv": [], "npz": [], "plots": []}
        ),
    )

    exit_code = cli.main(["run", "hartmann", "--output", str(output_dir)])

    assert exit_code == 0
    assert recorded[0][0] == "solve"
    assert recorded[1][0] == "outputs"
    assert f'"case": "{case.name}"' in capsys.readouterr().out


def test_cli_dispatches_direct_toml_run(monkeypatch: pytest.MonkeyPatch):
    recorded: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "run_from_toml",
        lambda path: recorded.update(path=path) or {"case": "demo"},
    )

    assert cli.main(["/tmp/demo_case.toml"]) == 0
    assert recorded["path"] == "/tmp/demo_case.toml"
    # `run case.toml` from older scripts reaches the same path.
    assert cli.main(["run", "/tmp/old_case.toml"]) == 0
    assert recorded["path"] == "/tmp/old_case.toml"


def test_cli_returns_nonzero_for_recorded_unconverged_steady_result():
    assert cli._summary_exit_code({"solver_mode": "steady", "converged": False}) == 2
    assert cli._summary_exit_code({"solver_mode": "steady", "converged": True}) == 0
    assert cli._summary_exit_code({"solver_mode": "transient", "converged": False}) == 0


@pytest.mark.parametrize("name", ("hartmann", "shercliff", "hunt"))
def test_fully_developed_cli_resolution_reaches_case(name):
    case = cli._build_case(SimpleNamespace(case=name, ha=5.0, output=None, ny=5, nz=7))
    assert (case.geometry.ny, case.geometry.nz) == (5, 7)


def test_cli_case_builders_reject_unknown_case():
    with pytest.raises(ValueError):
        cli._build_case(SimpleNamespace(case="unknown", ha=5.0, output="out"))


def test_python_module_entrypoint_delegates_to_cli_main(
    monkeypatch: pytest.MonkeyPatch,
):
    recorded: dict[str, object] = {}

    monkeypatch.setattr("lmhdx.io.main", lambda argv=None: recorded.update(argv=argv) or 0)
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("lmhdx", run_name="__main__")

    assert excinfo.value.code == 0
    assert recorded["argv"] is None


def test_run_config_uses_restart_bundle_and_writes_restart_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    output_dir = tmp_path / "run"
    case = cli._build_case(SimpleNamespace(case="hartmann", ha=5.0, output=str(output_dir)))
    case = case.__class__(
        **{
            **case.__dict__,
            "output": case.output.__class__(**{**case.output.__dict__, "directory": str(output_dir)}),
        }
    )
    config = RunConfig(
        case=case,
        logging=LoggingSpec(enabled=False),
        restart=RestartSpec(
            enabled=True,
            path=tmp_path / "restart.npz",
            reset_histories=False,
            write_restart=True,
            restart_filename="resume_state.npz",
        ),
    )
    bundle = SimpleNamespace(
        path=(tmp_path / "restart.npz").resolve(),
        state=SimpleNamespace(time=0.2),
        diagnostics=SimpleNamespace(
            time_history=[],
            u_max_history=[],
            mean_velocity_history=[],
            applied_forcing_history=[],
            residual_history=[],
            courant_like=[],
            ohmic_power=[],
            current_max_history=[],
            face_current_max_history=[],
            emf_max_history=[],
            lorentz_max_history=[],
            potential_residual_history=[],
            potential_iterations_history=[],
        ),
    )
    solution = SimpleNamespace(
        state=SimpleNamespace(time=0.4, residual=1e-3, u=cli.jnp.asarray([[1.0]])),
        diagnostics=SimpleNamespace(
            potential_residual_history=cli.jnp.asarray([1e-4]),
            potential_iterations_history=cli.jnp.asarray([12.0]),
        ),
        mesh=SimpleNamespace(),
        case_name=case.name,
    )

    monkeypatch.setattr(cli, "load_restart_bundle", lambda path: bundle)
    monkeypatch.setattr(
        cli,
        "validate_restart_bundle",
        lambda bundle, mesh, geometry_kind, case_name: None,
    )
    monkeypatch.setattr(cli, "case_mesh", lambda built_case: SimpleNamespace())
    perf_values = iter([10.0, 13.5])
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(
        cli,
        "_solve_case_with_optional_logger",
        lambda built_case, **kwargs: solution,
    )
    monkeypatch.setattr(
        cli,
        "write_solution_outputs",
        lambda solved, built_case, out_dir, write_npz, write_plots: {
            "paraview": [],
            "csv": [],
            "npz": [],
            "plots": [],
        },
    )
    monkeypatch.setattr(cli, "write_restart_npz", lambda solved, built_case, path: Path(path))

    summary = cli._run_config(config)

    assert summary["restart"]["enabled"] is True
    assert summary["restart"]["start_time"] == pytest.approx(0.2)
    assert summary["restart"]["output"] == "resume_state.npz"
    assert summary["execution_seconds"] == pytest.approx(3.5)
    assert '"restart"' in capsys.readouterr().out


def test_cli_validate_hartmann_branch_writes_analytic_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    output_dir = tmp_path / "validate"
    recorded = _stub_validation_cli(monkeypatch, case_name="hartmann_ha5", output_dir=output_dir)
    monkeypatch.setattr(
        cli,
        "hartmann_validation",
        lambda solved, ha: SimpleNamespace(y_profile=SimpleNamespace(l2_error=0.2, linf_error=0.3)),
    )
    monkeypatch.setattr(
        cli,
        "hartmann_acceptance",
        lambda solved, ha, l2_threshold, linf_threshold: SimpleNamespace(
            passed=True,
            l2_threshold=l2_threshold,
            linf_threshold=linf_threshold,
        ),
    )
    monkeypatch.setattr(
        cli,
        "write_analytic_comparison",
        lambda report, path, axis_name: recorded.update(analytic=path, axis=axis_name) or path,
    )
    monkeypatch.setattr(
        cli,
        "write_acceptance_report",
        lambda report, path: recorded.update(acceptance=path) or path,
    )
    exit_code = cli.main(["validate", "hartmann", "--ha", "5", "--output", str(output_dir)])

    assert exit_code == 0
    assert recorded["axis"] == "y"
    assert recorded["analytic"] == output_dir / "hartmann_ha5_analytic.json"
    assert recorded["acceptance"] == output_dir / "hartmann_ha5_acceptance.json"
    assert recorded["metrics"]["accepted"] == pytest.approx(1.0)
    assert '"y_l2_error": 0.2' not in capsys.readouterr().out


def test_solve_case_with_optional_logger_routes_steady_to_the_core(
    monkeypatch: pytest.MonkeyPatch,
):
    case = SimpleNamespace(name="demo")
    calls: list[tuple[str, object, dict]] = []

    def fake_transient(case, **kwargs):
        calls.append(("transient", case, kwargs))
        return "transient-ok"

    def fake_steady(case, **kwargs):
        calls.append(("steady", case, kwargs))
        return "steady-ok"

    monkeypatch.setattr(cli, "solve_fully_developed_transient", fake_transient)
    monkeypatch.setattr(cli, "solve_fully_developed", fake_steady)
    logger = object()

    assert cli._solve_case_with_optional_logger(case, solve_mode="transient", logger=logger) == "transient-ok"
    restart = SimpleNamespace(time=2.5)
    assert (
        cli._solve_case_with_optional_logger(case, solve_mode="steady", logger=logger, initial_state=restart)
        == "steady-ok"
    )
    assert calls == [
        (
            "transient",
            case,
            {
                "logger": logger,
                "initial_state": None,
                "initial_diagnostics": None,
                "append_diagnostics": False,
                "restart_info": None,
            },
        ),
        ("steady", case, {"logger": logger, "start_time": 2.5}),
    ]


def test_write_run_summary_respects_disabled_json_summary(tmp_path: Path):
    case = SimpleNamespace(name="demo", output=SimpleNamespace(write_json_summary=False))
    assert cli._write_run_summary({"case": "demo"}, case, tmp_path) is None


def test_run_config_requires_restart_path(tmp_path: Path):
    case = cli._build_case(SimpleNamespace(case="hartmann", ha=5.0, output=str(tmp_path)))
    config = RunConfig(
        case=case,
        logging=LoggingSpec(enabled=False),
        restart=RestartSpec(enabled=True, path=None),
    )

    with pytest.raises(ValueError, match="restart.path"):
        cli._run_config(config)


def test_run_branch_quiet_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    case = cli._build_case(SimpleNamespace(case="hartmann", ha=5.0, output=str(tmp_path)))
    recorded: dict[str, object] = {}

    monkeypatch.setattr(cli, "_build_case", lambda args: case)
    monkeypatch.setattr(
        cli,
        "_run_config",
        lambda config: recorded.update(enabled=config.logging.enabled) or {"case": case.name},
    )

    exit_code = cli.main(["run", "hartmann", "--output", str(tmp_path), "--quiet"])

    assert exit_code == 0
    assert recorded["enabled"] is False


# Meshes and tabulated fields.


def test_streaming_solver_logger_prints_live_solver_sections():
    stream = StringIO()
    logger = StreamingSolverLogger(LoggingSpec(step_stride=1), stream=stream)
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    logger.emit_header(
        **_header(case, _build_mesh(case), "steady", "staggered core / fast diagonalization"),
        restart=RestartLogInfo(enabled=False),
    )
    logger.emit_step(_sample_record())
    logger.emit_footer(case.name, 0.1, 1e-6)

    text = stream.getvalue()
    assert "LMhdX solver" in text
    assert f"case={case.name}" in text
    assert "potential_residual=" in text
    assert "linear_residual=" in text
    assert "charge=" in text
    assert "ohmic_power=" in text
    assert "progress=" in text
    assert "remaining=" in text
    assert "completed case=" in text


def _sample_record(step_index: int = 1) -> SolverStepRecord:
    return SolverStepRecord(
        step_index=step_index,
        time=0.1 * step_index,
        u_max=1.0,
        mean_velocity=0.5,
        current_max=0.2,
        lorentz_max=0.08,
        residual=1e-6,
        potential_residual=1e-5,
        potential_iterations=12.0,
        linear_residual=1e-6,
        linear_iterations=8.0,
        applied_forcing=1.0,
        courant_like=0.05,
        ohmic_power=0.01,
        volumetric_flow_rate=0.9,
        div_current_max=1e-8,
        charge_balance_residual=2e-9,
        gauge_residual=1e-10,
        interface_current_residual=1e-8,
        potential_initial_residual=2e-5,
        linear_initial_residual=3e-6,
    )


def test_streaming_solver_logger_respects_disable_stride_and_restart_sections():
    disabled_stream = StringIO()
    disabled_logger = StreamingSolverLogger(LoggingSpec(enabled=False), stream=disabled_stream)
    disabled_logger.emit_step(_sample_record())
    assert disabled_stream.getvalue() == ""

    step_stream = StringIO()
    extra_stream = StringIO()
    logger = StreamingSolverLogger(LoggingSpec(step_stride=2, print_footer=False), stream=step_stream)
    logger.add_stream(extra_stream)
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    logger.emit_header(
        **_header(case, _build_mesh(case), "steady", "staggered core / fast diagonalization"),
        restart=RestartLogInfo(enabled=True, path="restart.npz", start_time=0.2, reset_histories=False),
    )
    logger.emit_step(_sample_record(step_index=2))
    logger.emit_footer(case.name, 0.2, 1e-6)

    text = step_stream.getvalue()
    assert "potential=staggered core" in text
    assert "restart=restart.npz" in text
    assert "step=2" not in text
    assert "completed case=" not in text
    assert extra_stream.getvalue() == text


def test_streaming_solver_logger_progress_falls_back_without_header_context():
    stream = StringIO()
    logger = StreamingSolverLogger(LoggingSpec(step_stride=1), stream=stream)
    logger.emit_step(_sample_record(step_index=3))
    text = stream.getvalue()
    assert "elapsed=" in text
    assert "average_step=" in text


# Output (plan H4.4): restart, ParaView files, summary and log for every run type.


def _vtk(path):
    """The cell arrays of a ParaView file, by name, and its type and extent."""
    root = ElementTree.parse(path).getroot()
    arrays = {i.get("Name"): np.array(i.text.split(), dtype=float) for i in root.iter("DataArray")}
    return root.get("type"), root[0].get("WholeExtent"), arrays


_RUNS = {
    "euler": (lambda: duct(5, 12, cells_in_layer=3), Solver(dt=0.01, steps=8), 4),
    "thermal": (lambda: _heated_duct()[0], Solver(dt=2.0e-3, steps=40), 10),
    "q2d": (lambda: q2d(shape=(16, 16)), Solver(dt=0.01, steps=20, history=5), 5),
}


@pytest.mark.parametrize("name", _RUNS)
def test_a_run_in_chunks_or_from_its_record_is_the_run_in_one_bit_for_bit(tmp_path, name):
    build, solver, every = _RUNS[name]
    problem, half = build(), dataclasses.replace(solver, steps=solver.steps // 2)
    whole = lmhdx.solve(problem, solver)
    for chunk in (0, every):  # an Output changes no number, whether the run is one chunk or several
        found = lmhdx.solve(problem, solver, Output(str(tmp_path / f"c{chunk}"), every=chunk))
        assert _same(found.fields, whole.fields) and _same(found.history, whole.history)
    # 20 + 20 steps are 40: the second half starts from the first half's record.
    lmhdx.solve(problem, half, Output(str(tmp_path / "half")))
    record = tmp_path / "half" / f"run_{half.steps:06d}.npz"
    again = lmhdx.solve(problem, half, Output(str(tmp_path / "again")), start=record)
    assert _same(again.fields, whole.fields)
    assert (tmp_path / "again" / f"run_{solver.steps:06d}.npz").exists()
    assert {p.name for p in (tmp_path / "again").glob("*.pvd")} == {f"run_from_{half.steps:06d}.pvd"}
    for file in ("run_%06d.npz" % solver.steps,):
        with np.load(tmp_path / "again" / file) as resumed, np.load(tmp_path / f"c{every}" / file) as chunked:
            assert all(np.array_equal(resumed[k], chunked[k]) for k in chunked.files if k != "notes")


def test_a_record_resumes_only_the_run_that_wrote_it(tmp_path):
    problem, solver = duct(5, 12, cells_in_layer=3), Solver(dt=0.01, steps=4)
    lmhdx.solve(problem, solver, Output(str(tmp_path)))
    record = tmp_path / "run_000004.npz"
    with pytest.raises(ValueError, match="another problem or solver"):
        lmhdx.solve(duct(6, 12, cells_in_layer=3), solver, start=record)
    with pytest.raises(ValueError, match="another problem or solver"):
        lmhdx.solve(problem, dataclasses.replace(solver, dt=0.02), start=record)
    # An import takes the state of any record of the same fields.
    imported = lmhdx.solve(
        duct(6, 12, cells_in_layer=3), solver, start=Start(path=str(record), mode="import")
    )
    assert imported.velocity[0].shape == (2, 12, 12)
    with np.load(record) as arrays:
        arrays = dict(arrays)
    hot = arrays["leaf0"].copy()
    hot[0, 0, 0] = np.nan
    bad, shape = "is not a lmhdx.record record of version 1", arrays["leaf0"].shape
    for change, message in (
        (dict(version=2), bad),
        (dict(leaf0=arrays["leaf0"][:, :3]), rf"leaf 0 is float64\(2, 3, {shape[2]}\)"),
        (dict(leaf0=hot), "leaf 0 is not finite"),
        (dict(leaf1=arrays["leaf1"].astype(np.float32)), "leaf 1 is float32"),
        (dict(leaf9=arrays["leaf1"]), bad),
    ):
        np.savez(tmp_path / "bad.npz", **{**arrays, **change})
        with pytest.raises(ValueError, match=message):
            lmhdx.solve(problem, solver, start=tmp_path / "bad.npz")
    np.savez(tmp_path / "bad.npz", **{k: v for k, v in arrays.items() if k != "leaf3"})
    with pytest.raises(ValueError, match=bad):
        lmhdx.solve(problem, solver, start=tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="multiples of output.every"):
        lmhdx.solve(problem, solver, Output(str(tmp_path / "x"), every=3))
    with pytest.raises(ValueError, match="concrete"):
        jax.grad(
            lambda b: (
                lmhdx.solve(problem.replace(B=Uniform(0.0, b, 0.0)), solver, Output(str(tmp_path / "y")))
                .velocity[0]
                .data.sum()
            )
        )(5.0)
    with pytest.raises(ValueError, match="no start"):
        lmhdx.solve(problem, start=record)


def test_a_steady_solve_writes_its_fields_once(tmp_path):
    problem = duct(20, 24)
    found = lmhdx.solve(problem, output=Output(str(tmp_path), name="hartmann"))
    assert _same(found.fields, lmhdx.solve(problem).fields)
    assert {p.name for p in tmp_path.iterdir()} == {
        "hartmann_000000.npz",
        "hartmann_000000.vtr",
        "hartmann.pvd",
        "hartmann.log",
        "hartmann.jsonl",
        "summary.json",
    }
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["accepted"] and summary["steps"] == 0 and summary["method"] == "stokes"
    assert summary["flow_rate"] == float(found.flow_rate) and summary["residual"] == float(found.residual)
    assert found.summary == summary
    state, step = read_record(tmp_path / "hartmann_000000.npz", found.fields)
    assert step == 0 and _same(state, found.fields)
    # ParaView reads the stretched grid as it is, and the arrays as they are, x fastest.
    kind, extent, arrays = _vtk(tmp_path / "hartmann_000000.vtr")
    grid = problem.grid
    assert kind == "RectilinearGrid" and extent == "0 1 0 24 0 24"
    assert np.allclose(arrays["Y"], grid.y, rtol=1e-12) and np.allclose(arrays["Z"], grid.z, rtol=1e-12)
    assert np.allclose(arrays["pressure"], np.transpose(found.pressure.data).ravel(), rtol=1e-12)
    centred = [np.moveaxis(np.asarray(v.data), a, 0) for a, v in enumerate(found.velocity)]
    centred = [np.moveaxis(0.5 * (c[:-1] + c[1:]), 0, a) for a, c in enumerate(centred)]
    expected = np.stack([np.transpose(c) for c in centred], axis=-1).ravel()
    assert np.allclose(arrays["velocity"], expected, rtol=1e-12, atol=1e-14)
    text = (tmp_path / "hartmann.log").read_text()
    assert text.startswith(
        "LMhdX solver\ncase=hartmann mode=steady solver=stokes geometry=cartesian cells=(1,24,24)"
    )
    assert "completed case=hartmann" in text


def test_a_pipe_is_written_as_a_structured_grid_at_its_true_points(tmp_path):
    problem = pipe(5, 6, 8, cells_in_layer=2)
    found = lmhdx.solve(problem, output=Output(str(tmp_path), fields=("velocity",)))
    kind, extent, arrays = _vtk(tmp_path / "run_000000.vts")
    assert (
        kind == "StructuredGrid"
        and extent == f"0 6 0 8 0 {problem.grid.shape[2]}"
        and "pressure" not in arrays
    )
    points = arrays["Points"].reshape(-1, 3)
    assert np.allclose(np.max(np.hypot(points[:, 0], points[:, 1])), problem.grid.x[-1], rtol=1e-12)
    r, theta = (
        problem.grid.x,
        problem.grid.y,
    )  # x runs fastest, then the angle: the first two points are on one ray
    assert np.allclose(points[1], [r[1] * np.cos(theta[0]), r[1] * np.sin(theta[0]), problem.grid.z[0]])
    assert np.allclose(points[len(r)], [r[0] * np.cos(theta[1]), r[0] * np.sin(theta[1]), problem.grid.z[0]])
    assert np.allclose(arrays["velocity"], np.transpose(found.velocity[0].data).ravel(), rtol=1e-12)


def test_a_rejected_chunk_stops_the_run_and_keeps_the_chunk_before(tmp_path):
    problem, solver = _heated_duct()[0], Solver(dt=2.0e-3, steps=40, advective_cfl_limit=4.0e-5)
    with pytest.raises(NumericalFailure, match="rejected at step") as caught:
        lmhdx.solve(problem, solver, Output(str(tmp_path), every=10))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert caught.value.summary == summary and not summary["accepted"] and summary["steps"] == 10
    assert summary["rejected"]["reasons"] == ["advective_cfl over 4e-05"] and summary["rejected"]["step"] > 10
    written = {p.name for p in tmp_path.iterdir()}
    assert {"run_000010.npz", "run_000020_rejected.npz", "run.pvd"} <= written
    assert not written & {"run_000020.npz", "run_000020.vtr"}
    # The rejected state imports but does not resume.
    with pytest.raises(ValueError, match="another problem or solver"):
        lmhdx.solve(problem, solver, start=tmp_path / "run_000020_rejected.npz")
    assert lmhdx.solve(
        problem,
        Solver(dt=2.0e-3, steps=10),
        start=Start(path=str(tmp_path / "run_000020_rejected.npz"), mode="import"),
    )
    with pytest.raises(NumericalFailure, match="rejected at step"):
        lmhdx.solve(problem, solver)  # without an Output the run raises just the same
    with pytest.raises(ValueError, match="exceeds buoyancy_courant_limit"):  # the start is held to it too
        lmhdx.solve(problem, Solver(dt=2.0e-3, steps=10, buoyancy_courant_limit=1.0e-3))
    # Conduction alone is not held to an advective limit.
    quiet = problem.replace(thermal=dataclasses.replace(problem.thermal, advection="off"))
    assert lmhdx.solve(
        quiet, Solver(dt=2.0e-3, steps=10, advective_cfl_limit=1.0e-9), Output(str(tmp_path / "q"))
    ).summary["accepted"]


def test_the_log_keeps_the_solvers_text_and_adds_a_json_line_per_chunk(tmp_path):
    lmhdx.solve(
        _heated_duct()[0], Solver(dt=2.0e-3, steps=40), Output(str(tmp_path), every=20, name="heated")
    )
    lines = (tmp_path / "heated.log").read_text().splitlines()
    assert lines[0] == "LMhdX solver"
    assert (
        lines[2]
        == "domain=(1.000000e+00,2.000000e+00,2.000000e+00) dt=2.000000e-03 end=8.000000e-02 max_steps=40"
    )
    assert [line.split()[0] for line in lines if line.startswith("step=")] == ["step=20", "step=40"]
    assert sum(line.startswith("progress=50.0%") for line in lines) == 1 and lines[-1].startswith(
        "completed case=heated"
    )
    records = [json.loads(line) for line in (tmp_path / "heated.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [20, 40] and records[1]["time"] == pytest.approx(0.08, rel=1e-12)
    assert records[1]["finite"] == 1.0 and "heat_balance_residual" in records[1]


def _write(tmp_path: Path, text: str, name: str = "case.toml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_a_file_with_an_output_writes_its_files_and_exits_by_the_summary(tmp_path, capsys):
    """``lmhdx case.toml`` prints the summary of ``[output]`` and exits 0, or 2 when a chunk is rejected."""
    text = (ROOT / "examples/heated_duct.toml").read_text() + '\n[output]\ndir = "out"\nevery = 10\n'
    path = _write(tmp_path, text)
    assert cli.main([str(path)]) == 0
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert capsys.readouterr().out.endswith(json.dumps(summary, indent=2) + "\n")  # after the log's text
    assert summary["accepted"] and summary["case"] == "heated_duct" and summary["steps"] == 40
    assert {f"heated_duct_{n:06d}.npz" for n in (10, 20, 30, 40)} <= {
        p.name for p in (tmp_path / "out").iterdir()
    }
    # A file resumes from a record, and its run ends on the same record as the run in one.
    again = _write(tmp_path, text + '[start]\npath = "out/heated_duct_000020.npz"\n', "again.toml")
    assert cli.main([str(again), "solver.steps=20", 'output.dir="again"', "output.every=20"]) == 0
    with (
        np.load(tmp_path / "out/heated_duct_000040.npz") as whole,
        np.load(tmp_path / "again/heated_duct_000040.npz") as part,
    ):
        assert all(np.array_equal(whole[k], part[k]) for k in whole.files if k != "notes")
    assert cli.main([str(path), "solver.advective_cfl_limit=4e-5", 'output.dir="late"']) == 2
    assert not json.loads((tmp_path / "late/summary.json").read_text())["accepted"]
    assert (tmp_path / "late/heated_duct_000020_rejected.npz").exists()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_command_runs_in_a_fresh_process_in_either_precision(tmp_path, dtype):
    """``python -m lmhdx case.toml`` in its own process writes records of the solver's dtype."""
    text = (ROOT / "examples/heated_duct.toml").read_text() + "\n[output]\nevery = 40\n"
    _write(tmp_path, text.replace("steps = 40", f'steps = 40\ndtype = "{dtype}"'))
    environment = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    result = subprocess.run(
        [sys.executable, "-m", "lmhdx", "case.toml"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads((tmp_path / "out/summary.json").read_text())["accepted"]
    with np.load(tmp_path / "out/heated_duct_000040.npz") as record:
        assert {record[k].dtype for k in record.files if k.startswith("leaf")} == {np.dtype(dtype)}
