import runpy
from dataclasses import fields
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from lmhdx import io as cli
from lmhdx.cases import (
    Diagnostics,
    LoggingSpec,
    MHDState,
    RestartLogInfo,
    RestartSpec,
    RunConfig,
    Solution,
    SolverStepRecord,
    StreamingSolverLogger,
    default_log_path,
    make_hartmann_case,
)
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


def test_streaming_solver_logger_prints_live_solver_sections(tmp_path: Path):
    stream = StringIO()
    logger = StreamingSolverLogger(LoggingSpec(step_stride=1), stream=stream)
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    mesh = _build_mesh(case)
    assert default_log_path(tmp_path, case.name) == tmp_path / f"{case.name}.log"
    shape = mesh.yz_shape
    logger.emit_header(
        case=case,
        mesh=mesh,
        mode="steady",
        potential_solver="staggered core / fast diagonalization",
        target_mean_velocity=None,
        reference_mean_velocity=None,
        restart=RestartLogInfo(enabled=False),
    )
    logger.emit_step(_sample_record())
    logger.emit_footer(
        Solution(
            mesh=mesh,
            state=MHDState(
                u=jnp.ones(shape),
                phi=jnp.zeros(shape),
                jy=jnp.zeros(shape),
                jz=jnp.zeros(shape),
                lorentz_x=jnp.zeros(shape),
                time=0.1,
                residual=1e-6,
            ),
            diagnostics=Diagnostics(
                residual_history=jnp.asarray([1e-3, 1e-6]),
                courant_like=jnp.asarray([0.05]),
                ohmic_power=jnp.asarray([0.01]),
                time_history=jnp.asarray([0.1]),
                u_max_history=jnp.asarray([1.0]),
                mean_velocity_history=jnp.asarray([0.5]),
                applied_forcing_history=jnp.asarray([1.0]),
                current_max_history=jnp.asarray([0.2]),
                face_current_max_history=jnp.asarray([0.18]),
                emf_max_history=jnp.asarray([0.12]),
                lorentz_max_history=jnp.asarray([0.08]),
                face_lorentz_max_history=jnp.asarray([0.07]),
                potential_residual_history=jnp.asarray([1e-5]),
                potential_iterations_history=jnp.asarray([12.0]),
                linear_residual_history=jnp.asarray([1e-6]),
                linear_iterations_history=jnp.asarray([8.0]),
                volumetric_flow_rate_history=jnp.asarray([0.9]),
                mean_current_magnitude_history=jnp.asarray([0.11]),
                lorentz_power_history=jnp.asarray([0.02]),
                div_current_max_history=jnp.asarray([1e-8]),
                charge_balance_residual_history=jnp.asarray([2e-9]),
                gauge_residual_history=jnp.asarray([1e-10]),
                interface_current_residual_history=jnp.asarray([1e-8]),
            ),
            case_name=case.name,
        )
    )

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
    mesh = _build_mesh(case)
    logger.emit_header(
        case=case,
        mesh=mesh,
        mode="steady",
        potential_solver="staggered core / fast diagonalization",
        target_mean_velocity=None,
        reference_mean_velocity=None,
        restart=RestartLogInfo(enabled=True, path="restart.npz", start_time=0.2, reset_histories=False),
    )
    logger.emit_step(_sample_record(step_index=2))
    logger.emit_footer(
        Solution(
            mesh=mesh,
            state=MHDState(
                u=jnp.ones(mesh.yz_shape),
                phi=jnp.zeros(mesh.yz_shape),
                jy=jnp.zeros(mesh.yz_shape),
                jz=jnp.zeros(mesh.yz_shape),
                lorentz_x=jnp.zeros(mesh.yz_shape),
                time=0.2,
                residual=1e-6,
            ),
            diagnostics=Diagnostics(
                residual_history=jnp.asarray([1e-6]),
                courant_like=jnp.asarray([0.0]),
                ohmic_power=jnp.asarray([0.0]),
            ),
            case_name=case.name,
        )
    )

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


# A thermal run's restart record and ParaView output (plan step 5.6).


def _thermal_duct(cells=(1, 6, 6), dt=2.0e-3):
    """A buoyant, heated Hartmann duct whose flow evolves: periodic along x, walls across."""
    from lmhdx.core3d import ChannelProblem
    from lmhdx.grid import NEUMANN, PERIODIC, BoundaryCondition, Grid, uniform_faces
    from lmhdx.thermal import ThermalProblem, thermal_boundary

    wrap, wall = BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN)
    grid = Grid(
        uniform_faces(cells[0], 0.0, 1.0),
        uniform_faces(cells[1], -1.0, 1.0),
        uniform_faces(cells[2], -1.0, 1.0),
    )
    flow = ChannelProblem(
        grid, (wrap, wall, wall), magnetic_field=(0.0, 4.0, 0.0), forcing=(1.0, 0.0, 0.0), dt=dt
    )
    walls = (
        wrap,
        thermal_boundary("heat_flux", conductivity=0.5),
        thermal_boundary("temperature", 301.0, 299.0),
    )
    return ThermalProblem(flow, walls, 2.0, 0.5, 300.0, expansion_coefficient=1e-2, gravity=(0.0, 0.0, -1.0))


def test_a_thermal_run_resumes_from_its_record_bit_for_bit(tmp_path: Path):
    """Six steps straight, or three, a record written and read back, and three more: the same state, bit for bit.

    A record refuses another unit label, another problem, and, before any state is made, a missing field, a mangled
    or nonfinite array, a time that is not a finite scalar of the compute dtype, and a version that is not the
    integer 1. Given the run's provenance it resumes only that run; without, it imports the state."""
    import dataclasses

    import jax

    from lmhdx.grid import CENTER, Field
    from lmhdx.thermal import (
        advance_thermal,
        build_thermal_step,
        initial_thermal_state,
        load_thermal_record,
        write_thermal_record,
    )

    thermal = _thermal_duct()
    grid = thermal.flow.grid
    step = build_thermal_step(thermal, dtype=jnp.float64)
    heat = Field(jnp.full(grid.shape, 0.5), (CENTER,) * 3, grid)
    start = initial_thermal_state(
        thermal, 300.0 + 0.1 * np.random.default_rng(37).standard_normal(grid.shape)
    )
    straight = advance_thermal(step, start, heat, 6)[0]
    half = advance_thermal(step, start, heat, 3)[0]
    run = {"heat_source": 0.5, "hold_flow": False}
    path = write_thermal_record(tmp_path / "state", thermal, half, units="nondimensional", run=run)
    loaded = load_thermal_record(path, thermal, units="nondimensional", run=run)
    resumed = advance_thermal(step, loaded, heat, 3)[0]
    load_thermal_record(path, thermal, units="nondimensional")  # an import, of the same state
    with pytest.raises(ValueError, match="other inputs"):
        load_thermal_record(path, thermal, units="nondimensional", run={**run, "heat_source": 0.6})
    assert path.suffix == ".npz" and float(straight.time) == pytest.approx(6 * 2.0e-3, rel=1e-15)
    for a, b in zip(jax.tree_util.tree_leaves(straight), jax.tree_util.tree_leaves(resumed)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError, match="units"):
        load_thermal_record(path, thermal, units="SI")
    with pytest.raises(ValueError, match="another problem: flow"):
        load_thermal_record(
            path,
            dataclasses.replace(thermal, flow=dataclasses.replace(thermal.flow, dt=1e-3)),
            units="nondimensional",
        )
    with np.load(path) as record:
        arrays = dict(record)
    hot, fast = arrays["temperature"].copy(), arrays["velocity_x"].copy()
    hot[0, 0, 0], fast[0, 0, 0] = np.nan, np.inf
    for change, message in (
        (dict(version=2), "version"),
        (dict(version=1.5), "version"),
        (dict(temperature=arrays["temperature"][:, :3]), "shape"),
        (dict(temperature=hot), "not finite"),
        (dict(velocity_x=fast), "not finite"),
        (dict(time=np.array([0.006, 0.006])), "time"),
        (dict(time=np.array(np.nan)), "time"),
        (dict(time=np.array(6)), "time"),
    ):
        np.savez(tmp_path / "bad.npz", **{**arrays, **change})
        with pytest.raises(ValueError, match=message):
            load_thermal_record(tmp_path / "bad.npz", thermal, units="nondimensional")
    np.savez(tmp_path / "bad.npz", **{name: value for name, value in arrays.items() if name != "time"})
    with pytest.raises(ValueError, match="lacks time"):
        load_thermal_record(tmp_path / "bad.npz", thermal, units="nondimensional")


def test_a_thermal_state_opens_as_a_rectilinear_grid_in_vtk_order(tmp_path: Path):
    """The file parses as a VTK RectilinearGrid of 2 x 3 x 4 cells; ``T = x + 10 y + 100 z`` reads back with x
    varying fastest, the velocity as each component's mean over its two faces, and the coordinates as the faces."""
    import xml.etree.ElementTree as ElementTree

    from lmhdx.core3d import zero_velocity
    from lmhdx.io import write_vtr
    from lmhdx.ops import cell_average
    from lmhdx.thermal import initial_thermal_state

    thermal = _thermal_duct(cells=(2, 3, 4))
    grid = thermal.flow.grid
    x, y, z = (np.asarray(centres) for centres in grid.centers)
    temperature = x[:, None, None] + 10.0 * y[None, :, None] + 100.0 * z[None, None, :]
    rest = zero_velocity(thermal.flow)
    velocity = tuple(
        component.replace_data(jnp.asarray(np.random.default_rng(41 + axis).standard_normal(component.shape)))
        for axis, component in enumerate(rest)
    )
    state = initial_thermal_state(thermal, temperature, velocity)
    centred = jnp.stack(
        [cell_average(field, axis).data for axis, field in enumerate(state.velocity)], axis=-1
    )
    path = write_vtr(tmp_path / "frame", grid.faces, temperature=state.temperature.data, velocity=centred)
    root = ElementTree.parse(path).getroot()
    assert (
        root.get("type") == "RectilinearGrid"
        and root.find("RectilinearGrid").get("WholeExtent") == "0 2 0 3 0 4"
    )
    arrays = {item.get("Name"): np.array(item.text.split(), dtype=float) for item in root.iter("DataArray")}
    assert np.allclose(arrays["temperature"], np.transpose(temperature).ravel(), rtol=1e-12)
    centred = [np.moveaxis(np.asarray(v.data), a, 0) for a, v in enumerate(velocity)]
    centred = [np.moveaxis(0.5 * (c[:-1] + c[1:]), 0, a) for a, c in enumerate(centred)]
    expected = np.stack([np.transpose(c) for c in centred], axis=-1).ravel()
    assert np.allclose(arrays["velocity"], expected, rtol=1e-12, atol=1e-14)
    for name, faces in zip("XYZ", grid.faces):
        assert np.allclose(arrays[name], faces, rtol=1e-12)


def _page_toml() -> str:
    page = Path(__file__).resolve().parents[1] / "docs/physics/thermal.md"
    return page.read_text().split("## A run from TOML", 1)[1].split("```toml\n", 1)[1].split("```", 1)[0]


def test_a_thermal_run_from_toml_records_frames_and_resumes_bit_for_bit(tmp_path: Path):
    """The page's case through ``lmhdx case.toml``: four records and frames, a series and a summary, exit 0. Its
    second half, rerun from the record at step 20, ends on the same record, bit for bit, in a series of its own."""
    import json

    case = tmp_path / "case.toml"
    case.write_text(_page_toml())
    assert cli.main([str(case)]) == 0
    out, labels = tmp_path / "out", [f"heated_duct_{step:06d}" for step in (10, 20, 30, 40)]
    written = {path.name for path in out.iterdir()}
    expected = {f"{label}.{suffix}" for label in labels for suffix in ("npz", "vtr")}
    assert expected | {"heated_duct.pvd", "thermal_summary.json"} <= written
    summary = json.loads((out / "thermal_summary.json").read_text())
    assert summary["accepted"] and summary["rejected"] is None and summary["steps"] == 40
    assert summary["limits"] == {
        "advective_cfl": 0.8,
        "advective_cfl_override": False,
        "buoyancy_courant": None,
    }
    assert summary["finite"] and summary["time"] == pytest.approx(0.08, rel=1e-12)
    assert summary["largest"]["heat_balance_residual"] <= 1e-8 and summary["largest"]["advective_cfl"] <= 0.8
    resume = tmp_path / "resume.toml"
    text = (
        _page_toml().replace("steps = 40", "steps = 20").replace('directory = "out"', 'directory = "again"')
    )
    resume.write_text(text + '\n[restart]\npath = "out/heated_duct_000020.npz"\n')
    assert cli.main([str(resume)]) == 0
    assert {path.name for path in (tmp_path / "again").glob("*.pvd")} == {"heated_duct_from_000020.pvd"}
    with (
        np.load(out / "heated_duct_000040.npz") as straight,
        np.load(tmp_path / "again/heated_duct_000040.npz") as again,
    ):
        assert straight.files == again.files
        assert all(np.array_equal(straight[name], again[name]) for name in straight.files)


def test_a_chunk_is_rejected_at_any_sample_past_its_limits():
    """A sample inside the chunk past a limit is not hidden by a valid last one; a NaN fails; no limit, no check."""
    good = dict(
        finite=np.ones(5, bool),
        backflow_boundary_valid=np.ones(5, bool),
        advective_cfl=np.full(5, 0.1),
        buoyancy_courant=np.zeros(5),
    )
    limits = {"advective_cfl": 0.8, "buoyancy_courant": None}
    fast = SimpleNamespace(**{**good, "advective_cfl": np.array([0.1, 0.9, 0.95, 0.2, 0.1])})
    assert cli._thermal_rejection(SimpleNamespace(**good), limits, 40) is None
    assert cli._thermal_rejection(fast, limits, 40) == {
        "step": 42,
        "reasons": ["advective_cfl over 0.8"],
        "advective_cfl": 0.9,
        "buoyancy_courant": 0.0,
    }
    assert cli._thermal_rejection(fast, {**limits, "advective_cfl": None}, 40) is None
    broken = {"finite": np.array([1, 1, 0, 1, 1], bool), "buoyancy_courant": np.array([0, 0, np.nan, 0, 0])}
    rejected = cli._thermal_rejection(
        SimpleNamespace(**{**good, **broken}), {**limits, "buoyancy_courant": 1.0}, 0
    )
    assert rejected["step"] == 3 and rejected["reasons"] == ["not finite", "buoyancy_courant over 1.0"]


@pytest.mark.parametrize(
    ("limit", "reason"),
    [
        ("advective_cfl_limit = 4.0e-5", "advective_cfl over 4e-05"),
        ("buoyancy_courant_limit = 9.0e-5", "buoyancy_courant over 9e-05"),
    ],
)
def test_a_thermal_run_stops_at_the_first_sample_past_a_limit(tmp_path: Path, limit: str, reason: str):
    """The page's case with its z walls held at zero: from rest and a uniform temperature the start passes either
    limit, and the flow and the buoyancy outgrow it inside the second chunk. The run exits 2 after the first
    chunk's record, writes the second as rejected and names its first sample; that record imports, not resumes."""
    import json

    from lmhdx.cases import _load_toml
    from lmhdx.io import thermal_run_from_toml
    from lmhdx.thermal import load_thermal_record

    page = _page_toml().replace('kind = "heat_flux"\nupper = 1.0', 'kind = "temperature"')
    case = tmp_path / "case.toml"
    case.write_text(page.replace("chunk = 10", f"chunk = 10\n{limit}"))
    assert 'kind = "temperature"' in page and cli.main([str(case)]) == 2
    out = tmp_path / "out"
    summary = json.loads((out / "thermal_summary.json").read_text())
    assert not summary["accepted"] and summary["steps"] == 10 and summary["rejected"]["chunk"] == 1
    assert 10 < summary["rejected"]["step"] < 20 and summary["rejected"]["reasons"] == [reason]
    assert summary["limits"]["advective_cfl_override"] == limit.startswith("advective")
    written = {path.name for path in out.iterdir()}
    assert {"heated_duct_000010.npz", "heated_duct_000020_rejected.npz", "heated_duct.pvd"} <= written
    assert not written & {
        "heated_duct_000020.npz",
        "heated_duct_000020.vtr",
        "heated_duct_000030_rejected.npz",
    }
    run, rejected = thermal_run_from_toml(_load_toml(case), tmp_path), out / "heated_duct_000020_rejected.npz"
    with pytest.raises(ValueError, match="other inputs"):
        load_thermal_record(rejected, run.problem, units=run.units, run=run.provenance)
    attempted = load_thermal_record(rejected, run.problem, units=run.units)
    assert summary["time"] == float(attempted.time)  # `time` is the attempted chunk's, as `last` is


def test_conduction_alone_is_not_held_to_an_advective_limit(tmp_path: Path):
    """With ``advection = "off"`` no advective Courant limit applies, at the start or after, whatever is configured."""
    import json

    page = _page_toml().replace("heat_source = 0.5", 'heat_source = 0.5\nadvection = "off"')
    case = tmp_path / "case.toml"
    case.write_text(page.replace("chunk = 10", "chunk = 10\nadvective_cfl_limit = 1.0e-9"))
    assert cli.main([str(case)]) == 0
    summary = json.loads((tmp_path / "out/thermal_summary.json").read_text())
    assert summary["accepted"] and summary["limits"]["advective_cfl"] is None and summary["steps"] == 40
    assert summary["largest"]["advective_cfl"] == 0.0  # nothing is carried, though the flow moves


def _diagnostics(columns: dict):
    from lmhdx.thermal import ThermalDiagnostics

    return ThermalDiagnostics(**columns)


def test_the_run_summary_reduces_each_chunk_as_the_whole_history_would():
    """The summary's finite flag, maxima and last sample, kept chunk by chunk, equal the whole history's at every
    chunking, a NaN inside a chunk included: it clears the flag, and a maximum it enters stays NaN."""
    from lmhdx.thermal import ThermalDiagnostics

    rng, steps = np.random.default_rng(271), 60
    whole = {name: rng.standard_normal(steps) for name in ThermalDiagnostics._fields}
    whole["finite"], whole["backflow_boundary_valid"] = np.ones(steps, bool), np.ones(steps, bool)
    whole["advective_cfl"][23], whole["finite"][23] = np.nan, False
    largest = {name: float(np.max(np.abs(whole[name]))) for name in cli._LARGEST}
    last = {name: float(values[-1]) for name, values in whole.items()}
    assert np.isnan(largest["advective_cfl"])
    for size in (1, 7, 20, 60):
        extremes = cli._RunExtremes()
        for start in range(0, steps, size):
            extremes.add(_diagnostics({name: values[start : start + size] for name, values in whole.items()}))
        assert extremes.finite is False
        np.testing.assert_equal(extremes.largest, largest)
        np.testing.assert_equal(extremes.last, last)


def test_the_run_summary_does_not_grow_with_the_step_count():
    """2000 chunks of 50 samples are 17 MB of diagnostics if kept; the reduction holds a few floats."""
    import tracemalloc

    from lmhdx.thermal import ThermalDiagnostics

    rng, extremes = np.random.default_rng(5), cli._RunExtremes()
    tracemalloc.start()
    for _ in range(2000):
        extremes.add(_diagnostics({name: rng.standard_normal(50) for name in ThermalDiagnostics._fields}))
    held, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    whole = 2000 * 50 * len(ThermalDiagnostics._fields) * 8
    assert held < 64 * 1024 < whole / 100, (held, whole)


def test_a_thermal_run_reports_the_same_at_any_chunk_size(tmp_path: Path):
    """The page's case run in chunks of 10, 20 and 40 ends on the same record and the same summary; with a limit
    the flow outgrows inside its second ten steps, chunks of 5 and 10 name the same first rejected sample."""
    import json

    def run(chunk: int, extra: str = "", page: str | None = None) -> tuple[int, dict, Path]:
        text = (page or _page_toml()).replace("chunk = 10", f"chunk = {chunk}{extra}")
        out = f"out_{chunk}{'_limited' if extra else ''}"
        case = tmp_path / f"{out}.toml"
        case.write_text(text.replace('directory = "out"', f'directory = "{out}"'))
        code = cli.main([str(case)])
        return code, json.loads((tmp_path / out / "thermal_summary.json").read_text()), tmp_path / out

    runs = {chunk: run(chunk) for chunk in (10, 20, 40)}
    for chunk in (20, 40):
        code, summary, out = runs[chunk]
        assert code == 0 and summary["accepted"] and summary["steps"] == runs[10][1]["steps"] == 40
        for key in ("finite", "last", "largest", "time"):
            assert summary[key] == runs[10][1][key], key
        with (
            np.load(out / "heated_duct_000040.npz") as record,
            np.load(runs[10][2] / "heated_duct_000040.npz") as reference,
        ):
            assert all(np.array_equal(record[name], reference[name]) for name in reference.files)
    walls = _page_toml().replace('kind = "heat_flux"\nupper = 1.0', 'kind = "temperature"')
    limited = {chunk: run(chunk, "\nadvective_cfl_limit = 4.0e-5", walls) for chunk in (5, 10)}
    assert limited[5][0] == limited[10][0] == 2
    assert limited[5][1]["rejected"]["step"] == limited[10][1]["rejected"]["step"]
    assert limited[5][1]["rejected"]["reasons"] == limited[10][1]["rejected"]["reasons"]


@pytest.mark.parametrize("x64", ["", "true"])
def test_the_thermal_command_runs_in_a_fresh_process_in_either_precision(tmp_path: Path, x64: str):
    """``python -m lmhdx case.toml`` in its own process: float32 by default, float64 with ``JAX_ENABLE_X64``."""
    import json
    import os
    import subprocess
    import sys

    (tmp_path / "case.toml").write_text(_page_toml())
    environment = {key: value for key, value in os.environ.items() if key != "JAX_ENABLE_X64"}
    environment |= {"JAX_ENABLE_X64": x64} if x64 else {}
    command = [sys.executable, "-m", "lmhdx", "case.toml"]
    result = subprocess.run(
        command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads((tmp_path / "out/thermal_summary.json").read_text())["accepted"]
    with np.load(tmp_path / "out/heated_duct_000040.npz") as record:
        assert record["temperature"].dtype == (np.float64 if x64 else np.float32)
        assert np.all(np.isfinite(record["temperature"]))


def test_a_thermal_toml_refuses_what_it_cannot_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Each named fault, before a steady flow is solved: an unknown table or key, a value of the wrong TOML type,
    a missing or malformed table, an orphan option, an unsupported boundary, an open axis, steps that are not a
    multiple of the chunk, and a record of another problem or, to resume, of a run with other inputs."""
    import lmhdx.steady
    from lmhdx.cases import _load_toml
    from lmhdx.io import thermal_run_from_toml

    page = _page_toml()
    (tmp_path / "base.toml").write_text(page)
    assert cli.main([str(tmp_path / "base.toml")]) == 0
    monkeypatch.setattr(lmhdx.steady, "solve_steady_state", lambda flow: pytest.fail("solved a steady flow"))
    page = page.replace("dt = 2.0e-3", 'dt = 2.0e-3\ninitial_velocity = "steady"')
    restart = '\n[restart]\npath = "out/heated_duct_000020.npz"\n'
    changes = (
        ("heat_source = 0.5", "heat_source = 0.5\nconductivity_of_walls = 1.0", "unknown keys"),
        ("[run]", "[solvers]\nkind = 1\n\n[run]", r"\[top level\] has unknown keys \['solvers'\]"),
        ("schema_version = 1", "schema_version = 2", "schema_version must be 1"),
        ("steps = 40", "steps = 40.5", "steps must be a TOML integer"),
        ("y = { cells = 8", "y = { cells = true", "cells must be a TOML integer"),
        ("heat_source = 0.5", 'heat_source = 0.5\nhold_flow = "false"', "hold_flow must be a TOML boolean"),
        ("magnetic_field = [0.0, 10.0, 0.0]", "magnetic_field = [0.0, 10.0]", "array of three numbers"),
        ('[thermal.y]\nkind = "heat_flux"\n', "", r"\[thermal.y\] is needed"),
        ("lower = 0.0, upper = 1.0 }", "lower = 0.0 }", r"\[grid.x\] lacks \['upper'\]"),
        ("x = { cells = 1, lower = 0.0, upper = 1.0 }", "x = 1", "x must be a TOML table"),
        ("y = { cells = 8,", "y = { cells_in_layer = 2, cells = 8,", "cells_in_layer needs layer_thickness"),
        ('[thermal.y]\nkind = "heat_flux"', '[thermal.y]\nkind = "periodic"', 'kind is "temperature" or'),
        (
            '[thermal.y]\nkind = "heat_flux"',
            '[thermal.y]\nkind = "heat_flux"\nupper_kind = "a"',
            "unknown keys",
        ),
        ('conditions = ["periodic", "wall", "wall"]', 'conditions = ["inflow", "wall", "wall"]', "open axis"),
        ("chunk = 10", "chunk = 15", "multiple of chunk"),
        ("[output]", f'{restart}mode = "fork"\n\n[output]', 'mode "resume" or "import"'),
        ("dt = 2.0e-3", "dt = 1.0e-3", "another problem"),
        ("heat_source = 0.5", "heat_source = 0.6", "other inputs"),
        ("heat_source = 0.5", "heat_source = 0.5\nhold_flow = true", "other inputs"),
    )
    for old, new, message in changes:
        assert old in page
        bad = page.replace(old, new)
        (tmp_path / "bad.toml").write_text(
            bad + (restart if message in ("another problem", "other inputs") else "")
        )
        with pytest.raises(ValueError, match=message):
            thermal_run_from_toml(_load_toml(tmp_path / "bad.toml"), tmp_path)
    # The same changed inputs import the record's state as the start of another experiment.
    (tmp_path / "import.toml").write_text(
        page.replace("heat_source = 0.5", "heat_source = 0.6") + restart + 'mode = "import"\n'
    )
    imported = thermal_run_from_toml(_load_toml(tmp_path / "import.toml"), tmp_path)
    assert imported.restart["mode"] == "import" and imported.provenance["heat_source"] == 0.6
