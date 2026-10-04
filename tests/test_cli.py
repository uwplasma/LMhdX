import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from lmhdx import cli
from lmhdx.cases import LoggingSpec, RestartSpec, RunConfig

pytestmark = pytest.mark.unit


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

    exit_code = cli.main(["/tmp/demo_case.toml"])

    assert exit_code == 0
    assert recorded["path"] == "/tmp/demo_case.toml"


def test_cli_returns_nonzero_for_recorded_unconverged_steady_result():
    assert cli._summary_exit_code({"solver_mode": "steady", "converged": False}) == 2
    assert cli._summary_exit_code({"solver_mode": "steady", "converged": True}) == 0
    assert cli._summary_exit_code({"solver_mode": "transient", "converged": False}) == 0


@pytest.mark.parametrize("name", ("hartmann", "shercliff", "hunt"))
def test_fully_developed_cli_resolution_reaches_case(name):
    case = cli._build_case(SimpleNamespace(case=name, ha=5.0, output=None, ny=5, nz=7))
    assert (case.geometry.ny, case.geometry.nz) == (5, 7)
    assert case.reference_phi_cell == ((10, 11) if name == "hunt" else (2, 3))


def test_cli_case_builders_reject_unknown_case():
    with pytest.raises(ValueError):
        cli._build_case(SimpleNamespace(case="unknown", ha=5.0, output="out"))


def test_python_module_entrypoint_delegates_to_cli_main(
    monkeypatch: pytest.MonkeyPatch,
):
    recorded: dict[str, object] = {}

    monkeypatch.setattr("lmhdx.cli.main", lambda argv=None: recorded.update(argv=argv) or 0)
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
