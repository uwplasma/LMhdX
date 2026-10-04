import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group(name="examples")]


@pytest.mark.curated
def test_portable_duct_tutorials_and_toml_first_run(tmp_path: Path):
    examples = Path(__file__).resolve().parents[1] / "examples"
    for script_name in ("hartmann_example.py", "hunt_example.py"):
        subprocess.run([sys.executable, examples / script_name], cwd=tmp_path, timeout=30, check=True)

    hartmann = json.loads(next((tmp_path / "artifacts").rglob("hartmann_summary.json")).read_text())
    hunt = json.loads(next((tmp_path / "artifacts").rglob("hunt_summary.json")).read_text())
    assert hartmann["analytical_profile"]["l2_error"] < 0.05
    # The Hartmann walls are thin walls of the core, so there is no fluid/wall
    # interface to report; charge conservation is checked on every cell instead.
    assert hunt["validation"]["interface_current_residual"] is None
    assert hunt["validation"]["div_current_max"] < 1.0e-8
    design = hunt["design"]
    assert design["verification"] == "cold certified steady solve"
    assert design["status"] == "converged"
    # ||R(u)|| / ||R(0)|| of the one CG solve, which is asked for 1e-9.
    assert design["steady_residual"] <= 1e-9
    assert design["flow_rate"] == pytest.approx(design["target_flow_rate"], rel=1e-8)
    assert design["relative_flow_error"] < 1e-8
    assert design["drive_derivative_wrt_flow"] == pytest.approx(1 / design["flow_per_unit_drive"])
    assert design["hydraulic_power"] == pytest.approx(
        design["drive"] * design["length"] * design["flow_rate"]
    )

    source = Path(__file__).resolve().parents[1] / "examples/hartmann_case.toml"
    case_path = tmp_path / source.name
    case_path.write_text(
        source.read_text().replace("../artifacts/examples/toml_hartmann", "artifacts/toml_hartmann")
    )
    subprocess.run([sys.executable, "-m", "lmhdx", case_path], cwd=tmp_path, timeout=30, check=True)
    summary = json.loads(next((tmp_path / "artifacts").rglob("hartmann_ha20_toml_summary.json")).read_text())
    assert summary["converged"] is True
    assert summary["status"] == "converged"
    assert summary["residual"] < 1.0e-8


def test_li_aln_wall_stack_example_runs_explicit_models(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "examples/li_aln_wall_stack_example.py"
    subprocess.run([sys.executable, script], cwd=tmp_path, timeout=60, check=True)
    summary_path = next((tmp_path / "artifacts").rglob("li_aln_wall_stack_summary.json"))
    summary = json.loads(summary_path.read_text())

    assert summary["inductionless_assumption_pass"] is True
    assert set(summary["models"]) == {"intact_aln", "bare_metal"}
    assert (
        summary["models"]["bare_metal"]["tangential_conductance_ratio"]
        > summary["models"]["intact_aln"]["tangential_conductance_ratio"]
    )
    assert all(
        model["validation"]["relative_residual"] < 1.0e-8
        and model["validation"]["charge_balance_relative"] < 1.0e-8
        for model in summary["models"].values()
    )
    # The intact coating insulates the walls; bare 316L has c Ha = 2.3, so about three times the gradient.
    models = summary["models"]
    assert (
        models["intact_aln"]["pressure_gradient_pa_m"] < 0.5 * models["bare_metal"]["pressure_gradient_pa_m"]
    )
    assert (summary_path.parent / "li_aln_wall_stack.png").is_file()


def test_fringe_duct_example_conserves_and_checks_its_gradient(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "examples/fringe_duct_example.py"
    subprocess.run([sys.executable, script], cwd=tmp_path, timeout=180, check=True)
    summary = json.loads(next((tmp_path / "artifacts").rglob("fringe_duct_summary.json")).read_text())

    checks = summary["checks"]
    assert checks["relative_residual"] <= 1e-9
    assert max(checks["relative_flow_rate_error"], checks["mass_balance"], checks["charge_balance"]) < 1e-12
    assert summary["upstream_gradient_relative_error"] < 5e-3
    assert summary["fringe_pressure_drop"] > 0.0
    assert summary["derivative_relative_error"] < 1e-6
