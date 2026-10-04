from io import StringIO
from pathlib import Path

import jax.numpy as jnp
import pytest

from lmhdx.cases import (
    Diagnostics,
    LoggingSpec,
    MHDState,
    RestartLogInfo,
    Solution,
    SolverStepRecord,
    StreamingSolverLogger,
    default_log_path,
    make_hartmann_case,
)
from lmhdx.fully_developed import case_mesh as _build_mesh

pytestmark = pytest.mark.unit


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
        potential_solver=case.time_stepper.potential_solver,
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
        potential_solver=case.time_stepper.potential_solver,
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
    assert "linear=solvax_pcg" in text
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
