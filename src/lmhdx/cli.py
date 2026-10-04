from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp

from .cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from .fully_developed import case_mesh, solve_fully_developed, solve_fully_developed_transient
from .io import (
    _portable_path,
    load_restart_bundle,
    validate_restart_bundle,
    write_paraview,
    write_restart_npz,
    write_solution_outputs,
)
from .specs import (
    RestartLogInfo,
    RunConfig,
    StreamingSolverLogger,
    default_log_path,
    load_run_config,
)
from .validation import (
    benchmark_solver,
    extract_centerline,
    extract_midplane_profile,
    hartmann_acceptance,
    hartmann_validation,
    validation_summary,
    write_acceptance_report,
    write_analytic_comparison,
    write_benchmark_report,
    write_metrics_json,
    write_profile_csv,
)


def _build_case(args: argparse.Namespace):
    builders = {"hartmann": make_hartmann_case, "shercliff": make_shercliff_case, "hunt": make_hunt_case}
    if args.case not in builders:
        raise ValueError(args.case)
    geometry = {key: getattr(args, key) for key in ("width", "height", "ny", "nz") if hasattr(args, key)}
    return builders[args.case](ha=args.ha, output_dir=args.output, **geometry)


def _solve_case_with_optional_logger(
    case,
    *,
    solve_mode: str,
    logger=None,
    initial_state=None,
    initial_diagnostics=None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
):
    if solve_mode == "transient":
        return solve_fully_developed_transient(
            case,
            logger=logger,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=append_diagnostics,
            restart_info=restart_info,
        )
    start_time = 0.0 if initial_state is None else float(initial_state.time)
    return solve_fully_developed(case, logger=logger, start_time=start_time)


def _runtime_summary(
    solution,
    case,
    out_dir: Path,
    outputs: dict[str, list[Path]],
    *,
    restart_info: dict[str, object] | None = None,
) -> dict[str, object]:
    diag = getattr(solution, "diagnostics", None)
    geometry = getattr(getattr(case, "geometry", None), "kind", "unknown")
    u_field = getattr(solution.state, "u", jnp.asarray([0.0]))

    def _latest(name: str) -> float | None:
        history = getattr(diag, name, jnp.asarray([]))
        return float(history[-1]) if getattr(history, "size", 0) else None

    summary = {
        "case": case.name,
        "geometry": geometry,
        "solver_kind": getattr(getattr(case, "solver", None), "kind", "fully_developed_inductionless"),
        "solver_mode": getattr(getattr(case, "solver", None), "mode", "steady"),
        "converged": getattr(solution, "converged", None),
        "status": getattr(solution, "status", "not_recorded"),
        "steps": int(getattr(solution, "steps", 0)),
        "time": float(solution.state.time),
        "residual": float(solution.state.residual),
        "u_max": float(jnp.max(jnp.abs(u_field))),
        "potential_residual": _latest("potential_residual_history"),
        "potential_iterations_used": _latest("potential_iterations_history"),
        "linear_residual": _latest("linear_residual_history"),
        "linear_iterations_used": _latest("linear_iterations_history"),
        "volumetric_flow_rate": _latest("volumetric_flow_rate_history"),
        "mean_current_magnitude": _latest("mean_current_magnitude_history"),
        "lorentz_power": _latest("lorentz_power_history"),
        "div_current_max": _latest("div_current_max_history"),
        "charge_balance_residual": _latest("charge_balance_residual_history"),
        "gauge_residual": _latest("gauge_residual_history"),
        "interface_current_residual": _latest("interface_current_residual_history"),
        "output": _portable_path(out_dir),
        "generated_files": {
            key: [_portable_path(path) for path in paths] for key, paths in outputs.items() if paths
        },
    }
    if restart_info is not None:
        summary["restart"] = restart_info
    return summary


def _write_run_summary(summary: dict[str, object], case, out_dir: Path) -> Path | None:
    if not getattr(case.output, "write_json_summary", True):
        return None
    path = out_dir / f"{case.name}_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path


def _summary_exit_code(summary: dict[str, object]) -> int:
    """Return failure only for a recorded unconverged steady result."""

    return 2 if summary.get("solver_mode") == "steady" and summary.get("converged") is False else 0


def _run_config(config: RunConfig) -> dict[str, object]:
    case = config.case
    output_dir = getattr(case, "output_dir", None)
    if output_dir is None and getattr(case, "output", None) is not None:
        output_dir = getattr(case.output, "directory", None)
    out_dir = Path(output_dir) if output_dir else Path.cwd() / "out" / case.name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = StreamingSolverLogger(config.logging) if config.logging.enabled else None
    log_handle = None
    log_path: Path | None = None
    if logger is not None:
        log_path = default_log_path(out_dir, case.name)
        log_handle = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        logger.add_stream(log_handle)
    solve_start = time.perf_counter()
    initial_state = None
    initial_diagnostics = None
    restart_log_info = RestartLogInfo(enabled=False)
    restart_summary: dict[str, object] | None = None
    if config.restart.enabled:
        if config.restart.path is None:
            raise ValueError("Restart is enabled but no restart.path was provided")
        restart_bundle = load_restart_bundle(config.restart.path)
        validate_restart_bundle(
            restart_bundle,
            mesh=case_mesh(case),
            geometry_kind=case.geometry.kind,
            case_name=case.name,
        )
        initial_state = restart_bundle.state
        initial_diagnostics = restart_bundle.diagnostics
        restart_log_info = RestartLogInfo(
            enabled=True,
            path=str(restart_bundle.path),
            start_time=float(restart_bundle.state.time),
            reset_histories=config.restart.reset_histories,
        )
        restart_summary = {
            "enabled": True,
            "input": str(restart_bundle.path),
            "start_time": float(restart_bundle.state.time),
            "reset_histories": bool(config.restart.reset_histories),
        }
    try:
        solution = _solve_case_with_optional_logger(
            case,
            solve_mode=case.solver.mode,
            logger=logger,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=bool(config.restart.enabled and not config.restart.reset_histories),
            restart_info=restart_log_info,
        )
    finally:
        if log_handle is not None:
            log_handle.close()
    outputs = write_solution_outputs(
        solution,
        case,
        out_dir,
        write_npz=getattr(case.output, "write_npz", True),
        write_plots=getattr(case.output, "write_plots", False),
    )
    if config.restart.write_restart:
        restart_filename = config.restart.restart_filename or f"{case.name}_restart.npz"
        restart_path = write_restart_npz(solution, case, out_dir / restart_filename)
        outputs.setdefault("restart", []).append(restart_path)
        if restart_summary is None:
            restart_summary = {"enabled": False}
        restart_summary["output"] = _portable_path(restart_path)
    if log_path is not None:
        outputs.setdefault("log", []).append(log_path)
    if config.input_path is not None and getattr(case.output, "copy_input_file", True):
        copied_input = out_dir / config.input_path.name
        shutil.copy2(config.input_path, copied_input)
        outputs.setdefault("input", []).append(copied_input)
    summary = _runtime_summary(solution, case, out_dir, outputs, restart_info=restart_summary)
    summary["execution_seconds"] = time.perf_counter() - solve_start
    summary_path = _write_run_summary(summary, case, out_dir)
    if summary_path is not None:
        outputs.setdefault("json", []).append(summary_path)
        summary["generated_files"]["json"] = [_portable_path(summary_path)]
    print(json.dumps(summary, indent=2))
    return summary


def run_from_toml(path: str | Path) -> dict[str, object]:
    config = load_run_config(path)
    return _run_config(config)


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if argv and Path(argv[0]).suffix == ".toml":
        return _summary_exit_code(run_from_toml(argv[0]))

    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        prog="lmhdx",
        description="Run and validate differentiable inductionless MHD cases.",
        epilog="A TOML case may also be passed directly: lmhdx CASE.toml",
        formatter_class=formatter,
    )
    subparsers = parser.add_subparsers(dest="command", title="commands", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a named built-in case.",
        description="Run a fully developed duct case.",
        formatter_class=formatter,
    )
    run_parser.add_argument(
        "case",
        choices=[
            "hartmann",
            "shercliff",
            "hunt",
        ],
        help="Built-in case or solver family.",
    )
    run_parser.add_argument("--ha", type=float, default=20.0, help="Hartmann number.")
    run_parser.add_argument("--output", default="./out", help="Output directory.")
    run_parser.add_argument(
        "--mode",
        choices=["steady", "transient"],
        default="steady",
        help="Solve mode for fully developed cases.",
    )
    run_parser.add_argument("--plots", action="store_true", help="Write summary plots.")
    run_parser.add_argument("--quiet", action="store_true", help="Disable solver logging.")
    geometry = run_parser.add_argument_group("geometry and resolution")
    geometry.add_argument("--width", type=float, default=2.0, help="Duct width.")
    geometry.add_argument("--height", type=float, default=2.0, help="Duct height.")
    geometry.add_argument("--ny", type=int, default=48, help="Cross-stream y cells.")
    geometry.add_argument("--nz", type=int, default=48, help="Cross-stream z cells.")

    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Time a bounded Hartmann solve.",
        description="Measure cold and warm runtime for a portable Hartmann case.",
        formatter_class=formatter,
    )
    bench_parser.add_argument("--repeats", type=int, default=3, help="Timed repetitions.")
    bench_parser.add_argument("--ha", type=float, default=20.0, help="Hartmann number.")
    bench_parser.add_argument("--ny", type=int, default=48, help="Mesh y cells.")
    bench_parser.add_argument("--nz", type=int, default=48, help="Mesh z cells.")
    bench_parser.add_argument("--output", default="", help="Optional JSON report path.")

    validate_parser = subparsers.add_parser(
        "validate",
        help="Solve a duct case and write validation metrics.",
        description="Solve a duct case and write profiles, diagnostics, and Hartmann acceptance metrics.",
        formatter_class=formatter,
    )
    validate_parser.add_argument("case", choices=["hartmann", "shercliff", "hunt"], help="Validation case.")
    validate_parser.add_argument("--ha", type=float, default=20.0, help="Hartmann number.")
    validate_parser.add_argument("--output", default="./out", help="Output directory.")
    validate_parser.add_argument(
        "--hartmann-l2-threshold", type=float, default=0.05, help="Hartmann L2 gate."
    )
    validate_parser.add_argument(
        "--hartmann-linf-threshold", type=float, default=0.1, help="Hartmann Linf gate."
    )

    args = parser.parse_args(argv)

    if args.command == "benchmark":
        payload = benchmark_solver(repeats=args.repeats, ha=args.ha, ny=args.ny, nz=args.nz)
        if args.output:
            write_benchmark_report(payload, args.output)
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "validate":
        case = _build_case(args)
        solution = _solve_case_with_optional_logger(case, solve_mode="steady", logger=None)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_paraview(solution, out_dir)
        write_profile_csv(out_dir / f"{case.name}_centerline.csv", extract_centerline(solution))
        z_profile = extract_midplane_profile(solution, axis="z")
        write_profile_csv(out_dir / f"{case.name}_midplane_z.csv", z_profile)
        payload = validation_summary(solution, case.name, ha=args.ha)
        payload.update(
            converged=getattr(solution, "converged", None),
            status=getattr(solution, "status", "not_recorded"),
            steps=int(getattr(solution, "steps", 0)),
        )
        if args.case == "hartmann":
            comparison = hartmann_validation(solution, args.ha)
            write_analytic_comparison(comparison, out_dir / f"{case.name}_analytic.json", axis_name="y")
            acceptance = hartmann_acceptance(
                solution,
                args.ha,
                l2_threshold=args.hartmann_l2_threshold,
                linf_threshold=args.hartmann_linf_threshold,
            )
            write_acceptance_report(acceptance, out_dir / f"{case.name}_acceptance.json")
            payload["accepted"] = float(acceptance.passed)
            payload["acceptance_l2_threshold"] = acceptance.l2_threshold
            payload["acceptance_linf_threshold"] = acceptance.linf_threshold
        write_metrics_json(payload, out_dir / f"{case.name}_metrics.json")
        print(json.dumps(payload, indent=2))
        return 2 if getattr(solution, "converged", None) is False else 0

    case = _build_case(args)
    case = replace(
        case,
        solver=replace(case.solver, mode=args.mode),
        output=replace(
            case.output,
            directory=args.output,
            write_npz=True,
            write_json_summary=True,
            write_plots=args.plots,
        ),
    )
    config = RunConfig(case=case)
    logging = config.logging
    if args.quiet:
        logging = replace(logging, enabled=False)
    if logging is not config.logging:
        config = replace(config, logging=logging)
    return _summary_exit_code(_run_config(config))
