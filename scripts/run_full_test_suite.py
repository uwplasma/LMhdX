#!/usr/bin/env python3
"""Run all or selected LMhdX tests in parallel within a declared budget.

A tier selects evidence markers and a shard selects files; combining them runs
one evidence tier over one file group, which keeps each pull-request job inside
the plan's wall-clock target without dropping any test from the matrix.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

_TEST_TIERS = {
    "unit": "unit and not (regression or slow or gpu or external)",
    "regression": "regression and not (slow or gpu or external)",
    "slow": "slow and not (gpu or external)",
    "gpu": "gpu",
    "external": "external",
}
_TEST_SHARDS = {
    "support": (
        "tests/test_cli.py",
        "tests/test_config.py",
        "tests/test_io.py",
        "tests/test_mesh.py",
        "tests/test_runtime_logging.py",
        "tests/test_units_and_wall_models.py",
        "tests/test_freemhd.py",
        "tests/test_benchmarks.py",
        "tests/test_example_runner.py",
        "tests/test_design.py",
    ),
    "operators": (
        "tests/test_advect.py",
        "tests/test_grid.py",
        "tests/test_ops.py",
        "tests/test_timeloop.py",
        "tests/test_staggered_laplacian.py",
        "tests/test_poisson.py",
        "tests/test_em.py",
        "tests/test_momentum_placement_oracle.py",
    ),
    "channel": ("tests/test_core3d.py", "tests/test_pipe.py", "tests/test_axial.py"),
    "plane": (
        *(
            f"tests/test_core3d.py::{name}"
            for name in (
                "test_implicit_and_explicit_viscosity_reach_the_same_steady_state",
                "test_the_plane_channel_converges_at_second_order",
                "test_plane_channel_matches_the_analytic_parabola",
                "test_a_transverse_field_reduces_the_channel_throughput",
            )
        ),
        "tests/test_coreflow.py",
    ),
    "steady": ("tests/test_steady.py",),
    # The shared-program tests of 2b.1, split from steady to keep both inside their budgets.
    "programs": tuple(
        f"tests/test_steady.py::{name}"
        for name in (
            "test_the_shared_program_holds_no_value_of_the_problem",
            "test_a_new_field_on_the_same_mesh_reuses_the_compiled_solve",
            "test_a_new_mesh_of_a_known_shape_is_solved_without_a_trace",
            "test_a_known_shape_is_solved_in_a_new_process_without_a_trace",
            "test_a_program_stored_on_another_cpu_is_never_loaded",
        )
    ),
    "fully_developed": ("tests/test_fully_developed.py",),
    "gradients": tuple(
        f"tests/test_steady.py::{name}"
        for name in (
            "test_the_adjoint_matches_finite_differences",
            "test_the_reused_primal_is_the_general_adjoint",
            "test_the_adjoint_matches_finite_differences_where_the_layers_are_thin",
            "test_the_adjoint_matches_finite_differences_in_a_varying_field",
            "test_a_rejected_root_cannot_produce_a_finite_objective_or_gradient",
            "test_a_failed_tangent_solve_is_rejected_eagerly_and_under_jit",
        )
    ),
    "physics": (
        "tests/test_physics.py",
        "tests/test_q2d_identities.py",
        "tests/test_solver.py",
    ),
}

_ALL_TESTS = tuple(dict.fromkeys(path.split("::")[0] for shard in _TEST_SHARDS.values() for path in shard))
_CHANGE_TEST_NAMES = {
    "__init__": "config cli example_runner",
    "__main__": "cli",
    "cases": "config solver physics fully_developed",
    "cli": "cli example_runner",
    "advect": "advect core3d",
    "axial": "axial example_runner",
    "core3d": "core3d timeloop advect steady coreflow axial",
    "coreflow": "coreflow freemhd",
    "design": "design fully_developed",
    "fully_developed": "fully_developed design config cli solver example_runner",
    "steady": "steady fully_developed axial",
    "bc": "ops staggered_laplacian core3d timeloop advect steady pipe axial",
    "grid": "grid ops staggered_laplacian core3d timeloop advect steady pipe",
    "timeloop": "timeloop",
    "ops": "ops staggered_laplacian core3d timeloop advect steady pipe axial",
    "em": "em",
    "poisson": "poisson core3d timeloop steady pipe axial",
    "pipe": "pipe",
    "io": "io cli example_runner",
    "mesh": "mesh solver physics",
    "physics": "solver physics",
    "q2d": "physics q2d_identities example_runner",
    "solvers": "solver physics",
    "specs": "config solver physics cli",
    "validation": "benchmarks physics solver example_runner cli",
}
_CHANGE_TESTS = {
    module: tuple(f"tests/test_{name}.py" for name in names.split())
    for module, names in _CHANGE_TEST_NAMES.items()
}
_NO_PYTHON_TEST_PREFIXES = ("docs/", ".github/")
_NO_PYTHON_TEST_FILES = {
    ".gitignore",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "ROADMAP.md",
    "plan.md",
}


def _changed_files(base: str) -> tuple[str, ...]:
    """Return committed, working-tree, and untracked paths relative to ``base``."""

    commands = (
        ("git", "diff", "--name-only", "--diff-filter=ACMRTUXB", f"{base}...HEAD"),
        ("git", "diff", "--name-only", "--diff-filter=ACMRTUXB", "HEAD"),
        ("git", "ls-files", "--others", "--exclude-standard"),
    )
    paths = []
    for command in commands:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        paths.extend(completed.stdout.splitlines())
    return tuple(dict.fromkeys(paths))


def _tests_for_changes(paths: tuple[str, ...]) -> tuple[str, ...]:
    """Select a conservative test set; unknown executable paths fail closed to all tests."""

    selected = []
    for path in paths:
        if path.startswith("tests/") and path.endswith(".py"):
            selected.append(path)
        elif path.startswith("src/lmhdx/data/benchmarks/"):
            selected.append("tests/test_freemhd.py")
        elif path.startswith("src/lmhdx/") and path.endswith(".py"):
            module = path.removeprefix("src/lmhdx/").removesuffix(".py")
            if module in _CHANGE_TESTS:
                selected.extend(_CHANGE_TESTS[module])
            else:
                return _ALL_TESTS
        elif path.startswith("examples/"):
            selected.append("tests/test_example_runner.py")
        elif path == "scripts/run_benchmarks.py":
            selected.append("tests/test_benchmarks.py")
        elif path == "validation/freemhd.py":
            selected.append("tests/test_freemhd.py")
        elif path in {
            "scripts/audit_architecture.py",
            "scripts/run_full_test_suite.py",
            "scripts/make_showcase_figures.py",
        }:
            selected.append("tests/test_config.py")
        elif path in {"pyproject.toml", "MANIFEST.in"}:
            selected.append("tests/test_config.py")
        elif path in _NO_PYTHON_TEST_FILES or path.startswith(_NO_PYTHON_TEST_PREFIXES):
            continue
        else:
            return _ALL_TESTS
    return tuple(dict.fromkeys(selected))


def _test_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source, environment.get("PYTHONPATH")) if value
    )
    return environment


def _shard_selection(shards: tuple[str, ...]) -> list[str]:
    """Return the pytest arguments selecting the union of ``shards``.

    A shard may claim single test functions (``file::test``) of a file another shard
    owns. The owner deselects them, so a test added to that file later runs with it.
    """
    entries = list(dict.fromkeys(entry for shard in shards for entry in _TEST_SHARDS[shard]))
    files = {entry for entry in entries if "::" not in entry}
    selection = [entry for entry in entries if entry.split("::", 1)[0] not in files or "::" not in entry]
    for shard, others in _TEST_SHARDS.items():
        if shard not in shards:
            for entry in others:
                if "::" in entry and entry.split("::", 1)[0] in files:
                    selection.extend(("--deselect", entry))
    return selection


def _default_workers() -> int:
    return max(1, min(6, os.cpu_count() or 1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--budget-seconds", type=float, default=600.0)
    parser.add_argument("--warning-seconds", type=float, default=300.0)
    parser.add_argument("--test-timeout-seconds", type=float)
    parser.add_argument("--no-coverage", action="store_true")
    parser.add_argument(
        "--changed-from",
        metavar="GIT_REF",
        help="run only tests affected since a Git ref; implies --no-coverage",
    )
    parser.add_argument("--no-compilation-cache", action="store_true")
    parser.add_argument("--shard", action="append", choices=tuple(_TEST_SHARDS), help="repeatable")
    parser.add_argument("--tier", choices=tuple(_TEST_TIERS))
    parser.add_argument("--coverage-fail-under", type=float, default=95.0)
    parser.add_argument("--coverage-xml", default="coverage.xml")
    parser.add_argument("--junit-xml", default="artifacts/tests/full-suite-junit.xml")
    parser.add_argument("tests", nargs="*", help="Test paths to run; defaults to the complete suite")
    args = parser.parse_args(argv)

    workers = args.workers
    if workers is None:
        workers = 1 if args.shard == ["support"] else _default_workers()
    if workers < 1:
        parser.error("--workers must be positive")
    if args.budget_seconds <= 0.0:
        parser.error("--budget-seconds must be positive")
    if args.warning_seconds <= 0.0:
        parser.error("--warning-seconds must be positive")
    if args.test_timeout_seconds is not None and args.test_timeout_seconds <= 0.0:
        parser.error("--test-timeout-seconds must be positive")
    if not 0.0 <= args.coverage_fail_under <= 100.0:
        parser.error("--coverage-fail-under must be between 0 and 100")
    if args.shard and args.tests:
        parser.error("--shard cannot be combined with explicit test paths")
    if args.changed_from and (args.shard or args.tests):
        parser.error("--changed-from cannot be combined with --shard or explicit test paths")

    selected_tests = (
        _tests_for_changes(_changed_files(args.changed_from)) if args.changed_from else args.tests
    )
    if args.changed_from and not selected_tests:
        print(f"LMhdX change gate: no Python tests affected since {args.changed_from}")
        return 0

    junit_path = os.path.abspath(args.junit_xml)
    os.makedirs(os.path.dirname(junit_path), exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        str(workers),
        "--dist",
        "worksteal",
        "--durations=20",
        f"--junitxml={junit_path}",
    ]
    if args.test_timeout_seconds is not None:
        command.append(f"--timeout={args.test_timeout_seconds:g}")
    coverage = not args.no_coverage and not args.changed_from
    if coverage:
        command.extend(
            [
                "--cov=src/lmhdx",
                "--cov-branch",
                "--cov-report=term-missing:skip-covered",
                f"--cov-report=xml:{args.coverage_xml}",
                f"--cov-fail-under={args.coverage_fail_under}",
            ]
        )
    if args.tier:
        command.extend(("-m", _TEST_TIERS[args.tier]))
    elif not args.shard and not selected_tests:
        command.extend(("-m", "not curated"))
    command.extend(_shard_selection(tuple(args.shard)) if args.shard else selected_tests or ["tests"])

    environment = _test_environment()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("JAX_ENABLE_X64", "true")
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("NUMEXPR_NUM_THREADS", "1")
    if args.no_compilation_cache:
        environment["JAX_ENABLE_COMPILATION_CACHE"] = "false"
        environment.pop("JAX_COMPILATION_CACHE_DIR", None)
    else:
        environment.setdefault(
            "JAX_COMPILATION_CACHE_DIR",
            os.path.join(tempfile.gettempdir(), "lmx-jax-cache"),
        )

    started = time.monotonic()
    print(
        f"LMhdX full test gate: workers={workers}, budget={args.budget_seconds:.0f}s, coverage={coverage}",
        flush=True,
    )
    if args.changed_from:
        print(f"LMhdX change gate selected: {', '.join(selected_tests)}", flush=True)
    try:
        completed = subprocess.run(
            command,
            env=environment,
            check=False,
            timeout=args.budget_seconds,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        print(
            f"LMhdX full test gate exceeded its {args.budget_seconds:.0f}s budget after {elapsed:.1f}s",
            file=sys.stderr,
        )
        return 124

    elapsed = time.monotonic() - started
    print(f"LMhdX full test gate completed in {elapsed:.1f}s", flush=True)
    if elapsed > args.warning_seconds:
        print(
            f"LMhdX full test gate exceeded its {args.warning_seconds:.0f}s warning budget",
            file=sys.stderr,
        )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
