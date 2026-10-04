#!/usr/bin/env python3
"""Inventory the LMhdX architecture and enforce the M2 slimming baseline."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path("src/lmhdx")
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "_build",
}
EXCLUDED_FILES = {".coverage", "coverage.xml"}
CURRENT_STATE_TERMS = ("legacy", "deprecated", "previously", "no longer", "backward compat")

RESEARCH_STAGE: set[str] = set()
COMPATIBILITY: set[str] = set()
VISUALIZATION = {"io.py"}
VALIDATION = {
    "validation.py",
}


def _role(name: str) -> str:
    if name in COMPATIBILITY:
        return "compatibility_facade"
    if name in RESEARCH_STAGE:
        return "research_stage_extension"
    if name in VISUALIZATION:
        return "visualization"
    if name in VALIDATION:
        return "validation_and_evidence"
    return "stable_core"


def _root_exports(root: Path) -> list[str]:
    tree = ast.parse((root / PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            exports = ast.literal_eval(node.value)
            if not isinstance(exports, list) or not all(isinstance(item, str) for item in exports):
                break
            return exports
    raise ValueError("lmhdx.__all__ must be a literal list for architecture auditing")


def _repository_files(root: Path) -> list[Path] | None:
    """Return tracked and untracked Git files, or ``None`` outside a worktree."""

    try:
        completed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return [
        root / item.decode()
        for item in completed.stdout.split(b"\0")
        if item and (root / item.decode()).is_file()
    ]


def _checkout_size(root: Path) -> int:
    """Measure tracked checkout bytes without counting local build products."""

    total = 0
    repository_files = _repository_files(root)
    paths = repository_files if repository_files is not None else root.rglob("*")
    for path in paths:
        relative = path.relative_to(root)
        if not path.is_file() or path.name in EXCLUDED_FILES or path.name.startswith(".coverage."):
            continue
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        total += path.stat().st_size
    return total


def _current_state_violations(root: Path) -> list[str]:
    paths = [root / name for name in ("README.md", "CONTRIBUTING.md", "CITATION.cff")]
    for directory, patterns in (
        ("src/lmhdx", ("*.py",)),
        ("validation", ("*.py",)),
        ("docs", ("*.md", "*.py")),
        ("examples", ("*.py", "*.toml")),
        ("scripts", ("*.py",)),
    ):
        paths.extend(path for pattern in patterns for path in (root / directory).rglob(pattern))
    violations = []
    for path in paths:
        if not path.is_file() or path == Path(__file__) or "src/lmhdx/data" in path.as_posix():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(term in line.casefold() for term in CURRENT_STATE_TERMS):
                violations.append(f"{path.relative_to(root)}:{line_number}")
    return violations


def _release_asset_candidates(root: Path) -> list[dict[str, Any]]:
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size <= 128 * 1024:
            continue
        relative = path.relative_to(root)
        generated_doc = relative.parts[:3] == ("docs", "_static", "generated")
        if not generated_doc:
            continue
        candidates.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "disposition": "github_or_zenodo_release",
            }
        )
    return sorted(candidates, key=lambda item: item["path"])


def build_inventory(root: Path = ROOT) -> dict[str, Any]:
    modules = []
    for path in sorted((root / PACKAGE_ROOT).glob("*.py")):
        text = path.read_text(encoding="utf-8")
        role = _role(path.name)
        modules.append(
            {
                "path": path.relative_to(root).as_posix(),
                "lines": len(text.splitlines()),
                "bytes": path.stat().st_size,
                "role": role,
                "owner": role,
            }
        )
    test_files = sorted((root / "tests").glob("test_*.py"))
    external_validation = sorted((root / "validation").glob("*.py"))
    maintenance_scripts = sorted((root / "scripts").glob("*.py"))
    examples = sorted(
        path.relative_to(root).as_posix()
        for pattern in ("*.py", "*.toml")
        for path in (root / "examples").glob(pattern)
        if path.is_file() and path.name != "catalog.toml"
    )
    catalog_path = root / "examples" / "catalog.toml"
    catalog = tomllib.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 2:
        raise ValueError("examples/catalog.toml must use schema_version = 2")
    curated = catalog.get("example", [])
    if not isinstance(curated, list) or not curated:
        raise ValueError("examples/catalog.toml must define at least one [[example]]")
    curated_paths = [str(item["path"]) for item in curated]
    allowed_example_statuses = {"stable", "research-stage"}
    allowed_runtime_tiers = {"portable", "accelerator-optional"}
    for item in curated:
        if item.get("status") not in allowed_example_statuses:
            raise ValueError(f"Unknown curated workflow status: {item.get('status')!r}")
        if item.get("runtime") not in allowed_runtime_tiers:
            raise ValueError(f"Unknown curated workflow runtime: {item.get('runtime')!r}")
        if not item.get("command") or not item.get("outputs") or not item.get("docs"):
            raise ValueError("Curated workflows need command, outputs, and docs")
        if not (root / str(item["docs"])).is_file():
            raise ValueError(f"Curated workflow docs do not exist: {item['docs']}")
    missing_curated = [path for path in curated_paths if not (root / path).is_file()]
    if missing_curated:
        raise ValueError(f"Curated examples do not exist: {missing_curated}")
    unclassified = sorted(set(examples) - set(curated_paths))
    stale = sorted(set(curated_paths) - set(examples))
    if unclassified or stale:
        raise ValueError(f"Workflow catalog drift: unclassified={unclassified}, stale={stale}")
    exports = _root_exports(root)
    release_candidates = _release_asset_candidates(root)
    checkout_bytes = _checkout_size(root)
    prose_violations = _current_state_violations(root)
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    return {
        "schema_version": 1,
        "inventory": {
            "package_module_count": len(modules),
            "total_package_lines": sum(module["lines"] for module in modules),
            "largest_package_module_lines": max(module["lines"] for module in modules),
            "maintained_core_lines": sum(
                module["lines"]
                for module in modules
                if module["role"] in {"stable_core", "compatibility_facade"}
            ),
            "test_file_count": len(test_files),
            "test_lines": sum(len(path.read_text(encoding="utf-8").splitlines()) for path in test_files),
            "external_validation_file_count": len(external_validation),
            "largest_external_validation_lines": max(
                (len(path.read_text(encoding="utf-8").splitlines()) for path in external_validation),
                default=0,
            ),
            "maintenance_script_count": len(maintenance_scripts),
            "package_modules": modules,
            "root_export_count": len(exports),
            "root_exports": exports,
            "example_count": len(examples),
            "examples": examples,
            "curated_example_count": len(curated_paths),
            "curated_examples": curated,
            "uncurated_example_count": len(set(examples) - set(curated_paths)),
            "checkout_bytes_excluding_build_artifacts": checkout_bytes,
            "release_asset_candidate_bytes": sum(item["bytes"] for item in release_candidates),
            "checkout_bytes_excluding_release_candidates": checkout_bytes
            - sum(item["bytes"] for item in release_candidates),
            "release_asset_candidates": release_candidates,
            "current_state_prose_violations": prose_violations,
            "dependencies": {
                "runtime": sorted(project.get("dependencies", [])),
                "optional": {
                    key: sorted(value)
                    for key, value in sorted(project.get("optional-dependencies", {}).items())
                },
            },
        },
        "targets": {
            "package_module_count_max": 25,
            "total_package_lines_max": 11129,
            "largest_package_module_lines_max": 1346,
            "maintained_core_lines_max": 10200,
            "test_file_count_max": 26,
            "test_lines_max": 9488,
            "external_validation_file_count_max": 3,
            "largest_external_validation_lines_max": 599,
            "maintenance_script_count_max": 4,
            "stable_root_exports_max": 33,
            "curated_examples_max": 6,
            "checkout_bytes_max": 2440100,
            "root_import_median_seconds_max": 0.25,
            "sdist_bytes_max": 512 * 1024,
            "wheel_bytes_max": 384 * 1024,
        },
    }


def measure_import(root: Path = ROOT, repeats: int = 5) -> dict[str, Any]:
    samples = []
    command = [
        sys.executable,
        "-c",
        "import sys, lmhdx; print(int('jax' in sys.modules))",
    ]
    jax_loaded = False
    for _ in range(repeats):
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=root, capture_output=True, check=False)
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            raise ValueError(completed.stderr.decode(errors="replace"))
        jax_loaded |= completed.stdout.strip() == b"1"
        samples.append(elapsed)
    return {
        "command": "python -c 'import lmhdx'",
        "repeats": repeats,
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "jax_loaded": jax_loaded,
    }


def inspect_wheel(path: str | Path) -> dict[str, Any]:
    """Return wheel size and flag files outside the package/metadata roots."""

    wheel = Path(path)
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
    forbidden = [name for name in members if not (name.startswith("lmhdx/") or ".dist-info/" in name)]
    return {
        "path": wheel.name,
        "bytes": wheel.stat().st_size,
        "member_count": len(members),
        "forbidden_members": forbidden,
        "typed_marker_present": "lmhdx/py.typed" in members,
    }


def inspect_sdist(path: str | Path) -> dict[str, Any]:
    """Return sdist size and flag files outside its reproducibility payload."""

    sdist = Path(path)
    with tarfile.open(sdist) as archive:
        members = [member.name for member in archive.getmembers() if member.isfile()]
    relative = [name.split("/", 1)[1] if "/" in name else name for name in members]
    allowed_files = {
        "LICENSE",
        "MANIFEST.in",
        "PKG-INFO",
        "README.md",
        "pyproject.toml",
        "setup.cfg",
    }
    allowed_roots = ("src/lmhdx/", "src/lmhdx.egg-info/")
    forbidden = [
        name for name in relative if name not in allowed_files and not name.startswith(allowed_roots)
    ]
    return {
        "path": sdist.name,
        "bytes": sdist.stat().st_size,
        "member_count": len(members),
        "forbidden_members": forbidden,
    }


def architecture_budget_errors(
    payload: dict[str, Any],
    *,
    wheel: str | Path | None = None,
    sdist: str | Path | None = None,
) -> list[str]:
    """Validate inventory, import timing, and optional distribution budgets."""

    inventory = payload["inventory"]
    targets = payload["targets"]
    checks = {
        "package_module_count": "package_module_count_max",
        "total_package_lines": "total_package_lines_max",
        "largest_package_module_lines": "largest_package_module_lines_max",
        "maintained_core_lines": "maintained_core_lines_max",
        "test_file_count": "test_file_count_max",
        "test_lines": "test_lines_max",
        "external_validation_file_count": "external_validation_file_count_max",
        "largest_external_validation_lines": "largest_external_validation_lines_max",
        "maintenance_script_count": "maintenance_script_count_max",
        "root_export_count": "stable_root_exports_max",
        "curated_example_count": "curated_examples_max",
        "checkout_bytes_excluding_build_artifacts": "checkout_bytes_max",
    }
    errors = [
        f"{value_key}={inventory[value_key]} exceeds {target_key}={targets[target_key]}"
        for value_key, target_key in checks.items()
        if inventory[value_key] > targets[target_key]
    ]
    if inventory["uncurated_example_count"]:
        errors.append("all examples must be listed in examples/catalog.toml")
    if inventory["release_asset_candidate_bytes"]:
        errors.append("generated release assets must not be tracked in Git")
    if inventory["current_state_prose_violations"]:
        errors.append(
            "user-facing files contain project-history language: "
            + ", ".join(inventory["current_state_prose_violations"])
        )
    measurement = payload.get("import_measurement")
    if measurement is not None:
        if measurement["median_seconds"] > targets["root_import_median_seconds_max"]:
            errors.append("root import exceeds its median wall-time budget")
        if measurement["jax_loaded"]:
            errors.append("import lmhdx must not eagerly import JAX")
    if wheel is not None:
        wheel_record = inspect_wheel(wheel)
        if wheel_record["bytes"] > targets["wheel_bytes_max"]:
            errors.append(
                f"wheel bytes={wheel_record['bytes']} exceeds wheel_bytes_max={targets['wheel_bytes_max']}"
            )
        if wheel_record["forbidden_members"]:
            errors.append(
                "wheel contains files outside lmhdx/ and dist-info/: "
                + ", ".join(wheel_record["forbidden_members"])
            )
        if not wheel_record["typed_marker_present"]:
            errors.append("wheel must contain the lmhdx/py.typed PEP 561 marker")
    if sdist is not None:
        sdist_record = inspect_sdist(sdist)
        if sdist_record["bytes"] > targets["sdist_bytes_max"]:
            errors.append(
                f"sdist bytes={sdist_record['bytes']} exceeds sdist_bytes_max={targets['sdist_bytes_max']}"
            )
        if sdist_record["forbidden_members"]:
            errors.append(
                "sdist contains files outside its source payload: "
                + ", ".join(sdist_record["forbidden_members"])
            )
    return errors


def write_inventory(output: Path, *, root: Path = ROOT, measure: bool = False) -> dict[str, Any]:
    payload = build_inventory(root)
    if measure:
        payload["import_measurement"] = measure_import(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="optionally write the current JSON inventory",
    )
    parser.add_argument("--measure-import", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    payload = build_inventory()
    if args.measure_import:
        payload["import_measurement"] = measure_import()
    if args.output is not None:
        write_inventory(args.output, measure=args.measure_import)
    errors = architecture_budget_errors(payload, wheel=args.wheel, sdist=args.sdist)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    inventory = payload["inventory"]
    print(
        f"modules={inventory['package_module_count']} "
        f"core_lines={inventory['maintained_core_lines']} "
        f"total_lines={inventory['total_package_lines']} "
        f"exports={inventory['root_export_count']} "
        f"curated_examples={inventory['curated_example_count']}"
    )
    if args.wheel is not None:
        wheel = inspect_wheel(args.wheel)
        print(f"wheel_bytes={wheel['bytes']} wheel_members={wheel['member_count']}")
    if args.sdist is not None:
        sdist = inspect_sdist(args.sdist)
        print(f"sdist_bytes={sdist['bytes']} sdist_members={sdist['member_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
