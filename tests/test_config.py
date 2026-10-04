import ast
import inspect
import json
import os
import re
import subprocess
import sys
import tarfile
import textwrap
import zipfile
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

import lmhdx
from lmhdx.cases import _wall_conductivity_from_conductance_ratio
from lmhdx.specs import _parse_boundary_value, load_run_config
from scripts.audit_architecture import (
    _checkout_size,
    architecture_budget_errors,
    build_inventory,
    inspect_sdist,
    inspect_wheel,
    measure_import,
)
from scripts.run_full_test_suite import _ALL_TESTS, _test_environment, _tests_for_changes

pytestmark = pytest.mark.unit


def test_ci_tiers_cover_collection_without_overlapping_pr_work():
    from scripts.run_full_test_suite import _TEST_TIERS

    def collect(expression):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "--collect-only",
                "-o",
                "addopts=",
                "-q",
                "-m",
                expression,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode in (0, 5), result.stdout + result.stderr
        return {line for line in result.stdout.splitlines() if line.startswith("tests/") and "::" in line}

    all_tests = collect("")
    tiers = {name: collect(expression) for name, expression in _TEST_TIERS.items()}
    assert all_tests and all_tests == set.union(*tiers.values())
    assert tiers["unit"] and tiers["regression"]
    assert not tiers["unit"] & tiers["regression"]
    deferred = tiers["slow"] | tiers["gpu"] | tiers["external"]
    assert not (tiers["unit"] | tiers["regression"]) & deferred

    workflow = Path(".github/workflows/ci.yml").read_text()
    for job in ("compatibility", "coverage"):
        # Read the job condition without relying on its step implementation.
        condition = workflow.split(f"\n  {job}:\n", 1)[1].split("    if: ", 1)[1].splitlines()[0]
        assert "github.event_name != 'pull_request'" in condition
    pr_job = re.split(r"\n  \w[\w-]*:\n", workflow.split("\n  pr-tests:\n", 1)[1])[0]
    assert "--no-coverage" in pr_job
    # Heavy evidence must keep a per-test timeout above the slowest recorded case.
    assert "--test-timeout-seconds 300" in pr_job

    entries = [
        (
            block.split("tier: ", 1)[1].splitlines()[0].strip(),
            block.split("shard: ", 1)[1].splitlines()[0].strip(' "'),
        )
        for block in pr_job.split("- tier: ")[1:]
        for block in ["tier: " + block]
    ]
    assert entries, pr_job
    covered: set[str] = set()
    for tier, shard in entries:
        selection = tiers[tier] if not shard else tiers[tier] & _shard_members(all_tests, shard)
        assert selection, f"PR job {tier}/{shard} selects nothing"
        assert not covered & selection, f"PR job {tier}/{shard} repeats work"
        covered |= selection
    assert covered == tiers["unit"] | tiers["regression"]


def _shard_members(all_tests: set[str], shards: str) -> set[str]:
    """Return the tests the runner selects for space-separated shards, with pytest's semantics.

    A path selects its file, ``file::test`` that function and its parametrizations,
    and ``--deselect`` removes every node id it prefixes.
    """
    from scripts.run_full_test_suite import _shard_selection

    arguments = _shard_selection(tuple(shards.split()))
    deselected = tuple(arguments[index + 1] for index, value in enumerate(arguments) if value == "--deselect")
    paths = [value for value in arguments if value != "--deselect" and value not in deselected]
    members = {
        test
        for test in all_tests
        for path in paths
        if test.split("::", 1)[0] == path or test == path or test.startswith(path + "[")
    }
    return {test for test in members if not test.startswith(deselected)}


@pytest.mark.parametrize("x64", ["false", "true"])
def test_explicit_precision_in_fresh_process(x64):
    code = """
import warnings
import jax
import jax.numpy as jnp
import lmhdx
from dataclasses import replace
from lmhdx import axial, cases, mesh, physics, q2d
initial = jax.config.x64_enabled
assert initial == EXPECTED
assert jax.config.jax_default_matmul_precision is None
case32 = lmhdx.make_hartmann_case(ha=2, ny=8, nz=8, dtype="float32")
assert jax.config.x64_enabled == initial
assert jax.config.jax_default_matmul_precision == "highest"
values = []
for dtype in ("float32", "float64"):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        case = lmhdx.make_hartmann_case(ha=2, ny=8, nz=8, dtype=dtype)
    assert len(recorded) == int(dtype == "float64" and not initial)
    if recorded:
        assert recorded[0].category is DeprecationWarning
    objective = lambda x: jnp.mean(lmhdx.solve_fully_developed_fields(case, forcing=x)[0])
    x = jnp.asarray(1., dtype=case.dtype)
    value, grad = jax.jit(jax.value_and_grad(objective))(x)
    tangent = jax.jit(lambda x: jax.jvp(objective, (x,), (jnp.ones_like(x),))[1])(x)
    assert value.dtype == grad.dtype == tangent.dtype == case.dtype
    assert bool(jnp.isfinite(grad))
    # Linear forcing response is an exact derivative oracle at fixed field.
    assert bool(jnp.allclose(grad, value, rtol=2e-5, atol=1e-7))
    assert bool(jnp.allclose(grad, tangent, rtol=2e-5, atol=1e-7))
    values.append(float(value))
    short = replace(case, time_stepper=replace(case.time_stepper, max_steps=2))
    result = lmhdx.solve(short)
    assert result.state.u.dtype == result.state.phi.dtype == case.dtype
assert abs(values[0] - values[1]) < 2e-6
lmhdx.enable_x64()
assert jnp.asarray(1.).dtype == jnp.float64
""".replace("EXPECTED", str(x64 == "true"))
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        timeout=90,
        env={
            **{key: value for key, value in os.environ.items() if key != "JAX_DEFAULT_MATMUL_PRECISION"},
            "JAX_ENABLE_X64": x64,
        },
    )


@pytest.mark.parametrize("choice", ["0", "cache"])
def test_first_case_enables_the_shared_disk_cache_by_default(tmp_path, choice):
    code = """
import os, jax, jaxlib, platform, lmhdx
assert "--xla_gpu_enable_triton_gemm=false" in os.environ["XLA_FLAGS"]
assert jax.config.jax_compilation_cache_dir is None
lmhdx.make_hartmann_case(ha=2, ny=4, nz=4)
unsafe = platform.system() == "Darwin" and tuple(map(int, jaxlib.__version__.split(".")[:2])) < (0, 10)
enabled = os.environ["LMHDX_COMPILATION_CACHE"] != "0" and not unsafe
assert (jax.config.jax_compilation_cache_dir == os.environ["LMHDX_COMPILATION_CACHE"]) == enabled
assert jax.config.jax_compilation_cache_max_size == (2**31 if enabled else -1)
# JAX reads and writes a size-bounded cache only through filelock; without it every entry fails.
import importlib.util
assert importlib.util.find_spec("filelock") is not None
"""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key != "XLA_FLAGS" and not key.startswith(("JAX_COMPILATION_CACHE", "JAX_PERSISTENT_CACHE"))
    }
    environment["LMHDX_COMPILATION_CACHE"] = choice if choice == "0" else str(tmp_path / choice)
    subprocess.run([sys.executable, "-c", code], check=True, timeout=90, env=environment)


def _write_minimal_config(
    tmp_path: Path,
    name: str,
    *,
    case_extra: str = "",
    geometry_kind: str | None = "rect_duct",
    geometry_extra: str = "",
    magnetic_kind: str = "constant",
    solver: str = "",
) -> Path:
    geometry_type = "" if geometry_kind is None else f'kind = "{geometry_kind}"'
    magnetic_value = "value = [0.0, 0.0, 1.0]" if magnetic_kind == "constant" else ""
    solver_table = f"[solver]\n{solver}" if solver else ""
    path = tmp_path / f"{name}.toml"
    path.write_text(
        f"""
[case]
name = "{name}"
{case_extra}

[geometry]
{geometry_type}
width = 1.0
height = 1.0
ny = 4
nz = 4
{geometry_extra}

[magnetic_field]
kind = "{magnetic_kind}"
{magnetic_value}

{solver_table}

[time_stepper]
dt = 0.1
t_final = 0.1
max_steps = 1

[[regions]]
name = "fluid"
kind = "fluid"
conductivity = 1.0

[[boundary_conditions]]
name = "wall"
kind = "no_slip"
""".strip()
    )
    return path


def test_draft_ci_defers_numerics_without_skipping_ready_or_coverage_jobs(tmp_path):
    workflow = Path(".github/workflows/ci.yml").read_text()
    assert "types: [opened, synchronize, reopened, ready_for_review, converted_to_draft]" in workflow
    assert "  push:\n    branches: [main]" in workflow
    assert "  workflow_dispatch:" in workflow and "  workflow_call:" in workflow
    script = textwrap.dedent(workflow.split("run: |\n", 1)[1].split("\n\n  ", 1)[0])
    for event in ("push", "workflow_dispatch", "workflow_call"):
        output = tmp_path / event
        rendered = script.replace("${{ github.event_name }}", event).replace("${{ github.base_ref }}", "main")
        subprocess.run(
            ["bash", "-e", "-c", rendered],
            check=True,
            env={**os.environ, "GITHUB_OUTPUT": str(output)},
        )
        assert output.read_text().splitlines() == ["full=true", "targeted=false"]
    for event, draft in (
        ("pull_request", True),
        ("pull_request", False),
        ("push", False),
        ("workflow_dispatch", False),
        ("workflow_call", False),
    ):
        for full, targeted in ((True, False), (False, True), (False, False)):
            expected = {
                "quality": full or targeted,
                "pr-impact": event == "pull_request" and not draft and targeted,
                "pr-tests": event == "pull_request" and not draft and full,
                "compatibility": event != "pull_request" and full,
                "coverage": event != "pull_request" and full,
            }
            for job, enabled in expected.items():
                expression = workflow.split(f"\n  {job}:\n", 1)[1].split("    if: ", 1)[1].splitlines()[0]
                for key, value in {
                    "github.event_name": event,
                    "github.event.pull_request.draft": str(draft).lower(),
                    "needs.scope.outputs.full": str(full).lower(),
                    "needs.scope.outputs.targeted": str(targeted).lower(),
                }.items():
                    expression = expression.replace(key, repr(value))
                # These job guards deliberately use only shell-compatible comparisons/boolean operators.
                result = subprocess.run(["bash", "-c", f"[[ {expression} ]]"], check=False)
                assert result.returncode == (0 if enabled else 1), (job, event, draft, full, targeted)


def test_every_test_file_reaches_the_combined_coverage():
    """A file outside every shard, or a shard the coverage job does not restore, scores zero."""
    from scripts.run_full_test_suite import _TEST_SHARDS

    sharded = {entry.split("::", 1)[0] for shard in _TEST_SHARDS.values() for entry in shard}
    present = {f"tests/{path.name}" for path in Path("tests").glob("test_*.py")}
    assert present - sharded == set(), "add these files to a shard in _TEST_SHARDS"
    assert sharded - present == set(), "these shard entries no longer exist"

    workflow = Path(".github/workflows/ci.yml").read_text()
    matrix = workflow.split("\n  compatibility:\n", 1)[1].split("shard: [", 1)[1].split("]", 1)[0]
    coverage = workflow.split("\n  coverage:\n", 1)[1].split("\n  pr-tests:\n", 1)[0]
    restored = re.findall(r"-(\w+)\n\s+fail-on-cache-miss: true", coverage)
    assert {shard.strip() for shard in matrix.split(",")} == set(_TEST_SHARDS)
    assert sorted(restored) == sorted(_TEST_SHARDS), "restore every shard's evidence before combining"


@pytest.mark.parametrize(
    "changed,full,targeted,docs,external",
    [
        (".github/workflows/ci.yml", "true", "false", "false", "false"),
        (".github/workflows/docs.yml", "true", "false", "true", "false"),
        (".github/workflows/external-validation.yml", "true", "false", "false", "true"),
        (".github/actions/setup-lmhdx/action.yml", "true", "false", "true", "true"),
        ("src/lmhdx/mesh.py", "true", "false", "true", "false"),
        ("src/lmhdx/q2d.py", "true", "false", "true", "false"),
        ("src/lmhdx/validation.py", "true", "false", "true", "false"),
        ("validation/freemhd.py", "true", "false", "false", "true"),
        ("tests/test_mesh.py", "true", "false", "false", "false"),
        ("scripts/run_full_test_suite.py", "false", "true", "false", "false"),
        ("docs/index.md", "false", "false", "true", "false"),
        ("README.md", "false", "false", "true", "false"),
    ],
)
def test_ci_scope_and_superseded_work_policy(tmp_path, changed, full, targeted, docs, external):
    root = Path(__file__).resolve().parents[1] / ".github/workflows"

    def git(*args):
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=LMhdX test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("commit", "--allow-empty", "-qm", "baseline")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    path = tmp_path / changed
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("BENCHMARK_B\n")
    git("add", changed)
    git("commit", "-qm", "candidate")
    groups = set()
    for name, expected in (
        ("ci", {"full": full, "targeted": targeted}),
        ("docs", {"docs": docs}),
        ("external-validation", {"b2": external}),
    ):
        workflow = (root / f"{name}.yml").read_text()
        script = textwrap.dedent(workflow.split("run: |\n", 1)[1].split("\n\n  ", 1)[0])
        script = script.replace("${{ github.event_name }}", "pull_request").replace(
            "${{ github.base_ref }}", "main"
        )
        output = tmp_path / f"{name}.outputs"
        subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", script],
            cwd=tmp_path,
            check=True,
            env={**os.environ, "GITHUB_OUTPUT": str(output)},
        )
        assert dict(line.split("=", 1) for line in output.read_text().splitlines()) == expected
        policy = workflow.split("concurrency:\n", 1)[1].split("\n\n", 1)[0]
        assert "cancel-in-progress: true" in policy and "github.event.pull_request.number" in policy
        group = policy.split("group: ", 1)[1].splitlines()[0]
        assert group not in groups
        groups.add(group)


def test_load_run_config_reads_complete_toml(tmp_path: Path):
    input_file = tmp_path / "hartmann.toml"
    input_file.write_text(
        """
[case]
name = "hartmann_toml_demo"
forcing = 1.0
initial_velocity = 0.0
reference_pressure_gradient = -1.0
reference_phi_cell = [0, 0]
notes = "unit test"

[geometry]
kind = "rect_duct"
width = 2.0
height = 2.0
length = 1.0
nx = 1
ny = 8
nz = 8
wall_thickness = [0.0, 0.0, 0.0, 0.0]
wall_cells = [0, 0, 0, 0]
wall_model = "resolved"
target_ha = 20.0

[magnetic_field]
kind = "constant"
value = [0.0, 0.0, 20.0]
ramp_start = 0.0
ramp_duration = 0.0

[solver]
kind = "fully_developed_inductionless"
mode = "steady"
preconditioner = "jacobi"
time_scheme = "implicit_euler"
coupling_iterations = 9
coupling_tolerance = 1.0e-7
coupling_acceleration = "aitken"
coupling_min_relaxation = 0.1
coupling_max_relaxation = 12.0
coupling_history_depth = 5
coupling_regularization = 1.0e-9
coupling_damping = 0.8

[time_stepper]
dt = 0.001
t_final = 0.01
max_steps = 10
potential_iterations = 100
potential_relaxation = 1.0
potential_solver = "cg"
steady_tolerance = 1e-8

[output]
directory = "./out"
write_paraview = true
write_csv_profiles = true
write_npz = true
write_json_summary = true
write_plots = false
copy_input_file = true
write_stride = 1
history_stride = 3

[logging]
enabled = true
banner = true
print_footer = true
flush = true
step_stride = 2

[restart]
enabled = true
path = "./previous_results.npz"
reset_histories = false
write_restart = true
restart_filename = "hartmann_restart.npz"

[[regions]]
name = "fluid"
kind = "fluid"
conductivity = 1.0
density = 1.0
viscosity = 0.01

[[boundary_conditions]]
name = "y_min_wall"
kind = "no_slip"
axis = "y"
side = "min"

[[boundary_conditions]]
name = "y_max_wall"
kind = "no_slip"
axis = "y"
side = "max"

[[boundary_conditions]]
name = "z_min_wall"
kind = "insulating"
axis = "z"
side = "min"

[[boundary_conditions]]
name = "z_max_wall"
kind = "insulating"
axis = "z"
side = "max"
""".strip()
    )

    config = load_run_config(input_file)

    assert config.case.name == "hartmann_toml_demo"
    assert config.case.geometry.kind == "rect_duct"
    assert config.case.geometry.wall_model == "resolved"
    assert config.case.output.directory == str((tmp_path / "out").resolve())
    assert config.case.output.history_stride == 3
    assert config.case.solver.kind == "fully_developed_inductionless"
    assert config.case.solver.mode == "steady"
    assert config.case.solver.preconditioner == "jacobi"
    assert config.case.solver.coupling_iterations == 9
    assert config.case.solver.coupling_acceleration == "aitken"
    assert config.case.solver.coupling_min_relaxation == pytest.approx(0.1)
    assert config.case.solver.coupling_max_relaxation == pytest.approx(12.0)
    assert config.case.solver.coupling_history_depth == 5
    assert config.case.solver.coupling_regularization == pytest.approx(1.0e-9)
    assert config.case.solver.coupling_damping == pytest.approx(0.8)
    assert config.case.time_stepper.potential_solver == "cg"
    assert config.logging.step_stride == 2
    assert config.restart.enabled is True
    assert config.restart.path == (tmp_path / "previous_results.npz").resolve()
    assert config.restart.reset_histories is False
    assert config.restart.write_restart is True
    assert config.restart.restart_filename == "hartmann_restart.npz"
    assert len(config.case.regions) == 1
    assert len(config.case.boundary_conditions) == 4


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"magnetic_kind": "analytic"}, "analytic magnetic-field"),
        ({"solver": 'kind = "invalid"'}, "Unsupported solver kind"),
        ({"geometry_kind": None}, "Missing required TOML key 'kind'"),
        ({"geometry_extra": "wall_thickness = [0.1, 0.2]"}, "must have length 4"),
        ({"solver": 'mode = "invalid"'}, "Unsupported solve mode"),
    ],
)
def test_load_run_config_rejects_invalid_inputs(tmp_path: Path, kwargs: dict[str, str | None], message: str):
    input_file = _write_minimal_config(tmp_path, "rejected", **kwargs)

    with pytest.raises(ValueError, match=message):
        load_run_config(input_file)


def test_shipped_example_toml_files_parse():
    root = Path(__file__).resolve().parents[1]

    for relative in ("examples/hartmann_case.toml",):
        config = load_run_config(root / relative)
        assert config.case.name
        assert config.case.regions


def test_tutorials_map_to_executable_examples_or_numerical_tests():
    root = Path(__file__).resolve().parents[1]
    tutorial_paths = sorted((root / "docs/tutorials").glob("*.md"))
    expected = {
        "differentiation.md",
        "fringing.md",
        "fully_developed.md",
        "q2d.md",
        "walls_and_fields.md",
    }
    assert {path.name for path in tutorial_paths} == expected

    catalog = tomllib.loads((root / "examples/catalog.toml").read_text())
    documented = {Path(item["docs"]).name for item in catalog["example"]}
    index = (root / "docs/index.md").read_text()
    for path in tutorial_paths:
        if path.name != "differentiation.md":
            assert path.name in documented
        assert f"tutorials/{path.stem}" in index
        assert "```python" in path.read_text()


def test_parse_boundary_value_accepts_scalar_and_vector_and_rejects_bad_inputs():
    assert _parse_boundary_value(None) is None
    assert _parse_boundary_value(1.25) == pytest.approx(1.25)
    assert _parse_boundary_value([1, 2, 3]) == pytest.approx((1.0, 2.0, 3.0))

    with pytest.raises(ValueError, match="length 3"):
        _parse_boundary_value([1, 2])

    with pytest.raises(ValueError, match="Unsupported boundary-condition value"):
        _parse_boundary_value({"bad": True})


@pytest.mark.parametrize(
    ("wall_thickness", "hartmann_half_spacing", "message"),
    ((0.0, 1.0, "wall_thickness"), (1.0, 0.0, "hartmann_half_spacing")),
)
def test_wall_conductivity_rejects_nonpositive_geometry(
    wall_thickness: float, hartmann_half_spacing: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _wall_conductivity_from_conductance_ratio(
            wall_conductance_ratio=1.0,
            fluid_conductivity=1.0,
            wall_thickness=wall_thickness,
            hartmann_half_spacing=hartmann_half_spacing,
        )


def test_precision_setup_pins_true_float32_contractions_and_keeps_a_user_choice():
    """Unset, JAX allows TensorFloat-32 contractions on Ampere GPUs (3e-4 from float64 on an A4000)."""
    import jax
    import jax.numpy as jnp

    setups = {
        "enable_x64": lmhdx.enable_x64,
        "case dtype": lambda: lmhdx.make_hartmann_case(ha=2, ny=8, nz=8, dtype="float32"),
        "ChannelProblem": lambda: lmhdx.duct_problem(hartmann=20.0, cells=24),
        "Q2DProblem": lambda: lmhdx.Q2DProblem(jnp.zeros((8, 8), dtype=jnp.float32)),
    }
    initial = jax.config.jax_default_matmul_precision
    try:
        for name, setup in setups.items():
            jax.config.update("jax_default_matmul_precision", None)
            setup()
            assert jax.config.jax_default_matmul_precision == "highest", name
            jax.config.update("jax_default_matmul_precision", "tensorfloat32")
            setup()
            assert jax.config.jax_default_matmul_precision == "tensorfloat32", name

        jax.config.update("jax_default_matmul_precision", None)
        lmhdx.enable_x64()
        left, right = jax.random.uniform(jax.random.PRNGKey(0), (2, 32, 32, 32), dtype=jnp.float64)

        def contract(a, b):
            return jnp.einsum("ijk,klm->ijlm", a, b)

        exact = contract(left, right)
        single = contract(left.astype(jnp.float32), right.astype(jnp.float32))
        assert float(jnp.max(jnp.abs(single - exact) / exact)) <= 1.0e-6
    finally:
        jax.config.update("jax_default_matmul_precision", initial)


EXPECTED_ROOT_API = {
    "enable_x64",
    "ChannelProblem",
    "duct_problem",
    "solve_steady_state",
    "advance",
    "enable_compilation_cache",
    "make_hartmann_case",
    "make_shercliff_case",
    "make_hunt_case",
    "make_q2d_case",
    "evolve_q2d",
    "solve_fully_developed_fields",
    "Q2DProblem",
    "solve",
    "generate_rect_duct_mesh",
    "generate_rect_duct_mesh_from_faces",
    "generate_layered_duct_mesh",
    "generate_layered_duct_mesh_from_fluid_faces",
    "generate_multilayer_duct_mesh",
    "WallLayer",
    "dynamic_to_kinematic_viscosity",
    "kinematic_to_dynamic_viscosity",
    "hartmann_number",
    "reynolds_number",
    "interaction_parameter",
    "magnetic_reynolds_number",
    "magnetic_field_from_hartmann",
    "wall_conductance_ratio",
    "effective_pinhole_conductance_ratio",
    "tangential_stack_conductance_ratio",
    "normal_stack_leakage_ratio",
    "equivalent_single_layer",
    "nested_wall_layer_resolution_summary",
}


def test_architecture_inventory_is_deterministic_without_timing() -> None:
    assert build_inventory() == build_inventory()
    assert Path(_test_environment()["PYTHONPATH"].split(os.pathsep)[0]) == Path("src").resolve()


def test_change_gate_selects_affected_tests_and_fails_closed() -> None:
    assert _tests_for_changes(("docs/index.md", "plan.md")) == ()
    assert _tests_for_changes(("src/lmhdx/axial.py",)) == (
        "tests/test_axial.py",
        "tests/test_example_runner.py",
    )
    assert _tests_for_changes(("src/lmhdx/q2d.py", "examples/q2d_vortex.py")) == (
        "tests/test_physics.py",
        "tests/test_q2d_identities.py",
        "tests/test_example_runner.py",
    )
    assert _tests_for_changes(("tests/test_io.py",)) == ("tests/test_io.py",)
    assert _tests_for_changes(("unknown executable",)) == _ALL_TESTS


def test_stable_root_api_is_small_lazy_and_resolvable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert set(lmhdx.__all__) == EXPECTED_ROOT_API
    assert EXPECTED_ROOT_API <= set(dir(lmhdx))
    assert all(callable(getattr(lmhdx, name)) for name in lmhdx.__all__)
    assert not [name for name in lmhdx.__all__ if not inspect.getdoc(getattr(lmhdx, name))]
    api_reference = Path("docs/reference/api.md").read_text()
    assert all(f"`{name}`" in api_reference for name in lmhdx.__all__)

    import jax

    updates = []
    monkeypatch.setattr("lmhdx.io.jax.config.update", lambda *args: updates.append(args))
    cache = lmhdx.enable_compilation_cache(
        tmp_path / "jax-cache", min_compile_time_secs=2.0, min_entry_size_bytes=4096, share_across_values=True
    )
    assert cache.is_dir()
    assert updates == [
        ("jax_compilation_cache_dir", str(cache)),
        ("jax_persistent_cache_min_entry_size_bytes", 4096),
        ("jax_persistent_cache_min_compile_time_secs", 2.0),
    ] + [("jax_use_simplified_jaxpr_constants", True)] * (
        "jax_use_simplified_jaxpr_constants" in jax.config.values
    )


def test_advanced_api_uses_owning_module() -> None:
    assert not hasattr(lmhdx, "solve_open_duct")
    from lmhdx.axial import solve_open_duct

    assert callable(solve_open_duct)


def test_unknown_root_attribute_has_standard_error() -> None:
    with pytest.raises(AttributeError, match="not_an_api"):
        lmhdx.not_an_api


def test_architecture_inventory_ignores_generated_egg_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "source.py").write_bytes(b"x")
    metadata = tmp_path / "package.egg-info"
    metadata.mkdir()
    (metadata / "PKG-INFO").write_bytes(b"generated")

    def missing_git(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("scripts.audit_architecture.subprocess.run", missing_git)
    assert _checkout_size(tmp_path) == 1


def test_root_import_is_lazy_and_within_budget() -> None:
    payload = build_inventory()
    payload["import_measurement"] = measure_import(repeats=3)
    assert architecture_budget_errors(payload) == []


def test_numerical_modules_do_not_import_optional_visualization() -> None:
    code = """
import sys
import lmhdx.io
assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)
assert not any(name == 'PIL' or name.startswith('PIL.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_wheel_audit_rejects_nonpackage_payload(tmp_path: Path) -> None:
    wheel = tmp_path / "lmx-test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("lmhdx/__init__.py", "")
        archive.writestr("lmhdx/py.typed", "")
        archive.writestr("lmx-1.dist-info/METADATA", "")
        archive.writestr("benchmarks/raw.bin", b"large output")
    assert inspect_wheel(wheel)["forbidden_members"] == ["benchmarks/raw.bin"]
    assert "outside lmhdx/" in architecture_budget_errors(build_inventory(), wheel=wheel)[0]


def test_root_api_is_pep561_marked_and_fully_annotated() -> None:
    assert (Path("src/lmhdx") / "py.typed").is_file()
    for name in lmhdx.__all__:
        value = getattr(lmhdx, name)
        if not (inspect.isfunction(value) or inspect.isclass(value)):
            continue
        signature = inspect.signature(value)
        assert signature.return_annotation is not inspect.Signature.empty, name
        assert all(
            parameter.annotation is not inspect.Parameter.empty
            for parameter in signature.parameters.values()
            if parameter.name not in {"self", "cls"}
        ), name


def test_sdist_audit_rejects_repository_tests(tmp_path: Path) -> None:
    source = tmp_path / "lmx-1" / "tests" / "test_solver.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"large output")
    sdist = tmp_path / "lmx-test.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(source, arcname="lmx-1/tests/test_solver.py")
    assert inspect_sdist(sdist)["forbidden_members"] == ["tests/test_solver.py"]
    assert "outside its source payload" in architecture_budget_errors(build_inventory(), sdist=sdist)[0]


def test_curated_examples_use_submodules_and_linear_scripts_are_editable() -> None:
    inventory = build_inventory()["inventory"]
    stable = set(lmhdx.__all__)
    for item in inventory["curated_examples"]:
        path = Path(item["path"])
        if path.suffix != ".py":
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imports = (
            node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "lmhdx"
        )
        root_imports = {alias.name for node in imports for alias in node.names}
        assert root_imports <= stable, f"{path} imports unsupported root APIs: {root_imports - stable}"
        linear_limits = {
            "fringe_duct_example.py": 160,
            "hartmann_example.py": 160,
            "hunt_example.py": 160,
            "li_aln_wall_stack_example.py": 260,
            "q2d_turbulence_demo.py": 140,
        }
        if path.name in linear_limits:
            assert ast.get_docstring(tree)
            assert "# Inputs:" in source and "# Run" in source
            assert len(source.splitlines()) <= linear_limits[path.name]
            functions = (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            assert all(node.name != "main" and ast.get_docstring(node) for node in functions)
            assert "argparse" not in source and "__name__" not in source


def test_curated_examples_declare_user_facing_contracts(tmp_path: Path) -> None:
    inventory = build_inventory()["inventory"]
    curated = inventory["curated_examples"]
    assert {item["path"] for item in curated} == set(inventory["examples"])
    assert len(curated) == 6
    for item in curated:
        assert item["command"]
        assert item["outputs"]
        assert item["runtime"] in {"portable", "accelerator-optional"}
        assert Path(item["docs"]).is_file()
    q2d = Path(__file__).resolve().parents[1] / "examples/q2d_turbulence_demo.py"
    root = q2d.parents[1]
    environment = {
        **os.environ,
        "PATH": str(tmp_path),
        "PYTHONPATH": os.pathsep.join((str(root / "src"), os.environ.get("PYTHONPATH", ""))),
    }
    subprocess.run([sys.executable, q2d], cwd=tmp_path, timeout=30, check=True, env=environment)
    summary_path = next((tmp_path / "artifacts").rglob("q2d_vortex_decay.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "completed"
    assert summary["frames"] == 41
    assert summary["diagnostics"]["kinetic_energy_final"] < summary["diagnostics"]["kinetic_energy_initial"]
    assert (summary_path.parent / summary["poster"]).is_file()
    if summary["movie"] is not None:
        assert (summary_path.parent / summary["movie"]).is_file()


def test_the_readme_duct_solves_in_a_fresh_process_without_enabling_x64():
    code = """
import jax, lmhdx
assert not jax.config.x64_enabled
solution = lmhdx.solve(lmhdx.duct_problem(hartmann=20.0, cells=32, wall_conductance=0.027))
assert jax.config.x64_enabled and solution.velocity[0].data.dtype == "float64"
"""
    environment = {key: value for key, value in os.environ.items() if key != "JAX_ENABLE_X64"}
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300, env=environment)
