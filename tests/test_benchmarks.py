import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import lmhdx.io as benchmarks
from lmhdx.io import (
    benchmark_solver,
    write_benchmark_report,
)

pytestmark = pytest.mark.unit


def test_device_environment_records_cpu_count_without_affinity(monkeypatch):
    import jax

    from scripts import run_benchmarks as runner

    monkeypatch.delattr(runner.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 14)
    assert runner._environment(jax)["cpu_count"] == 14


@pytest.mark.parametrize(
    "arguments",
    [
        ["--cases", "unknown"],
        ["--cases", ""],
        ["--steps", "0"],
        ["--repeats", "-1"],
        ["--q2d-sizes", ""],
        ["--core3d-sizes", "1"],
    ],
)
def test_device_benchmark_rejects_invalid_requests(arguments):
    from scripts.run_benchmarks import main

    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("verdict", [True, False, "exception"])
def test_device_benchmark_exit_matches_retained_evidence(tmp_path, monkeypatch, verdict):
    from scripts import run_benchmarks as runner

    def build(*args, **options):
        if verdict == "exception":
            raise RuntimeError("bounded allocation failure")
        return {"accepted": verdict}

    monkeypatch.setattr(runner, "_q2d_case", build)
    monkeypatch.setattr(runner, "_environment", lambda jax: {})
    output = tmp_path / "report.json"
    assert runner.main(["--cases", "q2d", "--q2d-sizes", "8", "--output", str(output)]) == (
        0 if verdict is True else 1
    )
    entry = json.loads(output.read_text())["cases"][0]
    assert entry["accepted"] is (verdict is True)
    if verdict == "exception":
        assert "bounded allocation failure" in entry["error"]


@pytest.mark.parametrize("status", ["completed", "courant_limit_exceeded", "energy_budget_exceeded"])
def test_device_benchmark_honors_q2d_solver_verdict(monkeypatch, status):
    import jax
    import jax.numpy as jnp

    import lmhdx
    from scripts.run_benchmarks import _q2d_case

    monkeypatch.setattr(
        lmhdx,
        "solve",
        lambda problem: SimpleNamespace(
            vorticity=jnp.zeros((8, 8)),
            status=status,
        ),
    )
    result = _q2d_case(jax, 8, 2, 1)
    assert result["solver_status"] == status
    assert result["accepted"] is (status == "completed")


def test_trajectory_timing_does_not_certify_host_synchronization(monkeypatch):
    from scripts import run_benchmarks as runner

    monkeypatch.setattr(
        runner,
        "_core3d_case",
        lambda jax, cells, steps, repeats, **options: {
            "accepted": True,
            "seconds_per_step": float(steps),
        },
    )
    result = runner._host_sync_case(None, 8, 2, 1)
    assert result["per_step_ratio"] == 4.0
    assert result["accepted"] and not result["host_sync_verified"]


def test_device_environment_records_precision_provenance(monkeypatch):
    from scripts import run_benchmarks as runner

    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=2")
    monkeypatch.setattr(runner, "_commit", lambda *root: "abc1234")
    header = "| NVIDIA-SMI 535.183.01    Driver Version: 535.183.01    CUDA Version: 12.2 |\n"
    queried = []

    def nvidia_smi(command, **options):
        queried.append(command[0])
        return SimpleNamespace(stdout=header)

    monkeypatch.setattr(runner.subprocess, "run", nvidia_smi)
    gpu = SimpleNamespace(
        __version__="0.10.2",
        devices=lambda: [SimpleNamespace(platform="gpu", device_kind="NVIDIA RTX A4000")],
        config=SimpleNamespace(jax_enable_x64=False, jax_default_matmul_precision="highest"),
    )
    environment = runner._environment(gpu)
    assert queried == ["nvidia-smi"]
    assert environment["jax_default_matmul_precision"] == "highest"
    assert environment["nvidia_tf32_override"] == "0"
    assert environment["xla_flags"] == "--xla_gpu_autotune_level=2"
    assert (environment["gpu_driver"], environment["cuda_version"]) == ("535.183.01", "12.2")
    assert len(environment["load_average"]) == 3
    assert runner._unquotable_float32(environment) is None

    def missing(command, **options):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(runner.subprocess, "run", missing)
    monkeypatch.delattr(runner.os, "getloadavg")
    environment = runner._environment(gpu)
    assert (environment["gpu_driver"], environment["cuda_version"]) == (None, None)
    assert environment["load_average"] is None


def test_device_environment_does_not_query_a_gpu_on_cpu(monkeypatch):
    import jax

    from scripts import run_benchmarks as runner

    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    monkeypatch.setattr(runner, "_commit", lambda *root: "abc1234")
    monkeypatch.setattr(runner, "_gpu_versions", lambda: pytest.fail("queried a GPU on a CPU host"))
    environment = runner._environment(jax)
    assert environment["platform"] == "cpu"
    assert environment["jax_default_matmul_precision"] == jax.config.jax_default_matmul_precision
    assert environment["nvidia_tf32_override"] is None
    assert (environment["gpu_driver"], environment["cuda_version"]) == (None, None)


_FLOAT32_GPU = {"platform": "gpu", "x64": False}


@pytest.mark.parametrize(
    ("environment", "written"),
    [
        (_FLOAT32_GPU, False),
        ({**_FLOAT32_GPU, "jax_default_matmul_precision": None}, False),
        ({**_FLOAT32_GPU, "jax_default_matmul_precision": "default"}, False),
        ({**_FLOAT32_GPU, "jax_default_matmul_precision": "tensorfloat32"}, False),
        ({**_FLOAT32_GPU, "jax_default_matmul_precision": "highest"}, True),
        ({"platform": "gpu", "x64": True, "jax_default_matmul_precision": None}, True),
        ({"platform": "cpu", "x64": False, "jax_default_matmul_precision": None}, True),
    ],
)
def test_device_benchmark_refuses_float32_gpu_reports_without_true_precision(
    tmp_path, monkeypatch, environment, written
):
    from scripts import run_benchmarks as runner

    built = []
    monkeypatch.setattr(runner, "_environment", lambda jax: environment)
    monkeypatch.setattr(
        runner, "_q2d_case", lambda *args, **options: built.append(args) or {"accepted": True}
    )
    output = tmp_path / "report.json"
    arguments = ["--cases", "q2d", "--q2d-sizes", "8", "--output", str(output)]
    if written:
        assert runner.main(arguments) == 0
        assert json.loads(output.read_text())["environment"] == environment
    else:
        with pytest.raises(SystemExit) as error:
            runner.main(arguments)
        assert error.value.code == 2
        assert not output.exists() and not built


def test_device_benchmark_discards_warmups_and_pools_alternating_runs(tmp_path):
    from scripts import run_benchmarks as runner

    calls = []
    compile_seconds, samples, _ = runner._timed(
        SimpleNamespace(block_until_ready=lambda value: value), lambda: calls.append(1), repeats=3, warmups=2
    )
    assert len(calls) == 6 and len(samples) == 3
    median, (low, high) = runner._median_ci([1.0, 2.0, 3.0, 4.0, 100.0])
    assert median == 3.0 and low <= median <= high <= 100.0

    def report(samples, commit="79ba6e4"):
        case = {"case": "q2d_evolve", "shape": [8, 8], "steps": 2, "compile_seconds": 1.0, "accepted": True}
        environment = {"platform": "cpu", "x64": True, "lmx_commit": commit, "load_average": [1.0, 1.0, 1.0]}
        return {"environment": environment, "cases": [{**case, **runner._timing(1.0, samples, 2)}]}

    paths = [tmp_path / name for name in ("a.json", "b.json", "c.json")]
    for path, contents in zip(
        paths, (report([2.0, 4.0]), report([6.0, 8.0]), report([1.0], "other")), strict=True
    ):
        path.write_text(json.dumps(contents))
    output = tmp_path / "pooled.json"
    assert runner.main(["--combine", str(paths[0]), str(paths[1]), "--output", str(output)]) == 0
    pooled = json.loads(output.read_text())
    assert pooled["environment"]["rounds"] == 2 and pooled["cases"][0]["warm_samples"] == [2.0, 4.0, 6.0, 8.0]
    assert pooled["cases"][0]["seconds_per_step"] == 2.5
    with pytest.raises(SystemExit) as error:
        runner.main(["--combine", str(paths[0]), str(paths[2]), "--output", str(output)])
    assert error.value.code == 2


_MATCHED = ("matched_contract",)
_SHARED = _MATCHED + ("shared",)
_EQUATIONS = _SHARED + ("equations",)
_SEMANTICS = "matched formulation semantics differ"
_STEADY_STEPS = _MATCHED + ("roles", "b2-production", "stopping_rules", "steady_steps_min")
_STOPPING = "matched stopping contract differs"
_REFERENCE_HEADER = "x_over_L,b_over_B0,b_uncertainty,pressure_observable,pressure_uncertainty"


def test_benchmark_solver_returns_positive_timings(monkeypatch: pytest.MonkeyPatch):
    times = iter([10.0, 10.4, 10.4, 10.7])

    monkeypatch.setattr(
        benchmarks,
        "make_hartmann_case",
        lambda ha, ny, nz: SimpleNamespace(name="hartmann_ha5"),
    )
    fields = SimpleNamespace(u=object(), phi=object())
    monkeypatch.setattr(
        "lmhdx.fully_developed.solve_fully_developed", lambda case: SimpleNamespace(fields=fields)
    )
    synchronized = []
    monkeypatch.setattr(benchmarks.jax, "block_until_ready", synchronized.append)
    monkeypatch.setattr(benchmarks.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(benchmarks.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(benchmarks.jax, "devices", lambda: [SimpleNamespace(device_kind="cpu")])
    monkeypatch.setattr(benchmarks.platform, "python_version", lambda: "3.13.7")

    report = benchmark_solver(repeats=2, ha=5.0, ny=16, nz=16)
    assert float(report["cold_seconds"]) > 0.0
    assert float(report["warm_seconds"]) == pytest.approx(0.3)
    assert report["warm_cv"] == 0.0
    assert report["backend"]
    assert synchronized == [(fields.u, fields.phi)] * 2


def test_benchmark_writer(tmp_path: Path):
    path = write_benchmark_report({"cold_seconds": 1.0, "warm_seconds": 0.5}, tmp_path / "benchmark.json")
    assert path.exists()
