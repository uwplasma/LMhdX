from __future__ import annotations

import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, TextIO

import jax.numpy as jnp
from jax import config as jax_config

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only used on Python < 3.11
    import tomli as tomllib

if TYPE_CHECKING:
    from .mesh import StructuredMesh

RegionKind = Literal["fluid", "solid"]
GeometryKind = Literal["rect_duct", "layered_duct"]
WallModel = Literal["auto", "thin", "resolved"]
SolverKind = Literal["fully_developed_inductionless"]
SolveMode = Literal["steady", "transient"]
BoundaryKind = Literal[
    "no_slip",
    "insulating",
    "conducting_wall",
    "inlet_velocity",
    "inlet_flow_rate",
    "outlet_pressure",
    "imposed_current_density",
]
MagneticFieldKind = Literal["constant", "analytic", "tabulated"]
PotentialSolverKind = Literal["auto", "jacobi", "cg", "cg_volume"]
PreconditionerKind = Literal["none", "jacobi"]
TimeSchemeKind = Literal["implicit_euler", "crank_nicolson"]
CouplingAccelerationKind = Literal["none", "aitken", "anderson"]


@dataclass(frozen=True)
class RegionSpec:
    """Material-region specification.

    ``viscosity`` is kinematic viscosity ``nu`` in ``m^2/s``. Dynamic
    viscosity ``mu`` should be converted before constructing a case.
    """

    name: str
    kind: RegionKind
    conductivity: float
    density: float | None = None
    viscosity: float | None = None
    wall_thickness: float | None = None


@dataclass(frozen=True)
class BoundaryCondition:
    name: str
    kind: BoundaryKind
    value: float | tuple[float, float, float] | None = None
    region: str | None = None
    axis: str | None = None
    side: str | None = None


@dataclass(frozen=True)
class MagneticFieldSpec:
    kind: MagneticFieldKind
    value: tuple[float, float, float] | None = None
    fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None
    table_path: str | None = None
    ramp_start: float = 0.0
    ramp_duration: float = 0.0


@dataclass(frozen=True)
class SolverConfig:
    kind: SolverKind = "fully_developed_inductionless"
    mode: SolveMode = "steady"
    preconditioner: PreconditionerKind = "jacobi"
    time_scheme: TimeSchemeKind = "implicit_euler"
    coupling_iterations: int = 12
    coupling_tolerance: float = 1e-8
    coupling_acceleration: CouplingAccelerationKind = "none"
    coupling_min_relaxation: float = 0.05
    coupling_max_relaxation: float = 100.0
    coupling_history_depth: int = 6
    coupling_regularization: float = 1.0e-8
    coupling_damping: float = 1.0


@dataclass(frozen=True)
class TimeStepperConfig:
    dt: float
    t_final: float
    max_steps: int
    potential_iterations: int = 400
    potential_tolerance: float | None = None
    potential_relaxation: float = 1.0
    potential_solver: PotentialSolverKind = "auto"
    steady_tolerance: float = 1e-8
    steady_potential_tolerance: float | None = None
    relaxation: float = 0.35
    velocity_update_limit: float = 1e-3


@dataclass(frozen=True)
class OutputSpec:
    directory: str | None = None
    write_paraview: bool = True
    write_csv_profiles: bool = True
    write_npz: bool = True
    write_json_summary: bool = True
    write_plots: bool = False
    copy_input_file: bool = True
    write_stride: int = 1
    history_stride: int = 0


@dataclass(frozen=True)
class GeometrySpec:
    kind: GeometryKind
    width: float
    height: float
    length: float = 1.0
    nx: int = 1
    ny: int = 64
    nz: int = 64
    wall_thickness: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    wall_cells: tuple[int, int, int, int] = (0, 0, 0, 0)
    target_ha: float | None = None
    target_side_layer: float | None = None
    hartmann_layer_cells: int | None = None
    wall_model: WallModel = "auto"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    geometry: GeometrySpec
    regions: tuple[RegionSpec, ...]
    magnetic_field: MagneticFieldSpec
    boundary_conditions: tuple[BoundaryCondition, ...]
    time_stepper: TimeStepperConfig
    solver: SolverConfig = field(default_factory=SolverConfig)
    output: OutputSpec = field(default_factory=OutputSpec)
    forcing: float = 1.0
    initial_velocity: float = 0.0
    reference_pressure_gradient: float = -1.0
    reference_phi_cell: tuple[int, int] = (0, 0)
    notes: str = ""
    dtype: str = "float64"

    def __post_init__(self):
        if self.dtype not in ("float32", "float64"):
            raise ValueError("case dtype must be 'float32' or 'float64'")
        if self.dtype == "float64" and not jax_config.x64_enabled:
            warnings.warn(
                "Call lmhdx.enable_x64() before constructing float64 cases or meshes; "
                "case construction is activating float64 implicitly.",
                DeprecationWarning,
                stacklevel=2,
            )
            from . import enable_x64

            enable_x64()
        # The case dtype is where LMhdX sets precision, float32 included; see lmhdx.enable_x64.
        from . import _pin_matmul_precision

        _pin_matmul_precision()

    @property
    def output_dir(self) -> Path | None:
        if self.output.directory is None:
            return None
        return Path(self.output.directory)


class NumericalFailure(RuntimeError):
    """Raised when a solver produces nonfinite numerical state."""


def require_finite(stage: str, **values) -> None:
    """Raise with field names when numerical output is nonfinite."""

    failed = [name for name, value in values.items() if not bool(jnp.all(jnp.isfinite(jnp.asarray(value))))]
    if failed:
        raise NumericalFailure(f"{stage} produced nonfinite {', '.join(failed)}")


@dataclass(frozen=True)
class MHDState:
    u: jnp.ndarray
    phi: jnp.ndarray
    jy: jnp.ndarray
    jz: jnp.ndarray
    lorentz_x: jnp.ndarray
    time: float
    residual: float


@dataclass(frozen=True)
class Diagnostics:
    residual_history: jnp.ndarray
    courant_like: jnp.ndarray
    ohmic_power: jnp.ndarray
    time_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    u_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    mean_velocity_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    applied_forcing_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    current_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    face_current_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    emf_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    lorentz_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    face_lorentz_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    potential_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    potential_iterations_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    linear_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    linear_iterations_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    volumetric_flow_rate_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    mean_current_magnitude_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    lorentz_power_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    div_current_max_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    charge_balance_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    gauge_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    interface_current_residual_history: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))


@dataclass(frozen=True)
class Solution:
    mesh: StructuredMesh
    state: MHDState
    diagnostics: Diagnostics
    case_name: str
    converged: bool | None = None
    status: str = "not_recorded"
    steps: int = 0

    @property
    def residual(self) -> float:
        """Terminal normalized solver residual."""

        return float(self.state.residual)

    @property
    def fields(self) -> MHDState:
        """Final fully developed MHD state."""

        return self.state


@dataclass(frozen=True)
class LoggingSpec:
    enabled: bool = True
    banner: bool = True
    print_footer: bool = True
    flush: bool = True
    step_stride: int = 1

    def is_enabled(self) -> bool:
        return bool(self.enabled)


@dataclass(frozen=True)
class RestartSpec:
    enabled: bool = False
    path: Path | None = None
    reset_histories: bool = True
    write_restart: bool = False
    restart_filename: str | None = None


@dataclass(frozen=True)
class RunConfig:
    case: CaseSpec
    logging: LoggingSpec = field(default_factory=LoggingSpec)
    restart: RestartSpec = field(default_factory=RestartSpec)
    input_path: Path | None = None


def _load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required TOML key '{key}'")
    return mapping[key]


def _optional_tuple(
    mapping: dict[str, Any], key: str, *, length: int | None = None, cast=float
) -> tuple[Any, ...] | None:
    if key not in mapping:
        return None
    values = tuple(cast(value) for value in mapping[key])
    if length is not None and len(values) != length:
        raise ValueError(f"TOML key '{key}' must have length {length}")
    return values


def _parse_regions(entries: list[dict[str, Any]]) -> tuple[RegionSpec, ...]:
    return tuple(
        RegionSpec(
            name=str(_require(entry, "name")),
            kind=str(_require(entry, "kind")),
            conductivity=float(_require(entry, "conductivity")),
            density=None if entry.get("density") is None else float(entry["density"]),
            viscosity=None if entry.get("viscosity") is None else float(entry["viscosity"]),
            wall_thickness=None if entry.get("wall_thickness") is None else float(entry["wall_thickness"]),
        )
        for entry in entries
    )


def _parse_boundary_value(value: Any) -> float | tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        values = tuple(float(component) for component in value)
        if len(values) != 3:
            raise ValueError("Boundary-condition vector values must have length 3")
        return values
    raise ValueError(f"Unsupported boundary-condition value {value!r}")


def _parse_boundaries(entries: list[dict[str, Any]]) -> tuple[BoundaryCondition, ...]:
    return tuple(
        BoundaryCondition(
            name=str(_require(entry, "name")),
            kind=str(_require(entry, "kind")),
            value=_parse_boundary_value(entry.get("value")),
            region=None if entry.get("region") is None else str(entry["region"]),
            axis=None if entry.get("axis") is None else str(entry["axis"]),
            side=None if entry.get("side") is None else str(entry["side"]),
        )
        for entry in entries
    )


def load_run_config(path: str | Path) -> RunConfig:
    input_path = Path(path).resolve()
    root = _load_toml(input_path)

    case_table = root.get("case", {})
    geometry_table = root.get("geometry", {})
    field_table = root.get("magnetic_field", {})
    solver_table = root.get("solver", {})
    time_table = root.get("time_stepper", {})
    output_table = root.get("output", {})
    logging_table = root.get("logging", {})
    restart_table = root.get("restart", {})
    regions_table = root.get("regions", [])
    boundaries_table = root.get("boundary_conditions", [])

    if field_table.get("kind") == "analytic":
        raise ValueError(
            "TOML input does not support analytic magnetic-field callables; use the Python API for that case"
        )

    geometry = GeometrySpec(
        kind=str(_require(geometry_table, "kind")),
        width=float(_require(geometry_table, "width")),
        height=float(_require(geometry_table, "height")),
        length=float(geometry_table.get("length", 1.0)),
        nx=int(geometry_table.get("nx", 1)),
        ny=int(_require(geometry_table, "ny")),
        nz=int(_require(geometry_table, "nz")),
        wall_thickness=_optional_tuple(geometry_table, "wall_thickness", length=4, cast=float)
        or (0.0, 0.0, 0.0, 0.0),
        wall_cells=_optional_tuple(geometry_table, "wall_cells", length=4, cast=int) or (0, 0, 0, 0),
        target_ha=None if geometry_table.get("target_ha") is None else float(geometry_table["target_ha"]),
        target_side_layer=None
        if geometry_table.get("target_side_layer") is None
        else float(geometry_table["target_side_layer"]),
        wall_model=str(geometry_table.get("wall_model", "auto")),
    )

    magnetic_field = MagneticFieldSpec(
        kind=str(_require(field_table, "kind")),
        value=_optional_tuple(field_table, "value", length=3, cast=float),
        fn=None,
        table_path=None
        if field_table.get("table_path") is None
        else str((input_path.parent / str(field_table["table_path"])).resolve()),
        ramp_start=float(field_table.get("ramp_start", 0.0)),
        ramp_duration=float(field_table.get("ramp_duration", 0.0)),
    )

    time_stepper = TimeStepperConfig(
        dt=float(_require(time_table, "dt")),
        t_final=float(_require(time_table, "t_final")),
        max_steps=int(_require(time_table, "max_steps")),
        potential_iterations=int(time_table.get("potential_iterations", 400)),
        potential_tolerance=None
        if time_table.get("potential_tolerance") is None
        else float(time_table["potential_tolerance"]),
        potential_relaxation=float(time_table.get("potential_relaxation", 1.0)),
        potential_solver=str(time_table.get("potential_solver", "auto")),
        steady_tolerance=float(time_table.get("steady_tolerance", 1e-8)),
        steady_potential_tolerance=None
        if time_table.get("steady_potential_tolerance") is None
        else float(time_table["steady_potential_tolerance"]),
        relaxation=float(time_table.get("relaxation", 0.35)),
        velocity_update_limit=float(time_table.get("velocity_update_limit", 1e-3)),
    )

    solver_mode = str(solver_table.get("mode", "steady"))
    if solver_mode not in {"steady", "transient"}:
        raise ValueError(f"Unsupported solve mode {solver_mode!r}")
    solver_kind = str(solver_table.get("kind", "fully_developed_inductionless"))
    if solver_kind != "fully_developed_inductionless":
        raise ValueError(f"Unsupported solver kind {solver_kind!r}")
    solver = SolverConfig(
        kind=solver_kind,
        mode=solver_mode,
        preconditioner=str(solver_table.get("preconditioner", "jacobi")),
        time_scheme=str(solver_table.get("time_scheme", "implicit_euler")),
        coupling_iterations=int(solver_table.get("coupling_iterations", 12)),
        coupling_tolerance=float(solver_table.get("coupling_tolerance", 1e-8)),
        coupling_acceleration=str(solver_table.get("coupling_acceleration", "none")),
        coupling_min_relaxation=float(solver_table.get("coupling_min_relaxation", 0.05)),
        coupling_max_relaxation=float(solver_table.get("coupling_max_relaxation", 100.0)),
        coupling_history_depth=int(solver_table.get("coupling_history_depth", 6)),
        coupling_regularization=float(solver_table.get("coupling_regularization", 1.0e-8)),
        coupling_damping=float(solver_table.get("coupling_damping", 1.0)),
    )
    output_dir = output_table.get("directory")
    if output_dir is not None:
        output_dir = str((input_path.parent / str(output_dir)).resolve())

    output = OutputSpec(
        directory=output_dir,
        write_paraview=bool(output_table.get("write_paraview", True)),
        write_csv_profiles=bool(output_table.get("write_csv_profiles", True)),
        write_npz=bool(output_table.get("write_npz", True)),
        write_json_summary=bool(output_table.get("write_json_summary", True)),
        write_plots=bool(output_table.get("write_plots", False)),
        copy_input_file=bool(output_table.get("copy_input_file", True)),
        write_stride=int(output_table.get("write_stride", 1)),
        history_stride=int(output_table.get("history_stride", 0)),
    )

    case = CaseSpec(
        name=str(_require(case_table, "name")),
        geometry=geometry,
        regions=_parse_regions(regions_table),
        magnetic_field=magnetic_field,
        boundary_conditions=_parse_boundaries(boundaries_table),
        time_stepper=time_stepper,
        solver=solver,
        output=output,
        forcing=float(case_table.get("forcing", 1.0)),
        initial_velocity=float(case_table.get("initial_velocity", 0.0)),
        reference_pressure_gradient=float(case_table.get("reference_pressure_gradient", -1.0)),
        reference_phi_cell=_optional_tuple(case_table, "reference_phi_cell", length=2, cast=int) or (0, 0),
        notes=str(case_table.get("notes", "")),
        dtype=str(case_table.get("dtype", "float64")),
    )

    logging = LoggingSpec(
        enabled=bool(logging_table.get("enabled", True)),
        banner=bool(logging_table.get("banner", True)),
        print_footer=bool(logging_table.get("print_footer", True)),
        flush=bool(logging_table.get("flush", True)),
        step_stride=int(logging_table.get("step_stride", 1)),
    )
    restart_enabled = bool(restart_table.get("enabled", False))
    restart_path = restart_table.get("path")
    restart = RestartSpec(
        enabled=restart_enabled,
        path=None if restart_path is None else (input_path.parent / str(restart_path)).resolve(),
        reset_histories=bool(restart_table.get("reset_histories", True)),
        write_restart=bool(restart_table.get("write_restart", restart_enabled)),
        restart_filename=None
        if restart_table.get("restart_filename") is None
        else str(restart_table["restart_filename"]),
    )
    return RunConfig(
        case=case,
        logging=logging,
        restart=restart,
        input_path=input_path,
    )


@dataclass(frozen=True)
class RestartLogInfo:
    enabled: bool = False
    path: str | None = None
    start_time: float = 0.0
    reset_histories: bool = True


@dataclass(frozen=True)
class SolverStepRecord:
    step_index: int
    time: float
    u_max: float
    mean_velocity: float
    current_max: float
    lorentz_max: float
    residual: float
    potential_residual: float
    potential_iterations: float
    linear_residual: float
    linear_iterations: float
    applied_forcing: float
    courant_like: float
    ohmic_power: float
    volumetric_flow_rate: float
    div_current_max: float
    gauge_residual: float
    interface_current_residual: float
    charge_balance_residual: float = 0.0
    potential_initial_residual: float = 0.0
    linear_initial_residual: float = 0.0


class StreamingSolverLogger:
    def __init__(self, config: LoggingSpec | None = None, *, stream: TextIO | None = None) -> None:
        self.config = config or LoggingSpec()
        self.streams: list[TextIO] = [stream or sys.stdout]
        self._start_time = time.perf_counter()
        self._max_steps: int | None = None
        self._target_final_time: float | None = None

    def add_stream(self, stream: TextIO) -> None:
        self.streams.append(stream)

    def _write(self, line: str = "") -> None:
        for stream in self.streams:
            print(line, file=stream, flush=self.config.flush)

    def emit_header(
        self,
        *,
        case: CaseSpec,
        mesh,
        mode: str,
        potential_solver: str,
        target_mean_velocity: float | None,
        reference_mean_velocity: float | None,
        restart: RestartLogInfo | None = None,
    ) -> None:
        if not self.config.is_enabled():
            return
        self._max_steps = int(case.time_stepper.max_steps)
        self._target_final_time = float(case.time_stepper.t_final)
        if self.config.banner:
            self._write("LMhdX solver")
        solver = getattr(case, "solver", None)
        solver_kind = getattr(solver, "kind", "fully_developed_inductionless")
        self._write(
            f"case={case.name} mode={mode} solver={solver_kind} geometry={case.geometry.kind} "
            f"cells=({mesh.nx},{mesh.ny},{mesh.nz})"
        )
        self._write(
            f"domain=({case.geometry.length:.6e},{case.geometry.width:.6e},{case.geometry.height:.6e}) "
            f"dt={case.time_stepper.dt:.6e} end={case.time_stepper.t_final:.6e} max_steps={self._max_steps}"
        )
        if solver is not None:
            self._write(
                f"linear=solvax_pcg preconditioner={solver.preconditioner} "
                f"coupling_steps={solver.coupling_iterations} coupling_tolerance={solver.coupling_tolerance:.6e}"
            )
        self._write(
            f"potential={potential_solver} max_steps={case.time_stepper.potential_iterations} "
            f"tolerance={case.time_stepper.potential_tolerance}"
        )
        self._write(
            f"field={case.magnetic_field.kind} value={case.magnetic_field.value} forcing={case.forcing:.6e} "
            f"target_mean_velocity={target_mean_velocity} reference_mean_velocity={reference_mean_velocity}"
        )
        if restart is not None and restart.enabled:
            self._write(
                f"restart={restart.path} start={restart.start_time:.6e} reset_histories={restart.reset_histories}"
            )

    def _progress_fraction(self, record: SolverStepRecord) -> float | None:
        candidates: list[float] = []
        if self._max_steps is not None and self._max_steps > 0:
            candidates.append(record.step_index / float(self._max_steps))
        if self._target_final_time is not None and self._target_final_time > 0.0:
            candidates.append(record.time / float(self._target_final_time))
        if not candidates:
            return None
        return min(max(max(candidates), 0.0), 1.0)

    @staticmethod
    def _format_seconds(seconds: float | None) -> str:
        if seconds is None or not math.isfinite(float(seconds)):
            return "unknown"
        total = max(float(seconds), 0.0)
        minutes, secs = divmod(total, 60.0)
        hours, minutes = divmod(minutes, 60.0)
        if hours >= 1.0:
            return f"{int(hours):02d}:{int(minutes):02d}:{secs:04.1f}"
        return f"{int(minutes):02d}:{secs:04.1f}"

    def emit_step(self, record: SolverStepRecord) -> None:
        if not self.config.is_enabled():
            return
        if record.step_index > 1 and (record.step_index - 1) % max(self.config.step_stride, 1) != 0:
            return
        elapsed = time.perf_counter() - self._start_time
        progress = self._progress_fraction(record)
        average_step = elapsed / max(record.step_index, 1)
        estimated_total = elapsed / progress if progress is not None and progress > 0.0 else None
        remaining = estimated_total - elapsed if estimated_total is not None else None
        self._write(
            f"step={record.step_index} time={record.time:.6e} residual={record.residual:.6e} "
            f"max_u={record.u_max:.6e} mean_u={record.mean_velocity:.6e} Q={record.volumetric_flow_rate:.6e}"
        )
        self._write(
            f"potential_residual={record.potential_residual:.6e} potential_steps={int(record.potential_iterations)} "
            f"linear_residual={record.linear_residual:.6e} linear_steps={int(record.linear_iterations)}"
        )
        self._write(
            f"max_j={record.current_max:.6e} max_lorentz={record.lorentz_max:.6e} "
            f"div_j={record.div_current_max:.6e} charge={record.charge_balance_residual:.6e} "
            f"gauge={record.gauge_residual:.6e} interface_current={record.interface_current_residual:.6e}"
        )
        self._write(
            f"forcing={record.applied_forcing:.6e} courant={record.courant_like:.6e} "
            f"ohmic_power={record.ohmic_power:.6e}"
        )
        if progress is None:
            self._write(f"elapsed={elapsed:.3f}s average_step={average_step:.3f}s")
        else:
            self._write(
                f"progress={100.0 * progress:.1f}% elapsed={elapsed:.3f}s average_step={average_step:.3f}s "
                f"remaining={self._format_seconds(remaining)} total={self._format_seconds(estimated_total)}"
            )
        self._write("")

    def emit_footer(self, solution: Solution) -> None:
        if not self.config.is_enabled() or not self.config.print_footer:
            return
        elapsed = time.perf_counter() - self._start_time
        self._write(
            f"completed case={solution.case_name} time={solution.state.time:.6e} "
            f"residual={solution.state.residual:.6e} elapsed={elapsed:.3f}s"
        )


def default_log_path(out_dir: str | Path, case_name: str) -> Path:
    out_dir = Path(out_dir)
    return out_dir / f"{case_name}.log"
