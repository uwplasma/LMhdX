"""Where a run goes: restart records, ParaView files, a summary and a log (see docs/how_to/restart_and_output.md).

``solve(problem, solver, output=Output("out", every=20))`` runs in compiled chunks of 20 steps. An accepted chunk
writes the record ``{name}_{step:06d}.npz`` (the state, exactly) and the ParaView file ``{name}_{step:06d}.vtr``
(``.vts`` on a polar grid), gathered in ``{name}.pvd``; the first rejected chunk stops the run and writes only
``{name}_{step:06d}_rejected.npz``, which imports but does not resume. ``summary.json`` holds ``accepted``, the
rejection and the last and largest diagnostics; the log is the solver's text (:class:`StreamingSolverLogger`)
beside a JSON line per chunk in ``{name}.jsonl``. A record resumes the run that wrote it bit for bit, because a
step boundary is the whole state; ``Start(path=, mode="import")`` takes only its fields.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import jax
import jax.numpy as jnp
import numpy as np

from .cases import LoggingSpec
from .grid import FACE, Field, Grid
from .ops import cell_average

__all__ = ["Output", "Start", "read_record", "write_record", "write_vtk"]


@dataclass(frozen=True)
class Output:
    """Where a run is written: ``dir`` and the file ``name`` (default the case's, else ``run``); ``every`` steps per chunk (0: one chunk).

    ``fields`` names the arrays drawn in ParaView (None: every field the run has); ``paraview``, ``restart``,
    ``summary`` and ``log`` switch the four kinds of file.
    """

    dir: str = "out"
    name: str = ""
    every: int = 0
    fields: tuple[str, ...] | None = None
    paraview: bool = True
    restart: bool = True
    summary: bool = True
    log: bool = True

    def __post_init__(self):
        if self.every < 0:
            raise ValueError("every is a number of steps, or 0 for one chunk")


@dataclass(frozen=True)
class Start:
    """Where an unsteady run starts: ``state`` ``"rest"`` or the ``"steady"`` flow, or the record at ``path``.

    ``mode`` ``"resume"`` continues the run that wrote the record (it refuses another problem or solver),
    ``"import"`` takes the state of any record of the same fields.
    """

    state: str = "rest"
    path: str | None = None
    mode: str = "resume"

    def __post_init__(self):
        if self.state not in ("rest", "steady") or self.mode not in ("resume", "import"):
            raise ValueError('state is "rest" or "steady" and mode "resume" or "import"')


# Restart records.

_SCHEMA, _VERSION = "lmhdx.record", 1


def digest(*objects) -> str:
    """A hash of problems and solvers: arrays by their bytes, functions by name, classes by their fields."""
    hasher = hashlib.sha256()

    def feed(item):
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            hasher.update(type(item).__name__.encode())
            item = [getattr(item, f.name) for f in dataclasses.fields(item)]
        if isinstance(item, dict):
            item = sorted(item.items())
        if isinstance(item, (list, tuple)):
            hasher.update(b"[")
            for part in item:
                feed(part)
        elif item is None or callable(item):
            hasher.update(f"{getattr(item, '__module__', '')}:{getattr(item, '__qualname__', item)}".encode())
        else:
            array = np.asarray(item)
            hasher.update(f"{array.shape}{array.dtype}".encode() + array.tobytes())

    feed(objects)
    return hasher.hexdigest()


def write_record(path: str | Path, state, identity: str, step: int, **notes) -> Path:
    """Write the leaves of ``state`` exactly, with the ``identity`` of its run, its ``step`` and ``notes`` (JSON)."""
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves = {f"leaf{i}": np.asarray(leaf) for i, leaf in enumerate(jax.tree.leaves(state))}
    meta = {"schema": _SCHEMA, "version": _VERSION, "identity": identity, "step": step}
    np.savez(path, notes=json.dumps(notes, sort_keys=True), **meta, **leaves)
    return path


def read_record(path: str | Path, template, identity: str | None = None):
    """Return ``(state, step)`` of a record, shaped like ``template``; refuse a record that does not fit.

    Every leaf must be finite and have the template's shape and dtype. Given ``identity`` the load is a resume
    and the record must come from that problem and solver; without, it is an import.
    """
    with np.load(path, allow_pickle=False) as record:
        arrays = {name: np.asarray(record[name]) for name in record.files}
    leaves, tree = jax.tree.flatten(template)
    try:
        ok = str(arrays["schema"]) == _SCHEMA and int(arrays["version"]) == _VERSION
        stored = [arrays[f"leaf{i}"] for i in range(len(leaves))]
        step = int(arrays["step"])
    except (KeyError, TypeError, ValueError):
        ok = False
    if not ok or len(arrays) != len(leaves) + 5:
        raise ValueError(f"{path} is not a {_SCHEMA} record of version {_VERSION} for this state")
    if identity is not None and str(arrays["identity"]) != identity:
        raise ValueError(f"{path} was written for another problem or solver (import it with mode='import')")
    for i, (found, wanted) in enumerate(zip(stored, leaves)):
        wanted = np.asarray(wanted)
        if found.shape != wanted.shape or found.dtype != wanted.dtype:
            raise ValueError(
                f"{path}: leaf {i} is {found.dtype}{found.shape}, not {wanted.dtype}{wanted.shape}"
            )
        if not np.all(np.isfinite(found)):
            raise ValueError(f"{path}: leaf {i} is not finite")
    return jax.tree.unflatten(tree, [jnp.asarray(a) for a in stored]), step


# ParaView files.


def _text(array) -> str:
    return " ".join(f"{float(value):.12e}" for value in np.ravel(array))


def write_vtk(path: str | Path, grid, **cells) -> Path:
    """Write named cell arrays as a ParaView file; returns ``path`` with the ``.vtr`` or ``.vts`` suffix.

    ``grid`` is a :class:`~lmhdx.grid.Grid` or the three axes' face coordinates. A Cartesian grid is a
    rectilinear grid (any stretching); a polar grid a structured one, with the points at ``(r cos t, r sin t, z)``.
    Each array is ``(nx, ny, nz)`` or has a trailing component axis of three, and is written x fastest.
    """
    faces, polar = (grid.faces, grid.is_polar) if isinstance(grid, Grid) else (grid, False)
    target = Path(path).with_suffix(".vts" if polar else ".vtr")
    target.parent.mkdir(parents=True, exist_ok=True)
    extent = " ".join(f"0 {np.size(axis) - 1}" for axis in faces)

    def entry(name, data, extra=""):
        return f'<DataArray type="Float64" Name="{name}"{extra} format="ascii">{_text(data)}</DataArray>'

    def cell(name, data):
        data = np.asarray(data)
        vector = data.ndim == 4
        return entry(
            name,
            np.transpose(data, (2, 1, 0, 3) if vector else (2, 1, 0)),
            ' NumberOfComponents="3"' if vector else "",
        )

    if polar:
        r, theta, z = np.meshgrid(*faces, indexing="ij")
        points = np.transpose(np.stack([r * np.cos(theta), r * np.sin(theta), z], -1), (2, 1, 0, 3))
        kind, where = (
            "StructuredGrid",
            "<Points>" + entry("Points", points, ' NumberOfComponents="3"') + "</Points>",
        )
    else:
        kind, where = (
            "RectilinearGrid",
            "<Coordinates>" + "".join(entry(n, a) for n, a in zip("XYZ", faces)) + "</Coordinates>",
        )
    target.write_text(
        f'<?xml version="1.0"?>\n<VTKFile type="{kind}" version="0.1" byte_order="LittleEndian">\n'
        f'<{kind} WholeExtent="{extent}">\n<Piece Extent="{extent}">\n<CellData>\n'
        + "\n".join(cell(name, data) for name, data in cells.items())
        + f"\n</CellData>\n{where}\n</Piece>\n</{kind}>\n</VTKFile>\n"
    )
    return target


def write_pvd(entries: list[tuple[float, str]], out_dir: str | Path, name: str = "series") -> Path:
    """Gather ``(time, file)`` frames of one series into ``{name}.pvd``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.pvd"
    datasets = "\n".join(f'<DataSet timestep="{t:.8f}" group="" part="0" file="{f}"/>' for t, f in entries)
    target.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        f"<Collection>\n{datasets}\n</Collection>\n</VTKFile>\n"
    )
    return target


def _centred(part) -> np.ndarray:
    """A field, or a 2-D array, as an ``(nx, ny, nz)`` array of cell values."""
    if isinstance(part, Field):
        return np.asarray(
            cell_average(part, part.offset.index(FACE)).data if FACE in part.offset else part.data
        )
    return np.asarray(part)[:, :, None]


def cells_of(fields: dict, names=None) -> dict:
    """The ParaView arrays of a run's ``fields``: a scalar per field, a three-component vector per tuple."""
    cells = {}
    for name, value in fields.items():
        parts = value if isinstance(value, (tuple, list)) else (value,)
        if (names is None or name in names) and all(isinstance(p, Field) or np.ndim(p) == 2 for p in parts):
            arrays = [_centred(p) for p in parts]
            cells[name] = (
                arrays[0] if len(arrays) == 1 else np.stack(arrays + [0 * arrays[0]] * (3 - len(arrays)), -1)
            )
    return cells


# The log: the solver's text, kept.


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
        name: str,
        mode: str,
        solver: str,
        geometry: str,
        cells: tuple,
        domain: tuple,
        dt: float,
        end: float,
        max_steps: int,
        potential: str,
        field: tuple,
        forcing: float,
        target_mean_velocity: float | None = None,
        reference_mean_velocity: float | None = None,
        restart: RestartLogInfo | None = None,
    ) -> None:
        if not self.config.is_enabled():
            return
        self._max_steps, self._target_final_time = int(max_steps), float(end)
        if self.config.banner:
            self._write("LMhdX solver")
        self._write(
            f"case={name} mode={mode} solver={solver} geometry={geometry} cells=({cells[0]},{cells[1]},{cells[2]})"
        )
        self._write(
            f"domain=({domain[0]:.6e},{domain[1]:.6e},{domain[2]:.6e}) dt={dt:.6e} end={end:.6e} max_steps={self._max_steps}"
        )
        self._write(f"potential={potential}")
        self._write(
            f"field={field[0]} value={field[1]} forcing={forcing:.6e} "
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

    def emit_footer(self, name: str, final: float, residual: float) -> None:
        if not self.config.is_enabled() or not self.config.print_footer:
            return
        elapsed = time.perf_counter() - self._start_time
        self._write(f"completed case={name} time={final:.6e} residual={residual:.6e} elapsed={elapsed:.3f}s")


# One run's files.

_STEP = {  # a history series to the log's field
    "divergence": "residual",
    "mass_residual": "residual",
    "charge_residual": "charge_balance_residual",
    "joule_power": "ohmic_power",
    "advective_cfl": "courant_like",
}


def rejection(series: dict, limits: dict, done: int) -> dict | None:
    """Return the first sample of a chunk's ``series`` outside the run's ``limits``, its step and reasons, or None.

    Every sample is checked, so a violation inside a chunk is not hidden by a valid last one; a NaN fails.
    """
    if not series:
        return None
    finite = series["finite"] if "finite" in series else np.all([np.isfinite(v) for v in series.values()], 0)
    checks = {"not finite": ~np.asarray(finite, bool)}
    if "backflow_boundary_valid" in series:
        checks["the flow returns through the outlet"] = ~np.asarray(series["backflow_boundary_valid"], bool)
    for name in ("advective_cfl", "buoyancy_courant"):
        if limits.get(name) is not None and name in series:
            checks[f"{name} over {limits[name]}"] = ~(np.asarray(series[name]) <= limits[name])
    failing = {reason: np.flatnonzero(mask) for reason, mask in checks.items()}
    samples = [int(found[0]) for found in failing.values() if found.size]
    if not samples:
        return None
    sample = min(samples)
    values = {
        n: float(np.asarray(series[n])[sample]) for n in ("advective_cfl", "buoyancy_courant") if n in series
    }
    return {
        "step": done + sample + 1,
        "reasons": [r for r, found in failing.items() if sample in found],
        **values,
    }


class Session:
    """The files of one run: fed an accepted or rejected chunk at a time, closed with the summary.

    ``header`` is the log's description of the run (see :meth:`StreamingSolverLogger.emit_header`), ``first``
    the step it starts at, ``grid`` the ParaView grid, ``limits`` and ``restart`` what the summary records.
    """

    def __init__(self, output: Output, identity, grid, dt, first, steps, header, limits=None, restart=None):
        output = dataclasses.replace(output, name=output.name or "run")
        self.output, self.identity, self.grid, self.dt, self.first = output, identity, grid, dt, first
        self.limits, self.restart = limits, restart
        self.dir, self.frames, self.series, self.rejected = Path(output.dir), [], [], None
        self.done, self.time, self.handles, self.log = first, first * dt, [], None
        self.dir.mkdir(parents=True, exist_ok=True)
        if output.log:
            self.log, mode = StreamingSolverLogger(), "a" if first else "w"
            self.handles = [
                open(self.dir / f"{output.name}{suffix}", mode, encoding="utf-8")
                for suffix in (".log", ".jsonl")
            ]
            self.log.add_stream(self.handles[0])
            info = None if restart is None else RestartLogInfo(True, restart["path"], first * dt, False)
            self.log.emit_header(
                name=output.name,
                **header,
                dt=dt,
                end=(first + steps) * dt,
                max_steps=first + steps,
                restart=info,
            )

    def _label(self, step: int) -> Path:
        return self.dir / f"{self.output.name}_{step:06d}"

    def accept(self, state, fields: dict, series: dict, step: int) -> None:
        """Record an accepted chunk that ends at ``step``: its record, its frame and its log lines."""
        o, self.done = self.output, step
        self.time = float(getattr(state, "time", step * self.dt))
        if o.restart:
            write_record(self._label(step), state, self.identity, step)
        if o.paraview:
            self.frames.append(
                (self.time, write_vtk(self._label(step), self.grid, **cells_of(fields, o.fields)).name)
            )
        self.series.append(series)
        if self.log and series:
            last = {name: float(np.asarray(v)[-1]) for name, v in series.items()}
            known = {_STEP[n]: v for n, v in last.items() if n in _STEP}
            speed = [np.max(np.abs(np.asarray(getattr(p, "data", p)))) for p in fields.get("velocity", ())]
            known["u_max"] = float(max(speed, default=math.nan))
            blank = {f.name: 0.0 for f in dataclasses.fields(SolverStepRecord)[2:]}
            self.log.emit_step(SolverStepRecord(step, self.time, **{**blank, **known}))
            self.handles[1].write(json.dumps({"step": step, "time": self.time, **last}) + "\n")
            self.handles[1].flush()

    def reject(self, state, series: dict, rejected: dict, step: int) -> None:
        """Stop at a rejected chunk that would end at ``step``: keep its state as ``{name}_{step:06d}_rejected.npz``."""
        self.rejected = rejected
        self.time = float(getattr(state, "time", self.time))
        if self.output.restart and state is not None:
            write_record(f"{self._label(step)}_rejected", state, "", step, rejected=rejected)
        self.series.append(series)
        if self.log:
            self.log._write(f"rejected at step {rejected['step']}: {'; '.join(rejected['reasons'])}")

    def abort(self) -> None:
        """Close the log of a run that raised something else."""
        for handle in self.handles:
            handle.close()

    def close(self, **info) -> dict:
        """Write the pvd and ``summary.json`` and close the log; return the summary."""
        o = self.output
        if self.frames:
            write_pvd(
                self.frames, self.dir, name=o.name if self.first == 0 else f"{o.name}_from_{self.first:06d}"
            )
        series = [s for s in self.series if s]
        merged = {
            n: np.concatenate([np.asarray(s[n]) for s in series]) for n in (series[0] if series else ())
        }
        summary = {
            "case": o.name,
            "accepted": self.rejected is None,
            "rejected": self.rejected,
            "steps": self.done - self.first,
            "time": self.time,
            "limits": self.limits,
            "restart": self.restart,
            "last": {n: float(v[-1]) for n, v in merged.items()},
            "largest": {n: float(np.max(np.abs(v))) for n, v in merged.items()},
        }
        summary |= info
        if o.summary:
            (self.dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if self.log:
            self.log.emit_footer(o.name, self.time, float(info.get("residual") or math.nan))
            self.abort()
        return summary
