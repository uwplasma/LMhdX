"""One verb, ``solve(problem, solver=None, start=None)``, over the kernels that exist.

:func:`solve` reads a :class:`lmhdx.problem.Problem` (or :class:`~lmhdx.problem.Q2D`) and calls
the kernel it describes, so a concrete problem gives the older entry point's result bit for bit.
``method`` maps to the kernels as follows: ``stokes`` is :func:`lmhdx.steady.solve_compiled`
(:func:`lmhdx.axial.solve_open_duct` on an :class:`~lmhdx.problem.Open` axis,
:func:`lmhdx.pipe.solve_pipe` on a polar grid), ``newton`` the same with advection on, ``euler`` the
projection step (:func:`lmhdx.core3d.advance`, or :func:`lmhdx.thermal.advance_thermal` with
``Problem.thermal``) and ``ifrk4`` :func:`lmhdx.q2d.solve_q2d`. ``auto`` takes ``euler`` if ``steps``
is given, else ``stokes`` or ``newton`` by the advection; ``rk3``, ``march`` and ``implicit`` are
reserved for plan steps H2.2 and H3 and raise.

An unsteady run is a loop of compiled chunks (``Output.every`` steps each; one chunk without an
:class:`~lmhdx.output.Output`), so a run in chunks, or restarted from a record, is the run in one, bit for bit.

:func:`jax.grad` through the leaves of a problem runs the implicit derivative of the steady solve.
The factorizations are built on the host from the primal values, so :func:`jax.jit` is for the
objective with the problem closed over; the exception is a conducting wall's numbers, which the Stokes solve
of a periodic duct takes as arguments (their potential factorization is built in the trace).
"""

from __future__ import annotations

import dataclasses
import functools
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import configure
from .axial import pressure_drop as _pressure_drop
from .axial import solve_open_duct
from .cases import NumericalFailure
from .cases import solve as _solve_model
from .core3d import advance, zero_velocity
from .fields import Uniform, host, is_traced, primal
from .grid import CENTER, Field
from .output import Output, Session, Start, digest, read_record, rejection
from .pipe import PipeProblem, solve_pipe
from .pipe import flow_rate as _pipe_flow_rate
from .problem import Q2D, Open, Problem, channel_kernel, hold_walls, thermal_kernel
from .q2d import Q2DProblem, _setup, evolve_q2d, solve_q2d
from .steady import _MAX_STEPS, solve_compiled, solve_steady_state

__all__ = ["Case", "Solution", "Solver", "solve"]

_RESERVED = {"rk3": "H2.2", "march": "H3", "implicit": "H2.2"}
_METHODS = ("auto", "stokes", "newton", "euler", "ifrk4")


@dataclass(frozen=True)
class Solver:
    """How a problem is solved.

    ``dt`` and ``steps`` are the time step and count of an unsteady run (a steady solve takes ``dt``
    only as the pseudo-step of its preconditioner). ``tol`` is the relative residual of the steady
    solve (kernel default 1e-9; the pipe's 1e-11). ``history`` is the stride of the thermal
    diagnostics (default every step) or of the Q2D frames (default none). ``precision`` is
    ``"state"`` or ``"mixed"``, ``dtype`` ``"float64"`` or ``"float32"``. A heated run is rejected, at its
    start or at the first step past them, when its advective Courant number exceeds ``advective_cfl_limit``
    (default the transport scheme's own) or its buoyancy Courant number ``buoyancy_courant_limit`` (default
    none), or when the start's divergence exceeds ``divergence_tolerance``.
    """

    method: str = "auto"
    dt: float | None = None
    steps: int | None = None
    tol: float | None = None
    precision: str = "state"
    dtype: str = "float64"
    checkpoint: bool = True
    history: int | None = None
    advective_cfl_limit: float | None = None
    buoyancy_courant_limit: float | None = None
    divergence_tolerance: float = 1.0e-10

    def resolve(self, problem) -> str:
        """Return the concrete method for ``problem``, refusing what the kernels cannot do."""
        if self.method in _RESERVED:
            raise NotImplementedError(
                f"method {self.method!r} arrives with plan step {_RESERVED[self.method]}"
            )
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {list(_METHODS)}, got {self.method!r}")
        q2d, method = isinstance(problem, Q2D), self.method
        if method == "auto":
            method = "ifrk4" if q2d else "euler" if self.steps is not None else "stokes"
            method = "newton" if method == "stokes" and not q2d and problem.advection != "off" else method
        allowed = ("ifrk4",) if q2d else ("euler", "stokes" if problem.advection == "off" else "newton")
        if method not in allowed:
            raise ValueError(f"method {method!r} does not fit this problem; use one of {allowed}")
        if method in ("euler", "ifrk4") and (self.dt is None or self.steps is None):
            raise ValueError(f"method {method!r} needs dt and steps")
        return method


class Case(NamedTuple):
    """A problem with its solver, start and output, as :func:`lmhdx.config.load` reads them from a TOML file."""

    problem: object
    solver: Solver | None = None
    start: object = None
    name: str = ""
    output: Output | None = None


class Solution(NamedTuple):
    """The result of :func:`solve`; ``fields`` holds the named arrays and each is also an attribute.

    ``velocity`` is the staggered components; ``pressure``, ``potential``, ``currents``,
    ``temperature`` and ``vorticity`` appear where the run has them. ``pressure_gradient`` is the
    drive of a periodic duct (the body force, or the one that meets ``flow_rate``), ``flow_rate``
    the flux through the first x face, ``residual`` the kernel's residual norm and ``summary`` what an
    :class:`~lmhdx.output.Output` wrote to ``summary.json`` (None without one).
    """

    fields: dict
    pressure_gradient: object = None
    flow_rate: object = None
    residual: object = None
    iterations: object = None
    history: object = None
    summary: object = None

    def __getattr__(self, name):
        try:
            return self.fields[name]
        except KeyError:
            raise AttributeError(name) from None

    def pressure_drop(self, start: float, end: float):
        """The pressure drop between two stations of an open duct."""
        return _pressure_drop(self.pressure, start, end)


def _flux(axial: Field):
    """The volumetric flux through the first x face of an axial velocity."""
    return jnp.sum(axial.data[0] * jnp.asarray(axial.grid.face_areas(0)[0], axial.data.dtype))


@functools.lru_cache(maxsize=16)
def _program(kernel, tolerance: float):
    """One compiled steady solve whose forcing, field scale and wall numbers are arguments, so each can be differentiated.

    The kernel holds the wall pattern only, so a new conductance or thickness reuses the program.
    """

    def run(forcing, scale, walls):
        found = solve_steady_state(
            kernel, forcing=forcing, field_scale=scale, walls=walls, tolerance=tolerance, max_steps=_MAX_STEPS
        )
        return found.velocity, found.pressure, found.potential, found.residual_norm

    return jax.jit(run)


def _steady(problem: Problem, solver: Solver) -> Solution:
    # Only the Stokes solve of a periodic duct differentiates wall numbers: they ride as arguments.
    carried = problem.advection == "off" and not any(isinstance(entry, Open) for entry in problem.bc)
    kernel, forcing, scale, factor, walls = channel_kernel(
        problem, 1.0 if solver.dt is None else solver.dt, solver.precision, carried=carried
    )
    held, walls = (None, walls) if carried else (walls, None)
    rate = problem.flow_rate
    if kernel.open_axis is not None:
        if any(primal(f, "the forcing") for f in problem.forcing):
            raise ValueError("an open duct is driven by its flow_rate, not a forcing")
        found = solve_open_duct(
            kernel, field_scale=scale, tolerance=1.0e-9 if solver.tol is None else solver.tol
        )
        fields = {"velocity": found.velocity, "pressure": found.pressure, "potential": found.potential}
        fields["currents"], residual = found.currents, found.residual_norm
        if is_traced(factor):
            fields, residual = jax.tree.map(lambda f: f * factor, fields), abs(factor) * residual
        found = Solution(fields, None, _flux(fields["velocity"][0]), residual, found.iterations)
        return hold_walls(found, held)
    unit = rate is not None
    if unit and problem.advection != "off":
        raise NotImplementedError(
            "a flow rate sets the drive of a Stokes duct; constant-flux control is plan H2.5"
        )
    if forcing is not None or is_traced(scale) or solver.tol is not None or walls is not None:
        tolerance = 1.0e-9 if solver.tol is None else solver.tol
        velocity, pressure, potential, residual = _program(kernel, tolerance)(
            kernel.forcing if forcing is None else tuple(forcing), scale, walls
        )
    else:
        found = solve_compiled(kernel)
        velocity, pressure, potential, residual = (
            found.velocity,
            found.pressure,
            found.potential,
            found.residual_norm,
        )
    drive = kernel.forcing[0] if forcing is None else forcing[0]
    if unit:
        drive = rate / _flux(velocity[0])
        velocity, pressure, potential = (
            jax.tree.map(lambda f: f * drive, part) for part in (velocity, pressure, potential)
        )
        residual = abs(drive) * residual
    fields = {"velocity": velocity, "pressure": pressure, "potential": potential}
    return hold_walls(Solution(fields, drive, _flux(velocity[0]), residual), held)


def _pipe(problem: Problem, solver: Solver) -> Solution:
    if (problem.density, problem.viscosity, problem.conductivity) != (1.0, 1.0, 1.0) or problem.thermal:
        raise NotImplementedError("a pipe is solved in unit properties, without heat")
    if problem.flow_rate is not None or not isinstance(problem.B, Uniform) or problem.advection != "off":
        raise NotImplementedError("a pipe takes a forcing, a Uniform field and the Stokes limit")
    ends = problem.ends(0)
    pipe = PipeProblem(
        problem.grid,
        float(host(problem.B.strength, "the field strength")),
        float(host(ends[0].conductance, "a wall conductance")),
        float(host(problem.forcing[2], "the forcing")),
    )
    velocity, potential = solve_pipe(pipe, **({} if solver.tol is None else {"tolerance": solver.tol}))
    return Solution(
        {"velocity": (velocity,), "potential": potential}, pipe.forcing, _pipe_flow_rate(velocity)
    )


class _Run(NamedTuple):
    """An unsteady run in chunks: ``advance(state, n)`` takes ``n`` steps and ``fields(state)`` names what to write."""

    state: object
    advance: Callable
    fields: Callable
    grid: object
    header: dict
    limits: dict | None = None
    check: Callable | None = None


def _header(problem, method: str) -> dict:
    """The log's description of a run, as :meth:`lmhdx.output.StreamingSolverLogger.emit_header` takes it."""
    mode = "steady" if method in ("stokes", "newton", "march") else "transient"
    if isinstance(problem, Q2D):
        shape = (*problem.initial_vorticity.shape, 1)
        return dict(mode=mode, solver=method, geometry="periodic plane", cells=shape, domain=(*problem.length, 0.0),
                    potential="spectral", field=("none", None), forcing=0.0)  # fmt: skip
    grid, b = problem.grid, problem.B
    field = (
        ("uniform", tuple(float(b.strength) * d for d in b.direction))
        if isinstance(b, Uniform)
        else (type(b).__name__.lower(), None)
    )
    return dict(
        mode=mode,
        solver=method,
        geometry="polar" if grid.is_polar else "cartesian",
        cells=grid.shape,
        domain=tuple(float(f[-1] - f[0]) for f in grid.faces),
        potential="polar / GMRES" if grid.is_polar else "staggered core / fast diagonalization",
        field=field,
        forcing=float(problem.forcing[0]),
    )


def _series(history) -> dict:
    """The per-step scalars of a chunk's history: its one-dimensional arrays."""
    items = history._asdict() if hasattr(history, "_asdict") else history
    return {name: np.asarray(v) for name, v in items.items() if np.ndim(v) == 1}


def _drive(run: _Run, problem, solver: Solver, output: Output | None, record: Start | None):
    """Take ``solver.steps`` steps in chunks of ``output.every`` (one chunk without an output).

    Each chunk is checked against the run's limits before it is written; the first one past them stops the run,
    which raises :class:`~lmhdx.cases.NumericalFailure` once the rejected state and the summary are written.
    Returns the end state, the history of all chunks and the summary (None without an output).
    """
    identity = _identity(problem, solver) if output or record else None
    state, first = run.state, 0
    if record is not None:
        state, first = read_record(record.path, state, identity if record.mode == "resume" else None)
    if run.check is not None:
        run.check(state)
    steps, session = solver.steps, None
    every = (output.every if output else 0) or steps
    if steps % every or (solver.history and every % solver.history):
        raise ValueError(
            f"steps ({steps}) and the history stride must be multiples of output.every ({every})"
        )
    if output is not None:
        info = None if record is None else {"path": record.path, "mode": record.mode, "step": first}
        session = Session(output, identity, run.grid, solver.dt, first, steps, run.header, run.limits, info)
    histories, rejected = [], None
    try:
        for done in range(first, first + steps, every):
            new, history = run.advance(state, every)
            concrete = not is_traced(new)
            series = _series(history) if concrete else {}
            if run.limits and (rejected := rejection(series, run.limits, done)):
                break
            state, done = new, done + every
            histories.append(history)
            if session:
                session.accept(state, run.fields(state), series, done)
    except NumericalFailure as error:  # a chunk the kernel itself refused (Q2D's gates)
        rejected, new, series = {"step": done + every, "reasons": [str(error)]}, None, {}
    except BaseException:
        if session:
            session.abort()
        raise
    summary = None
    if session:
        if rejected:
            session.reject(new, series, rejected, done + every)
        summary = session.close(method=run.header["solver"])
    if rejected:
        failure = NumericalFailure(f"the run was rejected at step {rejected['step']}: {rejected['reasons']}")
        failure.summary = summary
        raise failure
    history = histories[0] if len(histories) == 1 else jax.tree.map(lambda *p: jnp.concatenate(p), *histories)
    return state, history, summary


def _identity(problem, solver: Solver) -> str:
    """What a record must match to resume: the problem without its initial state, and the solver without its length."""
    if isinstance(problem, Q2D):
        problem = dataclasses.replace(problem, initial_vorticity=None)
    elif problem.thermal is not None:
        problem = problem.replace(thermal=dataclasses.replace(problem.thermal, initial_temperature=0.0))
    return digest(problem, dataclasses.replace(solver, steps=None, history=None))


class _Flow(NamedTuple):
    """What a projection step advances: the staggered velocity, and the last pressure that starts its solve."""

    velocity: tuple
    pressure: Field
    potential: Field


class _Plane(NamedTuple):
    """What a Q2D chunk advances: the dealiased spectrum (the whole state), and the fields it gives."""

    spectrum: object
    vorticity: object
    velocity_x: object
    velocity_y: object


def _q2d(problem: Q2D, solver: Solver, output, record) -> Solution:
    coefficients = dict(viscosity=problem.viscosity, hartmann_friction=problem.hartmann_friction)
    if is_traced(problem):
        if output is not None:
            raise ValueError("an Output writes concrete arrays: solve outside jit and grad")
        omega, ux, uy = evolve_q2d(
            problem.initial_vorticity,
            forcing=problem.forcing,
            length=problem.length,
            dt=solver.dt,
            steps=solver.steps,
            **coefficients,
        )
        return Solution({"vorticity": omega, "velocity": (ux, uy)})
    plane = Q2DProblem(
        problem.initial_vorticity,
        problem.forcing,
        problem.length,
        dt=solver.dt,
        steps=solver.steps,
        history_stride=solver.history or 0,
        **coefficients,
    )
    omega, last = plane.initial_vorticity, []
    spectrum = _setup(omega, plane.forcing, plane.length, plane.viscosity, plane.hartmann_friction, plane.dt)[
        0
    ]

    def advance_q2d(state, steps):
        result = solve_q2d(dataclasses.replace(plane, steps=steps, spectrum=state.spectrum))
        if not result.converged:
            raise NumericalFailure(f"the Q2D run ended {result.status}")
        last.append(result)  # a later chunk's first frame is where the one before ended
        history = {
            "frames": result.vorticity_history[len(last) > 1 :],
            "residual": jnp.asarray([result.residual]),
        }
        return _Plane(result.spectrum, result.vorticity, result.velocity_x, result.velocity_y), history

    fields = lambda s: {"vorticity": s.vorticity, "velocity": (s.velocity_x, s.velocity_y)}  # noqa: E731
    axes = [np.linspace(0.0, length, n + 1) for n, length in zip(omega.shape, problem.length)]
    zero = jnp.zeros_like(omega)
    run = _Run(
        _Plane(spectrum, omega, zero, zero),
        advance_q2d,
        fields,
        (*axes, np.array([0.0, 1.0])),
        _header(problem, "ifrk4"),
    )
    state, history, summary = _drive(run, problem, solver, output, record)
    return Solution(
        fields(state), residual=max(r.residual for r in last), history=history["frames"], summary=summary
    )


def _transient(problem: Problem, solver: Solver, start, output, record) -> Solution:
    kernel, forcing, scale, _, walls = channel_kernel(problem, solver.dt, solver.precision)
    if forcing is not None or is_traced(scale) or kernel.open_axis is not None or float(scale) != 1.0:
        raise NotImplementedError("an unsteady run takes a concrete drive and field, and no open axis")
    viscous = kernel.viscous_factorizations()
    velocity = zero_velocity(kernel) if start in (None, "rest") else start
    cell = Field(jnp.zeros(kernel.grid.shape, velocity[0].dtype), (CENTER,) * 3, kernel.grid)

    def step(state, steps):
        run = advance(
            kernel,
            steps,
            state.velocity,
            viscous=viscous,
            checkpoint=solver.checkpoint,
            pressure=state.pressure,
        )
        history = {"divergence": run.divergence_residual, "kinetic_energy": run.kinetic_energy}
        return _Flow(run.velocity, run.pressure, run.potential), history

    run = _Run(
        _Flow(velocity, cell, cell), step, lambda s: s._asdict(), kernel.grid, _header(problem, "euler")
    )
    state, history, summary = _drive(run, problem, solver, output, record)
    return hold_walls(Solution(state._asdict(), history=history, summary=summary), walls)


def _thermal(problem: Problem, solver: Solver, start, output, record) -> Solution:
    from .thermal import (
        _CELL,
        _SCHEMES,
        ThermalState,
        advance_thermal,
        build_thermal_step,
        initial_thermal_state,
        validate_thermal_inputs,
    )

    kernel, forcing, scale, _, walls = channel_kernel(problem, solver.dt, solver.precision)
    if forcing is not None or is_traced(scale) or float(scale) != 1.0:
        raise NotImplementedError("an unsteady run takes a concrete drive and field")
    heat_problem, thermal = thermal_kernel(problem, kernel), problem.thermal
    grid = kernel.grid
    if isinstance(start, ThermalState):
        state = start
    else:
        velocity = solve_steady_state(kernel).velocity if start == "steady" else None
        state = initial_thermal_state(heat_problem, thermal.initial_temperature, velocity)
    heat = Field(jnp.full(grid.shape, thermal.heat_source, dtype=jnp.result_type(float)), _CELL, grid)
    scheme = _SCHEMES[thermal.advection]
    courant = solver.advective_cfl_limit if thermal.advection != "off" else None
    limits = {
        "advective_cfl": scheme if courant is None else courant,
        "advective_cfl_override": courant is not None and courant != scheme,
        "buoyancy_courant": solver.buoyancy_courant_limit,
    }
    bounds = {"buoyancy_courant_limit": limits["buoyancy_courant"]}
    bounds |= {} if limits["advective_cfl"] is None else {"advective_cfl_limit": limits["advective_cfl"]}

    def check(state):
        if not is_traced(state, heat):
            validate_thermal_inputs(
                heat_problem, state, heat, divergence_tolerance=solver.divergence_tolerance, **bounds
            )

    thermal_step, stride = build_thermal_step(heat_problem, dtype=jnp.result_type(float)), solver.history or 1
    chunk = jax.jit(
        lambda s, steps: advance_thermal(
            thermal_step, s, heat, steps, checkpoint=solver.checkpoint, diagnostics=stride
        ),
        static_argnums=1,
    )
    fields = lambda s: {"velocity": s.velocity, "temperature": s.temperature}  # noqa: E731
    run = _Run(state, chunk, fields, grid, _header(problem, "euler"), limits, check)
    state, history, summary = _drive(run, problem, solver, output, record)
    return hold_walls(Solution(fields(state), history=history, summary=summary), walls)


def _written(solution: Solution, problem, solver: Solver, method: str, output: Output) -> Solution:
    """Write a steady solution: its fields as the one record, the file, the summary and the log."""
    session = Session(output, _identity(problem, solver), problem.grid, 1.0, 0, 0, _header(problem, method))
    session.accept(solution.fields, solution.fields, {}, 0)
    scalars = {
        n: float(getattr(solution, n))
        for n in ("pressure_gradient", "flow_rate", "residual")
        if getattr(solution, n) is not None
    }
    return solution._replace(summary=session.close(method=method, **scalars))


def solve(
    problem: object, solver: Solver | None = None, output: Output | None = None, start: object = None
) -> object:
    """Solve ``problem`` with ``solver``; the older models go to :func:`lmhdx.cases.solve`.

    ``start`` is the initial state of an unsteady run: for a thermal run ``None`` (rest, at the
    problem's initial temperature), ``"steady"`` (the steady flow, temperature at rest) or a
    :class:`lmhdx.thermal.ThermalState`; for a mechanical run a velocity or ``None``; for either
    the path of a record, or a :class:`~lmhdx.output.Start`, which resumes the run that wrote it (bit for bit)
    or imports its state. A steady solve takes none. A :class:`Case` brings its own solver, start and output
    unless they are given.

    With an :class:`~lmhdx.output.Output` the run writes its files (``Solution.summary`` is the summary); a
    steady solve writes its fields once. Without one it does no I/O and ``jax.jit``, ``grad`` and ``vmap``
    apply as before.
    """
    if isinstance(problem, Case):
        case = problem
        problem, solver, output = case.problem, solver or case.solver, output or case.output
        start = case.start if start is None else start
        if output is not None and not output.name:
            output = dataclasses.replace(output, name=case.name)
    if not isinstance(problem, (Problem, Q2D)):
        if solver is not None or start is not None or output is not None:
            raise TypeError(f"{type(problem).__name__} takes no solver, output or start")
        return _solve_model(problem)
    if output is not None and is_traced(problem):
        raise ValueError("an Output writes concrete arrays: solve outside jit and grad")
    if isinstance(start, (str, os.PathLike)) and start not in ("rest", "steady"):
        start = Start(path=os.fspath(start))
    if isinstance(start, Start) and not start.path:
        start = start.state
    record, start = (start, None) if isinstance(start, Start) else (None, start)
    solver = Solver() if solver is None else solver
    method = solver.resolve(problem)
    configure(dtype=solver.dtype)
    if isinstance(problem, Q2D):
        return _q2d(problem, solver, output, record)
    if problem.grid.is_polar:
        if method != "stokes":
            raise NotImplementedError("a pipe is solved in the Stokes limit")
        found = _pipe(problem, solver)
    elif method == "euler":
        return (_thermal if problem.thermal is not None else _transient)(
            problem, solver, start, output, record
        )
    else:
        if record is not None or start is not None or problem.thermal is not None:
            raise ValueError("a steady solve takes no start and no heat transport (give Solver(dt=, steps=))")
        found = _steady(problem, solver)
    return found if output is None else _written(found, problem, solver, method, output)
