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

:func:`jax.grad` through the leaves of a problem runs the implicit derivative of the steady solve.
The factorizations are built on the host from the primal values, so :func:`jax.jit` is for the
objective with the problem closed over; the exception is a conducting wall's numbers, which the Stokes solve
of a periodic duct takes as arguments (their potential factorization is built in the trace).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import configure
from .axial import pressure_drop as _pressure_drop
from .axial import solve_open_duct
from .cases import NumericalFailure
from .cases import solve as _solve_model
from .core3d import advance
from .fields import Uniform, host, is_traced, primal
from .grid import Field
from .pipe import PipeProblem, solve_pipe
from .pipe import flow_rate as _pipe_flow_rate
from .problem import Q2D, Open, Problem, channel_kernel, hold_walls, thermal_kernel
from .q2d import Q2DProblem, evolve_q2d, solve_q2d
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
    ``"state"`` or ``"mixed"``, ``dtype`` ``"float64"`` or ``"float32"``.
    """

    method: str = "auto"
    dt: float | None = None
    steps: int | None = None
    tol: float | None = None
    precision: str = "state"
    dtype: str = "float64"
    checkpoint: bool = True
    history: int | None = None

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
    """A problem with its solver and start, as :func:`lmhdx.config.load` reads them from a TOML file."""

    problem: object
    solver: Solver | None = None
    start: object = None
    name: str = ""


class Solution(NamedTuple):
    """The result of :func:`solve`; ``fields`` holds the named arrays and each is also an attribute.

    ``velocity`` is the staggered components; ``pressure``, ``potential``, ``currents``,
    ``temperature`` and ``vorticity`` appear where the run has them. ``pressure_gradient`` is the
    drive of a periodic duct (the body force, or the one that meets ``flow_rate``), ``flow_rate``
    the flux through the first x face, ``residual`` the kernel's residual norm.
    """

    fields: dict
    pressure_gradient: object = None
    flow_rate: object = None
    residual: object = None
    iterations: object = None
    history: object = None

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


def _q2d(problem: Q2D, solver: Solver) -> Solution:
    coefficients = dict(viscosity=problem.viscosity, hartmann_friction=problem.hartmann_friction)
    if is_traced(problem):
        omega, ux, uy = evolve_q2d(
            problem.initial_vorticity,
            forcing=problem.forcing,
            length=problem.length,
            dt=solver.dt,
            steps=solver.steps,
            **coefficients,
        )
        return Solution({"vorticity": omega, "velocity": (ux, uy)})
    result = solve_q2d(
        Q2DProblem(
            problem.initial_vorticity,
            problem.forcing,
            problem.length,
            dt=solver.dt,
            steps=solver.steps,
            history_stride=solver.history or 0,
            **coefficients,
        )
    )
    if not result.converged:
        raise NumericalFailure(f"the Q2D run ended {result.status}")
    fields = {"vorticity": result.vorticity, "velocity": (result.velocity_x, result.velocity_y)}
    return Solution(fields, residual=result.residual, history=result.vorticity_history)


def _transient(problem: Problem, solver: Solver, start) -> Solution:
    kernel, forcing, scale, _, walls = channel_kernel(problem, solver.dt, solver.precision)
    if forcing is not None or is_traced(scale) or kernel.open_axis is not None or float(scale) != 1.0:
        raise NotImplementedError("an unsteady run takes a concrete drive and field, and no open axis")
    run = advance(
        kernel, solver.steps, start, viscous=kernel.viscous_factorizations(), checkpoint=solver.checkpoint
    )
    fields = {"velocity": run.velocity, "pressure": run.pressure, "potential": run.potential}
    found = Solution(
        fields, history={"divergence": run.divergence_residual, "kinetic_energy": run.kinetic_energy}
    )
    return hold_walls(found, walls)


def _thermal(problem: Problem, solver: Solver, start) -> Solution:
    from .io import _thermal_rejection
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
    limits = {"advective_cfl": _SCHEMES[thermal.advection], "buoyancy_courant": None}
    concrete = not is_traced((state, heat))
    if concrete:
        validate_thermal_inputs(heat_problem, state, heat, divergence_tolerance=1.0e-10)
    step = build_thermal_step(heat_problem, dtype=jnp.result_type(float))
    stride = solver.history or 1
    final, history = jax.jit(
        lambda s: advance_thermal(
            step, s, heat, solver.steps, checkpoint=solver.checkpoint, diagnostics=stride
        )
    )(state)
    if concrete and (rejected := _thermal_rejection(history, limits, 0)) is not None:
        raise NumericalFailure(
            f"the thermal run was rejected at step {rejected['step']}: {rejected['reasons']}"
        )
    fields = {"velocity": final.velocity, "temperature": final.temperature}
    return hold_walls(Solution(fields, history=history), walls)


def solve(problem: object, solver: Solver | None = None, start: object = None) -> object:
    """Solve ``problem`` with ``solver``; the older models go to :func:`lmhdx.cases.solve`.

    ``start`` is the initial state of an unsteady run: for a thermal run ``None`` (rest, at the
    problem's initial temperature), ``"steady"`` (the steady flow, temperature at rest) or a
    :class:`lmhdx.thermal.ThermalState`; for a mechanical run a velocity or ``None``. A steady
    solve takes none. A :class:`Case` brings its own solver and start unless they are given.
    """
    if isinstance(problem, Case):
        problem, solver, start = (
            problem.problem,
            solver or problem.solver,
            problem.start if start is None else start,
        )
    if not isinstance(problem, (Problem, Q2D)):
        if solver is not None or start is not None:
            raise TypeError(f"{type(problem).__name__} takes no solver or start")
        return _solve_model(problem)
    solver = Solver() if solver is None else solver
    method = solver.resolve(problem)
    configure(dtype=solver.dtype)
    if isinstance(problem, Q2D):
        return _q2d(problem, solver)
    if problem.grid.is_polar:
        if method != "stokes":
            raise NotImplementedError("a pipe is solved in the Stokes limit")
        return _pipe(problem, solver)
    if method == "euler":
        return (_thermal if problem.thermal is not None else _transient)(problem, solver, start)
    if start is not None or problem.thermal is not None:
        raise ValueError("a steady solve takes no start and no heat transport (give Solver(dt=, steps=))")
    return _steady(problem, solver)
