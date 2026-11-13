"""TOML schema 3: a table is a constructor call (see docs/reference/cli.md).

``[problem]`` is ``Problem(**table)``, or ``preset = "duct"`` (``channel``, ``fringe_duct``, ``pipe``, ``q2d``)
with the preset's arguments; keys are keyword arguments, nested tables nested objects, and ``kind`` picks the
class where several fit. :func:`load` refuses an unknown key, a missing key, a wrong TOML type or an unknown
kind with the table path; :func:`dumps` writes a problem back as the same text; :func:`upgrade` converts
schema 1 (thermal) and 2 (case) files.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import re
import runpy
import types
import typing
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

import numpy as np

from .cases import NumericalFailure, WallLayer, _load_toml, tomllib
from .fields import Fringe, Func, Table, Uniform
from .grid import Grid, axis
from .output import Output, Start
from .problem import Q2D, Open, Problem, Sides, Thermal, Wall, channel, duct, fringe_duct, pipe, q2d
from .solver import Case, Solver, solve

__all__ = ["dumps", "load", "upgrade"]


_KINDS = {c.__name__.lower(): c for c in (Problem, Q2D, Wall, Sides, Open, Thermal, Uniform)}
_KINDS |= {c.__name__.lower(): c for c in (Fringe, Table, Func, WallLayer, Grid, Solver, Start, Output)}
# Written without a kind: nothing else fits.
_PLAIN = (Problem, Solver, Thermal, Grid, WallLayer, Wall, Output, Start)
_PRESETS = {f.__name__: f for f in (duct, channel, fringe_duct, pipe, q2d)}
_NAMES = {float: "a number", int: "an integer", bool: "a boolean", str: "a string"}
_NAMES |= {tuple: "an array", np.ndarray: "an array"}
_NUMBERS = lambda v: type(v) in (int, float) or isinstance(v, list) and all(map(_NUMBERS, v))  # noqa: E731
_IS = {
    bool: lambda v: isinstance(v, bool),
    int: lambda v: type(v) is int,
    float: lambda v: type(v) in (int, float),
    str: lambda v: isinstance(v, str),
    tuple: lambda v: isinstance(v, list),
    dict: lambda v: isinstance(v, dict),
    np.ndarray: lambda v: isinstance(v, dict) or isinstance(v, list) and _NUMBERS(v),
}


class _Mismatch(ValueError):
    """A value of the wrong TOML type for a hint, which another arm of a union may accept."""


def _at(where, key):
    return f"{where}.{key}" if where else key


def _arms(hint):
    """The alternatives of a union hint, without ``None`` (an absent key is how a field is left out)."""
    union = typing.get_origin(hint) in (typing.Union, types.UnionType)
    return [a for a in typing.get_args(hint) if a is not type(None)] if union else [hint]


@functools.cache
def _allowed(call):
    """The keyword hints and the parameters of a class or function."""
    parameters = inspect.signature(call).parameters
    hints = typing.get_type_hints(call.__init__ if inspect.isclass(call) else call)
    allowed = {}
    for name, p in parameters.items():
        if p.kind is p.VAR_KEYWORD:  # the presets pass the rest to Problem
            allowed |= {k: v for k, v in typing.get_type_hints(Problem.__init__).items() if k != "return"}
        else:
            hint = hints.get(name, object)
            allowed[name] = tuple[hint, ...] if p.kind is p.VAR_POSITIONAL else hint
    return allowed, parameters


def _build(call, table, where, base):
    """``call(**table)``, the table checked against the signature first."""
    allowed, parameters = _allowed(call)
    label = f"[{where}]" if where else "the file"
    if unknown := sorted(set(table) - set(allowed)):
        raise ValueError(f"{label} has unknown keys {unknown}; valid keys are {sorted(allowed)}")
    P = inspect.Parameter
    required = [
        n
        for n, p in parameters.items()
        if p.default is P.empty and p.kind in (P.POSITIONAL_OR_KEYWORD, P.KEYWORD_ONLY)
    ]
    if missing := [n for n in required if n not in table]:
        raise ValueError(f"{label} lacks {missing}")
    args, kwargs = (), {}
    for key, value in table.items():
        value = _convert(allowed[key], value, _at(where, key), base)
        value = str(Path(base) / value) if key in ("path", "dir") else value
        if key in parameters and parameters[key].kind is P.VAR_POSITIONAL:
            args = value
        else:
            kwargs[key] = value
    try:
        return call(*args, **kwargs)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} {error}") from None


def _convert(hint, value, where, base):
    """``value`` as ``hint`` says: a table becomes an object, a list a tuple or an array."""
    arms = _arms(hint)
    classes = [a for a in arms if a in _KINDS.values()]
    if isinstance(value, dict) and classes:
        kind, table = value.get("kind"), {k: v for k, v in value.items() if k != "kind"}
        if kind is not None and _KINDS.get(kind) not in classes:
            raise ValueError(
                f"[{where}] kind must be one of {[c.__name__.lower() for c in classes]}, got {kind!r}"
            )
        fits = (c for c in classes if set(table) <= set(_allowed(c)[0]))
        return _build(_KINDS[kind] if kind is not None else next(fits, classes[0]), table, where, base)
    for arm in arms:
        try:
            return _leaf(arm, value, where, base)
        except _Mismatch as error:  # any other error is of an arm the value fits: it is the one to report
            mismatch = error
    names = " or ".join(dict.fromkeys(_NAMES.get(typing.get_origin(a) or a, "a table") for a in arms))
    raise mismatch if len(arms) == 1 else ValueError(f"{where} must be {names}, got {value!r}")


def _leaf(hint, value, where, base):
    """``value`` against a hint that is not a table: a number, a string, an array, a function."""
    kind, args = typing.get_origin(hint) or hint, typing.get_args(hint)
    if hint is object:
        return np.asarray(value, float) if isinstance(value, list) and _NUMBERS(value) else value
    if hint is Callable:
        return _function(value, where, base)
    if not _IS.get(kind, lambda v: False)(value):
        raise _Mismatch(f"{where} must be {_NAMES.get(kind, 'a table')}, got {value!r}")
    if kind is np.ndarray:  # a face list, or the arguments of axis()
        return _build(axis, value, where, base) if isinstance(value, dict) else np.asarray(value, float)
    if kind is dict and args:
        return {k: _convert(args[1], v, _at(where, k), base) for k, v in value.items()}
    if kind is tuple and not args:
        return _deep(value)
    if kind is tuple:
        hints = (args[0],) * len(value) if args[-1] is Ellipsis else args
        if len(hints) != len(value):
            raise ValueError(f"{where} must have {len(hints)} entries, got {len(value)}")
        return tuple(_convert(h, v, f"{where}[{i}]", base) for i, (h, v) in enumerate(zip(hints, value)))
    return value


def _deep(value):
    return tuple(map(_deep, value)) if isinstance(value, list) else value


def _function(spec, where, base):
    """Import ``"module:function"`` or ``"file.py:function"`` (a file relative to the TOML file)."""
    module, _, name = str(spec).partition(":")
    if not name:
        raise ValueError(f"{where} must be 'module:function' or 'file.py:function', got {spec!r}")
    try:
        if module.endswith(".py"):
            return runpy.run_path(str(Path(base) / module))[name]
        return functools.reduce(getattr, name.split("."), import_module(module))
    except (ImportError, AttributeError, KeyError, OSError) as error:
        raise ValueError(f"{where}: cannot import {spec!r}: {error}") from None


def _override(root, item):
    path, sep, text = item.partition("=")
    *parents, leaf = path.split(".")
    try:
        table = functools.reduce(lambda t, k: t.setdefault(k, {}), parents, root) if sep else 1 / 0
        try:
            table[leaf] = tomllib.loads(f"value = {text}")["value"]
        except tomllib.TOMLDecodeError:  # a bare word is a string: solver.method=march
            table[leaf] = text
    except (AttributeError, TypeError, ZeroDivisionError):
        raise ValueError(f"override {item!r} is not table.key=value") from None


def _file(
    schema_version: int = 3,
    name: str = "",
    units: str = "",
    notes: str = "",
    solver: Solver | None = None,
    start: Start | None = None,
    output: Output | None = None,
):
    if schema_version != 3:
        raise ValueError(f"schema_version {schema_version} is not 3: `lmhdx upgrade FILE` converts the file")
    if start is not None and not start.path:  # a start without a record is the plain "steady", or the default
        start = None if start.state == "rest" else start.state
    return solver, start, name, output


def load(path: str | Path, *overrides: str) -> Case:
    """Read a schema 3 file as a :class:`lmhdx.solver.Case`; ``overrides`` are TOML, ``"solver.steps=100"``.

    ``solve(load(path))`` runs it.
    """
    root, base = _load_toml(path), Path(path).resolve().parent
    for item in overrides:
        _override(root, item)
    table = root.pop("problem", None)
    if table is None:
        raise ValueError("a case needs a [problem] table (older files: `lmhdx upgrade FILE`)")
    if not isinstance(table, dict):
        raise ValueError("problem must be a table")
    preset = table.pop("preset", None)
    if preset is not None and preset not in _PRESETS:
        raise ValueError(f"[problem] preset must be one of {sorted(_PRESETS)}, got {preset!r}")
    if preset:
        problem = _build(_PRESETS[preset], table, "problem", base)
    else:
        problem = _convert(Problem | Q2D, table, "problem", base)
    return Case(problem, *_build(_file, root, "", base))


def _tree(obj):
    """The nested tables, lists and scalars of ``obj``: what :func:`load` reads back."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        tree = {} if type(obj) in _PLAIN else {"kind": type(obj).__name__.lower()}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            try:
                same = f.default is not dataclasses.MISSING and np.shape(value) == np.shape(f.default)
                same = same and bool(np.all(value == f.default))
            except ValueError:  # a ragged nested tuple has no shape
                same = False
            if value is not None and not same:
                tree[f.name] = _tree(value)
        return tree
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        return (int if isinstance(obj, int) else float)(obj)
    if isinstance(obj, (list, tuple)):
        return [_tree(v) for v in obj]
    if callable(obj):
        name = f"{obj.__module__}:{obj.__qualname__}"
        if "<" in name or obj.__module__ == "__main__":
            raise ValueError(f"{name} cannot be imported back: define it in a module or a file.py")
        return name
    try:
        return (
            np.asarray(obj, float).tolist()
            if not isinstance(obj, dict)
            else {str(k): _tree(v) for k, v in obj.items()}
        )
    except (TypeError, ValueError):
        raise ValueError(f"cannot write a {type(obj).__name__} to TOML") from None


def _inline(value):
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_key(k)} = {_inline(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(map(_inline, value)) + "]"
    return json.dumps(value).replace("Infinity", "inf").replace("NaN", "nan")  # a float prints as its repr


def _key(key):
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)


def _toml(tree, prefix=""):
    """Dotted keys, one line each: ``problem.grid.x = [...]``."""
    return "\n".join(
        _toml(v, f"{prefix}{_key(k)}.") if isinstance(v, dict) and v else f"{prefix}{_key(k)} = {_inline(v)}"
        for k, v in tree.items()
    )


def dumps(problem, solver=None, start=None, name: str = "", output=None) -> str:
    """The TOML text of a problem with its solver, start and output, or of a :class:`Case`; :func:`load` reads it back.

    ``start`` is ``"steady"`` or a :class:`~lmhdx.output.Start`. A lambda, a function defined in a script, a
    traced value or a ``ThermalState`` start cannot be written.
    """
    if isinstance(problem, Case):
        problem, solver, start, name, output = problem
    if not (start is None or start == "steady" or isinstance(start, Start)):
        raise ValueError(f"only the start 'steady' or a Start is written, not {start!r}")
    tree = {"schema_version": 3, **({"name": name} if name else {}), "problem": _tree(problem)}
    tree |= {"solver": _tree(solver)} if solver is not None else {}
    tree |= {"start": _tree(Start(start) if isinstance(start, str) else start)} if start else {}
    tree |= {"output": _tree(output)} if output is not None else {}
    return _toml(tree) + "\n"


def _faces(count, half, layer, cells):
    """The wall-resolving faces of a fully developed section, with fewer layer cells on a coarse mesh."""
    for n in range(cells if layer < half else 0, 0, -1):
        try:
            return axis(count, -half, half, layer_thickness=layer, cells_in_layer=n, max_ratio=None)
        except ValueError:
            pass
    return axis(count, -half, half)


def _from_thermal(root):
    """A schema 1 (thermal) file as schema 3 tables, and the settings schema 3 cannot hold yet."""
    flow, heat, run, case = root["flow"], root.get("thermal", {}), root.get("run", {}), root["case"]
    grid, bc = {}, []
    for i, n in enumerate("xyz"):
        spec = root["grid"][n]
        grid[n] = (
            axis(**{"max_ratio": None, "cells_in_layer": 4} | spec).tolist()
            if "layer_thickness" in spec
            else spec
        )
        if flow["conditions"][i] == "periodic":
            bc.append("periodic")
            continue
        c, wall = flow.get("wall_conductance", (0, 0, 0))[i], heat[n]
        lower, upper = [
            ({"conductance": c} if c else {}) | {wall["kind"]: wall.get(e, 0.0)} for e in ("lower", "upper")
        ]
        bc.append(lower if lower == upper else {"kind": "sides", "lower": lower, "upper": upper})
    problem = {
        k: flow[k] for k in ("density", "viscosity", "conductivity", "forcing", "advection") if k in flow
    }
    problem |= {
        "grid": grid,
        "bc": bc,
        "thermal": {k: v for k, v in heat.items() if k not in ("x", "y", "z", "hold_flow")},
    }
    problem |= (
        {"B": {"kind": "uniform", "value": list(flow["magnetic_field"])}} if "magnetic_field" in flow else {}
    )
    tree = {"schema_version": 3, "name": case["name"], "units": case["units"], "problem": problem}
    limits = {
        k: run[k]
        for k in ("advective_cfl_limit", "buoyancy_courant_limit", "divergence_tolerance")
        if k in run
    }
    tree["solver"] = {"method": "euler", "dt": flow["dt"], "steps": run["steps"]} | limits
    out = {"dir": root["output"]["directory"], "every": run.get("chunk")}
    out |= {"paraview": root["output"].get("write_paraview"), "restart": root["output"].get("write_restart")}
    tree["output"] = {k: v for k, v in out.items() if v is not None}
    start = {"state": "steady"} if flow.get("initial_velocity") == "steady" else {}
    tree |= {"start": start | root.get("restart", {})} if start or "restart" in root else {}
    return tree, {"thermal": {k: heat[k] for k in ["hold_flow"] if heat.get(k)}}


def _from_case(root):
    """A schema 2 (case) rectangular duct as schema 3 tables, and the settings schema 3 cannot hold yet."""
    case, geometry, field, steps = (root[k] for k in ("case", "geometry", "magnetic_field", "time_stepper"))
    fluids = {
        (r["conductivity"], r.get("density") or 1.0, r.get("viscosity") or 1.0)
        for r in root["regions"]
        if r["kind"] == "fluid"
    }
    kinds = {b["kind"] for b in root.get("boundary_conditions", [])}
    shape = (geometry["kind"], field["kind"], field.get("ramp_duration", 0), len(fluids))
    if shape != ("rect_duct", "constant", 0, 1) or kinds - {"no_slip", "insulating"}:
        raise ValueError(
            "only a rectangular duct of one fluid with a constant field and insulating walls is converted; "
            "write [problem.bc] and [problem.B] by hand (Wall(conductance=) or Wall(layers=), Table, Func)"
        )
    ((sigma, rho, nu),) = fluids
    halves, b = (0.5 * geometry["width"], 0.5 * geometry["height"]), field["value"]
    along = 0 if abs(b[1]) >= abs(b[2]) else 1
    ha = halves[along] * np.hypot(b[1], b[2]) * np.sqrt(sigma / (rho * nu))
    layers = [halves[along] / (ha if k == along else np.sqrt(ha)) if ha else np.inf for k in (0, 1)]
    y, z = (_faces(geometry[n], h, w, 6).tolist() for n, h, w in zip(("ny", "nz"), halves, layers))
    problem = {"grid": {"x": {"cells": 1, "lower": 0.0, "upper": 1.0}, "y": y, "z": z}}
    problem |= {"B": {"kind": "uniform", "value": list(b)}, "forcing": [case.get("forcing", 1.0), 0.0, 0.0]}
    problem |= {k: v for k, v in zip(("conductivity", "density", "viscosity"), (sigma, rho, nu)) if v != 1.0}
    tree = {"schema_version": 3, "name": case["name"], "problem": problem}
    tree |= {"notes": case["notes"]} if case.get("notes") else {}
    tree["solver"] = {"dt": min(halves) ** 2 / nu}
    if root.get("solver", {}).get("mode") == "transient":
        count = min(steps["max_steps"], int(np.floor(steps["t_final"] / steps["dt"] + 1e-12)))
        tree["solver"] = {"method": "implicit", "dt": steps["dt"], "steps": count}
    notes = {k: root.get(k) for k in ("output", "logging", "restart")}
    return tree, notes | {"case": {k: case[k] for k in ["initial_velocity"] if case.get(k)}}


def upgrade(path: str | Path) -> str:
    """The schema 3 text of a schema 1 (thermal) or schema 2 (case) file.

    Schema 1's ``[output]``, ``[run]`` limits and chunk, and ``[restart]`` become ``[output]``, ``[solver]`` and
    ``[start]``. What schema 3 cannot hold (a held flow, a schema 2 file's output, logging and restart) heads
    the text as comments. A layered duct, a conducting wall and a tabulated or analytic field are refused.
    """
    root = _load_toml(path)
    try:
        if root.get("solver", {}).get("kind") == "thermal":
            tree, notes = _from_thermal(root)
        elif {"case", "geometry", "time_stepper"} <= set(root):
            tree, notes = _from_case(root)
        else:
            done = " (it is schema 3 already)" if "problem" in root else ""
            raise ValueError(f"{path} is not a run configuration of schema 1 (thermal) or 2 (case){done}")
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} is not a valid schema 1 or 2 file: {error!r}") from None
    comments = [
        f"# not carried over: [{n}] " + ", ".join(f"{k} = {v!r}" for k, v in t.items())
        for n, t in notes.items()
        if t
    ]
    return "\n".join(comments + [_toml(tree), ""])


def run(path: str | Path, *overrides: str) -> dict[str, object]:
    """Solve a schema 3 file and return its summary: the ``summary.json`` of its ``[output]``, else its scalars.

    A rejected run returns its summary too (``accepted`` false); a run without an ``[output]`` raises.
    """
    case = load(path, *overrides)
    try:
        solution = solve(case)
    except NumericalFailure as error:
        if error.summary is None:
            raise
        return error.summary
    if solution.summary is not None:
        return solution.summary
    history = getattr(solution.history, "_asdict", lambda: solution.history)()
    summary = {"case": case.name}
    for n in ("pressure_gradient", "flow_rate", "residual", "iterations"):
        summary |= {n: float(getattr(solution, n))} if getattr(solution, n) is not None else {}
    last = {k: float(np.ravel(v)[-1]) for k, v in history.items()} if isinstance(history, dict) else {}
    return summary | ({"last": last} if last else {})
