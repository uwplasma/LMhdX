"""TOML schema 3: a table is a constructor call.

``[problem]`` is ``Problem(**table)``, or ``preset = "duct"`` (any of :func:`lmhdx.problem.duct`,
``channel``, ``fringe_duct``, ``pipe``, ``q2d``) with the preset's arguments. Keys are keyword
arguments, nested tables nested objects, and ``kind`` picks the class where several fit (``B`` is
``uniform``, ``fringe``, ``table`` or ``func``; a ``bc`` entry ``wall``, ``sides`` or ``open``).
``[solver]`` is ``Solver(**table)`` and ``[start]`` takes ``state``, ``"rest"`` or ``"steady"``. An axis
of ``[problem.grid]`` is a list of faces or the arguments of :func:`lmhdx.grid.axis`; a ``Func`` field
is ``"module:function"`` or ``"file.py:function"``, and relative paths resolve against the file.

:func:`load` refuses an unknown key, a missing key, a wrong TOML type and an unknown ``kind`` with the
table path in the message, before anything is solved: an integer stands for a number, a float or a
boolean never for an integer. :func:`dumps` writes any problem without callables back as the same
text, and :func:`upgrade` converts a schema 1 (thermal) or schema 2 (case) file.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import re
import types
import typing
from collections.abc import Callable
from importlib import import_module, util
from pathlib import Path

import numpy as np

from .cases import WallLayer, _load_toml, tomllib
from .fields import Fringe, Func, Table, Uniform
from .grid import Grid, axis
from .problem import Q2D, Open, Problem, Sides, Thermal, Wall, channel, duct, fringe_duct, pipe, q2d
from .solver import Case, Solver, solve

__all__ = ["dumps", "load", "upgrade"]

_CLASSES = (Problem, Q2D, Wall, Sides, Open, Thermal, Uniform, Fringe, Table, Func, WallLayer, Grid, Solver)
_KINDS = {cls.__name__.lower(): cls for cls in _CLASSES}
_PLAIN = (Problem, Solver, Thermal, Grid, WallLayer, Wall)  # written without a kind: nothing else fits
_PRESETS = {preset.__name__: preset for preset in (duct, channel, fringe_duct, pipe, q2d)}
_NAMES = {
    float: "a number",
    int: "an integer",
    bool: "a boolean",
    str: "a string",
    tuple: "an array",
    np.ndarray: "an array",
}


class _Mismatch(ValueError):
    """A value of the wrong TOML type for a hint, which another arm of a union may accept."""


def _arms(hint):
    """The alternatives of a union hint, without ``None`` (an absent key is how a field is left out)."""
    union = typing.get_origin(hint) in (typing.Union, types.UnionType)
    return [arm for arm in typing.get_args(hint) if arm is not type(None)] if union else [hint]


@functools.cache
def _allowed(call):
    """Return the keyword hints of ``call`` (a class or function) and its parameters."""
    parameters = inspect.signature(call).parameters
    hints = typing.get_type_hints(call.__init__ if inspect.isclass(call) else call)
    allowed = {}
    for name, parameter in parameters.items():
        if parameter.kind is parameter.VAR_KEYWORD:  # the presets pass the rest to Problem
            allowed |= {k: v for k, v in typing.get_type_hints(Problem.__init__).items() if k != "return"}
        else:
            hint = hints.get(name, object)
            allowed[name] = tuple[hint, ...] if parameter.kind is parameter.VAR_POSITIONAL else hint
    return allowed, parameters


def _build(call, table, where, base):
    """Return ``call(**table)``, the table checked against its signature first."""
    allowed, parameters = _allowed(call)
    if unknown := sorted(set(table) - set(allowed)):
        raise ValueError(f"[{where}] has unknown keys {unknown}; valid keys are {sorted(allowed)}")
    named = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    if missing := [
        n for n, p in parameters.items() if p.default is p.empty and p.kind in named and n not in table
    ]:
        raise ValueError(f"[{where}] lacks {missing}")
    args, kwargs = (), {}
    for key, value in table.items():
        converted = _convert(allowed[key], value, f"{where}.{key}", base)
        if key == "path":
            converted = str(Path(base) / converted)
        if key in parameters and parameters[key].kind is inspect.Parameter.VAR_POSITIONAL:
            args = converted
        else:
            kwargs[key] = converted
    try:
        return call(*args, **kwargs)
    except (TypeError, ValueError) as error:
        raise ValueError(f"[{where}] {error}") from None


def _convert(hint, value, where, base):
    """Return ``value`` as ``hint`` says: a table becomes an object, a list a tuple or an array."""
    arms = _arms(hint)
    classes = [arm for arm in arms if arm in _CLASSES]
    if isinstance(value, dict) and classes:
        kind = value.get("kind")
        if kind is not None and _KINDS.get(kind) not in classes:
            raise ValueError(
                f"[{where}] kind must be one of {[c.__name__.lower() for c in classes]}, got {kind!r}"
            )
        fits = (c for c in classes if set(value) - {"kind"} <= set(_allowed(c)[0]))
        cls = _KINDS[kind] if kind is not None else next(fits, classes[0])
        return _build(cls, {k: v for k, v in value.items() if k != "kind"}, where, base)
    for arm in arms:
        try:
            return _leaf(arm, value, where, base)
        except _Mismatch as error:  # any other error is of an arm the value fits: it is the one to report
            mismatch = error
    names = " or ".join(dict.fromkeys(_NAMES.get(typing.get_origin(a) or a, "a table") for a in arms))
    raise mismatch if len(arms) == 1 else ValueError(f"{where} must be {names}, got {value!r}")


def _numbers(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        or (isinstance(value, list) and all(map(_numbers, value)))
    )


def _leaf(hint, value, where, base):
    """Check ``value`` against a hint that is not a table: a number, a string, an array, a function."""
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    kind = origin or hint
    if hint is object:
        return np.asarray(value, float) if isinstance(value, list) and _numbers(value) else value
    if hint is Callable:
        return _function(value, where, base)
    ok = {
        bool: lambda v: isinstance(v, bool),
        int: lambda v: isinstance(v, int) and not isinstance(v, bool),
        float: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        str: lambda v: isinstance(v, str),
        np.ndarray: lambda v: isinstance(v, dict) or isinstance(v, list) and _numbers(v),
        tuple: lambda v: isinstance(v, list),
        dict: lambda v: isinstance(v, dict),
    }.get(kind, lambda v: False)
    if not ok(value):
        raise _Mismatch(f"{where} must be {_NAMES.get(kind, 'a table')}, got {value!r}")
    if kind is np.ndarray:  # a face list, or the arguments of axis()
        return _build(axis, value, where, base) if isinstance(value, dict) else np.asarray(value, float)
    if kind is dict:
        return {k: _convert(args[1], v, f"{where}.{k}", base) for k, v in value.items()} if args else value
    if kind is tuple:
        if not args:
            return _nest(value)
        hints = (args[0],) * len(value) if args[-1] is Ellipsis else args
        if len(hints) != len(value):
            raise ValueError(f"{where} must have {len(hints)} entries, got {len(value)}")
        return tuple(_convert(h, v, f"{where}[{i}]", base) for i, (h, v) in enumerate(zip(hints, value)))
    return value


def _nest(value):
    return tuple(map(_nest, value)) if isinstance(value, list) else value


def _function(spec, where, base):
    """Import ``"module:function"`` or ``"file.py:function"`` (a file relative to the TOML file)."""
    module, _, name = spec.partition(":") if isinstance(spec, str) else (spec, "", "")
    if not name:
        raise ValueError(f"{where} must be 'module:function' or 'file.py:function', got {spec!r}")
    try:
        if module.endswith(".py"):
            found = util.spec_from_file_location(Path(module).stem, Path(base) / module)
            module = util.module_from_spec(found)
            found.loader.exec_module(module)
        else:
            module = import_module(module)
        for part in name.split("."):
            module = getattr(module, part)
        return module
    except (ImportError, AttributeError, OSError) as error:
        raise ValueError(f"{where}: cannot import {spec!r}: {error}") from None


def _table(root, key):
    if not isinstance(root.get(key, {}), dict):
        raise ValueError(f"{key} must be a table")
    return dict(root.get(key, {}))


def _start(state: str = "rest"):
    """``[start]``: the initial state of an unsteady run, ``"rest"`` or the ``"steady"`` flow."""
    if state not in ("rest", "steady"):
        raise ValueError(f'state is "rest" or "steady", got {state!r}')
    return None if state == "rest" else state


def _override(root, item):
    path, sep, text = item.partition("=")
    try:
        if not sep:
            raise ValueError
        *parents, leaf = path.split(".")
        table = root
        for key in parents:
            table = table.setdefault(key, {})
        try:
            table[leaf] = tomllib.loads(f"value = {text}")["value"]
        except tomllib.TOMLDecodeError:  # a bare word is a string: solver.method=march
            table[leaf] = text
    except (ValueError, AttributeError, TypeError):
        raise ValueError(f"override {item!r} is not table.key=value") from None


def load(path: str | Path, *overrides: str) -> Case:
    """Read a schema 3 file as a :class:`lmhdx.solver.Case`; ``overrides`` are ``"solver.steps=100"``.

    ``solve(load(path))`` runs it. The values of an override are TOML: ``problem.B.value=[0, 30, 0]``.
    """
    root, base = _load_toml(path), Path(path).resolve().parent
    for item in overrides:
        _override(root, item)
    if root.get("schema_version", 3) != 3:
        raise ValueError(
            f"schema_version {root['schema_version']!r} is not 3: convert it with `lmhdx upgrade FILE`"
        )
    keys = ("schema_version", "name", "units", "notes", "problem", "solver", "output", "start")
    if unknown := sorted(set(root) - set(keys)):
        raise ValueError(
            f"the file has unknown keys {unknown}; valid keys are {sorted(keys)} "
            "(a file of schema 1 or 2 is converted with `lmhdx upgrade FILE`)"
        )
    for key, hint in (("schema_version", int), ("name", str), ("units", str), ("notes", str)):
        if key in root:
            _leaf(hint, root[key], key, base)
    if "problem" not in root:
        raise ValueError("a case needs a [problem] table")
    if _table(root, "output"):
        raise ValueError("[output] takes no keys yet: Output arrives with plan step H4.4")
    table = _table(root, "problem")
    preset = table.pop("preset", None)
    if preset is None:
        problem = _convert(Problem | Q2D, table, "problem", base)
    elif preset in _PRESETS:
        problem = _build(_PRESETS[preset], table, "problem", base)
    else:
        raise ValueError(f"[problem] preset must be one of {sorted(_PRESETS)}, got {preset!r}")
    solver = _build(Solver, _table(root, "solver"), "solver", base) if "solver" in root else None
    return Case(problem, solver, _build(_start, _table(root, "start"), "start", base), root.get("name", ""))


def _is_default(field, value):
    try:
        return (
            field.default is not dataclasses.MISSING
            and np.shape(value) == np.shape(field.default)
            and bool(np.all(value == field.default))
        )
    except ValueError:  # a ragged nested tuple has no shape
        return False


def _tree(obj):
    """The nested tables, lists and scalars of ``obj``: what :func:`load` reads back."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        tree = (
            {"kind": type(obj).__name__.lower()} if type(obj) in _CLASSES and type(obj) not in _PLAIN else {}
        )
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            if value is not None and not _is_default(field, value):
                tree[field.name] = _tree(value)
        return tree
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        return obj.item() if hasattr(obj, "item") else obj
    if isinstance(obj, dict):
        return {str(key): _tree(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tree(value) for value in obj]
    if callable(obj):
        name = f"{obj.__module__}:{obj.__qualname__}"
        if "<" in name or obj.__module__ == "__main__":
            raise ValueError(f"{name} cannot be imported back: define it in a module or a file.py")
        return name
    try:
        return np.asarray(obj, float).tolist()
    except (TypeError, ValueError):
        raise ValueError(f"cannot write a {type(obj).__name__} to TOML") from None


def _inline(value):
    if isinstance(value, dict):
        return (
            "{ " + ", ".join(f"{_key(k)} = {_inline(v)}" for k, v in value.items()) + " }" if value else "{}"
        )
    if isinstance(value, list):
        return "[" + ", ".join(map(_inline, value)) + "]"
    if isinstance(value, bool):
        return str(value).lower()
    return json.dumps(value) if isinstance(value, str) else repr(value)


def _key(key):
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)


def _toml(tree, prefix=""):
    scalars = [f"{_key(k)} = {_inline(v)}" for k, v in tree.items() if not isinstance(v, dict)]
    tables = []
    for key, value in tree.items():
        if isinstance(value, dict):
            path = f"{prefix}{_key(key)}"
            header = "" if value and all(isinstance(v, dict) for v in value.values()) else f"[{path}]\n"
            tables.append(header + _toml(value, f"{path}."))
    return "\n\n".join(filter(None, ["\n".join(scalars), *tables]))


def dumps(problem, solver: Solver | None = None, start: str | None = None, name: str = "") -> str:
    """Return the TOML text of a problem, its solver and its start, or of a :class:`Case`.

    :func:`load` of the text gives back an equal pytree. A problem holding a lambda, a function
    defined in a script, a traced value or a ``ThermalState`` start cannot be written.
    """
    if isinstance(problem, Case):
        problem, solver, start, name = problem
    if start not in (None, "steady"):
        raise ValueError(f"only the start 'steady' is written, not {start!r}")
    tree = {"schema_version": 3} | ({"name": name} if name else {}) | {"problem": _tree(problem)}
    tree |= {"solver": _tree(solver)} if solver is not None else {}
    tree |= {"start": {"state": start}} if start else {}
    return _toml(tree) + "\n"


def _faces(count, half, layer, cells):
    """The wall-resolving faces of a fully developed section, with fewer layer cells on a coarse mesh."""
    for n in range(cells if layer < half else 0, 0, -1):
        try:
            return axis(count, -half, half, layer_thickness=layer, cells_in_layer=n, max_ratio=None)
        except ValueError:
            pass
    return axis(count, -half, half)


def _note(name, table):
    """One comment line for the keys of ``table`` that schema 3 cannot hold yet; empty for no keys."""
    return f"[{name}] " + ", ".join(f"{k} = {v!r}" for k, v in table.items()) if table else ""


def _from_thermal(root):
    """A schema 1 (thermal) file as schema 3 tables, and what schema 3 cannot hold yet."""
    flow, heat, run = root["flow"], root.get("thermal", {}), root.get("run", {})
    grid, bc = {}, []
    for index, name in enumerate("xyz"):
        spec = dict(root["grid"][name])
        if "layer_thickness" in spec:
            spec = axis(**{"max_ratio": None, "cells_in_layer": 4} | spec).tolist()
        grid[name] = spec
        if flow["conditions"][index] == "periodic":
            bc.append("periodic")
            continue
        wall, c = heat[name], flow.get("wall_conductance", (0, 0, 0))[index]
        ends = [
            ({"conductance": c} if c else {}) | {wall["kind"]: wall.get(end, 0.0)}
            for end in ("lower", "upper")
        ]
        bc.append(ends[0] if ends[0] == ends[1] else {"kind": "sides", "lower": ends[0], "upper": ends[1]})
    problem = {
        k: flow[k] for k in ("density", "viscosity", "conductivity", "forcing", "advection") if k in flow
    }
    problem |= {
        "grid": grid,
        "bc": bc,
        "thermal": {k: v for k, v in heat.items() if k not in ("x", "y", "z", "hold_flow")},
    }
    if "magnetic_field" in flow:
        problem["B"] = {"kind": "uniform", "value": list(flow["magnetic_field"])}
    tree = {
        "schema_version": 3,
        "name": root["case"]["name"],
        "units": root["case"]["units"],
        "problem": problem,
    }
    tree["solver"] = {"method": "euler", "dt": flow["dt"], "steps": run["steps"]}
    if flow.get("initial_velocity") == "steady":
        tree["start"] = {"state": "steady"}
    dropped = [_note("run", {k: v for k, v in run.items() if k != "steps"})]
    dropped += [_note("thermal", {"hold_flow": True})] * bool(heat.get("hold_flow"))
    return tree, dropped + [_note(k, root.get(k)) for k in ("output", "restart")]


def _from_case(root):
    """A schema 2 (case) rectangular duct as schema 3 tables, and what schema 3 cannot hold yet."""
    case, geometry, field, steps = (
        root["case"],
        root["geometry"],
        root["magnetic_field"],
        root["time_stepper"],
    )
    fluids = {
        (r["conductivity"], r.get("density") or 1.0, r.get("viscosity") or 1.0)
        for r in root["regions"]
        if r["kind"] == "fluid"
    }
    kinds = {b["kind"] for b in root.get("boundary_conditions", [])}
    if (
        geometry["kind"] != "rect_duct"
        or field["kind"] != "constant"
        or field.get("ramp_duration")
        or len(fluids) != 1
        or kinds - {"no_slip", "insulating"}
    ):
        raise ValueError(
            "only a rectangular duct of one fluid with a constant field and insulating walls is converted; "
            "write [problem.bc] and [problem.B] by hand (Wall(conductance=) or Wall(layers=), Table, Func)"
        )
    ((sigma, rho, nu),) = fluids
    halves, b = (0.5 * geometry["width"], 0.5 * geometry["height"]), field["value"]
    peak = np.abs(b[1:])
    along = 0 if peak[0] >= peak[1] else 1
    ha = halves[along] * np.hypot(*peak) * np.sqrt(sigma / (rho * nu))
    layer = [np.inf, np.inf] if not ha else [halves[along] / ha] * 2
    if ha:
        layer[1 - along] = halves[along] / np.sqrt(ha)
    faces = [_faces(geometry[n], h, w, 6).tolist() for n, h, w in zip(("ny", "nz"), halves, layer)]
    grid = {"x": {"cells": 1, "lower": 0.0, "upper": 1.0}, "y": faces[0], "z": faces[1]}
    problem = {
        "grid": grid,
        "B": {"kind": "uniform", "value": list(b)},
        "forcing": [case.get("forcing", 1.0), 0.0, 0.0],
    }
    problem |= {k: v for k, v in zip(("conductivity", "density", "viscosity"), (sigma, rho, nu)) if v != 1.0}
    tree = {"schema_version": 3, "name": case["name"], "problem": problem}
    tree |= {"notes": case["notes"]} if case.get("notes") else {}
    if root.get("solver", {}).get("mode") == "transient":
        count = min(steps["max_steps"], int(np.floor(steps["t_final"] / steps["dt"] + 1e-12)))
        tree["solver"] = {"method": "implicit", "dt": steps["dt"], "steps": count}
    else:
        tree["solver"] = {"dt": min(halves) ** 2 / nu}
    dropped = [_note(k, root.get(k)) for k in ("output", "logging", "restart")]
    return tree, dropped + [
        _note("case", {"initial_velocity": v}) for v in [case.get("initial_velocity")] if v
    ]


def upgrade(path: str | Path) -> str:
    """Return the schema 3 text of a schema 1 (thermal) or schema 2 (case) file.

    What schema 3 cannot hold yet (the output, logging and restart tables and the run limits, until
    plan step H4.4) is listed as comments at the top. A layered duct, a conducting wall and a tabulated or
    analytic field are refused, naming the tables to write by hand.
    """
    root = _load_toml(path)
    try:
        if root.get("solver", {}).get("kind") == "thermal":
            tree, dropped = _from_thermal(root)
        elif {"case", "geometry", "time_stepper"} <= set(root):
            tree, dropped = _from_case(root)
        else:
            done = " (it is schema 3 already)" if "problem" in root else ""
            raise ValueError(f"{path} is not a run configuration of schema 1 (thermal) or 2 (case){done}")
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} is not a valid schema 1 or 2 file: {error!r}") from None
    return "".join(f"# not carried over: {item}\n" for item in filter(None, dropped)) + _toml(tree) + "\n"


def run(path: str | Path, *overrides: str) -> dict[str, object]:
    """Solve a schema 3 file and return its scalars; the command line prints them."""
    case = load(path, *overrides)
    solution = solve(case)
    names = ("pressure_gradient", "flow_rate", "residual", "iterations")
    summary = {"case": case.name} | {
        n: float(v) for n in names if (v := getattr(solution, n, None)) is not None
    }
    history = solution.history._asdict() if hasattr(solution.history, "_asdict") else solution.history
    return summary | (
        {"last": {k: float(np.ravel(v)[-1]) for k, v in history.items()}} if isinstance(history, dict) else {}
    )
