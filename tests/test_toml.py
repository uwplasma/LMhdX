"""TOML schema 3 (plan H4.3): a table is a constructor call, so TOML and Python build the same objects.

Every preset and field kind is written, read back and compared as a pytree; a file solves bit for bit
as the Python problem does; the old schemas upgrade to files that build the kernels the old loaders built;
and every malformed key, kind and type is refused with the table path in the message.
"""

import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
import pytest

import lmhdx
from lmhdx import Open, Problem, Sides, Solver, Thermal, Uniform, Wall, duct
from lmhdx.cases import WallLayer, load_run_config
from lmhdx.config import dumps, load, upgrade
from lmhdx.fields import Fringe, Func, Table
from lmhdx.fully_developed import channel_problem
from lmhdx.grid import Grid, axis
from lmhdx.io import main
from lmhdx.output import Output, Start
from lmhdx.problem import channel, channel_kernel, fringe_duct, pipe, q2d

pytestmark = pytest.mark.numerical  # the api shard runs in the regression tier

ROOT = Path(__file__).resolve().parents[1]
_SMALL = dict(
    wall_conductance=0.02,
    half_length=1.0,
    upstream=2.0,
    downstream=2.0,
    spacing=1.0,
    cells=8,
    cells_in_layer=2,
)


def tilted(x, y, z, a):
    """A field for the ``Func`` tests: importable as ``test_toml:tilted``."""
    return 0 * x, a[0] * np.tanh(x / a[1]) + a[2], 0 * x


def _same(first, second) -> bool:
    return jax.tree.structure(first) == jax.tree.structure(second) and all(
        np.array_equal(a, b) for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True)
    )


def _write(tmp_path, text, name="case.toml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def _heated() -> Problem:
    """The heated duct of ``examples/heated_duct.toml``, in Python."""
    return Problem(
        Grid(axis(1, 0.0, 1.0), axis(8, -1.0, 1.0), axis(8, -1.0, 1.0)),
        bc=dict(x="periodic", y=Wall(heat_flux=0.0), z=Sides(Wall(heat_flux=0.0), Wall(heat_flux=1.0))),
        B=Uniform(0.0, 10.0, 0.0),
        forcing=(1.0, 0.0, 0.0),
        thermal=Thermal(
            1.0, 0.05, 0.0, expansion_coefficient=0.02, gravity=(0.0, 0.0, -1.0), heat_source=0.5
        ),
    )


_LAYERED = dict(
    y=Wall(layers=(WallLayer("li", 0.5, 0.1, 4),)), z=Wall(layers=(WallLayer("steel", 1e-12, 0.1, 4),))
)
_PROBLEMS = {
    "duct": lambda: duct(20, 24),
    "hunt": lambda: duct(10, 32, wall_conductance=0.05, walls=_LAYERED),
    "channel": lambda: channel(R=450, ha=10, cells=(8, 8, 8)),
    "fringe_duct": lambda: fringe_duct(10.0, **_SMALL),
    "open_inlet": lambda: fringe_duct(10.0, **_SMALL).replace(
        bc=(Open(np.linspace(0, 1, 64).reshape(8, 8)), Wall(0.02), Wall(0.02))
    ),
    "pipe": lambda: pipe(10, 16, 24, wall_conductance=0.02),
    "wall_leaves": lambda: duct(10, 24).replace(
        bc=(
            "periodic",
            Wall(conductivity=2.0, thickness=0.05),
            Sides(Wall(0.02, conducting=True), Wall(0.03)),
        )
    ),
    "q2d": lambda: q2d(shape=(8, 8)),
    "heated": _heated,
    "heated_walls": lambda: duct(10, 24).replace(
        bc=("periodic", Sides(Wall(temperature=1.0), Wall(heat_flux=2.0)), Wall(heat_flux=np.ones((1, 24)))),
        thermal=Thermal(1.0, 0.5, 0.0, gravity=(0, 0, -1), advection="off"),
    ),
    "fringe_field": lambda: duct(10, 24).replace(B=Fringe(10.0, 2.0, 0.5, True)),
    "table_field": lambda: duct(10, 24).replace(B=Table(str(ROOT / "field.npz"), 2.0)),
    "func_field": lambda: duct(10, 24).replace(B=Func(tilted, np.array([10.0, 2.0, 5.0]))),
    "slanted_field": lambda: duct(10, 24).replace(B=Uniform(1.0, 2.0, -2.0), density=2.0, viscosity=0.5),
}


@pytest.mark.parametrize("name", _PROBLEMS)
def test_every_preset_and_field_kind_round_trips(tmp_path, name):
    """dump, load: the same pytree (statics, leaves, treedef), and the text is a fixed point."""
    problem = _PROBLEMS[name]()
    text = dumps(problem)
    loaded = load(_write(tmp_path, text)).problem
    assert type(loaded) is type(problem) and _same(loaded, problem)
    assert dumps(loaded) == text


def test_solver_start_output_and_name_round_trip(tmp_path):
    solver = Solver("euler", 1e-3, 5, 1e-8, "mixed", "float32", False, 3, 0.5, 2.0, 1e-8)
    output = Output(str(tmp_path / "out"), "demo", 5, ("velocity",), False, True, False, True)
    for start in ("steady", Start(path="a.npz", mode="import")):
        text = dumps(duct(20, 24), solver, start, "demo", output)
        case = load(_write(tmp_path, text))
        expected = start if isinstance(start, str) else Start(path=str(tmp_path / "a.npz"), mode="import")
        assert (case.solver, case.start, case.name, case.output) == (solver, expected, "demo", output)
        assert dumps(case) == text.replace("a.npz", str(tmp_path / "a.npz"))
    assert load(_write(tmp_path, dumps(duct(20, 24)))).solver is None
    assert load(_write(tmp_path, dumps(duct(20, 24), output=Output()))).output == Output()
    with pytest.raises(ValueError, match="only the start 'steady'"):
        dumps(duct(20, 24), start=[1.0])
    with pytest.raises(ValueError, match="cannot be imported back"):
        dumps(duct(20, 24).replace(B=Func(lambda x, y, z, a: (x, y, z))))


_PRESET_TABLES = (
    ('preset = "duct"\nha = 20\ncells = 24', lambda: duct(20, 24)),
    (
        'preset = "duct"\nha = 10\ncells = 32\nwall_conductance = 0.05\nwalls = { y = { layers = [{ name = "li", conductivity = 0.5, thickness = 0.1, cells = 4 }] }, z = { layers = [{ name = "steel", conductivity = 1e-12, thickness = 0.1, cells = 4 }] } }',
        _PROBLEMS["hunt"],
    ),
    ('preset = "channel"\nR = 450\nha = 10\ncells = [8, 8, 8]', _PROBLEMS["channel"]),
    (
        'preset = "fringe_duct"\nha = 10\n' + "\n".join(f"{k} = {v}" for k, v in _SMALL.items()),
        _PROBLEMS["fringe_duct"],
    ),
    ('preset = "pipe"\nha = 10\nradial = 16\nazimuthal = 24\nwall_conductance = 0.02', _PROBLEMS["pipe"]),
    ('preset = "q2d"\nshape = [8, 8]', _PROBLEMS["q2d"]),
)


@pytest.mark.parametrize(("table", "build"), _PRESET_TABLES)
def test_a_preset_table_is_the_python_preset(tmp_path, table, build):
    assert _same(load(_write(tmp_path, f"[problem]\n{table}\n")).problem, build())


def test_the_heated_duct_example_is_the_python_problem_and_solves_bit_for_bit():
    case = load(ROOT / "examples/heated_duct.toml")
    assert (
        _same(case.problem, _heated())
        and case.solver == Solver(dt=2.0e-3, steps=40)
        and case.name == "heated_duct"
    )
    found, expected = lmhdx.solve(case), lmhdx.solve(_heated(), Solver(dt=2.0e-3, steps=40))
    assert _same(
        (found.temperature, found.velocity, found.history),
        (expected.temperature, expected.velocity, expected.history),
    )


def test_a_file_solves_as_the_python_problem_does(tmp_path):
    case = load(_write(tmp_path, '[problem]\npreset = "duct"\nha = 20\ncells = 24\n'))
    found, expected = lmhdx.solve(case), lmhdx.solve(duct(20, 24))
    assert _same(
        (found.velocity, found.pressure, found.potential),
        (expected.velocity, expected.pressure, expected.potential),
    )
    assert float(found.residual) == float(expected.residual)


def test_overrides_replace_values_before_anything_is_built(tmp_path):
    path = ROOT / "examples/heated_duct.toml"
    case = load(
        path,
        "solver.steps=7",
        "problem.B.value=[0, 30, 0]",
        "solver.method=euler",
        'problem.thermal.advection="off"',
    )
    assert (case.solver.steps, case.solver.method, case.problem.thermal.advection) == (7, "euler", "off")
    assert case.problem.B.strength == 30.0 and case.problem.B.direction == (0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="not table.key=value"):
        load(path, "solver")
    with pytest.raises(ValueError, match="solver.steps must be an integer"):
        load(path, "solver.steps=2.5")
    with pytest.raises(ValueError, match="not table.key=value"):
        load(path, "name.x=1")


def test_a_func_field_imports_from_a_module_or_a_file(tmp_path):
    (tmp_path / "bfield.py").write_text("def uniform(x, y, z, a):\n    return 0 * x, a[0] + 0 * x, 0 * x\n")
    body = '[problem]\nB = {{ kind = "func", fn = "{}", params = [3, 4] }}\n[problem.grid]\nx = [0, 1]\ny = [-1, 1]\nz = [-1, 1]\n'
    from_file = load(_write(tmp_path, body.format("bfield.py:uniform"))).problem.B
    from_module = load(_write(tmp_path, body.format("test_toml:tilted"))).problem.B
    assert from_file.fn(1.0, 2.0, 3.0, from_file.params)[1] == 3.0 and from_module.fn is tilted
    assert isinstance(from_file.params, np.ndarray) and (tmp_path / "bfield.py").is_file()


def test_relative_paths_resolve_against_the_file(tmp_path):
    case = load(_write(tmp_path, '[problem]\nB = { kind = "table", path = "field.npz" }\n' + _EXPLICIT))
    assert case.problem.B.path == str(tmp_path / "field.npz")


_BASE = '[problem]\npreset = "duct"\nha = 20\ncells = 24\n'
_EXPLICIT = "[problem.grid]\nx = [0, 1]\ny = [-1, 0, 1]\nz = [-1, 0, 1]\n"
_FIELD = (
    "[problem]\n" + _EXPLICIT
)  # the presets set B themselves, so a field of its own needs an explicit problem
_REFUSED = (
    (
        "[problem]\npreset = 'duct'\nha = 20\nbogus = 1\n",
        r"\[problem\] has unknown keys \['bogus'\]; valid keys are \[",
    ),
    ("[problem]\npreset = 'duct'\n", r"\[problem\] lacks \['ha'\]"),
    ("[problem]\npreset = 'duct'\nha = '20'\n", "problem.ha must be a number"),
    ("[problem]\npreset = 'duct'\nha = true\n", "problem.ha must be a number"),
    ("[problem]\npreset = 'duct'\nha = 20\ncells = 24.0\n", "problem.cells must be an integer"),
    ("[problem]\npreset = 'duct'\nha = 20\ncells = true\n", "problem.cells must be an integer"),
    ("[problem]\npreset = 'nope'\n", r"preset must be one of \['channel', 'duct'"),
    ("[problem]\npreset = 'duct'\nha = 20\nkind = 'q2d'\n", "unknown keys"),
    (_BASE + "[problem.walls]\ny = 3\n", "problem.walls.y must be a table or a string"),
    ("[problem]\nforcing = [1, 'a', 0]\n" + _EXPLICIT, r"problem.forcing\[1\] must be a number"),
    ("[problem]\nforcing = [1, 0]\n" + _EXPLICIT, "problem.forcing must have 3 entries, got 2"),
    ("[problem]\nforcing = 1\n" + _EXPLICIT, "problem.forcing must be an array"),
    (
        "[problem]\n" + _EXPLICIT + "[problem.B]\nkind = 'bogus'\n",
        r"\[problem.B\] kind must be one of \['uniform', 'fringe', 'table', 'func'\], got 'bogus'",
    ),
    ("[problem]\n" + _EXPLICIT + "[problem.B]\nkind = 'thermal'\n", "kind must be one of"),
    ("[problem]\n" + _EXPLICIT + "[problem.B]\nkind = 'fringe'\n", r"\[problem.B\] lacks \['strength'\]"),
    (
        "[problem]\n" + _EXPLICIT + "[problem.B]\nkind = 'fringe'\nstrength = 1\nvalue = 2\n",
        r"\[problem.B\] has unknown keys \['value'\]",
    ),
    (
        "[problem]\n" + _EXPLICIT + "[problem.thermal]\nspecific_heat = 1\nbogus = 2\n",
        r"\[problem.thermal\] has unknown keys \['bogus'\]",
    ),
    (
        "[problem]\n" + _EXPLICIT + "[problem.thermal]\nspecific_heat = 1\n",
        r"\[problem.thermal\] lacks \['thermal_conductivity', 'reference_temperature'\]",
    ),
    (
        "[problem]\nbc = { x = 'periodic' }\n" + _EXPLICIT,
        r"\[problem\] bc needs an entry for each axis, missing \['y', 'z'\]",
    ),
    (
        "[problem]\nbc = ['periodic', 'wall', { kind = 'wall', bogus = 1 }]\n" + _EXPLICIT,
        r"\[problem.bc\[2\]\] has unknown keys \['bogus'\]",
    ),
    ("[problem]\nbc = ['periodic', { temperature = 1, heat_flux = 2 }, 'wall']\n" + _EXPLICIT, "not both"),
    ("[problem.grid]\nx = [0, 1]\ny = [-1, 1]\n", r"\[problem.grid\] lacks \['z'\]"),
    (
        "[problem.grid]\nx = { cells = 2, lower = 0, upper = 1, bogus = 1 }\ny = [0, 1]\nz = [0, 1]\n",
        r"\[problem.grid.x\] has unknown keys",
    ),
    ("[problem.grid]\nx = [0, 1]\ny = [1, 0]\nz = [0, 1]\n", "strictly increasing"),
    ("[problem.grid]\nx = [0, 'a']\ny = [0, 1]\nz = [0, 1]\n", "problem.grid.x must be an array"),
    (_BASE + "[solver]\nsteps = 1.5\n", "solver.steps must be an integer"),
    (_BASE + "[solver]\ncheckpoint = 1\n", "solver.checkpoint must be a boolean"),
    (_BASE + "[solver]\nmethod = 3\n", "solver.method must be a string"),
    (_BASE + "[solver]\nbogus = 3\n", r"\[solver\] has unknown keys \['bogus'\]"),
    (_BASE + "[start]\nstate = 'warm'\n", r"\[start\] state is \"rest\" or \"steady\""),
    (_BASE + "[start]\nmode = 'copy'\n", r"\[start\] state is \"rest\" or \"steady\" and mode"),
    (_BASE + "[start]\npath = 'x.npz'\nbogus = 1\n", r"\[start\] has unknown keys \['bogus'\]"),
    (_BASE + "[output]\nevery = -1\n", r"\[output\] every is a number of steps"),
    (_BASE + "[output]\ndir = 3\n", "output.dir must be a string"),
    (_BASE + "[output]\nfields = [3]\n", "output.fields"),
    (_BASE + "[output]\nbogus = 1\n", r"\[output\] has unknown keys \['bogus'\]"),
    ("schema_version = 2\n" + _BASE, "schema_version 2 is not 3"),
    ("schema_version = 3.0\n" + _BASE, "schema_version must be an integer"),
    ("colour = 1\n" + _BASE, r"unknown keys \['colour'\]"),
    ("name = 3\n" + _BASE, "name must be a string"),
    ("units = 'm'\n", "a case needs a \\[problem\\] table"),
    ("problem = 3\n", "problem must be a table"),
    (
        _FIELD + "[problem.B]\nkind = 'func'\nfn = 'no_such_module_xyz:f'\n",
        "cannot import 'no_such_module_xyz:f'",
    ),
    (
        _FIELD + "[problem.B]\nkind = 'func'\nfn = 'nocolon'\n",
        "must be 'module:function' or 'file.py:function'",
    ),
    (_FIELD + "[problem.B]\nkind = 'func'\nfn = 'missing.py:f'\n", "cannot import 'missing.py:f'"),
    (_FIELD + "[problem.B]\nkind = 'func'\nfn = 'test_toml:nothing_here'\n", "cannot import"),
)


@pytest.mark.parametrize(("text", "message"), _REFUSED)
def test_a_malformed_file_is_refused_with_the_table_path(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load(_write(tmp_path, text))


# Schema 2 text of a rectangular duct; the keyword arguments fill the tables.
def _case(
    properties="",
    field="[0.0, 0.0, 20.0]",
    width=2.0,
    height=2.0,
    ny=24,
    nz=24,
    geometry="",
    mode="steady",
    extra="",
):
    return f"""[case]
name = "duct"
forcing = 2.0
{extra}
[geometry]
kind = "rect_duct"
width = {width}
height = {height}
ny = {ny}
nz = {nz}
{geometry}
[magnetic_field]
kind = "constant"
value = {field}
[solver]
mode = "{mode}"
[time_stepper]
dt = 0.01
t_final = 0.1
max_steps = 7
[[regions]]
name = "fluid"
kind = "fluid"
conductivity = 2.0
{properties}
[[boundary_conditions]]
name = "walls"
kind = "no_slip"
[[boundary_conditions]]
name = "electric"
kind = "insulating"
[output]
directory = "out"
"""


@pytest.mark.parametrize(
    ("text", "method"),
    (
        ((ROOT / "examples/hartmann_case.toml").read_text(), "auto"),
        (_case(), "auto"),
        (_case("density = 2.0\nviscosity = 0.25", "[0.0, 3.0, 0.5]", 3.0, 1.0, 20, 36), "auto"),
        (_case(field="[0.0, 0.0, 0.0]"), "auto"),
        (
            _case(properties="viscosity = 0.1", field="[0.0, 15.0, 0.0]", ny=40, nz=40, mode="transient"),
            "implicit",
        ),
    ),
)
def test_a_schema_2_file_upgrades_to_the_kernel_the_old_loader_built(tmp_path, text, method):
    old = _write(tmp_path, text, "old.toml")
    case = load(_write(tmp_path, upgrade(old), "new.toml"))
    # The older loader's kernel has unit drive and scales the solution by the case's forcing.
    expected = dataclasses.replace(
        channel_problem(load_run_config(old).case), dt=case.solver.dt, forcing=case.problem.forcing
    )
    assert channel_kernel(case.problem, case.solver.dt)[0] == expected
    assert case.solver.method == method and upgrade(old).startswith("# not carried over: [output]")
    assert (case.solver.steps, case.solver.dt) == (
        (7, 0.01) if method == "implicit" else (None, case.solver.dt)
    )


def test_the_upgraded_hartmann_example_solves_as_the_older_case_does(tmp_path):
    from lmhdx.fully_developed import solve_fully_developed_fields

    old = ROOT / "examples/hartmann_case.toml"
    case = load(_write(tmp_path, upgrade(old)))
    velocity = lmhdx.solve(case).velocity[0].data
    expected = solve_fully_developed_fields(load_run_config(old).case)[0]
    # The older entry point runs the same kernel in its own program: equal to round-off, 3e-15 here.
    np.testing.assert_allclose(velocity[0], expected, rtol=0, atol=1e-13 * np.abs(expected).max())


@pytest.mark.parametrize(
    ("text", "message"),
    (
        (_case().replace("rect_duct", "layered_duct"), "only a rectangular duct"),
        (_case().replace("constant", "tabulated"), "only a rectangular duct"),
        (_case().replace('kind = "insulating"', 'kind = "conducting_wall"'), "only a rectangular duct"),
        (_case(field="[0, 0, 1]\nramp_duration = 1.0"), "only a rectangular duct"),
        (
            _case(properties="[[regions]]\nname = 'b'\nkind = 'fluid'\nconductivity = 3.0"),
            "only a rectangular duct",
        ),
    ),
)
def test_a_schema_2_file_beyond_the_converter_is_refused(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        upgrade(_write(tmp_path, text))
    assert "# not carried over: [case] initial_velocity = 0.5" in upgrade(
        _write(tmp_path, _case(extra="initial_velocity = 0.5"))
    )


_SCHEMA1 = """
schema_version = 1

[case]
name = "heated_duct"
units = "nondimensional"

[solver]
kind = "thermal"

[grid]
x = { cells = 1, lower = 0.0, upper = 1.0 }
y = { cells = 8, lower = -1.0, upper = 1.0 }
z = { cells = 8, lower = -1.0, upper = 1.0 }

[flow]
conditions = ["periodic", "wall", "wall"]
magnetic_field = [0.0, 10.0, 0.0]
forcing = [1.0, 0.0, 0.0]
dt = 2.0e-3

[thermal]
specific_heat = 1.0
thermal_conductivity = 0.05
reference_temperature = 0.0
expansion_coefficient = 0.02
gravity = [0.0, 0.0, -1.0]
heat_source = 0.5

[thermal.y]
kind = "heat_flux"

[thermal.z]
kind = "heat_flux"
upper = 1.0

[run]
steps = 40
chunk = 10

[output]
directory = "out"
"""


def test_a_schema_1_file_upgrades_to_the_run_it_described(tmp_path):
    text = upgrade(_write(tmp_path, _SCHEMA1, "old.toml"))
    assert "not carried over" not in text
    case = load(_write(tmp_path, text, "new.toml"))
    # The file is the shipped example, plus the run's chunks and directory.
    assert _same(case.problem, load(ROOT / "examples/heated_duct.toml").problem)
    assert case.solver == Solver("euler", 2.0e-3, 40) and case.start is None and case.name == "heated_duct"
    assert case.output == Output(str(tmp_path / "out"), every=10)


_VARIANTS = (
    (  # a steady start
        (("dt = 2.0e-3", 'dt = 2.0e-3\ninitial_velocity = "steady"'),),
        lambda case, here: case.start == "steady",
    ),
    (  # temperature walls
        (('kind = "heat_flux"\nupper = 1.0', 'kind = "temperature"\nlower = 0.5\nupper = 1.0'),),
        lambda case, here: case.problem.bc[2] == Sides(Wall(temperature=0.5), Wall(temperature=1.0)),
    ),
    (  # conducting walls and central transport
        (("dt = 2.0e-3", 'dt = 2.0e-3\nwall_conductance = [0.0, 0.03, 0.03]\nadvection = "central"'),),
        lambda case, here: (
            case.problem.advection == "central"
            and channel_kernel(case.problem)[0].wall_conductance == (0.0, 0.03, 0.03)
        ),
    ),
    (  # a wall-resolving axis
        (
            (
                "cells = 8, lower = -1.0, upper = 1.0 }\nz",
                "cells = 12, lower = -1.0, upper = 1.0, layer_thickness = 0.1, cells_in_layer = 3 }\nz",
            ),
        ),
        lambda case, here: len(case.problem.grid.y) == 13,
    ),
    (  # limits, and a record to resume from
        (
            ("[run]\n", "[run]\nadvective_cfl_limit = 0.5\nbuoyancy_courant_limit = 2.0\n"),
            (
                '[output]\ndirectory = "out"',
                '[restart]\npath = "out/a.npz"\nmode = "import"\n[output]\nwrite_paraview = false\ndirectory = "out"',
            ),
        ),
        lambda case, here: (
            (case.solver.advective_cfl_limit, case.solver.buoyancy_courant_limit) == (0.5, 2.0)
            and case.start == Start(path=str(here / "out/a.npz"), mode="import")
            and not case.output.paraview
        ),
    ),
)


@pytest.mark.parametrize(("changes", "holds"), _VARIANTS)
def test_schema_1_variants_upgrade_to_the_same_kernels(tmp_path, changes, holds):
    text = _SCHEMA1
    for old, new in changes:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    case = load(_write(tmp_path, upgrade(_write(tmp_path, text, "old.toml")), "new.toml"))
    assert holds(case, tmp_path)


def test_a_held_flow_is_not_carried_over(tmp_path):
    text = _SCHEMA1.replace("[thermal]\n", "[thermal]\nhold_flow = true\n")
    assert "# not carried over: [thermal] hold_flow = True" in upgrade(_write(tmp_path, text, "old.toml"))


def test_files_that_are_not_run_configurations_are_refused_by_upgrade():
    skip = {".git", "artifacts", "build", ".venv", "node_modules"}
    files = [p for p in ROOT.rglob("*.toml") if not skip & set(p.relative_to(ROOT).parts)]
    names = {p.name for p in files}
    assert {
        "pyproject.toml",
        "catalog.toml",
        "hunt-ha20.toml",
        "samper-table-i.toml",
        "heated_duct.toml",
    } <= names
    for path in files:
        if path.name == "hartmann_case.toml":
            continue
        with pytest.raises(ValueError, match="not a run configuration"):
            upgrade(path)
    with pytest.raises(ValueError, match=r"schema 3 already"):
        upgrade(ROOT / "examples/heated_duct.toml")


def test_upgrade_reports_a_malformed_old_file(tmp_path):
    with pytest.raises(ValueError, match="not a valid schema 1 or 2 file"):
        upgrade(_write(tmp_path, '[solver]\nkind = "thermal"\n'))


def test_the_command_line_upgrades_and_runs_a_file(tmp_path, capsys):
    old = ROOT / "examples/hartmann_case.toml"
    assert main(["upgrade", str(old)]) == 0
    assert capsys.readouterr().out == upgrade(old)
    path = _write(tmp_path, _BASE + "[solver]\ntol = 1e-7\n")
    assert main([str(path), "solver.tol=1e-8"]) == 0
    summary = json.loads(capsys.readouterr().out)
    expected = lmhdx.solve(duct(20, 24), Solver(tol=1e-8))
    assert summary["pressure_gradient"] == 1.0 and summary["flow_rate"] == float(expected.flow_rate)
    assert summary["residual"] == float(expected.residual)
    with pytest.raises(ValueError, match="overrides apply to schema 3"):
        main([str(old), "solver.tol=1e-8"])


def test_the_wall_leaves_are_leaves_after_a_round_trip(tmp_path):
    """conductivity, thickness and the conducting flag survive TOML, and the numbers stay differentiable."""
    problem = _PROBLEMS["wall_leaves"]()
    loaded = load(_write(tmp_path, dumps(problem))).problem
    walls = loaded.bc[1], loaded.bc[2].lower
    assert (walls[0].conductivity, walls[0].thickness) == (2.0, 0.05) and walls[1].conducting
    assert _same(loaded, problem)
    assert len(jax.tree.leaves(loaded)) == len(jax.tree.leaves(problem)) >= 4
