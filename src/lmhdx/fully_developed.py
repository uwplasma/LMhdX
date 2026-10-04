"""Fully developed duct cases solved on the staggered core.

A :class:`~lmhdx.specs.CaseSpec` for a rectangular duct becomes a
:class:`~lmhdx.core3d.ChannelProblem` with one periodic axial cell, solved by the
conjugate-gradient steady solve of :mod:`lmhdx.steady` and reported on the
case's cross-section. That is the route :func:`lmhdx.solve` and
:func:`lmhdx.solve_fully_developed_fields` take.

*Mesh.* ``ny`` and ``nz`` are the fluid cells along ``y`` and ``z``. The faces
follow :func:`lmhdx.core3d.duct_problem`: the walls normal to the field are
clustered to the Hartmann layer ``delta = sqrt(rho nu / sigma) / |B|``, the
others to the side layer ``sqrt(a delta)``, with ``a`` the half-width along the
field, six cells in each layer (``hartmann_layer_cells`` overrides) and the
gentlest stretching that spans the duct. A 2 x 2 duct with unit properties
gets exactly the faces of ``duct_problem(hartmann=Ha, cells=n)``.

*Walls.* An insulating wall is the homogeneous Neumann closure. The conducting
walls of a ``layered_duct`` follow ``geometry.wall_model``. ``"thin"`` makes
each a sheet of conductance ``sigma_w t_w / sigma`` (:mod:`lmhdx.poisson`) with no
cells of its own, which needs equal walls on one axis. ``"resolved"`` gives the
walls of one axis ``wall_cells`` uniform cells of their own conductivity,
insulated outside: one wall, two different walls, or a layer that no boundary
names (``ChannelProblem.wall_layers``); the reported fields cover the fluid. The
default ``"auto"`` is thin where that holds and resolved otherwise. Conducting
walls on both axes are not represented.

*Field.* A constant field is three numbers; an analytic or tabulated one is
sampled at the cell centres as an :class:`~lmhdx.core3d.ImposedField`, and the
layers follow its peak transverse strength. An axial component is kept: it
adds no electromotive force to the axial flow, and on a varying field it can
drive a secondary flow, which the cell-centred solver dropped.

*Drive.* ``forcing`` is the axial force density. With zero forcing and an
``inlet_flow_rate`` boundary, the flow rate is met by scaling the unit-drive
solution, which is exact because the problem is linear in the drive.

The pseudo-time controls of the case (time step, relaxation, potential and
coupling iterations, steady tolerance) do not enter; the steady state is one
preconditioned CG solve to a relative residual of 1e-9. Without
:func:`lmhdx.enable_x64` it runs in float32 to 1e-5, which the solve reaches at
Ha 20 on 32 cells and not at Ha 100 on 48, where it raises; a float32 case
with float64 enabled is solved in float64 and returned in float32.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ._programs import host_array
from .bc import NEUMANN, PERIODIC, BoundaryCondition
from .core3d import ChannelProblem, ImposedField
from .em import lorentz_force, wall_insulated
from .grid import Grid, uniform_faces, wall_resolving_faces
from .mesh import StructuredMesh, generate_rect_duct_mesh_from_faces, sample_tabulated_cross_section_field
from .ops import divergence
from .specs import CaseSpec, Diagnostics, MHDState, Solution, require_finite
from .steady import shared_or_embedded, solve_steady_state

__all__ = [
    "case_mesh",
    "channel_problem",
    "core_applies",
    "solve_fully_developed",
    "solve_fully_developed_fields",
]

# Relative CG tolerance by the precision JAX computes in: float32 floors near 1e-6.
_TOLERANCE = {"float64": 1.0e-9, "float32": 1.0e-5}
_CELLS_IN_LAYER = 6
_SIDES = {"left": (1, 0), "right": (1, 1), "bottom": (2, 0), "top": (2, 1)}
_PAIRS = {"left_right": ("left", "right"), "top_bottom": ("bottom", "top")}
_IGNORED = {"no_slip", "inlet_velocity", "inlet_flow_rate", "outlet_pressure"}


def channel_problem(case: CaseSpec) -> ChannelProblem:
    """Return the staggered-core problem a fully developed case solves, at unit drive.

    Raise ``NotImplementedError`` for what the core does not represent: other
    geometries, thick or mismatched conducting walls, several fluids, or an
    imposed current. A constant field stays three numbers; an analytic or
    tabulated one is sampled at the cell centres as an :class:`ImposedField`.
    """
    _check(case)
    fluid = _fluid(case)
    density, viscosity, conductivity = (
        float(fluid.density or 1.0),
        float(fluid.viscosity or 1.0),
        fluid.conductivity,
    )
    geometry = case.geometry
    halves = (0.5 * geometry.width, 0.5 * geometry.height)
    # The layers follow the strongest transverse field, sampled on a uniform section.
    peak = [
        float(np.max(np.abs(component)))
        for component in _sampled_field(
            case, *(uniform_faces(n, -h, h) for n, h in zip((geometry.ny, geometry.nz), halves))
        )[1:]
    ]
    along = 0 if peak[0] >= peak[1] else 1
    # Ha on the half-width along the field; the layers are a/Ha and a/sqrt(Ha) there.
    hartmann = halves[along] * float(np.hypot(*peak)) * np.sqrt(conductivity / (density * viscosity))
    layers = [np.inf, np.inf]
    if hartmann > 0.0:
        layers = [halves[along] / hartmann] * 2
        layers[1 - along] = halves[along] / np.sqrt(hartmann)
    cells_in_layer = geometry.hartmann_layer_cells or _CELLS_IN_LAYER
    faces = [
        _faces(count, half, layer, cells_in_layer)
        for count, half, layer in zip((geometry.ny, geometry.nz), halves, layers, strict=True)
    ]
    insulating = BoundaryCondition(NEUMANN)
    grid = Grid(uniform_faces(1, 0.0, 1.0), *faces)
    field = _sampled_field(case, *faces)
    if case.magnetic_field.kind == "constant":
        field = tuple(float(component.flat[0]) for component in field)
    else:
        field = ImposedField(grid, tuple(component[None] for component in field))
    return ChannelProblem(
        grid=grid,
        conditions=(BoundaryCondition(PERIODIC), insulating, insulating),
        density=density,
        viscosity=viscosity,
        conductivity=conductivity,
        magnetic_field=field,
        forcing=(1.0, 0.0, 0.0),
        dt=min(halves) ** 2 / viscosity,
        **_walls(case, conductivity),
    )


def core_applies(case: CaseSpec) -> bool:
    """Return whether the staggered core represents ``case``; the other cases keep the cell-centred solve."""
    try:
        channel_problem(case)
    except NotImplementedError:
        return False
    return True


def case_mesh(case: CaseSpec) -> StructuredMesh:
    """Return the fluid cross-section the core solves ``case`` on, as a :class:`StructuredMesh`."""
    grid = channel_problem(case).grid
    return generate_rect_duct_mesh_from_faces(
        y_faces=jnp.asarray(grid.y_faces, dtype=case.dtype),
        z_faces=jnp.asarray(grid.z_faces, dtype=case.dtype),
        length=case.geometry.length,
    )


def solve_fully_developed_fields(
    case: CaseSpec,
    *,
    forcing: float | jax.Array | None = None,
    magnetic_field_scale: float | jax.Array = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return the steady velocity, potential, currents and Lorentz force on :func:`case_mesh`.

    All five are cell-centred ``(ny, nz)`` arrays in the case's dtype: axial
    velocity, potential (zero volume mean), the ``y`` and ``z`` currents
    averaged from their faces, and the axial Lorentz force density.
    A case the core does not represent is solved by the cell-centred route,
    :func:`lmhdx.cases.solve_fully_developed_fields`, on its own mesh.
    ``forcing`` and ``magnetic_field_scale`` are continuous design inputs,
    differentiable through the implicit solve of :func:`lmhdx.steady.solve_steady_state`;
    the case itself is static, so close over it outside :func:`jax.jit`. A
    solve that fails raises :class:`~lmhdx.specs.NumericalFailure` when called
    with concrete inputs and gives nonfinite fields under tracing.
    """
    if not core_applies(case):
        from .cases import solve_fully_developed_fields as cell_centred

        return cell_centred(case, forcing=forcing, magnetic_field_scale=magnetic_field_scale)
    problem = channel_problem(case)
    target = _target_flow_rate(case) if forcing is None else None
    drive = target if target is not None else (case.forcing if forcing is None else forcing)
    fields, _, _ = _driven(problem, target is not None, drive, magnetic_field_scale)
    fields = tuple(value.astype(case.dtype) for value in fields)
    if not any(isinstance(value, jax.core.Tracer) for value in (drive, magnetic_field_scale)):
        require_finite("fully developed solve", velocity=fields[0])
    return fields


def solve_fully_developed(case: CaseSpec, *, logger=None, start_time: float = 0.0) -> Solution:
    """Solve a fully developed case to its steady state on the core and report it as a :class:`Solution`.

    ``status`` is ``"converged"`` with ``residual`` the relative steady
    residual ``||R(u)|| / ||R(0)||``; a solve that fails raises. ``steps`` is
    zero: the solve is one CG call with no outer iterations. The diagnostics
    hold one record.
    """
    problem = channel_problem(case)
    mesh = case_mesh(case)
    target = _target_flow_rate(case)
    if logger is not None:
        mean = None if target is None else target / (case.geometry.width * case.geometry.height)
        logger.emit_header(
            case=case,
            mesh=mesh,
            mode="steady",
            potential_solver="staggered core / fast diagonalization / pcg",
            target_mean_velocity=mean,
            reference_mean_velocity=mean,
            restart=None,
        )
    fields, drive, evidence = _driven(
        problem, target is not None, case.forcing if target is None else target, 1.0
    )
    u, phi, jy, jz, lorentz = (value.astype(case.dtype) for value in fields)
    require_finite("fully developed solve", velocity=u, potential=phi, residual=evidence["residual"])
    areas = jnp.asarray(mesh.dy, dtype=u.dtype)[:, None] * jnp.asarray(mesh.dz, dtype=u.dtype)[None, :]
    flow_rate = jnp.sum(areas * u)
    area = jnp.sum(areas)
    record = {
        "residual_history": evidence["residual"],
        "courant_like": 0.0,
        "ohmic_power": jnp.sum(areas * (jy**2 + jz**2)) / max(problem.conductivity, 1e-300),
        "u_max_history": jnp.max(jnp.abs(u)),
        "mean_velocity_history": flow_rate / area,
        "applied_forcing_history": drive,
        "current_max_history": jnp.max(jnp.hypot(jy, jz)),
        "lorentz_max_history": jnp.max(jnp.abs(lorentz)),
        "volumetric_flow_rate_history": flow_rate,
        "mean_current_magnitude_history": jnp.sum(areas * jnp.hypot(jy, jz)) / area,
        "lorentz_power_history": jnp.sum(areas * u * lorentz),
        "div_current_max_history": evidence["div_current_max"],
    }
    values = {name: float(value) for name, value in jax.device_get(record).items()}
    diagnostics = Diagnostics(
        time_history=jnp.asarray([start_time]),
        **{name: jnp.asarray([value]) for name, value in values.items()},
    )
    state = MHDState(u, phi, jy, jz, lorentz, float(start_time), values["residual_history"])
    solution = Solution(mesh, state, diagnostics, case.name, converged=True, status="converged", steps=0)
    if logger is not None:
        logger.emit_footer(solution)
    return solution


def _driven(problem: ChannelProblem, fixed_flow: bool, drive, field_scale):
    """Scale the unit-drive program's fields by the drive, outside the program.

    ``drive`` is the force density, or the flow rate when ``fixed_flow`` is set.
    The problem is linear in the drive, so the scaling is exact; done outside the
    compiled solve, a derivative in the drive differentiates a product and never
    enters the solve's program, which is then neither linearized nor transposed.
    The relative residual does not depend on the drive, and the current
    divergence scales with it. A field scale that is not traced makes the solve
    a constant of any enclosing trace: it runs then, through the one compiled
    program, so a jitted objective, gradient or tangent in the drive compiles
    only the scaling around it.
    """
    dtype = jnp.result_type(float)
    if isinstance(field_scale, jax.core.Tracer):
        fields, flow_rate, evidence = _compiled(problem)(jnp.asarray(field_scale, dtype=dtype))
    else:
        fields, flow_rate, evidence = _executable(problem, dtype.name)(np.asarray(field_scale, dtype=dtype))
    forcing = jnp.asarray(drive, dtype=dtype)
    if fixed_flow:
        forcing = forcing / flow_rate
    evidence = {**evidence, "div_current_max": jnp.abs(forcing) * evidence["div_current_max"]}
    return tuple(forcing * value for value in fields), forcing, evidence


@functools.lru_cache(maxsize=16)
def _executable(problem: ChannelProblem, dtype: str):
    """The compiled unit-drive program itself, which runs on concrete inputs even inside a trace."""
    return shared_or_embedded(problem, _unit_drive, jax.ShapeDtypeStruct((), dtype))


def _unit_drive(problem: ChannelProblem):
    return _compiled(problem).__wrapped__


@functools.lru_cache(maxsize=16)
def _compiled(problem: ChannelProblem):
    """Compile one steady solve per problem at unit drive, with the field scale as its argument.

    One program replaces the dispatch of every operation from the host, which
    is most of an eager solve's time, cold or warm; :func:`_driven` scales it.
    The fields and the report share the program, so :func:`lmhdx.solve` and
    :func:`solve_fully_developed_fields` agree bit for bit and compile once
    between them, for every case dtype: the fields come back in the precision of
    the solve and the callers cast them.
    """

    def run(field_scale):
        solution = solve_steady_state(
            problem,
            forcing=(1.0, 0.0, 0.0),
            field_scale=field_scale,
            tolerance=_TOLERANCE[jnp.result_type(float).name],
        )
        velocity = solution.velocity
        dy, dz = (host_array(problem.grid, _widths, axis) for axis in (1, 2))
        weights = dy[:, None] * dz[None, :]
        fields, currents = _fields(problem, solution)
        # The solve's own residuals, at rest and at the root: evaluating them again doubled the program.
        scale = solution.initial_residual_norm
        evidence = {
            "residual": solution.residual_norm / jnp.maximum(scale, 1e-300),
            "div_current_max": jnp.max(jnp.abs(divergence(currents).data)),
        }
        return fields, jnp.sum(weights * velocity[0].data[0]), evidence

    return jax.jit(run)


def _widths(grid: Grid, axis: int) -> np.ndarray:
    return grid.widths[axis]


def _fields(problem: ChannelProblem, solution):
    velocity, potential = solution.velocity, solution.potential
    currents, field = solution.currents, solution.magnetic_field
    scalar = problem.scalar_conditions
    closed = tuple(wall_insulated(current, axis, scalar[axis]) for axis, current in enumerate(currents))
    force = lorentz_force(closed, field, scalar)
    current_y, current_z = currents[1].data[0], currents[2].data[0]
    jy = 0.5 * (current_y[:-1] + current_y[1:])
    jz = 0.5 * (current_z[:, :-1] + current_z[:, 1:])
    fields = (velocity[0].data[0], potential.data[0], jy, jz, force[0].data[0])
    return fields, currents


def _sampled_field(case: CaseSpec, y_faces: np.ndarray, z_faces: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return ``(B_x, B_y, B_z)`` of the case at the cell centres of a section, each ``(ny, nz)``."""
    y, z = (0.5 * (faces[1:] + faces[:-1]) for faces in (y_faces, z_faces))
    spec = case.magnetic_field
    if spec.kind == "constant":
        return tuple(np.full((y.size, z.size), float(value)) for value in spec.value)
    if spec.kind == "analytic":
        if spec.fn is None:
            raise ValueError("an analytic magnetic field needs fn")
        sampled = spec.fn(*jnp.meshgrid(jnp.asarray(y), jnp.asarray(z), indexing="ij"))
    elif spec.kind == "tabulated":
        if spec.table_path is None:
            raise ValueError("a tabulated magnetic field needs table_path")
        sampled = sample_tabulated_cross_section_field(
            spec.table_path, y=y[:, None] + 0 * z, z=0 * y[:, None] + z
        )
    else:
        raise ValueError(f"unsupported magnetic field kind {spec.kind!r}")
    sampled = np.asarray(sampled, dtype=np.float64)
    return tuple(np.broadcast_to(sampled[..., axis], (y.size, z.size)) for axis in range(3))


def _faces(count: int, half: float, layer: float, cells_in_layer: int) -> np.ndarray:
    """Wall-resolving faces as in ``duct_problem``; fewer layer cells when the mesh is too coarse."""
    if layer < half:
        for cells in range(cells_in_layer, 0, -1):
            try:
                return wall_resolving_faces(
                    count, -half, half, layer_thickness=layer, cells_in_layer=cells, max_ratio=None
                )
            except ValueError:
                if count % 2:
                    raise NotImplementedError(
                        "the staggered core needs an even cell count along each axis of a duct with a field"
                    ) from None
    return uniform_faces(count, -half, half)


def _fluid(case: CaseSpec):
    fluids = [region for region in case.regions if region.kind == "fluid"]
    if len(fluids) != 1:
        raise NotImplementedError("the staggered core solves one fluid region")
    return fluids[0]


def _check(case: CaseSpec) -> None:
    field = case.magnetic_field
    if case.solver.kind != "fully_developed_inductionless":
        raise ValueError("case must select the fully developed inductionless solver")
    if case.geometry.kind not in {"rect_duct", "layered_duct"}:
        raise NotImplementedError(f"the staggered core does not solve geometry {case.geometry.kind!r}")
    if field.kind == "constant" and field.value is None:
        raise ValueError("a constant magnetic field needs a value")
    if field.ramp_duration > 0.0:
        raise ValueError("steady fields require an unramped magnetic field")
    for boundary in case.boundary_conditions:
        if boundary.kind not in _IGNORED | {"insulating", "conducting_wall"}:
            raise NotImplementedError(f"the staggered core does not impose {boundary.kind!r} boundaries")


def _walls(case: CaseSpec, conductivity: float) -> dict:
    """Return the ``wall_conductance`` or the ``wall_layers`` of the case's conducting walls.

    A side is conducting when a ``conducting_wall`` boundary names it, or when it
    has wall cells that no boundary names and the first solid region conducts
    (the cell-centred solver's fallback); it is insulating when an
    ``insulating`` boundary names it or it has no wall. ``geometry.wall_model``
    chooses the closure: ``"thin"`` makes each wall a sheet of conductance
    ``sigma_w t_w / sigma``, which needs equal walls on an axis; ``"resolved"``
    gives the walls of one axis ``wall_cells`` cells of their own; ``"auto"``
    is thin where that holds and resolved otherwise.
    """
    regions = {region.name: region for region in case.regions}
    geometry, sides = case.geometry, {}
    for boundary in case.boundary_conditions:
        if boundary.kind not in {"insulating", "conducting_wall"}:
            continue
        names = _side_names(boundary)
        if boundary.kind == "insulating":
            for name in names:
                sides.setdefault(name, None)
            continue
        region = regions.get(boundary.region)
        if geometry.kind != "layered_duct" or region is None or not names:
            raise NotImplementedError("a conducting wall needs a layered duct, a solid region and its sides")
        for name in names:
            sides[name] = region
    solids = [region for region in case.regions if region.kind == "solid"]
    walled = geometry.kind == "layered_duct"
    for index, name in enumerate(_SIDES):
        if (
            name not in sides
            and walled
            and geometry.wall_cells[index]
            and solids
            and solids[0].conductivity > 0.0
        ):
            sides[name] = solids[0]
    ends = {}
    for index, name in enumerate(_SIDES):
        region = sides.get(name)
        if region is None or not region.conductivity or not conductivity:
            continue
        thickness, cells = geometry.wall_thickness[index], geometry.wall_cells[index]
        if thickness <= 0.0:
            raise NotImplementedError(f"conducting wall {name!r} has no thickness")
        ends[name] = (region.conductivity / conductivity, thickness, cells)
    axes = {_SIDES[name][0] for name in ends}
    pairs = {axis: [ends.get(name) for name, (index, _) in _SIDES.items() if index == axis] for axis in axes}
    thin = all(
        None not in pair and pair[0][0] * pair[0][1] == pair[1][0] * pair[1][1] for pair in pairs.values()
    )
    model = geometry.wall_model
    if model not in {"auto", "thin", "resolved"}:
        raise ValueError(f"wall_model must be 'auto', 'thin' or 'resolved', got {model!r}")
    if model == "thin" or (model == "auto" and thin):
        if not thin:
            raise NotImplementedError("thin walls need equal conducting walls on both sides of an axis")
        if len(axes) > 1:
            raise NotImplementedError("the thin-wall solve needs one insulating axis")
        conductance = [0.0, 0.0, 0.0]
        for axis, pair in pairs.items():
            conductance[axis] = pair[0][0] * pair[0][1]
        return {"wall_conductance": tuple(conductance)}
    if len(axes) > 1:
        raise NotImplementedError("walls resolved in cells conduct on one axis; the corners are not modelled")
    layers = [None, None, None]
    for axis, pair in pairs.items():
        if any(end is not None and end[2] < 1 for end in pair):
            raise NotImplementedError("a wall resolved in cells needs wall_cells")
        layers[axis] = tuple(None if end is None else (end[0], (end[1] / end[2],) * end[2]) for end in pair)
    return {"wall_layers": tuple(layers)}


def _side_names(boundary) -> tuple[str, ...]:
    """Return the walls a boundary names: ``left_right``, ``top_bottom``, a list, or an axis end."""
    side = (boundary.side or "").lower()
    if side in _PAIRS:
        return _PAIRS[side]
    if side in {"min", "max"} and (boundary.axis or "").lower() in {"y", "z"}:
        return ({"y": ("left", "right"), "z": ("bottom", "top")}[boundary.axis.lower()][side == "max"],)
    names = tuple(part.strip() for part in side.split(",") if part.strip())
    return names if all(name in _SIDES for name in names) else ()


def _target_flow_rate(case: CaseSpec) -> float | None:
    """Return the prescribed flow rate of a case driven by zero forcing."""
    if case.forcing != 0.0:
        return None
    for boundary in case.boundary_conditions:
        if boundary.kind == "inlet_flow_rate" and isinstance(boundary.value, (int, float)):
            return float(boundary.value)
    return None
