"""Fully developed steady and transient inductionless solvers."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from solvax import affine_fixed_point_gmres
from solvax import (
    aitken_relaxation as _solvax_aitken_relaxation,
)
from solvax import (
    anderson_mixing as _solvax_anderson_mixing,
)

from .mesh import StructuredMesh
from .physics import build_material_fields, magnetic_field_components, magnetic_field_from_hartmann
from .solvers import (
    _LINEAR_RESIDUAL_FLOOR,
    _MIN_STRICT_POTENTIAL_COUPLING_SOLVES,
    _POTENTIAL_COUPLING_NORMALIZED_GATE,
    _bounded_time_step_count,
    _build_mesh,
    _cell_metric,
    _compute_current_and_lorentz,
    _concat_history,
    _coupling_potential_tolerance,
    _emit_solver_header,
    _emit_solver_step,
    _enforce_target_mean_velocity,
    _enforce_velocity_bc,
    _face_current_emf_and_lorentz_max,
    _fully_developed_rhs,
    _has_uniform_spacing,
    _initial_solver_state,
    _integral_diagnostics,
    _limited_velocity_update,
    _nested_velocity_tolerance,
    _PotentialSystem,
    _prepare_potential_system,
    _reference_mean_velocity,
    _resolve_potential_solver,
    _solve_potential,
    _solve_velocity_system,
    _target_mean_velocity,
    _velocity_system_coefficients,
)
from .specs import (
    BoundaryCondition,
    CaseSpec,
    Diagnostics,
    GeometrySpec,
    MagneticFieldSpec,
    MHDState,
    OutputSpec,
    RegionSpec,
    RestartLogInfo,
    Solution,
    SolverConfig,
    StreamingSolverLogger,
    TimeStepperConfig,
    require_finite,
)

if TYPE_CHECKING:
    from .core3d import ChannelProblem
    from .q2d import Q2DProblem, Q2DResult
    from .steady import SteadySolution

_STEP_DIAGNOSTIC_NAMES = (
    "u_max_history",
    "mean_velocity_history",
    "applied_forcing_history",
    "residual_history",
    "courant_like",
    "ohmic_power",
    "current_max_history",
    "face_current_max_history",
    "emf_max_history",
    "lorentz_max_history",
    "face_lorentz_max_history",
    "potential_residual_history",
    "potential_iterations_history",
    "linear_residual_history",
    "linear_iterations_history",
    "volumetric_flow_rate_history",
    "mean_current_magnitude_history",
    "lorentz_power_history",
    "div_current_max_history",
    "charge_balance_residual_history",
    "gauge_residual_history",
    "interface_current_residual_history",
)
_VelocitySystem = tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    tuple[jnp.ndarray, ...],
    jnp.ndarray,
]


def _ha_to_b(ha: float, length_scale: float, conductivity: float, density: float, viscosity: float) -> float:
    """Return ``B`` for target ``Ha`` with ``viscosity`` interpreted as ``nu``."""

    return magnetic_field_from_hartmann(
        hartmann=ha,
        length_scale=length_scale,
        conductivity=conductivity,
        density=density,
        kinematic_viscosity=viscosity,
    )


def _wall_conductivity_from_conductance_ratio(
    *,
    wall_conductance_ratio: float,
    fluid_conductivity: float,
    wall_thickness: float,
    hartmann_half_spacing: float,
) -> float:
    if wall_thickness <= 0.0:
        raise ValueError(
            "wall_thickness must be positive when deriving wall conductivity from conductance ratio"
        )
    if hartmann_half_spacing <= 0.0:
        raise ValueError(
            "hartmann_half_spacing must be positive when deriving wall conductivity from conductance ratio"
        )
    return wall_conductance_ratio * fluid_conductivity * hartmann_half_spacing / wall_thickness


def _hunt_short_transient_controls(ha: float) -> TimeStepperConfig:
    if ha <= 20.0:
        return TimeStepperConfig(
            dt=0.002,
            t_final=1.0,
            max_steps=500,
            potential_iterations=400,
            relaxation=0.08,
            velocity_update_limit=2e-3,
        )
    if ha <= 100.0:
        return TimeStepperConfig(
            dt=0.002,
            t_final=1.0,
            max_steps=500,
            potential_iterations=400,
            relaxation=0.1,
            velocity_update_limit=1e-3,
        )
    return TimeStepperConfig(
        dt=0.002,
        t_final=1.0,
        max_steps=500,
        potential_iterations=400,
        relaxation=0.1,
        velocity_update_limit=1e-3,
    )


def _fully_developed_solver(mode: str = "steady") -> SolverConfig:
    return SolverConfig(
        kind="fully_developed_inductionless",
        mode=mode,
        preconditioner="jacobi",
        time_scheme="implicit_euler",
        coupling_iterations=16,
        coupling_tolerance=1e-8,
    )


def make_hartmann_case(
    ha: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 96,
    nz: int = 96,
    conductivity: float = 1.0,
    density: float = 1.0,
    viscosity: float = 1.0,
    output_dir: str | None = None,
    dtype: str = "float64",
) -> CaseSpec:
    """Build an insulating rectangular Hartmann-duct reference case."""

    bmag = _ha_to_b(ha, 0.5 * height, conductivity, density, viscosity)
    anchor = (ny // 2, nz // 2)
    return CaseSpec(
        name=f"hartmann_ha{int(ha)}",
        dtype=dtype,
        geometry=GeometrySpec(kind="rect_duct", width=width, height=height, ny=ny, nz=nz, target_ha=ha),
        regions=(RegionSpec("fluid", "fluid", conductivity, density, viscosity),),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 1.0 * bmag, 0.0)),
        boundary_conditions=(
            BoundaryCondition("walls", "no_slip"),
            BoundaryCondition("electric", "insulating"),
        ),
        time_stepper=TimeStepperConfig(
            dt=0.001, t_final=1.0, max_steps=400, potential_iterations=200, relaxation=0.1
        ),
        solver=_fully_developed_solver(),
        output=OutputSpec(directory=output_dir),
        forcing=1.0,
        reference_pressure_gradient=-1.0,
        reference_phi_cell=anchor,
        notes="Planar Hartmann-like reference configuration for solver smoke tests.",
    )


def make_shercliff_case(
    ha: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 96,
    nz: int = 96,
    conductivity: float = 1.0,
    density: float = 1.0,
    viscosity: float = 1.0,
    output_dir: str | None = None,
    dtype: str = "float64",
) -> CaseSpec:
    """Build an all-insulating rectangular Shercliff-duct case."""

    bmag = _ha_to_b(ha, 0.5 * width, conductivity, density, viscosity)
    anchor = (ny // 2, nz // 2)
    return CaseSpec(
        name=f"shercliff_ha{int(ha)}",
        dtype=dtype,
        geometry=GeometrySpec(kind="rect_duct", width=width, height=height, ny=ny, nz=nz, target_ha=ha),
        regions=(RegionSpec("fluid", "fluid", conductivity, density, viscosity),),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 1.0 * bmag, 0.0)),
        boundary_conditions=(
            BoundaryCondition("walls", "no_slip"),
            BoundaryCondition("electric", "insulating"),
        ),
        time_stepper=TimeStepperConfig(
            dt=0.001, t_final=1.5, max_steps=400, potential_iterations=225, relaxation=0.1
        ),
        solver=_fully_developed_solver(),
        output=OutputSpec(directory=output_dir),
        forcing=1.0,
        reference_pressure_gradient=-1.0,
        reference_phi_cell=anchor,
        notes="All-insulating Shercliff-style duct. Analytical validation hooks are staged through the benchmark and validation utilities.",
    )


def make_hunt_case(
    ha: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 72,
    nz: int = 72,
    wall_cells: int = 8,
    wall_thickness: float = 0.1,
    insulator_cells: int | None = None,
    insulator_thickness: float | None = None,
    fluid_conductivity: float = 1.0,
    wall_conductance_ratio: float = 0.05,
    wall_conductivity: float | None = None,
    insulator_conductivity: float | None = None,
    insulator_conductivity_ratio: float = 1e-12,
    density: float = 1.0,
    viscosity: float = 1.0,
    output_dir: str | None = None,
    dtype: str = "float64",
) -> CaseSpec:
    """Build a Hunt duct with conducting Hartmann and insulating side walls."""

    bmag = _ha_to_b(ha, 0.5 * width, fluid_conductivity, density, viscosity)
    if wall_conductivity is None:
        wall_conductivity = _wall_conductivity_from_conductance_ratio(
            wall_conductance_ratio=wall_conductance_ratio,
            fluid_conductivity=fluid_conductivity,
            wall_thickness=wall_thickness,
            hartmann_half_spacing=0.5 * height,
        )
    if insulator_cells is None:
        insulator_cells = wall_cells
    if insulator_thickness is None:
        insulator_thickness = wall_thickness
    if insulator_conductivity is None:
        insulator_conductivity = fluid_conductivity * insulator_conductivity_ratio
    anchor = ((ny + 2 * insulator_cells) // 2, (nz + 2 * wall_cells) // 2)
    controls = _hunt_short_transient_controls(ha)
    return CaseSpec(
        name=f"hunt_ha{int(ha)}",
        dtype=dtype,
        geometry=GeometrySpec(
            kind="layered_duct",
            width=width,
            height=height,
            ny=ny,
            nz=nz,
            wall_thickness=(insulator_thickness, insulator_thickness, wall_thickness, wall_thickness),
            wall_cells=(insulator_cells, insulator_cells, wall_cells, wall_cells),
            target_ha=ha,
        ),
        regions=(
            RegionSpec("fluid", "fluid", fluid_conductivity, density, viscosity),
            RegionSpec("conducting_wall", "solid", wall_conductivity, density, viscosity, wall_thickness),
            RegionSpec(
                "insulating_wall", "solid", insulator_conductivity, density, viscosity, insulator_thickness
            ),
        ),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 1.0 * bmag, 0.0)),
        boundary_conditions=(
            BoundaryCondition("walls", "no_slip"),
            BoundaryCondition(
                "conducting_hartmann_walls", "conducting_wall", region="conducting_wall", side="left_right"
            ),
            BoundaryCondition(
                "insulating_side_walls", "insulating", region="insulating_wall", side="top_bottom"
            ),
        ),
        time_stepper=controls,
        solver=_fully_developed_solver(),
        output=OutputSpec(directory=output_dir),
        forcing=1.0,
        reference_pressure_gradient=-1.0,
        reference_phi_cell=anchor,
        notes=(
            "Hunt-style duct with explicit conducting Hartmann-wall layers and insulating side-wall layers. "
            f"Default wall conductance ratio c={wall_conductance_ratio:g}."
        ),
    )


def _fully_developed_case_step(
    *,
    case: CaseSpec,
    mesh: StructuredMesh,
    materials,
    u_previous: jnp.ndarray,
    step_time: float,
    potential_solver: str,
    target_mean_velocity: float | None,
    preconditioner: str,
    coupling_iterations: int,
    coupling_tolerance: float,
    phi_previous: jnp.ndarray | None = None,
    velocity_system: _VelocitySystem | None = None,
    potential_system: _PotentialSystem | None = None,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    if velocity_system is None:
        velocity_system = _prepare_fully_developed_velocity_system(case, mesh, materials, step_time)
    by, bz, magnetic_reaction, velocity_coefficients, cell_metric = velocity_system
    forcing = jnp.asarray(case.forcing, dtype=by.dtype)
    fluid_mask = materials.fluid_mask
    active_mask = fluid_mask
    u_iter = u_previous
    phi_iter = jnp.zeros_like(u_previous) if phi_previous is None else phi_previous
    dt = case.time_stepper.dt
    fluid_weight = jnp.where(fluid_mask, cell_metric, 0.0)
    fluid_total_weight = jnp.maximum(jnp.sum(fluid_weight), 1e-20)
    velocity_residual = jnp.asarray(jnp.inf, dtype=u_previous.dtype)
    potential_residual = jnp.asarray(jnp.inf, dtype=u_previous.dtype)
    potential_iteration_count = jnp.asarray(0, dtype=jnp.int32)
    potential_initial_residual = jnp.asarray(0.0, dtype=u_previous.dtype)
    linear_iteration_count = jnp.asarray(0, dtype=jnp.int32)
    linear_residual = jnp.asarray(jnp.inf, dtype=u_previous.dtype)
    linear_initial_residual = jnp.asarray(0.0, dtype=u_previous.dtype)
    applied_forcing = forcing
    steady_mode = case.solver.mode == "steady"
    if potential_system is None:
        potential_system = _prepare_potential_system(
            mesh, materials.conductivity, case.reference_phi_cell, potential_solver
        )
    acceleration = case.solver.coupling_acceleration
    _validate_coupling_controls(case)
    previous_fixed_point_residual: jnp.ndarray | None = None
    coupling_relaxation = jnp.asarray(1.0, dtype=u_previous.dtype)
    anderson_iterates: list[jnp.ndarray] = []
    anderson_residuals: list[jnp.ndarray] = []
    strict_potential_solves = 0
    velocity_linear_tolerance = _nested_velocity_tolerance(coupling_tolerance, u_previous.dtype)
    if case.solver.time_scheme != "implicit_euler" and not steady_mode:
        raise NotImplementedError("fully_developed_inductionless currently supports implicit_euler only")

    for _ in range(max(1, coupling_iterations)):
        potential_iteration_tolerance = _coupling_potential_tolerance(
            case.time_stepper.potential_tolerance,
            velocity_residual=float(velocity_residual),
            coupling_tolerance=float(coupling_tolerance),
            flexible=potential_system.flexible,
        )
        potential_result = _solve_potential(
            mesh,
            materials.conductivity,
            fluid_mask,
            u_iter,
            by,
            bz,
            case.reference_phi_cell,
            case.time_stepper.potential_iterations,
            tolerance=potential_iteration_tolerance,
            relaxation=case.time_stepper.potential_relaxation,
            solver=potential_solver,
            initial_phi=phi_iter,
            system=potential_system,
            return_solver_residual=True,
        )
        requested_potential_tolerance = case.time_stepper.potential_tolerance
        if (
            requested_potential_tolerance is not None
            and potential_iteration_tolerance is not None
            and potential_iteration_tolerance <= requested_potential_tolerance
        ):
            strict_potential_solves += 1
        (
            phi,
            potential_residual,
            potential_iteration_count,
            potential_initial_residual,
            _potential_solver_residual,
        ) = potential_result
        require_finite(
            "potential solve",
            potential=phi,
            residual=potential_residual,
        )
        phi_iter = phi
        rhs_base, _ = _fully_developed_rhs(
            mesh=mesh,
            sigma=materials.conductivity,
            rho=materials.density,
            fluid_mask=fluid_mask,
            u=u_iter,
            phi=phi,
            by=by,
            bz=bz,
            forcing=jnp.asarray(0.0, dtype=u_previous.dtype),
        )
        rhs_base = rhs_base + magnetic_reaction * jnp.where(active_mask, u_iter, 0.0)
        if not steady_mode:
            rhs_base = rhs_base + jnp.where(active_mask, u_previous / dt, 0.0)
        if target_mean_velocity is None:
            rhs = rhs_base + jnp.where(active_mask, forcing / materials.density, 0.0)
            u_next, velocity_linear_residual, linear_iteration_count, linear_initial_residual = (
                _solve_velocity_system(
                    coefficients=velocity_coefficients,
                    cell_metric=cell_metric,
                    rhs=rhs,
                    active_mask=active_mask,
                    preconditioner=preconditioner,
                    max_steps=max(case.time_stepper.max_steps, case.solver.coupling_iterations * 25),
                    tolerance=velocity_linear_tolerance,
                )
            )
            applied_forcing = forcing
        else:
            unit_rhs = jnp.where(active_mask, 1.0 / materials.density, 0.0)
            u_base, velocity_linear_residual, linear_iteration_count, linear_initial_residual = (
                _solve_velocity_system(
                    coefficients=velocity_coefficients,
                    cell_metric=cell_metric,
                    rhs=rhs_base,
                    active_mask=active_mask,
                    preconditioner=preconditioner,
                    max_steps=max(case.time_stepper.max_steps, case.solver.coupling_iterations * 25),
                    tolerance=velocity_linear_tolerance,
                )
            )
            u_sensitivity, _, _, sensitivity_initial_residual = _solve_velocity_system(
                coefficients=velocity_coefficients,
                cell_metric=cell_metric,
                rhs=unit_rhs,
                active_mask=active_mask,
                preconditioner=preconditioner,
                max_steps=max(case.time_stepper.max_steps, case.solver.coupling_iterations * 25),
                tolerance=velocity_linear_tolerance,
            )
            linear_initial_residual = jnp.maximum(linear_initial_residual, sensitivity_initial_residual)
            mean_base = jnp.sum(fluid_weight * u_base) / fluid_total_weight
            mean_sensitivity = jnp.sum(fluid_weight * u_sensitivity) / fluid_total_weight
            applied_forcing = jnp.where(
                mean_sensitivity > 1e-20,
                (jnp.asarray(target_mean_velocity, dtype=u_previous.dtype) - mean_base) / mean_sensitivity,
                jnp.asarray(0.0, dtype=u_previous.dtype),
            )
            u_next = u_base + applied_forcing * u_sensitivity
        linear_residual = velocity_linear_residual
        u_next = _limited_velocity_update(
            u_iter,
            u_next,
            fluid_mask,
            max_delta=case.time_stepper.velocity_update_limit,
        )
        u_next = _enforce_velocity_bc(
            jnp.where(fluid_mask, u_next, 0.0),
            mesh,
            fluid_mask,
            interpolate_direct_fluid_walls=case.geometry.kind == "rect_duct",
        )
        u_next = _enforce_target_mean_velocity(u_next, mesh, fluid_mask, target_mean_velocity)
        fixed_point_residual = u_next - u_iter
        if acceleration == "aitken" and previous_fixed_point_residual is not None:
            coupling_relaxation = _solvax_aitken_relaxation(
                previous_fixed_point_residual,
                fixed_point_residual,
                coupling_relaxation,
                min_relaxation=case.solver.coupling_min_relaxation,
                max_relaxation=case.solver.coupling_max_relaxation,
            )
            u_next = u_iter + coupling_relaxation * fixed_point_residual
            u_next = _enforce_velocity_bc(
                jnp.where(fluid_mask, u_next, 0.0),
                mesh,
                fluid_mask,
                interpolate_direct_fluid_walls=case.geometry.kind == "rect_duct",
            )
            u_next = _enforce_target_mean_velocity(u_next, mesh, fluid_mask, target_mean_velocity)
        elif acceleration == "anderson":
            anderson_iterates.append(u_iter)
            anderson_residuals.append(fixed_point_residual)
            depth = case.solver.coupling_history_depth
            anderson_iterates = anderson_iterates[-depth:]
            anderson_residuals = anderson_residuals[-depth:]
            u_next = _solvax_anderson_mixing(
                jnp.stack(anderson_iterates),
                jnp.stack(anderson_residuals),
                regularization=case.solver.coupling_regularization,
                damping=case.solver.coupling_damping,
            )
            u_next = _enforce_velocity_bc(
                jnp.where(fluid_mask, u_next, 0.0),
                mesh,
                fluid_mask,
                interpolate_direct_fluid_walls=case.geometry.kind == "rect_duct",
            )
            u_next = _enforce_target_mean_velocity(u_next, mesh, fluid_mask, target_mean_velocity)
        # Convergence is defined by the unrelaxed fixed-point residual at the
        # current iterate.  If it passes, retain that certified iterate rather
        # than returning the subsequently extrapolated Anderson/Aitken point,
        # whose residual has not been evaluated.  Acceleration is used only to
        # choose the next iterate when another map evaluation is required.
        velocity_residual = jnp.max(jnp.abs(fixed_point_residual))
        velocity_converged = float(velocity_residual) <= float(coupling_tolerance)
        auxiliary_converged = (
            strict_potential_solves >= _MIN_STRICT_POTENTIAL_COUPLING_SOLVES
            and float(potential_residual) <= _POTENTIAL_COUPLING_NORMALIZED_GATE
            and float(linear_residual) <= max(float(coupling_tolerance), _LINEAR_RESIDUAL_FLOOR)
        )
        if velocity_converged and auxiliary_converged:
            break
        if velocity_converged:
            # Re-evaluate the auxiliary solves at the certified velocity until
            # their own gates pass; do not perturb it merely to accumulate the
            # required strict-solve evidence.
            previous_fixed_point_residual = None
            continue
        previous_fixed_point_residual = fixed_point_residual
        u_iter = u_next

    phi, potential_residual, potential_iteration_count, potential_initial_residual = _solve_potential(
        mesh,
        materials.conductivity,
        fluid_mask,
        u_iter,
        by,
        bz,
        case.reference_phi_cell,
        case.time_stepper.potential_iterations,
        tolerance=case.time_stepper.potential_tolerance,
        relaxation=case.time_stepper.potential_relaxation,
        solver=potential_solver,
        initial_phi=phi_iter,
        system=potential_system,
    )
    jy, jz, lorentz = _compute_current_and_lorentz(
        mesh,
        materials.conductivity,
        fluid_mask,
        u_iter,
        phi,
        by,
        bz,
    )
    face_current_max, emf_max, face_lorentz_max = _face_current_emf_and_lorentz_max(
        mesh,
        materials.conductivity,
        fluid_mask,
        u_iter,
        phi,
        by,
        bz,
    )
    mean_velocity = jnp.sum(fluid_weight * u_iter) / fluid_total_weight
    return (
        u_iter,
        phi,
        jy,
        jz,
        lorentz,
        velocity_residual,
        potential_residual,
        potential_iteration_count,
        linear_residual,
        linear_iteration_count,
        face_current_max,
        emf_max,
        face_lorentz_max,
        mean_velocity,
        applied_forcing,
        potential_initial_residual,
        linear_initial_residual,
    )


def _prepare_fully_developed_velocity_system(
    case: CaseSpec,
    mesh: StructuredMesh,
    materials,
    step_time: float,
) -> _VelocitySystem:
    """Assemble the velocity system once for a fixed material and field state."""

    _, by, bz = magnetic_field_components(case.magnetic_field, mesh, time=step_time)
    active_mask = materials.fluid_mask
    magnetic_reaction = jnp.where(
        active_mask,
        materials.conductivity * (by**2 + bz**2) / materials.density,
        0.0,
    )
    reaction = magnetic_reaction
    if case.solver.mode != "steady":
        reaction = reaction + jnp.where(active_mask, 1.0 / case.time_stepper.dt, 0.0)
    cell_metric = _cell_metric(mesh).astype(materials.conductivity.dtype)
    coefficients = tuple(
        coefficient * cell_metric
        for coefficient in _velocity_system_coefficients(mesh, materials.viscosity, reaction, active_mask)
    )
    return by, bz, magnetic_reaction, coefficients, cell_metric


def _fully_developed_converged(
    case: CaseSpec,
    *,
    velocity_residual: float,
    linear_residual: float,
    potential_residual: float,
) -> bool:
    """Apply the fully developed velocity, linear, and potential stop gate."""

    potential_gate = case.time_stepper.steady_potential_tolerance
    if potential_gate is None:
        potential_gate = case.time_stepper.potential_tolerance
    if potential_gate is None:
        potential_gate = case.time_stepper.steady_tolerance
    return bool(
        velocity_residual <= float(case.time_stepper.steady_tolerance)
        and linear_residual <= max(float(case.time_stepper.steady_tolerance), _LINEAR_RESIDUAL_FLOOR)
        and potential_residual <= float(potential_gate)
    )


def _prepare_fully_developed_case(case: CaseSpec, mesh: StructuredMesh | None = None):
    """Prepare shared mesh, materials, and potential algebra for field solves."""

    mesh = _build_mesh(case) if mesh is None else mesh
    casts = {
        name: getattr(mesh, name).astype(case.dtype)
        for name in ("x_faces", "y_faces", "z_faces", "point_coordinates", "sigma")
        if getattr(mesh, name) is not None and getattr(mesh, name).dtype != case.dtype
    }
    if casts:
        mesh = replace(mesh, **casts)
    materials = build_material_fields(case, mesh)
    potential_solver = _resolve_potential_solver(case.time_stepper.potential_solver, materials.fluid_mask)
    if potential_solver == "cg" and not _has_uniform_spacing(mesh):
        potential_solver = "cg_volume"
    potential_system = _prepare_potential_system(
        mesh, materials.conductivity, case.reference_phi_cell, potential_solver
    )
    return mesh, materials, potential_solver, potential_system


def _validate_coupling_controls(case: CaseSpec) -> None:
    """Reject coupling-acceleration controls outside their documented ranges."""

    solver = case.solver
    if solver.coupling_acceleration not in {"none", "aitken", "anderson"}:
        raise ValueError(f"Unsupported coupling acceleration {solver.coupling_acceleration!r}")
    if (
        solver.coupling_min_relaxation <= 0.0
        or solver.coupling_max_relaxation < solver.coupling_min_relaxation
    ):
        raise ValueError("Coupling relaxation bounds must satisfy 0 < min <= max")
    if solver.coupling_history_depth < 1:
        raise ValueError("Anderson coupling history depth must be positive")
    if solver.coupling_regularization < 0.0:
        raise ValueError("Anderson coupling regularization must be non-negative")
    if not 0.0 <= solver.coupling_damping <= 1.0:
        raise ValueError("Anderson coupling damping must lie in [0, 1]")


def _affine_fully_developed_solve(
    case: CaseSpec,
    mesh: StructuredMesh,
    materials,
    potential_solver: str,
    potential_system: _PotentialSystem,
    by: jnp.ndarray,
    bz: jnp.ndarray,
    *,
    forcing: float | jax.Array,
    tolerance: float,
):
    """Solve the linear duct problem as the fixed point ``u = G(u)``.

    ``G`` is one potential solve followed by one momentum solve, and it is
    affine in ``u``. SOLVAX GMRES therefore solves ``(I - dG) u = G(0)``
    directly. Returns the Krylov solution together with ``G`` and its two
    inner solves, so callers can report the residuals of the returned state.
    """

    fluid_mask = materials.fluid_mask
    reaction = jnp.where(
        fluid_mask,
        materials.conductivity * (by**2 + bz**2) / materials.density,
        0.0,
    )
    cell_metric = _cell_metric(mesh).astype(by.dtype)
    coefficients = tuple(
        coefficient * cell_metric
        for coefficient in _velocity_system_coefficients(mesh, materials.viscosity, reaction, fluid_mask)
    )
    inner_tolerance = _nested_velocity_tolerance(tolerance, by.dtype)
    source = jnp.asarray(forcing, dtype=by.dtype)
    max_steps = max(case.time_stepper.max_steps, case.solver.coupling_iterations * 25)

    def potential(velocity):
        return _solve_potential(
            mesh,
            materials.conductivity,
            fluid_mask,
            velocity,
            by,
            bz,
            case.reference_phi_cell,
            case.time_stepper.potential_iterations,
            tolerance=inner_tolerance,
            solver=potential_solver,
            system=potential_system,
        )

    def momentum(velocity, phi):
        rhs, _ = _fully_developed_rhs(
            mesh=mesh,
            sigma=materials.conductivity,
            rho=materials.density,
            fluid_mask=fluid_mask,
            u=velocity,
            phi=phi,
            by=by,
            bz=bz,
            forcing=source,
        )
        return _solve_velocity_system(
            coefficients=coefficients,
            cell_metric=cell_metric,
            rhs=rhs + reaction * jnp.where(fluid_mask, velocity, 0.0),
            active_mask=fluid_mask,
            preconditioner=case.solver.preconditioner,
            max_steps=max_steps,
            tolerance=inner_tolerance,
        )

    def mapping(velocity):
        velocity = momentum(velocity, potential(velocity)[0])[0]
        return _enforce_velocity_bc(
            velocity,
            mesh,
            fluid_mask,
            interpolate_direct_fluid_walls=case.geometry.kind == "rect_duct",
        )

    zero = jnp.zeros(mesh.yz_shape, dtype=by.dtype)
    restart = min(30, max(2, mesh.ny * mesh.nz))
    max_restarts = 50
    coupled = affine_fixed_point_gmres(
        mapping,
        zero,
        restart=restart,
        rtol=tolerance,
        max_restarts=max_restarts,
        transpose_rtol=tolerance,
        transpose_max_restarts=max_restarts,
    )
    return coupled, mapping, potential, momentum


def _fully_developed_affine_report(
    case: CaseSpec,
    mesh: StructuredMesh,
    materials,
    potential_solver: str,
    potential_system: _PotentialSystem,
    drive: float,
):
    """Solve the steady duct once and return its state with its certificates.

    The residual is ``||G(u) - u|| / ||G(0)||`` evaluated after the solve, so
    it does not depend on the Krylov recurrence. The potential and momentum
    residuals and iteration counts come from ``G``'s solves at ``u``.
    """

    _, by, bz = magnetic_field_components(case.magnetic_field, mesh)
    tolerance = max(float(case.time_stepper.steady_tolerance), 10.0 * float(jnp.finfo(by.dtype).eps))

    def report(source):
        coupled, mapping, potential, momentum = _affine_fully_developed_solve(
            case,
            mesh,
            materials,
            potential_solver,
            potential_system,
            by,
            bz,
            forcing=source,
            tolerance=tolerance,
        )
        velocity = coupled.x
        phi, potential_residual, potential_iterations, _ = potential(velocity)
        _, linear_residual, linear_iterations, _ = momentum(velocity, phi)
        source_norm = jnp.linalg.norm(mapping(jnp.zeros_like(velocity)))
        residual = coupled.residual_norm / jnp.where(source_norm > 0.0, source_norm, 1.0)
        return (
            velocity,
            phi,
            residual,
            coupled.iterations,
            potential_residual,
            potential_iterations,
            linear_residual,
            linear_iterations,
        )

    return by, bz, jax.jit(report)(jnp.asarray(drive, dtype=by.dtype))


def _diagnostic_record(
    *,
    case: CaseSpec,
    mesh: StructuredMesh,
    materials,
    u: jnp.ndarray,
    phi: jnp.ndarray,
    jy: jnp.ndarray,
    jz: jnp.ndarray,
    lorentz: jnp.ndarray,
    by: jnp.ndarray,
    bz: jnp.ndarray,
    residual,
    mean_velocity,
    applied_forcing,
    potential_residual,
    potential_iterations,
    linear_residual,
    linear_iterations,
    face_current_max,
    emf_max,
    face_lorentz_max,
) -> jnp.ndarray:
    """Stack one record of ``_STEP_DIAGNOSTIC_NAMES`` on the device."""

    integrals = _integral_diagnostics(
        mesh=mesh,
        sigma=materials.conductivity,
        fluid_mask=materials.fluid_mask,
        u=u,
        phi=phi,
        jy=jy,
        jz=jz,
        lorentz=lorentz,
        by=by,
        bz=bz,
        anchor=case.reference_phi_cell,
    )
    u_max = jnp.max(jnp.abs(u))
    return jnp.stack(
        (
            u_max,
            mean_velocity,
            applied_forcing,
            residual,
            u_max * case.time_stepper.dt / jnp.min(mesh.dy),
            jnp.mean(jy**2 + jz**2),
            jnp.max(jnp.sqrt(jy**2 + jz**2)),
            face_current_max,
            emf_max,
            jnp.max(jnp.abs(lorentz)),
            face_lorentz_max,
            potential_residual,
            potential_iterations,
            linear_residual,
            linear_iterations,
            *integrals,
        )
    )


def _emit_diagnostic_record(
    logger: StreamingSolverLogger | None,
    *,
    step_index: int,
    step_time: float,
    values: dict[str, float],
    potential_initial_residual: float = 0.0,
    linear_initial_residual: float = 0.0,
) -> None:
    if logger is None:
        return
    _emit_solver_step(
        logger,
        step_index=step_index,
        step_time=step_time,
        u_max_value=values["u_max_history"],
        mean_velocity=values["mean_velocity_history"],
        max_current=values["current_max_history"],
        max_lorentz=values["lorentz_max_history"],
        residual_value=values["residual_history"],
        potential_residual=values["potential_residual_history"],
        potential_iteration_count=values["potential_iterations_history"],
        linear_residual=values["linear_residual_history"],
        linear_iteration_count=values["linear_iterations_history"],
        applied_forcing=values["applied_forcing_history"],
        courant_like=values["courant_like"],
        ohmic=values["ohmic_power"],
        volumetric_flow_rate=values["volumetric_flow_rate_history"],
        div_current_max=values["div_current_max_history"],
        charge_balance_residual=values["charge_balance_residual_history"],
        gauge_residual=values["gauge_residual_history"],
        interface_current_residual=values["interface_current_residual_history"],
        potential_initial_residual=potential_initial_residual,
        linear_initial_residual=linear_initial_residual,
    )


def _history_diagnostics(
    case: CaseSpec,
    history_values: dict,
    initial_diagnostics: Diagnostics | None,
    append_diagnostics: bool,
) -> Diagnostics:
    stride = case.output.history_stride

    def retained_history(name, values):
        initial = (
            getattr(initial_diagnostics, name) if initial_diagnostics is not None and stride != 0 else None
        )
        return _concat_history(initial, jnp.asarray(values, dtype=float), append=append_diagnostics)

    return Diagnostics(**{name: retained_history(name, values) for name, values in history_values.items()})


def _solve_fully_developed_steady(
    case: CaseSpec,
    logger: StreamingSolverLogger | None,
    prepared,
    *,
    initial_state: MHDState | None,
    initial_diagnostics: Diagnostics | None,
    append_diagnostics: bool,
    restart_info: RestartLogInfo | None,
) -> Solution:
    """Report the steady state from one certified affine solve (ADR 0005, D19).

    The discrete fully developed problem is linear, so pseudo-time stepping
    toward it only adds a stopping floor. A prescribed flow rate is met by
    scaling the unit-drive solution, which is exact for a linear problem.
    """

    mesh, materials, potential_solver, potential_system = prepared
    _validate_coupling_controls(case)
    target_mean_velocity = _target_mean_velocity(case)
    start_time = 0.0 if initial_state is None else float(initial_state.time)
    _emit_solver_header(
        logger,
        case=case,
        mesh=mesh,
        mode="steady",
        potential_solver=f"{potential_solver} / solvax_pcg / affine_gmres",
        target_mean_velocity=target_mean_velocity,
        reference_mean_velocity=_reference_mean_velocity(case),
        restart=restart_info,
    )
    by, bz, solved = _fully_developed_affine_report(
        case,
        mesh,
        materials,
        potential_solver,
        potential_system,
        drive=case.forcing if target_mean_velocity is None else 1.0,
    )
    (
        u,
        phi,
        residual,
        iterations,
        potential_residual,
        potential_iterations,
        linear_residual,
        linear_iterations,
    ) = solved
    fluid_mask = materials.fluid_mask
    fluid_weight = jnp.where(fluid_mask, _cell_metric(mesh).astype(u.dtype), 0.0)
    fluid_total_weight = jnp.maximum(jnp.sum(fluid_weight), 1e-20)
    applied_forcing = jnp.asarray(case.forcing, dtype=u.dtype)
    if target_mean_velocity is not None:
        unit_mean = jnp.sum(fluid_weight * u) / fluid_total_weight
        resolved = jnp.abs(unit_mean) > 1e-20
        applied_forcing = jnp.where(resolved, target_mean_velocity / jnp.where(resolved, unit_mean, 1.0), 0.0)
        u, phi = applied_forcing * u, applied_forcing * phi
    jy, jz, lorentz = _compute_current_and_lorentz(mesh, materials.conductivity, fluid_mask, u, phi, by, bz)
    face_current_max, emf_max, face_lorentz_max = _face_current_emf_and_lorentz_max(
        mesh, materials.conductivity, fluid_mask, u, phi, by, bz
    )
    record = _diagnostic_record(
        case=case,
        mesh=mesh,
        materials=materials,
        u=u,
        phi=phi,
        jy=jy,
        jz=jz,
        lorentz=lorentz,
        by=by,
        bz=bz,
        residual=residual,
        mean_velocity=jnp.sum(fluid_weight * u) / fluid_total_weight,
        applied_forcing=applied_forcing,
        potential_residual=potential_residual,
        potential_iterations=potential_iterations,
        linear_residual=linear_residual,
        linear_iterations=linear_iterations,
        face_current_max=face_current_max,
        emf_max=emf_max,
        face_lorentz_max=face_lorentz_max,
    )
    record, iterations = jax.device_get((record, iterations))
    values = dict(zip(_STEP_DIAGNOSTIC_NAMES, map(float, record), strict=True))
    steps = int(iterations)
    _emit_diagnostic_record(logger, step_index=steps, step_time=start_time, values=values)
    require_finite(
        "fully developed solve",
        velocity=u,
        potential=phi,
        current_y=jy,
        current_z=jz,
        lorentz_force=lorentz,
        residual=values["residual_history"],
    )
    converged = _fully_developed_converged(
        case,
        velocity_residual=values["residual_history"],
        linear_residual=values["linear_residual_history"],
        potential_residual=values["potential_residual_history"],
    )
    state = MHDState(
        u=u,
        phi=phi,
        jy=jy,
        jz=jz,
        lorentz_x=lorentz,
        time=start_time,
        residual=values["residual_history"],
    )
    history_values = {"time_history": [start_time], **{name: [value] for name, value in values.items()}}
    solution = Solution(
        mesh=mesh,
        state=state,
        diagnostics=_history_diagnostics(case, history_values, initial_diagnostics, append_diagnostics),
        case_name=case.name,
        converged=converged,
        status="converged" if converged else "not_converged",
        steps=steps,
    )
    if logger is not None:
        logger.emit_footer(solution)
    return solution


def _solve_fully_developed(
    case: CaseSpec,
    logger: StreamingSolverLogger | None = None,
    *,
    mesh: StructuredMesh | None = None,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
) -> Solution:
    if case.output.history_stride < 0:
        raise ValueError("history_stride must be non-negative")
    prepared = _prepare_fully_developed_case(case, mesh)
    mesh, materials, potential_solver, potential_system = prepared
    if case.geometry.kind not in {"rect_duct", "layered_duct"}:
        raise NotImplementedError(
            f"Solver {case.solver.kind!r} does not yet support geometry {case.geometry.kind!r}"
        )
    if case.solver.mode == "steady":
        return _solve_fully_developed_steady(
            case,
            logger,
            prepared,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=append_diagnostics,
            restart_info=restart_info,
        )
    target_mean_velocity = _target_mean_velocity(case)
    interpolate_direct_fluid_walls = case.geometry.kind == "rect_duct"
    initial_u, initial_phi, initial_jy, initial_jz, initial_lorentz, start_time = _initial_solver_state(
        case=case,
        mesh=mesh,
        fluid_mask=materials.fluid_mask,
        interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
        initial_state=initial_state,
    )
    dt = case.time_stepper.dt
    steps = _bounded_time_step_count(
        start_time=start_time,
        dt=dt,
        t_final=case.time_stepper.t_final,
        max_steps=case.time_stepper.max_steps,
    )
    _emit_solver_header(
        logger,
        case=case,
        mesh=mesh,
        mode=case.solver.mode,
        potential_solver=f"{potential_solver} / solvax_pcg",
        target_mean_velocity=target_mean_velocity,
        reference_mean_velocity=_reference_mean_velocity(case),
        restart=restart_info,
    )

    u = initial_u
    phi = initial_phi
    jy = initial_jy
    jz = initial_jz
    lorentz = initial_lorentz
    stride = case.output.history_stride
    retained: list[tuple[float, jnp.ndarray]] = []
    fixed_velocity_system = (
        _prepare_fully_developed_velocity_system(case, mesh, materials, start_time)
        if case.magnetic_field.ramp_duration <= 0.0
        else None
    )

    for step_index in range(steps):
        step_time = float(start_time + (step_index + 1) * dt)
        u_before_step = u
        (
            u,
            phi,
            jy,
            jz,
            lorentz,
            residual,
            potential_residual,
            potential_iteration_count,
            linear_residual,
            linear_iteration_count,
            face_current_max,
            emf_max,
            face_lorentz_max,
            mean_velocity,
            applied_forcing,
            potential_initial_residual,
            linear_initial_residual,
        ) = _fully_developed_case_step(
            case=case,
            mesh=mesh,
            materials=materials,
            u_previous=u,
            step_time=step_time,
            potential_solver=potential_solver,
            target_mean_velocity=target_mean_velocity,
            preconditioner=case.solver.preconditioner,
            coupling_iterations=case.solver.coupling_iterations,
            coupling_tolerance=case.solver.coupling_tolerance,
            phi_previous=phi,
            velocity_system=fixed_velocity_system,
            potential_system=potential_system,
        )
        _, by_step, bz_step = magnetic_field_components(case.magnetic_field, mesh, time=step_time)
        record = _diagnostic_record(
            case=case,
            mesh=mesh,
            materials=materials,
            u=u,
            phi=phi,
            jy=jy,
            jz=jz,
            lorentz=lorentz,
            by=by_step,
            bz=bz_step,
            residual=jnp.maximum(residual, jnp.max(jnp.abs(u - u_before_step))),
            mean_velocity=mean_velocity,
            applied_forcing=applied_forcing,
            potential_residual=potential_residual,
            potential_iterations=potential_iteration_count,
            linear_residual=linear_residual,
            linear_iterations=linear_iteration_count,
            face_current_max=face_current_max,
            emf_max=emf_max,
            face_lorentz_max=face_lorentz_max,
        )
        if stride == 0:
            retained[:] = ((step_time, record),)
        elif step_index % stride == 0 or step_index == steps - 1:
            retained.append((step_time, record))
        if logger is not None:
            # Records stay on the device unless a logger streams them.
            _emit_diagnostic_record(
                logger,
                step_index=step_index + 1,
                step_time=step_time,
                values=dict(zip(_STEP_DIAGNOSTIC_NAMES, map(float, jax.device_get(record)), strict=True)),
                potential_initial_residual=float(potential_initial_residual),
                linear_initial_residual=float(linear_initial_residual),
            )

    records = jax.device_get(jnp.stack([record for _, record in retained])) if retained else None
    history_values: dict[str, object] = {"time_history": [step_time for step_time, _ in retained]}
    for index, name in enumerate(_STEP_DIAGNOSTIC_NAMES):
        history_values[name] = [] if records is None else records[:, index]
    residual_value = (
        float(history_values["residual_history"][-1])
        if records is not None
        else float(initial_state.residual if initial_state is not None else 0.0)
    )

    require_finite(
        "fully developed solve",
        velocity=u,
        potential=phi,
        current_y=jy,
        current_z=jz,
        lorentz_force=lorentz,
        residual=residual_value,
        residual_history=history_values["residual_history"],
        potential_residual_history=history_values["potential_residual_history"],
        linear_residual_history=history_values["linear_residual_history"],
    )
    state = MHDState(
        u=u,
        phi=phi,
        jy=jy,
        jz=jz,
        lorentz_x=lorentz,
        time=float(start_time + steps * dt),
        residual=residual_value,
    )
    solution = Solution(
        mesh=mesh,
        state=state,
        diagnostics=_history_diagnostics(case, history_values, initial_diagnostics, append_diagnostics),
        case_name=case.name,
        converged=None,
        status="completed",
        steps=steps,
    )
    if logger is not None:
        logger.emit_footer(solution)
    return solution


def solve_transient(
    case: CaseSpec,
    logger: StreamingSolverLogger | None = None,
    *,
    mesh: StructuredMesh | None = None,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
) -> Solution:
    """Advance a supported case in transient mode, optionally from a restart."""

    solver_kind = getattr(getattr(case, "solver", None), "kind", "fully_developed_inductionless")
    if solver_kind == "fully_developed_inductionless":
        transient_case = (
            case
            if case.solver.mode == "transient"
            else case.__class__(
                **{
                    **case.__dict__,
                    "solver": case.solver.__class__(**{**case.solver.__dict__, "mode": "transient"}),
                }
            )
        )
        return _solve_fully_developed(
            transient_case,
            logger=logger,
            mesh=mesh,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=append_diagnostics,
            restart_info=restart_info,
        )
    raise NotImplementedError(f"Solver kind {solver_kind!r} is not implemented for transient runs")


def solve_steady(
    case: CaseSpec,
    logger: StreamingSolverLogger | None = None,
    *,
    mesh: StructuredMesh | None = None,
    initial_state: MHDState | None = None,
    initial_diagnostics: Diagnostics | None = None,
    append_diagnostics: bool = False,
    restart_info: RestartLogInfo | None = None,
) -> Solution:
    """Solve a supported case to a certified steady state.

    A fully developed case is linear, so its steady state comes from one
    affine fixed-point GMRES solve, not from pseudo-time steps. ``status`` is
    ``"converged"`` only when the relative fixed-point residual
    ``||G(u) - u|| / ||G(0)||`` (``solution.residual``) meets
    ``steady_tolerance`` and the final potential and momentum solves meet
    their gates; otherwise it is ``"not_converged"``. ``steps`` counts GMRES
    iterations and the diagnostics hold one record. ``initial_state`` sets
    only the reported time, because the steady state does not depend on it.
    Use :func:`solve_transient` for time histories.
    """

    solver_kind = getattr(getattr(case, "solver", None), "kind", "fully_developed_inductionless")
    if solver_kind == "fully_developed_inductionless":
        steady_case = (
            case
            if case.solver.mode == "steady"
            else case.__class__(
                **{
                    **case.__dict__,
                    "solver": case.solver.__class__(**{**case.solver.__dict__, "mode": "steady"}),
                }
            )
        )
        return _solve_fully_developed(
            steady_case,
            logger=logger,
            mesh=mesh,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=append_diagnostics,
            restart_info=restart_info,
        )
    raise NotImplementedError(f"Solver kind {solver_kind!r} is not implemented for steady runs")


def solve(
    model: "ChannelProblem | CaseSpec | Q2DProblem",
) -> "SteadySolution | Solution | Q2DResult":
    """Solve a duct, a fully developed case, or a Q2D problem.

    A :class:`lmhdx.core3d.ChannelProblem` goes to the staggered core's steady
    solve, :func:`lmhdx.steady.solve_steady_state`, and so does a fully developed
    ``CaseSpec``, through :func:`lmhdx.fully_developed.solve_fully_developed`,
    which reports it on the case's cross-section; each is compiled once per
    problem. A steady case the core does not represent (thick or mismatched
    conducting walls, several fluids) keeps the cell-centred
    solve, :func:`solve_steady`. A transient ``CaseSpec`` runs the core's
    projection steps, :func:`lmhdx.fully_developed.solve_fully_developed_transient`,
    or, where the core does not apply, the pseudo-time loop of
    :func:`solve_transient`. A duct with an inlet and an outlet is
    solved by :func:`lmhdx.axial.solve_open_duct`.
    """

    from .core3d import ChannelProblem

    if isinstance(model, ChannelProblem):
        from .steady import solve_compiled

        return solve_compiled(model)
    if isinstance(model, CaseSpec):
        from .fully_developed import core_applies, solve_fully_developed, solve_fully_developed_transient

        if model.solver.mode == "transient":
            return solve_fully_developed_transient(model) if core_applies(model) else solve_transient(model)
        return solve_fully_developed(model) if core_applies(model) else solve_steady(model)
    from .q2d import Q2DProblem, solve_q2d

    if isinstance(model, Q2DProblem):
        return solve_q2d(model)
    raise TypeError(f"solve expects ChannelProblem, CaseSpec, or Q2DProblem, got {type(model).__name__}")


def solve_fully_developed_fields(
    case: CaseSpec,
    *,
    forcing: float | jax.Array | None = None,
    magnetic_field_scale: float | jax.Array = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return steady duct fields from the cell-centred affine solve of this module.

    This is the validation lane that plan step 4.6 retires; the public
    :func:`lmhdx.solve_fully_developed_fields` solves the same case on the
    staggered core (:mod:`lmhdx.fully_developed`). ``forcing`` and ``magnetic_field_scale`` are continuous design inputs.
    The coupled affine state uses a SOLVAX implicit tangent/transpose solve,
    so reverse mode does not retain potential, momentum, or coupling
    iterations. Meshes, material regions, boundary kinds, and solver controls
    are static; construct or close over ``case`` outside ``jit``.
    """

    if case.solver.kind != "fully_developed_inductionless":
        raise ValueError("case must select the fully developed inductionless solver")
    if case.geometry.kind not in {"rect_duct", "layered_duct"}:
        raise NotImplementedError(f"unsupported differentiable duct geometry {case.geometry.kind!r}")
    if case.magnetic_field.ramp_duration > 0.0:
        raise ValueError("steady differentiable fields require an unramped magnetic field")
    if _target_mean_velocity(case) is not None:
        raise NotImplementedError("fixed-flow differentiation is not yet supported")
    with jax.ensure_compile_time_eval():
        mesh, materials, potential_solver, potential_system = _prepare_fully_developed_case(case)
        _, by, bz = magnetic_field_components(case.magnetic_field, mesh)
    field_scale = jnp.asarray(magnetic_field_scale, dtype=by.dtype)
    by, bz = field_scale * by, field_scale * bz
    coupled, _, potential, _ = _affine_fully_developed_solve(
        case,
        mesh,
        materials,
        potential_solver,
        potential_system,
        by,
        bz,
        forcing=case.forcing if forcing is None else forcing,
        tolerance=case.solver.coupling_tolerance,
    )
    phi = potential(coupled.x)[0]
    jy, jz, lorentz = _compute_current_and_lorentz(
        mesh, materials.conductivity, materials.fluid_mask, coupled.x, phi, by, bz
    )
    return coupled.x, phi, jy, jz, lorentz
