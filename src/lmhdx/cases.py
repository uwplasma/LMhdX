"""Fully developed duct cases: the builders and the solve entry point.

A ``CaseSpec`` is solved on the staggered core by :mod:`lmhdx.fully_developed`:
steady cases by :func:`~lmhdx.fully_developed.solve_fully_developed`, transient
ones by :func:`~lmhdx.fully_developed.solve_fully_developed_transient`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .physics import magnetic_field_from_hartmann
from .specs import (
    BoundaryCondition,
    CaseSpec,
    GeometrySpec,
    MagneticFieldSpec,
    OutputSpec,
    RegionSpec,
    Solution,
    SolverConfig,
    TimeStepperConfig,
)

if TYPE_CHECKING:
    from .core3d import ChannelProblem
    from .q2d import Q2DProblem, Q2DResult
    from .steady import SteadySolution


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


def solve(
    model: "ChannelProblem | CaseSpec | Q2DProblem",
) -> "SteadySolution | Solution | Q2DResult":
    """Solve a duct, a fully developed case, or a Q2D problem.

    A :class:`lmhdx.core3d.ChannelProblem` goes to the staggered core's steady
    solve, :func:`lmhdx.steady.solve_steady_state`, and so does a steady
    ``CaseSpec``, through :func:`lmhdx.fully_developed.solve_fully_developed`,
    which reports it on the case's cross-section; each is compiled once per
    problem. A transient ``CaseSpec`` runs implicit Euler steps on the core,
    :func:`lmhdx.fully_developed.solve_fully_developed_transient`. A duct with an
    inlet and an outlet is solved by :func:`lmhdx.axial.solve_open_duct`.
    """

    from .core3d import ChannelProblem

    if isinstance(model, ChannelProblem):
        from .steady import solve_compiled

        return solve_compiled(model)
    if isinstance(model, CaseSpec):
        from .fully_developed import solve_fully_developed, solve_fully_developed_transient

        if model.solver.mode == "transient":
            return solve_fully_developed_transient(model)
        return solve_fully_developed(model)
    from .q2d import Q2DProblem, solve_q2d

    if isinstance(model, Q2DProblem):
        return solve_q2d(model)
    raise TypeError(f"solve expects ChannelProblem, CaseSpec, or Q2DProblem, got {type(model).__name__}")
