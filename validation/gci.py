"""Observed order, Richardson value and grid-convergence index from three meshes.

The procedure is Celik et al. 2008 (J. Fluids Eng. 130, 078001), the form the
ASME V&V 20 standard adopts: the observed order ``p`` solves

.. math::
   p = \\frac{\\bigl|\\ln|\\varepsilon_{32}/\\varepsilon_{21}| + q(p)\\bigr|}{\\ln r_{21}},
   \\qquad q(p) = \\ln\\frac{r_{21}^p - s}{r_{32}^p - s},

with ``eps_ij = phi_i - phi_j`` (1 the fine mesh), ``s`` the sign of
``eps_32/eps_21`` and ``r`` the refinement ratios; the fine-mesh index is
``GCI = 1.25 |phi_1 - phi_2| / |phi_1| / (r_21^p - 1)``. Three meshes fit
``p`` exactly, so they cannot test the asymptotic range by themselves (Roache's
ratio is one by construction): an order or index is quoted only where the
convergence is monotone (``s > 0``) and ``p`` is near the scheme's formal order.

``python -m validation.gci`` reruns the campaign behind the table in
``docs/validation/index.md`` from the package's own three-mesh sequences.
"""

from __future__ import annotations

import math
from typing import NamedTuple

__all__ = ["GridConvergence", "grid_convergence", "table_i_flow_rate"]


class GridConvergence(NamedTuple):
    order: float
    extrapolated: float
    relative_change: float
    extrapolated_error: float
    gci_fine: float
    monotone: bool


def grid_convergence(
    coarse: float,
    medium: float,
    fine: float,
    ratio: float,
    ratio_coarse: float | None = None,
    safety: float = 1.25,
) -> GridConvergence:
    """Return Celik's quantities for three solutions; ``ratio`` is ``h_2/h_1``, ``ratio_coarse`` ``h_3/h_2``."""
    r21, r32 = ratio, ratio if ratio_coarse is None else ratio_coarse
    e21, e32 = medium - fine, coarse - medium
    sign = math.copysign(1.0, e32 / e21)
    order = math.log(abs(e32 / e21)) / math.log(r21)
    for _ in range(200):  # fixed point; exact in one step when r21 == r32
        q = math.log((r21**order - sign) / (r32**order - sign))
        order, previous = abs(math.log(abs(e32 / e21)) + q) / math.log(r21), order
        if abs(order - previous) < 1e-12:
            break
    extrapolated = (r21**order * fine - medium) / (r21**order - 1.0)
    gci_fine = safety * abs(e21 / fine) / (r21**order - 1.0)
    return GridConvergence(
        order,
        extrapolated,
        abs(e21 / fine),
        abs((extrapolated - fine) / extrapolated),
        gci_fine,
        sign > 0,
    )


# Smolentsev et al. 2015 Table I meshes, Ny:layer_y:Nz:layer_z; each series scales by 1.5.
TABLE_I_RATIO = 1.5
TABLE_I_MESHES = {
    500: ((32, 4, 32, 4), (48, 6, 48, 6), (72, 9, 72, 9)),
    "high": ((64, 8, 32, 4), (96, 12, 48, 6), (144, 18, 72, 9)),
}


def table_i_flow_rate(hartmann: float, conductance: float, cells, layer, cells_across, layer_across) -> float:
    """Square-duct flow rate for unit ``-dP/dx`` on the staggered core (Table I's normalisation)."""
    import numpy as np

    from lmhdx import enable_x64
    from lmhdx.core3d import ChannelProblem
    from lmhdx.grid import PERIODIC, BoundaryCondition, Grid, uniform_faces, wall_resolving_faces
    from lmhdx.steady import solve_steady_state

    enable_x64()
    wall = BoundaryCondition("neumann")

    def faces(count, layer_cells, thickness):
        return wall_resolving_faces(
            count, -1.0, 1.0, layer_thickness=thickness, cells_in_layer=layer_cells, max_ratio=None
        )

    problem = ChannelProblem(
        grid=Grid(
            uniform_faces(1, 0.0, 1.0),
            faces(cells, layer, 1.0 / hartmann),
            faces(cells_across, layer_across, hartmann**-0.5),
        ),
        conditions=(BoundaryCondition(PERIODIC), wall, wall),
        conductivity=1.0,
        magnetic_field=(0.0, hartmann, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
        wall_conductance=(0.0, conductance, 0.0),
    )
    velocity = np.asarray(solve_steady_state(problem).velocity[0].data)[0]
    return float((velocity * np.asarray(problem.grid.cell_volumes())[0]).sum())


def _campaign():
    """Yield ``(row, case, reference, (coarse, medium, fine), ratio)`` for every three-mesh sequence."""
    from importlib.resources import files

    import numpy as np

    from lmhdx.coreflow import CoreFlow
    from lmhdx.pipe import PipeProblem, flow_rate, pipe_grid, solve_pipe

    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    table = tomllib.loads(
        files("lmhdx").joinpath("data/benchmarks/references/samper-table-i.toml").read_text(encoding="utf-8")
    )
    for case in sorted(table["cases"], key=lambda c: (c["hartmann_number"], c["hartmann_wall_conductance"])):
        hartmann, conductance = case["hartmann_number"], case["hartmann_wall_conductance"]
        meshes = TABLE_I_MESHES.get(hartmann, TABLE_I_MESHES["high"])
        values = tuple(table_i_flow_rate(hartmann, conductance, *mesh) for mesh in meshes)
        name = f"{'A1 Shercliff' if conductance == 0.0 else 'A2 Hunt c 0.01'}, Ha {hartmann:,}"
        yield "3/4/27", name, case["analytical_flow_rate"], values, TABLE_I_RATIO
    # Row 5: the conducting pipe, Ha 20, c 0.1, every length doubled (tests/test_pipe.py).
    pipe = tuple(
        flow_rate(
            solve_pipe(PipeProblem(pipe_grid(radial, azimuthal, 20.0, cells_in_layer=layer), 20.0, 0.1))[0]
        )
        for radial, azimuthal, layer in ((24, 32, 3), (48, 64, 6), (96, 128, 12))
    )
    yield "5", "pipe Ha 20, c 0.1 (spectral reference)", 0.015286900, pipe, 2.0

    # Row 7 (core-flow model): the ANL fringe excess against TM-228's 0.0178, cross-section and axis doubled.
    def anl(x):
        return 0.5 * (1.0 - np.sin(np.pi * np.clip(x, -3.0, 3.0) / 6.0))

    excess = []
    for cells, nodes in ((10, 81), (20, 161), (40, 321)):
        x = np.linspace(-10.0, 6.0, nodes)
        pressure = np.asarray(
            CoreFlow(x, nz=cells, ny=cells).solve(anl(x), c_t=0.02, c_s=0.02).pressure
        ).mean(axis=1)
        gradient = float(
            CoreFlow(np.linspace(0.0, 2.0, 9), nz=cells, ny=cells)
            .solve(np.ones(9), c_t=0.02, c_s=0.02)
            .pressure_drop
        )
        inside = (x > -6.0 - 1e-9) & (x < 2.0 + 1e-9)
        square = anl(x[inside]) ** 2
        local = gradient / 2.0 * float(np.sum(0.5 * (square[1:] + square[:-1]) * np.diff(x[inside])))
        excess.append(float(np.interp(-6.0, x, pressure) - np.interp(2.0, x, pressure)) - local)
    yield "7", "ANL fringe excess, core-flow model (TM-228)", 0.0932 - 0.0754, tuple(excess), 2.0


if __name__ == "__main__":
    print("| Row | Case | Fine value | Reference | Fine error | Order | Richardson error | GCI fine |")
    print("|---|---|---|---|---|---|---|---|")
    for row, name, reference, values, ratio in _campaign():
        g = grid_convergence(*values, ratio)
        order = f"{g.order:.2f}" if g.monotone else "oscillatory"
        print(
            f"| {row} | {name} | {values[2]:.8g} | {reference:.6g} | {values[2] / reference - 1:+.2e} | {order} "
            f"| {g.extrapolated / reference - 1:+.1e} | {g.gci_fine:.1e} |",
            flush=True,
        )
