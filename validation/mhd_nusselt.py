"""The MHD Nusselt campaign: Shercliff's square duct, LMhdX against a spectral solve and a published analysis.

``python -m validation.mhd_nusselt`` reruns the tables of ``docs/validation/mhd_nusselt.md``: about ten minutes
on a laptop CPU, most of it the 128-cell LMhdX solves.

The duct has insulating walls, a field along ``y``, Ha on the half-width ``a`` and ``D_h = 2a``, and no Joule
heat. Three thermal conditions are compared: H1 (the wall at one temperature around the section, a uniform axial
heat input), H2 (a uniform wall flux) and T (the wall at one temperature along the duct).
- LMhdX: ``developed_heat`` and ``developed_wall_temperature`` on ``duct_problem``'s 128-cell meshes.
- The reference: :func:`validation.shercliff.developed_nusselt`, Chebyshev collocation on its own velocity.
- Bluck, Wolfendale & Marquis (2015): their Figs. 4 (H1) and 6 (H2), read from the accepted manuscript's vector
  paths into ``bluck2015-shercliff-nusselt.json`` (its provenance is in ``benchmarks/provenance.json``). The two
  figures match the reference only with different Ha conventions, which the record states.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from validation.shercliff import developed_nusselt

HARTMANN = (1.0, 10.0, 10**1.5, 100.0, 10**2.5)
POINTS = (48, 48, 48, 64, 72)  # Chebyshev points converged to 1e-6 at each Ha
READINGS = (
    Path(__file__).resolve().parents[1]
    / "src/lmhdx/data/benchmarks/references/bluck2015-shercliff-nusselt.json"
)


def lmhdx_nusselt(hartmann: float, cells: int = 128) -> dict[str, float]:
    """H1, H2 and T on ``duct_problem(hartmann=..., cells=...)``, in float64."""
    import jax.numpy as jnp

    import lmhdx
    from lmhdx.core3d import duct_problem
    from lmhdx.grid import CENTER, PERIODIC, BoundaryCondition, Field
    from lmhdx.steady import solve_steady_state
    from lmhdx.thermal import ThermalProblem, developed_heat, developed_wall_temperature, thermal_boundary

    lmhdx.enable_x64()
    flow = duct_problem(hartmann=hartmann, cells=cells)
    velocity = solve_steady_state(flow, pseudo_step=1.0e3, linear_restart=600).velocity
    zero = Field(jnp.zeros(flow.grid.shape), (CENTER,) * 3, flow.grid)
    held, flux = thermal_boundary("temperature"), thermal_boundary("heat_flux", -1.0, -1.0, conductivity=1.0)

    def problem(wall):
        return ThermalProblem(
            flow, (BoundaryCondition(PERIODIC), wall, wall), 1.0, 1.0, 0.0, joule_heating=False
        )

    return {
        "H1": float(developed_heat(problem(held), velocity, zero, axial_gradient=1.0).nusselt),
        "H2": float(developed_heat(problem(flux), velocity, zero).nusselt),
        "T": float(developed_wall_temperature(problem(held), velocity).nusselt),
    }


def bluck_readings() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The read curves by heating, ``(Ha, Nu)`` with Ha on each figure's own axis."""
    figures = json.loads(READINGS.read_text())["figures"].values()
    return {fig["heating"]: (np.array(fig["hartmann"]), np.array(fig["nusselt"])) for fig in figures}


def main() -> None:
    curves = bluck_readings()
    print("Ha     | H1: LMhdX / spectral | H2: LMhdX / spectral | T: LMhdX / spectral | Bluck H1, Ha on a")
    for hartmann, points in zip(HARTMANN, POINTS):
        ours = lmhdx_nusselt(hartmann)
        reference = {heating: developed_nusselt(hartmann, points, heating=heating) for heating in ours}
        read = curves["H1"][1][np.argmin(np.abs(np.log10(curves["H1"][0] / hartmann)))]
        cells = " | ".join(
            f"{ours[h]:.4f} / {reference[h]:.4f} ({ours[h] / reference[h] - 1:+.2%})" for h in ours
        )
        print(f"{hartmann:6.4g} | {cells} | {read:.4f} ({read / reference['H1'] - 1:+.2%})", flush=True)
    print("Bluck H2, Ha on D_h = 2a: Ha_B | equivalent Ha | read | spectral")
    for hartmann_b in HARTMANN:
        read = curves["H2"][1][np.argmin(np.abs(np.log10(curves["H2"][0] / hartmann_b)))]
        spectral = developed_nusselt(hartmann_b / 2.0, 48 if hartmann_b < 300 else 64, heating="H2")
        print(
            f"{hartmann_b:6.4g} | {hartmann_b / 2:6.4g} | {read:.4f} | {spectral:.4f} ({read / spectral - 1:+.2%})"
        )


if __name__ == "__main__":
    main()
