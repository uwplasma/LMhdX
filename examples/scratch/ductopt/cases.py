"""Physical inputs and the three duct cases (stage_2_plan.txt Section 2, 4.4, 0.6).

PbLi properties at 573 K, Martelli, Venturini & Utili 2019 (Fusion Eng. Des.
138:183, [D3]); EU DEMO field/geometry, Federici et al. 2019 (Nucl. Fusion
59:066013, [M1]). Every number here is also written into the results JSON
(``INPUTS``) so a reader never has to trust this file blindly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import DesignBox, Station

# ---------------------------------------------------------------- PbLi, 573 K
PBLI = {
    "temperature_K": 573.0,
    "sigma": 8.84e5,  # S/m
    "rho": 9838.0,  # kg/m^3
    "mu": 2.15e-3,  # Pa s, dynamic
    "cp": 189.8,  # J/(kg K)
    "source": "Martelli, Venturini & Utili 2019, Fusion Eng. Des. 138:183 (D3)",
}

# ------------------------------------------------------------- tokamak field
DEMO = {
    "R0_m": 9.0,
    "B0_T": 5.9,
    "R_outboard_fw_m": 12.1,
    "R_inboard_fw_m": 5.9,
    "blanket_depth_outboard_m": 0.982,
    "blanket_depth_inboard_m": 0.755,
    "source": "Federici et al. 2019, Nucl. Fusion 59:066013 (M1); first-wall "
    "radii and local field are this review's estimate from the radial build, "
    "not a table value (plan Section 2.1)",
}

B_FW = 0.5  # T: lab-scale reference field at the first wall (plan 4.4), not the reactor 4-10 T

# --------------------------------------------------------- flow/design bounds
Q_DEFAULT = 2.4e-6  # m^3/s
V_MIN_DEFAULT = 0.010  # m/s
V_MAX_DEFAULT = 1.0  # m/s
BETA_LO, BETA_HI = 0.02, 5.0
V_MIN_SWEEP = (0.002, 0.005, 0.010, 0.020, 0.050)


def _radial_stations(R_fw: float, depth: float, sign: float, n: int) -> tuple[Station, ...]:
    """``n`` equal-length stations of a 1/R toroidal field over one radial run.

    ``sign = -1`` for inboard (R decreases going into the blanket, B rises),
    ``sign = +1`` for outboard (R increases, B falls).
    """
    edges = R_fw + sign * depth * np.linspace(0.0, 1.0, n + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    L = depth / n
    return tuple(Station(B=float(B_FW * R_fw / R), L=float(L), label=f"station {k}", R=float(R)) for k, R in enumerate(mids))


def case_r_outboard(Q=Q_DEFAULT, V_min=V_MIN_DEFAULT, V_max=V_MAX_DEFAULT, n_stations=6) -> DesignBox:
    """Radial duct at the outboard midplane: B falls going into the blanket."""
    stations = _radial_stations(DEMO["R_outboard_fw_m"], DEMO["blanket_depth_outboard_m"], +1.0, n_stations)
    return DesignBox(
        Q=Q, mu=PBLI["mu"], sigma=PBLI["sigma"], V_min=V_min, V_max=V_max,
        beta_lo=BETA_LO, beta_hi=BETA_HI, stations=stations,
    )


def case_r_inboard(Q=Q_DEFAULT, V_min=V_MIN_DEFAULT, V_max=V_MAX_DEFAULT, n_stations=6) -> DesignBox:
    """Radial duct at the inboard midplane: B rises going into the blanket."""
    stations = _radial_stations(DEMO["R_inboard_fw_m"], DEMO["blanket_depth_inboard_m"], -1.0, n_stations)
    return DesignBox(
        Q=Q, mu=PBLI["mu"], sigma=PBLI["sigma"], V_min=V_min, V_max=V_max,
        beta_lo=BETA_LO, beta_hi=BETA_HI, stations=stations,
    )


@dataclass(frozen=True)
class CaseP:
    """Vertical outboard-midplane poloidal duct, upward-flow leg (plan 0.6).

    A single station: R is constant along a vertical duct, so the toroidal
    field magnitude at the duct centre does not vary along the axis. It DOES
    vary ACROSS the duct (the cross-duct z-direction is radial), an effect
    the uniform-field box below neglects and ``validity.case_p_correction``
    measures exactly (probe_poloidal_gradient.py's construction).
    """

    box: DesignBox
    R_center_m: float
    B_center_T: float
    poloidal_length_m: float


def case_p(Q=Q_DEFAULT, V_min=V_MIN_DEFAULT, V_max=V_MAX_DEFAULT, poloidal_length=1.0) -> CaseP:
    R_center = DEMO["R_outboard_fw_m"] + 0.5 * DEMO["blanket_depth_outboard_m"]
    B_center = B_FW * DEMO["R_outboard_fw_m"] / R_center
    box = DesignBox(
        Q=Q, mu=PBLI["mu"], sigma=PBLI["sigma"], V_min=V_min, V_max=V_max,
        beta_lo=BETA_LO, beta_hi=BETA_HI,
        stations=(Station(B=B_center, L=poloidal_length, label="uniform (cross-duct 1/R neglected)"),),
    )
    return CaseP(box=box, R_center_m=R_center, B_center_T=B_center, poloidal_length_m=poloidal_length)


def inputs_record() -> dict:
    """Everything the figures/tables need to cite, frozen before any result is looked at."""
    return {
        "pbli": PBLI,
        "demo_geometry": DEMO,
        "B_fw_T": B_FW,
        "Q_m3s": Q_DEFAULT,
        "V_min_default_ms": V_MIN_DEFAULT,
        "V_max_default_ms": V_MAX_DEFAULT,
        "V_min_sweep_ms": list(V_MIN_SWEEP),
        "beta_bounds": [BETA_LO, BETA_HI],
    }
