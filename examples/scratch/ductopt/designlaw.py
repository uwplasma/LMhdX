"""The dimensionless design-law sweep: beta* sqrt(Ha*)/a* against Ha* (F2, T3).

Reproduces stage_2_plan.txt's P9 measurement (examples/scratch/duct_opt_probes/
probe_design_law.py) with the optimum located on the O12 scaled-mesh family and
mesh-refinement error bars (Roache 1994 [O14]; Celik et al. 2008 [O15]).

Convention here matches P9 exactly (not the physical (u, w) box of
``physics.DesignBox``): the AREA is held fixed at 4 (mu = sigma = 1, a =
1/sqrt(beta)), and ``H`` is the Hartmann number of the SQUARE duct at that
fixed area; the actual Hartmann number at the optimum, ``Ha*``, differs
because ``a`` grows as ``beta`` shrinks.
"""

from __future__ import annotations

import numpy as np

from .optimize import polish_w
from .physics import q_only


def dp_dx(
    beta: float, H: float, cells: int, cells_in_layer: int, centre: float = 0.0, ha_mesh: float | None = None
) -> tuple[float, float]:
    """Return ``(dp/dx, Ha)`` at fixed area 4, mu = sigma = 1, forcing = 1."""
    a = 1.0 / np.sqrt(beta)
    ha = H * a
    q = q_only(beta, ha, ha_mesh or ha, cells, cells_in_layer, centre)  # default: ha_mesh = ha, this IS the design point
    return (4.0 * beta / q) / a**2, ha


def polish_beta_star(
    H: float, cells: int, cells_in_layer: int, guess: float, *, spectral_points: int = 0, delta: float = 0.05
) -> dict:
    """The fixed-area optimum on the scaled-mesh family centred at ``guess`` (5A.C, O12).

    ``spectral_points`` > 0 uses the independent spectral reference instead of the staggered core.
    """
    if spectral_points:
        from validation.shercliff import quadrant_flow_rate

        def value(w):
            beta = float(np.exp(w))
            a = 1.0 / np.sqrt(beta)
            return (4.0 * beta / (4.0 * beta * quadrant_flow_rate(H * a, spectral_points, aspect=beta))) / a**2
    else:
        ha_mesh = 1.1 * H / np.sqrt(guess)

        def value(w):
            return dp_dx(float(np.exp(w)), H, cells, cells_in_layer, guess, ha_mesh)[0]

    result = polish_w(value, float(np.log(guess)), delta=delta)
    if result["interior"]:
        beta = result["beta_star"]
        ha_star = H / np.sqrt(beta)
        dp_square = value(0.0) if spectral_points else dp_dx(1.0, H, cells, cells_in_layer, guess, ha_mesh)[0]
        result.update(H=H, Ha_star=ha_star, s_star=beta * np.sqrt(ha_star), dp_star=result["value_star"],
                      dp_square=dp_square, reduction=1.0 - result["value_star"] / dp_square)
    return result


def richardson_gci(
    values: list[float], cells: list[int], safety_factor: float = 1.25, assumed_order: float = 2.0,
    order_range: tuple[float, float] = (1.0, 3.0),
) -> dict:
    """Three-mesh GCI (Celik et al. 2008 [O15]; Roache 1994 [O14]) for cells ``cells``, coarse to fine.

    At a constant refinement ratio the observed order is ``p = ln|(f2 - f1) / (f3 - f2)| / ln r``. It is used
    inside ``order_range``; outside it, or for a non-monotone series, the assumed order is used and
    ``order_is_assumed`` says so (R6: Tier 0 held the order at 2 without stating that it could not be observed).
    """
    if len(values) != 3 or len(cells) != 3:
        raise ValueError("GCI needs exactly three (value, cell-count) pairs")
    (f1, f2, f3), (n1, n2, n3) = values, cells
    r21, r32 = n2 / n1, n3 / n2
    e21, e32 = f2 - f1, f3 - f2
    observed = None
    if e21 * e32 > 0.0 and abs(r21 - r32) / r32 < 0.05:
        observed = float(np.log(abs(e21 / e32)) / np.log(r32))
    usable = observed is not None and order_range[0] <= observed <= order_range[1]
    p = observed if usable else assumed_order
    return {
        "order": p, "order_is_assumed": not usable, "order_observed": observed,
        "extrapolated": float(f3 + e32 / (r32**p - 1.0)),
        "gci_fine": float(safety_factor * abs(e32 / f3) / (r32**p - 1.0)),
    }
