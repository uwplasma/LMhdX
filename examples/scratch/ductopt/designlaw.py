"""The dimensionless design-law sweep: beta* sqrt(Ha*)/a* against Ha* (F2, T3).

Reproduces stage_2_plan.txt's P9 measurement (examples/scratch/duct_opt_probes/
probe_design_law.py) inside the package, with mesh-refinement error bars
(Roache 1994 [O14]; Celik et al. 2008 [O15]) at three of the H values.

Convention here matches P9 exactly (not the physical (u, w) box of
``physics.DesignBox``): the AREA is held fixed at 4 (mu = sigma = 1, a =
1/sqrt(beta)), and ``H`` is the Hartmann number of the SQUARE duct at that
fixed area; the actual Hartmann number at the optimum, ``Ha*``, differs
because ``a`` grows as ``beta`` shrinks.
"""

from __future__ import annotations

import numpy as np

from .physics import q_only


def dp_dx(beta: float, H: float, cells: int, cells_in_layer: int) -> tuple[float, float]:
    """Return ``(dp/dx, Ha)`` at fixed area 4, mu = sigma = 1, forcing = 1."""
    a = 1.0 / np.sqrt(beta)
    ha = H * a
    q = q_only(beta, ha, ha, cells, cells_in_layer)  # ha_mesh = ha exactly: this IS the design point
    return (4.0 * beta / q) / a**2, ha


def find_beta_star(
    H: float, cells: int, cells_in_layer: int, *, guess: float | None = None
) -> dict:
    """Quadratic-fit bracket search for the fixed-area optimum at Hartmann number ``H``.

    Nine samples around the Shercliff-formula guess ``(2.13/sqrt(H))**(4/3)``
    (stage_2_plan.txt Section 4.3), refined with one more quadratic fit
    around the best of the first pass.
    """
    if guess is None:
        guess = min((2.13 / np.sqrt(H)) ** (4.0 / 3.0), 1.0)
    factors = np.array([0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0])
    betas = guess * factors
    values = np.array([dp_dx(b, H, cells, cells_in_layer)[0] for b in betas])
    beta_star, dp_star, interior = _quad_refine(betas, values)
    if not interior:  # the first bracket missed the minimum; recentre once
        betas2 = beta_star * factors
        values2 = np.array([dp_dx(b, H, cells, cells_in_layer)[0] for b in betas2])
        beta_star, dp_star, interior = _quad_refine(betas2, values2)
    dp_square, ha_square = dp_dx(1.0, H, cells, cells_in_layer)
    _, ha_star = dp_dx(beta_star, H, cells, cells_in_layer)
    return {
        "H": H,
        "beta_star": beta_star,
        "Ha_star": ha_star,
        "s_star": beta_star * np.sqrt(ha_star),
        "dp_star": dp_star,
        "dp_square": dp_square,
        "reduction": 1.0 - dp_star / dp_square,
        "interior": interior,
        "n_evals": len(betas) + (0 if interior else len(betas)),
    }


def _quad_refine(betas: np.ndarray, values: np.ndarray) -> tuple[float, float, bool]:
    i = int(np.argmin(values))
    interior = 0 < i < len(betas) - 1
    if not interior:
        return float(betas[i]), float(values[i]), False
    x = np.log(betas[i - 1 : i + 2])
    y = values[i - 1 : i + 2]
    c2, c1, c0 = np.polyfit(x, y, 2)
    if c2 <= 0.0:
        return float(betas[i]), float(values[i]), False
    beta_star = float(np.exp(-c1 / (2.0 * c2)))
    dp_star = float(c0 - c1**2 / (4.0 * c2))
    return beta_star, dp_star, True


def sweep(
    H_values: tuple[float, ...],
    cells: int,
    cells_in_layer: int,
    refine_at: tuple[float, ...] = (),
    refine_cells: tuple[int, ...] = (),
) -> list[dict]:
    """Run ``find_beta_star`` at every H, with extra mesh resolutions at ``refine_at``."""
    rows = []
    guess = None
    for H in H_values:
        row = find_beta_star(H, cells, cells_in_layer, guess=guess)
        guess = row["beta_star"]  # warm-start the next H from this one
        if H in refine_at:
            row["refinement"] = [
                find_beta_star(H, c, cells_in_layer, guess=row["beta_star"]) for c in refine_cells
            ]
        rows.append(row)
    return rows


def richardson_gci(
    values: list[float], cells: list[int], safety_factor: float = 1.25, assumed_order: float = 2.0
) -> dict:
    """Three-mesh GCI (Celik et al. 2008 [O15]; Roache 1994 [O14]).

    ``cells`` ascending (coarse, medium, fine). The classical single-formula
    observed order, ``p = ln|eps32/eps21| / ln(r21)``, assumes a CONSTANT
    refinement ratio (``r21 == r32``); ours is not (48/64/96, ratios 1.333
    and 1.5), and with the small, closely-spaced differences this run
    measured (order 0.02-0.05 relative to the values themselves) that
    formula gave a spurious negative order on both refined H values -- a
    caught mistake, recorded in stage_2_plan.txt.

    The general non-constant-ratio order (Celik et al. 2008, eq. 3-5) needs
    an iterative fixed-point solve and is itself noisy on three points this
    close together. Since LMX's staggered discretization is independently
    verified second order elsewhere in the codebase (plan.md's own
    convergence-order measurements throughout Phase 1), the GCI here uses
    that ASSUMED order instead of the fragile 3-point fit, exactly as
    Celik et al. 2008 sec. 2.5.4 allows when the observed order is not
    usable. The observed order is still reported, for the record, whenever
    the refinement ratio is close enough to constant to compute it at all.
    """
    if len(values) != 3 or len(cells) != 3:
        raise ValueError("GCI needs exactly three (value, cell-count) pairs")
    (f1, f2, f3), (n1, n2, n3) = values, cells
    r21, r32 = n2 / n1, n3 / n2
    e32, e21 = f3 - f2, f2 - f1
    observed_order = None
    if e32 != 0.0 and e21 != 0.0 and abs(r21 - r32) / r32 < 0.05:
        observed_order = float(np.log(abs(e32 / e21)) / np.log(r21))
    p = assumed_order
    extrapolated = f3 + (f3 - f2) / (r32**p - 1.0)
    gci = safety_factor * abs((f3 - f2) / f3) / (r32**p - 1.0)
    return {
        "order": p,
        "order_is_assumed": True,
        "order_observed": observed_order,
        "extrapolated": float(extrapolated),
        "gci_fine": float(gci),
    }
