"""Block-coordinate optimizer on (u, w) = (ln area, ln aspect).

Why block-coordinate. The mesh a station needs depends only on ``w`` (the
aspect ratio); any ``u`` (area/velocity) at fixed ``w`` is a WARM, mesh-free
evaluation through the traced field scale (physics.objective). So an update
that changes only ``u`` is essentially free, and an update that changes ``w``
costs a fresh mesh (a fresh JIT compile, ~5-10 s on this host). Each outer
round therefore does:

1. an exact, mesh-free 1-D optimization in ``u`` at the current ``w``
   (bisection on the closed-form ``dW/du``, projected to the box);
2. a quadratic-fit bracket search in ``w`` (Nocedal & Wright ch. 8's
   derivative-free bracketing, [O7]; the same method as
   ``designlaw.find_beta_star``, which stage_2_plan.txt's P9 already
   validated): 5 samples around the current ``w``, a parabola through the
   three lowest, re-centre and repeat if the minimum falls on the bracket's
   edge.

An earlier version used a projected-Newton step with Armijo backtracking
(Bertsekas 1976/1982, [O5, O6]); it is not used because Armijo trials probe a
CONTINUUM of nearby ``w`` values, each one a cache miss (a fresh mesh), which
measured at 900+ seconds for one optimization here (recorded as a caught
mistake in stage_2_plan.txt, not shipped). The bracket search reuses the
SAME handful of ``w`` values across rounds (the recentred bracket overlaps
the previous one), so most of its evaluations hit the cache warm.

The KKT check (Nocedal & Wright ch. 12 [O7]) reports which bound is active
and its multiplier from ``dW/du`` at the converged point, and a
finite-difference ``dW/dw`` (interior variable, so its gradient must be
small).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .physics import DesignBox, objective, objective_value


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _bisect_u(box: DesignBox, w: float, cells: int, cells_in_layer: int, u_lo: float, u_hi: float) -> float:
    """Find the interior root of dW/du in [u_lo, u_hi], or return the active bound.

    dW/du is evaluated warm (same mesh, only field_scale changes), so this
    costs no new mesh regardless of how many bisection steps it takes.
    """
    g_lo = objective(box, u_lo, w, cells, cells_in_layer)["dW_du"]
    g_hi = objective(box, u_hi, w, cells, cells_in_layer)["dW_du"]
    if g_lo >= 0.0:  # W already increasing at the smallest area: stay at the lower bound
        return u_lo
    if g_hi <= 0.0:  # W still decreasing at the largest area: the upper bound is optimal
        return u_hi
    lo, hi = u_lo, u_hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        g = objective(box, mid, w, cells, cells_in_layer)["dW_du"]
        if g > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1.0e-10:
            break
    return 0.5 * (lo + hi)


def _quad_bracket(
    box: DesignBox, u: float, w_center: float, w_lo: float, w_hi: float, cells: int, cells_in_layer: int,
    factors: np.ndarray,
) -> tuple[float, float, bool, list[dict], int]:
    """One bracket of 5 log-spaced samples around ``w_center``; a parabola through the best 3."""
    ws = np.clip(w_center + np.log(factors), w_lo, w_hi)
    ws = np.unique(ws)  # bound-clipping can collapse samples near an edge
    values = np.array([objective_value(box, u, w, cells, cells_in_layer) for w in ws])
    evaluated = [{"w": float(w), "W": float(v)} for w, v in zip(ws, values)]
    i = int(np.argmin(values))
    if i == 0 or i == len(ws) - 1 or len(ws) < 3:
        return float(ws[i]), float(values[i]), False, evaluated, len(ws)
    x, y = ws[i - 1 : i + 2], values[i - 1 : i + 2]
    c2, c1, c0 = np.polyfit(x, y, 2)
    if c2 <= 0.0:
        return float(ws[i]), float(values[i]), False, evaluated, len(ws)
    w_star = _clip(-c1 / (2.0 * c2), w_lo, w_hi)
    W_star = float(c0 - c1**2 / (4.0 * c2))
    return w_star, W_star, True, evaluated, len(ws)


@dataclass
class OptimizeResult:
    u: float
    w: float
    W: float
    path: list = field(default_factory=list)  # each round's bracket evaluations and the chosen point
    meshes_built: int = 0
    converged: bool = False
    active_u_bound: str = "interior"  # "V_min" (u at u_hi), "V_max" (u at u_lo), or "interior"
    u_multiplier: float = 0.0
    dW_dw_final: float = 0.0


_BRACKET_FACTORS = np.array([0.6, 0.8, 1.0, 1.25, 1.6])  # log-spaced around the current w


def optimize(
    box: DesignBox,
    u0: float,
    w0: float,
    cells: int,
    cells_in_layer: int,
    *,
    max_rounds: int = 8,
    w_tol: float = 1.0e-3,
    record_path: bool = True,
    progress=None,
) -> OptimizeResult:
    """Minimize W(u, w) on [u_lo, u_hi] x [ln(beta_lo), ln(beta_hi)]."""
    u_lo, u_hi = box.u_lo, box.u_hi
    w_lo, w_hi = float(np.log(box.beta_lo)), float(np.log(box.beta_hi))
    u, w = _clip(u0, u_lo, u_hi), _clip(w0, w_lo, w_hi)
    path = []
    meshes_built = 0
    for round_index in range(max_rounds):
        u = _bisect_u(box, w, cells, cells_in_layer, u_lo, u_hi)
        w_new, W_new, interior, evaluated, n_new = _quad_bracket(
            box, u, w, w_lo, w_hi, cells, cells_in_layer, _BRACKET_FACTORS
        )
        meshes_built += n_new
        if record_path:
            path.append({"round": round_index, "u": u, "w_center": w, "w_next": w_new, "W": W_new,
                         "interior": interior, "evaluations": evaluated})
        if progress is not None:
            progress(round_index, u, w_new, W_new, interior)
        moved = abs(w_new - w)
        w = w_new
        if interior and moved < w_tol:
            u_final = _bisect_u(box, w, cells, cells_in_layer, u_lo, u_hi)
            final = objective(box, u_final, w, cells, cells_in_layer)
            dW_dw = _finite_diff_dw(box, u_final, w, cells, cells_in_layer, meshes_built_ref=[meshes_built])
            return _finish(box, u_final, w, final, path, meshes_built + 2, True, dW_dw[0], u_lo, u_hi)

    u_final = _bisect_u(box, w, cells, cells_in_layer, u_lo, u_hi)
    final = objective(box, u_final, w, cells, cells_in_layer)
    dW_dw, meshes_built = _finite_diff_dw(box, u_final, w, cells, cells_in_layer, meshes_built_ref=[meshes_built])
    converged = abs(dW_dw) <= 1.0e-4 * max(abs(final["W"]), 1e-300)
    return _finish(box, u_final, w, final, path, meshes_built, converged, dW_dw, u_lo, u_hi)


def _finite_diff_dw(box, u, w, cells, cells_in_layer, meshes_built_ref, h=2.0e-3):
    Wp = objective_value(box, u, w + h, cells, cells_in_layer)
    Wm = objective_value(box, u, w - h, cells, cells_in_layer)
    meshes_built_ref[0] += 2
    return (Wp - Wm) / (2.0 * h), meshes_built_ref[0]


def _finish(box, u, w, final, path, meshes_built, converged, dW_dw_final, u_lo, u_hi) -> OptimizeResult:
    if u >= u_hi - 1.0e-9:
        bound, multiplier = "V_min", -final["dW_du"]  # active upper area bound; W would fall further past it
    elif u <= u_lo + 1.0e-9:
        bound, multiplier = "V_max", final["dW_du"]
    else:
        bound, multiplier = "interior", 0.0
    return OptimizeResult(
        u=u,
        w=w,
        W=final["W"],
        path=path,
        meshes_built=meshes_built,
        converged=bool(converged),
        active_u_bound=bound,
        u_multiplier=float(multiplier),
        dW_dw_final=float(dW_dw_final),
    )


def taylor_test(box: DesignBox, u: float, w: float, cells: int, cells_in_layer: int) -> dict:
    """Taylor remainder test (Farrell et al. 2013, SIAM J. Sci. Comput., [O10]), on the
    EXACT, mesh-free gradient in ``u`` alone: perturbing ``w`` changes the mesh
    (a fresh ``wall_resolving_faces`` build and a fresh CG solve, tolerance
    1e-9 relative), so a joint-direction test was found to hit a genuine
    discretization/solve-tolerance noise floor around 1e-6 relative to W well
    before machine precision -- not the analytic gradient failing, but a real
    limit of comparing two independently meshed points at floating-point
    precision (recorded in stage_2_plan.txt: the joint test's first-order
    slope sat at 0.5-0.6, not 2, and the remainder stopped decreasing, then
    stayed flat, exactly the signature of a noise floor rather than a wrong
    gradient). The exact ``u``-gradient (a single warm JAX ``value_and_grad``
    call, no mesh change) is unaffected and gives a clean second-order slope;
    see ``fd_step_study`` for the appropriate check on the ``w`` gradient.
    """
    base = objective(box, u, w, cells, cells_in_layer)
    W0, grad = base["W"], base["dW_du"]
    hs = [1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5]
    zeroth, first = [], []
    for h in hs:
        Wh = objective(box, u + h, w, cells, cells_in_layer)["W"]
        zeroth.append(abs(Wh - W0))
        first.append(abs(Wh - W0 - h * grad))
    return {
        "h": hs,
        "remainder_zeroth": zeroth,
        "remainder_first": first,
        "slope_zeroth": _slope(hs, zeroth),
        "slope_first": _slope(hs, first),
        "gradient": grad,
    }


def fd_step_study(
    box: DesignBox, u: float, w: float, cells: int, cells_in_layer: int,
    hs: tuple[float, ...] = (3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4),
    noise_jump_threshold: float = 0.5,
) -> dict:
    """Step-size sensitivity of the central-difference ``dW/dw`` (Nocedal & Wright
    ch. 8 [O7]): the appropriate check for a finite-difference gradient across a
    mesh change (see ``taylor_test``'s docstring for why a remainder-order test
    does not apply here). ``hs`` is DECREASING; consecutive estimates are
    compared, and the first pair whose relative change exceeds
    ``noise_jump_threshold`` marks where mesh-to-mesh discretization/CG-solve
    noise (not the true derivative) starts to dominate -- expected, since
    ``dW/dw`` is small near a shallow optimum (measured here: about 1e-7
    against W ~ 4e-5, i.e. ~0.3 % relative) while each mesh's own CG solve
    floor is ~1e-9 relative to W, so ``|dW/dw| noise`` grows like (CG floor)
    / h and overtakes the signal below some h. Everything at or above that
    step is the "plateau" the reported spread is computed over; a plain
    ``estimates[-3:-1]`` slice was tried first and picked up exactly this
    noise (a caught mistake, recorded in stage_2_plan.txt).
    """
    estimates = [_finite_diff_dw(box, u, w, cells, cells_in_layer, meshes_built_ref=[0], h=h)[0] for h in hs]
    W0 = objective(box, u, w, cells, cells_in_layer)["W"]
    noise_floor_index = len(hs)
    for i in range(1, len(estimates)):
        denom = max(abs(estimates[i - 1]), 1.0e-300)
        if abs(estimates[i] - estimates[i - 1]) / denom > noise_jump_threshold:
            noise_floor_index = i
            break
    plateau = estimates[:noise_floor_index] if noise_floor_index > 1 else estimates
    spread = (max(plateau) - min(plateau)) / max(abs(sum(plateau) / len(plateau)), 1.0e-300)
    return {
        "h": list(hs),
        "dW_dw": estimates,
        "W": W0,
        "noise_floor_h": hs[noise_floor_index] if noise_floor_index < len(hs) else None,
        "plateau_relative_spread": spread,
    }


def _slope(hs: list[float], values: list[float]) -> float:
    logh = np.log(hs)
    logv = np.log(np.maximum(values, 1.0e-300))
    return float(np.polyfit(logh, logv, 1)[0])


def polish_w(value, w0: float, *, delta: float = 0.05, h: float = 0.01) -> dict:
    """The minimum of a smooth ``value(w)`` (the O12 scaled-mesh family centred at ``w0``).

    A quartic through five points ``w0 + delta * (-2 ... 2)`` locates the minimum; ``dW/dw`` there is read by
    central differences at ``h`` and ``2 h`` (their difference is the noise band), relative to the value.
    """
    ws = w0 + delta * np.arange(-2, 3)
    coefficients = np.polyfit(ws - w0, [value(w) for w in ws], 4)
    stationary = np.roots(np.polyder(coefficients))
    real = stationary[np.abs(stationary.imag) < 1e-12].real
    convex = [r for r in real if np.polyval(np.polyder(coefficients, 2), r) > 0.0 and abs(r) <= 2.0 * delta]
    if not convex:
        return {"interior": False, "w_star": None}
    w_star = w0 + float(min(convex, key=abs))
    v_star = value(w_star)
    slopes = [(value(w_star + step) - value(w_star - step)) / (2.0 * step) / v_star for step in (h, 2.0 * h)]
    return {"interior": True, "w_star": w_star, "beta_star": float(np.exp(w_star)), "value_star": v_star,
            "dW_dw_rel_h": slopes[0], "dW_dw_rel_2h": slopes[1], "delta": delta, "h": h}
