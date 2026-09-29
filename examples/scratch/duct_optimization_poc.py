"""Stage 2 Tier 0 proof of concept: minimum pumping power at fixed flow for a
single straight, insulated PbLi duct in a tokamak-like field, on LMX's
staggered core (branch optimization/duct-poc = main + PR #156).

See stage_2_plan.txt (repo root) for the full derivation, literature and exit
criteria this script implements (Section 5, steps 0.1-0.9). Scratch code:
not part of any PR, no changes to src/lmhdx.

STAGE-BASED, CHECKPOINTED. This sandbox does not keep a detached background
process alive between conversation turns (its VM is recycled between them),
so the pipeline cannot simply be launched once and polled later. Instead it
is a set of small, independent stages, each run to completion inside one
foreground call and its result merged into artifacts/duct_opt/checkpoint.json
immediately -- interrupting the whole run costs at most the one stage in
flight. List stages with ``--list``; run one with ``--stage NAME``; once
every stage is done, ``--finalize`` assembles artifacts/duct_opt/results.json
for duct_opt_figures.py.

    PYTHONPATH=. .venv/bin/python -u examples/scratch/duct_optimization_poc.py --list
    PYTHONPATH=. .venv/bin/python -u examples/scratch/duct_optimization_poc.py --stage <name>
    PYTHONPATH=. .venv/bin/python -u examples/scratch/duct_optimization_poc.py --finalize

Reduced from stage_2_plan.txt Section 5's full scope to fit this sandbox's
per-call budget: 3 Pareto points per case (not 5-6), 5 design-law H values
with 2 refined (not 8 with 3), 5 validity-map gamma values with 1 refined
(not 7 with 2). Noted in the write-up as a scratch-budget reduction, not a
scientific one -- every stage still carries its own exit-criterion checks.
"""

from __future__ import annotations

import argparse
import dataclasses
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import solvax  # noqa: E402
from ductopt import cases, checkpoint, designlaw, validity  # noqa: E402
from ductopt.optimize import fd_step_study, optimize, polish_w, taylor_test  # noqa: E402
from ductopt.physics import (  # noqa: E402
    DesignBox,
    cache_stats,
    objective,
    objective_value,
    spectral_value,
    tilted_flow,
)

import lmhdx  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "artifacts" / "duct_opt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELLS_EXPLORE = 48
CELLS_VERIFY = 96
CELLS_IN_LAYER = 6
CELLS_IN_LAYER_VERIFY = 9

PARETO = {"R_out": cases.V_MIN_SWEEP, "R_in": cases.V_MIN_SWEEP, "P": (cases.V_MIN_DEFAULT,)}  # O3: P is a check
MESHES = ((32, 4), (48, 6), (72, 9))  # C2: a constant refinement ratio of 1.5
DESIGN_LAW_H = (10.0, 20.0, 30.0, 50.0, 100.0, 150.0, 200.0, 300.0)
GCI_H = (30.0, 100.0, 300.0)  # the design-law points that also run the three meshes
HIGH_H = (500.0, 1000.0)  # O4: Ha* about 2,400 and 6,100 by the law
SPECTRAL_POINTS_HIGH = 96  # the reference at Ha* 2,400-6,100; checked against 128 before it is quoted
TILTS = (0.1, 0.2)  # B_p / B_T, 5.7 and 11.3 degrees (C8)
VALIDITY_GAMMAS = (0.005, 0.02, 0.05, 0.1, 0.2)  # delta k a sqrt(Ha) (R1); Tier 0's 0.05-2 x 0.1
VALIDITY_REFINE_GAMMA = (0.2,)  # reduced from 2

CASE_BUILDERS = {"R_out": cases.case_r_outboard, "R_in": cases.case_r_inboard}


# Restated exits of stage_2_plan.txt 5A (v4.1), fixed before any run and written into results.json.
# No tolerance is relaxed after a run; a measurement that cannot be made is reported open.
EXITS = {
    "a_taylor_slope_u": (1.9, 2.1),  # exact u part; the w part is the step study until an exact gradient exists (C6)
    "b_s_star_rel": 0.03,  # beta* sqrt(Ha*) within 3 % of the uniform-field 2.09
    "b_a_star_rel": 0.05,  # a* within 5 % of the closed form of 4.3
    "c_dW_dw_rel": 1.0e-6,  # w-stationarity, exact w-gradient only; V = V_min active with multiplier >= 0
    "d_W_rel": 0.01,  # after re-optimizing on 96/9 against 48/6
    "d_beta_rel": 0.03,
    "e_reference_rel": 0.01,  # rectangular spectral reference at every optimum
    "re_over_ha_max": 200.0,  # exit (f); Ha is not capped (O13), warm time within 2x of Ha 300 on the same mesh
    "gamma_sqrt_ha_max": 0.2,  # 2.5 definition
    "f_cost_factor": 2.0,
    # 5A.B, fixed 2026-09-28 before the sweep (the mesh study at Ha 200, gamma sqrt(Ha) 2 chose 48/6):
    "i_mesh_change_max": 0.10,  # 72/9 and half-spacing must move the excess by <= 10 % of its value
    "i_collapse_rel": 0.20,  # Ha 50 and Ha 200 excess agree within 20 % at matched gamma sqrt(Ha) >= 0.1
    # 5A.C, fixed 2026-09-29 before the runs:
    "c_stationarity_rel": 1.0e-6,  # |dW/dw| / W at the polished optimum, on the scaled-mesh family, h and 2h
    "gci_order_range": (1.0, 3.0),  # the observed order is used inside it, else the assumed 2 (and it is said)
    "f_high_ha_budget_s": 7200.0,  # O4 is a bounded attempt: stop and record at 2 h of CPU
}
# 5A.C1, written before the 2 and 50 mm/s points run (4.3: Ha* ~ V_min^(-2/3)); 5 mm/s is the Tier 0 measurement:
PREDICTIONS = {
    "Ha_star_2mm_outboard": (550.0, 590.0), "Ha_star_2mm_inboard_max": 700.0,
    "Ha_star_50mm": (65.0, 80.0), "Re_over_Ha_50mm": 23.0, "Re_over_sqrtHa_50mm": 190.0,
    "s_star_stations": (2.03, 2.16),
}
RAMP_HA = (50.0, 200.0)
RAMP_GAMMAS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)  # delta k a sqrt(Ha), 2.5 definition, ramp delta 0.2
RAMP_REFINE = (1.0, 2.0)
RAMP_S_STAR = 2.09  # beta* sqrt(Ha*) of the design law (Tier 0, uniform field)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _validity_checks(box: DesignBox, u: float, w: float) -> dict:
    beta = np.exp(w)
    a = box.a_of(u, w)
    V = box.Q / np.exp(u)
    rows = []
    for station in box.stations:
        Ha = station.B * a * np.sqrt(box.sigma / box.mu)
        nu = box.mu / cases.PBLI["rho"]
        Re = V * a / nu
        gamma_sqrt_ha = a / station.R * np.sqrt(Ha) if station.R else 0.0
        rows.append(
            {
                "label": station.label, "B_T": station.B, "Ha": Ha, "Re": Re,
                "Re_over_Ha": Re / Ha, "Re_over_sqrtHa": Re / np.sqrt(Ha), "N_interaction": Ha**2 / Re,
                "gamma_sqrt_Ha": gamma_sqrt_ha, "gamma_sqrt_Ha_leq_0.2": bool(gamma_sqrt_ha <= EXITS["gamma_sqrt_ha_max"]),
                "Re_over_Ha_leq_200": bool(Re / Ha <= EXITS["re_over_ha_max"]),
            }
        )
    return {"a_m": a, "b_m": beta * a, "beta": beta, "V_ms": V, "stations": rows}


def _finer_mesh_check(box: DesignBox, u: float, w: float) -> dict:
    coarse = objective(box, u, w, CELLS_EXPLORE, CELLS_IN_LAYER)
    fine = objective(box, u, w, CELLS_VERIFY, CELLS_IN_LAYER_VERIFY)
    return {
        "W_coarse": coarse["W"], "W_fine": fine["W"],
        "W_relative_change": abs(fine["W"] - coarse["W"]) / abs(fine["W"]),
    }


def _shercliff_reference(beta: float, ha: float) -> float:
    return (2.0 * ha) * (1.0 - 1.0 / ha - 0.852 / (beta * np.sqrt(ha))) ** -1.0


def stage_landscape(tag: str) -> None:
    box = cases.case_p().box if tag == "P" else CASE_BUILDERS[tag]()
    print(f"landscape[{tag}]...", flush=True)
    t0 = time.perf_counter()
    betas = np.geomspace(box.beta_lo, box.beta_hi, 9)
    v_lo = min(box.V_min, *cases.V_MIN_SWEEP) * 0.8  # cover every Pareto V_min line the figure overlays
    v_grid = np.geomspace(v_lo, box.V_max, 10)
    rows = []
    for beta in betas:
        w = float(np.log(beta))
        for v in v_grid:
            u = float(np.log(box.Q / v))
            res = objective(box, u, w, CELLS_EXPLORE, CELLS_IN_LAYER)
            rows.append({"V": v, "beta": beta, "W": res["W"]})
    monotone = True
    for beta in (betas[0], betas[len(betas) // 2], betas[-1]):
        w = float(np.log(beta))
        g_lo = objective(box, box.u_lo, w, CELLS_EXPLORE, CELLS_IN_LAYER)["dW_du"]
        g_hi = objective(box, box.u_hi, w, CELLS_EXPLORE, CELLS_IN_LAYER)["dW_du"]
        if g_lo > 0.0 or g_hi > 0.0:
            monotone = False
    elapsed = time.perf_counter() - t0
    print(f"  {len(rows)} evaluations, {elapsed:.0f} s, monotone-in-area={monotone}", flush=True)
    checkpoint.set_key(("landscape", tag), {"rows": rows, "time_s": elapsed, "monotone_in_area": monotone})


def stage_demo_r_out() -> None:
    box = cases.case_r_outboard()
    v_mid = float(np.sqrt(box.V_min * box.V_max))
    u0 = float(np.log(box.Q / v_mid))
    print("demo optimizer (Case R outboard, start: square duct, mid velocity)...", flush=True)

    def progress(round_index, u, w, W, interior):
        print(f"  round {round_index}: beta={np.exp(w):.4f} W={W:.4e} interior={interior}", flush=True)

    t0 = time.perf_counter()
    demo = optimize(box, u0, 0.0, CELLS_EXPLORE, CELLS_IN_LAYER, progress=progress)
    elapsed = time.perf_counter() - t0
    print(f"  done: {demo.meshes_built} meshes, {elapsed:.0f} s, converged={demo.converged}", flush=True)
    checkpoint.set_key(
        ("demo_optimizer", "R_out"),
        {"u0": u0, "w0": 0.0, "path": demo.path, "meshes_built": demo.meshes_built,
         "converged": demo.converged, "time_s": elapsed, "w_final": demo.w, "u_final": demo.u},
    )


def _pareto_point(box: DesignBox, v_min: float, w_guess: float) -> tuple[dict, float]:
    u0 = float(np.log(box.Q / v_min))
    sweep_box = DesignBox(
        Q=box.Q, mu=box.mu, sigma=box.sigma, V_min=v_min, V_max=box.V_max,
        beta_lo=box.beta_lo, beta_hi=box.beta_hi, stations=box.stations,
    )
    t0 = time.perf_counter()
    opt = optimize(sweep_box, u0, w_guess, CELLS_EXPLORE, CELLS_IN_LAYER, record_path=False)
    elapsed = time.perf_counter() - t0
    tt = taylor_test(sweep_box, opt.u, opt.w, CELLS_EXPLORE, CELLS_IN_LAYER)
    fd_study = fd_step_study(sweep_box, opt.u, opt.w, CELLS_EXPLORE, CELLS_IN_LAYER)
    checks = _validity_checks(sweep_box, opt.u, opt.w)
    finer = _finer_mesh_check(sweep_box, opt.u, opt.w)
    beta = float(np.exp(opt.w))
    row = {
        "V_min": v_min, "u": opt.u, "w": opt.w, "beta": beta, "a_m": sweep_box.a_of(opt.u, opt.w),
        "b_m": beta * sweep_box.a_of(opt.u, opt.w), "W": opt.W, "converged": opt.converged,
        "active_u_bound": opt.active_u_bound, "u_multiplier": opt.u_multiplier,
        "dW_dw_final": opt.dW_dw_final, "meshes_built": opt.meshes_built, "time_s": elapsed,
        "taylor_test": tt, "fd_step_study": fd_study, "validity": checks, "finer_mesh": finer,
    }
    print(
        f"    V_min={v_min * 1000:.1f} mm/s: beta*={beta:.4f} W*={opt.W:.3e} bound={opt.active_u_bound} "
        f"taylor_slope1={tt['slope_first']:.2f} finer_dW={finer['W_relative_change']:.2e} ({elapsed:.0f} s)",
        flush=True,
    )
    return row, opt.w


def stage_pareto_point(tag: str, index: int) -> None:
    """One (case, V_min) Pareto point, checkpointed immediately -- small enough to always
    finish inside one foreground call, and a kill mid-sweep costs at most this one point."""
    box_builder = CASE_BUILDERS.get(tag)
    box = box_builder() if tag != "P" else cases.case_p().box
    v_min = PARETO[tag][index]
    print(f"Pareto point[{tag}] V_min={v_min * 1000:.1f} mm/s...", flush=True)
    ckpt = checkpoint.load()
    existing = ckpt.get("pareto", {}).get(tag, [])
    # warm-start from the nearest already-computed point in V_min (else the demo)
    if existing:
        w_guess = min(existing, key=lambda r: abs(np.log(r["V_min"] / v_min)))["w"]
    else:
        w_guess = ckpt.get("demo_optimizer", {}).get("R_out", {}).get("w_final", 0.0)
    row, _ = _pareto_point(box, v_min, w_guess)
    existing = [r for r in existing if abs(r["V_min"] - v_min) > 1e-15] + [row]
    existing.sort(key=lambda r: r["V_min"])
    checkpoint.set_key(("pareto", tag), existing)


def stage_verify(tag: str, index: int) -> None:
    """C2-C4, C6, exit (c): the discrete optimum of one Pareto point on three meshes and by the reference."""
    v_min = PARETO[tag][index]
    print(f"verify[{tag}] V_min={v_min * 1000:.1f} mm/s...", flush=True)
    box = cases.case_p().box if tag == "P" else CASE_BUILDERS[tag]()
    box = dataclasses.replace(box, V_min=v_min)
    row = next(r for r in checkpoint.load()["pareto"][tag] if abs(r["V_min"] - v_min) < 1e-15)
    u, w0 = box.u_hi, row["w"]
    centre = float(np.exp(w0))
    out = {"V_min": v_min, "w_block_coordinate": w0, "meshes": []}
    for cells, layer in MESHES:
        result = polish_w(lambda w, c=cells, ll=layer: objective_value(box, u, w, c, ll, centre), w0)
        result.update(cells=cells, cells_in_layer=layer)
        result["u_multiplier"] = -objective(box, u, result["w_star"], cells, layer, centre)["dW_du"]
        out["meshes"].append(result)
        print(f"  {cells}/{layer}: beta*={result['beta_star']:.5f} W*={result['value_star']:.5e} "
              f"dW/dw={result['dW_dw_rel_h']:.1e} (2h {result['dW_dw_rel_2h']:.1e})", flush=True)
    fv = out["meshes"][1]
    spectral = polish_w(lambda w: spectral_value(box, u, w), w0)
    at_fv = spectral_value(box, u, fv["w_star"])
    out["spectral"] = {**spectral, "W_at_fv_beta": at_fv, "W_fv_over_spectral": fv["value_star"] / at_fv - 1.0,
                       "beta_fv_over_spectral": fv["beta_star"] / spectral["beta_star"] - 1.0}
    out["gci"] = {name: designlaw.richardson_gci([m[key] for m in out["meshes"]], [m["cells"] for m in out["meshes"]],
                                                 order_range=EXITS["gci_order_range"])
                  for name, key in (("W", "value_star"), ("beta", "beta_star"))}
    print(f"  spectral: beta*={spectral['beta_star']:.5f}; FV/spectral W at the FV beta {out['spectral']['W_fv_over_spectral']:+.2e}",
          flush=True)
    checkpoint.set_key(("verify", f"{tag}_{v_min * 1000:g}"), out)


def stage_dlaw(H: float) -> None:
    """C1, C2, O4: the fixed-area optimum on the scaled-mesh family, by the core and by the reference."""
    print(f"design law H={H:.0f}...", flush=True)
    guess = min((2.13 / np.sqrt(H)) ** (4.0 / 3.0), 1.0)
    rough = designlaw.polish_beta_star(H, *MESHES[0], guess, delta=0.15)  # a wide first pass on the coarse mesh
    start = rough["beta_star"] if rough["interior"] else guess
    out = {"H": H, "meshes": []}
    for cells, layer in MESHES if H in GCI_H else MESHES[1:] if H in HIGH_H else (MESHES[1],):
        t0 = time.perf_counter()
        result = designlaw.polish_beta_star(H, cells, layer, start)
        result.update(cells=cells, cells_in_layer=layer, wall_s=time.perf_counter() - t0)
        out["meshes"].append(result)
        print(f"  {cells}/{layer}: beta*={result['beta_star']:.5f} Ha*={result['Ha_star']:.1f} s*={result['s_star']:.4f} "
              f"reduction={100 * result['reduction']:.1f}% ({result['wall_s']:.0f} s)", flush=True)
    out["spectral"] = designlaw.polish_beta_star(H, 0, 0, start, spectral_points=SPECTRAL_POINTS_HIGH if H in HIGH_H else 48)
    print(f"  spectral: beta*={out['spectral']['beta_star']:.5f} s*={out['spectral']['s_star']:.4f}", flush=True)
    if len(out["meshes"]) == 3:
        out["gci"] = {name: designlaw.richardson_gci([m[key] for m in out["meshes"]], [m["cells"] for m in out["meshes"]],
                                                     order_range=EXITS["gci_order_range"])
                      for name, key in (("dp", "dp_star"), ("beta", "beta_star"))}
    checkpoint.set_key(("dlaw", f"{H:.0f}"), out)


def stage_tilt() -> None:
    """C8: W(tilt) / W(aligned) for the default optima and the equal-area squares, on the damped preconditioner."""
    print("tilt robustness...", flush=True)
    rows = []
    for tag in ("R_out", "R_in"):
        box = CASE_BUILDERS[tag]()
        pareto = next(r for r in checkpoint.load()["pareto"][tag] if abs(r["V_min"] - cases.V_MIN_DEFAULT) < 1e-15)
        station = box.stations[len(box.stations) // 2]  # one mid-run field; the ratio barely depends on the station
        area = float(np.exp(pareto["u"]))
        for design, beta in (("optimum", pareto["beta"]), ("square", 1.0)):
            a = float(np.sqrt(area / (4.0 * beta)))
            ha = station.B * a * float(np.sqrt(box.sigma / box.mu))
            base = None
            for tilt in (0.0, *TILTS):
                q, steps, wall = tilted_flow(beta, ha, tilt)
                power = box.mu * box.Q**2 * station.L / (a**4 * q)
                base = base or power
                rows.append({"case": tag, "design": design, "beta": beta, "Ha": ha, "B_p_over_B_T": tilt,
                             "W": power, "W_over_aligned": power / base, "cg_restarts": steps, "wall_s": wall})
                print(f"  {tag} {design} tilt {tilt}: W/W0={power / base:.5f} steps={steps} ({wall:.0f} s)", flush=True)
    checkpoint.set_key(("tilt",), rows)


def stage_validity(gamma: float, refine: bool = False) -> None:
    tag = f"gamma{gamma}" + ("_refine" if refine else "")
    print(f"validity map {tag}...", flush=True)
    beta_star = RAMP_S_STAR / float(np.sqrt(50.0))  # R8: the design-law aspect at this Ha, not Tier 0's Ha-195 optimum
    nx = 32 if refine else 16
    rows = []
    for beta, case_label in ((1.0, "square"), (beta_star, "beta*")):
        row = validity.three_d_excess(gamma, beta, ha_base=50.0, cells=24, cells_in_layer=4, nx=nx)
        row["case"] = case_label
        if refine:
            row["refinement_of"] = "nx16"
        rows.append(row)
        print(
            f"  case={case_label} excess={row['excess_percent']:+.3f}% "
            f"core/side={row['core_over_side_velocity']:.3f} ({row['elapsed_s']:.0f} s)",
            flush=True,
        )
    checkpoint.set_key(("validity_map", tag), rows)


def stage_ramp(ha: float, case: str) -> None:
    """5A.B: 3-D excess on the open duct with a monotone ramp, square or at the design law's own beta*."""
    beta = 1.0 if case == "square" else RAMP_S_STAR / float(np.sqrt(ha))
    print(f"ramp Ha={ha:.0f} {case} beta={beta:.4f}...", flush=True)
    rows = []

    def solve(gamma: float, label: str, **kwargs) -> dict:
        kwargs = {"cells": CELLS_EXPLORE, "cells_in_layer": CELLS_IN_LAYER, **kwargs}
        row = validity.ramp_excess(gamma, beta, ha, **kwargs)
        row.update(case=case, variant=label)
        rows.append(row)
        checkpoint.set_key(("ramp_excess", f"Ha{ha:.0f}_{case}"), rows)
        print(f"  gamma sqrt(Ha)={gamma} {label}: excess={row['excess_percent']:+.4f}% "
              f"x0={row['x0']:.3g} it={row['iterations']} ({row['elapsed_s']:.0f} s)", flush=True)
        return row

    for gamma in RAMP_GAMMAS:
        base = solve(gamma, "base")
        if gamma in RAMP_REFINE:
            solve(gamma, "cross_72_9", cells=72, cells_in_layer=9)
            solve(gamma, "half_spacing", spacing=0.5 * base["spacing"])


def ramp_summary(rows: list[dict]) -> dict:
    """Exit (i): the refinement changes and the Ha collapse test, against the tolerances fixed in EXITS."""
    key = lambda r: (r["case"], r["ha_mid"], r["gamma_sqrt_ha"])  # noqa: E731
    base = {key(r): r for r in rows if r["variant"] == "base"}
    changes = [
        {"case": r["case"], "ha": r["ha_mid"], "gamma_sqrt_ha": r["gamma_sqrt_ha"], "variant": r["variant"],
         "relative_change": abs(r["excess_percent"] - base[key(r)]["excess_percent"]) / abs(base[key(r)]["excess_percent"])}
        for r in rows if r["variant"] != "base"
    ]
    collapse = []
    for case in ("square", "beta_star"):
        for gamma in RAMP_GAMMAS:
            lo, hi = base.get((case, RAMP_HA[0], gamma)), base.get((case, RAMP_HA[-1], gamma))
            if lo and hi and gamma >= 0.1:
                e = sorted((lo["excess_percent"], hi["excess_percent"]))
                collapse.append({"case": case, "gamma_sqrt_ha": gamma, "excess_ha_lo": lo["excess_percent"],
                                 "excess_ha_hi": hi["excess_percent"], "relative_difference": 1.0 - e[0] / e[1]})
    return {
        "mesh_changes": changes,
        "mesh_flagged": [c for c in changes if c["relative_change"] > EXITS["i_mesh_change_max"]],
        "collapse": collapse,
        "collapsed": all(c["relative_difference"] <= EXITS["i_collapse_rel"] for c in collapse),
    }


def stage_case_p_correction() -> None:
    print("Case P exact 1/R field correction...", flush=True)
    from lmhdx.bc import PERIODIC, BoundaryCondition
    from lmhdx.core3d import ChannelProblem, ImposedField
    from lmhdx.design import channel_cross_section_weights, channel_flow_rate
    from lmhdx.grid import Grid, uniform_faces, wall_resolving_faces
    from lmhdx.steady import solve_steady_state

    cp = cases.case_p()
    ckpt = checkpoint.load()
    p_pareto = ckpt.get("pareto", {}).get("P", [])
    default = next((r for r in p_pareto if abs(r["V_min"] - 0.010) < 1e-12), None)
    if default is None:
        raise RuntimeError("run stage_pareto('P') first")
    beta_star, a_star = default["beta"], default["a_m"]

    WALL = BoundaryCondition("neumann")
    sigma, mu = cp.box.sigma, cp.box.mu

    def toroidal_solve(a: float, beta: float, rc: float, B: float) -> dict:
        ha = B * a * np.sqrt(sigma / mu)
        y = wall_resolving_faces(CELLS_EXPLORE, -1.0, 1.0, layer_thickness=1.0 / ha, cells_in_layer=CELLS_IN_LAYER, max_ratio=None)
        z = wall_resolving_faces(CELLS_EXPLORE, -beta, beta, layer_thickness=1.0 / np.sqrt(ha), cells_in_layer=CELLS_IN_LAYER, max_ratio=None)
        grid = Grid(uniform_faces(1, 0.0, 1.0), y, z)
        rc_over_a = rc / a
        yv, zv = np.asarray(grid.y_faces), np.asarray(grid.z_faces)
        A = ha * rc_over_a * 0.5 * np.log((rc_over_a + zv[None, :]) ** 2 + yv[:, None] ** 2)
        by = np.diff(A, axis=1) / grid.widths[2][None, :]
        bz = -np.diff(A, axis=0) / grid.widths[1][:, None]
        faces = (np.zeros(grid.face_shape(0)), by[None], bz[None])
        centres = (
            np.zeros(grid.shape), 0.5 * (faces[1][:, :-1] + faces[1][:, 1:]),
            0.5 * (faces[2][:, :, :-1] + faces[2][:, :, 1:]),
        )
        field = ImposedField(grid, centres, faces)
        common = dict(grid=grid, conditions=(BoundaryCondition(PERIODIC), WALL, WALL), conductivity=1.0,
                      forcing=(1.0, 0.0, 0.0), dt=1.0)
        problem_uniform = ChannelProblem(magnetic_field=(0.0, ha, 0.0), **common)
        problem_exact = ChannelProblem(magnetic_field=field, **common)
        u_uniform = solve_steady_state(problem_uniform, forcing=(1.0, 0.0, 0.0)).velocity[0].data[0]
        u_exact = solve_steady_state(problem_exact, forcing=(1.0, 0.0, 0.0)).velocity[0].data[0]
        q_uniform = float(channel_flow_rate(problem_uniform, u_uniform))
        q_exact = float(channel_flow_rate(problem_exact, u_exact))
        wq = channel_cross_section_weights(problem_exact) * u_exact  # R2: weight by the cell areas
        centroid = float(jnp.sum(wq * grid.centers[2][None, :]) / jnp.sum(wq)) / beta
        wq0 = channel_cross_section_weights(problem_uniform) * u_uniform
        centroid_uniform = float(jnp.sum(wq0 * grid.centers[2][None, :]) / jnp.sum(wq0)) / beta  # 0 by symmetry
        return {"a_m": a, "beta": beta, "Rc_over_a": rc_over_a, "Ha": ha, "q_uniform": q_uniform,
                "centroid_uniform_fraction": centroid_uniform,
                "q_exact": q_exact, "flow_ratio": q_exact / q_uniform, "flow_centroid_fraction": centroid}

    t0 = time.perf_counter()
    square = toroidal_solve(a_star, 1.0, cp.R_center_m, cp.B_center_T)
    optimum = toroidal_solve(a_star, beta_star, cp.R_center_m, cp.B_center_T)
    elapsed = time.perf_counter() - t0
    print(f"  square: flow_ratio={square['flow_ratio']:.6f} centroid={square['flow_centroid_fraction']:+.4f}", flush=True)
    print(f"  optimum: flow_ratio={optimum['flow_ratio']:.6f} centroid={optimum['flow_centroid_fraction']:+.4f}"
          f" ({elapsed:.0f} s)", flush=True)
    checkpoint.set_key(("case_p_correction",), {"square": square, "at_optimum": optimum, "time_s": elapsed})


STAGES: list[str] = (
    ["landscape_R_out", "demo_R_out", "landscape_R_in", "landscape_P"]
    + [f"pareto_{tag}_{i}" for tag, sweep in PARETO.items() for i in range(len(sweep))]
    + [f"verify_{tag}_{i}" for tag, sweep in PARETO.items() for i in range(len(sweep))]
    + [f"dlaw_{H:.0f}" for H in DESIGN_LAW_H]
    + ["tilt"]
    + [f"validity_{g}" for g in VALIDITY_GAMMAS]
    + [f"validity_refine_{g}" for g in VALIDITY_REFINE_GAMMA]
    + [f"ramp_{ha:.0f}_{case}" for ha in RAMP_HA for case in ("square", "beta_star")]
    + ["case_p_correction"]
)


def run_stage(name: str) -> None:
    if name == "landscape_R_out":
        stage_landscape("R_out")
    elif name == "landscape_R_in":
        stage_landscape("R_in")
    elif name == "landscape_P":
        stage_landscape("P")
    elif name == "demo_R_out":
        stage_demo_r_out()
    elif name.startswith("pareto_"):
        rest = name.removeprefix("pareto_")
        tag, idx_str = rest.rsplit("_", 1)
        stage_pareto_point(tag, int(idx_str))
    elif name.startswith("verify_"):
        tag, idx_str = name.removeprefix("verify_").rsplit("_", 1)
        stage_verify(tag, int(idx_str))
    elif name.startswith("dlaw_"):
        stage_dlaw(float(name.removeprefix("dlaw_")))
    elif name == "tilt":
        stage_tilt()
    elif name.startswith("validity_refine_"):
        stage_validity(float(name.removeprefix("validity_refine_")), refine=True)
    elif name.startswith("validity_"):
        stage_validity(float(name.removeprefix("validity_")))
    elif name.startswith("ramp_"):
        ha, case = name.removeprefix("ramp_").split("_", 1)
        stage_ramp(float(ha), case)
    elif name == "case_p_correction":
        stage_case_p_correction()
    else:
        raise ValueError(f"unknown stage {name!r}")


def finalize() -> None:
    """Assemble artifacts/duct_opt/results.json from the checkpoint, for duct_opt_figures.py."""
    ckpt = checkpoint.load()
    missing = [s for s in STAGES if not _stage_done(ckpt, s)]
    if missing:
        print(f"NOT finalizing: {len(missing)} stage(s) still missing: {missing}", flush=True)
        sys.exit(1)

    results = {
        "meta": {
            "git_sha": _git_sha(), "jax_version": jax.__version__, "solvax_version": solvax.__version__,
            "python_version": platform.python_version(), "host": platform.platform(),
            "cells_explore": CELLS_EXPLORE, "cells_verify": CELLS_VERIFY,
            "exits": EXITS, "stage_cost": ckpt.get("stage_cost", {}), "reduced_scope_note":
                "V_min sweep 3 points, design-law 5+2 H values, validity map 5+1 gamma values "
                "(stage_2_plan.txt's full counts are 5-6, 8+3, 7+2 respectively; reduced for this "
                "sandbox's per-call time budget, not for scientific reasons).",
        },
        "inputs": cases.inputs_record(),
    }
    for tag, key in (("R_out", "case_r_outboard"), ("R_in", "case_r_inboard"), ("P", "case_p")):
        box = cases.case_p().box if tag == "P" else CASE_BUILDERS[tag]()
        for row in ckpt["pareto"][tag]:  # flags follow EXITS as they stand now, not as the run stored them
            row["validity"] = _validity_checks(box, row["u"], row["w"])
        entry = {"name": key, "landscape": ckpt["landscape"][tag]["rows"],
                  "landscape_time_s": ckpt["landscape"][tag]["time_s"],
                  "W_monotone_decreasing_in_area": ckpt["landscape"][tag]["monotone_in_area"],
                  "pareto": ckpt["pareto"][tag]}
        if tag == "R_out":
            entry["demo_optimizer"] = ckpt["demo_optimizer"]["R_out"]
        results[key] = entry

    def design_row(out: dict) -> dict:
        base = next(m for m in out["meshes"] if m["cells"] == CELLS_EXPLORE)
        return {**base, "meshes": out["meshes"], "spectral": out["spectral"], "gci": out.get("gci")}

    results["design_law"] = {"rows": [design_row(ckpt["dlaw"][f"{H:.0f}"]) for H in DESIGN_LAW_H],
                             "high_ha": [design_row(ckpt["dlaw"][f"{H:.0f}"]) for H in HIGH_H if f"{H:.0f}" in ckpt["dlaw"]],
                             "time_s": sum(m["wall_s"] for out in ckpt["dlaw"].values() for m in out["meshes"])}
    results["verify"] = ckpt["verify"]
    results["tilt"] = ckpt["tilt"]

    validity_rows = []
    for g in VALIDITY_GAMMAS:
        validity_rows.extend(ckpt["validity_map"][f"gamma{g}"])
    for g in VALIDITY_REFINE_GAMMA:
        validity_rows.extend(ckpt["validity_map"][f"gamma{g}_refine"])
    results["validity_map"] = {"rows": validity_rows,
                               "time_s": sum(r.get("elapsed_s", 0) for r in validity_rows)}

    ramp_rows = [r for ha in RAMP_HA for case in ("square", "beta_star")
                 for r in ckpt["ramp_excess"][f"Ha{ha:.0f}_{case}"]]
    results["ramp_excess"] = {"rows": ramp_rows, "summary": ramp_summary(ramp_rows),
                              "time_s": sum(r["elapsed_s"] for r in ramp_rows)}
    results["case_p_correction"] = {"square": ckpt["case_p_correction"]["square"],
                                     "at_optimum": ckpt["case_p_correction"]["at_optimum"]}
    results["meta"]["total_time_s"] = (
        sum(results[k]["landscape_time_s"] for k in ("case_r_outboard", "case_r_inboard", "case_p"))
        + results["case_r_outboard"]["demo_optimizer"]["time_s"]
        + sum(sum(r["time_s"] for r in results[k]["pareto"]) for k in ("case_r_outboard", "case_r_inboard", "case_p"))
        + results["design_law"]["time_s"]
        + results["validity_map"]["time_s"]
        + results["ramp_excess"]["time_s"]
        + ckpt["case_p_correction"]["time_s"]
    )

    import json as _json
    out_path = OUT_DIR / "results.json"
    with out_path.open("w") as fh:
        _json.dump(results, fh, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    print(f"wrote {out_path}", flush=True)


def _stage_done(ckpt: dict, name: str) -> bool:
    if name.startswith("landscape_"):
        return name.removeprefix("landscape_") in ckpt.get("landscape", {})
    if name == "demo_R_out":
        return "R_out" in ckpt.get("demo_optimizer", {})
    if name.startswith("pareto_"):
        rest = name.removeprefix("pareto_")
        tag, idx_str = rest.rsplit("_", 1)
        v_min = PARETO[tag][int(idx_str)]
        rows = ckpt.get("pareto", {}).get(tag, [])
        return any(abs(r["V_min"] - v_min) < 1e-15 for r in rows)
    if name.startswith("verify_"):
        tag, idx_str = name.removeprefix("verify_").rsplit("_", 1)
        return f"{tag}_{PARETO[tag][int(idx_str)] * 1000:g}" in ckpt.get("verify", {})
    if name.startswith("dlaw_"):
        return f"{float(name.removeprefix('dlaw_')):.0f}" in ckpt.get("dlaw", {})
    if name == "tilt":
        return "tilt" in ckpt
    if name.startswith("validity_refine_"):
        return f"gamma{name.removeprefix('validity_refine_')}_refine" in ckpt.get("validity_map", {})
    if name.startswith("validity_"):
        return f"gamma{name.removeprefix('validity_')}" in ckpt.get("validity_map", {})
    if name.startswith("ramp_"):
        ha, case = name.removeprefix("ramp_").split("_", 1)
        rows = ckpt.get("ramp_excess", {}).get(f"Ha{float(ha):.0f}_{case}", [])
        return len(rows) == len(RAMP_GAMMAS) + 2 * len(RAMP_REFINE)
    if name == "case_p_correction":
        return "case_p_correction" in ckpt
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    if args.list:
        for s in STAGES:
            print(s)
        return
    if args.status:
        ckpt = checkpoint.load()
        for s in STAGES:
            print(f"{'DONE' if _stage_done(ckpt, s) else '    '}  {s}")
        return
    if args.finalize:
        finalize()
        return
    if args.stage is None:
        parser.error("pass --stage NAME, --list, --status or --finalize")

    print(f"lmhdx {lmhdx.__file__}, JAX {jax.__version__}, SOLVAX {solvax.__version__}", flush=True)
    print(f"host: {platform.platform()}, python {platform.python_version()}", flush=True)
    before, t0 = cache_stats(), time.perf_counter()
    run_stage(args.stage)
    after = cache_stats()
    checkpoint.set_key(("stage_cost", args.stage), {  # R7: measured in the process that ran the stage
        "wall_s": time.perf_counter() - t0, "meshes_built": after["meshes_built"] - before["meshes_built"],
        "compiles": after["compiles"] - before["compiles"], "compile_s": after["compile_s"] - before["compile_s"]})


if __name__ == "__main__":
    main()
