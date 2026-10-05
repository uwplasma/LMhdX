"""5A.A9: cost of a new aspect on main, cold and warm, by both routes (Ha 200, 48 x 48 cells).

(i)  the PoC's jit(value_and_grad(channel_flow_response)) on a fresh ChannelProblem;
(ii) design.linear_flow_response(CaseSpec) at a concrete field scale, whose compiled program is
     shared by every problem of one shape from the second problem on (steady.shared_or_embedded).
Each call is followed by a repeat for the warm time. Steps are the CG restarts of an eager solve.
"""

import dataclasses
import time

import jax

jax.config.update("jax_enable_x64", True)

from ductopt.physics import _build_problem, cache_stats  # noqa: E402

import lmhdx  # noqa: E402
from lmhdx.cases import MagneticFieldSpec  # noqa: E402
from lmhdx.fully_developed import channel_flow_response, linear_flow_response  # noqa: E402
from lmhdx.steady import solve_steady_state  # noqa: E402

HA, CELLS, LAYER = 200.0, 48, 6
BETAS = (0.30, 0.25, 0.20, 0.17, 0.15, 0.13, 0.11)


def timed(fn):
    t = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    return out, time.perf_counter() - t


print(f"Ha {HA:.0f}, {CELLS}x{CELLS} cells, {jax.__version__}, host cpu")
print("route (i): jit(value_and_grad(channel_flow_response)), one problem per beta")
for beta in BETAS:
    t = time.perf_counter()
    problem = _build_problem(beta, HA, CELLS, LAYER)
    build = time.perf_counter() - t
    fn = jax.jit(
        jax.value_and_grad(
            lambda s: channel_flow_response(problem, magnetic_field_scale=s).flow_per_unit_drive
        )
    )
    c0 = cache_stats()["compiles"]
    (q, _), cold = timed(lambda: fn(1.0))
    _, warm = timed(lambda: fn(1.0))
    steps = int(solve_steady_state(problem, forcing=(1.0, 0.0, 0.0), field_scale=1.0).steps)
    print(
        f"  beta {beta:.2f}: build {build:.2f} s, cold {cold:6.2f} s, warm {warm * 1e3:6.1f} ms, "
        f"compiles {cache_stats()['compiles'] - c0}, steps {steps}, q {float(q):.6e}"
    )

print("route (ii): linear_flow_response(CaseSpec) at a concrete field scale")
for i, beta in enumerate(BETAS):
    case = lmhdx.make_hartmann_case(ha=HA, width=2.0, height=2.0 * beta, ny=CELLS, nz=CELLS)
    case = dataclasses.replace(case, magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, HA, 0.0)))
    c0 = cache_stats()["compiles"]
    r, cold = timed(lambda: linear_flow_response(case))
    _, warm = timed(lambda: linear_flow_response(case))
    print(
        f"  #{i + 1} beta {beta:.2f}: first call {cold:6.2f} s, warm {warm * 1e3:6.1f} ms, "
        f"compiles {cache_stats()['compiles'] - c0}, q {float(r.flow_per_unit_drive):.6e}"
    )
