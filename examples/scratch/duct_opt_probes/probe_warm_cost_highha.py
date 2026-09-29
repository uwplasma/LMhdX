"""5A.C exit (f): warm solve time of one flow evaluation at Ha* 1,229 (H 300) and 6,109 (H 1000) on the same 48/6 mesh."""

import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from ductopt.physics import q_only  # noqa: E402

for H, beta in ((300.0, 0.06015), (1000.0, 0.02680)):
    ha = H / np.sqrt(beta)
    q_only(beta, ha, ha, 48, 6)  # compile
    times = []
    for scale in (1.0, 0.99, 1.01, 0.995, 1.005):
        start = time.perf_counter()
        q_only(beta, ha * scale, ha, 48, 6)
        times.append(time.perf_counter() - start)
    print(f"H {H:5.0f}: Ha* {ha:7.1f}, warm solve {1e3 * np.median(times):6.1f} ms (median of 5)", flush=True)
