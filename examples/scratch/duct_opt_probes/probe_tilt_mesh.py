"""5A.C8: mesh dependence of W(tilt)/W(aligned) for the Case R outboard optimum (10 mm/s) and its equal-area square."""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from ductopt import cases  # noqa: E402
from ductopt.physics import tilted_flow  # noqa: E402

box = cases.case_r_outboard()
station = box.stations[len(box.stations) // 2]
area = box.Q / cases.V_MIN_DEFAULT
for design, beta in (("optimum", 0.1497), ("square", 1.0)):
    a = float(np.sqrt(area / (4.0 * beta)))
    ha = station.B * a * float(np.sqrt(box.sigma / box.mu))
    for cells, layer in ((32, 4), (48, 6), (72, 9), (96, 12)):
        base = tilted_flow(beta, ha, 0.0, cells, layer)[0]
        ratios = [base / tilted_flow(beta, ha, tilt, cells, layer)[0] for tilt in (0.1, 0.2)]
        print(f"{design:8s} Ha {ha:6.1f} {cells}/{layer}: W(0.1)/W0 = {ratios[0]:.5f}  W(0.2)/W0 = {ratios[1]:.5f}", flush=True)
