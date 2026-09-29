"""Tilt law, exit (t_reference_rel): an independent spectral solve of a rectangular duct in an in-plane field
B = Ha (cos t, sin t) along (y, z), against the staggered core's flow. Dense Chebyshev collocation on the full
domain, so limited to Ha of about 100, as validation/shercliff.duct_flow is.

    mu lap(u) - |B|^2 u + B_y dphi/dz - B_z dphi/dy + 1 = 0,    lap(phi) = B_y du/dz - B_z du/dy,
    u = 0 and dphi/dn = 0 on all four walls (insulating).
"""

import numpy as np

from validation.shercliff import _differentiation_matrix, chebyshev_weights


def tilted_spectral_flow(ha: float, beta: float, tilt: float, points: int = 64) -> float:
    """Flow per unit drive on [-1, 1] x [-beta, beta]: the integral of u."""
    derivative, nodes = _differentiation_matrix(points)
    n, eye = points + 1, np.eye(points + 1)
    dy, dz = np.kron(derivative, eye), np.kron(eye, derivative) / beta
    lap = np.kron(derivative @ derivative, eye) + np.kron(eye, derivative @ derivative) / beta**2
    theta = np.arctan(tilt)
    by, bz = ha * np.cos(theta), ha * np.sin(theta)
    size = n * n
    operator = np.zeros((2 * size, 2 * size))
    source = np.zeros(2 * size)
    operator[:size, :size] = lap - (by**2 + bz**2) * np.eye(size)
    operator[:size, size:] = by * dz - bz * dy
    source[:size] = -1.0
    operator[size:, :size] = -(by * dz - bz * dy)
    operator[size:, size:] = lap
    y, z = np.repeat(nodes, n), np.tile(nodes, n)
    for row in np.flatnonzero((np.abs(y) > 1 - 1e-12) | (np.abs(z) > 1 - 1e-12)):
        operator[row, :], source[row] = 0.0, 0.0
        operator[row, row] = 1.0
        operator[size + row, :] = 0.0
        normal = dy[row] * np.sign(y[row]) if abs(y[row]) > 1 - 1e-12 else dz[row] * np.sign(z[row])
        operator[size + row, size:] = normal
    operator[size + size // 2, :] = 0.0
    operator[size + size // 2, size + size // 2] = 1.0  # gauge: phi = 0 at the centre
    solution = np.linalg.solve(operator, source)
    weights = chebyshev_weights(points)
    return float(weights @ solution[:size].reshape(n, n) @ weights) * beta


if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)
    from ductopt.physics import tilted_flow

    for ha, beta, tilt in ((30.0, 0.5, 0.0), (30.0, 0.5, 0.3), (60.0, 0.3, 0.1), (60.0, 0.3, 0.3), (100.0, 0.15, 0.2)):
        spectral = {p: tilted_spectral_flow(ha, beta, tilt, p) for p in (48, 64)}
        core = {c: tilted_flow(beta, ha, tilt, c, layer) for c, layer in ((48, 6), (72, 9), (96, 12))}
        print(f"Ha {ha:5.0f} beta {beta:.2f} tilt {tilt:.2f}: spectral 48/64 pts {spectral[48]:.8e} {spectral[64]:.8e} "
              f"(diff {spectral[48] / spectral[64] - 1:+.1e}); core/spectral-1: "
              + " ".join(f"{c}: {core[c] / spectral[64] - 1:+.2e}" for c in core), flush=True)
