"""Evolve a damped Q2D vortex field and render a poster and short movie.

Run ``python examples/q2d_turbulence_demo.py``. Outputs are written beneath
``artifacts/``; MP4 creation is skipped cleanly when FFmpeg is unavailable.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from lmhdx import Q2DProblem, solve

# Inputs: periodic mesh, physical coefficients, integration, and outputs.
OUTPUT_DIR = Path("artifacts/examples/q2d_turbulence")
SHAPE = (64, 64)
LENGTH = (2.0 * jnp.pi, 2.0 * jnp.pi)
VISCOSITY = 2.0e-3
HARTMANN_FRICTION = 4.0e-2
DT = 1.5e-2
STEPS = 160
HISTORY_STRIDE = 4

# Build a smooth multi-scale vortex field with zero circulation.
x = jnp.arange(SHAPE[0]) * LENGTH[0] / SHAPE[0]
y = jnp.arange(SHAPE[1]) * LENGTH[1] / SHAPE[1]
xx, yy = x[:, None], y[None, :]
initial_vorticity = (
    jnp.sin(xx) * jnp.sin(yy)
    + 0.55 * jnp.sin(2.0 * xx - 0.4) * jnp.sin(3.0 * yy)
    + 0.35 * jnp.cos(3.0 * xx + 2.0 * yy)
)
problem = Q2DProblem(
    initial_vorticity,
    length=LENGTH,
    viscosity=VISCOSITY,
    hartmann_friction=HARTMANN_FRICTION,
    dt=DT,
    steps=STEPS,
    history_stride=HISTORY_STRIDE,
)

# Run the same public solve entry point used by other LMhdX models.
print(f"Evolving Q2D vorticity on {SHAPE[0]} x {SHAPE[1]} for {STEPS} steps...", flush=True)
result = solve(problem)
print(f"  {result.status}; rendering the poster and movie...", flush=True)
frames = np.asarray(result.vorticity_history)
limit = float(np.max(np.abs(frames)))
energies = []
for frame in frames:
    omega_hat = np.fft.fftn(frame)
    kx = (2.0 * np.pi * np.fft.fftfreq(SHAPE[0], d=float(LENGTH[0]) / SHAPE[0]))[:, None]
    ky = (2.0 * np.pi * np.fft.fftfreq(SHAPE[1], d=float(LENGTH[1]) / SHAPE[1]))[None, :]
    k2 = kx**2 + ky**2
    psi_hat = np.divide(omega_hat, k2, out=np.zeros_like(omega_hat), where=k2 > 0.0)
    energies.append(
        0.5
        * np.mean(
            np.abs(np.fft.ifftn(1j * ky * psi_hat)) ** 2 + np.abs(np.fft.ifftn(-1j * kx * psi_hat)) ** 2
        )
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
extent = (0.0, float(LENGTH[0]), 0.0, float(LENGTH[1]))


def show_vorticity(axis, frame):
    """Render one smoothly interpolated periodic vorticity field."""

    return axis.imshow(
        frame.T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="bicubic",
    )


figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
for axis, frame, title in zip(axes[:2], (frames[0], frames[-1]), ("Initial", "Final"), strict=True):
    image = show_vorticity(axis, frame)
    axis.set(xlabel="$x$", ylabel="$y$", title=f"{title} vorticity")
figure.colorbar(image, ax=axes[:2], shrink=0.82, label="$\\omega$")
axes[2].semilogy(np.asarray(result.frame_times), energies, color="#432371", linewidth=2.2)
axes[2].set(xlabel="Time", ylabel="Kinetic energy", title="Hartmann-damped decay")
poster_path = OUTPUT_DIR / "q2d_vortex_decay.webp"
figure.savefig(poster_path, dpi=145)
plt.close(figure)

movie_path = None
if shutil.which("ffmpeg"):
    figure, axis = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    image = show_vorticity(axis, frames[0])
    title = axis.set_title("")
    axis.set(xlabel="$x$", ylabel="$y$")
    figure.colorbar(image, ax=axis, label="$\\omega$")

    def update(index):
        """Update one rendered vorticity frame."""

        image.set_data(frames[index].T)
        title.set_text(f"Q2D vorticity, t={float(result.frame_times[index]):.2f}")
        return image, title

    movie_path = OUTPUT_DIR / "q2d_vortex_decay.mp4"
    animation.FuncAnimation(figure, update, frames=len(frames), interval=80, blit=False).save(
        movie_path,
        writer=animation.FFMpegWriter(
            fps=12,
            bitrate=900,
            extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
        ),
        dpi=110,
    )
    plt.close(figure)

summary = {
    "model": "periodic SM82 Q2D vorticity",
    "status": result.status,
    "shape": list(SHAPE),
    "frames": len(frames),
    "poster": poster_path.name,
    "movie": None if movie_path is None else movie_path.name,
    "diagnostics": result.diagnostics.__dict__,
}
summary_path = OUTPUT_DIR / "q2d_vortex_decay.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(
    f"Kinetic energy (nondimensional) {energies[0]:.4g} -> {energies[-1]:.4g} over t = {float(result.frame_times[-1]):.3g}"
)
print(f"Wrote {summary_path}, {poster_path}" + ("" if movie_path is None else f", {movie_path}"))
