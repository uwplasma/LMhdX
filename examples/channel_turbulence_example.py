"""Turbulent channel in a spanwise magnetic field: Krasnov, Zikanov and Boeck (2011), Table 1.

Edit the inputs below, then run ``python examples/channel_turbulence_example.py``. The run is long (see
``docs/validation/channel_turbulence.md`` for the measured cost); it writes ``summary.json``, the mean
profile and a figure beneath ``artifacts/``, and resumes from ``state.npz`` there if it is stopped.

Plane Poiseuille flow at bulk Reynolds number 6667 with an insulating wall and a field of Hartmann
number 30 along the periodic spanwise axis, at constant bulk velocity, on 64^3 cells with the tanh
wall grid (A = 2) and the IMEX RK3 step. The spectral reference is Re_tau 306.69 and Ucl 1.2542; the
finite-difference scheme B of the paper gives 306.99 / 1.2459 on 64^3, 306.45 / 1.2532 on 128^3.
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import lmhdx
from lmhdx import Solver, channel, solve
from lmhdx.core3d import RK3, courant_step, project, zero_velocity
from lmhdx.problem import channel_kernel

# Inputs: the case, the numerics, and the statistics window (all times in half-width / bulk velocity).
OUTPUT_DIR = Path("artifacts/examples/channel_turbulence")
REYNOLDS, HARTMANN = 6667.0, 30.0
CELLS = (64, 64, 64)
WALL_STRETCH = dict(beta=2.0)  # z = tanh(A xi) / tanh(A), A = 2
DT = 0.025  # near the RK3 stage-map limit by the conservative estimate printed below; half of it changes nothing
CHUNK = 50  # steps between the flux correction and a statistics sample
DRIVE = 2.0e-3  # starting guess of the wall stress; the flux correction below supplies the rest
BURN_IN = 200.0  # time discarded before the statistics start
WINDOW = 200.0  # length of the statistics window
SEED, NOISE = 1, 0.2  # the start: laminar profile plus this much low-wavenumber noise
REFERENCE = dict(re_tau=306.69, ucl=1.2542)  # KZB 2011 Table 1, spectral DNS on 256^3
CHECKPOINT_EVERY = 40  # chunks

# Set up the problem: constant bulk velocity is held by the correction, so the drive is a plain force.
lmhdx.configure(dtype="float64")
problem = channel(re=REYNOLDS, ha=HARTMANN, field="spanwise", cells=CELLS, stretch=WALL_STRETCH)
problem = problem.replace(flow_rate=None, forcing=(DRIVE, 0.0, 0.0), advection="central")
grid, viscosity = problem.grid, problem.viscosity
kernel = channel_kernel(problem, DT)[0]
solver = Solver("rk3", dt=DT, steps=CHUNK)
nx = CELLS[0]
dy, dz, zc = (np.asarray(grid.widths[1]), np.asarray(grid.widths[2]), np.asarray(grid.centers[2]))
section = dy[:, None] * dz[None, :]
profile = 1.0 - zc**2
profile /= section.sum(0) @ profile / section.sum()  # a unit-bulk shape that vanishes at the walls
# Wall shear from a quadratic through the wall and the two nearest cell centres.
z1, z2 = dz[0] / 2, dz[0] + dz[1] / 2
slope = np.array([z2**2, -(z1**2)]) / (z1 * z2 * (z2 - z1))


def bulk(data):
    return np.einsum("xyz,yz->", data[:nx], section) / (nx * section.sum())


def wall_stress(mean):
    return 0.5 * viscosity * (slope @ mean[[0, 1]] + slope @ mean[[-1, -2]])


# Start from the laminar profile plus low-wavenumber noise, or resume from the checkpoint.
checkpoint = OUTPUT_DIR / "state.npz"
if checkpoint.exists():
    saved = np.load(checkpoint)
    velocity = tuple(f.replace_data(jnp.asarray(saved[f"u{i}"])) for i, f in enumerate(zero_velocity(kernel)))
    clock, sums, count = float(saved["clock"]), saved["sums"], int(saved["count"])
    drives, trace = list(saved["drives"]), list(saved["trace"])
    print(f"Resumed at t = {clock:.1f} from {checkpoint}", flush=True)
else:
    rng = np.random.default_rng(SEED)
    velocity = list(zero_velocity(kernel))
    for i in range(2):
        modes = np.fft.fftn(rng.standard_normal(velocity[i].data.shape))
        kx, ky = (
            np.fft.fftfreq(modes.shape[0]) * modes.shape[0],
            np.fft.fftfreq(modes.shape[1]) * modes.shape[1],
        )
        modes *= ((np.abs(kx) <= 4)[:, None] & (np.abs(ky) <= 4)[None, :])[:, :, None]
        noise = np.real(np.fft.ifftn(modes))
        velocity[i] = velocity[i].replace_data(jnp.asarray(NOISE * noise / noise.std() * (1.0 - zc**2)))
    velocity[0] = velocity[0].replace_data(velocity[0].data + 1.5 * jnp.asarray(1.0 - zc**2))
    velocity = project(tuple(velocity), kernel)[0]
    clock, sums, count, drives, trace = 0.0, np.zeros(CELLS[2]), 0, [], []

# Run in chunks: hold the bulk velocity at one, then sample the plane-averaged mean after the burn-in.
steps_per_chunk = CHUNK * DT  # time per chunk
elapsed, timed, first_clock = 0.0, 0, clock
print(f"Running {CELLS[0]}^3 cells, Re = {REYNOLDS:g}, Ha = {HARTMANN:g}, dt = {DT:g}...", flush=True)
while clock < BURN_IN + WINDOW:
    started = time.perf_counter()
    velocity = solve(problem, solver, start=velocity).velocity
    u = np.asarray(velocity[0].data)
    deficit = 1.0 - bulk(u)
    velocity = (velocity[0].replace_data(velocity[0].data + deficit * jnp.asarray(profile)), *velocity[1:])
    clock += steps_per_chunk
    if clock > first_clock + steps_per_chunk:  # the first chunk of the process compiles: not timed
        elapsed, timed = elapsed + time.perf_counter() - started, timed + 1
    mean = np.einsum("xyz,y->z", u[:nx], dy) / (nx * dy.sum())
    drives.append(DRIVE + deficit / steps_per_chunk)
    trace.append(float(np.sqrt(wall_stress(mean))) / viscosity)
    if clock > BURN_IN:
        sums, count = sums + mean, count + 1
    if len(trace) % 10 == 0:
        courant = DT / courant_step(velocity, RK3, 1.0)
        print(
            f"t = {clock:7.1f}  Re_tau = {trace[-1]:7.2f}  drive = {drives[-1]:.4e}  "
            f"RK3 limit used = {courant:.2f}  {1e3 * elapsed / max(timed, 1) / CHUNK:.0f} ms/step",
            flush=True,
        )
    if len(trace) % CHECKPOINT_EVERY == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint, clock=clock, sums=sums, count=count, drives=drives, trace=trace,
                 **{f"u{i}": np.asarray(v.data) for i, v in enumerate(velocity)})  # fmt: skip

# Statistics: friction velocity from the mean wall shear (and from the mean drive), and the centreline.
mean = sums / count
sym = 0.5 * (mean + mean[::-1])
re_tau = float(np.sqrt(wall_stress(mean))) / viscosity
re_tau_balance = float(np.sqrt(np.mean(drives[-count:]))) / viscosity
ucl = float(0.5 * (sym[CELLS[2] // 2] + sym[CELLS[2] // 2 - 1]))
summary = dict(
    re_tau=re_tau, re_tau_from_drive=re_tau_balance, ucl=ucl, reference=REFERENCE,
    re_tau_error=re_tau / REFERENCE["re_tau"] - 1.0, ucl_error=ucl / REFERENCE["ucl"] - 1.0,
    cells=CELLS, dt=DT, burn_in=BURN_IN, window=WINDOW, samples=count,
    wall_seconds_per_step=elapsed / max(timed, 1) / CHUNK, wall_seconds_total=elapsed,
)  # fmt: skip
print(json.dumps(summary, indent=2))

# Save the summary, the profile and one figure.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
np.savez(OUTPUT_DIR / "mean_profile.npz", z=zc, u=mean, trace=trace, drives=drives)
figure, (left, middle, right) = plt.subplots(1, 3, figsize=(10.5, 3.0), constrained_layout=True)
times = steps_per_chunk * np.arange(1, len(trace) + 1)
left.plot(times, trace, lw=0.8)
left.axhline(REFERENCE["re_tau"], color="k", ls="--", lw=0.8)
left.axvspan(BURN_IN, BURN_IN + WINDOW, color="C1", alpha=0.15)
left.set(xlabel="time", ylabel=r"$Re_\tau$ (instantaneous)", title="Stationarity")
middle.plot(zc, mean, label="LMhdX 64$^3$")
middle.axhline(REFERENCE["ucl"], color="k", ls="--", lw=0.8, label="spectral Ucl")
middle.set(xlabel="z", ylabel=r"$\langle u\rangle$", title="Mean velocity")
middle.legend(frameon=False)
yplus = (1.0 + zc[: CELLS[2] // 2]) * re_tau
right.semilogx(yplus, mean[: CELLS[2] // 2] / (re_tau * viscosity), label="LMhdX")
right.semilogx(yplus[yplus > 8], 2.5 * np.log(yplus[yplus > 8]) + 5.5, "k:", lw=0.8, label="log law")
right.set(xlabel="$y^+$", ylabel="$u^+$", title="Wall units")
right.legend(frameon=False)
figure.savefig(OUTPUT_DIR / "channel_turbulence.png", dpi=90)
print(f"Wrote {OUTPUT_DIR}/summary.json and channel_turbulence.png")
