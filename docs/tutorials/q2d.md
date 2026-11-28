# Evolve a Q2D vortex field

Under a strong transverse magnetic field, a conducting flow becomes nearly
two-dimensional, and its depth average obeys the Sommeria--Moreau (SM82) model.
LMhdX evolves the periodic vorticity of that model, with viscous diffusion and a
linear friction from the Hartmann layers. This page decays a field of vortices,
as `python examples/q2d_turbulence_demo.py` does, then differentiates the
evolution.

## Set up and run

{ref}`The Q2D model <q2d-model>` gives the equations, and {ref}`the spectral IFRK4 step <q2d-ifrk4>` the time step.

Import JAX's NumPy and LMhdX.

```python
import jax.numpy as jnp
import lmhdx
```

Build a smooth vortex field of several scales on a periodic 64 × 64 grid of
side $2\pi$.

```python
shape, length = (64, 64), (2.0 * jnp.pi, 2.0 * jnp.pi)
x = jnp.arange(shape[0]) * length[0] / shape[0]
y = jnp.arange(shape[1]) * length[1] / shape[1]
xx, yy = x[:, None], y[None, :]
initial_vorticity = (
    jnp.sin(xx) * jnp.sin(yy)
    + 0.55 * jnp.sin(2.0 * xx - 0.4) * jnp.sin(3.0 * yy)
    + 0.35 * jnp.cos(3.0 * xx + 2.0 * yy)
)
```

Set the viscosity, the Hartmann friction and the time step, and keep every
fourth frame.

```python
problem = lmhdx.Q2DProblem(
    initial_vorticity,
    length=length,
    viscosity=2.0e-3,
    hartmann_friction=4.0e-2,
    dt=1.5e-2,
    steps=160,
    history_stride=4,
)
```

Run it and print the final kinetic energy.

```python
result = lmhdx.solve(problem)
assert result.converged
print(result.status, f"{float(result.diagnostics.kinetic_energy_final):.5f}")
```

```text
completed 0.05444
```

The kinetic energy falls from 0.06776 to 0.05444 over $t = 2.4$.

The example also renders the poster below, a JSON diagnostic record, and an
MP4 when FFmpeg is installed.

```{image} ../_static/q2d_vortex_decay.webp
:alt: Initial and final Q2D vorticity beside the kinetic-energy decay
:align: center
```

The inputs:
- `Q2DProblem` accepts any two-dimensional initial-vorticity array and a
  vorticity source of the same shape. `lmhdx.make_q2d_case(...)` builds a
  Taylor--Green decay case of one mode.
- Lengths, viscosity, friction, time and forcing must use one consistent unit
  system.
- The solver projects out the zero Fourier mode, so the periodic problem has
  zero net circulation.
- `history_stride` turns on sparse field frames. By default the solver keeps
  only the final state.

## Differentiate the finite evolution

`evolve_q2d` is the field-only core for optimization. It skips host status and
output assembly, so continuous array, forcing, length, viscosity,
Hartmann-friction and timestep parameters stay traced. Differentiate an
energy objective with respect to the friction.

```python
import jax


def objective(friction):
    vorticity, velocity_x, velocity_y = lmhdx.evolve_q2d(
        problem.initial_vorticity,
        viscosity=problem.viscosity,
        hartmann_friction=friction,
        dt=problem.dt,
        steps=problem.steps,
    )
    return jnp.mean(vorticity**2) + 0.1 * jnp.mean(velocity_x**2 + velocity_y**2)


value, gradient = jax.jit(jax.value_and_grad(objective))(problem.hartmann_friction)
print(f"{float(value):.5f} {float(gradient):.5f}")
```

```text
0.31122 -1.49068
```

- This is the exact reverse derivative of the dealiased IFRK4 discretization.
- SOLVAX divides a long recurrence into segments and rematerializes one
  segment during the backward pass. With $N$ steps and checkpoint width $C$,
  retained trajectory state is $O(N/C+C)$. The default $C=\lceil\sqrt N\rceil$
  is `O(sqrt(N))`. Set `adjoint_checkpoint_size` only after measuring a
  different trade-off between memory and recomputation.
- `steps`, grid shape and checkpoint width are discrete static controls. Use
  `Q2DProblem` with `solve` when you need validated host diagnostics or saved
  frames. Use `evolve_q2d` inside `grad`, `jvp`, `vjp` or `vmap`.

## CPU and GPU execution

The same public API selects JAX's active backend.

**CPU against GPU.** A controlled office-host run used JAX 0.6.2, float32
fields, a $256\times256$ grid, 80 steps, one compilation run and five warm
repetitions. The final CPU and NVIDIA RTX A4000 vorticity fields agreed to
relative $L_2=2.38\times10^{-6}$ and $L_\infty=3.67\times10^{-6}$.

**Timings.**
- A float32 GPU timing counts only when its report records a true-float32
  `jax_default_matmul_precision`, because JAX's default lets Ampere cards use
  TensorFloat-32 (ADR 0005, D15).
- The GPU reports in `benchmarks/results` record `highest` (plan step 2.1). On
  one A4000, $256^2$ takes 0.863 ms per step in true float32.
- The CPU reports there are from the uncontrolled 2026-09-07 run and record no
  precision, so this page quotes no Q2D CPU/GPU ratio.
- LMhdX sets the precision to `'highest'` when you build a `Q2DProblem`, unless
  you have already chosen one.

These figures hold for this workload and hardware only. Timings exclude
compilation, use identical precision and inputs, and synchronize the final
field before stopping the clock. Q2D accepts sharded initial fields, but a
correct sharded run does not imply a parallel speedup.

**Check placement locally.** Run from the repository root in a fresh process:

```sh
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  python scripts/run_benchmarks.py --cases shard --q2d-sizes 64,256 \
  --steps 20 --repeats 3 --x64 1 --output artifacts/q2d-cpu4.json
```

Repeat with two devices and `--x64 0` for float32.
- Each run compares a single-device trajectory against the same input
  distributed over all devices, checks solver completion and field agreement,
  and records synchronized times.
- The grid's first dimension must be a multiple of the device count.
- [JAX CPU devices](https://docs.jax.dev/en/latest/config_options.html) share
  host cores and memory, so the reported efficiency compares logical-device
  placement. It is not strong scaling or a prediction for a GPU interconnect.
- Record thread settings and host load, since one CPU device can already use
  multiple threads.

## Interpretation

The solver reports kinetic energy, enstrophy, the integrated energy-budget
residual, the final field's spectral velocity divergence and the maximum
Courant number. Energy-budget and Courant reductions run every step, and the
divergence once, on the final velocity.

When the solver accepts a result:
- `status == "completed"` and `converged == True` only when fields and
  diagnostics are finite, the maximum Courant number is at most one, and the
  normalized energy-budget residual is at most `energy_budget_tolerance`
  (default `1e-3`). This is finite-trajectory acceptance, not steady
  convergence.
- Set the tolerance on `Q2DProblem` or `make_q2d_case`.
- An energy-budget failure returns the computed fields with status
  `energy_budget_exceeded`. A Courant failure takes precedence. Non-finite
  fields or diagnostics raise an error.
- The budget uses a time-step-by-time-step trapezoidal integral, whose error is
  second order even for an accurately evolved mode. Refine the time step and
  check the {ref}`energy normalization <q2d-energy-budget>` before relaxing the
  tolerance.
- `evolve_q2d` is a traced field kernel and performs no host acceptance check.
  Qualify optimization trajectories with `solve_q2d`.

SM82 is a basic quasi-two-dimensional closure. Use the 3-D inductionless
model for developing Hartmann layers, spatially varying transverse structure,
fringing fields, and geometries that break the depth-average assumptions. This Q2D path leaves out
the higher-order inertial corrections that Pothérat, Sommeria, and Moreau
describe.
