(numerics-time)=
# Time stepping

`lmhdx.core3d` advances the three-dimensional flow by a projection step, by implicit Euler or by a three-stage IMEX
Runge-Kutta scheme. The same step marches an open duct to its steady state and carries the heat transport's velocity.
[Code verification](../../validation/verification.md) checks its time accuracy with the field on.

(numerics-projection)=
## The projection step and its two stiffnesses

One fractional step solves for the potential, takes the Lorentz force from the face currents, advances momentum, and
returns the velocity to the discretely divergence-free space by a pressure solve.

The velocity part of the Lorentz force is a damping of rate $\lambda=\sigma B^2/\rho$, which would force
$\Delta t\propto Ha^{-2}$ if left explicit. The step applies it implicitly, as a shift of the conservative face force,

$$
H\mathbf u^{*}+\frac{\Delta t}{\rho}G p=\mathbf b,\qquad D\mathbf u^{*}=0,\qquad
H=(1+\Delta t\,\lambda)I\ \bigl(-\Delta t\,\nu\nabla^2\text{ if implicit}\bigr),
$$

with $\mathbf b$ the old velocity plus the forces. The same $H$ acts on the pressure gradient as on every other force.
So a pressure force in balance with the forcing leaves the fluid at rest to round-off, and the stage's fixed point is
the physical steady state. The pressure solves $D H^{-1} G$ ({ref}`Krylov methods <numerics-krylov>`). A time loop
starts each solve from the last step's pressure, so a steady flow takes 0 to 1 iterations in place of 13.

- **A varying field** takes the largest rate over the cells, because the fast solve needs one shift per component. A
  cell of rate $r$ then updates by $1-\Delta t\,r/(1+\Delta t\,\lambda)$, which lies inside $(0,1]$ wherever
  $r\le\lambda$. A smaller shift, such as the volume mean, would overshoot past $-1$ once
  $\Delta t\,r>2(1+\Delta t\,\lambda)$.
- **Boundary constraints.** Before it takes any divergence, the step sets the wall-normal velocity to zero on its wall
  faces and makes the duplicated first and last face of a periodic axis equal. Without either, the divergence carries a
  net boundary flux that the pressure cannot remove when every axis is periodic or Neumann.
- **Implicit diffusion.** `ChannelProblem.viscous_factorizations` factorizes
  $(1+\Delta t\,\lambda)I-\Delta t\,\nu\nabla^2$ on the free faces of each axis, so one solve removes both stiff
  terms. Integrating the plane channel to the same accuracy takes 29.0 s with explicit diffusion and 1.9 s with
  implicit diffusion at 16 cells, and 95.1 s against 2.6 s at 32, for errors that agree to three digits.
- **Explicit diffusion** is the default. `ChannelProblem.diffusive_step_limit` reports its limit and does not enforce
  it, so a caller sweeping a parameter sees the constraint instead of a step clipped without warning.

Steady Stokes flow between plates has the exact profile $f(1-y^2)/(2\nu)$. The step reproduces it at measured orders
2.01 and 2.04 over 8, 16 and 32 cells across the channel. The run omits convective transport.

(numerics-rk3)=
## The IMEX Runge-Kutta step

`advance(problem, steps, viscous=problem.viscous_factorizations(RK3), scheme=RK3)` takes the low-storage scheme of
Spalart, Moser and Rogers (1991). Each of its three stages is the constrained step above, with explicit weights
$(8/15, 5/12, 3/4)$, carried weights $(0, -17/60, -5/12)$ and Crank-Nicolson weights $(4/15, 1/15, 1/6)$ for the
damping and the viscous term. Transport and the rest of the Lorentz force are explicit, so each stage solves
$H_k\mathbf u+c_k\,\Delta t\,G p/\rho=\mathbf b_k$, $D\mathbf u=0$ exactly. A gradient force at balance leaves the
fluid at rest at every stage (peak speed under $10^{-13}$). `Euler` is the one-stage tableau.

**Stability.** For a shear mode carried by a uniform stream at $kh=\pi/2$ the RK3 amplification per step is
$|g|^2=1-C^4/12+C^6/36$, below one up to $C=\sqrt3$. Forward Euler with the same central flux has $|g|^2=1+C^2$ and
grows at every $C$. Limited transport stays bounded under both.

**Accuracy.** Against the same scheme at 64 steps:

- the Shatrov and Gerbeth manufactured flow (3-D, no slip, tanh-stretched $12^3$, $Re=100$) converges at order 2.30
  and 2.24 for RK3 (Euler 1.01, 1.18);
- an exactly damped shear mode converges at order 2.0 (Euler 1.1);
- the kinetic-energy change of an unsteady inviscid box converges at order 2.76, 2.95 and 2.99 (Euler gains energy);
- the energy budget with drive, Lorentz damping and viscosity closes at order 2.0.

The implicit terms are second order, the Crank-Nicolson limit. The transport alone would be third order. The
Orr-Sommerfeld mode of plane Poiseuille flow at $Re=8000$ grows at $2.6626\times10^{-3}$ on $32\times48$ cells against
the tabulated $2.66441\times10^{-3}$.

(numerics-cfl)=
## Step control

`courant_step` takes a fraction of the RK3 limit $C=\sqrt3$. The explicit Lorentz force below is stable while
$\Delta t\,\lambda<2$, and the code checks that bound. The open-duct march takes the smaller of a Courant step and a
damping step ({ref}`March or Newton <open-ducts-march-newton>`). The heat transport has its own Courant limits on
[Heat transport](../thermal.md).

(numerics-lorentz-shift)=
## What the shift costs a transient

```{image} ../../_static/numerics_shift.webp
:alt: Growth rate of a roll along the field under the implicit shift and the explicit force
:align: center
```

*The growth rate of a convection roll along the field, over its exact rate, as the time step grows. The implicit damping shift slows it, by up to the factor 1 + Δt σB²/ρ, and the explicit Lorentz force keeps the exact rate. Drawn by `python scripts/make_showcase_figures.py --only shift` from `docs/_static/data/thermal_buoyancy.json`.*

The shift is implicit, but the step takes the potential's share of the force, $-\sigma\nabla\phi\times\mathbf B$, from
the old velocity. Where the two balance, the Lorentz force nearly vanishes. That happens in a quasi-two-dimensional
core, or in a convection roll whose axis lies along the field, since the roll's electromotive force is a gradient and
drives no current. There the step reads
$(1+\Delta t\,\lambda)(\mathbf u^{n+1}-\mathbf u^n)=\Delta t\,\mathbf F(\mathbf u^n)$, so the motion evolves
$1+\Delta t\,\lambda$ times too slowly. The shift does not affect a steady state, but it does affect a transient.

- Against the Chebyshev growth rate of `validation.thermal.convection_growth_rate`, a roll along the field at
  $Pr=0.02$, $Ra=2\times10^4$ grows 5, 9 and 17 % too slowly at $\Delta t\,\lambda=0.066$, 0.13 and 0.27, and 34, 50
  and 66 % too slowly at 0.66, 1.3 and 2.7.
- `ChannelProblem(magnetic_damping="explicit")` (`Solver(magnetic_damping=...)` for a run in time) drops the shift and
  takes the whole Lorentz force from the old velocity. That is exact for the balanced motion and first order for the
  damped motion. The same roll then grows at the field-free rate to $10^{-6}$ (`tests/test_core3d.py`).
- In an open flow the shift slows a roll's drift as well as its growth. A roll carried by a plane Poiseuille flow at
  $\Delta t\,\lambda=0.34$, on 18 cells a wavelength, grows 26 % slower and drifts 14 % slower under the shift
  (`docs/_static/data/thermal_buoyancy.json`), so its growth per unit length falls about 13 %.

Steady solves and the open-duct march keep the shift.

(numerics-trajectory)=
## Running the step as one compiled trajectory

`lmhdx.core3d` compiles a whole run with `jax.lax.scan`, so the device runs ahead instead of waiting for the host
every step. On a laptop CPU, 200 steps of a 4x16x16 duct take 3.58 s through a host loop and 0.44 s through the scan.
`jax.checkpoint` wraps the step body, so reverse mode keeps one state per step and recomputes each step's interior.
Diagnostics come out as scan outputs, so a run reports its divergence residual and kinetic-energy history without
synchronising mid-trajectory. A test greps the step and loop sources for `float(`, `bool(`, `device_get` and `.item()`,
because one of those inside the loop would serialise the queue.
