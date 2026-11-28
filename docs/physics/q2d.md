# Quasi-two-dimensional flow

`lmhdx.q2d` evolves a periodic, depth-averaged flow in a strong uniform field normal to the flow plane. The
[Q2D tutorial](../tutorials/q2d.md) runs it, and the [Q2D anchors](../validation/q2d.md) validate it.

```{image} ../_static/q2d_turbulence_poster.webp
:alt: Decaying Q2D turbulence: three vorticity snapshots and the energy spectrum
:align: center
```

*Decaying Q2D turbulence on 256² cells, with three snapshots of the vorticity and the energy spectrum against a $k^{-3}$ guide.*

(q2d-model)=
## The model

The basic Sommeria-Moreau closure evolves the depth-averaged vorticity,

$$
\frac{\partial\omega}{\partial t}
+\mathbf u_\perp\cdot\nabla_\perp\omega
=\nu\nabla_\perp^2\omega-\frac{\omega}{\tau_H}+f_\omega,
$$

$$
\mathbf u_\perp=(\partial_y\psi,-\partial_x\psi),\qquad
-\nabla_\perp^2\psi=\omega.
$$

Here $1/\tau_H$ is the linear Hartmann-layer friction and $f_\omega$ is a vorticity source. On a periodic domain the
velocity is divergence free by construction. With kinetic energy $E=\langle|\mathbf u_\perp|^2\rangle/2$ and
enstrophy $Z=\langle\omega^2\rangle/2$,

$$
\frac{dE}{dt}=-2\nu Z-\frac{2E}{\tau_H}+\langle\psi f_\omega\rangle.
$$

The model is a depth-averaged strong-field approximation. It does not resolve Hartmann-layer profiles or replace the
3-D formulation where the field varies along the flow.

(q2d-ifrk4)=
## The spectral IFRK4 step

LMhdX works with real Fourier transforms of the periodic field (`rfftn`). In Fourier space the vorticity equation is $\partial_t\hat\omega=-L\hat\omega+\hat N(\hat\omega)$, with
$L=\nu\lambda_k+1/\tau_H$ and $\hat N=P\big[\hat f_\omega-\mathcal F(\mathbf u_\perp\cdot\nabla_\perp\omega)\big]$.
Here $\lambda_k$ is the periodic Poisson symbol, which SOLVAX supplies with the zero-mean spectral inversion for the
streamfunction, and $P$ is the two-thirds dealiasing mask. With $\mathcal E=e^{-\Delta t L}$, the fourth-order
integrating-factor Runge-Kutta step is

$$
\begin{aligned}
N_1&=\hat N(\hat\omega^n), &
N_2&=\hat N\big(\mathcal E^{1/2}(\hat\omega^n+\tfrac{\Delta t}{2}N_1)\big),\\
N_3&=\hat N\big(\mathcal E^{1/2}\hat\omega^n+\tfrac{\Delta t}{2}N_2\big), &
N_4&=\hat N\big(\mathcal E\hat\omega^n+\Delta t\,\mathcal E^{1/2}N_3\big),\\
\hat\omega^{n+1}&=\mathcal E\hat\omega^n+\tfrac{\Delta t}{6}\big(\mathcal E N_1+2\mathcal E^{1/2}(N_2+N_3)+N_4\big). &&
\end{aligned}
$$

The scheme integrates the viscous and Hartmann-friction terms exactly within each step. LMhdX owns the vorticity
dynamics, the velocity reconstruction, the energy identity and physical acceptance.

LMhdX compiles the largest stable integration segment. A positive `history_stride` divides a run into compiled
segments and moves only the requested vorticity frames to the host, and zero keeps no field history. Stored results
therefore do not grow with the number of time steps. Derivatives use a two-level checkpoint schedule
([Derivatives](numerics/derivatives.md)).

(q2d-energy-budget)=
## The energy budget

`solve_q2d` enforces `energy_budget_tolerance` (default $10^{-3}$) on the absolute integrated defect of the energy
identity, divided by the larger of the initial energy and the absolute integrated power. The smallest positive normal
value of the working dtype is a zero-energy floor. An absolute machine-epsilon floor would hide relative defects in
small-amplitude flows.

The solver integrates the power by the trapezoidal rule at every time step, whatever the saved-frame stride. This
diagnostic has second-order quadrature error even when the IFRK4 field evolution is more accurate. Refine the time step
when it fails. Even `completed` does not certify mesh independence or steady flow.
