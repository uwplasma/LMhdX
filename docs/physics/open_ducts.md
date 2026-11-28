(open-ducts)=
# Ducts with an inlet and an outlet

```{image} ../_static/open_ducts.webp
:alt: The ANL fringe field, the pressure along a duct through it, and the excess pressure drop against Ha
:align: center
```

*(a) The ANL fringe field. (b) The pressure along a duct through it at Ha 20, as the fringe tutorial sets it up. (c) The excess pressure drop of the fringe on the 3-D core against the core-flow model, with TM-228's thin-layer asymptotes dashed. Drawn by `python scripts/make_showcase_figures.py --only fringe`.*

`lmhdx.axial` gives the staggered core an inflow-outflow axis, the first, for a duct that runs through a varying field
such as a magnet's fringe. It solves the inertialess limit or, by marching or Newton's method, the steady flow with
inertia. `validation/open_pipe.py` does the same for the pipe. The evidence is on
[Open ducts: the ANL fringe and the ALEX pipe](../validation/open_ducts.md), and the
[fringing duct tutorial](../tutorials/fringing.md) runs one.

(open-ducts-bcs)=
## Boundary conditions

The conditions are those of HIMAG, FreeMHD, GridapMHD and the 2025 six-code benchmark (plan D26).

- At the inlet the velocity is LMhdX's own fully developed profile at the inlet field, solved on the same
  cross-section and scaled to the imposed flow rate. It is array-valued Dirichlet data on the inlet face
  (`lmhdx.grid.BoundaryCondition(kind, lower=profile, upper_kind=...)`).
- At the outlet every velocity component has zero axial gradient, and $p=0$.
- At both ends no normal current flows, so $\partial\varphi/\partial n=(\mathbf u\times\mathbf B)\cdot\mathbf n$.
  Removing the potential's mean fixes its gauge.
- The flow rate is exact and the pressure drop is an output. There is no extra unknown.

(open-ducts-lift)=
## The lift and the outlet face

**The inlet is a lift.** The fully developed profile carried unchanged along the duct is discretely divergence free, so
the solution is that lift plus a correction with no inlet flux. Upstream of any field change the lift is the discrete
solution.

**The operator stays symmetric.** On the correction's space the Stokes-limit operator is symmetric in the face-volume
inner product, once the two end faces own the half cell inside the duct. The outlet velocity is then an unknown with a
zero-gradient viscous flux, and the pressure gradient through it is the Dirichlet one. The solve is the preconditioned
CG of a periodic duct ([Solvers](numerics/solvers.md)), and it differentiates the same way.

**The outlet face** owns half a cell and, with inertia, carries the flux $u^2$ through its outer side. Without that
flux the Newton root was unstable in time, with a mode localized at the outlet that grew as $2^{\text{cells}}$
upstream. At Ha 20, Re 40 on $62\times12\times12$, Newton took 1,352 outer iterations (105 s warm) without the flux and
206 (16 s) with it, to the same drop within $4\times10^{-13}$. Adding it moved the fringe drop by $3\times10^{-5}$ on
the coarsest test mesh (spacing 1, Re 10), $2.5\times10^{-8}$ at Ha 20, Re 40 and under $10^{-10}$ in the other cases.

**A wide spectrum.** The pressure operator's lowest mode, the long axial wave, lies nine decades below
its largest eigenvalue across a Hartmann layer (0.04 against 7e7 at Ha 100). So the singularity test is per axis, and
the solver sets the computed zero eigenvalue of every Neumann or periodic axis to zero. The same ratio costs the
fast-diagonal transforms that many digits, so the pressure and potential solves take one float64 defect correction.
Without these the true residual stalled at 2e-7 and the charge balance at 1e-9.

On the ANL fringe from Ha 100 to 1600, flow rate, mass and charge hold to round-off
({ref}`Balances <validation-open-ducts-balances>`).

(open-ducts-march-newton)=
## March or Newton

With inertia the steady open duct solves

$$
\mathbf u\cdot\nabla\mathbf u=-\frac{1}{\rho}\nabla p+\nu\nabla^2\mathbf u+\frac{1}{\rho}\mathbf J\times\mathbf B,
\qquad \nabla\cdot\mathbf u=0,
$$

with the conditions above. `solve_open_duct` takes an advective duct (`advection="central"` or `"limited"`) from the
Stokes-limit solution to this state. The residual is the physical steady residual, the projected momentum balance.

**The march** is the RK3 scheme of [Time stepping](numerics/time.md), with the exact constrained stage, Crank-Nicolson
damping and viscosity, central transport, and the lift carrying the inlet.

- The step is the smaller of 0.9 of the Courant limit $\sqrt3/\sum_a\max|u_a|/\Delta x_a$ and $4/\lambda_{\max}$, with
  $\lambda_{\max}=\sigma B_{\max}^2/\rho$. In a varying field every cell takes the largest damping rate in the implicit
  part and gives the rest of its own rate to the explicit one. That split is stable to $\lambda_{\max}\Delta t$ of 6
  and not at 7 (Ha 100, Re 20).
- Fifty steps make a chunk, and `solvax.fixed_point_iteration` repeats chunks until the residual falls to `tolerance`
  of the lift's. A march that does not get there is reported as `stagnated` or `unsteady`.

**Newton** takes the same residual and start. The unknown is the correction to the lift. Each update is a flexible
GMRES solve (`solvax.gcrot`, 120 per cycle, tolerance 1e-3) preconditioned by the Stokes-limit operator's inverse, a CG
solve to 0.1. There are at most twelve updates. Both methods differentiate through `solvax.root_solve` and agree on the
drop and its gradient to $10^{-6}$ ([Derivatives](numerics/derivatives.md)).

**Which to use.** `method="auto"` marches while the Courant step $\Delta t_C$ is at most four damping steps
$\Delta t_D=4/\lambda_{\max}$, and takes Newton otherwise. The table gives warm medians on the loaded laptop, on
$62\times12\times12$ (Ha 20) and $62\times16\times16$ (Ha 100).

| case | Newton / march (s) | $\Delta t_C/\Delta t_D$ | march steps | Newton outer iterations |
|---|---|---|---|---|
| Ha 20, Re 40 | 15.9 / 5.35 | 1.6 | 250 | 206 |
| Ha 20, Re 20 | 5.2 / 3.4 | 3.1 | 150 | 68 |
| Ha 20, Re 10 | 1.7 / 3.9 | 6.3 | 200 | 28 |
| Ha 100, Re 20 | 33.8 / 41.6 | 76 | 1,150 | 69 |

The threshold of 4 lies between the 3.1 and the 6.3 of these ducts. It is fitted, not derived, and it moves with the
mesh through the Courant step.

(open-ducts-inertia)=
## Inertia and the three-dimensional core

The inertial share is $\Delta p(Re)-\Delta p(0)$ over the excess, with $Re=Ua/\nu$ and $N=Ha^2/Re$. On the ANL fringe it
is 0.42 % at Ha 100, Re 10 and 2.33 % at Ha 200, Re 40, and a fit gives share
$\approx 0.045\,\% \cdot Re^{1.31} Ha^{-0.16}$ ({ref}`Inertial share <validation-open-ducts-inertia>`).

In the Stokes limit the three-dimensional core does not meet the 1 % gate of rows 7 and 24 against TM-228's 0.0178.
The gap is the conductance the Hartmann and side layers add, which the [core-flow model](coreflow.md) leaves out. With
that conductance the two agree within 1 % at $c=0.1$, Ha $2\times10^4$
({ref}`The layers' conductance <validation-open-ducts-layers>`).

(open-ducts-pipe)=
## The pipe on an open polar axis

`validation/open_pipe.py` (`fringe_pipe` and `solve_open_pipe`) gives the polar grid an inlet and an outlet on its third
axis.

- The velocity is staggered, $u_r$ on radial faces, $u_\theta$ on azimuthal faces and $u_x$ on axial faces.
- The viscous operator is the Hessian of the discrete dissipation $\tfrac12\sum|\nabla\mathbf u|^2\,dV$ in cylindrical
  components, whose azimuthal entries carry the curvature terms $(\partial_\theta u_r-u_\theta)/r$ and
  $(\partial_\theta u_\theta+u_r)/r$.
- The electromotive force forms at the cell centres in polar components of the transverse field $B=Ha\,b(x)$ and
  averages to the faces. The Lorentz force is minus its adjoint, with Walker's thin wall through the polar
  fast-diagonal factorization (a sheet node per azimuthal mode, which also conducts axially).
- So the Stokes operator is symmetric and negative definite in the face-volume inner product by construction. The
  solve is one preconditioned CG, with the fully developed pipe of `lmhdx.pipe.solve_pipe` as the lift and component
  Helmholtz solves as the preconditioner. The polar factorization treats Neumann at both axial ends as singular, as it
  does a periodic axis.

On ALEX B1 (row 6) the section mean meets the upstream gradient and the integrated excess but not the station-wise
tolerance. Read at the walls parallel to the field, it meets that tolerance too, which matches the data if the ALEX
taps are side-wall taps. The packaged data does not record where they are
({ref}`The ALEX pipe <validation-open-ducts-pipe>`).
