# Equations and assumptions

LMhdX solves incompressible, inductionless liquid-metal MHD. The magnetic
Reynolds number is assumed small, so the imposed magnetic field $\mathbf B$ is
not evolved. The full isothermal formulation below uses density $\rho$,
kinematic viscosity $\nu$, conductivity $\sigma$, pressure $p$, and body drive
$\mathbf f$. The implemented model reductions are described below it:

$$
\nabla\cdot\mathbf u = 0,
$$

$$
\rho\left(\frac{\partial\mathbf u}{\partial t}
+\mathbf u\cdot\nabla\mathbf u\right)
=-\nabla p+\rho\nu\nabla^2\mathbf u
+\mathbf J\times\mathbf B+\mathbf f,
$$

$$
\mathbf J=\sigma\left(-\nabla\phi+\mathbf u\times\mathbf B\right),
\qquad \nabla\cdot\mathbf J=0.
$$

The electric potential therefore satisfies the variable-conductivity equation

$$
\nabla\cdot(\sigma\nabla\phi)
=\nabla\cdot\left[\sigma(\mathbf u\times\mathbf B)\right].
$$

Fully developed models set $\mathbf u=(u(y,z),0,0)$ and solve the coupled
cross-section equations. The three-dimensional staggered core
(`lmhdx.core3d`, `lmhdx.steady`) retains all three velocity, current and
Lorentz-force components; a duct with an inlet and an outlet (`lmhdx.axial`)
is solved in the inertialess limit, without $\mathbf u\cdot\nabla\mathbf u$, or
with it by Newton's method from that solution; the
pipe with an inlet and an outlet is solved in the inertialess limit only.
The inertialess core-flow model (`lmhdx.coreflow`) reduces the core of a
thin-walled duct in a field $B_y(x)$ to three two-dimensional equations
(ANL/FPP/TM-228, eqs. 4a–4c). Their verification and open gates are in the
[validation matrix](../validation/index.md).

No-slip walls set velocity to zero. Insulating walls impose zero normal
current. Conducting solid regions solve potential with their own conductivity
and enforce potential and normal-current continuity at interfaces. Pipe meshes
use cylindrical/mapped electric metrics; rectangular electric operators use
nonuniform Cartesian finite-volume metrics. Generic momentum uses a distinct
discretization: electric closure alone does not establish nonuniform or
cylindrical vector-momentum consistency.

Common dimensionless groups are

$$
Ha=BL\sqrt{\frac{\sigma}{\rho\nu}},\qquad
Re=\frac{UL}{\nu},\qquad
N=\frac{Ha^2}{Re},\qquad
Rm=\mu_0\sigma UL.
$$

The inductionless model requires $Rm\ll1$. Geometry length, field orientation,
wall conductance ratio, and velocity scale must accompany any reported value.

## Quasi-two-dimensional model

For a strong uniform field normal to a shallow flow plane, the basic
Sommeria--Moreau closure evolves depth-averaged vorticity:

$$
\frac{\partial\omega}{\partial t}
+\mathbf u_\perp\cdot\nabla_\perp\omega
=\nu\nabla_\perp^2\omega-\frac{\omega}{\tau_H}+f_\omega,
$$

$$
\mathbf u_\perp=(\partial_y\psi,-\partial_x\psi),\qquad
-\nabla_\perp^2\psi=\omega.
$$

Here $1/\tau_H$ is the linear Hartmann-layer friction and $f_\omega$ is a
vorticity source. On a periodic domain the velocity is divergence free by
construction. With kinetic energy $E=\langle|\mathbf u_\perp|^2\rangle/2$ and
enstrophy $Z=\langle\omega^2\rangle/2$,

$$
\frac{dE}{dt}=-2\nu Z-\frac{2E}{\tau_H}+\langle\psi f_\omega\rangle.
$$

`solve_q2d` enforces `energy_budget_tolerance` (default $10^{-3}$) on the
absolute integrated defect divided by the larger of initial energy and
absolute integrated power, with the smallest positive normal working-dtype
value as a zero-energy floor. An absolute machine-epsilon floor would hide
relative defects in small-amplitude flows.
Power is integrated by the trapezoidal rule at every time step, independently
of saved-frame stride. This diagnostic has second-order quadrature error even
when the IFRK4 field evolution is more accurate. Refine the time step when it
fails; `completed` still does not certify mesh independence or steady flow.
The Q2D model is a
depth-averaged strong-field approximation; it does not resolve Hartmann-layer
profiles or replace the 3-D formulation where the field varies along the flow.

## Conventions of the channel and duct literature

Two papers agree on a flow only after their scales are converted. `channel(...)` takes the bulk
Reynolds number, or $R$ of Boeck, Krasnov and Zienicke, and works in half-width, bulk velocity,
unit density and conductivity, so $\nu = 1/Re_{av}$ and $B = Ha\sqrt{\nu}$.

| Source | Length | Velocity | $Re$ | $Ha$ |
|---|---|---|---|---|
| Krasnov, Zikanov and Boeck (2011, 2012), Krasnov et al. 2013, Zikanov et al. 2014 | half-width $L$ | bulk $U$ | $LU/\nu$ | $BL\sqrt{\sigma/\rho\nu}$ |
| Shatrov and Gerbeth 2010 | half-width $h$ | bulk $v_b$ | the same | the same |
| Boeck, Krasnov and Zienicke 2007 | half-width $d$ | $R = U_0\delta/\nu$ on the laminar centreline $U_0$, $\delta = d/Ha$ | $Re = Ha\,R$, $Re_{av} = Ha\,R_{av}$ | $d/\delta$ |
| Krasnov et al. 2010 | half the smaller side | centreline of the laminar base flow | on that velocity | on that length |
| Zhao and Zikanov 2012 | the full gap $d$ | $U_0 = V/(B_0 d)$ fixed by the voltage | $U_0 d/\nu$ | $dB_0\sqrt{\sigma/\rho\nu}$ |
| Kobayashi 2008 | half-width | bulk | 29,000 is the 14,500 of Krasnov et al. | |

The ratio $R = Re/Ha$ is the same in every convention. A half-gap Hartmann number is half the
full-gap one. In a laminar Hartmann channel the bulk velocity is
$U_{av}/U_0 = (\cosh Ha - \sinh Ha/Ha)/(\cosh Ha - 1)$, which is 0.90008 at $Ha = 10$ and 0.95000
at $Ha = 20$, and $c_f = 2\tau_w/\rho U_{av}^2$ with $Re_\tau = Re_{av}\sqrt{c_f/2}$.
