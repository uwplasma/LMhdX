# Equations and assumptions

LMhdX solves incompressible liquid-metal flow in an imposed magnetic field. The field is fixed. The currents the flow
induces are too weak to change it, because the magnetic Reynolds number is small. LMhdX therefore solves only for the
electric potential and the current, together with the velocity and pressure. This page states the equations and the
reduced form each solver uses. [Numerics](numerics.md) describes how LMhdX discretizes them.

## The core equations

The full isothermal formulation uses density $\rho$, kinematic viscosity $\nu$, electrical conductivity $\sigma$,
pressure $p$ and a body drive $\mathbf f$:

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

No-slip walls set velocity to zero. Insulating walls impose zero normal current. Conducting solid regions solve
potential with their own conductivity and enforce potential and normal-current continuity at interfaces. Pipe meshes
use cylindrical or mapped electric metrics. Rectangular electric operators use nonuniform Cartesian finite-volume
metrics. Generic momentum uses a distinct discretization, so electric closure alone does not establish nonuniform or
cylindrical vector-momentum consistency.

Common dimensionless groups are

$$
Ha=BL\sqrt{\frac{\sigma}{\rho\nu}},\qquad
Re=\frac{UL}{\nu},\qquad
N=\frac{Ha^2}{Re},\qquad
Rm=\mu_0\sigma UL.
$$

The inductionless model requires $Rm\ll1$. Report the geometry length, field orientation, wall conductance ratio, and
velocity scale with any value.

## The model hierarchy

Each solver keeps part of these equations:

| Solver | What it keeps | Page |
|:--|:--|:--|
| Three-dimensional staggered core (`lmhdx.core3d`, `lmhdx.steady`) | all three components of velocity, current and Lorentz force | [time stepping](numerics/time.md), [steady solves](numerics/steady.md) |
| Fully developed duct or pipe | the coupled cross-section equations for $\mathbf u=(u(y,z),0,0)$ | [steady solves](numerics/steady.md) |
| Duct with an inlet and an outlet (`lmhdx.axial`) | the inertialess limit, without $\mathbf u\cdot\nabla\mathbf u$; or inertia, by Newton's method or by marching to the steady state | [open ducts](open_ducts.md) |
| Pipe with an inlet and an outlet | the inertialess limit only | {ref}`open pipe <open-ducts-pipe>` |
| Core-flow model (`lmhdx.coreflow`) | the core of a thin-walled duct in a field $B_y(x)$, as three two-dimensional equations (ANL/FPP/TM-228, eqs. 4a–4c) | [core-flow model](coreflow.md) |
| Quasi-two-dimensional model (`lmhdx.q2d`) | depth-averaged vorticity in a strong uniform field normal to a periodic plane | [Q2D](q2d.md) |
| Heat transport (`lmhdx.thermal`) | temperature carried and conducted, with Boussinesq buoyancy and Joule heat | [heat transport](thermal.md) |

Their verification and open gates are in the [validation matrix](../validation/index.md).

## Conventions of the channel and duct literature

Two papers agree on a flow only after you convert their scales. `channel(...)` takes the bulk
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
