# The core-flow model

`lmhdx.coreflow` solves the three-dimensional core of a thin-walled rectangular duct in a field $B_y(x)$ at large
Hartmann number and interaction parameter, following Hua, Walker, Picologlou and Reed (ANL/FPP/TM-228, 1988). It is
inertialess, and it is the reference that the three-dimensional open duct is compared with
([Open ducts](open_ducts.md)).

(coreflow-model)=
## The model

Inertia and viscosity act only in layers. In the core $\nabla p=\mathbf j\times\mathbf B$, so the pressure is constant
along field lines. The core then reduces to three functions of two variables on one quadrant ($0\le y\le a$,
$-1\le z\le 0$): the pressure $p(x,z)$, the Hartmann-wall potential $\varphi_t(x,z)$ and the side-wall potential
$\varphi_s(x,y)$. With $\beta=1/B$ and $K=\beta^2+a^2\beta'^2/3$, TM-228 eqs. (4a)-(4c) are

$$
\partial_x(\beta^2\partial_x p)+K\,\partial_{zz}p=\beta'\,\partial_z\varphi_t,\qquad
c_t\nabla^2\varphi_t=a\beta'\,\partial_z p,\qquad
c_s\nabla^2\varphi_s=-\beta\,\partial_x p(x,-1),
$$

with $\varphi_t=0$ and $\partial_z p=0$ at $z=0$, $\partial_y\varphi_s=0$ at $y=0$, the corner conditions
$\varphi_t(x,-1)=\varphi_s(x,a)$ and $c_t\partial_z\varphi_t=c_s\partial_y\varphi_s$ (7d, e), and the side-layer flux
closure (14),
$K\,\partial_z p(x,-1)=\beta'\varphi_t(x,-1)-a^{-1}\,\mathrm d_x\big(\beta\!\int_0^a\varphi_s\,\mathrm dy\big)$. The
closure states that the core and side-layer flux
$Q=-a\beta^2\!\int\partial_x p\,\mathrm dz-\beta\!\int\varphi_s\,\mathrm dy$ is the same at every station. In a
uniform field these give Walker's fully developed gradient $-\partial_x p=B^2/(1+a/c_t+a^2/(3c_s))$ at unit mean
velocity, and $c_t/(a+c_t)$ with perfectly conducting side walls.

(coreflow-discretization)=
## Discretization

The equations are the stationarity conditions of one functional, maximal in $p$ and minimal in the potentials,

$$
\mathcal L=\iint\Big[-\tfrac a2\big(\beta^2p_x^2+Kp_z^2\big)+\tfrac{c_t}2|\nabla\varphi_t|^2
-a\beta'p\,\partial_z\varphi_t\Big]\mathrm dx\,\mathrm dz
+\iint\tfrac{c_s}2|\nabla\varphi_s|^2\,\mathrm dx\,\mathrm dy-\int p(x,-1)\,q\,\mathrm dx,
$$

with $q=aK\partial_zp(x,-1)$ the flux into the side layer. LMhdX discretizes $\mathcal L$ itself, so the coupled
operator is symmetric by construction and indefinite.

- As in TM-228 section 3, the grid is staggered in $z$. It puts $\varphi_t$ on nodes from the corner to $z=0$, $p$ at
  the centres between them, and $\varphi_s$ on nodes in $y$.
- The two walls share the corner node. Its molecule splits between half a cell of each wall, which carries (7d, e)
  without further equations.
- The wall pressure in (4c) comes from the closure, $p(-1)=p_{1/2}-\tfrac{\Delta z}2\partial_zp(-1)$ (the
  higher-order expansion of TM-228 section 3.2). In $\mathcal L$ this is the term
  $\tfrac{\Delta z}{4a}\int q^2/K\,\mathrm dx$.
- The ends are fully developed (5a-f). There $p$ is given and $\partial_x\varphi=0$ is natural, and the solver then
  rescales the solution once to the imposed flow rate. As in TM-228, the solver floors $\beta$ at 1000 and counts the
  floored stations.

The solver takes pressure and potentials together, never segregated, because TM-228 reports that a segregated
iteration diverges for small wall conductance. The system is two-dimensional and small, so the solver factorizes it
directly (SuperLU on the host) inside `jax.lax.custom_linear_solve`. Derivatives with respect to the wall
conductances, the field scale and the drive are exact in both modes, and the adjoint reuses the factorization.

(coreflow-anl)=
## The ANL fringe

On the ANL fringe ($x_0=3$, $c_t=c_s=0.02$, $a=1$, a domain running past the fringe into the capped zero field) the
gated quantity is the three-dimensional excess of the drop between $x=-6$ and $x=2$ over the locally fully developed
drop. LMhdX computes both drops on the same mesh. The excess is 0.01791, 0.01783 and 0.01781 on stations 0.2, 0.1 and
0.05 apart with 10, 20 and 40 cells across each wall, against TM-228's $0.0932-0.0754=0.0178$ (1 %) and its Figure 10
value $0.126\,c^{1/2}=0.01782$.

LMhdX reports the absolute drop but does not gate it, because TM-228's own fully developed gradient integrates to
0.0776 over $[-6,2]$, not the quoted 0.0754. The drop is 0.0951 against TM-228's 0.0932 (2.0 %) on a domain of exactly
$[-6,2]$ with fully developed ends, and 0.0954 on the longer domain. The model neglects inertia, an error that scales
as $N^{-1/3}$ (Mistrangelo et al. 2021). At ALEX B2 ($N=540$) that error is as large as the three-dimensional excess,
which is why B2 keeps a 5-10 % tolerance.

(coreflow-layers)=
## The layers' conductance

TM-228's core flow is the $c\sqrt{Ha}\to\infty$ limit. At finite Ha the Hartmann and side layers add conductance, a
$1/Ha$ to $c_t$ and $k\,Ha^{-1/2}$ to $c_s$. Both conductances are inputs of `CoreFlow.solve`, which also takes them per
station (an array or a callable of $x$). `lmhdx.coreflow.layer_conductances(c_t, c_s, Ha, B, k)` builds them at the
local field, with $k$ from `lmhdx.coreflow.side_layer_coefficient(c, Ha)` (one fully developed solve on the core).
Where $Ha\,B < 1$ the closure freezes the correction (`floor`).

The local closure at $c=0.1$, Ha $2\times10^4$ on the ANL fringe takes about 2 s. `tests/test_coreflow.py` compares it
with the stored 3-D numbers, so no 3-D solve runs.

```python
import numpy as np

from lmhdx.coreflow import CoreFlow, layer_conductances

x = np.linspace(-10.0, 6.0, 161)
field = 0.5 * (1.0 - np.sin(np.pi * np.clip(x, -3.0, 3.0) / 6.0))  # the ANL fringe, B_y(x)
k = ((100, 400, 800, 1600, 3200, 1e4, 2e4), (2.585, 1.397, 1.242, 1.138, 1.041, 0.832, 0.636))
c_t, c_s = layer_conductances(0.1, 0.1, 2e4, field, k)  # c = 0.1 at Ha 2e4, per station
pressure = np.asarray(CoreFlow(x, nz=20, ny=20).solve(field, c_t=c_t, c_s=c_s).pressure).mean(axis=1)
print(f"drop over [-6, 2]: {np.interp(-6.0, x, pressure) - np.interp(2.0, x, pressure):.4f}")
# drop over [-6, 2]: 0.4126
```

Subtracting the locally fully developed drop, 0.37050, leaves an excess of 0.0421, against the 3-D core's 0.04209.
With this closure the core-flow model agrees with the 3-D core within 1 % at $c=0.1$, Ha $2\times10^4$, and within 5 %
at $c=0.02$. The comparison over Ha, and the verdict for rows 7 and 24, are on
{ref}`Open ducts <validation-open-ducts-layers>`.
