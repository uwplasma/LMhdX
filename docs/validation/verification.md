# Code verification

Verification asks whether LMhdX solves its equations correctly. When you refine the mesh, the error against a known
solution must fall at the scheme's design order. Validation, on the other pages, asks whether those equations describe
the flow.

This page checks the step every run in time takes, `lmhdx.thermal.build_thermal_step`, against solutions that share none
of its operators. These are a manufactured solution for each term and for all of them at once, and the exact decay of a
vortex between Hartmann walls. **Every term converges at second order in space. The step is first order in time, and
with the explicit Lorentz force its time error does not grow with the field.**

```{image} ../_static/verification_orders.webp
:alt: The observed order of accuracy of each term of the thermal step, L-infinity and RMS, near two for every term
:align: center
```

## Run it

This runs one term, conduction, on two meshes in about 4 s on a laptop CPU:

```python
import numpy as np

from validation.verification import manufactured_run

coarse, fine = (manufactured_run("conduction", cells)["rms"]["T"] for cells in (16, 32))
print(f"RMS error {coarse:.2e} on 16 cells, {fine:.2e} on 32: order {np.log2(coarse / fine):.2f}")
# RMS error 1.29e-03 on 16 cells, 3.51e-04 on 32: order 1.88
```

`validation/verification.py` holds every case (`TERMS`, `vortex_decay`, `hartmann_roll_growth`).
`python scripts/make_showcase_figures.py --only verification` reruns them all, in about 40 minutes on a laptop GPU,
and writes `docs/_static/data/verification.json` and the figure above. `tests/test_steady.py` runs the coupled case,
three single terms and the vortex's decay at small size.

## Each term on its own

**Method.**
- Each manufactured solution is a smooth velocity, temperature and, where tested, pressure.
- Its source is the residual of the continuous equations, taken at the step's old time, as the step takes its
  forcing. Only the term under test is on, besides the time derivative and the incompressibility the step always
  enforces.
- The meshes have $N$ = 8, 16, 32 and 64 cells along $x$ and $z$ (128 for the cases with two cells along $y$), and
  half as many along $y$.
- The time step falls as $h^2$ from 0.01 at $N=8$, so that a first-order time error falls with the mesh too.
- The script measures the errors, exact less computed, at $t=0.2$ on every velocity face or cell, in $L_\infty$ and
  RMS.

The field lies along $y$. A flow invariant along it has the electromotive force $B\nabla\psi$, which the potential
$B\psi$ cancels, so it carries no current. A flow with no vorticity along it carries $\sigma\mathbf u\times\mathbf B$.
Both vanish with their normal derivatives at the walls $z=\pm1$, so the potential and current are exact, conducting
walls included.

The observed orders on the finest pair of meshes:

| Term | Tested | Field | $L_\infty$ | $L_2$ |
|:--|:--|:-:|--:|--:|
| Momentum advection | $(\mathbf u\cdot\nabla)\mathbf u$, limited, triply periodic | $u_x$ | 1.76 | 2.07 |
| Viscous diffusion | $\nu\nabla^2\mathbf u$, no-slip walls | $u_x$ | 1.85 | 1.96 |
| Pressure gradient | $-\nabla p/\rho$, a manufactured pressure | $p$ | 1.96 | 2.00 |
| Lorentz force, balanced part | $\mathbf j=\sigma(-\nabla\phi+\mathbf u\times\mathbf B)=\mathbf 0$ | $u_x$ | 1.93 | 1.99 |
| Lorentz force, damped part | $\mathbf j\times\mathbf B/\rho=-(\sigma B^2/\rho)\,\mathbf u_\perp$ | $u_x$ | 1.98 | 1.98 |
| Lorentz force with conducting walls | $\nabla\cdot(\sigma\nabla\phi)=\nabla\cdot(\sigma\,\mathbf u\times\mathbf B)$ | $u_x$ | 1.93 | 1.94 |
| Braking by Hartmann walls | the exact eigenmode $u_z=U(y)\cos(kx)\,e^{st}$ | $u_z$ | 1.21 | 2.06 |
| Buoyancy | $-\beta\,(T-T_{\mathrm r})\,\mathbf g$, the exact velocity zero | $u_z$ | 1.99 | 1.99 |
| Heat conduction | $\kappa\nabla^2T$, isothermal walls | $T$ | 1.84 | 1.97 |
| Heat advection | $\mathbf u\cdot\nabla T$, van Leer, triply periodic | $T$ | 1.36 | 2.08 |
| Joule heating | $\lvert\mathbf j\rvert^2/(\sigma\rho c_p)$ of a held flow | $T$ | 1.93 | 1.99 |
| Conjugate conduction | $\nabla\cdot(k\nabla T)$, $[T]=[k\,\partial_zT]=0$ at the walls | $T$ | 1.94 | 2.00 |
| All terms together | conducting walls | $u_x$ | 1.93 | 2.01 |

```{image} ../_static/verification_orders.webp
:alt: The observed order of accuracy of each term of the thermal step, L-infinity and RMS, near two for every term
:align: center
```

- Every term converges at second order in RMS. Two $L_\infty$ orders are lower, for known reasons:
  - Heat advection's 1.36 comes from the van Leer limiter, which clips extrema.
  - The Hartmann-wall case refines only along $x$. The 48 cells across M1's walls are fixed, and their error is the
    floor that the $L_\infty$ error meets at $N=64$.
- The conjugate case has walls 0.25 thick, with four times the fluid's conductivity and twice its heat capacity. The
  temperature and heat flux are continuous across them, and its errors include the wall cells.
- The script also draws each term's error along a line and against the mesh, as PDFs in `docs/_static/generated/`,
  which git does not track.

## In time, with the field on

**The coupled case in time.** On 32 cells to $t=1$, the differences between successive halvings of $\Delta t$ fall at
first order.
- With the explicit Lorentz force, their constant on $u_x$ does not grow with the field. It is 2.20 with no field,
  2.14 at $\sigma B^2/\rho=10$ and 2.12 at 30.
- Under the implicit damping shift it is 3.6 and 7.7, which is the shift's transient error
  ([numerics](../physics/numerics.md), "What the shift costs a transient").
- At a small step, the shift's error is the floor at which spatial refinement stops.

**Braking by Hartmann walls.** Take a vortex invariant along the field, between the walls the field crosses. The walls'
Hartmann layers and the current that closes through them brake it. `validation.shercliff.vortex_decay_rate` solves its
decay exactly, by Chebyshev collocation across the walls, with the resolved wall as a Robin condition on the potential.
For Case 2's steel (conductivity ratio 1.246, $t_w/a=0.087$) at Ha 235 it is $0.0933\,\sigma B^2/\rho$ at $k=\pi/2$, 22
times the insulating walls' $1/Ha$. `validation.verification.vortex_decay` runs the vortex on Case 2's M1 cross-section,
with 48 cells, 6 in each Hartmann layer and 6 in each wall:
- Doubling the cells across the duct, in the layers or in the walls changes the rate by under 0.1 %.
- Along the vortex's wavelength, the braking is weak by about $\sin^2(\pi/n)$ on $n$ cells, which gives 14.1, 3.7, 0.97
  and 0.29 % on 8, 16, 32 and 64. Case 2's rolls, 3.6–4.0 cm long on M1's 5 mm axial cells, have 7–8 cells a wavelength.
  - The cause is averaging. The step averages the electromotive force onto the current's faces, and the current back
    onto the velocity's. Each average multiplies a wave by $\cos(k\Delta x/2)$, so the Lorentz force does not brake a
    grid-scale wave across the field at all.
  - The collocated current-conservative schemes do the same (Ni et al. 2007, as restated by Gelfgat and Zikanov 2018).
    They interpolate $\mathbf u\times\mathbf B$ to the faces and the currents back to the cells.
- At Case 2's $\Delta t\,\sigma B^2/\rho$ of 0.34 (M1) and 0.27 (M2), the shift weakens the braking by 27 and 23 %,
  the factor $1+\Delta t\,\lambda$. The explicit force is within 1.9 %. Insulating walls are within 2.7 % on 16 cells
  and 1.3 % on 32.

The [thermal](../physics/thermal.md) page ("Growth rates") compares heated rolls between Hartmann walls with Sommeria
and Moreau's model.

## Manufactured solutions elsewhere

The other parts of LMhdX carry their own, on the pages of the code they check:
- the thin conducting wall's potential, second order ([numerics](../physics/numerics.md), the staggered grid);
- the RK3 projection step on Shatrov and Gerbeth's manufactured flow, order 2.2–2.3
  ([numerics](../physics/numerics.md), "The IMEX Runge-Kutta step");
- the open duct with heat and buoyancy, a coupled manufactured solution, second order
  ([heat transport](../physics/thermal.md), "Supported envelope");
- the staggered core on stretched meshes ([validation](index.md), row 8).

## Sources

- M.-J. Ni, R. Munipalli, N. B. Morley, P. Huang and M. A. Abdou, J. Comput. Phys. 227 (2007) 174–204 and
  205–228, doi:10.1016/j.jcp.2007.07.025 and 10.1016/j.jcp.2007.07.023.
- A. Yu. Gelfgat and O. Zikanov, *Computational modeling of magnetoconvection: effects of discretization method, grid
  refinement and grid stretching*, arXiv:1805.04710 (2018).
- J. Sommeria and R. Moreau, J. Fluid Mech. 118 (1982) 507–518.
- C. J. Roy, J. Comput. Phys. 205 (2005) 131–156 (order-of-accuracy verification with manufactured solutions).
