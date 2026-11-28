# The ANL fringe duct and the ALEX pipe

This page holds the evidence for the open duct and the open pipe of [Ducts with an inlet and an
outlet](../physics/open_ducts.md), rows 6, 7 and 24 of the [validation matrix](index.md). The duct runs through the ANL
fringe against TM-228 and LMhdX's [core-flow model](../physics/coreflow.md). The pipe runs through the ALEX B1 magnet
against the measurement.

The ANL fringe has $x_0=3$, $c=0.02$ on all four walls and $B_y$ alone. Its buffers are 15 half-widths upstream and 10
downstream, with 0.25 axial spacing over $[-6,3]$ for 70 axial cells and over the ramp $[-3,3]$ for 60. Every run is in
the Stokes limit unless it names a Reynolds number. Drops are in units of $\sigma U_0B_0^2a$.

(validation-open-ducts-balances)=
## Balances

| Ha, cells | mesh | flow rate | mass | charge | uniform-region $\partial_xp$ | CG iterations |
|---|---|---|---|---|---|---|
| 100, 24 | 70 × 24 × 24 | 4.4e-16 | 1.0e-16 | 2.2e-13 | 3.5e-5 | 1,444 |
| 100, 32 | 60 × 32 × 32 | 2.2e-16 | 1.9e-16 | 8.7e-14 | 4.3e-5 | 774 |
| 400, 48 | 60 × 48 × 48 | 2.2e-16 | 1.9e-16 | 7.3e-14 | 5.6e-5 | 2,030 |
| 1600, 96 | 60 × 96 × 96 | 2.2e-16 | 1.2e-16 | 7.5e-14 | 6.0e-5 | 4,846 |

The flow rate is the largest error over every axial face relative to the imposed one. Mass and charge are the largest
net flux out of a cell relative to the largest gross flux through a cell (the motional flux, for charge). The gradient
is the relative difference of the mean pressure gradient over $-12<x<-8$ from the fully developed one. Doubling both
buffers (to 30 and 20) moves the drop over $[-6,2]$ by 3.5e-9 of itself and the axial current through the window's two
end faces by 3e-7 (row 28 asks for 0.5 %).

(validation-open-ducts-fringe)=
## The excess on the three-dimensional core

`lmhdx.axial.fringe_duct` is the core-flow gate's geometry at finite Hartmann number. The excess is
$\Delta p-\Delta p_{FD}$ over $[-6,2]$. Here $\Delta p_{FD}$ is LMhdX's own fully developed gradient at the local
field, integrated with 24 Gauss points over the ramp. All runs used one A4000.

| Ha | mesh | $\Delta p$ | $\Delta p_{FD}$ | excess | CG iterations | warm |
|---|---|---|---|---|---|---|
| 100 | 60 × 32² | 0.2276 | 0.1651 | 0.06257 | 774 | 6.3 s |
| 200 | 60 × 32² | 0.1815 | 0.1288 | 0.05272 | 1,748 | 10.1 s |
| 400 | 60 × 48² | 0.1559 | 0.1109 | 0.04503 | 2,030 | 21 s |
| 800 | 60 × 64² | 0.1398 | 0.1008 | 0.03893 | 2,993 | 91 s |
| 1600 | 60 × 96² | 0.1289 | 0.0947 | 0.03419 | 4,846 | 183 s |
| 3200 | 70 × 128² | 0.1209 | 0.0905 | 0.03042 | 8,321 | 820 s |

- **The mesh is converged.** At Ha 400, $41\times32^2$ (axial spacing 0.5), $70\times48^2$ (0.25) and $118\times64^2$
  (0.125) give 0.04461, 0.04504 and 0.04514, a change of 0.2 % at the last refinement.
- **The excess decays slowly.** It shrinks by a factor of 0.78–0.80 per doubling of Ha, a power $Ha^{-0.35}$, not
  $Ha^{-1/2}$. A free power law extrapolates to 0.0158–0.0168, 6–11 % below the core-flow model's
  0.01783 (and TM-228's 0.0178).
- **Rows 7 and 24 are not met here.** At Ha 3200 the excess is still 71 % above the core-flow value. The absolute drop
  falls from 0.228 to 0.121, against TM-228's 0.0932. At Ha $10^4$ on the Ha 3200 mesh the drop is 0.11152, the
  locally fully developed one 0.08575 and the excess 0.02576, 44 % above 0.01783, a slope of $Ha^{-0.15}$. The meshes
  at high Ha below check that mesh.

(validation-open-ducts-inertia)=
## The inertial share

The inertial share is $\Delta p(Re)-\Delta p(0)$ over the excess, with $Re=Ua/\nu$ on the mean velocity and
$N=Ha^2/Re$. The Stokes-preconditioned Newton solve measured it on the ANL fringe with buffers at the core spacing
(`fringe_duct(growth=1.0)`, $124\times32^2$, cell Reynolds number $\le 8$). The relative residual is $\le 10^{-10}$,
the mass balance $\le 1.5\times10^{-16}$ and the charge balance $\le 1.9\times10^{-13}$.

| Ha | Re | N | $\Delta p$ | inertial | share | outer iterations | cold time |
|---|---|---|---|---|---|---|---|
| 100 | 0 | ∞ | 0.227655 | | | (245 CG) | |
| 100 | 10 | 1000 | 0.227919 | 0.000265 | 0.42 % | 39 | 89 s (46 s warm) |
| 100 | 20 | 500 | 0.228346 | 0.000691 | 1.10 % | 247 | 12 min |
| 100 | 33 | 303 | 0.228977 | 0.001323 | 2.11 % | 4,743 | 1.9 h |
| 200 | 0 | ∞ | 0.181530 | | | (448 CG) | |
| 200 | 10 | 4000 | 0.181739 | 0.000210 | 0.40 % | 37 | 3.3 min |
| 200 | 20 | 2000 | 0.182048 | 0.000518 | 0.98 % | 229 | 21 min |
| 200 | 40 | 1000 | 0.182760 | 0.001230 | 2.33 % | 18,932 | 21.8 h |

Krylov recycling took the outer iterations at Re 20 from 313 to 224 (Ha 100) and from 289 to 206 (Ha 200), but from
5,302 to 11,202 at Ha 100, Re 33. The solve does not use it. The outer counts predate the outlet face's $u^2$ flux,
whose mode caused much of their growth with Re. With the default buffer growth (cell Reynolds number 66) the Ha 100,
Re 33 run stalled past Re 31.35.

- At fixed Re the share depends little on Ha. At fixed N it changes, 0.42 % at Ha 100, Re 10 against 2.33 % at Ha 200,
  Re 40, both at N 1000. So the data do not support the plan's $\beta N^{-1/3}$ term on Ha 100–200, Re 10–40.
- A least-squares fit of $\ln$ share on $\ln Re$ and $\ln Ha$ gives share
  $\approx 0.045\,\% \cdot Re^{1.31} Ha^{-0.16}$. A fit in Re alone gives $0.021\,\% \cdot Re^{1.30}$. The 2.3 % of
  #218 on a $70\times32^2$ mesh agrees with the 2.33 % here.
- Station by station, the inertial change of $-\partial_x\bar p$ is $\le 2.3\,\%$ of the local gradient wherever
  $N b(x)^2 \ge 125$, and of order the gradient only where $N b^2 \lesssim 30$ (downstream of $x=1$), where the
  gradient itself is small.

The full campaign notes are in [numerics.md at
b97ad3a](https://github.com/uwplasma/LMhdX/blob/b97ad3a8525a20502279961d5b4201d158d971e9/docs/physics/numerics.md#inertia-on-the-open-axis-the-inertial-share-19d).

(validation-open-ducts-layers)=
## The layers' conductance (rows 7 and 24)

LMhdX's fully developed gradient lies above Walker's thin-wall value $1/(1+1/c_t+1/(3c_s))$ at every finite Ha.
Counting the Hartmann layers as a conductance $1/Ha$ added to $c_t$ and inverting Walker's formula for the side walls
gives $c_{s,\mathrm{eff}} = c + k\,Ha^{-1/2}$, the classical side-layer flux.

- At $c=0.02$, $k$ = 1.91, 1.46, 1.28, 1.19, 1.11 and 1.07 at Ha 400, 800, 1600, 3200, $10^4$ and $2\times10^4$. At
  $c=0.1$ it is 1.40, 1.24, 1.14, 1.04, 0.83 and 0.64.
- Relative to the wall the added conductance is $k/(c\sqrt{Ha})$, 55 % at $c=0.02$, Ha $10^4$, but 8 % at $c=0.1$.
- In the core-flow model, doubling $c_s$ from 0.02 to 0.04 raises the excess from 0.0178 to 0.0282.

**Meshes at high Ha.** Each refinement moves the excess by under 0.1 %. At Ha $10^4$ (#188's run reused the Ha 3200
mesh), $70\times192^2$ gives 0.025772, against 0.02576 on $70\times128^2$, and $118\times128^2$ gives 0.025783. At
$c=0.1$, Ha $2\times10^4$, tightening the CG tolerance from $10^{-7}$ to $10^{-8}$ moves the excess by
$1\times10^{-12}$, and $70\times256^2$ gives 0.042063 against 0.042087 on $70\times192^2$ (0.06 %). That run takes
25,000–26,000 CG iterations, about 100 minutes on one A4000.

**Two closures.** The global closure takes $c_t = c + 1/Ha$ and $c_s = c_{s,\mathrm{eff}}(Ha)$, uniform along the duct.
The local closure applies the same laws at the local field, $c_t = c + 1/(Ha\,B)$ and $c_s = c + k(Ha\,B)/\sqrt{Ha\,B}$,
with $k$ interpolated in $\log Ha$ (`lmhdx.coreflow.layer_conductances`). Freezing the correction at $Ha\,B < 100$ or
$1000$ instead of 1 moves the result by at most 0.6 % at $2\times10^4$ and 11 % at 3200, the ranges in the table.

| Ha | $c=0.02$: 3-D core | global | local | $c=0.1$: 3-D core | global | local |
|---|---|---|---|---|---|---|
| 400 | 0.04503 | 0.04967 | 0.0895 | 0.05510 | 0.05522 | 0.0786 |
| 800 | 0.03893 | 0.03865 | 0.0639 | 0.05077 | 0.04973 | 0.0641 |
| 1600 | 0.03419 | 0.03261 | 0.0471 | 0.04751 | 0.04621 | 0.0549 |
| 3200 | 0.03042 | 0.02855 | 0.0329–0.0369 | 0.04506 | 0.04379 | 0.0492 |
| $10^4$ | 0.02577 | 0.02415 | 0.0274–0.0280 | 0.04230 | 0.04124 | 0.0433–0.0439 |
| $2\times10^4$ | 0.02375 | 0.02235 | 0.0248–0.0249 | 0.04209 | 0.04022 | 0.0418–0.0421 |
| $\infty$ | | 0.01783 | 0.01783 | | 0.03901 | 0.03901 |

- The two closures bracket the 3-D core from Ha 800 up, at both conductances, and both tend to the core-flow model at
  $Ha\to\infty$.
- The 3-D core approaches the local closure. At $c=0.02$ it lies 6–8 % below at Ha $10^4$ and 4–5 % below at
  $2\times10^4$. At $c=0.1$ it lies 2–4 % below at $10^4$ and inside the closure's own 0.6 % range at $2\times10^4$.
- The fully developed parts agree to 0.05 %, 0.08365 against 0.08366, and 0.37050 against 0.37066 at $2\times10^4$.

**Extrapolation.** At $c=0.02$, fits in $Ha^{-1/2}$ and $Ha^{-1}$ through Ha 400–$2\times10^4$ give 0.0183–0.0211
(3–18 % above 0.01783), and the excess is still 33 % above it at $2\times10^4$. At $c=0.1$ the same fits give
0.0395–0.0405 (1–4 % above 0.03901). Reaching TM-228's value to 1 % at $c=0.02$ needs the side-layer term below about
1.5 % of $c$, that is $c\sqrt{Ha}\gtrsim 70$ and Ha ≳ $10^7$. No three-dimensional solve reaches that.

**Verdict.** The study found no modelling error in the 3-D core, the core-flow model or the buffers. The gap is a
physics difference. TM-228's inertialess core flow is the $c\sqrt{Ha}\to\infty$ limit, without the layers'
$O(Ha^{-1})$ and $O(Ha^{-1/2})$ conductance. With that conductance at the local field the core-flow model agrees with
the 3-D core within 1 % at $c=0.1$, Ha $2\times10^4$ (row 24's comparison, at finite Ha), and within 5 % at $c=0.02$.
On 2026-10-04 the owner ruled that row 24 is met against the layer-corrected core-flow model. Row 7's 1 % gate at
$c=0.02$ against 0.0178 is not met and not reachable on the three-dimensional core. The owner closed it as a physical
model difference, not a waiver.

(validation-open-ducts-pipe)=
## The ALEX pipe (row 6)

Buoyancy drives the observables of Mistrangelo et al. 2025, and LMhdX does not package its reference values. So
row 6 runs on the ALEX B1 pipe (Ha 6600, N 10700, $c=0.027$, a thin conducting wall).
`lmhdx/data/benchmarks/references/alex-b1-pipe.csv` packages the measured field table and the digitized excess pressure
gradient, with an uncertainty of 0.002. The Stokes limit applies, since $N=10700$ under the magnet and the duct's
inertial share at $N=1000$ is 0.4 %.

**The open pipe's checks.** On the test mesh ($10\times12\times14$, Ha 20, $c=0.05$, CPU) the operator is symmetric to
2.5e-14, the energy identity holds to 1e-15, and charge, flow rate and mass hold to 5e-14, 1e-13 and 1e-14. Under a
uniform field the open pipe keeps the two-dimensional pipe's profile to 4e-13 and its gradient to 4e-13. The adjoint of
the drop in the field scale matches central differences to 2.8e-10.

**ALEX B1** runs on one A4000, with the field table interpolated by a monotone cubic, $x\in[-15,10]$, and axial spacing
0.25 over $[-6,7.5]$. The observable is $-\partial_x\bar p/(\sigma U B_0^2)$ minus its plateau over $x\ge7.5$, at the
16 tabulated stations.

| Ha | mesh $r\times\theta\times x$ | CG iterations | warm | upstream | integrated | RMS / $L_\infty$ (in uncertainties) |
|---|---|---|---|---|---|---|
| 400 | 24 × 32 × 41 (spacing 0.5) | 1,076 | 2.1 s | 0.0295 | +8.3 % | 2.51 / 4.2 |
| 1600 | 32 × 48 × 41 (0.5) | 3,281 | 23 s | 0.0272 | −0.9 % | 2.36 / 4.9 |
| 6600 | 48 × 64 × 76 | 11,245 | 156 s | 0.02664 | −3.3 % | 2.43 / 5.08 |
| 6600 | 48 × 128 × 76 | 11,698 | 617 s | 0.02668 | −3.2 % | 2.44 / 5.08 |
| 6600 | 64 × 64 × 76 (8 cells in the layer) | 11,671 | 437 s | 0.02664 | −3.3 % | 2.43 / 5.08 |
| 6600 | 48 × 64 × 140 (spacing 0.125) | 11,437 | 740 s | 0.02665 | −3.3 % | 2.44 / 5.13 |

- The cross-section is converged. Doubling the azimuth or going to 64 radial cells moves no station by more than 3e-5.
- Upstream, LMhdX gives 0.0266 against 0.0250 measured, inside the 0.002 uncertainty. The integrated excess is within
  3.3 %, against a frozen gate of 10 %.
- Station by station the section mean is outside the data, with a weighted RMS of 2.4 against the frozen 1 and an
  $L_\infty$ of 5.1 against 2. Upstream of the pole edge the model peaks at $x=-3$ (0.0279 against 0.0220), where the
  axial currents of the fringe add to the local gradient. Downstream (from $x=-1$ to 2) it falls faster than the
  measurement, close to the locally fully developed $0.026\,b^2$. Neither feature changes with the mesh.
- **The section mean does not meet row 6 at its frozen station-wise tolerance.** It meets row 6 on the integrated
  excess and on the upstream gradient.

**The wall reads differently.** A pressure tap reads the wall, and behind a fringe the pressure is not uniform across
the section. Read at the wall (`wall_pressure_gradient`, the outermost cell, Ha 6600, $48\times64\times76$), the
weighted RMS and $L_\infty$ are 2.43 and 5.08 for the section mean, 1.79 and 3.82 for the wall ring average, and 3.11
and 6.60 at the walls normal to the field ($\theta=0,\pi$). At the walls parallel to the field ($\theta=\pm\pi/2$) they
are 0.86 and 1.39, inside the frozen 1 and 2. The packaged table does not record where ALEX put its taps, so this is a
statement about the observable, not a tuned parameter. The model reproduces the data if they are side-wall taps, and
not if they are section means or Hartmann-wall taps. The original scanned report must settle which. The draft Newton
solve showed that inertia is about $30\times$ too small to close the gap.

Station values at $x=-15,-12.5,-10,-7.5,-5,-3,-2,-1,0,1,2,3,4,5,7.5,10$ (Ha 6600, $48\times64\times76$):

- model, section mean: 0.0265, 0.0267, 0.0266, 0.0265, 0.0251, 0.0279, 0.0230, 0.0159, 0.0054, 0.0023, −0.0001,
  −0.0005, −0.0002, −0.0001, 0, 0;
- model, walls parallel to the field: 0.0264, 0.0263, 0.0260, 0.0250, 0.0228, 0.0193, 0.0180, 0.0167, 0.0138, 0.0097,
  0.0059, 0.0027, 0.0009, 0.0002, 0, 0;
- measured: 0.0250, 0.0252, 0.0250, 0.0247, 0.0235, 0.0220, 0.0205, 0.0185, 0.0155, 0.0125, 0.0085, 0.0050, 0.0025,
  0.0010, 0, 0.

**Sensitivity.** The next table scores variations at Ha 1600 on $32\times48\times41$ against the measurement. RMS and
$L_\infty$ are in uncertainties, and "upstream" is the model's mean gradient for $x<-10$, where the measured value is
0.0250.

| variation | upstream | section mean | side wall |
|---|---|---|---|
| baseline (monotone cubic, $c=0.027$) | 0.0272 | 2.36 / 4.89 | 0.79 / 1.21 |
| linear interpolation | 0.0272 | 2.35 / 4.90 | 0.80 / 1.23 |
| cubic spline | 0.0271 | 2.35 / 4.91 | 0.77 / 1.23 |
| $b+0.03$ (the table's uncertainty) | 0.0270 | 2.41 / 4.56 | 0.63 / 1.01 |
| $b-0.03$ | 0.0256 | 2.35 / 5.19 | 1.01 / 1.77 |
| ramp shifted $+0.5$ | 0.0272 | 2.07 / 4.25 | 0.51 / 0.98 |
| ramp shifted $-0.5$ | 0.0272 | 2.71 / 5.75 | 1.17 / 2.05 |
| $c=0.020$ | 0.0205 | 2.72 / 5.47 | 2.15 / 3.68 |
| $c=0.035$ | 0.0347 | 3.72 / 6.09 | 2.50 / 4.74 |
| $c=0.054$ | 0.0520 | 8.61 / 13.47 | 7.77 / 13.43 |

The upstream gradient fixes the wall conductance. The data admit $c$ of about 0.025 to 0.029, and $c=0.020$ or 0.035
miss it by 18 % and 39 %. No variation closes the gap for the section mean, but the choice of observable does. This
study did not test a finite-thickness wall, the field's $B_x$ component or inertia.
