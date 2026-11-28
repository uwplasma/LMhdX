# Turbulent channel in a spanwise field

LMhdX's first turbulence test is a channel flow between two walls, with the
magnetic field parallel to the walls and across the flow. The reference is
Table 1 of Krasnov, Zikanov and Boeck (Comput. Fluids 50, 46, 2011). The
centreline velocity matches within 1 %. The friction Reynolds number is 8.7 %
high, outside the 2 % target, and this page gives the suspected reason.

```{image} ../_static/channel_turbulence.webp
:alt: Stationarity, mean velocity and wall units
:align: center
```

Run `python examples/channel_turbulence_example.py`, but expect more than an
hour on a CPU. It is also the scheduled test `tests/test_channel_turbulence.py`
(plan step H5.1), marked `slow` and outside the pull-request tiers, which the
`turbulence` workflow runs weekly on the uwplasma Macs.

## Case

- Plane Poiseuille flow between insulating walls at $z=\pm1$, periodic in $x$
  ($L_x=2\pi$) and $y$ ($L_y=\pi$), at constant bulk velocity.
- The field of Hartmann number $Ha=30$ along $y$, bulk Reynolds number
  $Re=6667$ (interaction parameter $Ha^2/Re=0.135$).
- Mesh $64^3$ with the tanh wall grid $z=\tanh(A\xi)/\tanh A$, $A=2$.
- IMEX RK3 with Crank-Nicolson viscous and damping terms, central transport,
  and $\Delta t=0.025$, which is 0.9 to 1.2 of the nominal stage-map limit and
  0.5 of it at $\Delta t=0.0125$. The limit's peak rate is a conservative sum
  over the axes.
- Times are in half-widths per bulk velocity. One time unit is
  $Re_\tau^2/Re\approx17$ viscous times.

A flux correction every 50 steps, or 1.25 time units, holds the bulk velocity
by adding the deficit $1-U_b$ as a divergence-free unit-bulk profile
$\propto 1-z^2$. The body force is a constant $2\times10^{-3}$, so the drive the
run needed is the force plus the mean correction per unit time. The run reports
$Re_\tau$ from the mean wall shear, from a quadratic through the wall and the
two nearest cell centres, and independently from that mean drive. The two agree
to 0.2 %. $U_{cl}$ is the mean over the two central cells.

The start is the laminar profile plus low-wavenumber noise of amplitude 0.2 in
modes up to 4. The instantaneous $Re_\tau$ peaks near 385 at $t=12$ and
settles by $t\approx100$.

## Statistics window

After a burn-in of 200, the run samples the plane-averaged mean profile every
1.25 time units for 200 time units, 160 samples. The window is about 3,300
viscous times, or 32 flow-throughs of $L_x$ at the bulk velocity. The Re_tau
trace over the window has a standard deviation of 3.4 and a block-mean standard
error of 0.7, or 0.2 %. The first and second half give 333.6 and 333.2. The
paper does not state its averaging time.

## Result and cost

| | $Re_\tau$ | $U_{cl}$ |
|---|---|---|
| spectral, 256³ (KZB 2011) | 306.69 | 1.2542 |
| scheme B 64³ / 128³ / 256³ (KZB 2011) | 306.99 / 306.45 / 304.72 | 1.2459 / 1.2532 / 1.2575 |
| LMhdX 64³, $\Delta t=0.025$ | 333.4 (+8.7 %) | 1.2415 (−1.0 %) |
| LMhdX 64³, $\Delta t=0.0125$, 100 time units | 334.7 (+9.1 %) | 1.2431 (−0.9 %) |

$U_{cl}$ is within the plan's 1.5 % but just below the finite-difference
spread. $Re_\tau$ is **not** within the plan's 2 %. It is 8.7 % high, far
outside both the statistical error and the time-step effect.

The probable cause is the scheme. The staggered central scheme adds no
numerical dissipation, and at $\Delta x^+\approx33$, $\Delta y^+\approx16$
nothing removes energy from the unresolved scales, so the friction is higher. The paper's scheme B is
dissipative at $O(\Delta t\,\Delta x^2)$ and matches the spectral value at this
resolution. That reading is a hypothesis until the $96^3$ and $128^3$ runs,
with about 4 and 8 times the cells, show $Re_\tau$ falling toward 306.

The run used one shared 14-core Mac (load average 40 to 100), float64 and one
XLA CPU process. It cost:
- 0.29 s per step at $\Delta t=0.025$, or 1 h 18 min of stepping for 16,000
  steps and 400 time units;
- 0.22 s per step at $\Delta t=0.0125$, which restarted from the $t=200$ state
  with 25 time units of burn-in.

The 128³ case has about 8 times the cells and twice the steps, about 20 hours
at the same speed, so it remains a GPU run.

The scheduled test gates $U_{cl}$ at 1.5 %, the regime and the agreement
between shear and drive. It carries the plan's 2 % $Re_\tau$ gate as a strict
expected failure, so the test reports when the number moves inside the gate.
