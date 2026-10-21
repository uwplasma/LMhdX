# Q2D literature anchors (validation rows 13 and 15)

Both campaigns run `lmhdx.q2d.evolve_q2d` from `validation/q2d_campaigns.py`
(`python -m validation.q2d_campaigns tithof` or `sommeria`), on the latest stack
(Python 3.14, JAX 0.11.2, SOLVAX 0.28.2, float64, CPU).

## Row 15: the critical Reynolds number of the Kolmogorov-like flow

Tithof, Suri, Pallantla, Grigoriev and Schatz (J. Fluid Mech. 828, 837, 2017,
Section 4.2) linearise the flow $u_s\propto\sin\kappa y$ of
$\partial_t u+\beta u\cdot\nabla u=-\nabla p+\nu\nabla^2u-\alpha u+f$ and give the neutral
curve $Re_n(q)$ of perturbations of wavenumber $q\kappa$ (their eq. 4.1: three
Fourier modes, unbounded domain). For $\beta=1$ the minimum is $Re_c=7.60$ at
$q_c=0.465$, with the experiment's $\alpha=0.064\,\mathrm{s^{-1}}$,
$\nu=3.26\times10^{-6}\,\mathrm{m^2/s}$ and magnet width $w=1.27$ cm, so
$\kappa=\pi/w$ and $\alpha/(\nu\kappa^2)=0.3208$; $Re=Uw/\nu$ with $U$ the rms speed.
`lmhdx.q2d` solves the $\beta=1$ model (velocity and forcing rescale by $\beta$ in
the others), so the campaign takes a box of one forcing period in $y$ and $2\pi/(q\kappa)$ in $x$,
the exact steady flow plus a $10^{-8}$ perturbation in the modes
$e^{i\kappa(ny+qx)}$, $n=-1,0,1$, and reads the growth rate from the perturbation
norm between $t=40$ and 120 (units $\nu=\kappa=1$, 16² cells, $dt=0.02$). The neutral $Re$
is the zero of the growth rate by the secant rule, to $2\times10^{-4}$.

| q | Eq. (4.1) Re_n | lmhdx.q2d Re_n | difference |
|---|---|---|---|
| 0.400 | 7.7880 | 7.8182 | +0.39 % |
| 0.440 | 7.6672 | 7.7071 | +0.52 % |
| 0.465 | 7.6465 | 7.6940 | +0.62 % |
| 0.500 | 7.6810 | 7.7422 | +0.80 % |
| 0.540 | 7.8063 | 7.8883 | +1.05 % |

A parabola through the measured curve gives $Re_c=7.694$ ($+1.23$ % from the
published 7.60) at $q_c=0.462$ ($-0.64$ % from 0.465): **met, within the 2 % of
row 15**. The simulated curve lies 0.4–1.1 % above eq. (4.1), as expected of a
three-mode truncation compared with the full periodic problem. The published
7.60 is also 0.6 % below eq. (4.1) evaluated at this $\alpha/(\nu\kappa^2)$ (7.646), the
rounding of the printed parameters. `tests/test_q2d_identities.py` gates the neutral
Reynolds number at $q_c$ within 2 % of 7.60 and 1 % of eq. (4.1) (about 7 s).

## Row 13: the electrically driven vortex lattice

Sommeria (J. Fluid Mech. 170, 139, 1986) drives a lattice of vortices in mercury
and measures the inverse-cascade spectrum $E(k)=C\,\varepsilon^{2/3}k^{-5/3}$;
the plan's range for $C$ is 3–7 (taken from plan Section 3.3, not re-read from the
paper). The campaign forces $\omega_f=F\sin(6x)\sin(6y)$ on $[0,2\pi]^2$ (six
periods per side, steady), with friction $\alpha=1$, viscosity $2\times10^{-3}$,
128², $dt=2\times10^{-3}$ (halved at the largest $F$ to hold the Courant number), from a random
vorticity of amplitude $10^{-2}$, run to $t=150$ and averaged over the last 90.
The flux $\Pi(k)$ is the cumulative spectral transfer of the advection term, so
$\varepsilon$ in the law is the local inverse flux $|\Pi(k)|$ and does not assume what
friction removes. $C$ is the mean of $E(k)k^{5/3}/|\Pi(k)|^{2/3}$ over the shells
$k=3$–6 (below the forcing shell $|k|=6\sqrt2\approx8.5$), and the friction Reynolds
number is $Re_\alpha=u_\mathrm{rms}k_f/\alpha$ with $k_f=6$.

| F | seed | Re_α | E(1)/E | ε injected | \|Π\| at k = 4 | C |
|---|---|---|---|---|---|---|
| 30 | 1 | 6.7 | 0.002 | 1.42 | 0.11 | 1.31 |
| 30 | 2 | 6.7 | 0.001 | 1.41 | 0.11 | 1.30 |
| 100 | 1 | 14.0 | 0.007 | 6.14 | 1.08 | 2.80 |
| 100 | 2 | 14.0 | 0.007 | 6.14 | 1.02 | 2.91 |
| 300 | 1 | 27.7 | 0.016 | 24.50 | 7.53 | 4.18 |
| 300 | 2 | 27.9 | 0.017 | 23.51 | 6.64 | 4.58 |
| 600 | 1 | 42.7 | 0.034 | 57.56 | 19.57 | 5.37 |
| 600 | 2 | 42.6 | 0.023 | 56.44 | 20.91 | 5.13 |
| 1200 | 1 | 66.4 | 0.055 | 128.90 | 54.52 | 6.33 |
| 1200 | 2 | 65.7 | 0.057 | 123.48 | 64.13 | 5.79 |

**Result: C is inside 3–7 for $Re_\alpha\ge28$ (4.2–6.3) and below it at
$Re_\alpha\le14$ (2.8–2.9, and 1.3 at 6.7), so row 13 is met only for the upper part
of its range $Re_\alpha$ 1–30 on this definition.** The two seeds agree to 0.1 at
$Re_\alpha\le14$ and to 0.7 at the largest. $C$ rises with $Re_\alpha$ as the inertial
range widens; the box has only the shells $k=1$–5 below the forcing, so the plateau is
short and the value depends on the band (3–6 here), which is a limit of this
campaign. **Condensation is not reached**: the fundamental holds 0.1–5.7 % of the
energy, rising with $Re_\alpha$; a condensate would hold most of it. Larger $Re_\alpha$
and longer runs are needed, and Sommeria's own definition of $Re_\alpha$ and his forcing
geometry were not confirmed, so the mapping of the row's "Re_α 1–30" to these
runs is an assumption. One statistical run per $F$ and seed is a single window of
90 time units (600 to 6,000 turnovers of the forcing scale).
