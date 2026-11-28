# Q2D literature anchors

This page checks LMhdX's depth-averaged (Q2D) model against two published
experiments. The first is the onset of instability in a Kolmogorov-like flow
(Tithof et al. 2017), where LMhdX finds the critical Reynolds number within
1.2 %. The second is the energy spectrum of a driven vortex lattice (Sommeria
1986), where the spectrum's constant falls in the expected range only at the
higher friction Reynolds numbers. These are validation rows 13 and 15.

Each point of the neutral curve takes a few seconds. From the repository root,
this computes the row $q=0.465$ of the first table below:

```python
from validation.q2d_campaigns import tithof_formula, tithof_neutral

print(f"eq. (4.1) {float(tithof_formula(0.465)):.4f}, lmhdx.q2d {float(tithof_neutral(0.465)):.4f}")
```

```text
eq. (4.1) 7.6465, lmhdx.q2d 7.6940
```

Both campaigns run `lmhdx.q2d.evolve_q2d` from `validation/q2d_campaigns.py`
(`python -m validation.q2d_campaigns tithof` or `sommeria`), on the latest stack
(Python 3.14, JAX 0.11.2, SOLVAX 0.28.2, float64, CPU).

## Row 15: the critical Reynolds number of the Kolmogorov-like flow

Tithof, Suri, Pallantla, Grigoriev and Schatz (J. Fluid Mech. 828, 837, 2017,
Section 4.2) linearise the flow $u_s\propto\sin\kappa y$ of
$\partial_t u+\beta u\cdot\nabla u=-\nabla p+\nu\nabla^2u-\alpha u+f$ and give
the neutral curve $Re_n(q)$ of perturbations of wavenumber $q\kappa$ (their
eq. 4.1, with three Fourier modes on an unbounded domain). For $\beta=1$ the minimum
is $Re_c=7.60$ at $q_c=0.465$. The experiment has
$\alpha=0.064\,\mathrm{s^{-1}}$, $\nu=3.26\times10^{-6}\,\mathrm{m^2/s}$ and
magnet width $w=1.27$ cm, so $\kappa=\pi/w$ and $\alpha/(\nu\kappa^2)=0.3208$.
The Reynolds number is $Re=Uw/\nu$, with $U$ the rms speed.

`lmhdx.q2d` solves the $\beta=1$ model (other $\beta$ rescale velocity and
forcing). The campaign:
- takes a box of one forcing period in $y$ and $2\pi/(q\kappa)$ in $x$;
- starts from the exact steady flow plus a $10^{-8}$ perturbation in the modes
  $e^{i\kappa(ny+qx)}$, $n=-1,0,1$;
- reads the growth rate from the perturbation norm between $t=40$ and 120 in
  units where $\nu=\kappa=1$, on 16² cells with $dt=0.02$;
- finds the neutral $Re$ as the zero of the growth rate by the secant rule, to
  $2\times10^{-4}$.

| q | Eq. (4.1) Re_n | lmhdx.q2d Re_n | difference |
|---|---|---|---|
| 0.400 | 7.7880 | 7.8182 | +0.39 % |
| 0.440 | 7.6672 | 7.7071 | +0.52 % |
| 0.465 | 7.6465 | 7.6940 | +0.62 % |
| 0.500 | 7.6810 | 7.7422 | +0.80 % |
| 0.540 | 7.8063 | 7.8883 | +1.05 % |

A parabola through the measured curve gives $Re_c=7.694$, $+1.23$ % from the
published 7.60, at $q_c=0.462$, $-0.64$ % from 0.465. **Row 15 is met, within
its 2 %.**
- The simulated curve lies 0.4–1.1 % above eq. (4.1), as expected of a
  three-mode truncation against the full periodic problem.
- The published 7.60 is 0.6 % below eq. (4.1) at this $\alpha/(\nu\kappa^2)$,
  7.646, from the rounding of the printed parameters.
- `tests/test_q2d_identities.py` gates the neutral Reynolds number at $q_c$
  within 2 % of 7.60 and 1 % of eq. (4.1), in about 7 s.

## Row 13: the electrically driven vortex lattice

Sommeria (J. Fluid Mech. 170, 139, 1986) drives a lattice of vortices in
mercury and measures the inverse-cascade spectrum
$E(k)=C\,\varepsilon^{2/3}k^{-5/3}$. The plan's range for $C$, 3–7, comes from
plan Section 3.3 and was not re-read from the paper.

The campaign applies the steady forcing $\omega_f=F\sin(6x)\sin(6y)$ on
$[0,2\pi]^2$, six periods per side, with friction $\alpha=1$, viscosity
$2\times10^{-3}$ and 128² cells. The time step is $dt=2\times10^{-3}$, halved
at the largest $F$ for the Courant number. Each run starts from a random
vorticity of amplitude $10^{-2}$, runs to $t=150$ and averages over the last 90.
- The flux $\Pi(k)$ is the cumulative spectral transfer of the advection term,
  so $\varepsilon$ in the law is the local inverse flux $|\Pi(k)|$ and does not
  assume what friction removes.
- $C$ is the mean of $E(k)k^{5/3}/|\Pi(k)|^{2/3}$ over the shells $k=3$–6,
  below the forcing shell $|k|=6\sqrt2\approx8.5$.
- The friction Reynolds number is $Re_\alpha=u_\mathrm{rms}k_f/\alpha$ with
  $k_f=6$.

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

**Result.** C is inside 3–7 for $Re_\alpha\ge28$, at 4.2–6.3, and below it at
$Re_\alpha\le14$, at 2.8–2.9 and 1.3 at 6.7. **So row 13 is met only for the
upper part of its range $Re_\alpha$ 1–30 on this definition.**
- The two seeds agree to 0.1 at $Re_\alpha\le14$ and to 0.7 at the largest.
- $C$ rises with $Re_\alpha$ as the inertial range widens. The box has only the
  shells $k=1$–5 below the forcing, so the plateau is short and the value
  depends on the band, 3–6 here.
- **The runs do not reach condensation.** The fundamental holds 0.1–5.7 % of
  the energy, rising with $Re_\alpha$, and a condensate would hold most of it.
  Larger $Re_\alpha$ and longer runs are needed.
- This campaign did not confirm Sommeria's own definition of $Re_\alpha$ or his
  forcing geometry, so the mapping of the row's "Re_α 1–30" to these runs is an
  assumption.
- Each run averages one window of 90 time units, 600 to 6,000 turnovers of the
  forcing scale.
