(thermal-convection)=
# Buoyant convection and open ducts

Warm metal rises under the Boussinesq force $\mathbf f_b=-\rho_0\beta\,(T-T_r)\,\mathbf g$ of the
[heat transport page](../thermal.md), and a magnetic field damps the motion. This page checks that
coupling against the onset of convection, its growth rates and a heated cavity, and then covers heat
carried through an open duct.

## Onset of convection

A fluid layer heated from below, between rigid plates held at fixed temperatures, becomes unstable
at $Ra_c=1707.76$, and a vertical field delays the onset (Chandrasekhar 1961, Ch. IV).
`python scripts/make_showcase_figures.py --only onset` measures the growth rate of the critical roll
just below and just above the critical Rayleigh number, at Chandrasekhar's critical wavenumber, for
$Q=Ha^2$ from 0 to 1000, at the time steps $\Delta t_0=0.05/(Q+Ra_c^{1/2})$ and $\Delta t_0/2$. Power
iteration on the time stepper gives the rate. The script scales the perturbation back to $10^{-6}$
every half time unit, and the rate settles to five digits by the third rescaling.

The reference is Chandrasekhar's eigenproblem solved by Chebyshev collocation
(`validation.thermal.chandrasekhar_onset`). It reproduces his tabulated values for $Q\le100$, as
quoted by Basak & Kumar (2015), to within 0.01 %. On the exact momentum stage, LMhdX's onset does not
depend on the time step, as $\Delta t_0$ and $\Delta t_0/2$ agree to 0.01 %. It lies 0.2 to 0.8 % below the reference across the
tenfold rise of $Ra_c$, the error of the 32-cell layer mesh. The tests require the Bénard onset at
$Q=0$ within 4 % of 1707.76.

```{image} ../../_static/thermal_magnetoconvection.webp
:alt: Critical Rayleigh number against the Chandrasekhar number, the critical rolls at Q = 0 and 1000, and the error at two time steps
:align: center
```

(thermal-growth-rates)=
## Growth rates

**Growth rates.** Beyond the onset, the rate at which a roll grows checks the buoyancy, the
transport and the Lorentz force together. `validation.thermal.convection_growth_rate` solves the
linear problem by Chebyshev collocation at any Prandtl number, in a field along any axis and over a
plane Poiseuille flow. Its checks are 1707.76, Chandrasekhar's 3757 at $Q=100$, and Orr and
Sommerfeld's $Re_c=5772.22$. Power iteration on the thermal step measures the rate for one
wavelength of the critical roll at $Pr=0.02$ (PbLi's) and $Ra=2\times10^4$:
- Without a field, the rate is 0.35 % low on 8 cells across, then +0.09 % and −0.16 % on 16 and 32.
- In a field of $Q=100$ across the roll's axis or vertical, it is within 0.17 % and 0.10 % on 32
  cells.
- In a field along the roll's axis, which drives no current, the roll grows at the field-free rate
  when the Lorentz force is explicit. Under the default implicit shift it grows too slowly, by the
  factor that [time stepping](../numerics/time.md) derives, 17 % at $\Delta t\,\sigma B^2/\rho=0.27$
  (16 cells, $Q=10^4$).
- Carried by a plane Poiseuille flow (Re 400 on the half-height, $Ra=10^5$), the rate is within
  +1.7 to +3.3 % with limited momentum and van Leer heat transport on 18 to 48 cells a wavelength.
- With Case 2's ingredients at its $\Delta t\,\sigma B^2/\rho=0.34$, in that flow and a field along
  the roll's axis (18 cells, $Q=8100$), the rate is 24.6 % slow under the implicit shift and +1.65 %
  with an explicit force. In a field across the axis ($Q=10^4$, $Ra=4\times10^5$) it is 6.0 % and
  2.5 % slow.
- Along the field between insulating Hartmann walls at Ha 235 ($Pr=0.0196$, $Ra=10^5$), against
  Sommeria and Moreau's model with the exact friction of [code verification](../../validation/verification.md),
  the rate is +3.5 % on 32 × 48 × 32 cells at $\Delta t\,\sigma B^2/\rho=0.34$ and +1.5 % on
  64 × 96 × 64. Under the implicit shift it is 11 % slow (`validation.verification.hartmann_roll_growth`).

## The heated cavity and a convecting layer

A differentially heated square cavity (de Vahl Davis 1983, $Pr=0.71$) checks the steady nonlinear
flow. The peak velocities and the hot wall's mean Nusselt number are within 0.1 % at $Ra=10^3$ on
64 cells, 0.4 % at $10^4$ on 64, 0.1 % at $10^5$ on 128 and 0.6 % at $10^6$ on 128, and they converge
at second order. `python scripts/make_showcase_figures.py --only buoyancy` reruns the growth rates
and the cavity and writes `docs/_static/data/thermal_buoyancy.json`.

At $Ra=10^4$ and $Pr=0.1$, a periodic $4\times4\times1$ layer on 64 × 64 × 32 cells convects in cells
that keep rearranging. A horizontal field of $Q=1000$ damps every motion that varies along it and
leaves steady rolls along the field, as in the NaK experiments of Burr & Müller (2002). The page does
not compare this transient run with a reference (`--only convection` redraws it).

```{image} ../../_static/thermal_convection_3d.webp
:alt: A liquid-metal layer heated from below, without a field and in a horizontal field
:align: center
```

(thermal-open-duct)=
## Heat on an open duct

An inflow-outflow duct carries its heat on a held or an evolving flow.
- **Held.** Pass the steady velocity of `axial.solve_open_duct` in the state and build the step with
  `hold_flow=True`. A held flow takes no buoyancy.
- **Evolving** (ADR 0009 D37). Build the step without `hold_flow=True`. The flow advances through the
  stage that `axial.solve_open_duct` marches, with its inlet profile carried as a lift and D26's zero
  axial gradient at the outlet. With buoyancy, the outlet holds the hydrostatic pressure of its last
  layer of cells, so a stratified outlet at rest stays at rest. LMhdX refuses gravity along the
  open axis. `lmhdx.solve` starts on the steady flow (`start="steady"`) or on the inlet profile. The
  step is first order in time. `tests/test_steady.py` measures the outlet's reach upstream, buoyant
  onset, returning flow, and a coupled manufactured solution, second order with van Leer transport.
- **Returning flow.** Flow returning through the outlet comes from outside the model, so its
  enthalpy is boundary data. `outlet_backflow="error"`, the default, refuses such a run.
  `validate_thermal_inputs` rejects any returning outlet face, even when the net flow leaves, and a
  step run without validation reports `backflow_boundary_valid` as false.
  `outlet_backflow="reservoir_flux"` brings in $C\,u_n T_\mathrm{res}$, the Danckwerts condition
  $C u_n T_b - k\,\partial_n T_b = C u_n T_\mathrm{res}$ (not $T_b = T_\mathrm{res}$), with
  `backflow_temperature` given to each step as a number or a profile. The outlet's conduction
  operator stays homogeneous Neumann. Recirculation inside the duct needs neither, because its
  upwind donor is another cell. The outlet reports `outlet_outflow_rate`, `outlet_backflow_rate`
  and `outlet_backflow_enthalpy_in`.

**A heated Hunt duct.** At Ha 20, with thin conducting Hartmann walls ($c = 0.05$), insulating side
walls, 8 half-widths long, a developed inlet at flow rate 4 and temperature 0, one side wall taking
in a unit flux and Joule heat on, the heat balance closes to $2.5\times10^{-14}$ of its largest term
at every step over 400 steps. At the steady state the enthalpy leaving, 124.6, is the Joule heat,
110.3, plus the net wall heat, 14.3. The axial cells give $Pe_h/2 = 5$, so the page quotes this duct
for its balance, not for a Nusselt number.

**The thermal entrance.** Near the inlet the local Nusselt number falls along the duct toward its
developed value. The test heats plates with a uniform flux downstream of an inlet at 0, in the held
developed Hartmann flow with van Leer transport, and compares the local Nusselt number with the
Graetz eigen-expansion (`validation.thermal.graetz`), which leaves out axial conduction. At $Pe=400$, on 64 cells across and 135 along, LMhdX stays within
1.3 % beyond $x^+=10^{-3}$ and 0.3 % beyond $10^{-2}$, at Ha 0, 10 and 50. Closer to the inlet the
difference is axial conduction. At $Pe=1600$ the difference at $x^+=10^{-3}$ drops from 1.3 % to
0.14 %, while refining the mesh leaves it unchanged.

```{image} ../../_static/thermal_entrance.webp
:alt: Temperature above the bulk in 3-D, local Nusselt number along heated Hartmann plates against the Graetz expansion, and its error
:align: center
```

**A side-layer jet.** A square duct at Ha 1000 and $Pe=2000$, 24 half-widths long, takes in a uniform
flux through one side wall, without Joule heat, on a held flow. With conducting Hartmann walls
($c=0.05$) the side layers carry jets at 8.4 times the mean velocity, and the wall's hottest point
rises 25 % less above the inlet temperature than in the insulating duct, 25.0 % on 32 × 64 and on
40 × 80 cells. This is a heat-transfer result in a given flow at that flow rate. It is not a
conjugate-wall result. It does not mean a lower pumping cost, because at the same flow rate
conducting walls raise the MHD pressure drop. The figure at the top of the
[heat transport page](../thermal.md) shows both ducts.

**Crediting a load.** `load_attribution(step, state, background, load, steps)` runs the held flow
with and without a load from the same start and reports what the load itself adds to the outlet
enthalpy, the conduction out and the storage. At constant properties the temperature is affine in
the heat source with upwind or no transport, so the difference is the load's own response. Van
Leer's face value is not affine, so `load_attribution` refuses a van Leer step, and with a buoyant
flow the response is not affine in the load. In a plug flow with $\lambda=\rho_0c_pu/k$ the credited
fraction on 32 cells is 0.788 at $\lambda=5$, against an exact 0.801. The gap is first-order upwind diffusion, at $Pe_h/2 = 0.08$.

The full open-duct text, with the plug-flow orders and the load's derivation, is
[on the page at main's commit](https://github.com/uwplasma/LMhdX/blob/b97ad3a8525a20502279961d5b4201d158d971e9/docs/physics/thermal.md#an-open-duct).
