# The duct design law

A duct that carries a fixed flow across a strong magnetic field is cheapest to pump when it is deep along the field
and narrow across it. At a fixed cross-section area one aspect ratio costs least, and a one-line law gives it. This
page finds that optimum with LMhdX, sizes one blanket duct with it, and screens shapes for a heated duct.

**The law.** Let $a$ be the half-width along the field, $b = \beta a$ the half-width across it, and
$Ha = B a\sqrt{\sigma/\mu}$. At a fixed flow rate and a fixed cross-section area, the insulated duct of least pumping
power obeys

$$
s^* = \beta^*\sqrt{Ha^*} \to 2.08 ,
$$

from $s^* = 1.975$ at $Ha^* = 14$ to $2.078$ at $Ha^* = 6{,}109$ on the spectral reference. Shercliff's thin-layer
friction law, $\Delta p/L = \mu V Ha / [a^2 (1 - \alpha/(\beta\sqrt{Ha}) - 1/Ha)]$, explains it. The balance falls where
the side layers remove two fifths of the flow, $s^* = 2.5\alpha$, which gives 2.13 with Smolentsev's (2021)
$\alpha = 0.852$ and 2.06 with the $\alpha = 0.825$ of Müller & Bühler. The optimum's pumping power is 9 % below the
equal-area square duct's at $Ha^* = 14$, 61 % below at 1,229 and 73 % below at 6,109. Nishio et al. (2025) computed
the same interior minimum, and their numbers imply $s^* \approx 2.08$. LMhdX adds the finite-$Ha$ values, a check
against an independent solver and three-dimensional solves, and the tilted field.

## Run it

The pressure drop at each shape comes from {ref}`drive elimination <numerics-drive-elimination>`.

```bash
python examples/duct_design_law.py  # about 20 s on a CPU
```

```text
Design law over H = (10.0, 30.0, 100.0, 300.0), 48^2 cells, five solves per H...
H    10: beta* 0.53471, Ha*    13.7, s* 1.9774, 8.9% below the square
H    30: beta* 0.27147, Ha*    57.6, s* 2.0599, 26.3% below the square
H   100: beta* 0.12337, Ha*   284.7, s* 2.0816, 46.1% below the square
H   300: beta* 0.05958, Ha*  1229.0, s* 2.0888, 61.1% below the square
Blanket duct: 6 stations at 0.465-0.497 T, area 2.40 cm^2
design: beta* 0.14851, a 20.10 mm, b 2.99 mm, W* 4.1495e-05 W, 41.7% below the square
Wrote artifacts/examples/duct_design_law/duct_design_law_summary.json and artifacts/examples/duct_design_law/duct_design_law.png
```

```{image} ../_static/duct_design_law.webp
:alt: The optimum s* against H, approaching the thin-layer law
:align: center
```

*The optimum's $s^* = \beta^*\sqrt{Ha^*}$ against $H$, approaching the thin-layer law's 2.13. Drawn by `python examples/duct_design_law.py`.*

The example finds $\beta^*$ for four values of $H$, the Hartmann number of the equal-area square duct. It then sizes
one PbLi duct (573 K) that runs 0.982 m radially through an outboard blanket in a $1/R$ field of 0.5 T at the first
wall, at $Q = 2.4\times10^{-6}$ m³/s. There the smallest allowed mean velocity, 10 mm/s, sets the area. `jax.grad` of
the solve in its field scale confirms the sign of $d\Delta p/d\ln A$, so the aspect ratio is the one choice left.

The steps below find the optimum for one $H$, 30, as the example does. Work in double precision, because the optimum
is a shallow minimum that float32 solves would blur.

```python
from dataclasses import replace

import numpy as np

from lmhdx import enable_x64
from lmhdx.core3d import ChannelProblem
from lmhdx.grid import PERIODIC, BoundaryCondition, Grid, uniform_faces, wall_resolving_faces
from lmhdx.steady import channel_flow_rate, solve_compiled

enable_x64()
CELLS, CELLS_IN_LAYER = 48, 6
```

`unit_duct` builds a unit insulated duct of aspect `beta` on a mesh made for a central aspect and scaled across the
field. Every trial then shares one compiled program, and the pressure drop is smooth in $\beta$.

```python
def unit_duct(beta, ha_mesh, centre):
    y = wall_resolving_faces(
        CELLS, -1.0, 1.0, layer_thickness=1.0 / ha_mesh, cells_in_layer=CELLS_IN_LAYER, max_ratio=None
    )
    side = min(1.0 / np.sqrt(ha_mesh), 0.25 * centre)
    z = wall_resolving_faces(
        CELLS, -centre, centre, layer_thickness=side, cells_in_layer=CELLS_IN_LAYER, max_ratio=None
    )
    wall = BoundaryCondition("neumann")
    return ChannelProblem(
        grid=Grid(uniform_faces(1, 0.0, 1.0), y, z * (beta / centre)),
        conditions=(BoundaryCondition(PERIODIC), wall, wall),
        conductivity=1.0,
        magnetic_field=(0.0, ha_mesh, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
    )
```

`pressure_drop` returns the drop of aspect `beta` at a unit mean velocity through area 4, so the square duct has
$a = 1$ and $Ha = H$. One solve gives the flow per unit drive $q$, and $\Delta p = \mu Q L / (a^4 q)$.

```python
def pressure_drop(beta, H, centre):
    a = np.sqrt(1.0 / beta)  # area 4 = 4 a b
    ha_mesh = 1.15 * H * np.sqrt(1.0 / centre)  # the mesh's Ha, above every trial's
    station = replace(unit_duct(beta, ha_mesh, centre), magnetic_field=(0.0, H * a, 0.0))
    q = float(channel_flow_rate(station, solve_compiled(station).velocity[0].data[0]))
    return 4.0 / (a**4 * q)
```

Start from the law's guess, solve five trials 0.1 apart in $\ln\beta$, and take the minimum of the quartic through them.

```python
H = 30.0
centre = (2.08 / np.sqrt(H)) ** (4.0 / 3.0)  # beta from s* = 2.08 with Ha = H / sqrt(beta)
offsets = 0.1 * np.arange(-2, 3)
quartic = np.polyfit(offsets, [pressure_drop(centre * np.exp(x), H, centre) for x in offsets], 4)
x = min(
    (root.real for root in np.roots(np.polyder(quartic))
     if abs(root.imag) < 1e-12 and abs(root.real) <= 0.2 and np.polyval(np.polyder(quartic, 2), root.real) > 0),
    key=abs,
)
beta = centre * np.exp(x)
```

Compare it with the square duct of the same area.

```python
ha = H / np.sqrt(beta)
saving = 1.0 - pressure_drop(beta, H, centre) / pressure_drop(1.0, H, 1.0)
print(f"beta* {beta:.5f}, Ha* {ha:.1f}, s* {beta * np.sqrt(ha):.4f}, {100 * saving:.1f}% below the square")
```

```text
beta* 0.27147, Ha* 57.6, s* 2.0599, 26.3% below the square
```

This is the example's line for $H = 30$. The example repeats it for four values of $H$, then sums the pressure drop
over six stations of the $1/R$ field for the blanket duct.

## What is established

On 48/6 cells the example reproduces the study's optima to the digits printed. The study's own checks are these:
- $\beta^*$ has a grid convergence index of 0.18–0.42 % across 32/4, 48/6 and 72/9 cells at $H$ = 30, 100 and 300.
- The core's $s^*$ is 0.1–0.8 % above the spectral reference (`validation.shercliff`, with its `aspect` keyword).
- `validation.open_axis.ramp_excess` compares the locally fully developed station sum with a 3-D open duct in a
  monotone 20 % ramp. For a square duct the sum is good to 1 % only up to $\gamma\sqrt{Ha} \approx 0.2$, where $\gamma$
  is the field's relative change per half-width. The design-law duct stays under 0.35 % up to $\gamma\sqrt{Ha} = 2$.
  The excess does not collapse on $\gamma\sqrt{Ha}$ across $Ha$ 50 and 200, so these limits hold per $Ha$.

One point of that comparison runs in about a minute on a CPU.

```python
from validation.open_axis import ramp_excess

enable_x64()
row = ramp_excess(0.2, 1.0, 200.0, cells=48, cells_in_layer=6)  # gamma sqrt(Ha), beta, Ha
print(row["excess_percent"])  # 0.448
```

## A tilted field

A poloidal field tilts $B$ within the cross-section, and the slender optimum is sensitive to that tilt where the
square duct is not. When the study re-optimizes the duct in a field tilted by $B_p/B_T$, the optimum and its penalty
collapse on $\kappa = \sqrt{Ha^*_0}\,B_p/B_T$ to within 5 %:

$$
\beta^{*2} = \beta_0^{*2} + (B_p/B_T)^2 ,\qquad
\left(\Delta p^*/\Delta p^*_0\right)^2 = 1 + \frac{0.122\,\kappa^2}{1 + \kappa/3.01} .
$$

A penalty of at most 10 % needs $\kappa < 1.7$. A duct kept at its aligned shape loses to the square beyond
$\kappa \approx 8$–12 at $Ha^*_0$ 1,232–6,127. The study fitted both laws to the same 20 points ($Ha^*_0$ 58–6,127,
$B_p/B_T \le 0.3$, one uniform tilt per duct), so their errors are in-sample.

## A thermal design

The developed peaks come from {ref}`fully developed heat transfer <thermal-developed-h1h2>`, and the open-duct check
from {ref}`heat on an open duct <thermal-open-duct>`.

The design above takes its area from a smallest allowed velocity, which stands in for heat removal. A thermal design
asks which shape and flow remove the heat within a temperature limit at the least pumping power, either at a fixed
duty per channel or at a fixed flux on the first wall. The two favour different shapes, so this page screens both at
one fixed area. The study's
[Stage 3](https://github.com/uwplasma/LMhdX-Duct-Optimization/blob/stage3-v1/results/stage3-thermal/README.md)
also optimizes the area (the record below).

The specification is illustrative, not an engineering requirement:

| | |
|:--|:--|
| Fluid | Pb-17Li at 573 K, constant properties from Martelli et al. (2019), Table 1: $\rho$ 9838 kg/m³, $c_p$ 189.8 J/(kg K), $k$ 20.4 W/(m K), $\mu$ 2.153 mPa s, $\sigma$ 8.835×10⁵ S/m, $Pr = 0.020$ |
| Duct | straight, area 2.4×10⁻⁴ m², length 0.982 m, insulating walls, a uniform 0.5 T along $a$ |
| Heat | through the first wall $z=-b$, of width $2a$ parallel to the field: 1.5 kW per channel, or the square duct's flux, 98.6 kW/m² |
| Inlet | 573 K |
| Limits | fluid and wall-interface peaks of at most 625 K; $Re/Ha \le 200$ ($Re$ on $D_h$, $Ha$ on $a$); at least 95 % of the duty leaving at the outlet, checked on the candidate in an open duct |
| Cost | the hydraulic power $Q\,\Delta p$, with no pump efficiency |

The correlations cover 508–800 K for $c_p$, 508–625 K for $\mu$ and 600–800 K for $\sigma$, so $\sigma$ at 573 K is
an extrapolation 27 K below its range. The 625 K limit is the top of the viscosity's range. Buoyancy is left out, and
the flow is held fixed. The workflow reports $N_a = Ha_a^2/Re_a$, $Ri_a = Gr_a/Re_a^2$ and
$Ri_a/N_a = \rho g\beta\Delta T/(\sigma U B^2)$ as scales, with the temperature scale $q''a/k$,
$\beta = 1.21\times10^{-4}$ /K and $|g| = 9.81$ m/s². They are not a verdict.

Each trial solves the developed flow of its shape under a unit pressure gradient and scales it to the trial's flow,
which is exact because the Stokes flow is linear in its drive. The workflow recomputes the Joule heat from the scaled
velocity, solves the temperature with `developed_heat` and takes its peaks from the 573 K inlet with
`DevelopedHeat.anchored_peaks`. A rejected trial records why. The search sweeps 7 × 9 points in $\ln(b/a)$ and
$\ln Q$, bisects each observed lower edge of the feasible flows, and refines the cheapest shape by golden section. It
assumes one crossing between bracketing samples and one minimum between the cheapest shape's neighbours. The shape is
a host design variable, so this page claims no derivative through the geometry.

The whole workflow is one script, which needs none of the earlier blocks. It takes about 45 s on a CPU.

````{dropdown} The workflow script
```python
import jax.numpy as jnp
import numpy as np

import lmhdx
from lmhdx.core3d import ChannelProblem, zero_velocity
from lmhdx.steady import channel_flow_rate
from lmhdx.grid import CENTER, DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition, Field, Grid
from lmhdx.grid import uniform_faces, wall_resolving_faces
from lmhdx.steady import solve_compiled
from lmhdx.thermal import ThermalProblem, build_thermal_step, developed_heat, initial_thermal_state
from lmhdx.thermal import load_attribution, thermal_boundary, validate_developed_inputs, validate_thermal_inputs

lmhdx.enable_x64()

# Pb-17Li at 573 K, held constant: Martelli et al. (2019), Table 1.
RHO, CP, K, MU, SIGMA = 9838.0, 189.8, 20.40, 2.153e-3, 8.835e5  # kg/m^3, J/(kg K), W/(m K), Pa s, S/m
BETA, GRAVITY = 1.21e-4, 9.81  # 1/K, the same table; m/s^2, in any direction: the flow is held, so g only flags
# An illustrative specification, not an engineering one.
AREA, LENGTH, FIELD = 2.4e-4, 0.982, 0.5  # m^2 and m, fixed; T, uniform, along the half-width a
DUTY = 1500.0  # W, through the first wall z = -b, parallel to the field
T_IN, T_MAX = 573.0, 625.0  # K; 625 K is the top of the viscosity correlation's range
LAMINAR, DELIVERED = 200.0, 0.95  # the largest Re/Ha (Re on D_h, Ha on a); the least share of the duty delivered
ASPECTS, FLOWS = np.geomspace(0.08, 1.0, 7), np.geomspace(4e-6, 6e-5, 9)  # b/a, and m^3/s
WRAP, WALL = BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN)
INSULATED = thermal_boundary("heat_flux", conductivity=K)


def duct(aspect, cells=32, layer=4):
    """The duct of aspect b/a at the fixed area, and its flow under a pressure gradient of 1 Pa/m."""
    a = np.sqrt(AREA / (4.0 * aspect))
    hartmann = FIELD * a * np.sqrt(SIGMA / MU)
    y, z = (
        wall_resolving_faces(cells, -1.0, 1.0, layer_thickness=t, cells_in_layer=layer, max_ratio=None)
        for t in (1.0 / hartmann, min(1.0 / (aspect * np.sqrt(hartmann)), 0.25))
    )
    flow = ChannelProblem(
        Grid(uniform_faces(1, 0.0, LENGTH), a * y, aspect * a * z),
        (WRAP, WALL, WALL),
        density=RHO,
        viscosity=MU / RHO,
        conductivity=SIGMA,
        magnetic_field=(0.0, FIELD, 0.0),
        forcing=(1.0, 0.0, 0.0),
    )
    unit = solve_compiled(flow).velocity
    return dict(aspect=aspect, a=a, hartmann=hartmann, flow=flow, unit=unit,
                rate=float(channel_flow_rate(flow, unit[0].data[0])))


def trial(duct, flow_rate, duty=DUTY):
    """A design's developed temperatures, anchored at the inlet, its cost, and every reason to reject it."""
    flow, drive = duct["flow"], flow_rate / duct["rate"]
    velocity = tuple(component.replace_data(drive * component.data) for component in duct["unit"])
    first_wall = thermal_boundary("heat_flux", -duty / (2.0 * duct["a"] * LENGTH), 0.0, conductivity=K)
    thermal = ThermalProblem(flow, (WRAP, INSULATED, first_wall), CP, K, T_IN)
    none = Field(jnp.zeros(flow.grid.shape), (CENTER,) * 3, flow.grid)
    record = dict(aspect=duct["aspect"], flow_rate=flow_rate, duty=duty, velocity=velocity,
                  pressure_drop=drive * LENGTH, hydraulic_power=flow_rate * drive * LENGTH)
    try:
        validate_developed_inputs(
            thermal, velocity, none, flow_rate_tolerance=1e-12, velocity_tolerance=1e-9 * flow_rate / AREA
        )
    except ValueError as error:
        return {**record, "rejected": [f"invalid: {error}"]}
    result = developed_heat(thermal, velocity, none)
    record["wall_peak"], record["fluid_peak"] = (float(t) for t in result.anchored_peaks(T_IN, LENGTH))
    record["joule"] = RHO * CP * float(result.axial_gradient) * flow_rate * LENGTH - duty
    record["reynolds"] = RHO * flow_rate / AREA * result.hydraulic_diameter / MU
    record["re_over_ha"] = record["reynolds"] / duct["hartmann"]
    rejected = [
        f"{name} {record[name]:.1f} K over {T_MAX:.0f} K"
        for name in ("wall_peak", "fluid_peak")
        if not record[name] <= T_MAX
    ]
    if not record["re_over_ha"] <= LAMINAR:
        rejected.append(f"Re/Ha {record['re_over_ha']:.0f} over {LAMINAR:.0f}")
    return {**record, "rejected": rejected}


def lowest_flow(duct, below, above, duty=DUTY, tolerance=1e-3):
    """Bisect ln Q between an infeasible and a feasible flow, which must bracket one crossing of the boundary."""
    if trial(duct, above, duty)["rejected"] or not trial(duct, below, duty)["rejected"]:
        raise RuntimeError(
            f"{below:.3g} and {above:.3g} m^3/s do not bracket b/a = {duct['aspect']:.3g}'s boundary"
        )
    while above / below - 1.0 > tolerance:
        middle = np.sqrt(below * above)
        below, above = (below, middle) if not trial(duct, middle, duty)["rejected"] else (middle, above)
    return trial(duct, above, duty)


# Sweep the bounded design set; at each aspect, refine each observed lower edge of the feasible flows.
ducts = [duct(aspect) for aspect in ASPECTS]
sweep = [[trial(each, flow_rate) for flow_rate in FLOWS] for each in ducts]
edges = []
for each, row in zip(ducts, sweep):
    feasible = [not record["rejected"] for record in row]
    print(f"b/a {each['aspect']:.3f}: " + " ".join("o" if ok else "x" for ok in feasible))
    if feasible[0]:  # the lowest sampled flow already passes: the flow bound, not a limit, is active
        edges.append(row[0])
    edges += [
        lowest_flow(each, FLOWS[j - 1], FLOWS[j])
        for j in range(1, len(FLOWS))
        if feasible[j] > feasible[j - 1]
    ]
print("lowest flows: " + ", ".join(f"b/a {r['aspect']:.3f} {r['flow_rate']:.4e}" for r in edges))

# Refine the cheapest edge's aspect by golden section in ln(b/a) between its sampled neighbours.
best = min(edges, key=lambda record: record["hydraulic_power"])
index = int(np.searchsorted(ASPECTS, best["aspect"]))
low, high = np.log(ASPECTS[max(index - 1, 0)]), np.log(ASPECTS[min(index + 1, len(ASPECTS) - 1)])
bracket = (
    0.8 * best["flow_rate"],
    1.25 * max(r["flow_rate"] for r in edges if low <= np.log(r["aspect"]) <= high),
)
evaluated = {}


def cheapest(x):
    if x not in evaluated:
        evaluated[x] = lowest_flow(duct(float(np.exp(x))), *bracket)
    return evaluated[x]["hydraulic_power"]


ratio = (np.sqrt(5.0) - 1.0) / 2.0
left, right = high - ratio * (high - low), low + ratio * (high - low)
while high - low > 0.02:  # in ln(b/a); this assumes one minimum between the neighbours
    if cheapest(left) < cheapest(right):
        high, right = right, left
        left = high - ratio * (high - low)
    else:
        low, left = left, right
        right = low + ratio * (high - low)
candidate = min([*evaluated.values(), *edges], key=lambda record: record["hydraulic_power"])
square = next(record for record in edges if record["aspect"] == 1.0)

saving = 1.0 - candidate["hydraulic_power"] / square["hydraulic_power"]
for name, record in (("square", square), ("candidate", candidate)):
    a = np.sqrt(AREA / (4.0 * record["aspect"]))
    nu, speed, flux = MU / RHO, record["flow_rate"] / AREA, record["duty"] / (2.0 * a * LENGTH)
    # On the half-width a throughout: N_a = Ha_a^2 / Re_a = sigma B^2 a / (rho U), the Lorentz force over inertia.
    # Buoyancy on the flux's temperature scale q'' a / k: Ri_a = Gr_a / Re_a^2 against inertia, and
    # Ri_a / N_a = Gr_a / (Re_a Ha_a^2) = rho g beta dT / (sigma U B^2), the velocity buoyancy drives against the
    # Lorentz force over the forced one. Scales of a forced-convection screening, not a verdict on a design.
    reynolds, hartmann = speed * a / nu, FIELD * a * np.sqrt(SIGMA / MU)
    grashof = GRAVITY * BETA * (flux * a / K) * a**3 / nu**2
    record["numbers"] = dict(Re_a=reynolds, Ha_a=hartmann, N_a=hartmann**2 / reynolds, Ri_a=grashof / reynolds**2)
    record["numbers"]["Ri_a/N_a"] = record["numbers"]["Ri_a"] / record["numbers"]["N_a"]
    viscous = record["hydraulic_power"] - record["joule"]  # a developed Stokes flow's pumping is its dissipation
    print(
        f"{name}: b/a {record['aspect']:.4f}, Q {record['flow_rate']:.4e} m^3/s, dp {record['pressure_drop']:.1f} Pa, "
        f"W {record['hydraulic_power']:.3e} W, wall {record['wall_peak']:.2f} K, flux {flux / 1e3:.1f} kW/m^2, "
        f"Re_Dh/Ha_a {record['re_over_ha']:.0f}, Pe_Dh {record['reynolds'] * MU * CP / K:.0f}, "
        f"Rm {4e-7 * np.pi * SIGMA * speed * a:.1e}, viscous/duty {viscous / record['duty']:.1e}, "
        f"Joule/duty {record['joule'] / record['duty']:.1e}"
    )
    print("  " + ", ".join(f"{key} {value:.4g}" for key, value in record["numbers"].items()) + f", Gr_a {grashof:.3g}")
print(f"at the same duty, the candidate needs {100 * saving:.1f} % less pumping power than the square duct")

# Check the candidate: a finer mesh, then its flow carried down an open duct from an inlet at T_IN.
refined = lowest_flow(
    duct(candidate["aspect"], cells=48, layer=6), 0.9 * candidate["flow_rate"], 1.1 * candidate["flow_rate"]
)
print(
    f"48/6 cells: Q {100 * (refined['flow_rate'] / candidate['flow_rate'] - 1):+.2f} %, "
    f"W {100 * (refined['hydraulic_power'] / candidate['hydraulic_power'] - 1):+.2f} %"
)


def open_duct(record, refine=1):
    """Carry a design's developed flow down 12 axial cells from an inlet at T_IN, and credit its duty by difference.

    The duty is the cell source q A / V along the first wall: a face flux enters its cell exactly as that source.
    """
    profile, section = record["velocity"][0].data[0], record["velocity"][0].grid
    grid = Grid(uniform_faces(12, 0.0, LENGTH), section.y_faces, section.z_faces)
    dt = 0.5 * (LENGTH / 12) / float(jnp.max(profile)) / refine  # a Courant number of 0.5 / refine
    flow = ChannelProblem(
        grid,
        (BoundaryCondition(DIRICHLET, lower=np.asarray(profile), upper_kind=NEUMANN), WALL, WALL),
        density=RHO,
        viscosity=MU / RHO,
        conductivity=SIGMA,
        magnetic_field=(0.0, FIELD, 0.0),
        dt=dt,
    )
    inlet = thermal_boundary("temperature", T_IN, upper_kind="heat_flux", conductivity=K)
    thermal = ThermalProblem(flow, (inlet, INSULATED, INSULATED), CP, K, T_IN)
    rest = zero_velocity(flow)
    held = (rest[0].replace_data(jnp.broadcast_to(profile, rest[0].shape)), *rest[1:])
    start, nothing = initial_thermal_state(thermal, T_IN, held), Field(jnp.zeros(grid.shape), (CENTER,) * 3, grid)
    source = np.zeros(grid.shape)
    source[:, :, 0] = record["duty"] / np.sum(grid.cell_volumes()[:, :, 0])
    load = Field(jnp.asarray(source), (CENTER,) * 3, grid)
    validate_thermal_inputs(thermal, start, load, divergence_tolerance=1e-12)
    steps = int(np.ceil(6.0 * AREA * LENGTH / record["flow_rate"] / dt))  # six residence times
    step = build_thermal_step(thermal, dtype=jnp.float64, hold_flow=True)
    return load_attribution(step, start, nothing, load, steps)


credits = [open_duct(candidate, refine) for refine in (1, 2)]
delivered = [float(credit.delivered / credit.imposed) for credit in credits]
peaks = [float(credit.loaded.temperature_max) for credit in credits]
axial_peclet = candidate["flow_rate"] / AREA * (LENGTH / 12) * RHO * CP / K  # numerical / physical diffusivity x 2
print(
    f"open duct (Pe_h/2 {axial_peclet / 2:.0f}): "
    f"{100 * delivered[0]:.3f} % of the duty leaves at the outlet, {float(credits[0].conducted):.2f} W "
    f"back out of the inlet, storage {float(credits[0].stored):.1e} W; hottest cell {peaks[0]:.2f} K; "
    f"at half the time step {100 * delivered[1]:.3f} % and {peaks[1]:.2f} K"
)
assert delivered[0] >= DELIVERED and max(peaks) <= T_MAX and abs(float(credits[0].stored)) <= 1e-6 * DUTY


def largest_duty(duct, flow_rate, tolerance=1e-3):
    """The largest duty a design carries within its limits, or None where it fails them at no duty at all.

    More duty only heats the duct, so the feasible duties run from zero to that one, and bisection finds it.
    """
    if trial(duct, flow_rate, 0.0)["rejected"]:
        return None
    low, high = 0.0, DUTY
    while not trial(duct, flow_rate, high)["rejected"]:
        low, high = high, 2.0 * high
    while high - low > tolerance * high:
        middle = 0.5 * (low + high)
        low, high = (middle, high) if not trial(duct, flow_rate, middle)["rejected"] else (low, middle)
    return low


# Capacity: the most duty each sampled shape removes for the square duct's pumping power.
budget = square["hydraulic_power"]
capacity = [(each["aspect"], largest_duty(each, np.sqrt(budget * each["rate"] / LENGTH))) for each in ducts]
print("capacity at the square's power: " + ", ".join(
    f"b/a {r:.3f} " + ("infeasible" if duty is None else f"{duty / DUTY:.2f}") for r, duty in capacity
))

# At a fixed first-wall flux instead, the square's: each shape's duty grows with its heated width 2a, so compare
# the pumping power per metre of first wall.
flux = DUTY / (2.0 * ducts[-1]["a"] * LENGTH)
per_width = []
for each in ducts:
    duty = flux * 2.0 * each["a"] * LENGTH
    flows = FLOWS * duty / DUTY  # the same sweep, scaled to this duty
    feasible = [not trial(each, flow_rate, duty)["rejected"] for flow_rate in flows]
    j = next(j for j in range(1, len(flows)) if feasible[j] > feasible[j - 1])
    record = lowest_flow(each, flows[j - 1], flows[j], duty)
    per_width.append((each["aspect"], record["hydraulic_power"] / (2.0 * each["a"])))
print(f"at the square's flux, {flux / 1e3:.1f} kW/m^2, W per metre of first wall: " + ", ".join(
    f"b/a {r:.3f} {w:.3e}" for r, w in per_width
))
```
````

```text
b/a 0.080: x x x x x o o o o
b/a 0.122: x x x x x o o o o
b/a 0.186: x x x x x o o o o
b/a 0.283: x x x x x o o o o
b/a 0.431: x x x x x o o o o
b/a 0.656: x x x x x x o o o
b/a 1.000: x x x x x x x o x
lowest flows: b/a 0.080 1.7965e-05, b/a 0.122 1.7358e-05, b/a 0.186 1.7508e-05, b/a 0.283 1.8325e-05, b/a 0.431 1.9969e-05, b/a 0.656 2.3264e-05, b/a 1.000 3.1078e-05
square: b/a 1.0000, Q 3.1078e-05 m^3/s, dp 395.4 Pa, W 1.229e-02 W, wall 624.99 K, flux 98.6 kW/m^2, Re_Dh/Ha_a 117, Pe_Dh 184, Rm 1.1e-03, viscous/duty 4.4e-06, Joule/duty 3.8e-06
  Re_a 4583, Ha_a 78.46, N_a 1.343, Ri_a 0.02053, Ri_a/N_a 0.01529, Gr_a 4.31e+05
candidate: b/a 0.1436, Q 1.7330e-05 m^3/s, dp 127.2 Pa, W 2.205e-03 W, wall 625.00 K, flux 37.4 kW/m^2, Re_Dh/Ha_a 16, Pe_Dh 68, Rm 1.6e-03, viscous/duty 9.9e-07, Joule/duty 4.8e-07
  Re_a 6744, Ha_a 207, N_a 6.355, Ri_a 0.1742, Ri_a/N_a 0.02741, Gr_a 7.92e+06
at the same duty, the candidate needs 82.1 % less pumping power than the square duct
48/6 cells: Q -0.03 %, W +0.69 %
open duct (Pe_h/2 270): 99.968 % of the duty leaves at the outlet, 0.49 W back out of the inlet, storage 4.1e-12 W; hottest cell 624.74 K; at half the time step 99.968 % and 624.74 K
capacity at the square's power: b/a 0.080 1.85, b/a 0.122 2.04, b/a 0.186 2.01, b/a 0.283 1.81, b/a 0.431 1.54, b/a 0.656 1.26, b/a 1.000 1.00
at the square's flux, 98.6 kW/m^2, W per metre of first wall: b/a 0.080 1.755e+00, b/a 0.122 6.949e-01, b/a 0.186 5.039e-01, b/a 0.283 4.700e-01, b/a 0.431 5.012e-01, b/a 0.656 5.930e-01, b/a 1.000 7.933e-01
```

These are forced-convection screening results. For the candidate, buoyancy against inertia is not small,
with $Ri_a = 0.17$ against 0.02 for the square. Against the Lorentz force at the flow speed it is small, with $Ri_a/N_a$
0.027 for the candidate and 0.015 for the square. This page has not qualified buoyancy, its direction, the electrical
walls' closure or the thermal entrance. Mixed-convection MHD studies show buoyancy reorganizing the flow and the wall
temperatures in heated ducts (Zikanov et al. 2013; Vetcha, Smolentsev and Abdou; Mistrangelo et al. 2025), so a
coupled buoyant calculation with the direction of gravity declared has to come before an engineering choice.

**At a fixed duty per channel** the two ducts compare as follows.

| | Square | Candidate |
|:--|:--|:--|
| $b/a$ | 1 | 0.144 |
| half-widths $a \times b$ | 7.75 × 7.75 mm | 20.4 × 2.94 mm |
| first-wall flux | 98.6 kW/m² | 37.4 kW/m² |
| flow | 3.11×10⁻⁵ m³/s (0.129 m/s) | 1.73×10⁻⁵ m³/s (0.072 m/s) |
| $\Delta p$ | 395 Pa | 127 Pa |
| hydraulic power | 1.23×10⁻² W | 2.21×10⁻³ W |
| wall-to-bulk at the outlet | 26.1 K | 5.6 K |
| Joule heat | 5.7×10⁻³ W | 7.2×10⁻⁴ W |
| $Re_a$, $Ha_a$, $N_a$ | 4,583, 78.5, 1.34 | 6,744, 207, 6.36 |
| $Ri_a$, $Ri_a/N_a$ | 0.021, 0.015 | 0.174, 0.027 |
| $Re_{D_h}/Ha_a$, $Pe_{D_h}$, $Rm$ | 117, 184, 1.1×10⁻³ | 16, 68, 1.6×10⁻³ |
| viscous and Joule heat over the duty | 4.4×10⁻⁶, 3.8×10⁻⁶ | 9.9×10⁻⁷, 4.8×10⁻⁷ |

- Both ducts run at the 625 K wall limit. The slender duct needs 44 % less flow, because the duty enters through its
  wide side wall at a lower flux and crosses a thin gap.
- Joule and viscous heat are each under 10⁻⁵ of the duty here, which justifies leaving viscous heat out of this
  example. That does not carry over to every strong-field duct.
- The candidate's $b/a$ of 0.1436 is the isothermal law's $\beta^* = 0.145$ at $H = 78.5$. Near it the lowest
  feasible flow changes by about 1 % across $b/a$ 0.12–0.19, so pressure sets the shape and the temperature limit
  sets the flow.
- $N_a = 1.34$ for the square duct, so its inertia is not small against the Lorentz force. A straight developed
  laminar flow has no inertial term, but its entrance region and stability would feel it.

**At a fixed first-wall flux** the comparison is per metre of first wall, since covering a wall takes $1/(2a)$
channels per metre and a wide channel takes more duty. Of the sampled shapes, $b/a = 0.28$ is the cheapest at
0.470 W/m, 41 % below the square's 0.793 W/m, and $b/a = 0.08$ costs 1.755 W/m. The best shape therefore depends on
whether the application fixes the duty or the flux.

**Checks.** The square duct's pressure drop is 1.48 % below the spectral solver's (`validation.shercliff`) on the
screening mesh, and converges at second order: −0.66 %, −0.37 % and −0.17 % on 48/6, 64/8 and 96/12 cells. Its Joule
heat equals `core3d.energy_budget`'s to $10^{-9}$, below its hydraulic power. On 48/6 cells the candidate's minimum
flow moves −0.03 % and its power +0.69 %. In an open duct of twelve axial cells from a 573 K inlet, 99.968 % of its
duty leaves at the outlet and 0.49 W conducts back out of the inlet. The storage is 4×10⁻¹² W, half the time step
changes neither number, and the hottest cell is 624.74 K. The axial cell Péclet number, $Pe_h/2 = 270$, shows the run
is stationary, not that it is accurate in space, and twelve cells do not resolve the entrance region. So this checks
the delivered duty, not the screening's peak. At the square duct's pumping power the slender shapes carry up to 2.04
times the duty, at $b/a$ 0.122.

This is a screening optimum within the declared model, not an engineering design. The model is forced convection in
the fluid only, with the wall-interface temperature, constant properties, laminar developed flow and one uniform
field. Conjugate walls, a varying field and the coupled buoyant flow are later steps.

## Limits and the record

The law assumes insulated walls, the Stokes limit and an isothermal fluid. The 3-D check covers $Ha \le 200$ and one
ramp amplitude, so nothing here is a reactor-$Ha$ 3-D number. The 2-D law reaches $Ha^* = 6{,}109$.

The full study is in [LMhdX-Duct-Optimization](https://github.com/uwplasma/LMhdX-Duct-Optimization), with its
optimizer, checkpointed runs, figure pipeline and
[write-up](https://github.com/uwplasma/LMhdX-Duct-Optimization/blob/stage2-v1/results/stage2/README.md). A
[reproduction on LMhdX 1.10.0](https://github.com/uwplasma/LMhdX-Duct-Optimization/blob/stage2-v1/results/stage2-repro-1.10.0/README.md)
is bit-identical on the fully developed path. The release
[stage2-v1](https://github.com/uwplasma/LMhdX-Duct-Optimization/releases/tag/stage2-v1) carries
[results.json](https://github.com/uwplasma/LMhdX-Duct-Optimization/releases/download/stage2-v1/stage2-results.json),
the [figures](https://github.com/uwplasma/LMhdX-Duct-Optimization/releases/download/stage2-v1/stage2-figures.zip) and
the [tables](https://github.com/uwplasma/LMhdX-Duct-Optimization/releases/download/stage2-v1/stage2-tables.zip).

The thermal design of plan step 5.5 is the study's forced-convection screening of Case P,
[Stage 3](https://github.com/uwplasma/LMhdX-Duct-Optimization/blob/stage3-v1/results/stage3-thermal/README.md), with
the follow-ups [Stage 3 v2](https://github.com/uwplasma/LMhdX-Duct-Optimization/blob/stage3-v2/results/stage3-v2/README.md)
and [Stage 3 v3](https://github.com/uwplasma/LMhdX-Duct-Optimization/blob/579f4d481584d28e85d472df28e615de937bb3c3/results/stage3-v3/README.md).
Each shape takes its cheapest flow that keeps the developed peaks within 625 K, and the study minimises the hydraulic
power over area and aspect under `Re/Ha ≤ 200` and an area cap.

| | Fixed duty, 1.5 kW | Fixed first-wall flux, 96.8 kW/m² |
|:--|:--|:--|
| The optimum | Stage 2's design, to 0.1 % | b/a 0.28, against Stage 2's 0.15 |
| Stage 2's design costs | the same (0.9997 at a common flow) | 21 % more per metre of first wall (26 % at a common flow) |
| The flow it needs | 7.2 times Stage 2's velocity bound | |
| On 72/9 cells | within 0.4 % | within 0.4 % |
| Open duct, 12–48 axial cells (v2) | ≥ 99.96 % delivered; wall face 624.936 ± 0.003 K | ≥ 99.96 % delivered; wall face 624.952 ± 0.001 K |

Every gate of Stage 3 and v2 passes. The fixed-flux optimum's stationarity, 1.8e-6 against 1e-6 in Stage 3, is met
by v3, which the study declared before its run. Stage 3's polish had placed the minimum 2.55e-6 short in ln b/a, and
one Newton step brings the slope to 1.7e-10. It is a study, not a library feature, and `examples/duct_design_law.py`
still sizes a duct by its velocity bound, `MIN_VELOCITY`.

The study's own limits:
- An illustrative field cap of about 0.48 T, insulating electrical walls, fixed properties and fully developed
  inertialess flow. The small margins to 625 K leave out the walls, buoyancy and the properties' uncertainty, so they
  are not a design margin.
- The `Re/Ha ≤ 200` filter mixes length scales. The Hartmann-layer Reynolds number is $(Re/Ha)(a/D_h)$, with
  $D_h/a = 4\beta/(1+\beta)$, so the filter does not hold it at one value across aspect ratios. A rerun will apply
  each threshold in its own definition. The two reported designs stay feasible under the corrected filter, which does
  not show they stay optimal.
- The study optimizes the aspect ratio by derivative-free bracketing on rebuilt meshes. Gradients in the design
  variables come with LMhdX 2.0. A blanket design would add the gross channel pitch, the total heat removal and the
  coverage as constraints.

On LMhdX 1.11.0 the four designs reproduce stage3-v1's results.json exactly. The releases list each asset's SHA-256:
[stage3-v1](https://github.com/uwplasma/LMhdX-Duct-Optimization/releases/tag/stage3-v1)
([results.json](https://github.com/uwplasma/LMhdX-Duct-Optimization/releases/download/stage3-v1/stage3-results.json),
`2dda7498809683860b889cefb67cc174ae65bf4b5614bfaf05d5c0a9bf1f98ca`) and
[stage3-v2](https://github.com/uwplasma/LMhdX-Duct-Optimization/releases/tag/stage3-v2)
([results.json](https://github.com/uwplasma/LMhdX-Duct-Optimization/releases/download/stage3-v2/stage3-v2-results.json),
`7b4fd3b008fcc9ef5a4461853fd6e7c18ff7eae03843e37a408d84df37acb32a`).
