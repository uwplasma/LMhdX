# The duct design law

At a fixed flow rate, an insulated rectangular duct in a strong transverse field
has one aspect ratio of least pumping power. With $a$ the half-width along the
field, $b = \beta a$ the half-width across it and $Ha = B a\sqrt{\sigma/\mu}$,
the optimum at a fixed cross-section area obeys

$$
s^* = \beta^*\sqrt{Ha^*} \to 2.08 ,
$$

from $s^* = 1.975$ at $Ha^* = 14$ to $2.078$ at $Ha^* = 6{,}109$ on the
spectral reference. Shercliff's thin-layer friction law,
$\Delta p/L = \mu V Ha / [a^2 (1 - \alpha/(\beta\sqrt{Ha}) - 1/Ha)]$, puts the
balance where the side layers remove two fifths of the flow, $s^* = 2.5\alpha$:
2.13 with $\alpha = 0.852$ as Smolentsev (2021) gives it, 2.06 with the
$\alpha = 0.825$ that Müller & Bühler print. The optimum is deep along the
field and narrow across it, and its advantage grows with the field: 9 % below
the square duct of the same area at $Ha^* = 14$, 61 % at 1,229 and 73 % at 6,109.

The optimum itself is not new. Nishio et al. (2025) computed the same interior
minimum (fixed area and flow, insulating walls, 10 T), and their numbers imply
$s^* \approx 2.08$; Müller &
Bühler's friction formula gives it in one line. LMhdX's study states the law
with its finite-$Ha$ values, checks it against an independent solver and
three-dimensional solves, and extends it to a tilted field.

## Run it

```bash
python examples/duct_design_law.py  # about 15 s on a CPU
```

```text
H    10: beta* 0.53471, Ha*    13.7, s* 1.9774, 8.9% below the square
H    30: beta* 0.27147, Ha*    57.6, s* 2.0599, 26.3% below the square
H   100: beta* 0.12337, Ha*   284.7, s* 2.0816, 46.1% below the square
H   300: beta* 0.05958, Ha*  1229.0, s* 2.0888, 61.1% below the square
design: beta* 0.14851, a 20.10 mm, b 2.99 mm, W* 4.1495e-05 W, 41.7% below the square
```

The example does two things. First, the design law: at a fixed area, with $H$
the Hartmann number of the equal-area square duct, it finds $\beta^*$ for each
$H$. Second, one design: a PbLi duct (573 K) running 0.982 m radially through
an outboard blanket in a $1/R$ field of 0.5 T at the first wall, at
$Q = 2.4\times10^{-6}$ m³/s. There the pumping power falls as the duct grows,
so the smallest allowed mean velocity, 10 mm/s, sets the area. The sign of
$d\Delta p/d\ln A$, exact through `jax.grad` of the solve in its field scale,
confirms that this bound is active. The aspect ratio is then the one choice
left.

Each trial $\beta$ scales one mesh across the field, so the discrete pressure
drop is smooth in $\beta$, and a quartic through five trials finds the minimum.
All trials have the same mesh shape and share one compiled program.

## What is established

On 48/6 cells, the example reproduces the study's optima to the digits printed.
The study's own evidence:

- **Mesh convergence.** Re-solved on 32/4, 48/6 and 72/9 cells at $H$ = 30,
  100 and 300, $\beta^*$ has a grid convergence index of 0.18–0.42 %.
- **An independent solver.** The core's $s^*$ sits 0.1–0.8 % above the
  spectral reference (`validation.shercliff`, with its `aspect` keyword).
- **The station sum in a varying field.** `validation.open_axis.ramp_excess`
  compares the locally fully developed sum with a 3-D solve of an open duct in
  a monotone 20 % ramp. For a square duct the sum is good to 1 % only up to
  $\gamma\sqrt{Ha} \approx 0.2$, where $\gamma$ is the field's relative change
  per half-width. The design-law duct stays under 0.35 % up to
  $\gamma\sqrt{Ha} = 2$. The excess does not collapse on $\gamma\sqrt{Ha}$
  across $Ha$ 50 and 200, so these limits are quoted per $Ha$.

```python
from validation.open_axis import ramp_excess

row = ramp_excess(0.2, 1.0, 200.0, cells=48, cells_in_layer=6)  # gamma sqrt(Ha), beta, Ha
print(row["excess_percent"])  # 0.448
```

## A tilted field

A poloidal field tilts $B$ within the cross-section, and the slender optimum is
sensitive to that tilt where the square duct is not. Re-optimized in a field
tilted by $B_p/B_T$, the optimum and its penalty collapse on
$\kappa = \sqrt{Ha^*_0}\,B_p/B_T$ to within 5 %:

$$
\beta^{*2} = \beta_0^{*2} + (B_p/B_T)^2 ,\qquad
\left(\Delta p^*/\Delta p^*_0\right)^2 = 1 + \frac{0.122\,\kappa^2}{1 + \kappa/3.01} .
$$

A penalty of at most 10 % needs $\kappa < 1.7$. A duct kept at its aligned
shape loses to the square beyond $\kappa \approx 8$–12 at $Ha^*_0$ 1,232–6,127.
Both laws are fitted to the same 20 points ($Ha^*_0$ 58–6,127, $B_p/B_T \le 0.3$,
one uniform tilt per duct), so their errors are in-sample.

## A thermal design

The design above takes its area from a smallest allowed velocity, which stands
in for heat removal. A thermal design asks the question directly: which shape
and flow remove a fixed duty within a temperature limit, at the least pumping
power?

**The specification** is illustrative. None of these values is an engineering
requirement.
- **The fluid** is Pb-17Li at 573 K, with constant properties from Martelli et
  al. (2019), Table 1: $\rho$ 9838 kg/m³, $c_p$ 189.8 J/(kg K), $k$ 20.4 W/(m K),
  $\mu$ 2.153 mPa s and $\sigma$ 8.835×10⁵ S/m, so $Pr = 0.020$.
- **The correlations' ranges differ:** 508–800 K for $c_p$, 508–625 K for
  $\mu$, and 600–800 K for $\sigma$. The temperature limit, 625 K, is the top of
  the viscosity's range.
- **The duct** is straight, with the design's area, 2.4×10⁻⁴ m², and length,
  0.982 m. Its walls are insulating, in a uniform 0.5 T along $a$.
- **The duty** is 1.5 kW, entering through the first wall $z=-b$. That is a side
  wall, parallel to the field, as for a channel behind a first wall whose
  toroidal field lies in its plane. The duty is held fixed as the shape
  changes, so the flux is $P/(2aL)$. It is deposited in the fluid's row of cells
  along that wall, which tends to the surface flux as the mesh refines.
- **The inlet** is at 573 K.
- **The limits:**
  - fluid and wall-interface peaks of at most 625 K;
  - $Re/Ha \le 200$, for laminar Hartmann layers;
  - at least 95 % of the duty leaving at the outlet.
- **The cost** is the hydraulic power $Q\,\Delta p$, with no assumed pump
  efficiency.

**Each trial** follows four steps:
1. Solve the developed flow of its shape under a unit pressure gradient.
2. Scale that flow to the trial's flow. The Stokes flow is linear in its drive,
   and the Joule heat is recomputed from the scaled velocity.
3. Solve the developed temperature with `developed_heat`.
4. Take its peaks over the length from the inlet's 573 K with
   `DevelopedHeat.anchored_peaks`.

A rejected trial records why.

**The search** has three stages:
1. A 7 × 9 sweep in $\ln(b/a)$ and $\ln Q$.
2. Bisection of each observed lower edge of the feasible flows, between a
   sampled infeasible flow and a feasible one.
3. A golden-section refinement of the cheapest shape.

Nothing assumes the temperature falls monotonically with flow. The square
duct's feasible flows form an interval, closed above by $Re/Ha$. The shape is a
host design variable, so no derivative through the geometry is claimed.

```python
import jax.numpy as jnp
import numpy as np

import lmhdx
from lmhdx.core3d import ChannelProblem, zero_velocity
from lmhdx.fully_developed import channel_flow_rate
from lmhdx.grid import CENTER, DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition, Field, Grid
from lmhdx.grid import uniform_faces, wall_resolving_faces
from lmhdx.steady import solve_compiled
from lmhdx.thermal import ThermalProblem, advance_thermal, build_thermal_step, developed_heat
from lmhdx.thermal import initial_thermal_state, load_attribution, thermal_boundary, validate_developed_inputs

lmhdx.enable_x64()

# Pb-17Li at 573 K, held constant: Martelli et al. (2019), Table 1.
RHO, CP, K, MU, SIGMA = 9838.0, 189.8, 20.40, 2.153e-3, 8.835e5  # kg/m^3, J/(kg K), W/(m K), Pa s, S/m
# An illustrative specification, not an engineering one.
AREA, LENGTH, FIELD = 2.4e-4, 0.982, 0.5  # m^2 and m, fixed; T, uniform, along the half-width a
DUTY = 1500.0  # W, through the first wall z = -b, parallel to the field
T_IN, T_MAX = 573.0, 625.0  # K; 625 K is the top of the viscosity correlation's range
LAMINAR, DELIVERED = 200.0, 0.95  # the largest Re/Ha; the least fraction of the duty leaving at the outlet
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
    rate = float(channel_flow_rate(flow, unit[0].data[0]))
    thermal = ThermalProblem(flow, (WRAP, INSULATED, INSULATED), CP, K, T_IN)
    return dict(aspect=aspect, a=a, hartmann=hartmann, unit=unit, rate=rate, thermal=thermal)


def first_wall(grid, duty):
    """The duty in the fluid's row of cells along the first wall: its surface flux as the mesh refines."""
    data = np.zeros(grid.shape)
    data[:, :, 0] = duty / np.sum(grid.cell_volumes()[:, :, 0])
    return Field(jnp.asarray(data), (CENTER,) * 3, grid)


def trial(duct, flow_rate, duty=DUTY):
    """A design's developed temperatures, anchored at the inlet, its cost, and every reason to reject it."""
    thermal, drive = duct["thermal"], flow_rate / duct["rate"]
    velocity = tuple(component.replace_data(drive * component.data) for component in duct["unit"])
    load = first_wall(thermal.flow.grid, duty)
    record = dict(aspect=duct["aspect"], flow_rate=flow_rate, duty=duty, velocity=velocity,
                  pressure_drop=drive * LENGTH, hydraulic_power=flow_rate * drive * LENGTH)
    try:
        validate_developed_inputs(
            thermal, velocity, load, flow_rate_tolerance=1e-12, velocity_tolerance=1e-9 * flow_rate / AREA
        )
    except ValueError as error:
        return {**record, "rejected": [f"invalid: {error}"]}
    result = developed_heat(thermal, velocity, load)
    record["wall_peak"], record["fluid_peak"] = (float(t) for t in result.anchored_peaks(T_IN, LENGTH))
    record["joule"] = RHO * CP * float(result.axial_gradient) * flow_rate * LENGTH - duty
    record["re_over_ha"] = RHO * flow_rate / AREA * result.hydraulic_diameter / MU / duct["hartmann"]
    rejected = [
        f"{name} {record[name]:.1f} K over {T_MAX:.0f} K"
        for name in ("wall_peak", "fluid_peak")
        if not record[name] <= T_MAX
    ]
    if not record["re_over_ha"] <= LAMINAR:
        rejected.append(f"Re/Ha {record['re_over_ha']:.0f} over {LAMINAR:.0f}")
    return {**record, "rejected": rejected}


def lowest_flow(duct, below, above, tolerance=1e-3):
    """Bisect ln Q between an infeasible and a feasible flow, which must bracket the boundary."""
    if trial(duct, above)["rejected"] or not trial(duct, below)["rejected"]:
        raise RuntimeError(
            f"{below:.3g} and {above:.3g} m^3/s do not bracket b/a = {duct['aspect']:.3g}'s boundary"
        )
    while above / below - 1.0 > tolerance:
        middle = np.sqrt(below * above)
        below, above = (below, middle) if not trial(duct, middle)["rejected"] else (middle, above)
    return trial(duct, above)


# Sweep the bounded design set; at each aspect, refine each observed lower edge of the feasible flows.
ducts = [duct(aspect) for aspect in ASPECTS]
sweep = [[trial(each, flow_rate) for flow_rate in FLOWS] for each in ducts]
edges = []
for each, row in zip(ducts, sweep):
    feasible = [not record["rejected"] for record in row]
    print(f"b/a {each['aspect']:.3f}: " + " ".join("o" if ok else "x" for ok in feasible))
    edges += [
        lowest_flow(each, FLOWS[j - 1], FLOWS[j])
        for j in range(1, len(FLOWS))
        if feasible[j] > feasible[j - 1]
    ]

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
while high - low > 0.02:  # in ln(b/a)
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
    print(
        f"{name}: b/a {record['aspect']:.4f}, Q {record['flow_rate']:.4e} m^3/s, dp {record['pressure_drop']:.1f} Pa, "
        f"W {record['hydraulic_power']:.3e} W, wall {record['wall_peak']:.2f} K, Joule {record['joule']:.1e} W, "
        f"Re/Ha {record['re_over_ha']:.0f}"
    )
print(f"the candidate needs {100 * saving:.1f} % less pumping power than the square duct")

# Check the candidate: a finer mesh, then its flow carried down an open duct from an inlet at T_IN.
refined = lowest_flow(
    duct(candidate["aspect"], cells=48, layer=6), 0.9 * candidate["flow_rate"], 1.1 * candidate["flow_rate"]
)
profile, section = candidate["velocity"][0].data[0], candidate["velocity"][0].grid
grid = Grid(uniform_faces(12, 0.0, LENGTH), section.y_faces, section.z_faces)
dt = 0.5 * (LENGTH / 12) / float(jnp.max(profile))  # a Courant number of 0.5 along the duct
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
open_duct = ThermalProblem(flow, (inlet, INSULATED, INSULATED), CP, K, T_IN)
rest = zero_velocity(flow)
start = initial_thermal_state(
    open_duct, T_IN, (rest[0].replace_data(jnp.broadcast_to(profile, rest[0].shape)), *rest[1:])
)
step = build_thermal_step(open_duct, dtype=jnp.float64, hold_flow=True)
steps = int(np.ceil(6.0 * AREA * LENGTH / candidate["flow_rate"] / dt))  # six residence times
nothing, load = Field(jnp.zeros(grid.shape), (CENTER,) * 3, grid), first_wall(grid, DUTY)
credit = load_attribution(step, start, nothing, load, steps)
open_peak = float(advance_thermal(step, start, load, steps)[1].temperature_max[-1])
delivered = float(credit.delivered / credit.imposed)
print(
    f"48/6 cells: Q {100 * (refined['flow_rate'] / candidate['flow_rate'] - 1):+.2f} %, "
    f"W {100 * (refined['hydraulic_power'] / candidate['hydraulic_power'] - 1):+.2f} %"
)
print(f"open duct: {100 * delivered:.2f} % of the duty leaves at the outlet, peak {open_peak:.2f} K")
assert delivered >= DELIVERED and open_peak <= T_MAX


def largest_duty(duct, flow_rate, tolerance=1e-3):
    """The largest duty a design carries within its limits: more duty only heats it, so bisect from zero."""
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
print(
    "capacity at the square's power: " + ", ".join(f"b/a {r:.3f} {duty / DUTY:.2f}" for r, duty in capacity)
)
```

```text
b/a 0.080: x x x x x o o o o
b/a 0.122: x x x x x o o o o
b/a 0.186: x x x x x o o o o
b/a 0.283: x x x x x o o o o
b/a 0.431: x x x x x o o o o
b/a 0.656: x x x x x x o o o
b/a 1.000: x x x x x x x o x
square: b/a 1.0000, Q 3.0548e-05 m^3/s, dp 388.7 Pa, W 1.187e-02 W, wall 625.00 K, Joule 5.5e-03 W, Re/Ha 115
candidate: b/a 0.1436, Q 1.7267e-05 m^3/s, dp 126.8 Pa, W 2.189e-03 W, wall 625.00 K, Joule 7.1e-04 W, Re/Ha 16
the candidate needs 81.6 % less pumping power than the square duct
48/6 cells: Q +0.05 %, W +0.85 %
open duct: 99.97 % of the duty leaves at the outlet, peak 624.91 K
capacity at the square's power: b/a 0.080 1.84, b/a 0.122 2.03, b/a 0.186 2.00, b/a 0.283 1.81, b/a 0.431 1.54, b/a 0.656 1.26, b/a 1.000 1.00
```

The workflow takes about 30 s on a CPU.

| | Square | Candidate |
|:--|:--|:--|
| $b/a$ | 1 | 0.144 |
| half-widths $a \times b$ | 7.75 × 7.75 mm | 20.4 × 2.94 mm |
| flow | 3.05×10⁻⁵ m³/s (0.127 m/s) | 1.73×10⁻⁵ m³/s (0.072 m/s) |
| $\Delta p$ | 389 Pa | 127 Pa |
| hydraulic power | 1.19×10⁻² W | 2.19×10⁻³ W |
| wall-to-bulk at the outlet | 25.7 K | 5.5 K |
| Joule heat | 5.5×10⁻³ W | 7.1×10⁻⁴ W |
| $Re/Ha$ | 115 | 16 |

**What the result shows:**
- **Both ducts run at the 625 K wall limit.** The slender duct needs 43 % less
  flow. The duty enters through its wide side wall and crosses a thin gap, so
  its wall runs only 5.5 K above the bulk.
- **Joule heat is a burden, not cooling,** and it is under 10⁻⁵ of the duty.
- **The shape is still the isothermal law's:** 0.1436, against
  $\beta^* = 0.145$ at $H = 78.5$. Near the optimum the minimum flow changes by
  under 1 % across $b/a$ 0.12–0.19. Pressure therefore sets the shape, and the
  temperature limit sets the flow, in place of the smallest allowed velocity.

**The reference checks.** The square duct's pressure drop is 1.48 % below
the spectral solver's (`validation.shercliff`) on the screening mesh. It
converges at second order: −0.66 %, −0.37 % and −0.17 % on 48/6, 64/8 and 96/12
cells. Its Joule heat equals `core3d.energy_budget`'s to $10^{-9}$, and is
below its hydraulic power, which in a developed Stokes flow is the viscous
plus the Joule dissipation.

**The candidate's checks:**
- **On 48/6 cells** its minimum flow moves +0.05 % and its power +0.85 %.
- **In an open duct.** Twelve axial cells carry its profile from an inlet at
  573 K, and `thermal.load_attribution` credits the duty by difference:
  - 99.97 % of it leaves at the outlet, and the rest conducts back out of the
    inlet;
  - the open duct's peak, with its entrance region, is 624.91 K, so the
    developed screening (developed from the inlet) is conservative here.
- **At the square duct's pumping power** the slender shapes carry up to 2.03
  times the duty, at $b/a$ 0.122.

This is a screening optimum within the declared model, not an engineering
design. The model is:
- fluid only;
- the wall-interface temperature, not the solid's;
- constant properties;
- laminar developed flow;
- one uniform field;
- no buoyancy.

Conjugate walls, a varying field and the coupled buoyant flow are later steps.

## Limits

Insulated walls and the Stokes limit; the law above is for an isothermal fluid. The 3-D check covers
$Ha \le 200$ and one ramp amplitude, so nothing here is a reactor-$Ha$ 3-D
number. The 2-D law reaches $Ha^* = 6{,}109$.

## The record

The full study lives in
[LMhdX-Duct-Optimization](https://github.com/TylerBrandes/LMhdX-Duct-Optimization).
That repository holds the optimizer, the checkpointed runs, the figure pipeline
and the
[write-up](https://github.com/TylerBrandes/LMhdX-Duct-Optimization/blob/stage2-v1/results/stage2/README.md).
It also holds a
[reproduction on LMhdX 1.10.0](https://github.com/TylerBrandes/LMhdX-Duct-Optimization/blob/stage2-v1/results/stage2-repro-1.10.0/README.md),
bit-identical on the fully developed path. The release
[stage2-v1](https://github.com/TylerBrandes/LMhdX-Duct-Optimization/releases/tag/stage2-v1)
carries
[results.json](https://github.com/TylerBrandes/LMhdX-Duct-Optimization/releases/download/stage2-v1/stage2-results.json),
the [figures](https://github.com/TylerBrandes/LMhdX-Duct-Optimization/releases/download/stage2-v1/stage2-figures.zip)
and the [tables](https://github.com/TylerBrandes/LMhdX-Duct-Optimization/releases/download/stage2-v1/stage2-tables.zip).
