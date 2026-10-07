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
python examples/duct_design_law.py  # about 20 s on a CPU
```

```text
Design law over H = (10.0, 30.0, 100.0, 300.0), 48^2 cells, five solves per H...
H    10: beta* 0.53471, Ha*    13.7, s* 1.9774, 8.9% below the square
H    30: beta* 0.27147, Ha*    57.6, s* 2.0599, 26.3% below the square
H   100: beta* 0.12337, Ha*   284.7, s* 2.0816, 46.1% below the square
H   300: beta* 0.05958, Ha*  1229.0, s* 2.0888, 61.1% below the square
Blanket duct: 6 stations at 0.465-0.497 T, area 2.40 cm^2...
design: beta* 0.14851, a 20.10 mm, b 2.99 mm, W* 4.1495e-05 W, 41.7% below the square
Wrote artifacts/examples/duct_design_law/duct_design_law_summary.json and artifacts/examples/duct_design_law/duct_design_law.png
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

## Limits

Insulated walls, the Stokes limit and an isothermal fluid. The 3-D check covers
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
