# Fully developed heat transfer

These tables check the [fully developed duct and pipe](../physics/thermal/developed.md) against exact
series, closed forms and Chebyshev or spectral solves of the same equations. They verify the solver,
and none is a comparison with experiment.

```{image} ../_static/thermal_developed_duct.webp
:alt: Developed Nusselt numbers against their exact series and the Hartmann plates' closed form
:align: center
```

*The developed Nusselt numbers against their exact series, and the Hartmann plates against their closed form.*

## The developed duct

H1 and H2 Nusselt numbers. The flow is the package's own steady solve or a uniform slug:

| Duct | Exact | 16 / 32 / 64 cells | Order |
|:--|--:|:--|:--|
| square | 3.60795 | +1.35 % / +0.35 % / +0.087 % | 1.97, 1.99 |
| aspect ratio 0.5 | 4.12330 | +2.7 % / +0.73 % / +0.18 % | 1.92, 1.98 |
| square, slug flow | 7.11354 | −1.5 % / −0.38 % / −0.094 % | 1.98, 1.99 |
| square, H2 (uniform flux on every wall) | 3.08738 | +1.41 % / +0.36 % / +0.090 % | 1.98, 1.99 |

The H2 case tests the heat-flux path of the design workflow. Its reference,
`validation.thermal.h2_series`, expands the velocity in sines and the temperature as a lift plus
cosines, and bases Nu on the perimeter mean of the wall temperature. The MHD Nusselt campaign's
spectral solve agrees with it to 1e-8, and the package's Richardson value, 3.08733, to 2e-5. Shah &
London's (1978) 3.091, which the review of #240 cited, is 0.12 % above all three. Bluck et al. (2015)
tabulate 3.0871.

The H1 values lie on Shah & London's polynomial fit for rectangular ducts, with coefficients from
the `ht` library. The comparison never gates on the fit and does not consult the book's tables. The
tests check the aspect-0.5 duct only at 64 cells.

## Hartmann plates

Hartmann flow between plates at $y=\pm a$, each taking in a uniform heat flux with no source and no
Joule heat, has $Nu=4/\int_0^1F_H^2\,ds$ with $D_h=4a$. Here $F_H$ is the integral of the
mean-normalized Hartmann velocity $f_H=(1-\cosh(Hs)/\cosh H)/(1-\tanh H/H)$. The reference, by
quadrature, runs from $140/17$ at Ha 0 to 12 as Ha grows. Given the cell means of the analytical
velocity across one cell with insulated ends, the plates' own $Nu$ converges as follows:

| Flow | Exact | 32 / 64 / 128 cells | Order |
|:--|--:|:--|:--|
| Poiseuille, Ha 0 | 8.23529 | −2.6e‑7 / −1.6e‑8 / −1.0e‑9 | 4.0 |
| Ha 10 | 10.25858 | −1.2e‑5 / −7.6e‑7 / −4.7e‑8 | 4.0 |
| slug | 12 | −1.9e‑3 / −4.9e‑4 / −1.2e‑4 | 2.0 |

With the velocity's cell means the discrete fluxes are exact, and the $O(h^2)$ error of $T_w-T_b$ is
$+h^2\theta''_w/6$, half from the wall value and half from the bulk. Since
$k\theta''=\rho_0c_p\gamma u$, no slip removes it. With point values of the velocity the order falls
to 2. Slug flow keeps it, and its discrete answer is exactly $12/(1+h^2/2)$. The public four-wall
`nusselt` of that fixture averages in the insulated ends, and so gives a different number.

## The uniform wall temperature

$Nu_T$ from `developed_wall_temperature`:

| Duct | Reference | 16 / 32 / 64 cells | Order |
|:--|--:|:--|:--|
| square, slug flow | $\pi^2/2=4.93480$; discretely $(8/h^2)\sin^2(\pi h/4)$ | exact to 3e-16 at each | n/a |
| square | 2.97753 | +0.63 % / +0.16 % / +0.040 % | 1.97 |
| plates, Poiseuille (32 / 64 / 128) | 7.54070 | −4.6e‑4 / −1.1e‑4 / −2.9e‑5 | 2.0 |
| plates, Ha 10 (32 / 64 / 128) | 8.96255 | −6.5e‑4 / −1.6e‑4 / −4.1e‑5 | 2.0 |

The square duct's reference is a Chebyshev collocation of the same eigenproblem, extrapolated from
40, 48 and 56 nodes at its observed order 2.0. The package's own Richardson value, 2.97749, agrees
with it to 1.5e-5. Shah & London (1978) list 2.976, 0.05 % below both. The slug-flow row is exact
because slug flow's sampled sines are exact eigenfunctions of the cell-centred Dirichlet Laplacian.
The plates carry the cell means of the analytical velocity, with $D_h=4a$, and their
references are one-dimensional Chebyshev solves.

## The pipe

| Pipe: mean Nu on the diameter, $D_h=2R$ | Exact | 16 / 32 / 64 radial cells | Order |
|:--|--:|:--|:--|
| H (uniform flux; H1 and H2 coincide in a pipe), Poiseuille | 48/11 = 4.36364 | +0.28 % / +0.071 % / +0.018 % | 2.0 |
| H, slug | 8 | −0.39 % / −0.098 % / −0.024 % | 2.0 |
| T (one wall temperature), Poiseuille | 3.65679 | +0.077 % / +0.019 % / +0.0049 % | 2.0 |
| T, slug | $j_{0,1}^2=5.78319$ | −0.12 % / −0.031 % / −0.0077 % | 2.0 |

The value 3.65679 comes from a radial Chebyshev solve of the same eigenproblem across the diameter.
