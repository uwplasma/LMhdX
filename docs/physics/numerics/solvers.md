(numerics-solvers)=
# Linear and nonlinear solvers

```{image} ../../_static/device_scaling.webp
:alt: Time per step against size for the 3-D core and Q2D, on a CPU and a GPU
:align: center
```

*Time per step against problem size for the 3-D core and Q2D, on a CPU and one GPU, in both precisions, with the G4 targets drawn.*

LMhdX inverts its operators with direct fast-diagonal solves where the grid allows, and with preconditioned Krylov
methods elsewhere. The operators come from [the staggered grid](grid.md). SOLVAX supplies the generic algorithms.

(numerics-poisson)=
## Fast-diagonal Poisson and Helmholtz solves

On a tensor-product grid the Laplacian is the Kronecker sum of three one-dimensional operators, each symmetric once
the cell widths are folded in. `lmhdx.poisson` diagonalizes them on the host, so a solve is three tensor contractions
and one elementwise divide. It reads the one-dimensional operators out of `lmhdx.ops` by applying the assembled
Laplacian to unit vectors, so the factorization cannot differ from the stencil the rest of the code uses.

- Its cost does not grow with the Hartmann number the way an iteration count does.
- Its answer is exact to round-off, not to a tolerance.
- The solve is a linear map, so it differentiates without taping any iteration.

A pure Neumann or fully periodic problem is singular. The solver removes the constant from the right-hand side and
returns a field with zero volume-weighted mean. Inhomogeneous boundary data belongs in the right-hand side, so the
solver refuses a condition that carries a value. A guard rejects any axis operator that is not symmetric under the cell
widths.

The same factorization with a shift solves Helmholtz problems. `ChannelProblem.viscous_factorizations` factorizes
$(1+\Delta t\,\lambda)I-\Delta t\,\nu\nabla^2$ for the implicit viscous step ([Time stepping](time.md)), and the pipe
uses component Helmholtz solves, one radial eigendecomposition per azimuthal mode. Thin conducting walls keep the
Kronecker structure ([Thin and resolved conducting walls](grid.md)).

(numerics-krylov)=
## Krylov methods

- **CG and PCG.** The steady Stokes-limit operator is symmetric in the face-volume inner product, so steady ducts,
  open ducts in the Stokes limit and the open pipe solve it by preconditioned conjugate gradients.
- **The constrained pressure solve.** The projection step solves $D H^{-1} G$ by SOLVAX's constrained solve. It takes
  one iteration where $H$ commutes with the gradient (periodic axes) and a few next to a wall.
- **Newton-Krylov.** An open duct with inertia can take Newton's method, with each update a flexible GMRES solve
  (`solvax.gcrot`) preconditioned by the Stokes-limit operator's inverse
  ({ref}`March or Newton <open-ducts-march-newton>`).
- **GMRES** solves the tangent and transposed systems of a differentiated fixed point ([Derivatives](derivatives.md)).

(numerics-preconditioners)=
## Preconditioners

One projection step preconditions the steady CG solve. In a uniform field it uses the field-line solve and the peak
Joule rate. In a varying field, since plan step 2b.4, its damping is one global rate much smaller than the peak.
`lmhdx.steady._varying_field_rate` is

$$
r = (\nu\lambda_1)^{3/4}\,(\sigma|\mathbf B|^2_{\max}/\rho)^{1/4},
$$

with $\lambda_1$ the sum of the smallest Laplacian eigenvalues of the three axes, about $2(\pi/2)^2$ in a duct of
half-width one. The rate is 33, 66 and 132 at Ha 100, 400 and 1600, against peak rates of $10^4$ to $2.6\times10^6$.

- **Why one small rate.** The slowest preconditioned modes carry almost no Joule dissipation and need less damping, not
  more. A damping that varies in space, by component or by mode reopens the Schur-complement deficit of the projection
  and costs more iterations. The count is flat within 10 % over a wide band of rates, and the chosen rate falls inside
  it.
- **The gain.** On the ANL fringe duct CG iterations fall 3.2×, 2.8×, 2.1× and 1.9× at Ha 100, 400, 1600 and 3200.
  The count still grows, as $Ha^{0.83}$ on the ANL meshes against $Ha^{0.68}$ before.
- **Nothing else changes.** Uniform fields keep their iteration counts and fields bit for bit. On the varying cases the
  fields agree to the solve tolerance, and the ANL excess is the same to five digits.
- **Derivatives use the same CG.** `jax.value_and_grad` of the mean velocity in the field scale falls from 1.77 to 0.80
  s warm at Ha 100 and from 5.98 to 2.40 s at Ha 300.

The alternatives that lost to this rate are in ["The varying-field preconditioner (2b.4)" at
b97ad3a](https://github.com/uwplasma/LMhdX/blob/b97ad3a8525a20502279961d5b4201d158d971e9/docs/physics/numerics.md#the-varying-field-preconditioner-2b4).

(numerics-defect-correction)=
## Defect correction and the tolerance rule

The steady residual of `lmhdx.steady` projects twice. The fast pressure solve is exact only to round-off, and on a
stretched mesh that leaves about $10^{-13}$ of the removed gradient outside the divergence-free fields. The CG
preconditioner ends in a projection, so it cannot see that part and CG stalls on it. A varying field's force has a
large gradient part. On the four-cell varying duct of the tests the floor was $5.0\times10^{-10}$ of the right-hand
side at Ha 20 and $6.4\times10^{-7}$ at Ha 100. The second projection is one defect correction of the pressure solve.
It brings the floors to $1.2\times10^{-13}$ and $3\times10^{-12}$. Every varying-field duct measured, the ANL fringe
included, certifies at the default tolerance $10^{-9}$ through Ha 300.

Above Ha 300 a floor remains. On the fringe mesh of the tests (24 cells, 16 axial) CG stalls at $1.5\times10^{-10}$ of
its right-hand side at Ha 300, $7.5\times10^{-10}$ at Ha 600 and $1.5\times10^{-8}$ at Ha 1000. So the tolerance is
$10^{-9}$ through Ha 300, $10^{-8}$ to Ha 600 and $10^{-7}$ to Ha 1000, with `linear_max_restarts=600` (36,000 CG
iterations) above Ha 300. The three solves take 5,581, 12,056 and 19,299 iterations.
`test_the_fringe_certifies_at_the_tolerance_rule` holds the fringe to that rule. The open duct adds one more defect
correction ({ref}`Boundary conditions <open-ducts-bcs>`).

(numerics-solvax)=
## LMhdX and SOLVAX

LMhdX owns geometry metrics and material coefficients, boundary and interface equations, MHD coupling and dimensional
scaling, the charge, mass, momentum and power residuals, and case-level convergence and validation. SOLVAX owns PCG,
GMRES/FGMRES and fixed-point iteration, the preconditioners (Jacobi, line, additive, deflation and Schur), tridiagonal
and sparse direct solves, solver state and termination metadata, implicit linear differentiation, and checkpointed
exact reverse mode for long recurrences. LMhdX calls these algorithms with MHD-specific operator actions, then
certifies the returned state in physical units.

(numerics-preconditioner-table)=
## The varying-field preconditioner before and after

The table is an A/B comparison on the office host (JAX 0.6.2, SOLVAX 0.19.0), `main` adc0340 against the 2b.4
branch, in fresh processes. The test cases ran on the CPU and the ANL duct of #184 on one A4000.

| case | mesh | iterations, before to after | warm (s) | cold (s) |
|---|---|---|---|---|
| uniform duct, Ha 300 (field lines) | $1\times48^2$ | 44 to 44 | 0.03 to 0.03 | 2.0 to 1.9 |
| uniform duct, Ha 1000 (field lines) | $1\times48^2$ | 76 to 76 | 0.05 to 0.07 | 1.8 to 1.9 |
| varying duct of the tests, Ha 20 | $4\times24^2$ | 122 to 83 | 0.16 to 0.10 | 1.9 to 1.9 |
| varying duct, Ha 100, $c=0.05$ | $4\times24^2$ | 793 to 346 | 0.74 to 0.32 | 2.8 to 2.4 |
| varying duct, Ha 300 | $4\times24^2$ | 2,903 to 977 | 2.9 to 0.84 | 5.3 to 3.0 |
| periodic fringe of the tests, Ha 300 | $16\times24^2$ | 5,634 to 1,361 | 28 to 6.6 | 30 to 8.8 |
| periodic fringe, Ha 600 | $16\times24^2$ | 12,092 to 2,489 | 59 to 13 | 62 to 16 |
| periodic fringe, Ha 1000 | $16\times24^2$ | 19,585 to 3,874 | 95 to 19 | 99 to 22 |
| ANL fringe duct, Ha 100 | $70\times32^2$ | 783 to 244 | 3.8 to 1.2 | 23 to 20 |
| ANL fringe duct, Ha 400 | $70\times48^2$ | 2,047 to 728 | 24 to 8.6 | 46 to 30 |
| ANL fringe duct, Ha 1600 | $70\times96^2$ | 4,848 to 2,283 | 239 to 111 | 265 to 137 |
| ANL fringe duct, Ha 3200 | $70\times128^2$ | 8,307 to 4,353 | 810 to 425 | 841 to 456 |
| ANL fringe duct, Ha $10^4$ (after only) | $70\times128^2$ | 13,406 | 1,307 | 1,339 |
