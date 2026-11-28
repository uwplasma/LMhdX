(numerics-derivatives)=
# Derivatives

LMhdX chooses the derivative algorithm as part of each numerical method. A converged solve differentiates by its
implicit equations, and a finite transient differentiates its discrete update.

## The derivative policy

- **Converged linear or steady nonlinear equations** use an implicit tangent and adjoint system. The reverse pass then
  costs one more transposed solve, and that cost does not depend on the number of primal iterations.
- **Finite transient models** differentiate the discrete update. Generic 3-D ducts use an implicit electric VJP and
  exact checkpointed projection and outer recurrences
  ({ref}`One compiled trajectory <numerics-trajectory>`).
- **Q2D** uses the same two-level schedule. Its retained trajectory state is $O(N/C+C)$ for $N$ steps and width $C$,
  with a square-root default width.

JAX traces field arrays and continuous physical coefficients. Mesh topology, array shapes, iteration limits,
checkpoint widths, convergence strings, logging and file output are static or host-side.

## Implicit derivatives and adjoints

- **Drive elimination.** $df/dQ=1/G$ is analytic ({ref}`Drive elimination <numerics-drive-elimination>`).
- **Fast solves.** A fast-diagonal solve is a linear map, so it differentiates without taping any iteration
  ({ref}`Fast-diagonal solves <numerics-poisson>`).
- **Steady CG.** Derivatives use the same preconditioned CG as the primal solve.
- **The open duct with inertia.** Derivatives come from the implicit function theorem on the fixed point of one step
  (`solvax.root_solve`), with tangent and transposed solves by GMRES on that step's linearization. The march and
  Newton's method agree on the drop and its gradient to $10^{-6}$.
- **The core-flow model.** The direct factorization sits inside `jax.lax.custom_linear_solve`, and the adjoint reuses
  it ({ref}`Discretization <coreflow-discretization>`).

## Acceptance

Gradient acceptance combines an independent analytical, finite-difference or transpose check with primal and adjoint
residuals, compiled memory scaling, warm runtime, and parity between CPU and GPU. Some of the checks:

- the open duct's adjoint of the drop in the field scale matches central differences to 7e-10 on the test mesh, and
  the open pipe's to 2.8e-10;
- RK3 gradients through a short window pass the dot-product identity to $6\times10^{-15}$ and a Taylor-remainder test
  with slope 2.0;
- on the varying duct, gradients before and after the 2b.4 preconditioner agree to $10^{-11}$ and
  $5\times10^{-10}$ relative at Ha 100 and 300.
