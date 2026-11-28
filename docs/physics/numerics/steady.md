(numerics-steady)=
# Steady solves

A steady duct flow in LMhdX is the root of a residual, found by a solver and certified in physical units. Two ideas
keep these solves cheap and differentiable. The drive drops out of a fully developed problem, and the steady state is
a root that implicit differentiation handles.

(numerics-drive-elimination)=
## Drive elimination

At fixed field, materials and geometry, the fully developed inductionless problem is linear in the drive $f$. So the
volumetric flow rate is $Q=Gf$ for a single response $G$ that one solve measures. `lmhdx.steady.channel_flow_response`
uses this. The drive that delivers a requested flow rate is exactly $f=Q_{\rm target}/G$, and its derivative is
$df/dQ=1/G$. The tests check this analytic derivative against automatic differentiation of the solve.

An optimizer asked to find the drive would return a scalar good only to its own tolerance. It would also spend its
budget rediscovering a linear relation instead of exploring the inputs that change the flow, such as wall conductance,
aspect ratio and field strength.

For a duct driven by a uniform pressure gradient, the drop over a length $L$ is $fL$ and the hydraulic power is $fLQ$.
That is the isothermal power of a fully developed segment. It excludes entry and exit losses, manifolds and all
thermal effects, so it is not a blanket pumping budget.

(numerics-steady-root)=
## The steady state as a differentiable root

In the Stokes limit the steady problem is linear and its operator is symmetric in the face-volume inner product, so
`lmhdx.steady` solves it by CG, preconditioned by one projection step ([Solvers](solvers.md)). The residual projects
twice, so the certificate reads the same field that CG converged
({ref}`Defect correction <numerics-defect-correction>`). The default tolerance is $10^{-9}$.

With inertia the steady state of an open duct is the fixed point of one time step, found by marching or by Newton's
method ({ref}`March or Newton <open-ducts-march-newton>`). Either way, `solvax.root_solve` rebuilds the root and
differentiates it.

A converged steady state differentiates by the implicit function theorem, not through the iterations. The reverse
pass costs one more transposed solve, whatever the number of primal iterations ([Derivatives](derivatives.md)).

## Where it is checked

The drive elimination is the "Run it" example of [Numerical methods](../numerics.md). The steady Stokes profile
between plates and the open-duct balances are on [Time stepping](time.md) and
[Open ducts](../../validation/open_ducts.md).
