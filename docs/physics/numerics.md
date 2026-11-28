# Numerical methods

This page explains how LMhdX turns the equations of a liquid-metal duct flow into numbers: the grid, the time step,
the linear solvers, the inlet and outlet of an open duct, and how derivatives are taken. Read it when you want to know
what a solver option does, or why you can trust a result. The long tables of measured results are at the end, in
{ref}`Reference results <numerics-reference-results>`.

```{image} ../_static/verification_orders.webp
:alt: The observed order of accuracy of each term of the time step, L-infinity and RMS, near two for every term
:align: center
```

Every term of the time step converges at second order in space. The [code verification](../validation/verification.md)
page has the details.

## Run it

The drive elimination of the next section is the quickest way to see the numerics at work. One solve measures how much
flow a unit drive gives, and the drive for any flow rate follows exactly. The example takes under half a minute on a
laptop CPU.

```python
import jax

from lmhdx.core3d import duct_problem
from lmhdx.steady import channel_flow_rate, channel_flow_response, solve_steady_state

jax.config.update("jax_enable_x64", True)
problem = duct_problem(hartmann=20.0, cells=24)  # square duct, insulating walls, Stokes limit
response = channel_flow_response(problem)  # one solve at unit drive measures G
drive = float(response.drive_for(0.02))  # f = Q / G
solution = solve_steady_state(problem, forcing=(drive, 0.0, 0.0))
flow = float(channel_flow_rate(problem, solution.velocity[0].data[0]))
print(f"G = {float(response.flow_per_unit_drive):.5f}, drive {drive:.4f}, flow rate {flow:.6f}")
# G = 0.15614, drive 0.1281, flow rate 0.020000
```

LMhdX has one discretization for ducts. The staggered core (`lmhdx.grid`, `lmhdx.ops`, `lmhdx.core3d`, `lmhdx.steady`,
`lmhdx.axial`) puts velocity on cell faces and pressure and potential at cell centres. It solves periodic and open
ducts, the pipe, and every fully developed `Problem`: thin or resolved conducting walls, wall stacks, varying and
axial fields, odd meshes, and transient runs by implicit Euler. Step 4.6 removed the cell-centred fully developed
solver that LMhdX used until 1.7.

## Fully developed design: eliminating the drive

At fixed field, materials and geometry, the fully developed inductionless problem is linear in the drive. So the
volumetric flow rate is $Q=Gf$ for a single response $G$ that one solve measures. `lmhdx.steady.channel_flow_response`
uses this. The drive that delivers a requested flow rate is exactly $f=Q_{\rm target}/G$, and its derivative is
$df/dQ=1/G$.

The elimination matters for more than the solves it saves. An optimizer asked to find the drive would return a scalar
good only to its own tolerance. It would also spend its budget rediscovering a linear relation instead of exploring
the inputs that change the flow: wall conductance, aspect ratio and field strength. The tests check the analytic
derivative against automatic differentiation of the solve.

For a duct driven by a uniform pressure gradient, the drop over a length $L$ is $fL$ and the hydraulic power is $fLQ$.
That is the isothermal power of a fully developed segment. It excludes entry and exit losses, manifolds and all
thermal effects, so it is not a blanket pumping budget.

## Staggered grid and wall-resolving coordinates

`lmhdx.grid` holds the geometry of the conservative 3-D core. A `Grid` stores strictly increasing face coordinates per
axis as host-side metadata. It is hashable and JAX never traces it, so stencil bookkeeping and any eigendecomposition
happen once, at trace time. A `Field` pairs a traced array with its staggered offset. Along an axis, `CENTER` puts the
value at the cell centre and `FACE` puts it on the lower face, so a face field has one extra entry on that axis. This
is the marker-and-cell layout, in which the normal velocity already lies where a conservative face flux needs it.

High-Hartmann ducts must resolve two layers of very different thickness. The Hartmann layer scales as $a/Ha$ and the
side layer as $a/\sqrt{Ha}$. Three coordinate families are available: `uniform_faces` (unstretched, the control),
`geometric_faces` (each cell a fixed ratio larger than the last) and `tanh_faces` (clustered at both walls).
`wall_resolving_faces` starts from the requirement. Given a layer thickness, it puts a requested number of cells
inside the layer while bounding the growth ratio. If the cell count cannot meet the request it raises an error rather
than return an unresolved mesh.

LMhdX writes a wall condition once, as the ghost value that reproduces it, and `lmhdx.ops` differences every face with
the same expression. A cell-centred value lies half a cell from the wall, so a prescribed value $g$ needs
$p_{\rm ghost}=2g-p_0$ and a prescribed normal derivative $q$ needs $p_{\rm ghost}=p_0\mp q\,\Delta x_0$.

`face_gradient` maps a cell field to the faces normal to one axis, and `divergence` maps three face fields back to
cells as the net flux per unit volume. Under the cell volumes and the face weights $A_f d_f$ they are exact discrete
adjoints,

$$
\langle p,\nabla\!\cdot\mathbf u\rangle_V=-\langle\mathbf u,\nabla p\rangle_{Ad},
$$

for a wall-impermeable flux. The tests check this to a relative 1e-14 on a stretched mesh. That identity makes a
projection idempotent and stops the Lorentz force from doing spurious work in the core of a high-Hartmann duct.

Two accuracy properties are deliberate, and tests pin them. The wall flux is the two-point difference
$(p_0-g)/(\Delta x_0/2)$, first order at the wall, because that stencil keeps the assembled Laplacian symmetric and
the fluxes conservative. On a stretched mesh the interior two-point gradient is centred between cell centres, not on
the face, so its truncation error is first order in the spacing change. The solution's order there is a
manufactured-solution question, and the step that owns it verifies it.

`lmhdx.poisson` inverts that Laplacian with a direct solve. On a tensor-product grid the operator is the Kronecker sum
of three one-dimensional operators, each symmetric once you fold in the cell widths. Diagonalizing them on the host
reduces a solve to three tensor contractions and one elementwise divide. LMhdX reads the one-dimensional operators out
of `lmhdx.ops` by applying the assembled Laplacian to unit vectors, so the factorization cannot differ from the
stencil the rest of the code uses.

The direct solve matters here for three reasons. Its cost does not grow with the Hartmann number the way an iteration
count does. Its answer is exact to round-off, not to a tolerance. And the solve is a linear map, so it differentiates
without taping any iteration. A pure Neumann or fully periodic problem is singular. The solver removes the constant
from the right-hand side, and the returned field has zero volume-weighted mean. Inhomogeneous boundary data is affine,
not linear, and belongs in the right-hand side, so the solver refuses a condition that carries a value rather than
linearize it without warning. A guard rejects any axis operator that is not symmetric under the cell widths. A future
three-point wall stencil would trip it.

`lmhdx.ops` builds the electric coupling on those operators, following Ni et al. Its rule is that one face-normal
current

$$
J_{n,f}=\sigma_f\left[-\frac{\phi_N-\phi_P}{d_{PN}}
+(\mathbf u_f\times\mathbf B_f)\cdot\mathbf n_f\right]
$$

is the single source of truth. The potential equation is the divergence of exactly that flux, and `lmhdx.ops` rebuilds
the Lorentz force from the same numbers. The reason is quantitative. In the core the momentum balance is
$-\nabla p+\mathbf J\times\mathbf B=0$ to $O(Ha^{-2})$, so $Ha^2$ amplifies an $O(\Delta)$ mismatch between the two
parts of $\mathbf J$, and the mismatch shows up as a spurious core current.

The force uses Ni's face form,

$$
(\mathbf J\times\mathbf B)_c=\frac{1}{\Omega_c}\sum_f J_{n,f}\,s_f\,
(\mathbf r_f-\mathbf r_c)\times\mathbf B_f,
$$

which never forms a cell-centred current vector and samples the magnetic field on the faces, so it stays correct where
the field varies along the duct. The face conductivity is the distance-weighted harmonic mean. That mean is the series
resistance of the two half-cells, and so the right average across a fluid-wall jump. The arithmetic mean would let a
poorly conducting wall draw too much current.

The electromotive force and the Lorentz force use the same interpolation path, in opposite directions. The code
averages a velocity component from its faces to the cell centres and carries it to the current faces. It averages the
force from the current faces to the cell centres (Ni's face form above) and carries it to the velocity faces. Every
cell-to-face step is `lmhdx.ops.face_average`, the average of the piecewise-constant cell field over the control
volume straddling the face, $(h_L c_L+h_R c_R)/(h_L+h_R)$. Every face-to-cell step is its transpose,
`lmhdx.ops.face_average_adjoint`, under the cell volumes and the face weights $A_f d_f$, including the polar metric.
The force map is then exactly minus the adjoint of the electromotive map, the discrete form of
$\int\mathbf u\cdot(\mathbf J\times\mathbf B)=-\int\mathbf J\cdot(\mathbf u\times\mathbf B)$. So the Lorentz force
does exactly minus the Joule dissipation, and the steady Stokes operator is symmetric in the face-volume inner product
on a stretched mesh, to round-off. Distance-weighted interpolation is exact for a linear field where the face average
is not. On a stretched mesh, though, its transpose is not an average. With it, the steady operator was asymmetric by
1e-2 on a Ha 100 layer mesh, and the ohmic identity was off by up to 3e-3. The two interpolations coincide on uniform
cells. On the layer meshes of the validation ladder the face average moves each insulating duct flow rate towards the
spectral reference, by at most 0.3 % of it. The pipe solver of `lmhdx.pipe` uses the same pair and takes the polar
rotation of the field at the cell centres.

A thin conducting wall of conductance ratio $c=\sigma_w t_w/(\sigma a)$ is a sheet with a potential $\varphi_w$ of its
own (Walker's condition). Current crosses the half cell between the fluid and the wall,
$J_n=\sigma(\varphi_P-\varphi_w)/(h_P/2)$, and the sheet carries that current along itself,
$J_n=-c\,\sigma\nabla_\tau^2\varphi_w$, with the five-point surface Laplacian in the metric of the wall. On the
wall-normal axis the sheet is one more node, weighted by $c$ times the wall area where a cell is weighted by its
volume. The tangential operators act on it as on a cell, and the potential operator stays a Kronecker sum. The solve
is therefore the same three contractions as for an insulating wall (`lmhdx.poisson.fast_diagonal_thin_wall_poisson`
and the `wall_conductance` option of the polar factorization), with no inner Krylov iteration, and it is symmetric in
the cell volumes. The earlier choice, the adjacent cell value as the wall potential, was first order. Now a
manufactured wall potential converges at second order, the ohmic identity closes to round-off through the wall, and
the conducting pipe reaches second order under joint refinement. Where two conducting walls meet, the corner node
joins the sheets in series, so one delivers what the other receives. A rank-four Woodbury correction per mode of the
third axis removes the conduction that the Kronecker sum would otherwise give the corner edge. The half-cell current
into a sheet carries no electromotive force, so it exerts no Lorentz force either, and `lmhdx.core3d.electric_state`
closes the wall faces before it forms the force. Its product with a field tangential to the wall would be work that
the Joule dissipation does not include. With conducting side walls in a uniform field, that work left the steady
operator asymmetric by 1.1e-3 at Ha 20. Closing the faces moves the flow rate by at most 2.2e-4 of itself on 24 and
48 cells.

The imposed field may vary in space. `ChannelProblem.magnetic_field` takes three numbers, or an
`lmhdx.core3d.ImposedField` of cell-centred arrays (three arrays become one). Both paths carry each component to the
current face with the same interpolation and multiply it there. So the adjoint pairing above holds for any field, and
a constant field given as arrays reproduces the three numbers bit for bit. `lmhdx.core3d.fringe_field` builds the
fringe of the ANL benchmark (square duct, $x_0=3$). With $s=x-x_c$ and $k=\pi/(2x_0)$, the midplane profile is
$B_y=\tfrac12 B_0[1-\sin(ks)]$ over $|s|\le x_0$. As Votyakov et al. (2009) ask of a fringe model, the field also has
the companion that makes it divergence and curl free, $B_y+iB_x=\tfrac12 B_0[1-\sin k(s+iy)]$:

$$
B_x=-\tfrac12 B_0\cos(ks)\sinh(ky),\qquad
B_y=\tfrac12 B_0\left[1-\sin(ks)\cosh(ky)\right].
$$

The profile is not analytic at $s=\pm x_0$, so no harmonic field keeps it on the whole midplane. At those points
$B_x$ vanishes, which keeps the divergence continuous, and $B_y$ jumps by $\tfrac12 B_0(\cosh ky-1)$, 7.0 % of $B_0$
at $|y|=1$. Both components come from one flux function, $B_x=\partial_y A$ and $B_y=-\partial_x A$. Each face-normal
value is the mean of the field over its face, a difference of $A$. So the discrete divergence of the faces cancels to
round-off (0 on uniform cells, 2.6e-16 on a tanh mesh). The cells take the average of their two faces.

## The projection step and its two stiffnesses

`lmhdx.core3d` assembles one fractional step. It solves for the potential, the face currents of `lmhdx.ops` give the
Lorentz force, momentum advances, and a pressure Poisson solve returns the velocity to the discretely divergence-free
space.

The step treats the two stiffnesses differently, on purpose. The velocity part of the Lorentz force is a damping of
rate $\sigma B^2/\rho$, which would force $\Delta t\propto Ha^{-2}$ if left explicit. The step applies it implicitly,
as a correction to the conservative face force,

$$
H\mathbf u^{*}+\frac{\Delta t}{\rho}G p=\mathbf b,\qquad D\mathbf u^{*}=0,\qquad
H=(1+\Delta t\,\lambda)I\ \bigl(-\Delta t\,\nu\nabla^2\text{ if implicit}\bigr),
$$

with $\mathbf b$ the old velocity plus the forces. The same $H$ acts on the pressure gradient as on every other force.
So a pressure force in balance with the forcing leaves the fluid at rest to round-off, and the stage's fixed point is
the physical steady state. The pressure solves $D H^{-1} G$ by SOLVAX's constrained solve. That takes one iteration
where $H$ commutes with the gradient (periodic axes) and a few next to a wall. A time loop starts each solve from the
last step's pressure, so a steady flow takes 0 to 1 iterations in place of 13. The earlier stage scaled the increment
per component and projected it. It lost the rest state under a gradient force whenever $\lambda$ differed between
components, with a peak speed of $4.4\times10^{-2}$ at $B=10$, $\Delta t=0.1$.

The fast solve needs one shift per component, so a varying field takes the largest rate over the cells. In the
diagonal model a cell of rate $r$ then updates by $1-\Delta t\,r/(1+\Delta t\,\lambda)$, which lies inside $(0,1]$
wherever $r\le\lambda$. A smaller shift, such as the volume mean, would overshoot past $-1$ once
$\Delta t\,r>2(1+\Delta t\,\lambda)$. The steady preconditioner of `lmhdx.steady` took the peak $|\mathbf B|^2$ for
the same reason. Since plan step 2b.4, a varying field's steady preconditioner takes one much smaller rate instead
(see "The varying-field preconditioner" below). The step leaves viscosity explicit.
`ChannelProblem.diffusive_step_limit` reports its limit and does not enforce it, so a caller sweeping a parameter sees
the constraint instead of a step clipped without warning. The magnetic stiffness grows as $Ha^2$ and must go. The
viscous one depends only on the mesh and can stay until an implicit viscous solve is in place.

**What the shift costs a transient.** The shift is implicit, but the step takes the potential's share of the force,
$-\sigma\nabla\phi\times\mathbf B$, from the old velocity. Where the two balance, the Lorentz force nearly vanishes.
That happens in a quasi-two-dimensional core, or in a convection roll whose axis lies along the field, since the
roll's electromotive force is a gradient and drives no current. There the step reads
$(1+\Delta t\,\lambda)(\mathbf u^{n+1}-\mathbf u^n)=\Delta t\,\mathbf F(\mathbf u^n)$, so the motion evolves
$1+\Delta t\,\lambda$ times too slowly. The shift does not affect a steady state, but it does affect a transient.
Against the Chebyshev growth rate of `validation.thermal.convection_growth_rate`, a roll along the field at
$Pr=0.02$, $Ra=2\times10^4$ grows 5, 9 and 17 % too slowly at $\Delta t\,\lambda=0.066$, 0.13 and 0.27, and 34, 50 and
66 % too slowly at 0.66, 1.3 and 2.7. `ChannelProblem(magnetic_damping="explicit")`
(`Solver(magnetic_damping=...)` for a run in time) drops the shift and takes the whole Lorentz force from the old
velocity. That choice is exact for the balanced motion and first order for the damped motion. By the forward-Euler
bound on a damping it is stable while $\Delta t\,\lambda<2$, and the code checks that bound. The same roll then grows
at the field-free rate to $10^{-6}$ (`tests/test_core3d.py`). Steady solves and the open-duct march keep the shift.

In an open flow the shift slows a roll's drift as well as its growth, but by different amounts. Take a roll carried by
a plane Poiseuille flow in a field along its axis at $\Delta t\,\lambda=0.34$, on 18 cells a wavelength. Under the
shift it grows 26 % slower and drifts 14 % slower than in the explicit run on the same cells
(`docs/_static/data/thermal_buoyancy.json`), so its growth per unit length falls about 13 %.

The [code verification](../validation/verification.md) page verifies its time accuracy with the field on, every
term's spatial order and the braking by Hartmann walls.

The step imposes two constraints before it takes any divergence. A wall-normal velocity is zero on its wall faces,
and on a periodic axis the step makes the duplicated first and last face equal. Without either, the discrete
divergence carries a net boundary flux. With every axis periodic or Neumann the pressure cannot remove that constant,
so the projection would return a field that is still not divergence free.

The step can also take diffusion implicitly. `ChannelProblem.viscous_factorizations` factorizes
$(1+\Delta t\,\lambda)I-\Delta t\,\nu\nabla^2$ at each velocity position and folds the magnetic damping into the
shift. One solve then removes both stiff terms, and accuracy bounds the step rather than the mesh. That operator
separates exactly as the pressure Laplacian does, on the free faces of each axis. A walled face axis has its two
boundary faces prescribed, and a periodic one carries a duplicate. Solving on anything else would either invent a
value for a prescribed face or treat one face as two unknowns.

The gain grows with refinement, because the explicit limit falls as the square of the cell size. Integrating the plane
channel to the same accuracy takes 29.0 s with explicit diffusion and 1.9 s with implicit diffusion at 16 cells, and
95.1 s against 2.6 s at 32, a 16-fold and then a 36-fold reduction, for errors that agree to three digits.

Steady Stokes flow between plates has the exact profile $f(1-y^2)/(2\nu)$. The step reproduces it and converges at
second order. The measured orders are 2.01 and 2.04 over 8, 16 and 32 cells across the channel. The run omits
convective transport. That is the Stokes limit, which suits blanket interaction parameters, and this page states it
rather than implies it.

The steady residual of `lmhdx.steady` projects twice. One projection's pressure comes from the fast diagonal solve,
which is exact only to round-off. On a stretched mesh that round-off leaves about $10^{-13}$ of the removed gradient
outside the divergence-free fields. The conjugate-gradient preconditioner ends in a projection, so it cannot see that
part of a residual, and CG stalls on it. A uniform duct's force is nearly solenoidal and the leak does not matter. A
varying field's force has a large gradient part. On the four-cell varying duct of the tests the floor was
$5.0\times10^{-10}$ of the right-hand side at Ha 20 and $6.4\times10^{-7}$ at Ha 100. The second projection is one
defect correction of the pressure solve. With it the floors fall to $1.2\times10^{-13}$ and $3\times10^{-12}$, and the
certificate reads the same field that CG converged. Every varying-field duct measured, the ANL fringe included,
certifies at the default tolerance $10^{-9}$ through Ha 300, and iteration counts on uniform ducts do not change.

Above Ha 300 a floor remains. On the fringe mesh of the tests (24 cells, 16 axial) CG stalls at $1.5\times10^{-10}$ of
its right-hand side at Ha 300, $7.5\times10^{-10}$ at Ha 600 and $1.5\times10^{-8}$ at Ha 1000. At Ha 1000, round-off
in the potential solve leaves the operator asymmetric by $6\times10^{-7}$. Steps 1.9b–1.9d therefore take a tolerance
of $10^{-9}$ through Ha 300, $10^{-8}$ to Ha 600 and $10^{-7}$ to Ha 1000, with `linear_max_restarts=600` (36,000 CG
iterations) above Ha 300. The three solves take 5,581, 12,056 and 19,299 iterations.
`test_the_fringe_certifies_at_the_tolerance_rule` holds the fringe to that rule.

### The IMEX Runge-Kutta step

`advance(problem, steps, viscous=problem.viscous_factorizations(RK3), scheme=RK3)` takes the low-storage scheme of
Spalart, Moser and Rogers (1991). It runs three stages of the constrained stage above, with explicit weights
$(8/15, 5/12, 3/4)$, carried weights $(0, -17/60, -5/12)$ and Crank-Nicolson weights $(4/15, 1/15, 1/6)$ for the
damping and the viscous term. Transport and the part of the Lorentz force that is not damping are explicit, so each
stage solves $H_k\mathbf u+c_k\,\Delta t\,G p/\rho=\mathbf b_k$, $D\mathbf u=0$ exactly. A gradient force at balance
leaves the fluid at rest at every stage (peak speed under $10^{-13}$, $B_y=0$ and $10$, $\Delta t=0.1$ to
$10^{-3}$). `Euler` is the one-stage tableau.

**Stability.** For a shear mode carried by a uniform stream at $kh=\pi/2$ the RK3 amplification per step is
$|g|^2=1-C^4/12+C^6/36$, below one up to $C=\sqrt3$ (`courant_step` takes a fraction of it). Forward Euler with the
same central flux has $|g|^2=1+C^2$ and grows at every $C$. Limited transport stays bounded under both.

**Accuracy.** Against the same scheme at 64 steps:

- the Shatrov and Gerbeth manufactured flow (3-D, no slip, tanh-stretched $12^3$, $Re=100$) converges at order 2.30
  and 2.24 for RK3 (Euler 1.01, 1.18);
- an exactly damped shear mode converges at order 2.0 (Euler 1.1);
- the kinetic-energy change of an unsteady inviscid box converges at order 2.76, 2.95 and 2.99 (Euler gains energy);
- the energy budget with drive, Lorentz damping and viscosity closes at order 2.0.

The Orr-Sommerfeld mode of plane Poiseuille flow at $Re=8000$ grows at $2.6626\times10^{-3}$ on $32\times48$ cells
against the tabulated $2.66441\times10^{-3}$, and the step moves it by $9\times10^{-7}$. The implicit terms are second
order, which is the Crank-Nicolson limit. The transport alone would be third order. Gradients through a short window
pass the dot-product identity to $6\times10^{-15}$ and a Taylor-remainder test with slope 2.0.

## Running the step as one compiled trajectory

A Python loop around the projection step dispatches every operation from the host. On an accelerator that means a
round trip per step instead of a queue the device can run ahead on. That is why an early audit found one GPU slower
than a laptop CPU on a small duct. `lmhdx.core3d` compiles the whole run with `jax.lax.scan` instead. On a laptop CPU,
200 steps of a 4x16x16 duct take 3.58 s through the host loop and 0.44 s through the scan, an 8x speedup with no
accelerator involved. The 32-cell case gives the same ratio, because the scan removes per-step dispatch, not
arithmetic.

`jax.checkpoint` wraps the step body, so reverse mode keeps one state per step and recomputes each step's interior
rather than storing every intermediate. Diagnostics come out as scan outputs, so a run reports its divergence residual
and kinetic-energy history without synchronising in the middle of the trajectory. A test greps the step and loop
sources for `float(`, `bool(`, `device_get` and `.item()`, because one of those inside the loop would serialise the
queue and undo the change.

## LMhdX and SOLVAX

LMhdX owns geometry metrics and material coefficients; boundary and interface equations; MHD coupling and dimensional
scaling; charge, mass, momentum, and power residuals; and case-level convergence and validation. SOLVAX owns PCG,
GMRES/FGMRES, and fixed-point iteration; Jacobi, line, additive, deflation, and Schur preconditioners; tridiagonal and
sparse direct solves; and solver state, termination metadata, implicit linear differentiation, and checkpointed exact
reverse mode for long recurrences.

LMhdX calls these algorithms with MHD-specific operator actions, then certifies the returned state in physical units.
Solver policy stays reusable, and geometry and physics stay out of SOLVAX.

## Q2D spectral evolution

The periodic Q2D path uses full complex Fourier transforms, the two-thirds dealiasing rule for the vorticity-advection
product, and fourth-order integrating-factor Runge--Kutta time stepping. The scheme integrates the viscous and
Hartmann-friction terms exactly within each step. SOLVAX supplies the reusable periodic Poisson symbol and zero-mean
spectral inversion for the streamfunction. LMhdX owns vorticity dynamics, velocity reconstruction, the energy
identity, and physical acceptance.

LMhdX JIT compiles the largest stable integration segment. A positive `history_stride` divides a run into compiled
segments and transfers only the requested vorticity frames to the host, and zero keeps no field history. Stored
results therefore do not grow with the number of time steps.

## Inertialess core-flow model

`lmhdx.coreflow` solves the three-dimensional core of a thin-walled rectangular duct in a field $B_y(x)$ at large
Hartmann number and interaction parameter, following Hua, Walker, Picologlou and Reed (ANL/FPP/TM-228, 1988). Inertia
and viscosity act only in layers. In the core $\nabla p=\mathbf j\times\mathbf B$, so the pressure is constant along
field lines. The core then reduces to three functions of two variables on one quadrant ($0\le y\le a$,
$-1\le z\le 0$): the pressure $p(x,z)$, the Hartmann-wall potential $\varphi_t(x,z)$ and the side-wall potential
$\varphi_s(x,y)$. With $\beta=1/B$ and $K=\beta^2+a^2\beta'^2/3$, TM-228 eqs. (4a)--(4c) are

$$
\partial_x(\beta^2\partial_x p)+K\,\partial_{zz}p=\beta'\,\partial_z\varphi_t,\qquad
c_t\nabla^2\varphi_t=a\beta'\,\partial_z p,\qquad
c_s\nabla^2\varphi_s=-\beta\,\partial_x p(x,-1),
$$

with $\varphi_t=0$ and $\partial_z p=0$ at $z=0$, $\partial_y\varphi_s=0$ at $y=0$, the corner conditions
$\varphi_t(x,-1)=\varphi_s(x,a)$ and $c_t\partial_z\varphi_t=c_s\partial_y\varphi_s$ (7d, e), and the side-layer flux
closure (14),
$K\,\partial_z p(x,-1)=\beta'\varphi_t(x,-1)-a^{-1}\,\mathrm d_x\big(\beta\!\int_0^a\varphi_s\,\mathrm dy\big)$. The
closure states that the core and side-layer flux
$Q=-a\beta^2\!\int\partial_x p\,\mathrm dz-\beta\!\int\varphi_s\,\mathrm dy$ is the same at every station. In a
uniform field these give Walker's fully developed gradient $-\partial_x p=B^2/(1+a/c_t+a^2/(3c_s))$ at unit mean
velocity, and $c_t/(a+c_t)$ with perfectly conducting side walls.

The equations are the stationarity conditions of one functional, maximal in $p$ and minimal in the potentials,

$$
\mathcal L=\iint\Big[-\tfrac a2\big(\beta^2p_x^2+Kp_z^2\big)+\tfrac{c_t}2|\nabla\varphi_t|^2
-a\beta'p\,\partial_z\varphi_t\Big]\mathrm dx\,\mathrm dz
+\iint\tfrac{c_s}2|\nabla\varphi_s|^2\,\mathrm dx\,\mathrm dy-\int p(x,-1)\,q\,\mathrm dx,
$$

with $q=aK\partial_zp(x,-1)$ the flux into the side layer. LMhdX discretizes $\mathcal L$ itself, so the coupled
operator is symmetric by construction and indefinite. As in TM-228 section 3, the grid is staggered in $z$. It puts
$\varphi_t$ on nodes from the corner to $z=0$ and $p$ at the centres between them, and $\varphi_s$ on nodes in $y$.
The two walls share the corner node. Its molecule splits between half a cell of the Hartmann wall and half a cell of
the side wall, which carries (7d, e) without further equations. LMhdX extrapolates the wall pressure in (4c) with the
closure, $p(-1)=p_{1/2}-\tfrac{\Delta z}2\partial_zp(-1)$ (the higher-order expansion of TM-228 section 3.2). In
$\mathcal L$ this is the term $\tfrac{\Delta z}{4a}\int q^2/K\,\mathrm dx$. The ends are fully developed (5a--f).
There $p$ is given and $\partial_x\varphi=0$ is natural, and the solver then rescales the solution once to the imposed
flow rate. As in TM-228, the solver floors $\beta$ at 1000 and counts the floored stations.

The solver takes pressure and potentials together, never segregated. TM-228 reports that a segregated iteration
diverges for small wall conductance. The system is two-dimensional and small, so the solver factorizes it directly
(SuperLU on the host) inside `jax.lax.custom_linear_solve`. Derivatives with respect to the wall conductances, the
field scale and the drive are exact in both modes, and the adjoint reuses the factorization.

On the ANL fringe ($x_0=3$, $c_t=c_s=0.02$, $a=1$, a domain running past the fringe into the capped zero field) the
gated quantity is the three-dimensional excess of the drop between $x=-6$ and $x=2$ over the locally fully developed
drop. LMhdX computes both drops on the same mesh. The excess is 0.01791, 0.01783 and 0.01781 on stations 0.2, 0.1 and
0.05 apart with 10, 20 and 40 cells across each wall, against TM-228's $0.0932-0.0754=0.0178$ (1 %) and its Figure 10
value $0.126\,c^{1/2}=0.01782$. LMhdX reports the absolute drop but does not gate it, because TM-228's own fully
developed gradient integrates to 0.0776 over $[-6,2]$, not the quoted 0.0754. The drop is 0.0951 against TM-228's
0.0932 (2.0 %) on a domain of exactly $[-6,2]$ with fully developed ends, and 0.0954 on the longer domain. The model
neglects inertia, an error that scales as $N^{-1/3}$ (Mistrangelo et al. 2021). At ALEX B2 ($N=540$) that error is as
large as the three-dimensional excess, which is why B2 keeps a 5--10 % tolerance.

## Inlet and outlet: the non-periodic axial direction

`lmhdx.axial` gives the staggered core an inflow-outflow axis (the first), with the boundary conditions of HIMAG,
FreeMHD, GridapMHD and the 2025 six-code benchmark (plan D26):

- At the inlet the velocity is LMhdX's own fully developed profile at the inlet field, solved on the same
  cross-section and scaled to the imposed flow rate. It is array-valued Dirichlet data on the inlet face
  (`lmhdx.grid.BoundaryCondition(kind, lower=profile, upper_kind=...)`).
- At the outlet every velocity component has zero axial gradient, and $p=0$.
- At both ends no normal current flows, which means
  $\partial\varphi/\partial n=(\mathbf u\times\mathbf B)\cdot\mathbf n$. Removing the potential's mean fixes its
  gauge.
- The flow rate is exact and the pressure drop is an output. There is no extra unknown.

How the solve works:

- **The inlet is a lift.** The fully developed profile carried unchanged along the duct is discretely divergence free,
  so the solution is that lift plus a correction with no inlet flux. Upstream of any field change the lift is the
  discrete solution.
- **The operator stays symmetric.** On the correction's space the Stokes-limit operator is symmetric in the
  face-volume inner product, once the two end faces own the half cell inside the duct. The outlet velocity is then an
  unknown with a zero-gradient viscous flux, and the pressure gradient through it is the Dirichlet one. The solve is
  the preconditioned CG of a periodic duct, and it differentiates the same way.
- **The singularity test is per axis.** The pressure operator is non-singular, but its lowest mode, the long axial
  wave, lies nine decades below the largest eigenvalue across a Hartmann layer (0.04 against 7e7 at Ha 100). A test
  against that largest eigenvalue took it for a null mode.
- **One defect correction.** The same ratio costs the fast-diagonal transforms that many digits. So the pressure and
  potential solves of an open duct take one float64 defect correction, and the solver sets the computed zero
  eigenvalue of every Neumann or periodic axis to zero. Without these the true residual stalled at 2e-7 and the charge
  balance at 1e-9.

On the ANL fringe ($x_0=3$, $c=0.02$ on all four walls, $B_y$ alone, buffers of 15 half-widths upstream and 10
downstream, 0.25 axial spacing over $[-6,3]$ for 70 axial cells and over the ramp $[-3,3]$ for 60), in the Stokes
limit:

- flow rate, mass and charge hold to round-off from Ha 100 to 1600 (table in
  {ref}`Reference results <numerics-open-duct-balances-on-the-anl-fringe>`);
- doubling both buffers (to 30 and 20) moves the drop over $[-6,2]$ by 3.5e-9 of itself and the axial current through
  the window's two end faces by 3e-7 (row 28 asks for 0.5 %);
- the adjoint of the drop in the field scale matches central differences to 7e-10 on the test mesh.

Timings (fresh processes, alternating, two rounds; "build" is the host assembly including the fully developed inlet
solve):

- On the CPU floor stack (JAX 0.6.2, load 17–26), the test mesh ($23\times12\times12$, Ha 10) builds in 14.1–14.9 s,
  takes 2.7–5.8 s for its first solve and 0.18–0.21 s warm. Ha 100 on $70\times24\times24$ builds in 15.1–15.5 s,
  takes 45.8–51.6 s for its first solve and 41.5–44.1 s warm (1,444 iterations, 29 ms each).
- For one A4000 (JAX 0.10.2), the warm times and iteration counts are in the 1.9d table. The iteration count is the
  cost, and plan step 2b.4's global damping rate cuts it 1.9–3.2× (below).

### The ANL fringe on the three-dimensional core (1.9d)

`lmhdx.axial.fringe_duct`, solved in the Stokes limit, reproduces the geometry of the core-flow gate at finite
Hartmann number, with the fringe, walls and buffers above. The excess is $\Delta p-\Delta p_{FD}$ over $[-6,2]$ in
units of $\sigma U_0B_0^2a$. Here $\Delta p_{FD}$ is LMhdX's own two-dimensional fully developed gradient at the local
field (same Hartmann number, cross-section mesh and walls), integrated with 24 Gauss points over the ramp. Results
from Ha 100 to 3200 are in {ref}`Reference results <numerics-anl-fringe-excess-on-the-three-dimensional-core>`.

- **The mesh is converged.** At Ha 400, $41\times32^2$ (axial spacing 0.5), $70\times48^2$ (0.25) and $118\times64^2$
  (0.125) give 0.04461, 0.04504 and 0.04514, a change of 0.2 % at the last refinement. The Hartmann number, not the
  mesh, sets the gap to the core-flow value.
- **The excess decays slowly.** It shrinks by a constant factor of 0.78–0.80 per doubling of Ha, which is a power
  $Ha^{-0.35}$. That is not the $Ha^{-1/2}$ of the plan's fit, which does not fit these points. A free power law
  extrapolates to 0.0158–0.0168 depending on the points used, 6–11 % below the core-flow model's 0.01783 (and
  TM-228's 0.0178).
- **The three-dimensional core does not meet the 1 % gate of rows 7 and 24.** At Ha 3200 the excess is still 71 %
  above the core-flow value, and the extrapolation is not accurate to 1 %. The absolute drop falls from 0.228 to
  0.121, against TM-228's 0.0932.

#### Inertia on the open axis: the inertial share (1.9d)

`solve_open_duct` takes an advective open duct (`advection="central"` or `"limited"`) from the Stokes-limit solution
to its steady state, by marching or by Newton's method (H3, from 2026-10-26). `method="auto"` marches while the
Courant step is at most four damping steps, and takes Newton otherwise.

**The march.**

- The stepper is the RK3 scheme of `lmhdx.core3d`, with the exact constrained stage at each of three substeps,
  Crank-Nicolson damping and viscosity, central transport, and the lift carrying the inlet.
- The step is the smaller of 0.9 of the Courant limit $\sqrt3/\sum_a\max|u_a|/\Delta x_a$ and $4/\lambda_{\max}$,
  with $\lambda_{\max}=\sigma B_{\max}^2/\rho$. In a varying field every cell takes the largest damping rate in the
  implicit part and gives the rest of its own rate to the explicit one. That split is stable to
  $\lambda_{\max}\Delta t$ of 6 and not at 7 (Ha 100, Re 20).
- Fifty steps make a chunk, and `solvax.fixed_point_iteration` repeats chunks until the physical steady residual (the
  projected momentum balance) falls to `tolerance` of the lift's. If a march does not get there, the solver watches it
  for sixteen chunks more and reports it as `stagnated` (flat or too slow) or `unsteady` (the residual grows by half,
  or the change between chunks turns through sixty degrees).
- Derivatives come from the implicit function theorem on the fixed point of one step (`solvax.root_solve`), with
  tangent and transposed solves by GMRES on that step's linearization.

**Newton** takes the same residual and start. The unknown is the correction to the lift, extended off the
divergence-free fields by the identity. Each update is a flexible GMRES solve (`solvax.gcrot`, 120 per cycle,
tolerance 1e-3) preconditioned by the Stokes-limit operator's inverse, which is a conjugate-gradient solve to 0.1.
There are at most twelve updates. The march's `solvax.root_solve` rebuilds and differentiates the root, so the two
methods agree on the drop and its gradient to $10^{-6}$, and a test checks this.

**Which to use.** The table gives warm medians on the loaded laptop, on $62\times12\times12$ (Ha 20) and
$62\times16\times16$ (Ha 100). The ratio $\Delta t_C/\Delta t_D$ is the Courant step of the inlet flow over the
damping step $4/\lambda_{\max}$.

| case | Newton / march (s) | $\Delta t_C/\Delta t_D$ | march steps | Newton outer iterations |
|---|---|---|---|---|
| Ha 20, Re 40 | 15.9 / 5.35 | 1.6 | 250 | 206 |
| Ha 20, Re 20 | 5.2 / 3.4 | 3.1 | 150 | 68 |
| Ha 20, Re 10 | 1.7 / 3.9 | 6.3 | 200 | 28 |
| Ha 100, Re 20 | 33.8 / 41.6 | 76 | 1,150 | 69 |

A step that damping limits shrinks with $Ha^2$, and Newton's outer iterations grow with Re, so `auto` marches for a
ratio of 4 or less. The threshold lies between the 3.1 and the 6.3 of four measured ducts. It is fitted, not derived,
and it moves with the mesh through the Courant step.

**The outlet** owns half a cell and carries the flux $u^2$ through its outer side.

- Without it (the zero edge of a wall face) the Newton root was unstable in time. A mode localized at the outlet grew
  as $2^{\text{cells}}$ upstream, at about 10 per unit time at Ha 20, Re 20, whatever the step.
- Adding it moved the fringe drop by $3\times10^{-5}$ on the coarsest test mesh (spacing 1, Re 10), $2.5\times10^{-8}$
  at Ha 20, Re 40 and under $10^{-10}$ in the other cases.
- That mode was also Newton's near-critical one. At Ha 20, Re 40 on $62\times12\times12$, Newton took 1,352 outer
  iterations (105 s warm) without the flux and 206 (16 s) with it, to the same drop within $4\times10^{-13}$. So this
  mode caused much of the growth of the outer count with Re in the campaigns below. Those campaigns (Ha 100, Re 33;
  Ha 200, Re 40) were not re-run.

**The inertial share** is $\Delta p(Re)-\Delta p(0)$ over the excess, with $Re=Ua/\nu$ on the mean velocity and
$N=Ha^2/Re$. The Stokes-preconditioned Newton solve measured it, before the march existed, on the ANL fringe with
buffers at the core spacing (`fringe_duct(growth=1.0)`, $124\times32^2$, cell Reynolds number $\le 8$; table in
{ref}`Reference results <numerics-inertial-share-on-the-anl-fringe>`). With the default buffer growth (cell Reynolds
number 66) the Ha 100, Re 33 run stalled past Re 31.35. Transport-aware preconditioners and Krylov recycling were
measured and not adopted. Recycling took the outer iterations from 313 to 224 and from 289 to 206 at Re 20, but from
5,302 to 11,202 at Ha 100, Re 33. Those studies, the earlier $70\times32^2$ campaign and the choice of 120 per GMRES
cycle are in [numerics.md at b97ad3a,
"Inertia on the open axis: the inertial share
(1.9d)"](https://github.com/uwplasma/LMhdX/blob/b97ad3a8525a20502279961d5b4201d158d971e9/docs/physics/numerics.md#inertia-on-the-open-axis-the-inertial-share-19d).

**The fit.**

- At fixed Re the share depends little on Ha. For Ha 100 and 200 it is 0.42 and 0.40 % at Re 10, and 1.10 and
  0.98 % at Re 20.
- At fixed N the share changes. It is 0.42 % at Ha 100, Re 10 against 2.33 % at Ha 200, Re 40, both at N 1000. The
  data do not support the plan's $\beta N^{-1/3}$ term on this range (Ha 100–200, Re 10–40).
- A least-squares fit of $\ln$ share on $\ln Re$ and $\ln Ha$ over the six points gives share
  $\approx 0.045\,\% \cdot Re^{1.31} Ha^{-0.16}$. A fit in Re alone gives $0.021\,\% \cdot Re^{1.30}$. The local
  exponent between successive Re is 1.39 and 1.30 at Ha 100 and 1.30 and 1.25 at Ha 200, so it falls slowly with Re.
- The 2.3 % of #218 on the $70\times32^2$ mesh and the 2.33 % here on the $124\times32^2$ mesh agree.
- Station by station, the inertial change of $-\partial_x\bar p$ is $\le 2.3\,\%$ of the local gradient wherever the
  local $N b(x)^2 \ge 125$, and of order the gradient only where $N b^2 \lesssim 30$ (downstream of $x=1$), where the
  gradient itself is small.

#### The gap is the layers' conductance (rows 7 and 24)

The three-dimensional core and the core-flow model agree once the model carries the conductance the Hartmann and side
layers add, which TM-228's core flow leaves out. The gap above is that finite-Hartmann-number effect. At $c=0.02$ it
decays too slowly for the 1 % gate to be reachable.

**Meshes at high Ha.** Each refinement below moves the excess by under 0.1 %:

- At Ha $10^4$ (#188's run reused the Ha 3200 mesh), $70\times192^2$ gives 0.025772, against 0.02576 on
  $70\times128^2$. Halving the axial spacing ($118\times128^2$) gives 0.025783.
- At $c=0.1$, Ha $2\times10^4$, tightening the CG tolerance from $10^{-7}$ to $10^{-8}$ moves the excess by
  $1\times10^{-12}$, and $70\times256^2$ gives 0.042063 against 0.042087 on $70\times192^2$ (0.06 %).
- Ha $2\times10^4$ on $70\times192^2$ takes 25,000–26,000 CG iterations, about 100 minutes on one A4000. The flow
  rate, mass and charge hold to 2e-16, 2e-16 and 1–2e-13.

**The layers act as extra wall conductance.** LMhdX's fully developed gradient lies above Walker's thin-wall value
$1/(1+1/c_t+1/(3c_s))$ at every finite Ha. Count the Hartmann layers as a conductance $1/Ha$ added to $c_t$, and
invert Walker's formula for the side walls. That gives $c_{s,\mathrm{eff}} = c + k\,Ha^{-1/2}$, the classical
side-layer flux:

- at $c=0.02$, $k$ = 1.91, 1.46, 1.28, 1.19, 1.11 and 1.07 at Ha 400, 800, 1600, 3200, $10^4$ and $2\times10^4$;
- at $c=0.1$, $k$ = 1.40, 1.24, 1.14, 1.04, 0.83 and 0.64 at the same Ha;
- relative to the wall the added conductance is $k/(c\sqrt{Ha})$, which is 55 % at $c=0.02$, Ha $10^4$, but 8 % at
  $c=0.1$;
- the excess is sensitive to it. In the core-flow model, doubling $c_s$ from 0.02 to 0.04 raises the excess from
  0.0178 to 0.0282 (TM-228's fit is $0.126\,c^{1/2}$).

**The core-flow model at the layers' conductances.** Both conductances are inputs of `lmhdx.coreflow.CoreFlow.solve`,
so the comparison needs no code change. There are two closures, each with its own locally fully developed drop:

- The global closure takes $c_t = c + 1/Ha$ and $c_s = c_{s,\mathrm{eff}}(Ha)$, uniform along the duct. These are
  scalars, passed through the public API.
- The local closure applies the same laws at the local field, $c_t = c + 1/(Ha\,B)$ and
  $c_s = c + k(Ha\,B)/\sqrt{Ha\,B}$, with $k$ interpolated in $\log Ha$. `CoreFlow.solve` takes conductances per
  station (an array or a callable of $x$). `lmhdx.coreflow.layer_conductances(c_t, c_s, Ha, B, k)` builds them, with
  $k$ from `lmhdx.coreflow.side_layer_coefficient(c, Ha)` (one fully developed solve on the core). Where
  $Ha\,B < 1$ the closure freezes the correction (`floor`). Freezing it at $Ha\,B < 100$ or $1000$ instead moves the
  result by at most 0.6 % at $2\times10^4$ and 11 % at 3200.

The local closure at $c=0.1$, Ha $2\times10^4$ on the ANL fringe takes about 2 s. `tests/test_coreflow.py` compares it
with the stored 3-D numbers, so no 3-D solve runs.

```python
import numpy as np

from lmhdx.coreflow import CoreFlow, layer_conductances

x = np.linspace(-10.0, 6.0, 161)
field = 0.5 * (1.0 - np.sin(np.pi * np.clip(x, -3.0, 3.0) / 6.0))  # the ANL fringe, B_y(x)
k = ((100, 400, 800, 1600, 3200, 1e4, 2e4), (2.585, 1.397, 1.242, 1.138, 1.041, 0.832, 0.636))
c_t, c_s = layer_conductances(0.1, 0.1, 2e4, field, k)  # c = 0.1 at Ha 2e4, per station
pressure = np.asarray(CoreFlow(x, nz=20, ny=20).solve(field, c_t=c_t, c_s=c_s).pressure).mean(axis=1)
print(f"drop over [-6, 2]: {np.interp(-6.0, x, pressure) - np.interp(2.0, x, pressure):.4f}")
# drop over [-6, 2]: 0.4126
```

Subtracting the locally fully developed drop, 0.37050, leaves an excess of 0.0421, against the 3-D core's 0.04209.
Both closures against the 3-D core, from Ha 400 to $2\times10^4$ at both conductances, are in
{ref}`Reference results <numerics-excess-with-the-layers-conductance>`:

- **The two closures bracket the 3-D core.** They do so from Ha 800 up, at both conductances, and both tend to 1.9c's
  value (the core-flow model at $Ha\to\infty$) as Ha grows.
- **The 3-D core approaches the local closure.** At $c=0.02$ it lies 6–8 % below at Ha $10^4$ and 4–5 % below at
  $2\times10^4$. At $c=0.1$ it lies 2–4 % below at $10^4$ and inside the closure's own 0.6 % range at $2\times10^4$
  (0.04209 against 0.04184–0.04209).
- **The fully developed parts agree to 0.05 %.** The local closure's $\Delta p_{FD}$ matches the 3-D one. It is
  0.08365 against 0.08366, and 0.37050 against 0.37066 at $2\times10^4$.

**Extrapolation to TM-228's limit.** The free power law of #184 (1.9d above) is the wrong form, because the layers
enter through $k/(c\sqrt{Ha})$. Even the right form does not settle the gate:

- at $c=0.02$, fits in $Ha^{-1/2}$ and $Ha^{-1}$ through Ha 400–$2\times10^4$ give 0.0183–0.0211 (3–18 % above
  0.01783). The limit falls toward 0.01783 as the fits drop the lowest points, and the excess is still 33 % above it
  at $2\times10^4$;
- at $c=0.1$, the same fits give 0.0395–0.0405 (1–4 % above 0.03901), with residuals of 1 %. The excess falls only
  0.5 % from $10^4$ to $2\times10^4$, while the local closure falls 4 %;
- reaching TM-228's value itself to 1 % at $c=0.02$ needs the side-layer term below about 1.5 % of $c$, that is
  $c\sqrt{Ha}\gtrsim 70$ and Ha ≳ $10^7$. No three-dimensional solve reaches that, with or without 2b.4.

**Verdict for rows 7 and 24.** The study found no modelling error in the 3-D core, the core-flow model or the
buffers. The gap is a physics difference. TM-228's inertialess core flow is the $c\sqrt{Ha}\to\infty$ limit, without
the layers' $O(Ha^{-1})$ and $O(Ha^{-1/2})$ conductance. With that conductance at the local field the core-flow model
agrees with the 3-D core within 1 % at $c=0.1$, Ha $2\times10^4$ (row 24's comparison, at finite Ha), and within 5 %
at $c=0.02$.
**Final decision.** On 2026-10-04 the owner ruled that row 24 is met against the layer-corrected core-flow model
(through `lmhdx.coreflow.layer_conductances`). Row 7's 1 % gate at $c=0.02$ against 0.0178 is not met and not
reachable on the three-dimensional core. The owner closed it as a physical model difference, not a waiver.

### The ALEX pipe on an open polar axis (row 6)

Row 6 needs a measured or code-to-code reference that LMhdX's equations can reach. Mistrangelo et al. 2025 is a
magneto-convection benchmark. Buoyancy drives its observables, so a comparison would first need an energy equation and
a Boussinesq force, and LMhdX does not package its reference values. LMhdX does package the ALEX B1 pipe (Ha 6600,
N 10700, $c=0.027$, a thin conducting wall, the measured field table and the digitized excess pressure gradient with
an uncertainty of 0.002) in `lmhdx/data/benchmarks/references/alex-b1-pipe.csv`. Its equations are LMhdX's in the
Stokes limit, and the interaction parameter there is large ($N=10700$ under the magnet; the duct's inertial share at
$N=1000$ is 0.4 %, see above). So row 6 runs on the ALEX pipe.

**The open pipe** (`validation/open_pipe.py`, `fringe_pipe` and `solve_open_pipe`):

- The polar grid gets an inlet and an outlet on its third axis. The velocity is staggered: $u_r$ on radial faces,
  $u_\theta$ on azimuthal faces, $u_x$ on axial faces.
- The viscous operator is the Hessian of the discrete dissipation $\tfrac12\sum|\nabla\mathbf u|^2\,dV$ in cylindrical
  components, whose azimuthal entries carry the curvature terms $(\partial_\theta u_r-u_\theta)/r$ and
  $(\partial_\theta u_\theta+u_r)/r$.
- The solver forms the electromotive force at the cell centres in polar components of the transverse field
  $B=Ha\,b(x)$ and averages it to the faces. The Lorentz force is minus its adjoint under the charge balance, with
  Walker's thin wall through the polar fast-diagonal factorization (a sheet node per azimuthal mode, which also
  conducts axially).
- So the Stokes operator is symmetric and negative definite in the face-volume inner product by construction. The
  solve is one preconditioned CG, as in the duct, with the fully developed pipe of `lmhdx.pipe.solve_pipe` as the lift
  and component Helmholtz solves (one radial eigendecomposition per azimuthal mode) as the preconditioner. The polar
  factorization treats Neumann at both axial ends as singular, as it does a periodic axis.
- On the test mesh ($10\times12\times14$, Ha 20, $c=0.05$, CPU) the operator is symmetric to 2.5e-14, the energy
  identity (Lorentz work against Joule heat) holds to 1e-15, and charge, flow rate and mass hold to 5e-14, 1e-13 and
  1e-14. Under a uniform field the open pipe keeps the two-dimensional pipe's profile to 4e-13 and its gradient to
  4e-13. The adjoint of the drop in the field scale matches central differences to 2.8e-10.

**ALEX B1** runs in the Stokes limit on one A4000, with the field table interpolated by a monotone cubic,
$x\in[-15,10]$, and axial spacing 0.25 over $[-6,7.5]$. The observable is $-\partial_x\bar p/(\sigma U B_0^2)$ minus
its plateau over $x\ge7.5$, at the 16 tabulated stations. Meshes, station values and a sensitivity study are in
{ref}`Reference results <numerics-alex-b1-pipe-row-6>`.

- The cross-section is converged. Doubling the azimuth or going to 64 radial cells moves no station by more than 3e-5.
- Upstream, LMhdX gives 0.0266 against 0.0250 measured (inside the 0.002 uncertainty). The integrated excess is within
  3.3 % (the frozen gate is 10 %).
- Station by station the section-mean model is outside the data, with a weighted RMS of 2.4 against the frozen 1 and
  an $L_\infty$ of 5.1 against 2. Two features cause the miss, and neither changes with the mesh. Upstream of the pole
  edge the model peaks at $x=-3$ (0.0279 against 0.0220), where the axial currents of the fringe add to the local
  gradient. The measured points show no peak. Downstream (from $x=-1$ to 2) the model falls faster than the
  measurement, close to the locally fully developed $0.026\,b^2$.
- **The section mean therefore does not meet row 6 at its frozen station-wise tolerance.** It meets row 6 on the
  integrated excess and on the upstream gradient.

**The wall reads differently.** Those numbers score the area-mean pressure of each section. A pressure tap reads the
wall, and behind a fringe the pressure is not uniform across the section (the transverse differences of plan rows 24
and 27). Reading the same solutions at the wall (`validation/open_pipe.py`, `wall_pressure_gradient`, the outermost
cell, all Ha 6600, $48\times64\times76$) changes the score. The weighted RMS and $L_\infty$ are 2.43 and 5.08 for the
section mean, 1.79 and 3.82 for the wall ring average, and 3.11 and 6.60 at the walls normal to the field
($\theta=0,\pi$). At the walls parallel to the field ($\theta=\pm\pi/2$) they are 0.86 and 1.39, inside the frozen 1
and 2.

The packaged table does not record where ALEX put its taps, so this is a statement about the observable, not a tuned
parameter. The model reproduces the data if they are side-wall taps, and not if they are section means or
Hartmann-wall taps. The original report (scanned) must settle which. Explaining the shape therefore does not need the
inertia reading above. The draft Newton solve showed that inertia is about $30\times$ too small in any case.

**Sensitivity.** At Ha 1600, the interpolant, the table's 0.03 uncertainty and a half-radius shift of the ramp move
the side-wall score between 0.51 and 1.17, while the section mean's stays between 2.1 and 2.7. The upstream gradient
fixes the wall conductance. The data (0.0250 $\pm$ 0.002) admit $c$ of about 0.025 to 0.029, and $c=0.020$ or 0.035
miss it by 18 % and 39 %. None of these closes the gap for the section mean. The choice of observable does. This
study did not test a finite-thickness wall, the field's $B_x$ component or inertia, because they need `axial.py`,
which another lane holds.

### The varying-field preconditioner (2b.4)

In a varying field, one projection step preconditions the steady CG solve. Its damping is a single, global rate, much
smaller than the peak Joule rate. `lmhdx.steady._varying_field_rate` is

$$
r = (\nu\lambda_1)^{3/4}\,(\sigma|\mathbf B|^2_{\max}/\rho)^{1/4},
$$

the geometric mean of the slowest viscous rate $\nu\lambda_1$ of the mesh ($\lambda_1$ the sum of the smallest
Laplacian eigenvalues of the three axes, about $2(\pi/2)^2$ in a duct of half-width one) and the Hartmann braking rate
$\sqrt{\nu\lambda_1\,\sigma|\mathbf B|^2_{\max}/\rho}$. The rate is 33, 66 and 132 at Ha 100, 400 and 1600, against
peak rates of $10^4$ to $2.6\times10^6$.

- **Why one small, global rate.** The slowest preconditioned modes carry almost no Joule dissipation. They lie where
  the field is weak or absent, or they are odd-even along the duct, and they need less damping, not more. A damping
  that varies in space, by component or by mode reopens the Schur-complement deficit of the projection and costs more
  iterations. One global rate does not. The count is flat within 10 % over a wide band of rates, from
  $3\times10^{-4}$ to $3\times10^{-2}$ of the peak Joule rate at Ha 100 and from $2\times10^{-6}$ to $10^{-3}$ at
  Ha 1600, and the chosen rate falls inside the band each time.
- **Where it applies.** The rate applies only where the damped fallback applied, which is a varying field. Uniform
  fields keep the field-line solve and the peak rate, with the same iteration counts and fields bit for bit.
- **The gain.** On the ANL fringe duct CG iterations fall 3.2×, 2.8×, 2.1× and 1.9× at Ha 100, 400, 1600 and 3200
  (from 783 to 244, 2,047 to 728, 4,848 to 2,283, and 8,307 to 4,353; full table in
  {ref}`Reference results <numerics-varying-field-preconditioner-before-and-after>`). This meets plan step 2b.4's exit
  as the owner restated it, which asks for at least 1.9× fewer CG iterations. The count still grows, as $Ha^{0.83}$
  on the ANL meshes against $Ha^{0.68}$ before. The band's two edges are the unbraked modes (bottom) and the
  under-damped y-oscillatory modes at the coarse mid-duct cells (top), and one rate can only balance them.
- **Nothing else changes.** On the varying cases the fields agree to the solve tolerance (mean velocities to
  $10^{-13}$ relative, ANL drops to $3\times10^{-12}$), every solve certifies at the tolerance rule of #150, and the
  ANL excess is the same to five digits (0.06259, 0.04504, 0.03420, 0.03042), so rows 7 and 24 stay as #184 left
  them. Cold minus warm time, which is the compile and setup, is the same within noise.
- **Derivatives use the same CG.** `jax.value_and_grad` of the mean velocity in the field scale on the varying duct
  falls from 1.77 to 0.80 s warm at Ha 100 and from 5.98 to 2.40 s at Ha 300, and the gradients agree to $10^{-11}$
  and $5\times10^{-10}$ relative. The adjoint tests against central differences (varying duct Ha 20 and 100, the open
  duct's field-scale adjoint) pass.
- **Ha $10^4$.** Before 2b.4, `main` would need an estimated 20,000+ iterations. On the Ha 3200 mesh the drop over
  $[-6,2]$ is 0.11152, the locally fully developed one 0.08575 and the excess 0.02576, 44 % above the core-flow
  model's 0.01783. That mesh's convergence at Ha $10^4$ was not checked here. The flow rate, mass and charge hold to
  2e-16, 2e-16 and 1.2e-13. The excess continues the series, from 0.03042 to 0.02576 for a factor 3.1 in Ha. That is
  $Ha^{-0.15}$, flatter than the $Ha^{-0.35}$ above, so the 1 % gate of rows 7 and 24 is still not met.

The alternatives that lost to this rate are in full at b97ad3a. The study that led to the rate (Ritz vectors of the
slow modes, a core-flow coarse space, dampings varying in space, by component and by mode) is in
["The varying-field preconditioner
(2b.4)"](https://github.com/uwplasma/LMhdX/blob/b97ad3a8525a20502279961d5b4201d158d971e9/docs/physics/numerics.md#the-varying-field-preconditioner-2b4).
A flexible-CG Schur-complement inner solve cut iterations up to about 2.6× but lost wall time at every Hartmann
number measured. It is in ["The Schur-complement inner solve (measured, not
adopted)"](https://github.com/uwplasma/LMhdX/blob/b97ad3a8525a20502279961d5b4201d158d971e9/docs/physics/numerics.md#the-schur-complement-inner-solve-measured-not-adopted).
Field-line solves per axial station beat the global rate at no Hartmann number. They are in ["Field lines per axial
station (measured, not
adopted)"](https://github.com/uwplasma/LMhdX/blob/b97ad3a8525a20502279961d5b4201d158d971e9/docs/physics/numerics.md#field-lines-per-axial-station-measured-not-adopted).
Step 2b.4 is done with its exit restated. A coupled three-dimensional multigrid preconditioner is a separate
feasibility study, not part of 2b.4 (["Status: done (owner,
2026-10-04)"](https://github.com/uwplasma/LMhdX/blob/b97ad3a8525a20502279961d5b4201d158d971e9/docs/physics/numerics.md#status-done-owner-2026-10-04)).

## Derivative policy

The derivative algorithm is part of each numerical method. Converged linear or steady nonlinear equations use an
implicit tangent/adjoint system. The reverse pass then costs one more transposed solve, and that cost does not depend
on the number of primal iterations. Finite transient models differentiate the discrete update. Generic 3-D ducts use
an implicit electric VJP and exact checkpointed projection and outer recurrences. Q2D uses the same two-level
schedule. Its retained trajectory state is $O(N/C+C)$ for $N$ steps and width $C$, with a square-root default width.

JAX traces field arrays and continuous physical coefficients. Mesh topology, array shapes, iteration limits,
checkpoint widths, convergence strings, logging, and file output are static or host-side. Gradient acceptance combines
an independent analytical, finite-difference or transpose check with primal and adjoint residuals, compiled memory
scaling, warm runtime, and parity between CPU and GPU.

## Accuracy and performance

Analytical and manufactured tests check the observed order on refined meshes. Production claims also require stable
physical observables, stricter solver tolerances, and conservation gates. The timings measure JAX compilation and warm
execution separately. A fully developed solve assembles invariant potential and velocity coefficients and its
preconditioner once, then reuses them at each coupling step. Repeated cases also gain from the compilation cache. Keep
optional output and histories disabled when memory is the limiting resource.

(numerics-reference-results)=
## Reference results

These are the measured tables behind the sections above. The numbers are as recorded, and the hosts and stacks are
those named in each section.

(numerics-open-duct-balances-on-the-anl-fringe)=
### Open-duct balances on the ANL fringe

The numbers come from "Inlet and outlet" above, on the office host.

| Ha, cells | mesh | flow rate | mass | charge | uniform-region $\partial_xp$ | CG iterations |
|---|---|---|---|---|---|---|
| 100, 24 | 70 × 24 × 24 | 4.4e-16 | 1.0e-16 | 2.2e-13 | 3.5e-5 | 1,444 |
| 100, 32 | 60 × 32 × 32 | 2.2e-16 | 1.9e-16 | 8.7e-14 | 4.3e-5 | 774 |
| 400, 48 | 60 × 48 × 48 | 2.2e-16 | 1.9e-16 | 7.3e-14 | 5.6e-5 | 2,030 |
| 1600, 96 | 60 × 96 × 96 | 2.2e-16 | 1.2e-16 | 7.5e-14 | 6.0e-5 | 4,846 |

The flow rate is the largest error over every axial face relative to the imposed one. Mass and charge are the largest
net flux out of a cell relative to the largest gross flux through a cell (the motional flux, for charge). The gradient
is the relative difference of the mean pressure gradient over $-12<x<-8$ from the fully developed one.

(numerics-anl-fringe-excess-on-the-three-dimensional-core)=
### ANL fringe excess on the three-dimensional core

The numbers come from "The ANL fringe on the three-dimensional core (1.9d)" above, on the office host with one A4000.

| Ha | mesh | $\Delta p$ | $\Delta p_{FD}$ | excess | CG iterations | warm |
|---|---|---|---|---|---|---|
| 100 | 60 × 32² | 0.2276 | 0.1651 | 0.06257 | 774 | 6.3 s |
| 200 | 60 × 32² | 0.1815 | 0.1288 | 0.05272 | 1,748 | 10.1 s |
| 400 | 60 × 48² | 0.1559 | 0.1109 | 0.04503 | 2,030 | 21 s |
| 800 | 60 × 64² | 0.1398 | 0.1008 | 0.03893 | 2,993 | 91 s |
| 1600 | 60 × 96² | 0.1289 | 0.0947 | 0.03419 | 4,846 | 183 s |
| 3200 | 70 × 128² | 0.1209 | 0.0905 | 0.03042 | 8,321 | 820 s |

(numerics-inertial-share-on-the-anl-fringe)=
### Inertial share on the ANL fringe

The numbers come from "Inertia on the open axis" above. The runs use the ANL fringe on $124\times32^2$
(`growth=1.0`) and the Stokes-preconditioned Newton solve, with continuation from $Re=15$ in steps of 0.1 of the flow
rate. The relative residual is $\le 10^{-10}$, the mass balance $\le 1.5\times10^{-16}$ and the charge balance
$\le 1.9\times10^{-13}$. Drops are in units of $\sigma U B_0^2 a$.

| Ha | Re | N | $\Delta p$ | inertial | share | outer iterations | cold time |
|---|---|---|---|---|---|---|---|
| 100 | 0 | ∞ | 0.227655 | | | (245 CG) | |
| 100 | 10 | 1000 | 0.227919 | 0.000265 | 0.42 % | 39 | 89 s (46 s warm) |
| 100 | 20 | 500 | 0.228346 | 0.000691 | 1.10 % | 247 | 12 min |
| 100 | 33 | 303 | 0.228977 | 0.001323 | 2.11 % | 4,743 | 1.9 h |
| 200 | 0 | ∞ | 0.181530 | | | (448 CG) | |
| 200 | 10 | 4000 | 0.181739 | 0.000210 | 0.40 % | 37 | 3.3 min |
| 200 | 20 | 2000 | 0.182048 | 0.000518 | 0.98 % | 229 | 21 min |
| 200 | 40 | 1000 | 0.182760 | 0.001230 | 2.33 % | 18,932 | 21.8 h |

The counts and times are from the first campaign. The 2026-10-21 re-run with a fixed continuation (stages of 1.5 in Re
from 15) took 313 outer iterations at Ha 100, Re 20, 5,302 at Re 33, 289 at Ha 200, Re 20 and 18,932 at Re 40 (21.8 h
on a shared A4000). It gave the same shares and drops to the printed digits.

(numerics-excess-with-the-layers-conductance)=
### Excess with the layers' conductance

The numbers come from "The gap is the layers' conductance" above. Each entry is the excess over $[-6,2]$.

| Ha | $c=0.02$: 3-D core | global | local | $c=0.1$: 3-D core | global | local |
|---|---|---|---|---|---|---|
| 400 | 0.04503 | 0.04967 | 0.0895 | 0.05510 | 0.05522 | 0.0786 |
| 800 | 0.03893 | 0.03865 | 0.0639 | 0.05077 | 0.04973 | 0.0641 |
| 1600 | 0.03419 | 0.03261 | 0.0471 | 0.04751 | 0.04621 | 0.0549 |
| 3200 | 0.03042 | 0.02855 | 0.0329–0.0369 | 0.04506 | 0.04379 | 0.0492 |
| $10^4$ | 0.02577 | 0.02415 | 0.0274–0.0280 | 0.04230 | 0.04124 | 0.0433–0.0439 |
| $2\times10^4$ | 0.02375 | 0.02235 | 0.0248–0.0249 | 0.04209 | 0.04022 | 0.0418–0.0421 |
| $\infty$ | | 0.01783 | 0.01783 | | 0.03901 | 0.03901 |

(numerics-alex-b1-pipe-row-6)=
### ALEX B1 pipe (row 6)

The numbers come from "The ALEX pipe on an open polar axis" above. All meshes ran on one A4000.

| Ha | mesh $r\times\theta\times x$ | CG iterations | warm | upstream | integrated | RMS / $L_\infty$ (in uncertainties) |
|---|---|---|---|---|---|---|
| 400 | 24 × 32 × 41 (spacing 0.5) | 1,076 | 2.1 s | 0.0295 | +8.3 % | 2.51 / 4.2 |
| 1600 | 32 × 48 × 41 (0.5) | 3,281 | 23 s | 0.0272 | −0.9 % | 2.36 / 4.9 |
| 6600 | 48 × 64 × 76 | 11,245 | 156 s | 0.02664 | −3.3 % | 2.43 / 5.08 |
| 6600 | 48 × 128 × 76 | 11,698 | 617 s | 0.02668 | −3.2 % | 2.44 / 5.08 |
| 6600 | 64 × 64 × 76 (8 cells in the layer) | 11,671 | 437 s | 0.02664 | −3.3 % | 2.43 / 5.08 |
| 6600 | 48 × 64 × 140 (spacing 0.125) | 11,437 | 740 s | 0.02665 | −3.3 % | 2.44 / 5.13 |

Station values at $x=-15,-12.5,-10,-7.5,-5,-3,-2,-1,0,1,2,3,4,5,7.5,10$ (Ha 6600, $48\times64\times76$):

- model, section mean: 0.0265, 0.0267, 0.0266, 0.0265, 0.0251, 0.0279, 0.0230, 0.0159, 0.0054, 0.0023, −0.0001,
  −0.0005, −0.0002, −0.0001, 0, 0;
- model, walls parallel to the field: 0.0264, 0.0263, 0.0260, 0.0250, 0.0228, 0.0193, 0.0180, 0.0167, 0.0138, 0.0097,
  0.0059, 0.0027, 0.0009, 0.0002, 0, 0;
- measured: 0.0250, 0.0252, 0.0250, 0.0247, 0.0235, 0.0220, 0.0205, 0.0185, 0.0155, 0.0125, 0.0085, 0.0050, 0.0025,
  0.0010, 0, 0.

The next table gives the sensitivity of the station-wise shape at Ha 1600 on $32\times48\times41$, scored against the
measurement. RMS and $L_\infty$ are in uncertainties, and "upstream" is the model's mean gradient for $x<-10$, where
the measured value is 0.0250.

| variation | upstream | section mean | side wall |
|---|---|---|---|
| baseline (monotone cubic, $c=0.027$) | 0.0272 | 2.36 / 4.89 | 0.79 / 1.21 |
| linear interpolation | 0.0272 | 2.35 / 4.90 | 0.80 / 1.23 |
| cubic spline | 0.0271 | 2.35 / 4.91 | 0.77 / 1.23 |
| $b+0.03$ (the table's uncertainty) | 0.0270 | 2.41 / 4.56 | 0.63 / 1.01 |
| $b-0.03$ | 0.0256 | 2.35 / 5.19 | 1.01 / 1.77 |
| ramp shifted $+0.5$ | 0.0272 | 2.07 / 4.25 | 0.51 / 0.98 |
| ramp shifted $-0.5$ | 0.0272 | 2.71 / 5.75 | 1.17 / 2.05 |
| $c=0.020$ | 0.0205 | 2.72 / 5.47 | 2.15 / 3.68 |
| $c=0.035$ | 0.0347 | 3.72 / 6.09 | 2.50 / 4.74 |
| $c=0.054$ | 0.0520 | 8.61 / 13.47 | 7.77 / 13.43 |

The choice of interpolant (monotone cubic, linear, spline) changes nothing. The side-wall RMS stays between 0.77 and
0.80.

(numerics-varying-field-preconditioner-before-and-after)=
### Varying-field preconditioner before and after

The numbers come from "The varying-field preconditioner (2b.4)" above. They are an A/B comparison on the office host
(JAX 0.6.2, SOLVAX 0.19.0), `main` adc0340 against the 2b.4 branch, in fresh processes that alternate A and B. The
test cases ran on the CPU at load 7–11, and the ANL duct of #184 ran on one idle A4000.

| case | mesh | iterations, before to after | warm (s), before to after | cold (s), before to after |
|---|---|---|---|---|
| uniform duct, Ha 300 (field lines) | $1\times48^2$ | 44 to 44 | 0.03 to 0.03 | 2.0 to 1.9 |
| uniform duct, Ha 1000 (field lines) | $1\times48^2$ | 76 to 76 | 0.05 to 0.07 | 1.8 to 1.9 |
| varying duct of the tests, Ha 20 | $4\times24^2$ | 122 to 83 | 0.16 to 0.10 | 1.9 to 1.9 |
| varying duct, Ha 100, $c=0.05$ | $4\times24^2$ | 793 to 346 | 0.74 to 0.32 | 2.8 to 2.4 |
| varying duct, Ha 300 | $4\times24^2$ | 2,903 to 977 | 2.9 to 0.84 | 5.3 to 3.0 |
| periodic fringe of the tests, Ha 300 | $16\times24^2$ | 5,634 to 1,361 | 28 to 6.6 | 30 to 8.8 |
| periodic fringe, Ha 600 | $16\times24^2$ | 12,092 to 2,489 | 59 to 13 | 62 to 16 |
| periodic fringe, Ha 1000 | $16\times24^2$ | 19,585 to 3,874 | 95 to 19 | 99 to 22 |
| ANL fringe duct, Ha 100 (A4000) | $70\times32^2$ | 783 to 244 | 3.8 to 1.2 | 23 to 20 |
| ANL fringe duct, Ha 400 (A4000) | $70\times48^2$ | 2,047 to 728 | 24 to 8.6 | 46 to 30 |
| ANL fringe duct, Ha 1600 (A4000) | $70\times96^2$ | 4,848 to 2,283 | 239 to 111 | 265 to 137 |
| ANL fringe duct, Ha 3200 (A4000) | $70\times128^2$ | 8,307 to 4,353 | 810 to 425 | 841 to 456 |
| ANL fringe duct, Ha $10^4$ (A4000, after only) | $70\times128^2$ | 13,406 | 1,307 | 1,339 |
