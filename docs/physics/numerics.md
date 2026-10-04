# Numerical methods

LMhdX has one discretization for ducts. The staggered core (`lmhdx.grid`,
`lmhdx.ops`, `lmhdx.core3d`, `lmhdx.steady`, `lmhdx.axial`) places velocity on
faces and pressure and potential at cell centres; it solves periodic and open
ducts, the pipe, and every fully developed `CaseSpec` (`lmhdx.fully_developed`):
thin or resolved conducting walls, wall stacks, varying and axial fields, odd
meshes, and transient runs by implicit Euler. The cell-centred fully developed
solver that served those cases until 1.7 was removed in step 4.6.

## Fully developed design: eliminating the drive

At fixed field, materials and geometry the fully developed inductionless problem
is linear in the drive, so the volumetric flow rate is $Q=Gf$ for a single
response $G$ that one solve measures. `lmhdx.design` uses that directly: the drive
delivering a requested throughput is $f=Q_{\rm target}/G$ exactly, and its
derivative is $df/dQ=1/G$.

The point is not economy of solves alone. An optimizer asked to find the drive
would return a scalar good only to its own tolerance, and would spend its budget
rediscovering a linear relation instead of exploring the inputs that genuinely
change the flow: wall conductance, aspect ratio and field strength. The tests
check the analytic derivative against automatic differentiation of the solve, so
the elimination is verified rather than assumed.

For a duct driven by a uniform pressure gradient the drop over a length $L$ is
$fL$ and the hydraulic power is $fLQ$. That is the isothermal power of a fully
developed segment. It excludes entry and exit losses, manifolds and all thermal
effects, so it is not a blanket pumping budget.

## Staggered grid and wall-resolving coordinates

`lmhdx.grid` supplies the geometry the plan's conservative 3-D core is being built
on. A `Grid` stores strictly increasing face coordinates per axis as host-side
metadata: it is hashable and never traced, so stencil bookkeeping and any
eigendecomposition happen once at trace time. A `Field` pairs a traced array with
its staggered offset, where `CENTER` places the value at the cell centre along an
axis and `FACE` places it on the lower face; a face field therefore carries one
extra entry on that axis. This is the marker-and-cell layout, for which the
normal velocity already lives where a conservative face flux needs it.

High-Hartmann ducts require the mesh to resolve two very different layers: the
Hartmann layer scales as $a/Ha$ and the side layer as $a/\sqrt{Ha}$. Three
coordinate families are available. `uniform_faces` is the unstretched control,
`geometric_faces` grows successive cells by a fixed ratio, and `tanh_faces`
clusters symmetrically at both walls. `wall_resolving_faces` inverts the
requirement directly: given a layer thickness it places a requested number of
cells inside the layer while bounding the growth ratio, and raises when the cell
count cannot meet the request rather than returning an unresolved mesh.

`lmhdx.bc` expresses a wall condition once, as the ghost value that reproduces it,
and `lmhdx.ops` differences every face with the same expression. A cell-centred
value sits half a cell from the wall, so a prescribed value $g$ needs
$p_{\rm ghost}=2g-p_0$ and a prescribed normal derivative $q$ needs
$p_{\rm ghost}=p_0\mp q\,\Delta x_0$.

`face_gradient` maps a cell field to the faces normal to one axis and
`divergence` maps three face fields back to cells as the net flux per unit
volume. Under the cell volumes and the face weights $A_f d_f$ these are exact
discrete adjoints,

$$
\langle p,\nabla\!\cdot\mathbf u\rangle_V=-\langle\mathbf u,\nabla p\rangle_{Ad},
$$

for a wall-impermeable flux; the test suite checks this to a relative 1e-14 on a
stretched mesh. That identity is what makes a projection idempotent and stops the
Lorentz force doing spurious work in the core of a high-Hartmann duct.

Two accuracy properties are deliberate and pinned by test rather than left
implicit. The wall flux is the two-point difference $(p_0-g)/(\Delta x_0/2)$,
first order at the wall, because that stencil is what keeps the assembled
Laplacian symmetric and the fluxes conservative. On a stretched mesh the interior
two-point gradient is centred between cell centres rather than on the face, so
its truncation error is first order in the spacing change; solution order there
is a manufactured-solution question and is verified in the step that owns it.

`lmhdx.poisson` inverts that Laplacian directly. On a tensor-product grid the
operator is the Kronecker sum of three one-dimensional operators, each symmetric
once the cell widths are folded in, so diagonalizing them on the host reduces a
solve to three tensor contractions and one elementwise divide. The one-dimensional
operators are read out of `lmhdx.ops` by applying the assembled Laplacian to unit
vectors, so the factorization cannot drift away from the stencil the rest of the
code uses.

The consequences matter for this code in particular: the cost does not grow with
the Hartmann number the way an iteration count does, the answer is exact to
round-off instead of to a tolerance, and the solve is a linear map, so it
differentiates without taping any iteration. A pure Neumann or fully periodic
problem is singular; the constant is removed from the right-hand side and the
returned field has zero volume-weighted mean. Inhomogeneous boundary data is
affine rather than linear and belongs in the right-hand side, so a condition
carrying a value is refused instead of silently linearized. A guard rejects any
axis operator that is not symmetric under the cell widths, which is the tripwire
that a future three-point wall stencil would trip.

`lmhdx.em` builds the electric coupling on those operators, following Ni et al.
Its rule is that one face-normal current

$$
J_{n,f}=\sigma_f\left[-\frac{\phi_N-\phi_P}{d_{PN}}
+(\mathbf u_f\times\mathbf B_f)\cdot\mathbf n_f\right]
$$

is the single source of truth: the potential equation is the divergence of
exactly that flux and the Lorentz force is rebuilt from the same numbers. The
reason is quantitative. In the core the momentum balance is
$-\nabla p+\mathbf J\times\mathbf B=0$ to $O(Ha^{-2})$, so an $O(\Delta)$
inconsistency between the two parts of $\mathbf J$ is amplified by $Ha^2$ and
appears as a spurious core current.

The force uses Ni's face form,

$$
(\mathbf J\times\mathbf B)_c=\frac{1}{\Omega_c}\sum_f J_{n,f}\,s_f\,
(\mathbf r_f-\mathbf r_c)\times\mathbf B_f,
$$

which never forms a cell-centred current vector and samples the magnetic field on
the faces, so it remains correct where the field varies along the duct. Face
conductivity is the distance-weighted harmonic mean, the series resistance of the
two half-cells and therefore the right average across a fluid-wall jump; the
arithmetic mean would let a poorly conducting wall draw too much current.

The electromotive force and the Lorentz force travel the same interpolation path
in opposite directions. A velocity component is averaged from its faces to the
cell centres and carried to the current faces; the force is averaged from the
current faces to the cell centres, which is Ni's face form above, and carried to
the velocity faces. Every cell-to-face step is `lmhdx.ops.face_average`, the
average of the piecewise-constant cell field over the control volume straddling
the face, $(h_L c_L+h_R c_R)/(h_L+h_R)$, and every face-to-cell step is its
transpose `lmhdx.ops.face_average_adjoint` under the cell volumes and the face
weights $A_f d_f$, including the polar metric. The force map is then exactly
minus the adjoint of the electromotive map, the discrete form of
$\int\mathbf u\cdot(\mathbf J\times\mathbf B)=-\int\mathbf J\cdot(\mathbf u\times\mathbf B)$,
so the Lorentz force does exactly minus the Joule dissipation and the steady
Stokes operator is symmetric in the face-volume inner product on a stretched
mesh, to round-off. The distance-weighted interpolation is exact for a linear
field where the face average is not, but its transpose is not an average on a
stretched mesh: with it the steady operator was asymmetric by 1e-2 on a Ha 100
layer mesh and the ohmic identity was off by up to 3e-3. The two interpolations
coincide on uniform cells. On the layer meshes of the validation ladder the face
average moves each insulating duct flow rate towards the spectral reference, by
at most 0.3 % of it. The pipe solver of `lmhdx.pipe` uses the same pair, with the
polar rotation of the field taken at the cell centres.

A thin conducting wall of conductance ratio $c=\sigma_w t_w/(\sigma a)$ is a
sheet with a potential $\varphi_w$ of its own (Walker's condition). The fluid
reaches it across the half cell against the wall,
$J_n=\sigma(\varphi_P-\varphi_w)/(h_P/2)$, and the sheet carries that current
along itself, $J_n=-c\,\sigma\nabla_\tau^2\varphi_w$, with the five-point surface
Laplacian in the metric of the wall. On the wall-normal axis the sheet is one
more node, weighted by $c$ times the wall area where a cell is weighted by its
volume, so the tangential operators act on it as on a cell and the potential
operator stays a Kronecker sum. The solve is therefore the same three
contractions as for an insulating wall (`lmhdx.poisson.fast_diagonal_thin_wall_poisson`
and the `wall_conductance` option of the polar factorization), with no inner
Krylov iteration, and it is symmetric in the cell volumes. Taking the adjacent
cell value as the wall potential, as before, was first order: a manufactured
wall potential now converges at second order, the ohmic identity closes to
round-off through the wall, and the conducting pipe reaches second order
under joint refinement. Where two conducting walls meet, the corner node
joins the sheets in series, so one delivers what the other receives, and a
rank-four Woodbury correction per mode of the third axis removes the conduction
the Kronecker sum would otherwise give the corner edge. The half-cell current
into a sheet carries no electromotive force, so it exerts no Lorentz force
either: `lmhdx.core3d.electric_state` closes the wall faces before forming the
force. Its product with a field tangential to the wall would be work the Joule
dissipation never sees; with conducting side walls in a uniform field that left
the steady operator asymmetric by 1.1e-3 at Ha 20, and closing the faces moves
the flow rate by at most 2.2e-4 of itself on 24 and 48 cells.

The imposed field may vary in space. `ChannelProblem.magnetic_field` takes
three numbers, or an `lmhdx.core3d.ImposedField` of cell-centred arrays (three
arrays become one); both paths carry each component to the current face with the
same interpolation and multiply it there, so the adjoint pairing above holds for
any field, and a constant field given as arrays reproduces the three numbers bit
for bit. `lmhdx.core3d.fringe_field` builds the fringe of the ANL benchmark (square
duct, $x_0=3$): with $s=x-x_c$ and $k=\pi/(2x_0)$, the midplane profile
$B_y=\tfrac12 B_0[1-\sin(ks)]$ over $|s|\le x_0$, and, as Votyakov et al. (2009)
ask of a fringe model, the companion that makes it divergence and curl free,
$B_y+iB_x=\tfrac12 B_0[1-\sin k(s+iy)]$:

$$
B_x=-\tfrac12 B_0\cos(ks)\sinh(ky),\qquad
B_y=\tfrac12 B_0\left[1-\sin(ks)\cosh(ky)\right].
$$

The profile is not analytic at $s=\pm x_0$, so no harmonic field keeps it on the
whole midplane: $B_x$ vanishes there, which keeps the divergence continuous, and
$B_y$ jumps by $\tfrac12 B_0(\cosh ky-1)$, 7.0 % of $B_0$ at $|y|=1$. Both
components come from one flux function, $B_x=\partial_y A$ and $B_y=-\partial_x A$;
each face-normal value is the mean of the field over its face, a difference of
$A$, so the discrete divergence of the faces cancels to round-off (0 on uniform
cells, 2.6e-16 on a tanh mesh), and the cells take the average of their two
faces.

These modules supply geometry, operators, the scalar solve and the electric
coupling. The cell-centred fully developed solver keeps its own mesh,
`lmhdx.mesh`.

## The projection step and its two stiffnesses

`lmhdx.core3d` assembles one fractional step: the potential is solved, the face
currents of `lmhdx.em` give the Lorentz force, momentum advances, and a pressure
Poisson solve returns the velocity to the discretely divergence-free space.

The two stiffnesses are handled differently, on purpose. The velocity part of the
Lorentz force is a damping of rate $\sigma B^2/\rho$, which would force
$\Delta t\propto Ha^{-2}$ if left explicit. It is applied implicitly as a
correction to the conservative face force,

$$
\mathbf u^{*}=\mathbf u+\frac{\Delta t\,\mathbf r}{1+\Delta t\,\lambda},
$$

so the correction vanishes with the right-hand side and changes the path to a
steady state but not the steady state itself. The fast solve needs one shift per
component, so a varying field takes the largest rate over the cells; in the
diagonal model a cell of rate $r$ then updates by $1-\Delta t\,r/(1+\Delta t\,\lambda)$,
inside $(0,1]$ wherever $r\le\lambda$, where a smaller shift such as the volume
mean would overshoot past $-1$ once $\Delta t\,r>2(1+\Delta t\,\lambda)$. The
steady preconditioner of `lmhdx.steady` took the peak $|\mathbf B|^2$ for the same
reason; since plan step 2b.4 a varying field's steady preconditioner takes one
much smaller rate instead (see "The varying-field preconditioner" below). Viscosity is left explicit and its
limit is reported by `ChannelProblem.diffusive_step_limit` rather than enforced,
so a caller sweeping a parameter sees the constraint instead of a silently
clipped step. The magnetic stiffness grows as $Ha^2$ and must go; the viscous one
depends only on the mesh and can stay until an implicit viscous solve lands.

Two constraints are imposed before any divergence is taken: a wall-normal
velocity is zero on its wall faces, and on a periodic axis the duplicated first
and last face are made equal. Without either the discrete divergence carries a
net boundary flux, and with every axis periodic or Neumann the pressure cannot
remove that constant, so the projection would return a field that is still not
divergence free.

Diffusion may be taken implicitly instead. `ChannelProblem.viscous_factorizations`
factorizes $(1+\Delta t\,\lambda)I-\Delta t\,\nu\nabla^2$ at each velocity
position, folding the magnetic damping into the shift, so one solve removes both
stiff terms and the step is bounded by accuracy rather than by the mesh. That
operator separates exactly as the pressure Laplacian does, on the free faces of
each axis: a walled face axis has its two boundary faces prescribed and a
periodic one carries a duplicate, and solving on anything else would either
invent a value for a prescribed face or treat one face as two unknowns.

The gain grows with refinement, because the explicit limit falls as the square of
the cell size. Integrating the plane channel to the same accuracy takes 29.0 s
explicitly and 1.9 s implicitly at 16 cells, and 95.1 s against 2.6 s at 32, a
**16-fold and then 36-fold** reduction for errors that agree to three digits.

Steady Stokes flow between plates has the exact profile $f(1-y^2)/(2\nu)$. The
step reproduces it and **converges at second order**, measured as 2.01 and 2.04
over 8, 16 and 32 cells across the channel. Convective transport is omitted;
that is the Stokes limit, appropriate at blanket interaction parameters and
stated rather than implied.

The steady residual of `lmhdx.steady` projects twice. One projection's pressure
comes from the fast diagonal solve, exact only to round-off, and on a stretched
mesh that round-off leaves about $10^{-13}$ of the removed gradient outside the
divergence-free fields. The conjugate-gradient preconditioner ends in a
projection, so it cannot see that part of a residual, and CG stalls on it. A
uniform duct's force is nearly solenoidal and the leak does not matter; a varying
field's force has a large gradient part, and on the four-cell varying duct of the
tests the floor was $5.0\times10^{-10}$ of the right-hand side at Ha 20 and
$6.4\times10^{-7}$ at Ha 100. The second projection is one defect correction of
the pressure solve: the floors fall to $1.2\times10^{-13}$ and $3\times10^{-12}$,
and the certificate reads the same field CG converged. Every varying-field duct
measured, the ANL fringe included, certifies at the default tolerance $10^{-9}$
through Ha 300, with iteration counts on uniform ducts unchanged.

Above Ha 300 a floor remains. On the fringe mesh of the tests (24 cells, 16 axial)
CG stalls at $1.5\times10^{-10}$ of its right-hand side at Ha 300,
$7.5\times10^{-10}$ at Ha 600 and $1.5\times10^{-8}$ at Ha 1000, where round-off
in the potential solve leaves the operator asymmetric by $6\times10^{-7}$. Steps
1.9b–1.9d therefore take a tolerance of $10^{-9}$ through Ha 300, $10^{-8}$ to
Ha 600 and $10^{-7}$ to Ha 1000, with `linear_max_restarts=600` (36,000 CG
iterations) above Ha 300: the three solves take 5,581, 12,056 and 19,299.
`test_the_fringe_certifies_at_the_tolerance_rule` holds the fringe to that rule.

## Running the step as one compiled trajectory

A Python loop around the projection step dispatches every operation from the
host. On an accelerator that is the difference between a queue the device can run
ahead on and a round trip per step, and it is why the audit that opened this plan
found one GPU slower than a laptop CPU on a small duct. `lmhdx.timeloop` compiles
the whole run with `jax.lax.scan` instead. Measured on this laptop's CPU, 200
steps of a 4x16x16 duct take 3.58 s through the host loop and 0.44 s through the
scan, an **8x speedup before any accelerator is involved**; the 32-cell case gives
the same ratio, because what is removed is per-step dispatch rather than
arithmetic.

The step body is wrapped in `jax.checkpoint`, so reverse mode keeps one state per
step and recomputes each step's interior rather than storing every intermediate.
Diagnostics leave as scan outputs, so a run reports its divergence residual and
kinetic-energy history without synchronising mid-trajectory. A test greps the
step and loop sources for `float(`, `bool(`, `device_get` and `.item()`: one of
those inside the loop would serialise the queue and undo the change.

## LMhdX and SOLVAX

LMhdX owns:

- geometry metrics and material coefficients;
- boundary and interface equations;
- MHD coupling and dimensional scaling;
- charge, mass, momentum, and power residuals;
- case-level convergence and validation.

SOLVAX owns:

- PCG, GMRES/FGMRES, and fixed-point iteration;
- Jacobi, line, additive, deflation, and Schur preconditioning primitives;
- tridiagonal and sparse direct solves;
- solver state, termination metadata, implicit linear differentiation, and
  checkpointed exact reverse mode for long recurrences.

LMhdX calls these algorithms with MHD-specific operator actions and then certifies
the returned state in physical units. This keeps solver policy reusable without
moving geometry or physics into SOLVAX.

## Q2D spectral evolution

The periodic Q2D path uses full complex Fourier transforms, the two-thirds
dealiasing rule for the vorticity-advection product, and fourth-order
integrating-factor Runge--Kutta time stepping. Viscous and Hartmann-friction
terms are integrated exactly within each step. SOLVAX supplies the reusable
periodic Poisson symbol and zero-mean spectral inversion for the streamfunction;
LMhdX owns vorticity dynamics, velocity reconstruction, the energy identity, and
physical acceptance.

The largest stable integration segment is JIT compiled. A positive
`history_stride` divides a run into compiled segments and transfers only the
requested vorticity frames to the host; zero retains no field history. This
keeps primal result storage independent of the number of time steps.

## Inertialess core-flow model

`lmhdx.coreflow` solves the three-dimensional core of a thin-walled rectangular
duct in a field $B_y(x)$ at large Hartmann number and interaction parameter,
following Hua, Walker, Picologlou and Reed (ANL/FPP/TM-228, 1988). Inertia and
viscosity are confined to layers; in the core $\nabla p=\mathbf j\times\mathbf B$,
so the pressure is constant along field lines, and the core reduces to three
functions of two variables on one quadrant ($0\le y\le a$, $-1\le z\le 0$): the
pressure $p(x,z)$, the Hartmann-wall potential $\varphi_t(x,z)$ and the
side-wall potential $\varphi_s(x,y)$. With $\beta=1/B$ and
$K=\beta^2+a^2\beta'^2/3$, TM-228 eqs. (4a)--(4c) are

$$
\partial_x(\beta^2\partial_x p)+K\,\partial_{zz}p=\beta'\,\partial_z\varphi_t,\qquad
c_t\nabla^2\varphi_t=a\beta'\,\partial_z p,\qquad
c_s\nabla^2\varphi_s=-\beta\,\partial_x p(x,-1),
$$

with $\varphi_t=0$ and $\partial_z p=0$ at $z=0$, $\partial_y\varphi_s=0$ at
$y=0$, the corner conditions $\varphi_t(x,-1)=\varphi_s(x,a)$ and
$c_t\partial_z\varphi_t=c_s\partial_y\varphi_s$ (7d, e), and the side-layer flux
closure (14),
$K\,\partial_z p(x,-1)=\beta'\varphi_t(x,-1)-a^{-1}\,\mathrm d_x\big(\beta\!\int_0^a\varphi_s\,\mathrm dy\big)$,
which states that the core and side-layer flux
$Q=-a\beta^2\!\int\partial_x p\,\mathrm dz-\beta\!\int\varphi_s\,\mathrm dy$ is the
same at every station. In a uniform field these give Walker's fully developed
gradient $-\partial_x p=B^2/(1+a/c_t+a^2/(3c_s))$ at unit mean velocity, and
$c_t/(a+c_t)$ with perfectly conducting side walls.

The equations are the stationarity conditions of one functional, maximal in
$p$ and minimal in the potentials,

$$
\mathcal L=\iint\Big[-\tfrac a2\big(\beta^2p_x^2+Kp_z^2\big)+\tfrac{c_t}2|\nabla\varphi_t|^2
-a\beta'p\,\partial_z\varphi_t\Big]\mathrm dx\,\mathrm dz
+\iint\tfrac{c_s}2|\nabla\varphi_s|^2\,\mathrm dx\,\mathrm dy-\int p(x,-1)\,q\,\mathrm dx,
$$

with $q=aK\partial_zp(x,-1)$ the flux into the side layer. LMhdX discretizes
$\mathcal L$ itself, so the coupled operator is symmetric by construction and
indefinite. As in TM-228 section 3 the grid is staggered in $z$: $\varphi_t$ on
nodes from the corner to $z=0$, $p$ at the centres between them, $\varphi_s$ on
nodes in $y$. The corner node is shared, and its molecule is split between half
a cell of the Hartmann wall and half a cell of the side wall, which carries
(7d, e) without further equations. The wall pressure in (4c) is extrapolated
with the closure, $p(-1)=p_{1/2}-\tfrac{\Delta z}2\partial_zp(-1)$ (the
higher-order expansion of TM-228 section 3.2), which in $\mathcal L$ is the term
$\tfrac{\Delta z}{4a}\int q^2/K\,\mathrm dx$. The ends are fully developed
(5a--f): $p$ is given and $\partial_x\varphi=0$ is natural; the solution is then
rescaled once to the imposed flow rate. $\beta$ is floored at 1000, as in TM-228,
and the floored stations are counted.

Pressure and potentials are solved together, never segregated; TM-228 reports
a segregated iteration diverging for small wall conductance. The system is
two-dimensional and small, so it is factorized directly (SuperLU on the host)
inside `jax.lax.custom_linear_solve`: derivatives with respect to the wall
conductances, the field scale and the drive are exact in both modes, and the
adjoint reuses the factorization.

On the ANL fringe ($x_0=3$, $c_t=c_s=0.02$, $a=1$, a domain running past the
fringe into the capped zero field) the gated quantity is the three-dimensional
excess of the drop between $x=-6$ and $x=2$ over the locally fully developed
drop, both computed by LMhdX on the same mesh: 0.01791, 0.01783 and 0.01781 on
stations 0.2, 0.1 and 0.05 apart with 10, 20 and 40 cells across each wall,
against TM-228's $0.0932-0.0754=0.0178$ (1 %) and its Figure 10 value
$0.126\,c^{1/2}=0.01782$. The absolute drop is reported and not gated: 0.0951
against TM-228's 0.0932 (2.0 %) on a domain of exactly $[-6,2]$ with fully
developed ends, 0.0954 on the longer domain, because TM-228's own fully
developed gradient integrates to 0.0776 over $[-6,2]$, not the quoted 0.0754.
The model neglects inertia, an error that scales as $N^{-1/3}$
(Mistrangelo et al. 2021); at ALEX B2 ($N=540$) it is as large as the
three-dimensional excess, which is why B2 keeps a 5--10 % tolerance.

## Inlet and outlet: the non-periodic axial direction

`lmhdx.axial` gives the staggered core an inflow-outflow axis (the first),
with the boundary conditions of HIMAG, FreeMHD, GridapMHD and the 2025
six-code benchmark (plan D26). The inlet velocity is LMhdX's own fully
developed profile at the inlet field, solved on the same cross-section and
scaled to the imposed flow rate; it is array-valued Dirichlet data on the inlet
face (`lmhdx.bc.BoundaryCondition(kind, lower=profile, upper_kind=...)`). The
outlet has zero axial gradient of every velocity component and $p=0$. Neither
end carries normal current, $\partial\varphi/\partial n=(\mathbf u\times\mathbf B)\cdot\mathbf n$,
and the potential's gauge is fixed by removing its mean. The flow rate is exact
and the pressure drop is an output; there is no extra unknown.

The inlet enters as a lift. The fully developed profile carried unchanged along
the duct is discretely divergence free, so the solution is that lift plus a
correction with no inlet flux. The correction lives in a linear space on which
the Stokes-limit operator is symmetric in the face-volume inner product, once
the two end faces own the half cell inside the duct, as the outlet face must:
its velocity is an unknown with a zero-gradient viscous flux, and the pressure
gradient through it is the Dirichlet one. The solve is then the preconditioned
conjugate-gradient solve of a periodic duct and differentiates the same way.
Upstream of any field change the lift is the discrete solution.

Two numerical details make that hold at high Hartmann number. The pressure
operator is non-singular now, and its lowest mode, the long axial wave, sits
nine decades below the largest eigenvalue across a Hartmann layer (0.04
against 7e7 at Ha 100); judged against that largest eigenvalue it passed for a
null mode, so the singularity test is made per axis. The same ratio costs the
fast-diagonal transforms that many digits, so the pressure solve and the
potential solve of an open duct take one float64 defect correction, and the
computed zero eigenvalue of every Neumann or periodic axis is set to zero.
Without them the true residual stalled at 2e-7 and the charge balance at 1e-9.

Measured on the ANL fringe ($x_0=3$, $c=0.02$ on all four walls, $B_y$ alone,
buffers of 15 half-widths upstream and 10 downstream, 0.25 axial spacing over
$[-6,3]$ for 70 axial cells and over the ramp $[-3,3]$ for 60), in the Stokes
limit, on the office host:

| Ha, cells | mesh | flow rate | mass | charge | uniform-region $\partial_xp$ | CG iterations |
|---|---|---|---|---|---|---|
| 100, 24 | 70 × 24 × 24 | 4.4e-16 | 1.0e-16 | 2.2e-13 | 3.5e-5 | 1,444 |
| 100, 32 | 60 × 32 × 32 | 2.2e-16 | 1.9e-16 | 8.7e-14 | 4.3e-5 | 774 |
| 400, 48 | 60 × 48 × 48 | 2.2e-16 | 1.9e-16 | 7.3e-14 | 5.6e-5 | 2,030 |
| 1600, 96 | 60 × 96 × 96 | 2.2e-16 | 1.2e-16 | 7.5e-14 | 6.0e-5 | 4,846 |

The flow rate is the largest error over every axial face relative to the
imposed one; mass and charge are the largest net flux out of a cell relative to
the largest gross (motional, for charge) flux through a cell; the gradient is
the relative difference of the mean pressure gradient over $-12<x<-8$ from the
fully developed one. Doubling both buffers (to 30 and 20) moves the drop over
$[-6,2]$ by 3.5e-9 of itself and the axial current through the window's two end
faces by 3e-7 (row 28 asks for 0.5 %). The adjoint of the drop in the field
scale matches central differences to 7e-10 on the test mesh.

Cold and warm solves (fresh processes, alternating, two rounds; build is the
host assembly including the fully developed inlet solve): on the CPU floor stack
(JAX 0.6.2, load 17–26) the test mesh ($23\times12\times12$, Ha 10) builds in
14.1–14.9 s, solves first in 2.7–5.8 s and warm in 0.18–0.21 s; Ha 100 on
$70\times24\times24$ builds in 15.1–15.5 s, solves first in 45.8–51.6 s and warm
in 41.5–44.1 s (1,444 iterations, 29 ms each). On one A4000 (JAX 0.10.2) the
warm solve takes 6.3 s at Ha 100 ($60\times32^2$), 21 s at Ha 400
($60\times48^2$), 91 s at Ha 800 ($60\times64^2$) and 183 s at Ha 1600
($60\times96^2$). The iteration count, 774 to 4,846, is the cost; the coarse
space of plan step 2b.4 is aimed at it.

### The ANL fringe on the three-dimensional core (1.9d)

`lmhdx.axial.fringe_duct` solved in the Stokes limit reproduces the geometry of
the core-flow gate above at finite Hartmann number: $x_0=3$, $c=0.02$ on all
four walls, $B_y$ alone, buffers of 15 and 10 half-widths. The excess is
$\Delta p-\Delta p_{FD}$ over $[-6,2]$ in units of $\sigma U_0B_0^2a$, with
$\Delta p_{FD}$ LMhdX's own two-dimensional fully developed gradient at the
local field and the same Hartmann number, cross-section mesh and walls,
integrated with 24 Gauss points over the ramp. Office host, one A4000:

| Ha | mesh | $\Delta p$ | $\Delta p_{FD}$ | excess | CG iterations | warm |
|---|---|---|---|---|---|---|
| 100 | 60 × 32² | 0.2276 | 0.1651 | 0.06257 | 774 | 6.3 s |
| 200 | 60 × 32² | 0.1815 | 0.1288 | 0.05272 | 1,748 | 10.1 s |
| 400 | 60 × 48² | 0.1559 | 0.1109 | 0.04503 | 2,030 | 21 s |
| 800 | 60 × 64² | 0.1398 | 0.1008 | 0.03893 | 2,993 | 91 s |
| 1600 | 60 × 96² | 0.1289 | 0.0947 | 0.03419 | 4,846 | 183 s |
| 3200 | 70 × 128² | 0.1209 | 0.0905 | 0.03042 | 8,321 | 820 s |

Three meshes at Ha 400 ($41\times32^2$ at axial spacing 0.5, $70\times48^2$ at
0.25, $118\times64^2$ at 0.125) give 0.04461, 0.04504 and 0.04514: the mesh
moves the excess by 0.2 % at the last refinement, so the Hartmann number, not
the mesh, sets the gap to the core-flow value. Successive excesses shrink by a
constant factor 0.78–0.80 per doubling of Ha, a power $Ha^{-0.35}$, not the
$Ha^{-1/2}$ of the plan's fit, which does not fit these points. A free power
law extrapolates to 0.0158–0.0168 depending on which points it uses, 6–11 %
below the core-flow model's 0.01783 (and TM-228's 0.0178). The 1 % gate of rows
7 and 24 on the three-dimensional core is therefore **not met**: at Ha 3200 the
excess is still 71 % above the core-flow value, and the extrapolation is not
accurate to 1 %. The absolute drop falls from 0.228 to 0.121 over the same
range, against TM-228's 0.0932.

#### The gap is the layers' conductance (rows 7 and 24)

*Lane A, 2026-10-04.* The tools for this were converged meshes at Ha $10^4$ and
$2\times10^4$, a second wall conductance ($c=0.1$), and the core-flow model run at
the conductances the layers add. With them, the three-dimensional core and the
core-flow model agree. The gap above is a finite-Hartmann-number effect of the
Hartmann and side layers, which TM-228's core flow leaves out. At $c=0.02$ that
effect decays too slowly for the 1 % gate to be reachable.

**Meshes at high Ha.** The Ha $10^4$ run of #188 reused the Ha 3200 mesh. The
refinements below change the excess by less than 0.1 %:

- $70\times192^2$ gives 0.025772, against 0.02576 on $70\times128^2$;
- halving the axial spacing ($118\times128^2$) gives 0.025783;
- at $c=0.1$, Ha $2\times10^4$, tightening the CG tolerance from $10^{-7}$ to
  $10^{-8}$ moves the excess by $1\times10^{-12}$, and $70\times256^2$ gives 0.042063
  against 0.042087 on $70\times192^2$ (0.06 %).

Ha $2\times10^4$ on $70\times192^2$ takes 25,000–26,000 CG iterations, about 100
minutes on one A4000. The balances hold to round-off: flow rate 2e-16, mass
2e-16, charge 1–2e-13.

**The layers act as extra wall conductance.** LMhdX's own fully developed
gradient lies above Walker's thin-wall value $1/(1+1/c_t+1/(3c_s))$ at every
finite Ha. Count the Hartmann layers as conductance $1/Ha$ added to $c_t$, and
invert Walker's formula for the side walls. The result is
$c_{s,\mathrm{eff}} = c + k\,Ha^{-1/2}$, the classical side-layer flux:

- at $c=0.02$, $k$ = 1.91, 1.46, 1.28, 1.19, 1.11 and 1.07 at Ha 400, 800, 1600,
  3200, $10^4$ and $2\times10^4$;
- at $c=0.1$, $k$ = 1.40, 1.24, 1.14, 1.04, 0.83 and 0.64 at the same Ha.

Relative to the wall, the added conductance is $k/(c\sqrt{Ha})$: 55 % at
$c=0.02$, Ha $10^4$, but 8 % at $c=0.1$. The excess is sensitive to it. In the
core-flow model, doubling $c_s$ from 0.02 to 0.04 raises the excess from 0.0178
to 0.0282 (TM-228's fit is $0.126\,c^{1/2}$).

**The core-flow model at the layers' conductances.** Both conductances are
inputs of `lmhdx.coreflow.CoreFlow.solve`, so no code change is needed. Two
closures were run, each with its own locally fully developed drop:

- *global*: $c_t = c + 1/Ha$ and $c_s = c_{s,\mathrm{eff}}(Ha)$, uniform along
  the duct. Scalars, through the public API.
- *local*: the same laws at the local field, $c_t = c + 1/(Ha\,B)$ and
  $c_s = c + k(Ha\,B)/\sqrt{Ha\,B}$, with $k$ interpolated in $\log Ha$. This
  needs conductances that vary along $x$. `CoreFlow.solve` takes them per
  station (an array or a callable of $x$), and
  `lmhdx.coreflow.layer_conductances(c_t, c_s, Ha, B, k)` builds them, with
  $k$ from `lmhdx.coreflow.side_layer_coefficient(c, Ha)` (one fully developed
  solve on the core). Where $Ha\,B < 1$ the correction is frozen (`floor`). Freezing it at $Ha\,B < 100$ or
  $1000$ instead moves the result by at most 0.6 % at $2\times10^4$ and 11 % at 3200.

Excess over $[-6,2]$:

| Ha | $c=0.02$: 3-D core | global | local | $c=0.1$: 3-D core | global | local |
|---|---|---|---|---|---|---|
| 400 | 0.04503 | 0.04967 | 0.0895 | 0.05510 | 0.05522 | 0.0786 |
| 800 | 0.03893 | 0.03865 | 0.0639 | 0.05077 | 0.04973 | 0.0641 |
| 1600 | 0.03419 | 0.03261 | 0.0471 | 0.04751 | 0.04621 | 0.0549 |
| 3200 | 0.03042 | 0.02855 | 0.0329–0.0369 | 0.04506 | 0.04379 | 0.0492 |
| $10^4$ | 0.02577 | 0.02415 | 0.0274–0.0280 | 0.04230 | 0.04124 | 0.0433–0.0439 |
| $2\times10^4$ | 0.02375 | 0.02235 | 0.0248–0.0249 | 0.04209 | 0.04022 | 0.0418–0.0421 |
| $\infty$ | | 0.01783 | 0.01783 | | 0.03901 | 0.03901 |

What the table shows:

- **The two closures bracket the 3-D core** from Ha 800 up, at both
  conductances, and both tend to 1.9c's value as Ha grows.
- **The 3-D core approaches the local closure.** At $c=0.02$ it lies 6–8 % below
  at Ha $10^4$ and 4–5 % below at $2\times10^4$. At $c=0.1$ it lies 2–4 % below
  at $10^4$ and inside the closure's own 0.6 % range at $2\times10^4$ (0.04209
  against 0.04184–0.04209).
- **The fully developed parts agree to 0.05 %.** The local closure's
  $\Delta p_{FD}$ matches the 3-D one: 0.08365 against 0.08366, and 0.37050
  against 0.37066 at $2\times10^4$.

So at $c=0.1$, Ha $2\times10^4$, the three-dimensional core agrees with the
core-flow model to within 1 % once the model carries the layers' conductance at
the local field. That is row 24's comparison, at finite Ha.

**Extrapolation to TM-228's limit.** The free power law of #184 (fit $Ha^{-0.35}$,
limit 0.0158–0.0168) is the wrong form, because the layers enter through
$k/(c\sqrt{Ha})$. Even with the right form, the extrapolation does not settle
the gate:

- At $c=0.02$, fits in $Ha^{-1/2}$ and $Ha^{-1}$ through Ha 400–$2\times10^4$
  give 0.0183–0.0211 (3–18 % above 0.01783). The limit falls toward 0.01783 as
  the lowest points are dropped, and the excess is still 33 % above it at
  $2\times10^4$.
- At $c=0.1$, the same fits give 0.0395–0.0405 (1–4 % above 0.03901), with
  residuals of 1 %. The excess falls only 0.5 % from $10^4$ to $2\times10^4$,
  while the local closure falls 4 %.

Reaching TM-228's value itself to 1 % at $c=0.02$ needs the side-layer term
below about 1.5 % of $c$: $c\sqrt{Ha}\gtrsim 70$, Ha ≳ $10^7$. No
three-dimensional solve reaches that, with or without 2b.4.

The comparison at $c=0.1$, Ha $2\times10^4$ is a test
(`tests/test_coreflow.py`, from the stored 3-D numbers, so no 3-D solve runs):

```python
from lmhdx.coreflow import CoreFlow, layer_conductances

k = ((100, 400, 800, 1600, 3200, 1e4, 2e4), (2.585, 1.397, 1.242, 1.138, 1.041, 0.832, 0.636))
c_t, c_s = layer_conductances(0.1, 0.1, 2e4, field, k)   # field: B_y at the stations of x
result = CoreFlow(x, nz=20, ny=20).solve(field, c_t=c_t, c_s=c_s)
```

**Verdict for rows 7 and 24.**

- No modelling error was found in the three-dimensional core, the core-flow
  model or the buffers. The meshes are converged, the balances hold to
  round-off, and the fully developed parts agree to 0.05 %.
- The gap is a physics difference. TM-228's inertialess core flow is the
  $c\sqrt{Ha}\to\infty$ limit, without the layers' $O(Ha^{-1})$ and
  $O(Ha^{-1/2})$ conductance.
- Row 7's 1 % gate at $c=0.02$ against 0.0178 is **not met and not reachable on
  the three-dimensional core**.
- At finite Ha, the core-flow model with the layers' conductance at the local
  field agrees with the 3-D core within 1 % at $c=0.1$, Ha $2\times10^4$, and
  within 5 % at $c=0.02$.
- Restating rows 7 and 24 as that comparison is the plan owner's decision.

### The varying-field preconditioner (2b.4)

Plan step 2b.4 asked for a two-level preconditioner whose coarse space is the
core-flow model of 1.9c, on the premise (2b.0) that the continuum of small
preconditioned eigenvalues is the core-flow space. Ritz vectors from CG's own
Lanczos coefficients on the fringe duct (Ha 100, $31\times24^2$, 700 steps)
say otherwise. The slowest modes carry almost no Joule dissipation: their
preconditioned eigenvalue equals their viscous energy over the peak $\sigma B^2$.
They lie where the field is weak or absent (the field-free outlet buffer and the
end of the ramp) or are odd-even along the duct, a pattern the face averages of
the electromotive force cancel. They are not y-invariant core flows: their
velocity peaks mid-duct in $y$, and a uniform imposed field on the same open duct
needs as many iterations (1,304 against 1,383).

The core-flow coarse space was built and measured anyway: a streamfunction
$\psi(x,z)$ on every axial and spanwise face, times a Hartmann or a parabolic
profile in $y$, projected, 713 to 1,922 vectors, the coarse matrix probed with
the real operator (an exact Galerkin matrix). Added to the damped preconditioner
it takes Ha 100 from 1,383 to 1,316 iterations (balanced, BNN, 1,294); added to
the preconditioner below it takes Ha 400 from 780 to 723–768, for a setup of
one operator application per coarse vector. It is not adopted.

What the slow modes need is less damping, and the projection sets how it may be
reduced. A damping that varies in space (the local $\sigma B^2(x)$, alone, with a
matching weighted projection, or as a partition of unity between two rates), by
component (the field-line solve of the uniform duct on one or two components,
extended to the open axis) or by mode (an anisotropic Joule symbol, or the
electromotive average's symbol) measured 1.1 to 4.2 times *more* iterations on
the Ha 100 fringe duct (1,474 to 5,784 against 1,383): each reopens the
Schur-complement deficit of the projection, which moves the top of the spectrum
up (from 34 to 2,124 with the local rate). One global rate does not; at that rate
the anisotropic symbol is no better than the isotropic damping (511 against 455). The iteration count is flat within 10 % over a wide band of rates, from
$3\times10^{-4}$ to $3\times10^{-2}$ of the peak Joule rate at Ha 100 and from
$2\times10^{-6}$ to $10^{-3}$ at Ha 1600, and rises outside it. The rate chosen,
`lmhdx.steady._varying_field_rate`, is

$$
r = (\nu\lambda_1)^{3/4}\,(\sigma|\mathbf B|^2_{\max}/\rho)^{1/4},
$$

the geometric mean of the slowest viscous rate $\nu\lambda_1$ of the mesh
($\lambda_1$ the sum of the smallest Laplacian eigenvalues of the three axes,
about $2(\pi/2)^2$ in a duct of half-width one) and the Hartmann braking rate
$\sqrt{\nu\lambda_1\,\sigma|\mathbf B|^2_{\max}/\rho}$: 33, 66 and 132 at Ha 100,
400 and 1600 against peak rates of $10^4$ to $2.6\times10^6$, inside the band
each time. It applies only where the damped fallback applied, a varying field;
uniform fields keep the field-line solve and the peak rate.

A/B on the office host (JAX 0.6.2, SOLVAX 0.19.0, `main` adc0340 against this
branch, fresh processes alternating A/B; test cases on the CPU at load 7–11, the
ANL duct of #184 on one idle A4000):

| case | mesh | iterations before → after | warm (s) before → after | cold (s) before → after |
|---|---|---|---|---|
| uniform duct, Ha 300 (field lines) | $1\times48^2$ | 44 → 44 | 0.03 → 0.03 | 2.0 → 1.9 |
| uniform duct, Ha 1000 (field lines) | $1\times48^2$ | 76 → 76 | 0.05 → 0.07 | 1.8 → 1.9 |
| varying duct of the tests, Ha 20 | $4\times24^2$ | 122 → 83 | 0.16 → 0.10 | 1.9 → 1.9 |
| varying duct, Ha 100, $c=0.05$ | $4\times24^2$ | 793 → 346 | 0.74 → 0.32 | 2.8 → 2.4 |
| varying duct, Ha 300 | $4\times24^2$ | 2,903 → 977 | 2.9 → 0.84 | 5.3 → 3.0 |
| periodic fringe of the tests, Ha 300 | $16\times24^2$ | 5,634 → 1,361 | 28 → 6.6 | 30 → 8.8 |
| periodic fringe, Ha 600 | $16\times24^2$ | 12,092 → 2,489 | 59 → 13 | 62 → 16 |
| periodic fringe, Ha 1000 | $16\times24^2$ | 19,585 → 3,874 | 95 → 19 | 99 → 22 |
| ANL fringe duct, Ha 100 (A4000) | $70\times32^2$ | 783 → 244 | 3.8 → 1.2 | 23 → 20 |
| ANL fringe duct, Ha 400 (A4000) | $70\times48^2$ | 2,047 → 728 | 24 → 8.6 | 46 → 30 |
| ANL fringe duct, Ha 1600 (A4000) | $70\times96^2$ | 4,848 → 2,283 | 239 → 111 | 265 → 137 |
| ANL fringe duct, Ha 3200 (A4000) | $70\times128^2$ | 8,307 → 4,353 | 810 → 425 | 841 → 456 |
| ANL fringe duct, Ha $10^4$ (A4000, after only) | $70\times128^2$ | → 13,406 | → 1,307 | → 1,339 |

At Ha $10^4$ (this branch only; `main` would need an estimated 20,000+ iterations)
the drop over $[-6,2]$ is 0.11152, the locally fully developed one 0.08575 and the
excess 0.02576, 44 % above the core-flow model's 0.01783. The mesh is the Ha 3200
one and its convergence at Ha $10^4$ was not checked; the balances hold (flow rate
2e-16, mass 2e-16, charge 1.2e-13). The excess continues the series
(0.03042 → 0.02576 for a factor 3.1 in Ha, $Ha^{-0.15}$, flatter than the
$Ha^{-0.35}$ below), so the 1 % gate of rows 7 and 24 is still not met.

Uniform fields are untouched: same preconditioner, same iteration counts and
fields bit for bit. On the varying cases the fields agree to the solve tolerance
(mean velocities to $10^{-13}$ relative, ANL drops to $3\times10^{-12}$), every
solve certifies at the tolerance rule of #150, and the ANL excess is unchanged to
five digits (0.06259, 0.04504, 0.03420, 0.03042), so rows 7 and 24 stand as
#184 left them. Cold minus warm, the compile and setup, is unchanged within
noise. Derivatives take the same CG: `jax.value_and_grad` of the mean velocity
in the field scale on the varying duct falls from 1.77 to 0.80 s warm at Ha 100
and from 5.98 to 2.40 s at Ha 300, gradients equal to $10^{-11}$ and
$5\times10^{-10}$ relative; the adjoint tests against central differences
(varying duct Ha 20 and 100, the open duct's field-scale adjoint) pass.

The count still grows with the Hartmann number, as $Ha^{0.83}$ on the ANL meshes
against $Ha^{0.68}$ before, so the gain falls from 3.2× at Ha 100 to 1.9× at Ha 3200, because the band's two edges are the unbraked modes (bottom)
and the under-damped y-oscillatory modes at the coarse mid-duct cells (top), and
one rate can only balance them. Removing either edge needs a damping that is
not global, which the projection forbids in this form; a preconditioner that
solves the Schur complement (an inner iteration, so flexible CG) is the
remaining route and is not attempted here.

#### The Schur-complement inner solve (measured, not adopted)

The remaining route was built and measured as a prototype (outside the package):
outer CG made flexible (Notay's truncated FCG(1), `beta = -(z, q)/(p, q)`, one
extra inner product per step), and a preconditioner that runs $k$ steps of PCG
on a model Stokes operator, $\mathbb P(\nu\nabla^2 - D)\mathbb P$ on the
constrained divergence-free fields, preconditioned by the projection step
above. The inner iteration solves the Schur complement of the projection, so
the model's damping $D$ may vary in space without the deficit that made a
varying damping lose. Three models: the global rate above, the peak rate, and
the local Lorentz rate on each face component,
$D_i = \alpha\,\sigma(|\mathbf B|^2 - B_i^2)/\rho$ averaged to the faces.

What it shows:

- The Schur deficit of the projection is not what limits CG. Solving it exactly
  for the global rate changes little: ANL Ha 100 244 → 229 iterations, Ha 400
  728 → 806, periodic fringe Ha 300 1,361 → 937–951 ($k$ 4 and 16); the peak
  rate gives 1,390.
- A varying damping now helps, but only scaled well below the Joule rate.
  At $\alpha=1$ the local model over-brakes the core flows, which the true
  operator brakes only through the Hartmann layers (an $O(Ha)$, not
  $O(Ha^2)$, rate), and solving it more exactly makes CG worse (periodic fringe
  1,182 at $k=4$, 8,068 at $k=16$). At $\alpha=0.1$ the iterations fall 1.5–2.6×.
- The gain saturates near 2.6× (2.5× at Ha 1600, 2.6× at 3200 and $10^4$) and
  does not pay for the inner steps. More inner steps buy almost nothing
  (Ha 1600: 921 at $k=4$, 867 at $k=8$), so the model, not the inner accuracy,
  sets the count; one outer step with $k=4$ costs about
  3× a plain one.

A/B on the office host, one A4000 (JAX 0.10.2, `main` 1ae695f, fresh processes;
the ANL duct of #184 with the tolerance rule of #150; warm times; local model
$\alpha=0.1$, one projection inside the model, unless noted):

| case | mesh | plain CG: iterations, warm | Schur, $k$: iterations, warm |
|---|---|---|---|
| periodic fringe Ha 300 (CPU, JAX 0.6.2) | $16\times24^2$ | 1,361, 6.7 s | global $k$ 4: 937, 18.1 s; local $k$ 8 (two projections): 861, 26.6 s |
| ANL Ha 100 | $70\times32^2$ | 244, 1.2 s | global $k$ 4: 229, 3.5 s; local $k$ 8 (two projections): 160, 3.9 s |
| ANL Ha 400 | $70\times48^2$ | 728, 8.6 s | $k$ 2: 551, 13.3 s; $k$ 4: 450, 16.0 s; $k$ 8 (two projections): 358, 25.8 s |
| ANL Ha 1600 | $70\times96^2$ | 2,283, 111 s | $k$ 4: 921, 142 s; $k$ 8: 867, 218 s |
| ANL Ha 3200 | $70\times128^2$ | 4,353, 424 s | $k$ 4: 1,655, 492 s |
| ANL Ha $10^4$ (cold, one solve) | $70\times128^2$ | 13,406, 1,339 s (#188) | $k$ 4: 5,212, 1,581 s |

Every variant reproduces the plain solve's drop within the solve tolerance
($4\times10^{-10}$ relative at Ha 3200). It loses wall time at every Hartmann
number measured: by 1.16× at Ha 3200 and 1.18× at Ha $10^4$, the closest.
It is not adopted, and the code is not kept; the uniform-field path was never
touched. (The Ha $10^4$ row compares cold single solves on the two cards of the
host, the plain one from #188.) The model's count grows almost as fast as the
plain one, as $Ha^{0.76}$ at $k=4$ against $Ha^{0.83}$, so the gain stays near
2.6× and an inner solve would have to cost less than 1.6 plain steps to win.
A preconditioner that wins here has to brake the core as the Hartmann layers do
(the field-line form of the uniform duct, along $y$ at the local field), not by a
local rate; on the open axial direction that needs the field-line solve per axial
station rather than in the axial eigenbasis, and is left open.

## Derivative policy

The derivative algorithm is part of each numerical method. Converged linear or
steady nonlinear equations use an implicit tangent/adjoint system, so reverse
cost is one additional transposed solve and does not depend on the number of
primal iterations. Finite transient models differentiate the discrete update.
Generic 3-D ducts use an implicit electric VJP and exact checkpointed
projection and outer recurrences. Q2D uses the same two-level schedule;
retained trajectory state is $O(N/C+C)$ for $N$ steps and width $C$, with a
square-root default.

Field arrays and continuous physical coefficients are traced. Mesh topology,
array shapes, iteration limits, checkpoint widths, convergence strings,
logging, and file output are static or host-side. Gradient acceptance combines
an independent analytical/finite-difference/transpose check with primal and
adjoint residuals, compiled memory scaling, warm runtime, and CPU/GPU parity.

## Accuracy and performance

Analytical and manufactured tests check observed order on refined meshes.
Production claims additionally require stable physical observables, stricter
solver tolerances, and conservation gates. JAX compilation and warm execution
are measured separately. A fully developed solve assembles invariant potential
and velocity coefficients and its preconditioner once, then reuses them at each
coupling step. Repeated cases also benefit from the compilation cache; optional
output and histories should remain disabled when memory is the limiting resource.
