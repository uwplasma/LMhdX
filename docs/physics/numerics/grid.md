(numerics-grid)=
# The staggered grid and the electric scheme

Every three-dimensional solver in LMhdX uses one finite-volume grid and one electric scheme. The periodic and open
ducts, the pipe, every fully developed `Problem` and the heat transport share them. The fast solves that invert these
operators are on [Solvers](solvers.md).

## The staggered grid

`lmhdx.grid` holds the geometry. A `Grid` stores strictly increasing face coordinates per axis as host-side metadata.
It is hashable and JAX never traces it, so stencil bookkeeping happens once, at trace time. A `Field` pairs a traced
array with its staggered offset. Along an axis, `CENTER` puts the value at the cell centre and `FACE` puts it on the
lower face. This is the marker-and-cell layout. Velocity lives on cell faces, and pressure and potential at cell
centres, so the normal velocity already lies where a conservative face flux needs it.

LMhdX writes a wall condition once, as the ghost value that reproduces it. A prescribed value $g$ needs
$p_{\rm ghost}=2g-p_0$ and a prescribed normal derivative $q$ needs $p_{\rm ghost}=p_0\mp q\,\Delta x_0$.

`face_gradient` maps a cell field to the faces normal to one axis, and `divergence` maps three face fields back to
cells as the net flux per unit volume. Under the cell volumes and the face weights $A_f d_f$ they are exact discrete
adjoints for a wall-impermeable flux,

$$
\langle p,\nabla\!\cdot\mathbf u\rangle_V=-\langle\mathbf u,\nabla p\rangle_{Ad}.
$$

The tests check this to a relative 1e-14 on a stretched mesh. The identity makes a projection idempotent and stops the
Lorentz force from doing spurious work in the core of a high-Hartmann duct.

Two accuracy properties are deliberate, and tests pin them. The wall flux is the two-point difference
$(p_0-g)/(\Delta x_0/2)$, first order at the wall, because that stencil keeps the assembled Laplacian symmetric and
the fluxes conservative. On a stretched mesh the interior two-point gradient is centred between cell centres, not on
the face, so its truncation error is first order in the spacing change. The solution's order is a
manufactured-solution question, and [code verification](../../validation/verification.md) answers it. Every term of
the time step converges at second order in space.

(numerics-wall-coordinates)=
## Wall-resolving coordinates

The Hartmann layer scales as $a/Ha$ and the side layer as $a/\sqrt{Ha}$. Three coordinate families resolve them.
`uniform_faces` is unstretched, the control. `geometric_faces` makes each cell a fixed ratio larger than the last, and
`tanh_faces` clusters cells at both walls. `wall_resolving_faces` starts from a layer thickness and puts a requested
number of cells inside the layer while it bounds the growth ratio. If the cell count cannot meet the request, it
raises an error rather than return an unresolved mesh.

(numerics-electric)=
## The current-conservative electric scheme

`lmhdx.ops` builds the electric coupling following Ni et al. One face-normal current is the single source of truth,

$$
J_{n,f}=\sigma_f\left[-\frac{\phi_N-\phi_P}{d_{PN}}
+(\mathbf u_f\times\mathbf B_f)\cdot\mathbf n_f\right].
$$

The potential equation is the divergence of exactly that flux, and the Lorentz force comes from the same numbers. In
the core the momentum balance is $-\nabla p+\mathbf J\times\mathbf B=0$ to $O(Ha^{-2})$, so $Ha^2$ amplifies an
$O(\Delta)$ mismatch between the two parts of $\mathbf J$ into a spurious core current. The force uses Ni's face form,

$$
(\mathbf J\times\mathbf B)_c=\frac{1}{\Omega_c}\sum_f J_{n,f}\,s_f\,
(\mathbf r_f-\mathbf r_c)\times\mathbf B_f.
$$

It never forms a cell-centred current vector and samples the field on the faces, so it stays correct where the field
varies along the duct.

The electromotive force and the Lorentz force use the same interpolation path in opposite directions. Every
cell-to-face step is `lmhdx.ops.face_average`, $(h_L c_L+h_R c_R)/(h_L+h_R)$, and every face-to-cell step is its
transpose `lmhdx.ops.face_average_adjoint`, including the polar metric. The force map is then exactly minus the
adjoint of the electromotive map, the discrete form of
$\int\mathbf u\cdot(\mathbf J\times\mathbf B)=-\int\mathbf J\cdot(\mathbf u\times\mathbf B)$. So the Lorentz force does
exactly minus the Joule dissipation, and the steady Stokes operator is symmetric in the face-volume inner product on a
stretched mesh, to round-off. With distance-weighted interpolation instead, the steady operator was asymmetric by 1e-2
on a Ha 100 layer mesh and the ohmic identity was off by up to 3e-3. On the layer meshes of the validation ladder the
face average moves each insulating duct flow rate towards the spectral reference, by at most 0.3 % of it. The pipe
solver of `lmhdx.pipe` uses the same pair and takes the polar rotation of the field at the cell centres.

(numerics-walls)=
## Thin and resolved conducting walls

A resolved wall is a solid region that solves the potential with its own conductivity. Potential and normal current
are continuous at the interface. The face conductivity is the distance-weighted harmonic mean, the series resistance of
the two half-cells. The arithmetic mean would let a poorly conducting wall draw too much current.

A thin wall of conductance ratio $c=\sigma_w t_w/(\sigma a)$ is a sheet with a potential $\varphi_w$ of its own
(Walker's condition). Current crosses the half cell between the fluid and the wall,
$J_n=\sigma(\varphi_P-\varphi_w)/(h_P/2)$, and the sheet carries it along itself, $J_n=-c\,\sigma\nabla_\tau^2\varphi_w$,
with the five-point surface Laplacian in the wall's metric. On the wall-normal axis the sheet is one more node,
weighted by $c$ times the wall area. The potential operator stays a Kronecker sum, so the solve is the same three
contractions as for an insulating wall (`lmhdx.poisson.fast_diagonal_thin_wall_poisson` and the `wall_conductance`
option of the polar factorization). It has no inner Krylov iteration and is symmetric in the cell volumes.

- A manufactured wall potential converges at second order, the ohmic identity closes to round-off through the wall,
  and the conducting pipe reaches second order under joint refinement.
- Where two conducting walls meet, the corner node joins the sheets in series. A rank-four Woodbury correction per
  mode of the third axis removes the conduction that the Kronecker sum would give the corner edge.
- The half-cell current into a sheet carries no electromotive force, so it exerts no Lorentz force.
  `lmhdx.core3d.electric_state` closes the wall faces before it forms the force. Without that, conducting side walls in
  a uniform field left the steady operator asymmetric by 1.1e-3 at Ha 20. Closing the faces moves the flow rate by at
  most 2.2e-4 of itself on 24 and 48 cells.

(numerics-imposed-field)=
## Imposed fields that vary

`ChannelProblem.magnetic_field` takes three numbers, or an `lmhdx.core3d.ImposedField` of cell-centred arrays. Both
paths carry each component to the current face with the same interpolation. So the adjoint pairing holds for any
field, and a constant field given as arrays reproduces the three numbers bit for bit.

`lmhdx.core3d.fringe_field` builds the fringe of the ANL benchmark (square duct, $x_0=3$). With $s=x-x_c$ and
$k=\pi/(2x_0)$, the midplane profile is $B_y=\tfrac12 B_0[1-\sin(ks)]$ over $|s|\le x_0$. As Votyakov et al. (2009)
ask, the field also has the companion that makes it divergence and curl free,

$$
B_x=-\tfrac12 B_0\cos(ks)\sinh(ky),\qquad
B_y=\tfrac12 B_0\left[1-\sin(ks)\cosh(ky)\right].
$$

At $s=\pm x_0$, $B_x$ vanishes and $B_y$ jumps by $\tfrac12 B_0(\cosh ky-1)$, 7.0 % of $B_0$ at $|y|=1$. Each
face-normal value is a difference of one flux function $A$, so the discrete divergence cancels to round-off (0 on
uniform cells, 2.6e-16 on a tanh mesh).
