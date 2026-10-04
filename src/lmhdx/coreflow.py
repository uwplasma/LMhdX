"""Inertialess core-flow model of a thin-walled rectangular duct in a varying field.

At large Hartmann number and interaction parameter, inertia and viscosity are
confined to layers and the core obeys ``grad p = j x B`` and Ohm's law. For a
field ``B_y(x)`` across a duct of half-width 1 (side walls at ``z = -1, 1``) and
half-height ``a`` (Hartmann walls at ``y = -a, a``), the pressure is constant
along field lines and the three-dimensional core reduces to three functions of
two variables (Hua, Walker, Picologlou & Reed, ANL/FPP/TM-228, 1988, eqs. 4a-4c):
the core pressure ``p(x, z)``, the Hartmann-wall potential ``phi_t(x, z)`` and the
side-wall potential ``phi_s(x, y)``. With ``beta = 1/B`` and
``K = beta^2 + a^2 beta'^2 / 3``, on the quadrant ``0 <= y <= a``, ``-1 <= z <= 0``::

    d/dx(beta^2 dp/dx) + K d2p/dz2 = beta' dphi_t/dz          (4a, no normal flow at y = a)
    c_t (d2/dx2 + d2/dz2) phi_t    = a beta' dp/dz             (4b, charge in the Hartmann wall)
    c_s (d2/dx2 + d2/dy2) phi_s    = -beta dp/dx (x, -1)       (4c, charge in the side wall)

with ``phi_t = 0`` and ``dp/dz = 0`` at ``z = 0``, ``dphi_s/dy = 0`` at ``y = 0``,
the corner conditions ``phi_t(x, -1) = phi_s(x, a)`` and
``c_t dphi_t/dz = c_s dphi_s/dy`` (7d, e), and the side-layer flux closure (14)
``K dp/dz(x, -1) = beta' phi_t(x, -1) - (beta int_0^a phi_s dy)' / a``, which
says that the core and side-layer flux ``Q = -a beta^2 int dp/dx dz - beta int phi_s dy``
does not change along the duct. Walker's thin-wall limits follow in a uniform
field: the fully developed gradient is ``-dp/dx = B^2 / (1 + a/c_t + a^2/(3 c_s))``
at unit mean velocity, and ``c_t/(a + c_t)`` with perfectly conducting side walls.

The equations are the stationarity conditions of one quadratic functional,
maximal in ``p`` and minimal in the potentials::

    L = int int [-a/2 (beta^2 p_x^2 + K p_z^2) + c_t/2 |grad phi_t|^2 - a beta' p dphi_t/dz] dx dz
        + int int c_s/2 |grad phi_s|^2 dx dy - int p(x, -1) q dx,   q = a K dp/dz(x, -1)

which is why the coupled operator is symmetric and indefinite. It is discretized
directly, so the matrix is symmetric by construction. As in TM-228 section 3 the
grid is staggered in ``z``: ``phi_t`` sits on nodes from the corner to ``z = 0``,
``p`` at the cell centres between them; ``phi_s`` sits on nodes in ``y`` and the
corner node is shared, so its molecule is split between the Hartmann wall (half a
cell in ``z``) and the side wall (half a cell in ``y``) and carries (7d, e)
without further equations. Every term is a finite-volume molecule. The pressure
at the side wall is extrapolated from the first centre with the closure,
``p(-1) = p_1 - dz/2 dp/dz(-1)``, the higher-order expansion TM-228 section 3.2
asks for; in the functional this is the term ``dz/(4a) int q^2/K dx``. The ends
are fully developed (5a-f): ``p`` is given, ``dphi/dx = 0`` is natural. The
solution is then rescaled once to the imposed flow rate (6).

The pressure and potentials are solved together, never segregated: TM-228 found
a segregated scheme divergent for small wall conductance. The system is small
and two-dimensional, so it is solved directly (SuperLU on the host) inside
:func:`jax.lax.custom_linear_solve`; derivatives with respect to the wall
conductances, the field scale and the drive are exact, and the adjoint is the
same factorization. ``beta = 1/B`` is floored at ``beta_max`` (TM-228 caps it at
1000), and the number of floored nodes is returned.

The model neglects inertia: its error scales as ``N^(-1/3)`` (Mistrangelo et al.,
Fusion Eng. Des. 173, 2021), which at ALEX B2 (N = 540) is as large as the
three-dimensional excess it computes.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "CoreFlow",
    "CoreFlowResult",
    "fully_developed_gradient",
    "layer_conductances",
    "midplane_field",
    "side_layer_coefficient",
]


def fully_developed_gradient(c_t, c_s, aspect: float = 1.0):
    """Return Walker's fully developed ``-dp/dx`` at unit field and unit mean velocity."""
    return 1.0 / (1.0 + aspect / c_t + aspect**2 / (3.0 * c_s))


def midplane_field(field) -> tuple[np.ndarray, np.ndarray]:
    """Return the cell-centre ``x`` and ``B_y`` nearest ``y = 0`` of an :class:`lmhdx.core3d.ImposedField`."""
    grid = field.grid
    x, y = (np.asarray(values) for values in grid.centers[:2])
    b_y = np.asarray(field.components[1])[:, int(np.argmin(np.abs(y))), :]
    return x, b_y.mean(axis=-1)


def layer_conductances(c_t, c_s, hartmann: float, field, k, *, floor: float = 1.0):
    """Return the wall conductances with the layers' finite-Ha conductance at the local field.

    At a finite Hartmann number the Hartmann layers conduct like ``1/Ha`` of extra
    Hartmann-wall conductance and the side layers like ``k/sqrt(Ha)`` of extra
    side-wall conductance; at a station of field ``B`` the local Hartmann number is
    ``Ha |B|``, so ``c_t + 1/(Ha |B|)`` and ``c_s + k/sqrt(Ha |B|)`` (#204: with
    these the 3-D core and the core-flow model agree within 1 % at c 0.1,
    Ha 2e4). ``k`` is a number, a callable of the local Hartmann number, or a
    table ``(hartmann_numbers, k_values)`` interpolated in ``log Ha``, for example
    from :func:`side_layer_coefficient`. Where ``Ha |B|`` is below ``floor`` the
    correction is frozen at ``floor``. ``c_t``, ``c_s`` and ``field`` are numbers
    or one value per station; the result is two arrays for :meth:`CoreFlow.solve`.
    """
    local = jnp.maximum(hartmann * jnp.abs(jnp.asarray(field, dtype=jnp.result_type(float))), floor)
    if callable(k):
        coefficient = k(local)
    elif isinstance(k, tuple):
        table, values = (np.asarray(item, dtype=float) for item in k)
        coefficient = jnp.interp(jnp.log(local), jnp.asarray(np.log(table)), jnp.asarray(values))
    else:
        coefficient = k
    return c_t + 1.0 / local, c_s + coefficient / jnp.sqrt(local)


def side_layer_coefficient(c: float, hartmann: float, *, cells: int = 48) -> float:
    """Return ``k`` of the side layers in a square duct with both walls of conductance ratio ``c``.

    The fully developed gradient of LMhdX's own 2-D solve on the staggered core
    (:func:`lmhdx.core3d.duct_problem`, thin walls on both axes) is set equal to
    Walker's ``1/(1 + 1/c_t + 1/(3 c_s))`` with ``c_t = c + 1/Ha``, which is
    inverted for ``c_s = c + k/sqrt(Ha)``. One steady solve; ``cells`` must
    resolve the layers (48 to 64 cells from Ha 400 to 2e4, #204).
    """
    from .core3d import ChannelProblem, duct_problem
    from .steady import solve_steady_state

    base = duct_problem(hartmann=hartmann, cells=cells, wall_conductance=c)
    section = ChannelProblem(
        grid=base.grid,
        conditions=base.conditions,
        magnetic_field=(0.0, hartmann, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=1.0,
        wall_conductance=(0.0, c, c),
    )
    velocity = solve_steady_state(section, tolerance=1e-10, linear_max_restarts=600).velocity[0].data
    area = np.asarray(section.grid.face_areas(0)[0])
    gradient = 4.0 / float(np.sum(np.asarray(velocity)[0] * area)) / hartmann**2
    side = 1.0 / (3.0 * (1.0 / gradient - 1.0 - 1.0 / (c + 1.0 / hartmann)))
    return float((side - c) * np.sqrt(hartmann))


class CoreFlowResult(NamedTuple):
    """Core-flow solution on one quadrant, scaled to the requested mean velocity.

    ``pressure`` is ``(nx + 1, nz)`` at the ``z`` cell centres, ``phi_top`` is
    ``(nx + 1, nz + 1)`` on the ``z`` nodes (the last column is ``z = 0``) and
    ``phi_side`` is ``(nx + 1, ny + 1)`` on the ``y`` nodes (the last column is the
    corner). ``axial_flux`` is the quadrant flux on each ``x`` face and
    ``pressure_drop`` is ``p(x_first) - p(x_last)``.
    """

    pressure_drop: jax.Array
    axial_flux: jax.Array
    pressure: jax.Array
    phi_top: jax.Array
    phi_side: jax.Array
    floored_nodes: jax.Array


def _operator(rows: int, columns: int, entries) -> sp.csr_matrix:
    """Return a sparse map whose row ``m`` is ``sum coef[m] u[col[m]]`` over ``entries``; ``col < 0`` is zero."""
    index = np.arange(rows)
    r, c, v = [], [], []
    for col, coef in entries:
        col = np.broadcast_to(np.asarray(col), (rows,))
        coef = np.broadcast_to(np.asarray(coef, dtype=float), (rows,))
        keep = col >= 0
        r.append(index[keep])
        c.append(col[keep])
        v.append(coef[keep])
    return sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(rows, columns))


def _pairs(left: sp.csr_matrix, right: sp.csr_matrix):
    """Return ``(row m, column of left, column of right, product)`` for every pair sharing a row."""
    counts_left, counts_right = np.diff(left.indptr), np.diff(right.indptr)
    left_rows = np.repeat(np.arange(left.shape[0]), counts_left)
    repeat = counts_right[left_rows]
    entry = np.repeat(np.arange(left.nnz), repeat)
    offset = np.arange(repeat.sum()) - np.repeat(np.cumsum(repeat) - repeat, repeat)
    other = right.indptr[left_rows[entry]] + offset
    return left_rows[entry], left.indices[entry], right.indices[other], left.data[entry] * right.data[other]


class CoreFlow:
    """Mesh and assembled structure of the TM-228 core-flow problem on a uniform ``x`` grid.

    ``x`` holds the ``nx + 1`` uniformly spaced stations (the ends are fully
    developed), ``aspect`` is ``a`` (the Hartmann-wall half-height over the side-wall
    half-width), and ``nz`` and ``ny`` are the cells across the Hartmann and side
    walls. Everything here is host data fixed by the mesh; the field, conductances
    and drive enter :meth:`solve` as traced values.
    """

    def __init__(self, x, *, aspect: float = 1.0, nz: int = 16, ny: int = 16):
        x = np.asarray(x, dtype=float)
        h = np.diff(x)
        if x.ndim != 1 or x.size < 4 or np.any(h <= 0) or np.ptp(h) > 1e-9 * h.mean():
            raise ValueError("x must hold at least four increasing, uniformly spaced stations")
        if aspect <= 0.0 or nz < 2 or ny < 2:
            raise ValueError("aspect must be positive and nz, ny at least 2")
        self.x, self.aspect, self.nz, self.ny = x, float(aspect), int(nz), int(ny)
        self.h, self.dz, self.dy = float(h.mean()), 1.0 / nz, aspect / ny
        nx = x.size - 1
        stride = 2 * nz + ny
        self.size = (nx + 1) * stride
        i_nodes, i_faces = np.arange(nx + 1), np.arange(nx)
        self.wx = np.full(nx + 1, self.h)
        self.wx[[0, -1]] *= 0.5
        wz = np.ones(nz)
        wz[0] = 0.5
        wy = np.ones(ny + 1)
        wy[[0, -1]] = 0.5

        def p(i, j):
            return i * stride + j

        def t(i, j):
            return np.where(j >= nz, -1, i * stride + nz + np.minimum(j, nz - 1))

        def s(i, k):
            return np.where(k >= ny, t(i, 0), i * stride + 2 * nz + np.minimum(k, ny - 1))

        def grid(first, second):
            a, b = np.meshgrid(first, second, indexing="ij")
            return a.ravel(), b.ravel()

        n, dz, dy, hh = self.size, self.dz, self.dy, self.h
        fi, fj = grid(i_faces, np.arange(nz))
        ni, nj = grid(i_nodes, np.arange(nz - 1))
        ti, tj = grid(i_nodes, np.arange(nz))
        si, sk = grid(i_faces, np.arange(ny + 1))
        yi, yk = grid(i_nodes, np.arange(ny))
        # The side-wall integral of phi_s at each station, its face mean and its central slope.
        side = [(s(i_nodes, k), wy[k] * dy) for k in range(ny + 1)]
        side_face = [(s(i_faces + shift, k), 0.5 * wy[k] * dy) for shift in (0, 1) for k in range(ny + 1)]
        up, down = np.minimum(i_nodes + 1, nx), np.maximum(i_nodes - 1, 0)
        span = (up - down) * hh
        side_slope = [
            (s(end, k), sign * wy[k] * dy / span)
            for end, sign in ((up, 1), (down, -1))
            for k in range(ny + 1)
        ]
        # q = a beta' phi_t(-1) - (beta Phi_s)' = beta' (a phi_t(-1) - Phi_s) - beta Phi_s'.
        closure = _operator(nx + 1, n, [(t(i_nodes, 0), self.aspect)] + [(c, -w) for c, w in side])
        slope_of_side = _operator(nx + 1, n, side_slope)
        # Each term (left, right, weights) contributes 1/2 sum_m w_m (left u)_m (right u)_m to the functional.
        self._terms = [
            (_operator(fi.size, n, [(p(fi + 1, fj), 1 / hh), (p(fi, fj), -1 / hh)]), None, "px"),
            (_operator(ni.size, n, [(p(ni, nj + 1), 1 / dz), (p(ni, nj), -1 / dz)]), None, "pz"),
            (_operator(fi.size, n, [(t(fi + 1, fj), 1 / hh), (t(fi, fj), -1 / hh)]), None, "tx"),
            (_operator(ti.size, n, [(t(ti, tj + 1), 1 / dz), (t(ti, tj), -1 / dz)]), None, "tz"),
            (_operator(si.size, n, [(s(si + 1, sk), 1 / hh), (s(si, sk), -1 / hh)]), None, "sx"),
            (_operator(yi.size, n, [(s(yi, yk + 1), 1 / dy), (s(yi, yk), -1 / dy)]), None, "sy"),
            # -a beta' p dphi_t/dz; at the corner cell it cancels -p q's own a beta' phi_t(-1).
            (
                _operator(ti.size, n, [(p(ti, tj), 1.0)]),
                _operator(ti.size, n, [(t(ti, tj + 1), 1 / dz), (np.where(tj > 0, t(ti, tj), -1), -1 / dz)]),
                "coupling",
            ),
            # The rest of -p q: -beta dp/dx(x, -1) Phi_s on the faces.
            (
                _operator(nx, n, [(p(i_faces + 1, 0), 1 / hh), (p(i_faces, 0), -1 / hh)]),
                _operator(nx, n, side_face),
                "side",
            ),
            (closure, None, "qa"),
            (slope_of_side, None, "qb"),
            (closure, slope_of_side, "qab"),
        ]
        self._wz = wz
        self._wy = wy
        blocks = []
        for index, (left, right, _) in enumerate(self._terms):
            right = left if right is None else right
            for a, b in ((left, right), (right, left)):
                m, r, c, v = _pairs(a, b)
                blocks.append((np.full(m.size, index), m, r, c, 0.5 * v))
        term, row_m, r, c, v = (np.concatenate(parts) for parts in zip(*blocks))
        dirichlet = np.zeros(n, dtype=bool)
        dirichlet[p(0, np.arange(nz))] = dirichlet[p(nx, np.arange(nz))] = True
        free = np.flatnonzero(~dirichlet)
        position = np.full(n, -1)
        position[free] = np.arange(free.size)
        self.free = free
        self._inlet = np.zeros(n)
        self._inlet[p(0, np.arange(nz))] = 1.0
        offsets = np.cumsum([0] + [left.shape[0] for left, _, _ in self._terms])
        weight_index = offsets[term] + row_m
        keep = ~dirichlet[r]
        inner = keep & ~dirichlet[c]
        pair = position[r[inner]] * free.size + position[c[inner]]
        unique, ids = np.unique(pair, return_inverse=True)
        self._a = (weight_index[inner], v[inner], ids.ravel(), unique.size)
        self.rows, self.columns = unique // free.size, unique % free.size
        edge = keep & dirichlet[c]
        self._b = (weight_index[edge], v[edge] * self._inlet[c[edge]], position[r[edge]])
        self._layout = (nx, stride)
        self._factors: dict[bytes, object] = {}
        self._jitted = jax.jit(self._solve, static_argnums=7)

    def _weights(self, beta, slope, c_t, c_s):
        """Return the concatenated per-row weights of every energy term."""
        a, h, dz, dy, nz, ny = self.aspect, self.h, self.dz, self.dy, self.nz, self.ny
        beta_face = 0.5 * (beta[1:] + beta[:-1])
        square_face = 0.5 * (beta[1:] ** 2 + beta[:-1] ** 2)
        stiffness = beta**2 + a**2 * slope**2 / 3.0
        wx = jnp.asarray(self.wx)
        wz = jnp.asarray(self._wz)
        wy = jnp.asarray(self._wy)
        # Conductances at the stations; the x-fluxes take the mean of a face's two stations.
        c_t, c_s = (jnp.broadcast_to(jnp.asarray(c, dtype=beta.dtype), beta.shape) for c in (c_t, c_s))
        c_t_face, c_s_face = (0.5 * (c[1:] + c[:-1]) for c in (c_t, c_s))
        closure = dz * wx / (2.0 * a * stiffness)
        weights = {
            "px": jnp.repeat(-a * square_face * h * dz, nz),
            "pz": jnp.repeat(-a * stiffness * wx * dz, nz - 1),
            "tx": jnp.repeat(c_t_face, nz) * h * dz * jnp.tile(wz, beta.size - 1),
            "tz": jnp.repeat(c_t * wx * dz, nz),
            "sx": jnp.repeat(c_s_face, ny + 1) * h * dy * jnp.tile(wy, beta.size - 1),
            "sy": jnp.repeat(c_s * wx * dy, ny),
            "coupling": jnp.repeat(-2.0 * a * slope * wx * dz, nz),
            "side": -2.0 * beta_face * h,
            "qa": closure * slope**2,
            "qb": closure * beta**2,
            "qab": -2.0 * closure * slope * beta,
        }
        return jnp.concatenate([weights[name] for _, _, name in self._terms])

    def _coefficients(self, field, field_scale, beta_max):
        b = field_scale * jnp.asarray(field, dtype=jnp.result_type(float))
        beta = 1.0 / jnp.maximum(jnp.abs(b), 1.0 / beta_max)
        h = self.h
        interior = (beta[2:] - beta[:-2]) / (2.0 * h)
        first = (-3.0 * beta[0] + 4.0 * beta[1] - beta[2]) / (2.0 * h)
        last = (3.0 * beta[-1] - 4.0 * beta[-2] + beta[-3]) / (2.0 * h)
        slope = jnp.concatenate([first[None], interior, last[None]])
        return beta, slope, jnp.sum(jnp.abs(b) * beta_max < 1.0)

    def operator(self, field, *, c_t, c_s, field_scale=1.0, beta_max=1000.0) -> sp.csr_matrix:
        """Return the assembled coupled operator on the free unknowns as a host sparse matrix."""
        field = jnp.asarray(field, dtype=jnp.result_type(float))
        c_t, c_s = self._stations(c_t), self._stations(c_s)
        values = np.asarray(jax.jit(self._assemble)(field, c_t, c_s, field_scale, beta_max)[0])
        size = self.free.size
        return sp.csr_matrix((values, (self.rows, self.columns)), shape=(size, size))

    def _assemble(self, field, c_t, c_s, field_scale, beta_max):
        """Return the operator values, the inlet column of the right-hand side, ``beta`` and the floored count."""
        beta, slope, floored = self._coefficients(field, field_scale, beta_max)
        weights = self._weights(beta, slope, c_t, c_s)
        index, coef, ids, count = self._a
        values = jax.ops.segment_sum(coef * weights[index], ids, count)
        b_index, b_coef, b_rows = self._b
        inlet = jax.ops.segment_sum(b_coef * weights[b_index], b_rows, self.free.size)
        return values, inlet, beta, floored

    def solve(
        self, field, *, c_t, c_s, field_scale=1.0, drive=1.0, mean_velocity=1.0, beta_max=1000.0
    ) -> CoreFlowResult:
        """Solve for ``p``, ``phi_t`` and ``phi_s`` together; differentiable in every traced argument.

        ``c_t`` and ``c_s`` are the Hartmann- and side-wall conductance ratios: a
        number, an array with one value per station of ``x``, or a callable of
        ``x`` returning one (see :func:`layer_conductances`). A face of the ``x``
        grid conducts at the mean of its two stations, so the functional, and the
        operator, stay symmetric, and the derivatives in every conductance value
        are exact.
        ``field`` is ``B_y`` at the stations of ``x`` (for example from
        :func:`lmhdx.core3d.fringe_field` through :func:`midplane_field`, or any 1-D
        array), multiplied by ``field_scale``. The inlet pressure is ``drive`` and
        the outlet zero; unless ``mean_velocity`` is None the solution is then
        rescaled once so that the mean axial velocity is ``mean_velocity``.
        """
        field = jnp.asarray(field, dtype=jnp.result_type(float))
        if field.shape != self.x.shape:
            raise ValueError(f"the field must be sampled at the {self.x.size} stations of x")
        rescale = mean_velocity is not None
        target = mean_velocity if rescale else 1.0
        c_t, c_s = self._stations(c_t), self._stations(c_s)
        return self._jitted(field, c_t, c_s, field_scale, drive, target, beta_max, rescale)

    def _stations(self, conductance):
        """Return a conductance as one number or one value per station: a callable is evaluated at ``x``."""
        if callable(conductance):
            conductance = conductance(self.x)
        if jnp.ndim(conductance) not in (0, 1) or (
            jnp.ndim(conductance) == 1 and jnp.shape(conductance) != self.x.shape
        ):
            raise ValueError(
                f"a conductance is a number or one value at each of the {self.x.size} stations of x"
            )
        return conductance

    def _host_solve(self, values, target):
        """Solve with a SuperLU factorization, reused while the matrix values repeat (forward and adjoint)."""
        key = np.ascontiguousarray(values, dtype=np.float64).tobytes()
        if key not in self._factors:
            size = self.free.size
            matrix = sp.csc_matrix((np.frombuffer(key), (self.rows, self.columns)), shape=(size, size))
            self._factors = {key: spla.splu(matrix)}
        return self._factors[key].solve(np.asarray(target, dtype=np.float64))

    def _solve(self, field, c_t, c_s, field_scale, drive, mean_velocity, beta_max, rescale):
        values, inlet, beta, floored = self._assemble(field, c_t, c_s, field_scale, beta_max)
        size = self.free.size
        rhs = -drive * inlet
        rows, columns = jnp.asarray(self.rows), jnp.asarray(self.columns)

        def matvec(vector):
            return jax.ops.segment_sum(values * vector[columns], rows, size)

        def direct(_, target):
            shape = jax.ShapeDtypeStruct(target.shape, target.dtype)
            return jax.pure_callback(self._host_solve, shape, values, target, vmap_method="sequential")

        unknowns = jax.lax.custom_linear_solve(matvec, rhs, direct, symmetric=True)
        state = (jnp.asarray(self._inlet) * drive).at[self.free].set(unknowns)
        return self._result(state, beta, drive, mean_velocity if rescale else None, floored)

    def _result(self, state, beta, drive, mean_velocity, floored) -> CoreFlowResult:
        nx, stride = self._layout
        nz, a, h, dz = self.nz, self.aspect, self.h, self.dz
        slabs = state.reshape(nx + 1, stride)
        pressure, top, side = slabs[:, :nz], slabs[:, nz : 2 * nz], slabs[:, 2 * nz :]
        top = jnp.concatenate([top, jnp.zeros((nx + 1, 1), top.dtype)], axis=1)
        side = jnp.concatenate([side, top[:, :1]], axis=1)
        side_total = side @ jnp.asarray(self._wy * self.dy)
        side_mean = 0.5 * (side_total[1:] + side_total[:-1])
        gradient = jnp.sum(pressure[1:] - pressure[:-1], axis=1) * dz / h
        flux = (
            -a * 0.5 * (beta[1:] ** 2 + beta[:-1] ** 2) * gradient - 0.5 * (beta[1:] + beta[:-1]) * side_mean
        )
        scale = 1.0 if mean_velocity is None else a * mean_velocity / jnp.mean(flux)
        return CoreFlowResult(
            drive * scale, flux * scale, pressure * scale, top * scale, side * scale, floored
        )
