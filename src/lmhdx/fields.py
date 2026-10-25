"""Imposed magnetic fields of a :class:`lmhdx.problem.Problem`: four small pytrees.

Each field samples itself on a grid with ``sample(grid)``, which returns what the
kernels take (three numbers, or an :class:`lmhdx.core3d.ImposedField`) and a
``scale`` that multiplies it. The scale is the one channel the kernels
differentiate a field through (``field_scale`` of :func:`lmhdx.steady.solve_steady_state`),
so a field's continuous leaf is its amplitude: ``Uniform.strength``,
``Fringe.strength``, ``Table.scale``. Under :func:`jax.grad` the field is sampled
at the primal value and ``scale`` carries the derivative, which is exact because the
force is linear in the amplitude. Sampling needs a concrete primal, so differentiate
with :func:`jax.grad` or :func:`jax.value_and_grad`, or close over the field when
using :func:`jax.jit`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

try:
    from jax.extend.core import concrete_or_error
except ImportError:  # pragma: no cover - older JAX keeps it in jax.core
    from jax.core import concrete_or_error

from .cases import _interpolate_tabulated_field, load_tabulated_field
from .core3d import ImposedField, fringe_field

__all__ = ["Func", "Fringe", "Table", "Uniform"]


def pytree(data: tuple[str, ...] = (), meta: tuple[str, ...] = ()):
    """Register a dataclass as a pytree whose ``data`` fields are the leaves and ``meta`` the static ones."""

    def register(cls):
        return jax.tree_util.register_dataclass(cls, data_fields=list(data), meta_fields=list(meta))

    return register


def is_traced(*values) -> bool:
    """Whether any of the pytrees ``values`` holds a tracer."""
    return any(
        isinstance(leaf, jax.core.Tracer) for value in values for leaf in jax.tree_util.tree_leaves(value)
    )


def host(value, what: str) -> np.ndarray:
    """Return ``value`` for the host-side setup of a kernel, which a derivative cannot pass through."""
    if is_traced(value):
        raise NotImplementedError(
            f"{what} is traced, but its derivative needs the in-trace factorizations of Program H; "
            "close over it, or difference the solve"
        )
    return np.asarray(value, dtype=float)


def primal(value, what: str) -> float:
    """Return the host value of a leaf, its primal under :func:`jax.grad`; refuse an abstract tracer."""
    try:
        return float(concrete_or_error(None, value, what))
    except jax.errors.ConcretizationTypeError as error:
        raise TypeError(
            f"{what} is abstract (inside jit); the kernels set up on the host, so close over it "
            "and differentiate with jax.grad outside jit"
        ) from error


def split(value, what: str) -> tuple[float, object]:
    """Return ``(reference, scale)`` of a leaf: its host value, and the factor that restores its derivative."""
    reference = primal(value, what)
    if not is_traced(value):
        return reference, 1.0
    return (reference or 1.0), value / (reference or 1.0)


@pytree(("strength",), ("direction",))
@dataclass(frozen=True, eq=False, init=False)
class Uniform:
    """A uniform field, ``strength`` times the fixed unit vector ``direction``.

    ``Uniform(bx, by, bz)`` or ``Uniform(strength=, direction=)``. The derivative is the
    one through the amplitude; turning the field changes ``direction``, which is static.
    """

    strength: float
    direction: tuple[float, float, float]

    def __init__(
        self, *value: float, strength: float | None = None, direction: tuple[float, ...] = (0.0, 1.0, 0.0)
    ) -> None:
        if value:
            value = value[0] if len(value) == 1 else value
            if len(value) != 3:
                raise ValueError("a uniform field has three components")
            reference = np.array([primal(v, "a field component") for v in value])
            norm = float(np.linalg.norm(reference))
            if norm:
                direction = tuple(float(c) for c in reference / norm)
            elif is_traced(value):
                raise ValueError("a traced field needs a nonzero primal to give it a direction")
            strength = sum(v * d for v, d in zip(value, direction)) if is_traced(value) else norm
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "direction", tuple(direction))

    def sample(self, grid):
        reference, scale = split(self.strength, "the field strength")
        return tuple(reference * d for d in self.direction), scale


@pytree(("strength",), ("half_length", "centre", "solenoidal"))
@dataclass(frozen=True, eq=False)
class Fringe:
    """The ANL fringe of :func:`lmhdx.core3d.fringe_field`, ``strength`` times its profile."""

    strength: float
    half_length: float = 3.0
    centre: float = 0.0
    solenoidal: bool = False

    def sample(self, grid):
        reference, scale = split(self.strength, "the fringe strength")
        field = fringe_field(
            grid,
            half_length=self.half_length,
            centre=self.centre,
            strength=reference,
            solenoidal=self.solenoidal,
        )
        return field, scale


@pytree(("scale",), ("path",))
@dataclass(frozen=True, eq=False)
class Table:
    """A tabulated field: an NPZ with ``y, z, bx, by, bz`` (a cross-section) or ``x, y, z, ...``, times ``scale``."""

    path: str
    scale: float = 1.0

    def sample(self, grid):
        data = load_tabulated_field(self.path)
        centres = dict(zip("xyz", np.meshgrid(*grid.centers, indexing="ij")))
        sampled = _interpolate_tabulated_field(
            data, **{name: centres[name] for name in "xyz" if name in data}
        )
        return ImposedField(
            grid, tuple(np.broadcast_to(sampled[..., k], grid.shape) for k in range(3))
        ), self.scale


@pytree(("params",), ("fn",))
@dataclass(frozen=True, eq=False)
class Func:
    """A field ``fn(x, y, z, params) -> (Bx, By, Bz)`` sampled at the cell centres; ``params`` is any pytree."""

    fn: callable
    params: object = ()

    def sample(self, grid):
        params = jax.tree_util.tree_map(lambda leaf: host(leaf, "a Func parameter"), self.params)
        sampled = self.fn(*np.meshgrid(*grid.centers, indexing="ij"), params)
        return ImposedField(
            grid, tuple(np.broadcast_to(np.asarray(c, float), grid.shape) for c in sampled)
        ), 1.0
