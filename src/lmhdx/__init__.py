"""Small, lazy convenience API for LMhdX.

Research and advanced APIs live in their named modules, for example
``lmhdx.axial`` and ``lmhdx.cases``. Keeping the package root deliberately
small makes supported concepts discoverable and avoids importing JAX-heavy
solver modules until a symbol is used.
"""

from __future__ import annotations

import os
from importlib import import_module

# Cold runs first: cuBLAS instead of Triton GEMMs halves the GPU compile of a steady
# solve (9.4 s to 4.2 s on an RTX A4000) at about 10 % warm cost. XLA reads the flag
# when the GPU backend starts, so it is set on import; an explicit user flag wins.
if (
    "xla_gpu_enable_triton_gemm" not in os.environ.get("XLA_FLAGS", "")
    and os.environ.get("LMHDX_XLA_DEFAULTS") != "0"
):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_triton_gemm=false"
    ).strip()

__all__ = [
    "enable_x64",
    "enable_compilation_cache",
    "make_hartmann_case",
    "make_shercliff_case",
    "make_hunt_case",
    "make_q2d_case",
    "evolve_q2d",
    "solve_fully_developed_fields",
    "Q2DProblem",
    "solve",
    "ChannelProblem",
    "duct_problem",
    "solve_steady_state",
    "advance",
    "generate_rect_duct_mesh_from_faces",
    "WallLayer",
    "dynamic_to_kinematic_viscosity",
    "kinematic_to_dynamic_viscosity",
    "hartmann_number",
    "reynolds_number",
    "interaction_parameter",
    "magnetic_reynolds_number",
    "magnetic_field_from_hartmann",
    "wall_conductance_ratio",
    "effective_pinhole_conductance_ratio",
    "tangential_stack_conductance_ratio",
    "normal_stack_leakage_ratio",
    "equivalent_single_layer",
    "nested_wall_layer_resolution_summary",
]


def enable_x64() -> None:
    """Enable float64 arrays process-wide; call before constructing meshes or tracing.

    This also sets JAX's ``jax_default_matmul_precision`` to ``'highest'`` when
    that option is unset. ``'highest'`` is the same level as ``'float32'``.
    Unset, JAX lets Ampere and newer GPUs run float32 matrix products and
    convolutions in TensorFloat-32. On an RTX A4000, float32 contractions then
    differed from float64 by 3e-4 relative. With the pinned precision the
    difference was 2.6-6.1e-7. CPUs and float64 arrays are unaffected.

    A value the user already chose is kept, whether set through
    ``jax.config.update`` or the ``JAX_DEFAULT_MATMUL_PRECISION`` environment
    variable. Constructing a case with a ``dtype``, a ``ChannelProblem`` or a
    ``Q2DProblem`` applies the same rule.
    """
    from jax import config

    config.update("jax_enable_x64", True)
    _pin_matmul_precision()


def _pin_matmul_precision() -> str:
    """Set ``jax_default_matmul_precision`` to ``'highest'`` if unset; return the value in effect.

    The option is part of JAX's compilation key, so functions traced before the
    pin are traced again afterwards rather than silently keeping TensorFloat-32.
    """
    from jax import config

    if config.jax_default_matmul_precision is None:
        config.update("jax_default_matmul_precision", "highest")
    _enable_default_cache(config)
    return config.jax_default_matmul_precision


def _enable_default_cache(config) -> None:
    """Keep compiled solves on disk unless the user chose otherwise, so a new process starts warm.

    Programs are shared across field values, so a new Hartmann number on the same
    mesh reuses them (compile 4.4 s to 0.45 s on an A4000) at a 35-60 % warm cost.
    Only compiles over one second are kept, in at most 2 GiB. ``LMHDX_COMPILATION_CACHE=0``
    disables it; a path in it sets the directory. It stays off on macOS with jaxlib
    older than 0.10, which can crash reading back a large cached CPU program.
    """
    choice = os.environ.get("LMHDX_COMPILATION_CACHE", "")
    if choice == "0" or config.jax_compilation_cache_dir or _cache_read_unsafe():
        return
    from .io import enable_compilation_cache

    enable_compilation_cache(choice or None, min_compile_time_secs=1.0, share_across_values=True)
    config.update("jax_compilation_cache_max_size", 2**31)


def _cache_read_unsafe() -> bool:
    import platform

    import jaxlib

    major, minor = (int(part) for part in jaxlib.__version__.split(".")[:2])
    return platform.system() == "Darwin" and (major, minor) < (0, 10)


_EXPORTS = {
    "enable_compilation_cache": ("lmhdx.io", "enable_compilation_cache"),
    "make_hartmann_case": ("lmhdx.cases", "make_hartmann_case"),
    "make_shercliff_case": ("lmhdx.cases", "make_shercliff_case"),
    "make_hunt_case": ("lmhdx.cases", "make_hunt_case"),
    "make_q2d_case": ("lmhdx.q2d", "make_q2d_case"),
    "evolve_q2d": ("lmhdx.q2d", "evolve_q2d"),
    "solve_fully_developed_fields": ("lmhdx.fully_developed", "solve_fully_developed_fields"),
    "ChannelProblem": ("lmhdx.core3d", "ChannelProblem"),
    "duct_problem": ("lmhdx.core3d", "duct_problem"),
    "solve_steady_state": ("lmhdx.steady", "solve_steady_state"),
    "advance": ("lmhdx.core3d", "advance"),
    "Q2DProblem": ("lmhdx.q2d", "Q2DProblem"),
    "solve": ("lmhdx.cases", "solve"),
    "generate_rect_duct_mesh_from_faces": ("lmhdx.cases", "generate_rect_duct_mesh_from_faces"),
    "WallLayer": ("lmhdx.cases", "WallLayer"),
    "dynamic_to_kinematic_viscosity": ("lmhdx.cases", "dynamic_to_kinematic_viscosity"),
    "kinematic_to_dynamic_viscosity": ("lmhdx.cases", "kinematic_to_dynamic_viscosity"),
    "hartmann_number": ("lmhdx.cases", "hartmann_number"),
    "reynolds_number": ("lmhdx.cases", "reynolds_number"),
    "interaction_parameter": ("lmhdx.cases", "interaction_parameter"),
    "magnetic_reynolds_number": ("lmhdx.cases", "magnetic_reynolds_number"),
    "magnetic_field_from_hartmann": ("lmhdx.cases", "magnetic_field_from_hartmann"),
    "wall_conductance_ratio": ("lmhdx.cases", "wall_conductance_ratio"),
    "effective_pinhole_conductance_ratio": ("lmhdx.cases", "effective_pinhole_conductance_ratio"),
    "tangential_stack_conductance_ratio": ("lmhdx.cases", "tangential_stack_conductance_ratio"),
    "normal_stack_leakage_ratio": ("lmhdx.cases", "normal_stack_leakage_ratio"),
    "equivalent_single_layer": ("lmhdx.cases", "equivalent_single_layer"),
    "nested_wall_layer_resolution_summary": ("lmhdx.cases", "nested_wall_layer_resolution_summary"),
}


def __getattr__(name: str):
    """Load a documented root export on first access."""

    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'lmhdx' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals plus the lazy public surface for discovery."""

    return sorted(set(globals()) | set(__all__))
