# Install LMhdX

This page installs LMhdX, picks the device JAX runs on, and explains the two
settings LMhdX changes for speed. You need Python 3.11–3.13 and JAX 0.7.2 or
later. A plain install runs on the CPU and needs nothing else.

```console
python -m venv .venv
source .venv/bin/activate
python -m pip install lmhdx
lmhdx --help
```

`lmhdx --help` lists the commands `run`, `benchmark` and `validate`. JAX 0.6.2,
the last release for Python 3.10, is not supported: its CUDA build crashes when
a thermal step's compiled loop runs a second time at production sizes, a fault
that later releases do not have. SOLVAX's dependency equinox excludes JAX 0.7.0
and 0.7.1.

## Plotting and accelerators

Add plotting support with:

```console
python -m pip install "lmhdx[visualization]"
```

The standard JAX package uses the CPU. For NVIDIA, AMD or Apple accelerators,
follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
then check which devices JAX sees:

```console
python -c "import jax; print(jax.devices())"
```

A run uses JAX's default device: one GPU when a CUDA `jaxlib` is installed,
else the CPU with XLA's own threads. Multi-device execution is not supported
yet.

| Setting | Default | Change it when |
| --- | --- | --- |
| `JAX_PLATFORMS=cpu` | GPU when a CUDA `jaxlib` is installed | Small sections, where the CPU can beat a GPU; time your case |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | JAX preallocates most GPU memory | Several processes share one GPU |

## From source

For development, install the repository in editable mode and run the tests:

```console
git clone https://github.com/uwplasma/LMhdX.git
cd LMhdX
python -m pip install -e ".[dev,docs]"
python scripts/run_full_test_suite.py
```

## What LMhdX sets for you

`import lmhdx` is lazy and does not start JAX. The first case you build turns
on JAX's persistent compilation cache in `~/.cache/lmhdx/jax_compilation`. A
new process, or a new Hartmann number on the same mesh, then reuses compiled
solves instead of compiling them again.

- Compiles over one second are kept, in at most 2 GiB.
- Programs are shared across field values, which makes a repeated warm solve
  35–60 % slower. For long loops over one compiled function, call
  `lmhdx.enable_compilation_cache(share_across_values=False)` first.
- Set `LMHDX_COMPILATION_CACHE=0` to turn the cache off, or set it to a
  directory to move it.
- The cache stays off on macOS with jaxlib older than 0.10, which can crash
  reading back a large cached program.

On import, LMhdX also adds `--xla_gpu_enable_triton_gemm=false` to `XLA_FLAGS`.
This halves the GPU compile of a steady solve at about 10 % warm cost. A value
you set yourself wins, and `LMHDX_XLA_DEFAULTS=0` leaves `XLA_FLAGS` untouched.
