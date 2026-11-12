# Install LMhdX

LMhdX supports Python 3.11–3.13 with JAX 0.7.2 or later. JAX 0.6.2, the last
release for Python 3.10, is not supported: its CUDA build crashes when a
thermal step's compiled loop runs a second time at production sizes, a fault
that later releases do not have. SOLVAX's dependency equinox excludes JAX
0.7.0 and 0.7.1.

```console
python -m venv .venv
source .venv/bin/activate
python -m pip install lmhdx
lmhdx --help
```

Install plotting support with `python -m pip install "lmhdx[visualization]"`.
The standard JAX package uses the CPU. For NVIDIA, AMD, or Apple accelerator
options, follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
and verify the selected devices:

```console
python -c "import jax; print(jax.devices())"
```

For source development:

```console
git clone https://github.com/uwplasma/LMhdX.git
cd LMhdX
python -m pip install -e ".[dev,docs]"
python scripts/run_full_test_suite.py
```

LMhdX imports lazily, so `import lmhdx` does not initialize JAX. The first case
you build turns on JAX's persistent compilation cache in `~/.cache/lmhdx`, so a
new process, or a new Hartmann number on the same mesh, reuses compiled solves
instead of compiling them again. Compiles over one second are kept, in at most
2 GiB. Programs are shared across field values, which makes a repeated warm
solve 35–60 % slower; for long loops over one compiled function, call
`lmhdx.enable_compilation_cache(share_across_values=False)` first. Set
`LMHDX_COMPILATION_CACHE=0` to turn the cache off, or to a directory to move it.
It stays off on macOS with jaxlib older than 0.10, which can crash reading
back a large cached program.

On import, LMhdX also adds `--xla_gpu_enable_triton_gemm=false` to `XLA_FLAGS`,
which halves the GPU compile of a steady solve at about 10 % warm cost. A value
you set yourself wins; `LMHDX_XLA_DEFAULTS=0` leaves `XLA_FLAGS` untouched.

A run uses JAX's default device: one GPU when a CUDA `jaxlib` is installed, else the CPU with XLA's own threads. Multi-device execution is research stage.

| Setting | Default | Change it when |
| --- | --- | --- |
| `JAX_PLATFORMS=cpu` | GPU when a CUDA `jaxlib` is installed | Small sections, where the CPU can beat a GPU; time your case |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | JAX preallocates most GPU memory | Several processes share one GPU |
| `LMHDX_COMPILATION_CACHE` | on, `~/.cache/lmhdx` | `0` turns it off; a path moves it |
| `share_across_values=False` | shared across field values | Long loops over one compiled function |
| `LMHDX_XLA_DEFAULTS=0` | cuBLAS GEMMs on GPU | Warm speed matters more than compile time |
