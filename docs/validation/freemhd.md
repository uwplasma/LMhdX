# FreeMHD comparison

This page explains how LMhdX is run side by side with FreeMHD, an independent
liquid-metal MHD code built on OpenFOAM. The case is the ALEX B2 square duct
in a fringing field. Each code passes or fails on its own checks; their
pressure difference is reported, not gated, because the two runs do not
compute the same state.

To write and hash the inputs without Docker:

```console
python -m validation.freemhd --preflight --output artifacts/freemhd-b2
```

FreeMHD is an external validator, not an LMhdX runtime dependency. One module,
`validation/freemhd.py`, holds the whole comparison:

1. it writes the FreeMHD B2 input: the block mesh, the foam dictionaries, the
   field `B_y(x)/B0` interpolated linearly between the anchors of
   `alex-b2-square.csv`, and the pressure taps and boundary-flux probes;
2. it checks the FreeMHD source snapshot against the commit and SHA-256 pins in
   `[free_mhd_discretization_reference]` of `alex-b2-square.toml`;
3. it runs FreeMHD in Docker and reads the native log and probe tables;
4. it solves the same duct with the core-flow model
   (`lmhdx.coreflow.CoreFlow`): aspect 1, thin walls `c_t = c_s = 0.07`, unit
   mean velocity;
5. it writes `record.json` with both observables and the SHA-256 of every input
   and output.

## What is compared

The two codes do not compute the same state. FreeMHD runs the frozen two-update
transient harness smoke from a uniform plug on a 5 x 5 grid. LMhdX gives the
steady inertialess core flow. The cross-code pressure difference is therefore
reported, not gated, and so is LMhdX against the ALEX `pressure_observable`
column. Each code passes or fails on its own gates:

- FreeMHD: the `[harness_smoke_execution]` limits on step count, time step,
  Courant number, mass, current and interface-current balance, and interface
  current activity, plus a log without fatal markers;
- LMhdX: a finite solution, `beta = 1/B` floored only where the tabulated field
  is below `1/beta_max` (the CSV field is zero for `x/L >= 7.5`), and an axial
  flux constant along the duct to a relative spread of `1e-4`.

The observable is the side-wall tap (`y = 0`, `|z| = 1`) minus the top tap
(`y = a`, `z = 0`), in units `sigma U B0^2 L`, minus its mean over the plateau
`x/L <= -7.5` or `x/L >= 5`. The core pressure is constant along field lines,
so LMhdX takes the side tap at the first `z` cell centre and the top tap at the
last one.

## Running it

The reproducible build uses
[`freemhd_install`](https://github.com/rogeriojorge/freemhd_install) commit
`36f409d294ba3170d64d4073378d5ef68401072f`, FreeMHD commit
`14b54a3e8e1a05b6ee4c98331995abaaae96e7a5`, and OpenFOAM v2206. The weekly
external-validation workflow rebuilds that image, runs the module and uploads
the evidence directory. Pull requests that touch the module, the core-flow
model or the B2 data run the same workflow.

```console
python -m validation.freemhd \
  --freemhd-image freemhd-install:latest \
  --freemhd-install-dir freemhd_install \
  --freemhd-source-repo FreeMHD \
  --output artifacts/freemhd-b2
```

`--preflight`, shown at the top, writes and hashes the inputs without Docker.
The exit code is 0 when both codes pass their own gates. Record the image ID next to the evidence:

```console
docker image inspect freemhd-install:latest --format '{{.Id}}'
```

The gated comparison with the ALEX experiment belongs to the steady 3-D solver,
not to this module.
