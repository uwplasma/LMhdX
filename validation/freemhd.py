"""FreeMHD and LMhdX on the ALEX B2 fringing-field duct, with provenance.

``python -m validation.freemhd --output DIR`` writes the pinned FreeMHD B2
input, checks the FreeMHD source snapshot against
``[free_mhd_discretization_reference]`` of ``alex-b2-square.toml``, runs FreeMHD
in Docker, observes its native output, solves the same geometry with
:class:`lmhdx.coreflow.CoreFlow` and writes ``record.json``. ``--preflight``
stops after materializing and hashing the inputs, without Docker.

The two sides are not the same state. FreeMHD runs the frozen two-update
transient harness smoke from a uniform plug (Ha 2900, N 540, a 5x5 grid);
LMhdX solves the steady inertialess core flow (TM-228, the ``N -> infinity``
limit). Each side is gated only on its own execution: FreeMHD on the
``[harness_smoke_execution]`` limits that apply to it, LMhdX on a finite
solution, on flooring only where the tabulated field is below ``1/beta_max``
and on a constant axial flux. The cross-code pressure difference and LMhdX
against the ALEX ``pressure_observable`` column are reported, not gated; the
gated experimental comparison belongs to the steady 3-D solver.

The observable is the side-wall tap (``y = 0, |z| = 1``) minus the top tap
(``y = a, z = 0``) in units ``sigma U B0^2 L``, minus its mean over the
plateau ``x <= -7.5`` or ``x >= 5``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import tomllib
from itertools import pairwise
from pathlib import Path, PurePosixPath

import numpy as np

_DATA = Path(__file__).resolve().parents[1] / "src" / "lmhdx" / "data" / "benchmarks"
SPEC = _DATA / "specs" / "alex-b2-square.toml"
REFERENCE = _DATA / "references" / "alex-b2-square.csv"
CASE_ID = "B2-fringing-square"
DT = 1.0 / 540000.0
PLATEAU = (-7.5, 5.0)
BETA_MAX = 1000.0
AXIAL_FLUX_SPREAD_MAX = 1.0e-4
_SOURCE_NAMES = "momentum electric limiter scheme_macro limiter_registration nvd vector_transform".split()
_SKELETON = "blockMeshDict controlDict liquid/fvSchemes liquid/fvSolution".split()
_OBJECTS = (
    "b2PressureTaps massIn massOut currentIn currentOut currentIntoSolid currentIntoSolidMagnitude".split()
)
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def load_spec() -> dict:
    return tomllib.loads(SPEC.read_text(encoding="utf-8"))


def load_reference() -> dict[str, np.ndarray]:
    with REFERENCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}


def artifact_sha256(path: str | Path, kind: str | None = None) -> str:
    """SHA-256 of a regular file, or of a tree: sorted ``(kind, relative path, file hash)`` frames."""

    path = Path(path)
    kind = kind or ("tree" if path.is_dir() else "file")
    if path.is_symlink():
        raise ValueError(f"refusing to hash a symlink: {path.name}")
    if kind == "file":
        if not path.is_file():
            raise ValueError(f"not a regular file: {path.name}")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if kind != "tree" or not path.is_dir():
        raise ValueError(f"not a {kind}: {path.name}")
    digest = hashlib.sha256(b"LMX-ARTIFACT-TREE-v1\0")
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        mode = os.lstat(child).st_mode
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError(f"tree holds a link or special file: {child.name}")
        kind_tag = b"d" if stat.S_ISDIR(mode) else b"f"
        name = child.relative_to(path).as_posix().encode()
        payload = kind_tag + len(name).to_bytes(8, "big") + name
        if kind_tag == b"f":
            payload += bytes.fromhex(artifact_sha256(child, "file"))
        digest.update(len(payload).to_bytes(8, "big") + payload)
    return digest.hexdigest()


def snapshot_freemhd_source(source_repo: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Copy the pinned, clean, tracked FreeMHD/OpenFOAM sources after checking their SHA-256."""

    repository, destination = Path(source_repo).resolve(), Path(output_dir)
    reference = load_spec()["free_mhd_discretization_reference"]

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-C", str(repository), *args], capture_output=True, text=True)

    top = git("rev-parse", "--show-toplevel")
    if top.returncode or Path(top.stdout.strip()).resolve() != repository:
        raise ValueError("the FreeMHD source must be a Git worktree root")
    if git("rev-parse", "HEAD").stdout.strip() != reference["repository_commit"]:
        raise ValueError("the FreeMHD HEAD is not the pinned commit")
    files = {}
    for name in _SOURCE_NAMES:
        relative = reference[f"{name}_source"]
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            raise ValueError(f"noncanonical pinned source path {relative}")
        tracked = git("ls-files", "--stage", "--error-unmatch", "--", relative)
        if tracked.returncode or not tracked.stdout.startswith("100"):
            raise ValueError(f"pinned source is not a tracked regular file: {relative}")
        files[relative] = reference[f"{name}_source_sha256"]
        if artifact_sha256(repository / relative, "file") != files[relative]:
            raise ValueError(f"pinned source SHA-256 differs: {relative}")
    for relative in files:
        (destination / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination / relative)
    manifest = {"commit": reference["repository_commit"], "files": dict(sorted(files.items()))}
    manifest["openfoam_release"] = reference["openfoam_release"]
    (destination / "source-pin.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _foam(path: Path, body: str, class_name: str = "dictionary") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {class_name};\n    object {path.name};\n}}\n\n"
    path.write_text(header + body.strip() + "\n", encoding="utf-8")


def _field_expression(count: int) -> str:
    """Piecewise-linear ``B_y(x)`` through the anchors ``(xa, ba), (xb, bb), ...`` as ramp sums."""

    labels = [chr(97 + index) for index in range(count)]
    slopes = [f"(b{right}-b{left})/(x{right}-x{left})" for left, right in pairwise(labels)]
    terms = [f"b{labels[0]}", f"{slopes[0]}*(x-x{labels[0]})"]
    terms += [
        f"({slopes[i]}-{slopes[i - 1]})*pos(x-x{labels[i]})*(x-x{labels[i]})" for i in range(1, count - 1)
    ]
    return "+".join(terms)


_BLOCK_MESH = """
scale 1;
xMin -15; xMax 10; Ly 1; Ly_wall 1.02; physicalHalfWidth 0.0439;
Nx 8; Ny 5; Nz 5; N_wall 1;
vertices
(
 ($xMin -$Ly -$Ly) ($xMax -$Ly -$Ly) ($xMax $Ly -$Ly) ($xMin $Ly -$Ly)
 ($xMin -$Ly $Ly) ($xMax -$Ly $Ly) ($xMax $Ly $Ly) ($xMin $Ly $Ly)
 ($xMin -$Ly -$Ly_wall) ($xMax -$Ly -$Ly_wall) ($xMax $Ly -$Ly_wall) ($xMin $Ly -$Ly_wall)
 ($xMin -$Ly_wall -$Ly_wall) ($xMax -$Ly_wall -$Ly_wall) ($xMax $Ly_wall -$Ly_wall) ($xMin $Ly_wall -$Ly_wall)
 ($xMin -$Ly_wall -$Ly) ($xMax -$Ly_wall -$Ly) ($xMax $Ly_wall -$Ly) ($xMin $Ly_wall -$Ly)
 ($xMin -$Ly $Ly_wall) ($xMax -$Ly $Ly_wall) ($xMax $Ly $Ly_wall) ($xMin $Ly $Ly_wall)
 ($xMin -$Ly_wall $Ly_wall) ($xMax -$Ly_wall $Ly_wall) ($xMax $Ly_wall $Ly_wall) ($xMin $Ly_wall $Ly_wall)
 ($xMin -$Ly_wall $Ly) ($xMax -$Ly_wall $Ly) ($xMax $Ly_wall $Ly) ($xMin $Ly_wall $Ly)
);
blocks
(
 hex (0 1 2 3 4 5 6 7) liquid ($Nx $Ny $Nz) simpleGrading (1 1 1)
 hex (12 13 9 8 16 17 1 0) solidWalls ($Nx $N_wall $N_wall) simpleGrading (1 1 1)
 hex (11 10 14 15 3 2 18 19) solidWalls ($Nx $N_wall $N_wall) simpleGrading (1 1 1)
 hex (8 9 10 11 0 1 2 3) solidWalls ($Nx $Ny $N_wall) simpleGrading (1 1 1)
 hex (28 29 5 4 24 25 21 20) solidWalls ($Nx $N_wall $N_wall) simpleGrading (1 1 1)
 hex (7 6 30 31 23 22 26 27) solidWalls ($Nx $N_wall $N_wall) simpleGrading (1 1 1)
 hex (4 5 6 7 20 21 22 23) solidWalls ($Nx $Ny $N_wall) simpleGrading (1 1 1)
 hex (16 17 1 0 28 29 5 4) solidWalls ($Nx $N_wall $Nz) simpleGrading (1 1 1)
 hex (3 2 18 19 7 6 30 31) solidWalls ($Nx $N_wall $Nz) simpleGrading (1 1 1)
);
edges ();
boundary
(
 inlet { type patch; faces ((0 4 7 3)); }
 sink { type patch; faces ((2 6 5 1)); }
 outerWalls
 {
  type wall;
  faces
  (
   (12 16 0 8) (8 0 3 11) (11 3 19 15) (3 7 31 19) (7 23 27 31) (4 20 23 7) (28 24 20 4)
   (16 28 4 0) (13 17 16 12) (17 29 28 16) (29 25 24 28) (25 21 20 24) (21 22 23 20)
   (22 26 27 23) (30 26 27 31) (18 30 31 19) (14 18 19 15) (10 14 15 11) (9 10 11 8)
   (13 9 8 12) (13 17 1 9) (9 1 2 10) (10 2 18 14) (17 29 5 1) (2 6 30 18) (29 25 21 5)
   (5 21 22 6) (6 22 26 30)
  );
 }
);
mergePatchPairs ();
"""
_LIQUID_SCHEMES = """
ddtSchemes { default Euler; }
gradSchemes { default cellLimited leastSquares 1.0; }
divSchemes { default Gauss linear; div(rhoPhi,U) Gauss limitedLinear 1.0; div(phi,alpha) Gauss vanLeer; div(phirb,alpha) Gauss interfaceCompression; div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear uncorrected; } interpolationSchemes { default linear; } snGradSchemes { default uncorrected; }
"""
_LIQUID_SOLUTION = """
solvers {
 "alpha.liquidMetal.*" { nAlphaCorr 1; nAlphaSubCycles 1; cAlpha 1; solver PBiCG; preconditioner DILU; tolerance 1e-12; relTol 0; }
 p_rgh { solver PCG; preconditioner DIC; tolerance 1e-10; relTol 0; maxIter 4000; }
 p_rghFinal { $p_rgh; }
 "(U).*" { solver PBiCG; preconditioner DILU; tolerance 1e-10; relTol 0; maxIter 400; }
 "(h|T).*" { solver PBiCG; preconditioner DILU; tolerance 1e-12; relTol 0; maxIter 0; }
 potE { solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; maxIter 600; }
 potEFinal { $potE; }
}
PIMPLE { correctPhi yes; momentumPredictor yes; nCorrectors 1; nOuterCorrectors 1; nNonOrthogonalCorrectors 0; }
potentialFlow { nNonOrthogonalCorrectors 0; PhiRefCell 0; PhiRefValue 0; }
potE { nCorrectors 0; nNonOrthogonalCorrectors 0; PotERefCell 0; PotERefValue 0; }
"""
_SOLID_SCHEMES = """
ddtSchemes { default Euler; } gradSchemes { default Gauss linear; } divSchemes { default Gauss linear; }
laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; } snGradSchemes { default corrected; }
"""
_SOLID_SOLUTION = """
solvers { "(h|T).*" { solver PBiCG; preconditioner DILU; tolerance 1e-12; relTol 0; maxIter 0; } potE { solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; maxIter 600; } potEFinal { $potE; } }
PIMPLE { nNonOrthogonalCorrectors 0; } potE { nCorrectors 0; nNonOrthogonalCorrectors 0; PotERefCell 0; PotERefValue 0; }
"""
_B0 = 'B0 { internalField uniform (0 23.2379000772445 0); boundaryField { ".*" { type zeroGradient; value $internalField; } } }'
_COUPLED = "type compressible::turbulentTemperatureCoupledBaffleMixed;"
_LIQUID_CHANGE = f"""
alpha.liquidMetal {{ internalField uniform 1; boundaryField {{ inlet {{ type fixedValue; value uniform 1; }} ".*" {{ type zeroGradient; }} }} }}
U {{ internalField uniform (1 0 0); boundaryField {{ inlet {{ type flowRateInletVelocity; volumetricFlowRate 4; extrapolateProfile yes; value uniform (1 0 0); }} sink {{ type zeroGradient; value uniform (1 0 0); }} ".*" {{ type noSlip; }} }} }}
T {{ internalField uniform 300; boundaryField {{ ".*" {{ type fixedValue; value uniform 300; }} "liquid_to_.*" {{ {_COUPLED} Tnbr T; kappaMethod fluidThermo; value uniform 300; }} inlet {{ type fixedValue; value uniform 300; }} sink {{ type fixedValue; value uniform 300; }} }} }}
p_rgh {{ internalField uniform 0; boundaryField {{ sink {{ type fixedValue; value uniform 0; }} inlet {{ type zeroGradient; }} ".*" {{ type fixedFluxPressure; value uniform 0; }} }} }}
p {{ internalField uniform 0; boundaryField {{ ".*" {{ type calculated; value uniform 0; }} }} }}
{_B0}
potE {{ internalField uniform 0; boundaryField {{ inlet {{ type zeroGradient; }} sink {{ type zeroGradient; }} "liquid_to_.*" {{ {_COUPLED} Tnbr potE; kappaMethod lookup; kappa elcond; kappaName elcond; value uniform 0; }} }} }}
"""
_SOLID_CHANGE = f"""
T {{ internalField uniform 300; boundaryField {{ outerWalls {{ type fixedValue; value uniform 300; }} "solidWalls_to_.*" {{ {_COUPLED} Tnbr T; kappaMethod solidThermo; value uniform 300; }} }} }}
{_B0}
potE {{ internalField uniform 0; boundaryField {{ outerWalls {{ type zeroGradient; value uniform 0; }} "solidWalls_to_.*" {{ {_COUPLED} Tnbr potE; kappaMethod lookup; kappa elcond; kappaName elcond; value uniform 0; }} }} }}
"""


_INITIAL_FIELDS = """
B0|volVectorField|[1 0 -2 0 0 -1 0]|uniform (0 23.2379000772445 0)
JxB|volVectorField|[1 -2 -2 0 0 0 0]|uniform (0 0 0)
T|volScalarField|[0 0 0 1 0 0 0]|uniform 300
U|volVectorField|[0 1 -1 0 0 0 0]|uniform (1 0 0)
alpha.liquidMetal|volScalarField|[0 0 0 0 0 0 0]|uniform 1
p|volScalarField|[1 -1 -2 0 0 0 0]|uniform 0
p_rgh|volScalarField|[1 -1 -2 0 0 0 0]|uniform 0
potE|volScalarField|[1 2 -3 0 0 -1 0]|uniform 0
"""


def _set_expr(reference: dict[str, np.ndarray]) -> str:
    x, b = reference["x_over_L"].tolist(), reference["b_over_B0"].tolist()
    variables = ['"x=pos().x()"', '"Bscale=sqrt(540)"']
    variables += [f'"x{chr(97 + i)}={value:.17g}"' for i, value in enumerate(x)]
    variables += [f'"b{chr(97 + i)}={value:.17g}"' for i, value in enumerate(b)]
    anchors = json.dumps({"b_over_B0": b, "x_over_L": x}, sort_keys=True, separators=(",", ":"))
    return f"""
lmxFieldSource "{REFERENCE.name}";
lmxFieldSourceSHA256 "{artifact_sha256(REFERENCE, "file")}";
lmxFieldAnchorsSHA256 "{hashlib.sha256(anchors.encode()).hexdigest()}";
lmxInterpolation linear;
lmxExtrapolation forbidden;
expressions
(
 B0
 {{
  field B0;
  dimensions [1 0 -2 0 0 -1 0];
  variables ({" ".join(variables)});
  expression #{{ vector(0,Bscale*({_field_expression(len(x))}),0) #}};
 }}
);
"""


def _function_objects(sample_x: list[float]) -> str:
    """Pressure taps at ``(x, 0.8, 0)`` (top) then ``(x, 0, 0.8)`` (side), and boundary fluxes."""

    points = "\n  ".join(
        f"({x:.17g} {y:.17g} {z:.17g})" for y, z in ((0.8, 0.0), (0.0, 0.8)) for x in sample_x
    )
    every = "executeControl timeStep; executeInterval 1; writeControl timeStep; writeInterval 1;"
    blocks = [
        f"b2PressureTaps\n {{\n  type probes; libs (sampling); region liquid;\n  {every}\n"
        f"  fixedLocations true; interpolationScheme cell; fields (p);\n  probeLocations\n (\n  {points}\n );\n }}"
    ]
    patches = "inlet sink inlet sink liquid_to_solidWalls liquid_to_solidWalls".split()
    fields = "rhoPhi rhoPhi jn jn jn jn".split()
    for name, patch, field in zip(_OBJECTS[1:], patches, fields):
        operation = "sumMag" if name.endswith("Magnitude") else "sum"
        blocks.append(
            f"{name}\n {{\n  type surfaceFieldValue; libs (fieldFunctionObjects); region liquid;\n  {every}\n"
            f"  regionType patch; name {patch}; operation {operation}; fields ({field}); writeFields false;\n }}"
        )
    return "functions\n{\n " + "\n ".join(blocks) + "\n}"


def materialize_freemhd_input(output_dir: str | Path) -> str:
    """Write the deterministic two-update FreeMHD B2 case and return its tree SHA-256."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination.name}")
    mu = 540.0 / 2900.0**2  # rho = U = L = sigma = 1: mu = N / Ha^2, B0 = sqrt(N)
    fluid = f"""
thermoType {{ type heRhoThermo; mixture pureMixture; transport const; thermo hConst; equationOfState rhoConst; specie specie; energy sensibleInternalEnergy; }}
mixture {{ specie {{ molWeight 1; }} equationOfState {{ rho 1; }} thermodynamics {{ Cp 1; Cv 1; Hf 0; }} transport {{ mu {mu:.17g}; Pr 1; }} }}
elcond [-1 -3 3 0 0 2 0] 1;"""
    # c_w = elcond * thickness = 3.5 * 0.02 = 0.07.
    solid = """
thermoType { type heSolidThermo; mixture pureMixture; transport constIso; thermo hConst; equationOfState rhoConst; specie specie; energy sensibleEnthalpy; }
mixture { specie { molWeight 1; } transport { kappa 1; } thermodynamics { Hf 0; Cp 1; } equationOfState { rho 1; } }
elcond 3.5;"""
    sample_x = np.linspace(-15.0 + 25.0 / 16.0, 10.0 - 25.0 / 16.0, 8).tolist()
    control = f"""
application epotMultiRegionInterFoam; startFrom startTime; startTime 0; stopAt endTime;
endTime {2 * DT:.17g}; deltaT {DT:.17g}; adjustTimeStep off; maxCo 0.4; maxAlphaCo 0.3; maxDeltaT {DT:.17g};
writeControl timeStep; writeInterval 2; purgeWrite 0; writeFormat ascii; writePrecision 16; timeFormat general; timePrecision 16;
runTimeModifiable false; BtStartTime 0; BtDuration 0; JConservativeForm true;
lmxSteadyStepsRequired 3; {_function_objects(sample_x)}
"""
    files = {
        "constant/g": ("dimensions [0 1 -2 0 0 0 0];\nvalue (0 0 0);", "uniformDimensionedVectorField"),
        "constant/regionProperties": "regions ( fluid (liquid) solid (solidWalls) );",
        "constant/liquid/fvOptions": "{}",
        "constant/liquid/turbulenceProperties": "simulationType laminar;",
        "constant/liquid/thermophysicalProperties": "phases (liquidMetal air);\npMin 0;\nsigma [1 0 -2 0 0 0 0] 0;",
        "constant/liquid/thermophysicalProperties.liquidMetal": fluid,
        "constant/liquid/thermophysicalProperties.air": fluid.replace("0 0 2 0] 1;", "0 0 2 0] 0;"),
        "constant/solidWalls/thermophysicalProperties": solid,
        "system/controlDict": control,
        "system/blockMeshDict": _BLOCK_MESH,
        "system/fvSchemes": "ddtSchemes {} gradSchemes {} divSchemes {} laplacianSchemes {} "
        "interpolationSchemes {} snGradSchemes {}",
        "system/fvSolution": "PIMPLE { nOuterCorrectors 1; }",
    }
    for line in _INITIAL_FIELDS.strip().splitlines():
        name, class_name, dimensions, internal = line.split("|")
        boundary = 'boundaryField { ".*" { type calculated; value $internalField; } }'
        files[f"0/{name}"] = (f"dimensions {dimensions};\ninternalField {internal};\n{boundary}", class_name)
    field = _set_expr(load_reference())
    for region, bodies in (
        ("liquid", (_LIQUID_SCHEMES, _LIQUID_SOLUTION, _LIQUID_CHANGE, field)),
        ("solidWalls", (_SOLID_SCHEMES, _SOLID_SOLUTION, _SOLID_CHANGE, field)),
    ):
        names = ("fvSchemes", "fvSolution", "changeDictionaryDict", "setExprFieldsDict")
        files |= {f"system/{region}/{name}": body for name, body in zip(names, bodies)}
    for region in ("", "liquid/", "solidWalls/"):
        files[f"system/{region}decomposeParDict"] = "numberOfSubdomains 2;\nmethod scotch;"
    for relative, body in files.items():
        _foam(destination / relative, *((body,) if isinstance(body, str) else body))
    return artifact_sha256(destination, "tree")


def run_freemhd(input_dir: Path, output_dir: Path, image: str, nproc: int, timeout: float) -> float:
    """Run the case in the pinned image; copy the log, controls and function-object tables out."""

    output_dir.mkdir(parents=True)
    container = f"lmhdx-b2-{os.getpid()}-{time.time_ns()}"
    shell = f"""
source /usr/lib/openfoam/openfoam2206/etc/bashrc
set -euo pipefail
work=/tmp/lmx-b2-case
rm -rf "$work" && mkdir -p "$work"
rsync -a /input/ "$work/" && cd "$work"
blockMesh -fileHandler collated
splitMeshRegions -cellZonesOnly -overwrite -fileHandler collated
for region in liquid solidWalls; do
  changeDictionary -region "$region" -fileHandler collated
  setExprFields -region "$region" -fileHandler collated
done
decomposePar -allRegions -force -fileHandler collated
cp system/controlDict /output/controlDict.used
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
mpirun --oversubscribe -np {nproc} epotMultiRegionInterFoam -parallel 2>&1 | tee /output/run.log
mkdir /output/postProcessing
for name in {" ".join(_OBJECTS)}; do
  path="$(find postProcessing -type d -name "$name" -print -quit)"
  test -n "$path" && cp -a "$path" "/output/postProcessing/$name"
done
"""
    command = ["docker", "run", "--rm", "--name", container, "--entrypoint", "/bin/bash"]
    command += ["--mount", f"type=bind,src={input_dir.resolve()},dst=/input,readonly"]
    command += ["--mount", f"type=bind,src={output_dir.resolve()},dst=/output", image, "-lc", shell]
    started = time.perf_counter()
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)
    if done.returncode:
        raise RuntimeError(f"FreeMHD exited {done.returncode}:\n{(done.stdout + done.stderr)[-4000:]}")
    return time.perf_counter() - started


def _table(path: Path, width: int, times: np.ndarray) -> np.ndarray:
    rows = [
        [float(value) for value in re.findall(_NUMBER, line)]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    values = np.asarray(rows, dtype=float)
    if values.shape != (times.size, width) or not np.allclose(values[:, 0], times, rtol=0, atol=1e-18):
        raise ValueError(f"FreeMHD table {path.parent.name}/{path.name} has an unexpected shape or times")
    return values[:, 1:]


def observe_freemhd(output_dir: str | Path) -> dict[str, object]:
    """Read the native FreeMHD log and function-object tables into the smoke observables."""

    root = Path(output_dir)
    log = (root / "run.log").read_text(encoding="utf-8")
    markers = ("FOAM FATAL", "Segmentation fault", "MPI_ABORT", "killed")
    fatal = [marker for marker in markers if marker.lower() in log.lower()]
    if re.search(r"(?im)^(?!.*trapping enabled).*Floating point exception", log):
        fatal.append("Floating point exception")
    if re.search(r"(?i)(?:^|[\s=,(])(?:nan|[-+]?inf)(?:$|[\s,;)])", log):
        fatal.append("non-finite value")
    if fatal or re.search(r"(?m)^End\s*$", log) is None:
        raise ValueError(f"FreeMHD log reports a failure: {fatal or ['no End']}")
    times = np.asarray([float(value) for value in re.findall(r"(?m)^Time = (\S+)\s*$", log)])
    courant = np.asarray(
        re.findall(rf"(?m)^Region: liquid Courant Number mean:\s*({_NUMBER})\s+max:\s*({_NUMBER})", log),
        dtype=float,
    )[-times.size :]
    post = root / "postProcessing"
    (probe,) = (post / "b2PressureTaps").rglob("p")
    headers = re.findall(r"(?m)^# Probe \d+ \(([^)]+)\)$", probe.read_text())
    points = np.asarray([[float(value) for value in point.split()] for point in headers])
    if points.shape != (16, 3) or not (
        np.allclose(points[:8, 1:], (0.8, 0.0)) and np.allclose(points[8:, 1:], (0.0, 0.8))
    ):
        raise ValueError("FreeMHD pressure taps are not the materialized top/side pairs")
    taps = _table(probe, 17, times)[-1]
    fluxes = {
        name: _table(next((post / name).rglob("surfaceFieldValue.dat")), 2, times)[:, 0]
        for name in _OBJECTS[1:]
    }
    x = points[:8, 0]
    observable = taps[8:] - taps[:8]  # side minus top, in rho U^2 = sigma U B0^2 L / N
    observable = (observable - observable[(x <= PLATEAU[0]) | (x >= PLATEAU[1])].mean()) / 540.0
    magnitude = np.abs(fluxes["currentIntoSolidMagnitude"])
    return {
        "steps": int(times.size),
        "dt": np.diff(np.concatenate(([0.0], times))).tolist(),
        "courant_max": courant[:, 1].tolist() if courant.size else [],
        "mass_balance": float(np.max(np.abs(fluxes["massIn"] + fluxes["massOut"])) / 4.0),
        "current_balance": float(
            np.max(np.abs(fluxes["currentIn"] + fluxes["currentOut"])) / math.sqrt(540.0)
        ),
        "interface_current_balance": float(
            np.max(np.abs(fluxes["currentIntoSolid"]) / np.maximum(magnitude, 1e-30))
        ),
        "interface_current_activity": float(np.max(magnitude) / math.sqrt(540.0)),
        "x_over_L": x.tolist(),
        "pressure_observable": observable.tolist(),
    }


def freemhd_failures(observed: dict[str, object], limits: dict[str, float]) -> list[str]:
    """The ``[harness_smoke_execution]`` gates that apply to the FreeMHD run."""

    dt, courant = np.asarray(observed["dt"]), np.asarray(observed["courant_max"])
    balances = ("mass_balance", "current_balance", "interface_current_balance")
    passed = {
        "steps": observed["steps"] == limits["executed_steps"]
        and np.all(np.abs(dt - DT) <= limits["dt_absolute_tolerance"]),
        "courant": courant.shape == dt.shape and np.all(courant <= limits["courant_max"]),
        **{gate: observed[gate] <= limits[f"{gate}_max"] for gate in balances},
        "interface_current_activity": observed["interface_current_activity"]
        >= limits["interface_current_activity_min"],
        "pressure": np.all(np.isfinite(observed["pressure_observable"])),
    }
    return [gate for gate, ok in passed.items() if not ok]


def solve_core_flow(nx: int = 400, nz: int = 32, ny: int = 32) -> dict[str, object]:
    """Steady inertialess core flow in the B2 duct (aspect 1, ``c_t = c_s`` from ``[wall]``, unit mean velocity)."""

    from lmhdx.coreflow import CoreFlow

    spec, reference = load_spec(), load_reference()
    geometry, c_w = spec["geometry"], spec["wall"]["wall_conductance_ratio"]
    x = np.linspace(geometry["x_over_L_min"], geometry["x_over_L_max"], nx + 1)
    field = np.interp(x, reference["x_over_L"], reference["b_over_B0"])
    started = time.perf_counter()
    result = CoreFlow(x, aspect=1.0, nz=nz, ny=ny).solve(
        field, c_t=c_w, c_s=c_w, mean_velocity=1.0, beta_max=BETA_MAX
    )
    pressure, flux = np.asarray(result.pressure), np.asarray(result.axial_flux)
    wall_seconds = time.perf_counter() - started
    # Pressure is constant along B: the side tap is the first z centre (half a cell from z = -1,
    # not extrapolated), the top tap the last z centre (half a cell from z = 0).
    observable = pressure[:, 0] - pressure[:, -1]
    observable = observable - observable[(x <= PLATEAU[0]) | (x >= PLATEAU[1])].mean()
    return {
        "mesh": {"nx": nx, "nz": nz, "ny": ny},
        "wall_seconds": wall_seconds,
        "finite": bool(np.all(np.isfinite(pressure)) and np.all(np.isfinite(flux))),
        "floored_nodes": int(result.floored_nodes),
        "weak_field_nodes": [int(np.sum(np.abs(field) * BETA_MAX < bound)) for bound in (1 - 1e-6, 1 + 1e-6)],
        "axial_flux_spread": float(np.ptp(flux) / abs(flux.mean())),
        "x_over_L": x.tolist(),
        "pressure_observable": observable.tolist(),
    }


def core_flow_failures(observed: dict[str, object]) -> list[str]:
    weak = observed["weak_field_nodes"]  # |B| clearly below / not clearly above 1/beta_max
    passed = {
        "finite": observed["finite"],
        "floored_nodes": min(weak) <= observed["floored_nodes"] <= max(weak),
        "axial_flux": observed["axial_flux_spread"] <= AXIAL_FLUX_SPREAD_MAX,
    }
    return [gate for gate, ok in passed.items() if not ok]


def _difference(lmhdx: dict, x_ref, reference) -> dict[str, float]:
    delta = np.interp(x_ref, lmhdx["x_over_L"], lmhdx["pressure_observable"]) - np.asarray(reference)
    return {"rms": float(np.sqrt(np.mean(delta**2))), "linf": float(np.max(np.abs(delta)))}


def run(args: argparse.Namespace) -> int:
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    limits = load_spec()["harness_smoke_execution"]
    artifacts = {"spec": artifact_sha256(SPEC), "reference": artifact_sha256(REFERENCE)}
    artifacts["lmhdx_coreflow"] = artifact_sha256(_DATA.parents[1] / "coreflow.py")
    artifacts["freemhd_input"] = materialize_freemhd_input(root / "freemhd_input")
    skeleton = args.freemhd_install_dir / "cases" / "hunt_demo"
    if skeleton.is_dir():
        artifacts["freemhd_install_skeleton"] = {
            name: artifact_sha256(skeleton / "system" / name, "file") for name in _SKELETON
        }
    if args.freemhd_source_repo.is_dir():
        snapshot_freemhd_source(args.freemhd_source_repo, root / "freemhd_source")
        artifacts["freemhd_source"] = artifact_sha256(root / "freemhd_source")
    elif not args.preflight:
        raise FileNotFoundError("the pinned FreeMHD source repository is required")
    record: dict[str, object] = {"schema_version": 4, "case_id": CASE_ID, "artifacts": artifacts}
    if args.preflight:
        (root / "preflight.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    lmhdx = solve_core_flow()
    lmhdx["failures"] = core_flow_failures(lmhdx)
    seconds = run_freemhd(
        root / "freemhd_input", root / "freemhd_output", args.freemhd_image, args.nproc, args.timeout
    )
    if artifact_sha256(root / "freemhd_input", "tree") != artifacts["freemhd_input"]:
        raise ValueError("the FreeMHD input changed during the run")
    freemhd = observe_freemhd(root / "freemhd_output") | {
        "wall_seconds": seconds,
        "image": args.freemhd_image,
    }
    freemhd["failures"] = freemhd_failures(freemhd, limits)
    artifacts["freemhd_output"] = artifact_sha256(root / "freemhd_output")
    reference = load_reference()
    record |= {
        "freemhd": freemhd,
        "lmhdx": lmhdx,
        "reported_not_gated": {
            "lmhdx_minus_freemhd": _difference(lmhdx, freemhd["x_over_L"], freemhd["pressure_observable"]),
            "lmhdx_minus_alex": _difference(lmhdx, reference["x_over_L"], reference["pressure_observable"]),
            "reason": "FreeMHD is a two-update transient smoke from a plug; LMhdX is the steady inertialess core flow.",
        },
        "execution_pass": not (lmhdx["failures"] or freemhd["failures"]),
    }
    (root / "record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    failures = {"lmhdx": lmhdx["failures"], "freemhd": freemhd["failures"]}
    print(json.dumps({"failures": failures, "reported_not_gated": record["reported_not_gated"]}, indent=2))
    return 0 if record["execution_pass"] else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m validation.freemhd", description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="materialize and hash the inputs; no Docker")
    parser.add_argument(
        "--freemhd-image", default=os.environ.get("LMHDX_FREEMHD_IMAGE", "freemhd-install:latest")
    )
    for flag, default in (("install-dir", "freemhd_install"), ("source-repo", "lmx_external_codes/FreeMHD")):
        variable = "LMHDX_FREEMHD_" + flag.upper().replace("-", "_")
        parser.add_argument(f"--freemhd-{flag}", type=Path, default=Path(os.environ.get(variable, default)))
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1200.0)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
