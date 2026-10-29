#!/usr/bin/env bash
# Rebuild / refresh the `microduck_rl` conda environment from uv.lock.
#
# WHY THIS EXISTS
#   The repo's dependency truth is `pyproject.toml` + `uv.lock`. A bare
#   `pip install -e .` does NOT reproduce it: bam pins `protobuf<4.0`, which
#   cascades onnx down to 1.17.0 (no wheel -> source build fails), and bam
#   pulls the broken stub wheel `zmq==0.0.0`. All three are neutralised by
#   `[tool.uv] override-dependencies` in pyproject.toml, which only `uv` reads.
#   So we export the resolved lockfile and feed THAT to uv pip.
#
#   Two more traps this script handles:
#   * `uv export` emits `-e .` as the first line. pip resolves it relative to
#     the requirements file, so the export MUST live in the repo root (a /tmp
#     file installs the wrong project). We ALSO strip that line before
#     installing: uv pip treats it as a project and then rejects the two names
#     of bam (branch `mjlab_frictionloss` in pyproject vs pinned commit
#     62bd8ce1 in the lock) as conflicting URLs. The project is installed
#     afterwards with `--no-deps -e .` so its console scripts and the
#     `mjlab.tasks` entry point are still registered.
#   * conda's libstdc++ can be older than what the torch wheel needs
#     (`GLIBCXX_3.4.30 not found`). We don't install conda's cudatoolkit /
#     cudnn / pytorch on purpose: warp 1.12 bundles CUDA 12.9 and torch's
#     cu12 wheels bring their own runtime. Mixing them is the usual source of
#     "libcudnn not found" / wrong-CUDA-version breakage.
#
# USAGE
#   bash scripts/sync_conda.sh              # create if missing, then sync
#   bash scripts/sync_conda.sh --recreate   # delete the env first (destructive)
#   ENV_NAME=other bash scripts/sync_conda.sh
#
# VERIFY (afterwards): see the gate block printed at the end.

set -euo pipefail

ENV_NAME="${ENV_NAME:-microduck_rl}"
PY_VERSION="${PY_VERSION:-3.12}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="$REPO_ROOT/requirements-conda.txt"

RECREATE=0
for arg in "$@"; do
    case "$arg" in
        --recreate) RECREATE=1 ;;
        -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

cd "$REPO_ROOT"

# conda's shell hook is not active in non-interactive shells.
if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda not on PATH" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "==> repo:  $REPO_ROOT"
echo "==> env:   $ENV_NAME (python $PY_VERSION)"

if [[ "$RECREATE" == "1" ]]; then
    echo "==> removing existing env (destructive, --recreate)"
    conda deactivate 2>/dev/null || true
    conda env remove -n "$ENV_NAME" -y
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> creating conda env"
    conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
fi

ENV_PREFIX="$(conda env list | awk -v n="$ENV_NAME" '$1==n {print $NF}')"
if [[ -z "$ENV_PREFIX" || ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "error: could not resolve the env prefix for $ENV_NAME" >&2
    exit 1
fi
echo "==> prefix: $ENV_PREFIX"

# NOTE: written into the repo root on purpose (see the `-e .` trap above).
#
# The `-e .` line is STRIPPED before installing. Reason: uv pip resolves it as
# a real project, and the project metadata names bam by branch
# (`[tool.uv.sources] branch = "mjlab_frictionloss"`) while the lock pins it by
# commit (62bd8ce1...). Both describe the same thing, but uv pip refuses:
#   "Requirements contain conflicting URLs for package better-actuator-models"
# The project itself is installed afterwards with --no-deps (= its
# dependencies are exactly what the lockfile just installed), which sidesteps
# the URL comparison entirely and still writes the console scripts
# (train/play/publish/...) and the `mjlab.tasks` entry point.
echo "==> exporting locked requirements -> $REQ_FILE"
uv export --no-hashes --no-dev --format requirements-txt \
    | grep -v -x -e '-e .' -e '--editable .' > "$REQ_FILE"

echo "==> installing (uv pip into the conda prefix; no venv)"
uv pip install --python "$ENV_PREFIX/bin/python" -r "$REQ_FILE"

echo "==> installing the project itself (editable, deps already satisfied)"
uv pip install --python "$ENV_PREFIX/bin/python" --no-deps -e "$REPO_ROOT"

echo "==> installing pip itself (some tooling still shells out to it)"
uv pip install --python "$ENV_PREFIX/bin/python" pip

cat <<EOF

==> done. verification gates (run these, they are the whole point):

  conda activate $ENV_NAME

  # G1 interpreter + CUDA
  python -c "import sys,torch; print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

  # G2 warp backend really on cuda
  python -c "import warp as wp; wp.init(); print(wp.get_device('cuda'))"

  # G3 task registry discovered / console script present
  train 2>&1 | head -2

  # G4 single-duck smoke train (watch nvidia-smi in another shell)
  train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5 --gpu-ids '[0]'

  # G5 the ball asset still imports and bam resolves
  python -c "import bam.model, mujoco; print('ok')"
EOF
