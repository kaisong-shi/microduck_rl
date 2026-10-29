#!/usr/bin/env bash
# CPU deployment rehearsal: load EVERY walk-model 61D policy into one process.
#
#   ./scripts/infer_all_walk.sh                       # velstand in the walk slot
#   ./scripts/infer_all_walk.sh --walk alpha_walking  # old walk + separate stand
#   ./scripts/infer_all_walk.sh --dry-run             # print the command only
#   ./scripts/infer_all_walk.sh --debug --vin 7.0     # extra flags go to infer_policy.py
#
# All seven sessions stay resident on the shared 61D obs buffer and are swapped
# by command / key press, exactly like the daemon on the robot. Roller policies
# need --roller (scene_rollers.xml, interleaved passive joints) and therefore a
# separate process, so they are deliberately not included here.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."   # XML paths in infer_policy.py are relative

MODELS="${MODELS_DIR:-models/microduck_policies}"
WALK=velstand
DRY_RUN=0
FORWARD=()

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--walk)    WALK="${2:?--walk needs a policy name or path}"; shift 2 ;;
        -d|--models)  MODELS="${2:?--models needs a directory}"; shift 2 ;;
        -n|--dry-run) DRY_RUN=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            FORWARD+=("$1"); shift ;;   # forwarded to infer_policy.py
    esac
done

resolve() {   # policy name or path -> path
    local p="$1"
    [[ "$p" == */* ]] || p="$MODELS/$p"
    [[ "$p" == *.onnx ]] || p="$p.onnx"
    printf '%s' "$p"
}

WALK_PATH="$(resolve "$WALK")"

POLICIES=(
    "$WALK_PATH"
    "$MODELS/alpha_stand.onnx"
    "$MODELS/alpha_sitstand.onnx"
    "$MODELS/alpha_ground_pick.onnx"
    "$MODELS/ball_kick_left.onnx"
    "$MODELS/ball_kick_right.onnx"
    "$MODELS/roulade.onnx"
)

missing=()
for f in "${POLICIES[@]}"; do
    [[ -f "$f" ]] || missing+=("$f")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    printf 'missing policy file(s):\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    printf '\ndownload the set with:\n  hf download pollen-robotics/microduck-policies --include "*.onnx" --local-dir %s\n' "$MODELS" >&2
    exit 1
fi

cmd=(
    uv run scripts/infer_policy.py --new-cmd-obs
    --walking      "${POLICIES[0]}"
    --standing     "${POLICIES[1]}"
    --sitstand     "${POLICIES[2]}"
    --ground-pick  "${POLICIES[3]}"
    --kick-left    "${POLICIES[4]}"
    --kick-right   "${POLICIES[5]}"
    --roulade      "${POLICIES[6]}"
)
[[ ${#FORWARD[@]} -eq 0 ]] || cmd+=("${FORWARD[@]}")

printf 'walk slot : %s\n' "$WALK_PATH"
printf 'loading   : %d policies (61D obs / 14D action)\n' "${#POLICIES[@]}"
printf 'command   :'
printf ' %q' "${cmd[@]}"
printf '\n'

[[ $DRY_RUN -eq 1 ]] && exit 0

exec "${cmd[@]}"
