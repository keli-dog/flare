#!/usr/bin/env bash
# Launch parallel tests_seen eval shards on one or more GPUs.
# Usage:
#   bash scripts/run_tests_seen_parallel.sh
#   FROM_IDX=201 NUM_SHARDS=4 DN=flare_paper bash scripts/run_tests_seen_parallel.sh
#   bash scripts/run_tests_seen_parallel.sh --no-kill   # do not stop existing jobs

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SPLIT="${SPLIT:-tests_seen}"
TO_IDX="${TO_IDX:-1533}"
DN="${DN:-flare_paper}"
NUM_SHARDS="${NUM_SHARDS:-4}"
GPU_ID="${GPU_ID:-0}"
DISPLAY_ID="${DISPLAY_ID:-1}"
KILL_OLD="${KILL_OLD:-1}"

if [[ "${1:-}" == "--no-kill" ]]; then
    KILL_OLD=0
fi

mkdir -p results/logs results/shard_logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="results/logs/run_${SPLIT}_parallel_${DN}_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%F %T')] $*"
}

export DN SPLIT

# Auto-detect resume point from analyze_recs if FROM_IDX not set
if [[ -z "${FROM_IDX:-}" ]]; then
    FROM_IDX="$(python - <<'PY'
import glob, os, pickle

dn = os.environ.get("DN", "flare_paper")
split = os.environ.get("SPLIT", "tests_seen")
pattern = f"results/analyze_recs/{split}_anaylsis_recs_from_*_{dn}.p"
episodes = set()
for path in glob.glob(pattern):
    try:
        for r in pickle.load(open(path, "rb")):
            ep = r.get("number_of_this_episode")
            if ep is not None:
                episodes.add(int(ep))
    except Exception:
        pass
print(max(episodes) + 1 if episodes else 0)
PY
)"
    log "Auto-detected FROM_IDX=${FROM_IDX} (max completed episode + 1)"
else
    log "Using FROM_IDX=${FROM_IDX}"
fi

TOTAL=$((TO_IDX - FROM_IDX))
if (( TOTAL <= 0 )); then
    log "ERROR: nothing to run (FROM_IDX=${FROM_IDX}, TO_IDX=${TO_IDX})"
    exit 1
fi

CHUNK=$(( (TOTAL + NUM_SHARDS - 1) / NUM_SHARDS ))

exec > >(tee -a "$MASTER_LOG") 2>&1

log "========== FLARE parallel eval =========="
log "REPO_ROOT=${REPO_ROOT}"
log "SPLIT=${SPLIT}  FROM_IDX=${FROM_IDX}  TO_IDX=${TO_IDX}  DN=${DN}"
log "NUM_SHARDS=${NUM_SHARDS}  CHUNK≈${CHUNK}  GPU=${GPU_ID}  DISPLAY=:${DISPLAY_ID}"
log "Master log: ${MASTER_LOG}"
log "========================================="

if [[ "$KILL_OLD" == "1" ]]; then
    if pgrep -f "${DN}" >/dev/null 2>&1; then
        log "Stopping existing processes matching DN=${DN} ..."
        pkill -f "${DN}" || true
        sleep 2
    else
        log "No existing processes for DN=${DN}"
    fi
fi

export ALFRED_ROOT="${ALFRED_ROOT:-/media/ubuntu/Student/gengzeyu/flare/alfred}"
export FILM="$REPO_ROOT"
export MMP_RESULTS_DIR="${MMP_RESULTS_DIR:-MMP_results}"
export DISPLAY=":${DISPLAY_ID}"

log "ALFRED_ROOT=${ALFRED_ROOT}"
log "MMP_RESULTS_DIR=${MMP_RESULTS_DIR}"

if ! command -v conda >/dev/null 2>&1; then
    log "WARN: conda not in PATH; ensure flare env is active"
else
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate flare
    log "conda env: flare"
fi

PIDS=()
START="${FROM_IDX}"
for ((i = 0; i < NUM_SHARDS; i++)); do
    END=$((START + CHUNK))
    if (( END > TO_IDX )); then
        END=$TO_IDX
    fi
    if (( START >= END )); then
        break
    fi

    SHARD_LOG="results/shard_logs/${SPLIT}_${START}_${END}_${DN}.log"
    log "Starting shard $((i + 1)): [${START}, ${END}) -> ${SHARD_LOG}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup bash eval.sh "${SPLIT}" "${START}" "${END}" "${DN}" \
        >> "${SHARD_LOG}" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    log "  PID=${pid}"

  START=$END
done

log ""
log "Launched ${#PIDS[@]} shard(s): PIDs=${PIDS[*]}"
log "Monitor GPU:  watch -n 30 nvidia-smi"
log "Monitor logs: tail -f results/shard_logs/${SPLIT}_*_${DN}.log"
log "Summarize:    bash scripts/summarize_tests_seen.sh"
log ""

sleep 3
if ps -p "${PIDS[0]}" >/dev/null 2>&1; then
    log "First shard still running (OK)."
else
    log "WARN: first shard may have exited early; check shard logs."
fi

nvidia-smi || true
