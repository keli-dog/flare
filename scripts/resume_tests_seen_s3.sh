#!/usr/bin/env bash
# Resume missing tests_seen episodes [874, 1203) in parallel (s3 crash recovery).
#
# Usage (from repo root, via SSH):
#   bash scripts/resume_tests_seen_s3.sh
#   NUM_SHARDS=2 bash scripts/resume_tests_seen_s3.sh   # if 4-way OOM
#   bash scripts/resume_tests_seen_s3.sh --no-kill
#
# Does NOT delete old logs. New outputs:
#   results/shard_logs/s3_r{1..4}.log
#   results/logs/log_tests_seen_from_{874,957,...}_flare_paper.txt
#   results/analyze_recs/tests_seen_anaylsis_recs_from_*_flare_paper.p
#   results/leaderboard/actseqs_test_seen_flare_paper_*_*.p

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SPLIT="${SPLIT:-tests_seen}"
FROM_IDX="${FROM_IDX:-874}"
TO_IDX="${TO_IDX:-1203}"
DN="${DN:-flare_paper}"
NUM_SHARDS="${NUM_SHARDS:-4}"
GPU_ID="${GPU_ID:-0}"
DISPLAY_ID="${DISPLAY_ID:-1}"
KILL_OLD="${KILL_OLD:-1}"

if [[ "${1:-}" == "--no-kill" ]]; then
    KILL_OLD=0
fi

mkdir -p results/logs results/shard_logs

log() {
    echo "[$(date '+%F %T')] $*"
}

TOTAL=$((TO_IDX - FROM_IDX))
if (( TOTAL <= 0 )); then
    log "ERROR: nothing to run (FROM_IDX=${FROM_IDX}, TO_IDX=${TO_IDX})"
    exit 1
fi

CHUNK=$(( (TOTAL + NUM_SHARDS - 1) / NUM_SHARDS ))
MASTER_LOG="results/logs/resume_s3_${DN}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

log "========== FLARE s3 resume (tests_seen) =========="
log "REPO_ROOT=${REPO_ROOT}"
log "Range: [${FROM_IDX}, ${TO_IDX})  (${TOTAL} episodes)"
log "DN=${DN}  NUM_SHARDS=${NUM_SHARDS}  CHUNK≈${CHUNK}"
log "GPU=${GPU_ID}  DISPLAY=:${DISPLAY_ID}"
log "Master log: ${MASTER_LOG}"
log "================================================="

if [[ "$KILL_OLD" == "1" ]]; then
    if pgrep -f "${DN}" >/dev/null 2>&1; then
        log "Stopping existing processes matching DN=${DN} ..."
        pkill -f "${DN}" || true
        sleep 2
    fi
fi

export ALFRED_ROOT="${ALFRED_ROOT:-/media/ubuntu/Student/gengzeyu/flare/alfred}"
export FILM="$REPO_ROOT"
export MMP_RESULTS_DIR="${MMP_RESULTS_DIR:-MMP_results}"
export DISPLAY=":${DISPLAY_ID}"

CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate flare
    log "conda env: flare ($(which python))"
else
    log "WARN: conda not in PATH"
fi

launch_shard() {
    local start="$1" end="$2" shard_log="$3"
    nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" bash -c "
        set -u
        cd '${REPO_ROOT}'
        if [[ -n '${CONDA_BASE}' && -f '${CONDA_BASE}/etc/profile.d/conda.sh' ]]; then
            source '${CONDA_BASE}/etc/profile.d/conda.sh'
            conda activate flare
        fi
        export ALFRED_ROOT='${ALFRED_ROOT}'
        export FILM='${REPO_ROOT}'
        export MMP_RESULTS_DIR='${MMP_RESULTS_DIR}'
        export DISPLAY='${DISPLAY}'
        echo \"[shard] python=\$(which python 2>/dev/null || echo missing)\"
        echo \"[shard] range=[${start}, ${end})\"
        exec bash eval.sh '${SPLIT}' '${start}' '${end}' '${DN}'
    " >> "${shard_log}" 2>&1 &
}

PIDS=()
SHARD_LOGS=()
START="${FROM_IDX}"
shard_no=0
for ((i = 0; i < NUM_SHARDS; i++)); do
    END=$((START + CHUNK))
    if (( END > TO_IDX )); then
        END=$TO_IDX
    fi
    if (( START >= END )); then
        break
    fi
    shard_no=$((shard_no + 1))
    SHARD_LOG="results/shard_logs/s3_r${shard_no}.log"
    log "Starting shard ${shard_no}: [${START}, ${END}) -> ${SHARD_LOG}"
    launch_shard "${START}" "${END}" "${SHARD_LOG}"
    PIDS+=("$!")
    SHARD_LOGS+=("${SHARD_LOG}")
    log "  PID=${PIDS[-1]}"
    START=$END
done

log ""
log "Launched ${#PIDS[@]} shard(s): PIDs=${PIDS[*]}"
log "Monitor:  tail -f results/shard_logs/s3_r*.log"
log "Progress: python scripts/_check_eps.py  (or see below)"
log ""

sleep 5
alive=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if ps -p "${pid}" >/dev/null 2>&1; then
        alive=$((alive + 1))
        log "Shard $((i + 1)) PID=${pid} running OK."
    else
        log "ERROR: Shard $((i + 1)) PID=${pid} exited early."
        tail -n 15 "${SHARD_LOGS[$i]}" 2>/dev/null | sed 's/^/  /' || true
    fi
done

if (( alive == 0 )); then
    log "All shards died. Try: NUM_SHARDS=1 bash scripts/resume_tests_seen_s3.sh"
    exit 1
fi

nvidia-smi || true
