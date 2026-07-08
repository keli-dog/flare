#!/usr/bin/env bash
# Summarize tests_seen parallel eval results (analyze_recs + optional actseqs).
# Usage:
#   bash scripts/summarize_tests_seen.sh
#   DN=flare_paper TO_IDX=1533 bash scripts/summarize_tests_seen.sh

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SPLIT="${SPLIT:-tests_seen}"
TO_IDX="${TO_IDX:-1533}"
DN="${DN:-flare_paper}"

mkdir -p results/logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="results/logs/summary_${SPLIT}_${DN}_${TIMESTAMP}.txt"
SUMMARY_TXT="results/summary_${SPLIT}_${DN}.txt"

exec > >(tee -a "$SUMMARY_LOG" | tee "$SUMMARY_TXT") 2>&1

echo "========== FLARE eval summary =========="
echo "Generated: $(date '+%F %T')"
echo "SPLIT=${SPLIT}  TO_IDX=${TO_IDX}  DN=${DN}"
echo "Summary log: ${SUMMARY_LOG}"
echo "Summary txt: ${SUMMARY_TXT}"
echo "========================================"
echo ""

export DN SPLIT TO_IDX REPO_ROOT
python - <<'PY'
import glob
import os
import pickle
import re

dn = os.environ["DN"]
split = os.environ["SPLIT"]
to_idx = int(os.environ["TO_IDX"])

rec_paths = sorted(glob.glob(f"results/analyze_recs/{split}_anaylsis_recs_from_*_{dn}.p"))
act_paths = sorted(glob.glob(f"results/leaderboard/actseqs_test_{split.split('_')[1]}_{dn}_*.p"))

print("## analyze_recs (per shard)")
all_recs = []
seen_eps = set()
for path in rec_paths:
    recs = pickle.load(open(path, "rb"))
    m = re.search(r"from_(\d+)_to_(\d+)", path)
    span = f"[{m.group(1)}, {m.group(2)})" if m else path
    succ = sum(1 for r in recs if r.get("success"))
    print(f"  {os.path.basename(path)}")
    print(f"    span {span}  count={len(recs)}  approx_success={succ}")
    for r in recs:
        ep = r.get("number_of_this_episode")
        if ep is not None:
            if ep in seen_eps:
                print(f"    WARN: duplicate episode {ep} in {path}")
            seen_eps.add(ep)
    all_recs.extend(recs)

print("")
print("## analyze_recs (total)")
print(f"  Unique episodes: {len(seen_eps)} / {to_idx}")
if all_recs:
    approx_sr = sum(1 for r in all_recs if r.get("success")) / len(all_recs)
    print(f"  Total records:   {len(all_recs)}")
    print(f"  Approx SR:       {100 * approx_sr:.2f}%  (tests_seen uses all_completed proxy)")
else:
    print("  No analyze_recs files found.")

print("")
print("## log files (episode markers)")
log_paths = sorted(glob.glob(f"results/logs/log_{split}_from_*_{dn}.txt"))
for path in log_paths:
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
        n_ep = text.count("episode # is")
        n_act = text.count("list of actions is")
        print(f"  {os.path.basename(path)}: episode_markers={n_ep}, list_of_actions={n_act}")
    except Exception as e:
        print(f"  {path}: read error {e}")

print("")
print("## leaderboard actseqs")
if act_paths:
    total_act = 0
    for path in act_paths:
        acts = pickle.load(open(path, "rb"))
        total_act += len(acts)
        print(f"  {os.path.basename(path)}: {len(acts)}")
    print(f"  Total actseq chunks: {total_act}")
else:
    print("  (none yet)")

print("")
print("## Note")
print("  tests_seen sets args.test=True -> no successes/*.p / fails/*.p")
print("  Paper Table 1 Test Seen SR uses official eval; local approx from analyze_recs.")
PY

echo ""
echo "Done."
