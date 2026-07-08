#!/usr/bin/env python3
"""Analyze valid_seen task difficulty vs MMP plan complexity (offline stats).

Example:
  export ALFRED_ROOT=/path/to/alfred
  python scripts/analyze_split_difficulty.py
"""

import argparse
import collections
import json
import os
import statistics
import sys
from pathlib import Path

from _paths import REPO_ROOT

COMPLEX_TYPES = {
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
    "pick_two_obj_and_place",
    "pick_and_place_with_movable_recep",
}


def resolve_split_path(alfred_root: str) -> Path:
    for name in ("oct21.json", "oct24.json"):
        path = Path(alfred_root) / "data" / "splits" / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No oct21/oct24 split under {alfred_root}/data/splits/")


def load_records(alfred_root: str):
    split = json.load(open(resolve_split_path(alfred_root), encoding="utf-8"))
    tasks = split["valid_seen"]
    mmp = json.load(open(REPO_ROOT / "MMP_results" / "valid_seen.json", encoding="utf-8"))
    langs = json.load(open(REPO_ROOT / "planner" / "valid_seen_langs.json", encoding="utf-8"))

    records = []
    missing = 0
    for idx, t in enumerate(tasks):
        tt = t["task"].split("-")[0]
        trial = t["task"].split("/")[-1]
        ridx = t["repeat_idx"]

        cands = [(k, d) for k, d in mmp.items() if trial in d.get("root", "")]
        data = {}
        if len(cands) == 1:
            _, data = cands[0]
        elif len(cands) > 1:
            task_key = next((k for k in langs if k.endswith(trial)), None)
            matched = None
            if task_key and str(ridx) in langs[task_key]:
                snippet = langs[task_key][str(ridx)].split("<<goal>>")[0].strip().lower()
                for k, d in cands:
                    if snippet[:25].replace(" ", "") in k.lower().replace(" ", ""):
                        matched = d
                        break
            data = matched or cands[min(ridx, len(cands) - 1)][1]
        else:
            missing += 1

        triplet = data.get("triplet", [])
        if triplet and isinstance(triplet[0], str):
            n_triplet = len(triplet)
        else:
            n_triplet = len(triplet)

        records.append(
            {
                "idx": idx,
                "type": tt,
                "trial": trial,
                "repeat_idx": ridx,
                "n_triplet": n_triplet,
                "n_low": len(data.get("low_actions", [])),
                "sliced": int(
                    any("Sliced" in str(x) for x in (data.get("low_classes") or []))
                ),
            }
        )

    print(f"Loaded {len(records)} tasks, missing MMP match: {missing}")
    return records


def summarize(name, recs):
    if not recs:
        return
    types = collections.Counter(r["type"] for r in recs)
    print(f"\n=== {name} (n={len(recs)}) ===")
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({100 * c / len(recs):.1f}%)")
    print(f"  avg triplet steps: {statistics.mean(r['n_triplet'] for r in recs):.2f}")
    print(f"  avg low actions:   {statistics.mean(r['n_low'] for r in recs):.2f}")
    print(f"  sliced ratio:      {100 * sum(r['sliced'] for r in recs) / len(recs):.1f}%")
    print(
        f"  complex type ratio:{100 * sum(1 for r in recs if r['type'] in COMPLEX_TYPES) / len(recs):.1f}%"
    )
    print(
        f"  simple type ratio: {100 * sum(1 for r in recs if r['type'] == 'pick_and_place_simple') / len(recs):.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze valid_seen difficulty vs MMP plans")
    parser.add_argument("--alfred_root", default=os.environ.get("ALFRED_ROOT", ""))
    args = parser.parse_args()

    if not args.alfred_root:
        print("Set ALFRED_ROOT or pass --alfred_root", file=sys.stderr)
        sys.exit(1)

    records = load_records(args.alfred_root)

    summarize("First 32 [0,32)", records[:32])
    summarize("Next 32 [32,64)", records[32:64])
    summarize("Middle [320,352)", records[320:352])
    summarize("Last 32 [788,820)", records[-32:])
    summarize("Rest [32,820)", records[32:])
    summarize("ALL valid_seen", records)

    q = len(records) // 4
    for i in range(4):
        summarize(f"Quartile Q{i + 1} [{i * q},{(i + 1) * q})", records[i * q : (i + 1) * q])

    first, rest = records[:32], records[32:]
    print("\n=== First 32 vs Rest [32,820) ===")
    print(
        "avg triplet:",
        round(statistics.mean(r["n_triplet"] for r in first), 2),
        "vs",
        round(statistics.mean(r["n_triplet"] for r in rest), 2),
    )
    print(
        "avg low actions:",
        round(statistics.mean(r["n_low"] for r in first), 2),
        "vs",
        round(statistics.mean(r["n_low"] for r in rest), 2),
    )
    heat_clean_cool = {
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
    }
    print(
        "heat/cool/clean %:",
        round(100 * sum(1 for r in first if r["type"] in heat_clean_cool) / 32, 1),
        "vs",
        round(100 * sum(1 for r in rest if r["type"] in heat_clean_cool) / len(rest), 1),
    )
    print(
        "simple %:",
        round(100 * sum(1 for r in first if r["type"] == "pick_and_place_simple") / 32, 1),
        "vs",
        round(
            100 * sum(1 for r in rest if r["type"] == "pick_and_place_simple") / len(rest), 1
        ),
    )

    window = 32
    scores = []
    for i in range(0, len(records) - window + 1, 16):
        chunk = records[i : i + window]
        scores.append(
            (
                i,
                statistics.mean(r["n_triplet"] for r in chunk),
                100 * sum(1 for r in chunk if r["type"] in COMPLEX_TYPES) / window,
            )
        )
    easiest = min(scores, key=lambda x: (x[2], x[1]))
    hardest = max(scores, key=lambda x: (x[2], x[1]))
    print("\n=== Sliding window (32 tasks, step 16) ===")
    print(f"Hardest window start={hardest[0]}: complex%={hardest[2]:.1f}, avg_triplet={hardest[1]:.2f}")
    print(f"Easiest window start={easiest[0]}: complex%={easiest[2]:.1f}, avg_triplet={easiest[1]:.2f}")
    print(f"First window [0,32): complex%={scores[0][2]:.1f}, avg_triplet={scores[0][1]:.2f}")
    rank = sorted(range(len(scores)), key=lambda i: (scores[i][2], scores[i][1]))
    rank0 = rank.index(0)
    print(f"First-32 window ranks #{rank0 + 1}/{len(scores)} by difficulty (1=hardest)")


if __name__ == "__main__":
    main()
