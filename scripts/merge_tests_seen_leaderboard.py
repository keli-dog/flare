#!/usr/bin/env python3
"""Merge shard actseq pickles into ALFRED leaderboard JSON (tests_seen only).

Eval writes one pickle per shard:
  results/leaderboard/actseqs_test_seen_{dn}_{from}_{to}.p

This script merges them (dedupe by trial id), validates count, writes JSON.

Usage:
  python scripts/merge_tests_seen_leaderboard.py --dn flare_paper
  python scripts/merge_tests_seen_leaderboard.py --dn flare_paper --dry-run

Output:
  leaderboard_jsons/tests_actseqs_{json_name}.json

Submit JSON at:
  https://leaderboard.allenai.org/alfred/submissions/public
"""

import argparse
import glob
import json
import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def filter_actions(actions):
    if isinstance(actions, dict):
        return [actions]
    if not isinstance(actions, list):
        return []
    out = []
    for action in actions:
        if action.get("action") in ("LookDown_0", "LookUp_0"):
            continue
        out.append(action)
    return out


def load_merged_actseqs(dn: str):
    pattern = os.path.join(
        REPO_ROOT, f"results/leaderboard/actseqs_test_seen_{dn}_*.p"
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No actseq pickles: {pattern}")

    by_trial = {}
    per_file = []
    for path in paths:
        entries = pickle.load(open(path, "rb"))
        n = 0
        for item in entries:
            if not item or item == "":
                continue
            key = list(item.keys())[0]
            trial = key[1] if isinstance(key, tuple) else key
            actions = filter_actions(item[key])
            by_trial[trial] = actions
            n += 1
        per_file.append((os.path.basename(path), n, len(entries)))

    merged = [{trial: actions} for trial, actions in sorted(by_trial.items())]
    return merged, paths, per_file


def main():
    parser = argparse.ArgumentParser(description="Merge tests_seen actseqs for ALFRED leaderboard")
    parser.add_argument("--dn", default="flare_paper", help="set_dn used in eval.sh")
    parser.add_argument("--json-name", default=None, help="output basename (default: same as --dn)")
    parser.add_argument("--expected", type=int, default=1533)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="write JSON even if count < expected (for inspection only)",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate only, do not write JSON")
    args = parser.parse_args()

    json_name = args.json_name or args.dn
    merged, paths, per_file = load_merged_actseqs(args.dn)

    print("## actseq shard files")
    for name, n_used, n_raw in per_file:
        print(f"  {name}: {n_raw} entries, {n_used} with actseq")
    print(f"\nUnique trials after merge: {len(merged)} / {args.expected}")

    if len(merged) != args.expected and not args.allow_partial:
        print(
            f"\nERROR: expected {args.expected} trials, got {len(merged)}. "
            "Finish eval first, or use --allow-partial for preview.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ALFRED format: both keys; tests_unseen empty if not evaluated yet
    results = {"tests_seen": merged, "tests_unseen": []}

    if args.dry_run:
        print("\nDry run OK.")
        return

    out_dir = os.path.join(REPO_ROOT, "leaderboard_jsons")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"tests_actseqs_{json_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, sort_keys=True)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nWrote: {out_path} ({size_mb:.1f} MB)")
    print("\nNext: upload this JSON to ALFRED leaderboard (see gengzeyu.md §8).")


if __name__ == "__main__":
    main()
