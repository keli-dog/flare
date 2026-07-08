#!/usr/bin/env python3
"""Compare MMP plans against ALFRED expert high-level plans (Table-2 style, offline).

There is NO built-in plan-accuracy script in this repo. This tool:
  1. Loads tasks from oct21.json (same order as eval.sh)
  2. Reads expert plan from traj JSON plan.high_pddl
  3. Converts expert + MMP plans to FLARE-style triplets
  4. Reports strict / relaxed match rates (LLM-Planner static HLP ACC style)

Requires ALFRED_ROOT pointing to a local alfred clone with data/ installed.

Example (server):
  export ALFRED_ROOT=/media/ubuntu/Student/gengzeyu/flare/alfred
  python compare_plan_gt.py --split valid_seen --from_idx 0 --to_idx 32 \\
      --mmp-dirs MMP_results MMP_results_qwen --labels GPT-4 Qwen
"""

import argparse
import json
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from planner.postprocess import parse_plan_to_triplet  # noqa: E402

try:
    import alfred_utils.gen.constants as alfred_constants
except ImportError:
    alfred_constants = None

INTERACTION_ACTIONS = {
    "PickupObject",
    "PutObject",
    "SliceObject",
    "CleanObject",
    "HeatObject",
    "CoolObject",
    "ToggleObject",
}

SKIP_ACTIONS = {"GotoLocation", "OpenObject", "CloseObject", "End", "NoOp"}

RECEPTACLE_ALIASES = {
    "sink": "SinkBasin",
    "sinkbasin": "SinkBasin",
    "bathtub": "BathtubBasin",
    "bathtubbasin": "BathtubBasin",
    "garbagecan": "GarbageCan",
    "trashcan": "GarbageCan",
    "floorlamp": "FloorLamp",
    "diningtable": "DiningTable",
    "coffeetable": "CoffeeTable",
    "sidetable": "SideTable",
    "countertop": "CounterTop",
    "stoveburner": "StoveBurner",
    "tvstand": "TVStand",
    "toiletpaperhanger": "ToiletPaperHanger",
}

DEFAULT_RECEP_FOR_ACTION = {
    "CleanObject": "SinkBasin",
    "HeatObject": "Microwave",
    "CoolObject": "Fridge",
}


def load_split_tasks(alfred_root, split, from_idx, to_idx):
    for name in ("oct21.json", "oct24.json"):
        path = os.path.join(alfred_root, "data/splits", name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)[split][from_idx:to_idx]
    raise FileNotFoundError(
        f"Split file not found under {alfred_root}/data/splits/ (need oct21.json)"
    )


def load_traj(alfred_root, task):
    json_path = os.path.join(
        alfred_root,
        "data/json_2.1.0",
        task["task"],
        "pp",
        f"ann_{task['repeat_idx']}.json",
    )
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_instruction_key(traj, repeat_idx):
    instruction = traj["turk_annotations"]["anns"][repeat_idx]["task_desc"]
    for desc in traj["turk_annotations"]["anns"][repeat_idx]["high_descs"]:
        instruction += desc
    return instruction


def normalize_name(name):
    if not name or name in ("0", "-", "None", "none"):
        return "0"
    key = str(name).strip().replace(" ", "").lower()
    if key in RECEPTACLE_ALIASES:
        return RECEPTACLE_ALIASES[key]
    if alfred_constants is not None:
        mapped = alfred_constants.OBJECTS_LOWER_TO_UPPER.get(key)
        if mapped:
            return mapped
    # fallback: PotatoSliced style
    if key.endswith("sliced"):
        base = key[:-6]
        base_norm = RECEPTACLE_ALIASES.get(base, base.capitalize())
        return base_norm + "Sliced"
    return key.capitalize()


def normalize_triplet(action, obj, recep):
    return [action, normalize_name(obj), normalize_name(recep)]


def triplet_to_str(triplets):
    if not triplets:
        return "(empty)"
    parts = []
    for item in triplets:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            parts.append(f"{item[0]}({item[1]},{item[2]})")
        else:
            parts.append(str(item))
    return " -> ".join(parts)


def expert_high_pddl_to_triplets(traj):
    """Convert ALFRED expert plan.high_pddl to FLARE-style triplets."""
    triplets = []
    last_goto = "0"

    for entry in traj.get("plan", {}).get("high_pddl", []):
        discrete = entry.get("discrete_action") or {}
        action = discrete.get("action", "")
        args = discrete.get("args") or []

        if action in SKIP_ACTIONS:
            if action == "GotoLocation" and args:
                last_goto = normalize_name(args[0])
            continue

        if action not in INTERACTION_ACTIONS:
            continue

        if action == "PutObject" and len(args) >= 2:
            triplets.append(normalize_triplet(action, args[0], args[1]))
        elif action == "PickupObject" and len(args) >= 1:
            triplets.append(normalize_triplet(action, args[0], last_goto))
        elif action == "SliceObject" and len(args) >= 1:
            triplets.append(normalize_triplet(action, args[0], last_goto))
        elif action in ("CleanObject", "HeatObject", "CoolObject") and len(args) >= 1:
            triplets.append(
                normalize_triplet(action, args[0], DEFAULT_RECEP_FOR_ACTION[action])
            )
        elif action == "ToggleObject" and len(args) >= 1:
            triplets.append(normalize_triplet(action, args[0], last_goto))
        last_goto = "0"

    return triplets


def mmp_entry_to_triplets(plan_entry):
    if not plan_entry:
        return []
    triplets = plan_entry.get("triplet") or []
    if triplets and isinstance(triplets[0], str):
        parsed = parse_plan_to_triplet(plan_entry)
    elif triplets and isinstance(triplets[0], list):
        parsed = [[t[0], t[1], t[2]] for t in triplets if len(t) >= 3]
    else:
        parsed = []
    return [normalize_triplet(a, o, r) for a, o, r in parsed]


def strict_match(pred, gt):
    return pred == gt


def relaxed_match(pred, gt):
    """Match action+object; receptacle wildcard if either side uses '0'."""
    if len(pred) != len(gt):
        return False
    for p, g in zip(pred, gt):
        if p[0] != g[0] or p[1] != g[1]:
            return False
        if p[2] != g[2] and "0" not in (p[2], g[2]):
            return False
    return True


def step_match_ratio(pred, gt):
    if not pred and not gt:
        return 1.0
    n = max(len(pred), len(gt))
    if n == 0:
        return 0.0
    hits = 0
    for i in range(min(len(pred), len(gt))):
        if pred[i] == gt[i]:
            hits += 1
        elif relaxed_match([pred[i]], [gt[i]]):
            hits += 0.5
    return hits / n


def first_mismatch(pred, gt):
    n = max(len(pred), len(gt))
    for i in range(n):
        if i >= len(pred):
            return i, "(missing step)", gt[i]
        if i >= len(gt):
            return i, pred[i], "(missing step)"
        if pred[i] != gt[i]:
            return i, pred[i], gt[i]
    return None


def evaluate_plans(mmp_dir, split, tasks, alfred_root):
    plans = json.load(open(os.path.join(mmp_dir, f"{split}.json"), encoding="utf-8"))
    rows = []
    for idx, task in enumerate(tasks):
        traj = load_traj(alfred_root, task)
        instr_key = build_instruction_key(traj, task["repeat_idx"])
        gt = expert_high_pddl_to_triplets(traj)
        pred = mmp_entry_to_triplets(plans.get(instr_key))
        trial = task["task"].split("/")[-1]
        rows.append(
            {
                "episode_idx": idx,
                "trial": trial,
                "repeat_idx": task["repeat_idx"],
                "task_type": task["task"].split("-")[0],
                "goal": traj["turk_annotations"]["anns"][task["repeat_idx"]]["task_desc"],
                "gt": gt,
                "pred": pred,
                "strict": strict_match(pred, gt),
                "relaxed": relaxed_match(pred, gt),
                "step_ratio": step_match_ratio(pred, gt),
                "mismatch": first_mismatch(pred, gt),
            }
        )
    return rows


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {}
    return {
        "count": n,
        "strict_acc": sum(r["strict"] for r in rows) / n,
        "relaxed_acc": sum(r["relaxed"] for r in rows) / n,
        "avg_step_ratio": sum(r["step_ratio"] for r in rows) / n,
        "empty_plans": sum(1 for r in rows if not r["pred"]),
    }


def write_report(path, split, from_idx, to_idx, labels, all_rows):
    lines = []
    lines.append("=" * 72)
    lines.append("MMP Plan vs Expert GT Comparison (offline, Table-2 style)")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Split: {split}  indices: [{from_idx}, {to_idx})")
    lines.append("GT source: ALFRED traj plan.high_pddl -> FLARE triplets")
    lines.append("Strict: exact triplet sequence match (LLM-Planner static HLP ACC style)")
    lines.append("Relaxed: match action+object; recep '0' is wildcard")
    lines.append("=" * 72)

    for label, rows in zip(labels, all_rows):
        s = summarize(rows)
        lines.append("")
        lines.append(f"## {label}")
        lines.append(f"Strict plan accuracy:  {100 * s['strict_acc']:.1f}% ({int(s['strict_acc'] * s['count'])}/{s['count']})")
        lines.append(f"Relaxed plan accuracy: {100 * s['relaxed_acc']:.1f}% ({int(s['relaxed_acc'] * s['count'])}/{s['count']})")
        lines.append(f"Avg step match ratio:  {s['avg_step_ratio']:.3f}")
        lines.append(f"Empty plans:           {s['empty_plans']}")

    if len(all_rows) == 2:
        lines.append("")
        lines.append("## GPT vs Qwen plan (ignore GT)")
        a, b = all_rows[0], all_rows[1]
        same = sum(1 for x, y in zip(a, b) if x["pred"] == y["pred"])
        lines.append(f"Same triplet sequence: {same}/{len(a)}")

    lines.append("")
    lines.append("## Per-episode Details")
    base_rows = all_rows[0]
    for i, row in enumerate(base_rows):
        lines.append("")
        flags = []
        for label, rows in zip(labels, all_rows):
            r = rows[i]
            flags.append(f"{label}: strict={'OK' if r['strict'] else 'FAIL'} relaxed={'OK' if r['relaxed'] else 'FAIL'}")
        lines.append(f"[{row['episode_idx']:3d}] {' | '.join(flags)}")
        lines.append(f"      trial: {row['trial']}  repeat: {row['repeat_idx']}  type: {row['task_type']}")
        lines.append(f"      goal:  {row['goal'][:90]}")
        lines.append(f"      GT:    {triplet_to_str(row['gt'])}")
        for label, rows in zip(labels, all_rows):
            r = rows[i]
            lines.append(f"      {label:5s}: {triplet_to_str(r['pred'])}")
            if r["mismatch"] and not r["strict"]:
                mi, pred_step, gt_step = r["mismatch"]
                lines.append(f"             first diff @ step {mi}: pred={pred_step}  gt={gt_step}")

    lines.append("")
    lines.append("=" * 72)
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Compare MMP plans with ALFRED expert GT.")
    parser.add_argument("--split", default="valid_seen")
    parser.add_argument("--from_idx", type=int, default=0)
    parser.add_argument("--to_idx", type=int, default=32)
    parser.add_argument(
        "--alfred-root",
        default=os.environ.get("ALFRED_ROOT", ""),
        help="Path to alfred repo (or set ALFRED_ROOT)",
    )
    parser.add_argument(
        "--mmp-dirs",
        nargs="+",
        default=["MMP_results"],
        help="One or more MMP result directories",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for each --mmp-dirs entry",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Report path (default: results/compare_plan_gt_{split}_{from}_{to}.txt)",
    )
    args = parser.parse_args()

    if not args.alfred_root:
        print("ERROR: set ALFRED_ROOT or pass --alfred-root", file=sys.stderr)
        sys.exit(1)

    labels = args.labels or [os.path.basename(d.rstrip("/\\")) for d in args.mmp_dirs]
    if len(labels) != len(args.mmp_dirs):
        print("ERROR: --labels count must match --mmp-dirs", file=sys.stderr)
        sys.exit(1)

    tasks = load_split_tasks(args.alfred_root, args.split, args.from_idx, args.to_idx)
    all_rows = []
    for mmp_dir in args.mmp_dirs:
        path = os.path.join(REPO_ROOT, mmp_dir) if not os.path.isabs(mmp_dir) else mmp_dir
        if not os.path.isfile(os.path.join(path, f"{args.split}.json")):
            print(f"ERROR: missing {path}/{args.split}.json", file=sys.stderr)
            sys.exit(1)
        all_rows.append(evaluate_plans(path, args.split, tasks, args.alfred_root))

    out = args.output or os.path.join(
        REPO_ROOT,
        "results",
        f"compare_plan_gt_{args.split}_{args.from_idx}_{args.to_idx}.txt",
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    report = write_report(out, args.split, args.from_idx, args.to_idx, labels, all_rows)
    print(report)
    print(f"Report saved to: {out}")


if __name__ == "__main__":
    main()
