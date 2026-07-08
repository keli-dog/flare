#!/usr/bin/env python3
"""Compare GPT-4 (MMP_results) vs Qwen (MMP_results_qwen) eval on the same split slice.

Example:
  export ALFRED_ROOT=/path/to/alfred
  python scripts/compare_gpt_qwen_eval.py --split valid_seen --from_idx 0 --to_idx 32
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime

from _paths import REPO_ROOT


def load_split_tasks(alfred_root, split, from_idx, to_idx):
    splits_path = os.path.join(alfred_root, "data/splits/oct21.json")
    if not os.path.isfile(splits_path):
        splits_path = os.path.join(alfred_root, "data/splits/oct24.json")
    with open(splits_path) as f:
        all_tasks = json.load(f)[split]
    return all_tasks[from_idx:to_idx]


def load_instruction_key(alfred_root, task):
    json_path = os.path.join(
        alfred_root,
        "data/json_2.1.0",
        task["task"],
        "pp",
        f"ann_{task['repeat_idx']}.json",
    )
    with open(json_path) as f:
        traj = json.load(f)
    r_idx = task["repeat_idx"]
    instruction = traj["turk_annotations"]["anns"][r_idx]["task_desc"]
    for desc in traj["turk_annotations"]["anns"][r_idx]["high_descs"]:
        instruction += desc
    return instruction, traj


def load_mmp_plans(mmp_dir, split):
    path = os.path.join(mmp_dir, f"{split}.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_eval_records(results_dir, split, from_idx, to_idx, dn):
    successes_path = os.path.join(
        results_dir,
        "successes",
        f"{split}_successes_from_{from_idx}_to_{to_idx}_{dn}.p",
    )
    failures_path = os.path.join(
        results_dir,
        "fails",
        f"{split}_failures_from_{from_idx}_to_{to_idx}_{dn}.p",
    )
    records = {}
    if os.path.isfile(successes_path):
        for entry in pickle.load(open(successes_path, "rb")):
            key = (entry["trial"], entry["repeat_idx"])
            records[key] = {**entry, "success": True}
    if os.path.isfile(failures_path):
        for entry in pickle.load(open(failures_path, "rb")):
            key = (entry["trial"], entry["repeat_idx"])
            records[key] = {**entry, "success": False}
    return records


def run_eval(repo_root, split, from_idx, to_idx, dn, mmp_dir):
    env = os.environ.copy()
    env["MMP_RESULTS_DIR"] = mmp_dir
    env["FILM"] = repo_root
    cmd = ["bash", "eval.sh", split, str(from_idx), str(to_idx), dn]
    print(f"[RUN] MMP_RESULTS_DIR={mmp_dir} {' '.join(cmd)}")
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


def summarize_records(records):
    if not records:
        return {"count": 0, "success": 0, "sr": 0.0, "avg_spl": 0.0, "avg_gc": 0.0}
    success = sum(1 for r in records.values() if r.get("success"))
    count = len(records)
    avg_spl = sum(r.get("success_spl", 0.0) for r in records.values()) / count
    avg_gc = sum(r.get("goal_condition_success", 0.0) for r in records.values()) / count
    return {
        "count": count,
        "success": success,
        "sr": success / count if count else 0.0,
        "avg_spl": avg_spl,
        "avg_gc": avg_gc,
    }


def triplet_to_str(triplet):
    if not triplet:
        return "(empty)"
    parts = []
    for item in triplet:
        if isinstance(item, list) and len(item) >= 3:
            parts.append(f"{item[0]}({item[1]},{item[2]})")
        else:
            parts.append(str(item))
    return " -> ".join(parts)


def write_report(
    output_path,
    split,
    from_idx,
    to_idx,
    tasks,
    alfred_root,
    gpt_plans,
    qwen_plans,
    gpt_eval,
    qwen_eval,
    gpt_dn,
    qwen_dn,
):
    lines = []
    lines.append("=" * 72)
    lines.append("GPT-4 vs Qwen MMP Eval Comparison")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Split: {split}  indices: [{from_idx}, {to_idx})  ({len(tasks)} entries)")
    lines.append(f"GPT plan dir: MMP_results          eval dn: {gpt_dn}")
    lines.append(f"Qwen plan dir: MMP_results_qwen    eval dn: {qwen_dn}")
    lines.append("=" * 72)
    lines.append("")

    gpt_sum = summarize_records(gpt_eval)
    qwen_sum = summarize_records(qwen_eval)

    lines.append("## Summary")
    lines.append(f"{'Metric':<28} {'GPT-4':>12} {'Qwen':>12} {'Delta':>12}")
    lines.append("-" * 72)
    lines.append(
        f"{'Evaluated episodes':<28} {gpt_sum['count']:>12} {qwen_sum['count']:>12} {qwen_sum['count'] - gpt_sum['count']:>12}"
    )
    lines.append(
        f"{'Success count':<28} {gpt_sum['success']:>12} {qwen_sum['success']:>12} {qwen_sum['success'] - gpt_sum['success']:>12}"
    )
    lines.append(
        f"{'Success rate (SR)':<28} {gpt_sum['sr']:>11.1%} {qwen_sum['sr']:>11.1%} {qwen_sum['sr'] - gpt_sum['sr']:>+11.1%}"
    )
    lines.append(
        f"{'Avg success SPL':<28} {gpt_sum['avg_spl']:>12.3f} {qwen_sum['avg_spl']:>12.3f} {qwen_sum['avg_spl'] - gpt_sum['avg_spl']:>+12.3f}"
    )
    lines.append(
        f"{'Avg goal-condition rate':<28} {gpt_sum['avg_gc']:>12.3f} {qwen_sum['avg_gc']:>12.3f} {qwen_sum['avg_gc'] - gpt_sum['avg_gc']:>+12.3f}"
    )
    lines.append("")

    plan_same = 0
    plan_diff = 0
    plan_missing_gpt = 0
    plan_missing_qwen = 0

    both_success = both_fail = gpt_only = qwen_only = 0
    disagree = []

    lines.append("## Per-episode Details")
    lines.append("")

    for idx, task in enumerate(tasks):
        ep_no = from_idx + idx
        instruction, traj = load_instruction_key(alfred_root, task)
        key = (traj["task_id"], int(task["repeat_idx"]))
        goal = instruction[:80] + ("..." if len(instruction) > 80 else "")

        gpt_plan = gpt_plans.get(instruction)
        qwen_plan = qwen_plans.get(instruction)
        if gpt_plan is None:
            plan_missing_gpt += 1
        if qwen_plan is None:
            plan_missing_qwen += 1

        gpt_trip = (gpt_plan or {}).get("triplet", [])
        qwen_trip = (qwen_plan or {}).get("triplet", [])
        if gpt_trip == qwen_trip and gpt_plan is not None and qwen_plan is not None:
            plan_same += 1
            plan_tag = "SAME"
        elif gpt_plan is not None and qwen_plan is not None:
            plan_diff += 1
            plan_tag = "DIFF"
        else:
            plan_tag = "MISSING"

        gpt_r = gpt_eval.get(key)
        qwen_r = qwen_eval.get(key)
        gpt_ok = gpt_r.get("success") if gpt_r else None
        qwen_ok = qwen_r.get("success") if qwen_r else None

        if gpt_ok is True and qwen_ok is True:
            both_success += 1
        elif gpt_ok is False and qwen_ok is False:
            both_fail += 1
        elif gpt_ok is True and qwen_ok is False:
            gpt_only += 1
            disagree.append((ep_no, key, "GPT only", goal))
        elif gpt_ok is False and qwen_ok is True:
            qwen_only += 1
            disagree.append((ep_no, key, "Qwen only", goal))

        lines.append(f"[{ep_no:3d}] {plan_tag} | GPT: {_fmt_result(gpt_r)} | Qwen: {_fmt_result(qwen_r)}")
        lines.append(f"      trial: {key[0]}")
        lines.append(f"      goal:  {goal}")
        if plan_tag == "DIFF":
            lines.append(f"      GPT plan:  {triplet_to_str(gpt_trip)}")
            lines.append(f"      Qwen plan: {triplet_to_str(qwen_trip)}")
        if gpt_r:
            lines.append(
                f"      GPT  metrics: SPL={gpt_r.get('success_spl', 0):.3f} "
                f"GC={gpt_r.get('goal_condition_success', 0):.3f} steps={gpt_r.get('steps_taken', '?')}"
            )
        if qwen_r:
            lines.append(
                f"      Qwen metrics: SPL={qwen_r.get('success_spl', 0):.3f} "
                f"GC={qwen_r.get('goal_condition_success', 0):.3f} steps={qwen_r.get('steps_taken', '?')}"
            )
        lines.append("")

    lines.append("=" * 72)
    lines.append("## Plan Comparison")
    lines.append(f"Same triplet:     {plan_same}")
    lines.append(f"Different triplet:{plan_diff}")
    lines.append(f"Missing GPT plan: {plan_missing_gpt}")
    lines.append(f"Missing Qwen plan:{plan_missing_qwen}")
    lines.append("")
    lines.append("## Outcome Comparison (where both eval ran)")
    lines.append(f"Both success:  {both_success}")
    lines.append(f"Both fail:     {both_fail}")
    lines.append(f"GPT only win:  {gpt_only}")
    lines.append(f"Qwen only win: {qwen_only}")
    if disagree:
        lines.append("")
        lines.append("Episodes with different outcomes:")
        for ep_no, key, tag, goal in disagree:
            lines.append(f"  [{ep_no}] {tag} | {key[0]} | {goal}")
    lines.append("=" * 72)

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


def _fmt_result(record):
    if record is None:
        return "N/A"
    return "SUCCESS" if record.get("success") else "FAIL"


def main():
    parser = argparse.ArgumentParser(description="Compare GPT vs Qwen MMP eval results")
    parser.add_argument("--split", default="valid_seen")
    parser.add_argument("--from_idx", type=int, default=0)
    parser.add_argument("--to_idx", type=int, default=32)
    parser.add_argument("--repo_root", default=REPO_ROOT)
    parser.add_argument("--alfred_root", default=os.environ.get("ALFRED_ROOT", ""))
    parser.add_argument("--gpt_mmp_dir", default="MMP_results")
    parser.add_argument("--qwen_mmp_dir", default="MMP_results_qwen")
    parser.add_argument("--gpt_dn", default="gpt_mmp_32", help="--set_dn for GPT eval")
    parser.add_argument("--qwen_dn", default="qwen_mmp_32", help="--set_dn for Qwen eval")
    parser.add_argument(
        "--output",
        default="results/compare_gpt_qwen_valid_seen_0_32.txt",
        help="Output text report path",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run eval.sh for both GPT and Qwen before comparing",
    )
    args = parser.parse_args()

    if not args.alfred_root:
        print("Set ALFRED_ROOT or pass --alfred_root", file=sys.stderr)
        sys.exit(1)
    os.environ["ALFRED_ROOT"] = args.alfred_root

    gpt_mmp = os.path.join(args.repo_root, args.gpt_mmp_dir)
    qwen_mmp = os.path.join(args.repo_root, args.qwen_mmp_dir)

    if args.run:
        run_eval(args.repo_root, args.split, args.from_idx, args.to_idx, args.gpt_dn, gpt_mmp)
        run_eval(args.repo_root, args.split, args.from_idx, args.to_idx, args.qwen_dn, qwen_mmp)

    tasks = load_split_tasks(args.alfred_root, args.split, args.from_idx, args.to_idx)
    gpt_plans = load_mmp_plans(gpt_mmp, args.split)
    qwen_plans = load_mmp_plans(qwen_mmp, args.split)

    results_dir = os.path.join(args.repo_root, "results")
    gpt_eval = load_eval_records(
        results_dir, args.split, args.from_idx, args.to_idx, args.gpt_dn
    )
    qwen_eval = load_eval_records(
        results_dir, args.split, args.from_idx, args.to_idx, args.qwen_dn
    )

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.repo_root, output_path)

    report = write_report(
        output_path,
        args.split,
        args.from_idx,
        args.to_idx,
        tasks,
        args.alfred_root,
        gpt_plans,
        qwen_plans,
        gpt_eval,
        qwen_eval,
        args.gpt_dn,
        args.qwen_dn,
    )
    print(report)
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
