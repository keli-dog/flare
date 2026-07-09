#!/usr/bin/env python3
"""Analyze tests_seen eval results from analyze_recs + shard logs."""

import collections
import glob
import pickle
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REC_GLOB = str(REPO / "results/analyze_recs/tests_seen_*_flare_paper.p")
LOG_GLOBS = [
    REPO / "results/logs/log_tests_seen_*_flare_paper.txt",
    REPO / "results/shard_logs/s*.log",
]


def load_episode_records():
    by_ep = {}
    for path in sorted(glob.glob(REC_GLOB)):
        recs = pickle.load(open(path, "rb"))
        for r in recs:
            ep = r.get("number_of_this_episode")
            if ep is None:
                continue
            by_ep[int(ep)] = r
    return by_ep


def parse_log_markers():
    """Parse episode endings from log files."""
    episodes = {}
    ep_re = re.compile(r"episode # is (\d+)")
    for pattern in LOG_GLOBS:
        for path in sorted(glob.glob(str(pattern))):
            text = open(path, encoding="utf-8", errors="replace").read()
            blocks = text.split("===================================================")
            for block in blocks:
                m = ep_re.search(block)
                if not m:
                    continue
                ep = int(m.group(1))
                info = episodes.get(ep, {
                    "probably_success": False,
                    "max_steps": False,
                    "interact_fail": False,
                    "llm_giveup": False,
                })
                if "This episode is probably Success!" in block:
                    info["probably_success"] = True
                if "This outputted" in block:
                    info["max_steps"] = True
                if "Interact API failed 10 times" in block:
                    info["interact_fail"] = True
                if "LLM gaved up" in block:
                    info["llm_giveup"] = True
                episodes[ep] = info
    return episodes


def main():
    by_ep = load_episode_records()
    log_eps = parse_log_markers()

    total = len(by_ep)
    successes = sum(1 for r in by_ep.values() if r.get("success"))
    sr = successes / total if total else 0

    print("=" * 60)
    print("tests_seen 当前已完成分析 (flare_paper / GPT-4 MMP)")
    print("=" * 60)
    print(f"已完成 episode 数: {total} / 1533")
    print(f"缺失 episode 数:     {1533 - total}")
    print(f"Success (all_completed): {successes} ({100*sr:.2f}%)")
    print(f"Failure:               {total - successes} ({100*(1-sr):.2f}%)")
    print()

    # By task type
    by_type = collections.defaultdict(lambda: {"n": 0, "succ": 0})
    for r in by_ep.values():
        tt = r.get("task_type") or "unknown"
        by_type[tt]["n"] += 1
        if r.get("success"):
            by_type[tt]["succ"] += 1

    print("## 按任务类型 SR")
    print(f"{'task_type':<45} {'n':>5} {'SR':>8}")
    print("-" * 60)
    for tt, d in sorted(by_type.items(), key=lambda x: -(x[1]["n"])):
        rate = d["succ"] / d["n"] if d["n"] else 0
        print(f"{tt:<45} {d['n']:>5} {100*rate:>7.1f}%")
    print()

    # Failure analysis from analyze_recs fields
    fail_recs = [r for r in by_ep.values() if not r.get("success")]
    goal_not_found = sum(1 for r in fail_recs if not r.get("goal_found"))
    has_errs = sum(1 for r in fail_recs if r.get("errs"))
    avg_pointer_fail = (
        sum(r.get("action_pointer") or 0 for r in fail_recs) / len(fail_recs)
        if fail_recs else 0
    )
    avg_pointer_succ = (
        sum(r.get("action_pointer") or 0 for r in by_ep.values() if r.get("success"))
        / successes
        if successes else 0
    )

    print("## 失败原因（analyze_recs 字段）")
    print(f"  goal_found=False:     {goal_not_found} / {len(fail_recs)} failures")
    print(f"  errs 非空:            {has_errs} / {len(fail_recs)} failures")
    print(f"  失败 avg action_pointer: {avg_pointer_fail:.2f}")
    print(f"  成功 avg action_pointer: {avg_pointer_succ:.2f}")
    print()

    # Log-based failure modes (subset with log markers)
    log_total = len([ep for ep in by_ep if ep in log_eps])
    probably = sum(1 for ep in by_ep if log_eps.get(ep, {}).get("probably_success"))
    max_steps = sum(1 for ep in by_ep if log_eps.get(ep, {}).get("max_steps"))
    interact = sum(1 for ep in by_ep if log_eps.get(ep, {}).get("interact_fail"))
    giveup = sum(1 for ep in by_ep if log_eps.get(ep, {}).get("llm_giveup"))

    print("## 失败模式（日志标记，可重叠）")
    print(f"  有日志记录的 episode: {log_total}")
    print(f"  probably Success:     {probably}")
    print(f"  步数耗尽 (outputted): {max_steps}")
    print(f"  Interact API 10次失败: {interact}")
    print(f"  LLM gave up:          {giveup}")
    print()

    # Missing range
    missing = [i for i in range(1533) if i not in by_ep]
    if missing:
        print("## 缺失区间")
        for a, b in [(0, 204), (204, 537), (537, 870), (870, 1203), (1203, 1533)]:
            m = [x for x in missing if a <= x < b]
            if m:
                print(f"  [{a},{b}): {len(m)} 条 (e.g. {m[0]}..{m[-1]})")
        print()

    # Shard comparison
    print("## 各分片 SR（按 episode 区间）")
    ranges = [(0, 204, "旧单路"), (204, 537, "s1"), (537, 870, "s2"),
              (870, 1203, "s3"), (1203, 1533, "s4")]
    for a, b, name in ranges:
        recs = [by_ep[i] for i in range(a, b) if i in by_ep]
        if not recs:
            print(f"  {name} [{a},{b}): 0/{b-a} 完成")
            continue
        s = sum(1 for r in recs if r.get("success"))
        print(f"  {name} [{a},{b}): {len(recs)}/{b-a} 完成, SR={100*s/len(recs):.1f}% ({s}/{len(recs)})")

    print()
    print("## 说明")
    print("  tests_seen 的 success=all_completed（完成全部高层动作），非官方 leaderboard SR。")
    print("  论文 Table 1 Test Seen ~40% 需官方 eval；本地 all_completed 通常偏高或口径不同。")


if __name__ == "__main__":
    main()
