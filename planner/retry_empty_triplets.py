#!/usr/bin/env python3
"""Retry generate_plans entries with empty triplet (e.g. after Connection error)."""

import json
import os
import time
from argparse import ArgumentParser
from collections import defaultdict

from generate_plans import PLANNER_DIR, build_prompt, build_request_kwargs, get_client, load_task_json


def main():
    parser = ArgumentParser()
    parser.add_argument("--dn", default="minimax-m3")
    parser.add_argument("--split", default="valid_seen", choices=["tests_seen", "tests_unseen", "valid_seen", "valid_unseen"])
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "MiniMax-M3"))
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.minimaxi.com/v1"))
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise SystemExit("Set LLM_API_KEY or MINIMAX_API_KEY")

    alfred_root = os.environ["ALFRED_ROOT"]
    client = get_client(api_key=api_key, base_url=args.base_url)

    output_path = os.path.join(PLANNER_DIR, f"planner_results/{args.dn}/turbo-bias-{args.split}_result.json")
    if not os.path.isfile(output_path):
        raise SystemExit(f"Missing {output_path}")

    result = defaultdict(dict, json.load(open(output_path, encoding="utf-8")))
    splits_path = os.path.join(alfred_root, "data/splits/oct21.json")
    if not os.path.isfile(splits_path):
        splits_path = os.path.join(alfred_root, "data/splits/oct24.json")
    splits = json.load(open(splits_path, encoding="utf-8"))

    retrived = json.load(open(
        os.path.join(PLANNER_DIR, f"few_examples_from_song/few-song-{args.split}_retrieved_keys_clip_Img1_Txt1_panoramic.json"),
        encoding="utf-8",
    ))
    few_examples = json.load(open(os.path.join(PLANNER_DIR, "few_examples_from_song/few_examples.json"), encoding="utf-8"))

    retry = []
    for task in splits[args.split]:
        data = load_task_json(task)
        r_idx = task["repeat_idx"]
        instruction = data["turk_annotations"]["anns"][r_idx]["task_desc"]
        for desc in data["turk_annotations"]["anns"][r_idx]["high_descs"]:
            instruction += desc
        if not result.get(instruction, {}).get("triplet"):
            retry.append((task, instruction, data, r_idx))

    print(f"Retry {args.split} / {args.dn}: {len(retry)} empty triplet(s)")
    if not retry:
        print("Nothing to do.")
        return

    still_failed = []
    for i, (task, instruction, data, r_idx) in enumerate(retry):
        task_id = task["task"]
        goal = "".join(data["ann"]["goal"][j].strip() + " " for j in range(len(data["ann"]["goal"]) - 1))
        inst_list = [x for sub in data["ann"]["instr"] for x in sub]
        high_descs = "".join(inst_list[j].strip() + " " for j in range(len(inst_list) - 1))
        inst = [few_examples[k] for k in retrived[task_id][str(r_idx)]]
        text = build_prompt(inst, goal, high_descs)

        ok = False
        for attempt in range(args.max_retries):
            try:
                resp = client.chat.completions.create(**build_request_kwargs(args.model, text))
                result[instruction]["root"] = os.path.join("data/json_feat_2.1.0", task_id)
                result[instruction]["triplet"] = [resp.choices[0].message.content]
                result[instruction].setdefault("low_actions", [])
                result[instruction].setdefault("low_classes", [])
                result[instruction].setdefault("high_idxs", [])
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=4)
                print(f"[{i + 1}/{len(retry)}] OK")
                ok = True
                break
            except Exception as e:
                print(f"[{i + 1}/{len(retry)}] attempt {attempt + 1}: {e}")
                time.sleep(3 * (attempt + 1))
        if not ok:
            still_failed.append(instruction)
            print(f"[{i + 1}/{len(retry)}] FAILED")
        time.sleep(0.5)

    empty = sum(1 for v in result.values() if not v.get("triplet"))
    print(f"Done. empty triplet left: {empty}")
    if still_failed:
        fail_path = os.path.join(PLANNER_DIR, f"{args.split}failed.json")
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(still_failed, f, indent=4)
        print(f"Updated {fail_path}")


if __name__ == "__main__":
    main()
