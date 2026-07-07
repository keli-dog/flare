import os
import json
import random
import time
from collections import defaultdict
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from openai import OpenAI
from tqdm import tqdm

random.seed(2)

DEFAULT_BASE_URL = "https://llm-1316nm7shgmikwyq.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-plus"

PLANNER_DIR = os.path.dirname(os.path.abspath(__file__))
failed = []
ALFRED_ROOT = os.environ["ALFRED_ROOT"]


def get_client(api_key=None, base_url=None):
    api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "Set LLM_API_KEY (or DASHSCOPE_API_KEY) before running generate_plans.py."
        )
    base_url = base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def load_task_json(task):
    json_path = os.path.join(
        f"{ALFRED_ROOT}/data/json_2.1.0",
        task["task"],
        "pp",
        "ann_%d.json" % task["repeat_idx"],
    )
    with open(json_path) as f:
        return json.load(f)


def build_prompt(inst, goal, high_descs):
    text = """Create a high-level plan for completing a household task using the allowed actions and objects.
Allowed actions: ToggleObject, CleanObject, HeatObject, PickupObject, SliceObject, CoolObject, PutObject
Allowed objects: AlarmClock, Apple, AppleSliced, ArmChair, BaseballBat, BasketBall, Bathtub, Bed, Book, Bowl, Box, Bread, BreadSliced, ButterKnife, CD, Cabinet, Candle, Cart, CellPhone, Cloth, CoffeeMachine, CoffeeTable, CounterTop, CreditCard, Cup, Desk, DeskLamp, DiningTable, DishSponge, Drawer, Dresser, Egg, FloorLamp, Fork, Fridge, GarbageCan, Glassbottle, HandTowel, Kettle, KeyChain, Knife, Ladle, Laptop, Lettuce, LettuceSliced, Microwave, Mug, Newspaper, Ottoman, Pan, Pen, Pencil, PepperShaker, Pillow, Plate, Plunger, Pot, Potato, PotatoSliced, RemoteControl, Safe, SaltShaker, Shelf, SideTable, Sink, SoapBar, SoapBottle, Sofa, Spatula, Spoon, SprayBottle, Statue, StoveBurner, TennisRacket, TissueBox, Toilet, ToiletPaper, ToiletPaperHanger, Tomato, TomatoSliced, Vase, Watch, WateringCan, WineBottle
"""

    for example in inst[:9]:
        text += f"""Task description: {example['goal']}
Step-by-step instructions: {example['instruction'][:-1]}
Next plan: {example['pddl'][:-2]}
"""

    text += f"""Task description: {goal[:-1]}
Step-by-step instructions: {high_descs[:-1]}
Next plan: """
    return text


def main(sp, destination, client, model, use_gpt4_bias=False, rate_limit_batch=100, rate_limit_sleep=5):
    bias = None
    if use_gpt4_bias:
        with open(os.path.join(PLANNER_DIR, "bias_gpt4.json")) as f:
            bias = json.load(f)

    retrieved_path = os.path.join(
        PLANNER_DIR, f"few_examples_from_song/few-song-{sp}_retrieved_keys_clip_Img1_Txt1_panoramic.json"
    )
    few_examples_path = os.path.join(PLANNER_DIR, "few_examples_from_song/few_examples.json")
    retrived = json.load(open(retrieved_path))
    few_examples = json.load(open(few_examples_path))
    splits_path = os.path.join(ALFRED_ROOT, "data/splits/oct21.json")
    if not os.path.isfile(splits_path):
        splits_path = os.path.join(ALFRED_ROOT, "data/splits/oct24.json")
    splits = json.load(open(splits_path))

    output_dir = os.path.join(PLANNER_DIR, f"planner_results/{destination}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"turbo-bias-{sp}_result.json")

    result = defaultdict(dict)
    cnt = 0
    for task in tqdm(splits[sp]):
        cnt += 1
        if cnt > rate_limit_batch:
            time.sleep(rate_limit_sleep)
            cnt = 0

        inst = []
        data = load_task_json(task)
        r_idx = task["repeat_idx"]
        task_id = task["task"]
        instruction = data["turk_annotations"]["anns"][r_idx]["task_desc"]
        for desc in data["turk_annotations"]["anns"][r_idx]["high_descs"]:
            instruction += desc

        goal = ""
        for i in range(len(data["ann"]["goal"]) - 1):
            goal += data["ann"]["goal"][i].strip() + " "
        inst_list = [i for sub_list in data["ann"]["instr"] for i in sub_list]
        high_descs = ""
        for i in range(len(inst_list) - 1):
            high_descs += inst_list[i].strip() + " "

        result[instruction]["root"] = os.path.join("data/json_feat_2.1.0", task["task"])
        result[instruction]["triplet"] = []
        result[instruction]["low_actions"] = []
        result[instruction]["low_classes"] = []
        result[instruction]["high_idxs"] = []

        keys = retrived[task_id][str(r_idx)]
        for k in keys:
            inst.append(few_examples[k])

        text = build_prompt(inst, goal, high_descs)
        try:
            request_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": text}],
                "temperature": 0,
                "max_tokens": 90,
                "stop": ["\n"],
            }
            if bias is not None:
                request_kwargs["logit_bias"] = bias

            response = client.chat.completions.create(**request_kwargs)
            result[instruction]["triplet"].append(response.choices[0].message.content)

            with open(output_path, "w") as f:
                json.dump(result, f, indent=4)

        except Exception as e:
            print(e)
            print(instruction)
            failed.append(instruction)
            with open(os.path.join(PLANNER_DIR, f"{sp}failed.json"), "w") as f:
                json.dump(failed, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dn", help="destination folder under planner_results/", default="qwen3.7-plus", type=str)
    parser.add_argument("--model", help="LLM model name", default=os.environ.get("LLM_MODEL", DEFAULT_MODEL), type=str)
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL", default=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL), type=str)
    parser.add_argument("--api-key", help="API key (overrides LLM_API_KEY env var)", default=None, type=str)
    parser.add_argument(
        "--split",
        help="Run a single split only (default: all four splits)",
        default=None,
        choices=["tests_seen", "tests_unseen", "valid_seen", "valid_unseen"],
    )
    parser.add_argument(
        "--use-gpt4-bias",
        help="Use GPT-4 logit_bias from bias_gpt4.json (OpenAI only)",
        action="store_true",
    )
    parser.add_argument("--rate-limit-batch", type=int, default=100)
    parser.add_argument("--rate-limit-sleep", type=int, default=5)
    args = parser.parse_args()

    client = get_client(api_key=args.api_key, base_url=args.base_url)
    splits = [args.split] if args.split else ["tests_seen", "tests_unseen", "valid_seen", "valid_unseen"]
    for split in splits:
        main(
            split,
            args.dn,
            client,
            args.model,
            use_gpt4_bias=args.use_gpt4_bias,
            rate_limit_batch=args.rate_limit_batch,
            rate_limit_sleep=args.rate_limit_sleep,
        )
