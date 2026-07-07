# MMP
This README provides instructions for generating subgoals with GPT-4 as part of our implementation of Multi-Modal Planner in FLARE.


## Embed language instructions
~~Run `embed-instructions.py` to embed language instructions with BERT encoder.~~


~~This will create `few_examples_from_song/train_few_instrucitons_emb.p` which contains language instruction embeddings.~~

We provide pre-generated `few_examples_from_song/train_few_instrucitons_emb.p`

## Embed language instructions
~~Run `embed-state.py` to embed agnet's egocentric view with CLIP encoder.~~

~~This will create `few_examples_from_song/train_few_clip_image_panoramic_emb.p` which contains agent's egocentrice view (panoramic) embeddings.~~

We provide pre-generated `few_examples_from_song/train_few_clip_image_panoramic_emb.p`

## Retrieve top-k in-context examples
Run `retriever.py` to retreive top-k (here. k=9) in-context examples for each tasks in valid and tests splits.

```
python retriever.py
```
This will create `few_examples_from_song/few-song-{sp}_retrieved_keys_clip_Img1_Txt1_panoramic.json` which contains retrieved in-context examples for each tasks.

## Generate plan with LLM (Qwen / GPT-4)

Install the OpenAI Python SDK (used in OpenAI-compatible mode):
```
pip install openai
```

Set API credentials via environment variables (do **not** hardcode keys in source):
```
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://llm-1316nm7shgmikwyq.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
export LLM_MODEL="qwen3.7-plus"
```

Generate plans:
```
cd planner
python generate_plans.py --dn qwen3.7-plus
# or test one split first:
python generate_plans.py --dn qwen3.7-plus --split valid_unseen
```

Postprocess LLM output into ALFRED executable action sequences:
```
python postprocess.py --dn qwen3.7-plus --output-mmp-dir ../MMP_results_qwen
```

This writes `planner_results/qwen3.7-plus/turbo-bias-{split}_result.json`.
With `--output-mmp-dir`, it also writes `MMP_results_qwen/{split}.json`.

Use your generated plans at inference time:
```
export MMP_RESULTS_DIR=MMP_results_qwen
bash eval.sh tests_unseen 0 64 flare
```

For original GPT-4 generation, pass `--use-gpt4-bias` and point `--base-url` / `--model` to OpenAI.
## Hardware 
Tested on:
- **GPU** - RTX A6000
- **CPU** - Intel(R) Core(TM) i7-12700K CPU @ 3.60GHz
- **RAM** - 64GB
- **OS** - Ubuntu 20.04