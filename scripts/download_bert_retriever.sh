#!/bin/bash
# Download bert-base-nli-mean-tokens into models/BERT_retriever/ for offline eval.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$REPO_ROOT/models/BERT_retriever/bert-base-nli-mean-tokens"

echo "Target: $TARGET"

if [ -d "$TARGET" ] && [ -f "$TARGET/config.json" ]; then
  echo "Local BERT model already exists."
  exit 0
fi

mkdir -p "$TARGET"

# Prefer planner env (Python 3.10 + newer huggingface_hub)
if command -v conda >/dev/null 2>&1 && conda env list | grep -q planner; then
  eval "$(conda shell.bash hook)"
  conda activate planner
fi

unset HF_ENDPOINT
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-https://hf-mirror.com}"

export BERT_TARGET="$TARGET"
python - <<'PY'
from sentence_transformers import SentenceTransformer
import os

target = os.environ["BERT_TARGET"]
model = SentenceTransformer("bert-base-nli-mean-tokens")
model.save(target)
print("Saved to", target)
PY

echo "Done. Sync $TARGET to server if needed."
