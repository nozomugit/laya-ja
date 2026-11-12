#!/usr/bin/env bash
# Combine the two teachers' labels, train the unified model with them (one seed), and score the public dev problems.
# Usage: bash tools/distill/pilot_train.sh RUNS_DIR   (the Drive folder that holds teacher/ and tasks/)
set -euo pipefail
RUNS="$1"; OUT="$RUNS/distill"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK=/content/distill
cd "$ROOT"
rm -rf /root/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B /root/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it /root/.cache/huggingface/xet
df -h /content | tail -1
if [ ! -d "$WORK/scored-b" ]; then
  python tools/distill/combine_teachers.py "$WORK/teacher-a.jsonl" "$WORK/teacher-b.jsonl" \
    --input "$WORK/teacher_input.jsonl" --documents "$WORK/documents.jsonl" --out "$WORK/scored-b" --score-teacher b
  rm -rf "$OUT/scored-b" && cp -r "$WORK/scored-b" "$OUT/"
fi
cat "$WORK/scored-b/provenance.json"; echo
pip -q install "transformers==4.57.6" 2>&1 | tail -1
pip -q install --no-deps "laya==0.4.31" 2>&1 | tail -1
python lora_tasks.py --regimes unified --nli-regime mix4-logic-teacher-jsem --teacher "$RUNS/teacher/teacher.jsonl" \
  --distill "$WORK/scored-b" --seeds 42 --out "$OUT/tasks" --result "$OUT/lora-tasks-distill.json" \
  --summary "$OUT/summary-distill.txt" 2>&1 | grep -v "step [0-9]* loss"
python heldout_eval.py --checkpoint "$OUT/tasks/unified-final" "$RUNS/tasks/unified-final" --out "$OUT/heldout-dev-distill.json"
date +%H:%M:%S
