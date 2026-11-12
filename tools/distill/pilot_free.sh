#!/usr/bin/env bash
# Free-form round: an open model writes new documents and questions from the kit's requests, both teachers label
# them, and the unified model is trained on the pilot questions plus these. One seed; public dev problems only.
# Usage: bash tools/distill/pilot_free.sh PILOT_DIR RUNS_DIR [SEALED_FILE]
# The sealed file, when given, is read only by the overlap filter. Finished steps are skipped on a rerun.
set -uo pipefail
PILOT="$1"; RUNS="$2"; SEALED="${3:-}"
OUT="$RUNS/distill"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK=/content/distill; FREE="$WORK/free"
A=Qwen/Qwen3.6-27B
B=google/gemma-4-31B-it
mkdir -p "$FREE" "$OUT/free"
cd "$ROOT"
sha() { python -c "from huggingface_hub import model_info;import sys;print(model_info(sys.argv[1]).sha)" "$1"; }
step() { echo "$(date +%H:%M:%S) $*"; }
pip -q install "transformers==5.22.1" 2>&1 | tail -1
REV_A="$(sha $A)"
[ -f "$FREE/requests.jsonl" ] || cp "$PILOT/open_model_generation_requests.jsonl" "$FREE/requests.jsonl"
if [ ! -f "$FREE/probe.jsonl" ]; then
  step "probe: 16 requests"
  head -16 "$FREE/requests.jsonl" > "$FREE/probe-requests.jsonl"
  python tools/distill/generate_free.py --requests "$FREE/probe-requests.jsonl" --model $A --revision "$REV_A" \
    --out "$FREE/probe.jsonl" > "$FREE/probe.log" 2>&1 || { tail -c 1200 "$FREE/probe.log"; exit 1; }
  rm -rf "$FREE/probe-input"
  python tools/distill/free_to_input.py "$FREE/probe.jsonl" --out "$FREE/probe-input" --heldout heldout/dev.jsonl
  [ "$(wc -l < "$FREE/probe-input/teacher_input.jsonl")" -ge 40 ] || { echo "probe produced too few usable questions"; exit 1; }
fi
if [ ! -f "$FREE/done-generate" ]; then
  step "generate"
  python tools/distill/generate_free.py --requests "$FREE/requests.jsonl" --model $A --revision "$REV_A" \
    --out "$FREE/replies.jsonl" > "$FREE/generate.log" 2>&1 || { tail -c 1200 "$FREE/generate.log"; cp "$FREE/replies.jsonl" "$OUT/free/" 2>/dev/null; exit 1; }
  touch "$FREE/done-generate"; cp "$FREE/replies.jsonl" "$OUT/free/"
fi
if [ ! -d "$FREE/input" ]; then
  step "convert"
  python tools/distill/free_to_input.py "$FREE/replies.jsonl" --out "$FREE/input" --heldout heldout/dev.jsonl $SEALED || exit 1
  cp -r "$FREE/input" "$OUT/free/"
fi
score() { # model output
  [ -f "$2" ] && return 0
  step "label with $1"
  python tools/distill/score_teacher.py --input "$FREE/input/teacher_input.jsonl" --model "$1" --revision "$(sha "$1")" \
    --four-bit --batch-size 16 --out "$2" > "$2.log" 2>&1 || { tail -c 900 "$2.log"; return 1; }
  cp "$2" "$OUT/free/"
}
score $A "$FREE/teacher-a.jsonl" || exit 1
score $B "$FREE/teacher-b.jsonl" || exit 1
if [ ! -d "$FREE/scored" ]; then
  python tools/distill/combine_teachers.py "$FREE/teacher-a.jsonl" "$FREE/teacher-b.jsonl" \
    --input "$FREE/input/teacher_input.jsonl" --documents "$FREE/input/documents.jsonl" --out "$FREE/scored" \
    --score-teacher b || exit 1
  cp -r "$FREE/scored" "$OUT/free/"
fi
cat "$FREE/scored/provenance.json"; echo
if [ ! -d "$WORK/scored-all" ]; then
  mkdir "$WORK/scored-all"
  for f in train tune; do cat "$WORK/scored-b/$f.jsonl" "$FREE/scored/$f.jsonl" > "$WORK/scored-all/$f.jsonl"; done
fi
wc -l "$WORK"/scored-all/*.jsonl
step "train"
pip -q install "transformers==4.57.6" 2>&1 | tail -1
pip -q install --no-deps "laya==0.4.31" 2>&1 | tail -1
python lora_tasks.py --regimes unified --nli-regime mix4-logic-teacher-jsem --teacher "$RUNS/teacher/teacher.jsonl" \
  --distill "$WORK/scored-all" --seeds 42 --out "$OUT/tasks-free" --result "$OUT/lora-tasks-free.json" \
  --summary "$OUT/summary-free.txt" 2>&1 | grep -v "step [0-9]* loss" | tail -40
python heldout_eval.py --checkpoint "$OUT/tasks-free/unified-final" --out "$OUT/heldout-dev-free.json" 2>&1 | grep -v "^  "
step "done"
