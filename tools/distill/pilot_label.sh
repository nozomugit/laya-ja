#!/usr/bin/env bash
# Measure the second teacher on the public dev problems, then label the pilot questions with both teachers.
# Usage: bash tools/distill/pilot_label.sh PILOT_DIR OUT_DIR
# Finished files are kept and partial ones resumed, so the script can be rerun after an interruption.
set -uo pipefail
PILOT="$1"; OUT="$2"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK=/content/distill
A=Qwen/Qwen3.6-27B
B=google/gemma-4-31B-it
mkdir -p "$WORK" "$OUT"
cd "$ROOT"
sha() { python -c "from huggingface_hub import model_info;import sys;print(model_info(sys.argv[1]).sha)" "$1"; }
score() { # model input output
  [ -f "$3" ] && return 0
  echo "$(date +%H:%M:%S) $1 -> $3"
  python tools/distill/score_teacher.py --input "$2" --model "$1" --revision "$(sha "$1")" --four-bit \
    --batch-size 16 --out "$3" > "$3.log" 2>&1 || { tail -c 900 "$3.log"; return 1; }
}
[ -f "$WORK/teacher_input.jsonl" ] || cp "$PILOT"/{teacher_input,documents}.jsonl "$WORK/"
(cd "$WORK" && sha256sum teacher_input.jsonl documents.jsonl)
[ -f /content/dev_in.jsonl ] || python tools/distill/dev_check.py prepare --out /content/dev_in.jsonl
score "$B" /content/dev_in.jsonl "/content/dev-$(basename $B).jsonl"
python tools/distill/dev_check.py report /content/dev-Qwen3.6-27B.jsonl "/content/dev-$(basename $B).jsonl" \
  --json "$OUT/dev-teachers.json"
cp /content/dev-*.jsonl "$OUT/" 2>/dev/null
score "$B" "$WORK/teacher_input.jsonl" "$WORK/teacher-b.jsonl" && cp "$WORK/teacher-b.jsonl" "$OUT/"
score "$A" "$WORK/teacher_input.jsonl" "$WORK/teacher-a.jsonl" && cp "$WORK/teacher-a.jsonl" "$OUT/"
if [ -f "$WORK/teacher-a.jsonl" ] && [ -f "$WORK/teacher-b.jsonl" ] && [ ! -d "$WORK/scored" ]; then
  python tools/distill/combine_teachers.py "$WORK/teacher-a.jsonl" "$WORK/teacher-b.jsonl" \
    --input "$WORK/teacher_input.jsonl" --documents "$WORK/documents.jsonl" --out "$WORK/scored" \
    && cp -r "$WORK/scored" "$OUT/" && cat "$WORK/scored/provenance.json"
fi
date +%H:%M:%S
