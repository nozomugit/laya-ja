#!/usr/bin/env bash
# Score the prepared prompts with the two direct judges on a CUDA GPU.
# Same prompts, quantized weights and llama.cpp commit as the CPU workflows; only the layers are offloaded.
# Usage: bash tools/teacher/label_gpu.sh OUT_DIR [BUILD_DIR]
# Finished chunks are skipped, so the script can be rerun after an interruption.
set -euo pipefail

OUT="$1"
WORK="${2:-/content/teacher-build}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
COMMIT=ccd360f060a0f17b87ee0d4c5d91ee84cfdff484
mkdir -p "$WORK" "$OUT"

if [ ! -x "$WORK/score" ]; then
  if [ ! -d "$WORK/runtime/.git" ]; then
    git init -q "$WORK/runtime"
    git -C "$WORK/runtime" fetch -q --depth=1 https://github.com/ggml-org/llama.cpp.git "$COMMIT"
    git -C "$WORK/runtime" checkout -q --detach FETCH_HEAD
  fi
  cmake -S "$WORK/runtime" -B "$WORK/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF > "$WORK/cmake.log" 2>&1 \
    || { tail -30 "$WORK/cmake.log"; exit 1; }
  cmake --build "$WORK/build" --target llama -j "$(nproc)" > "$WORK/build.log" 2>&1 \
    || { tail -40 "$WORK/build.log"; exit 1; }
  sed 's/mp.n_gpu_layers = 0;/mp.n_gpu_layers = 999;/' "$ROOT/tools/four-task/label.cpp" > "$WORK/label_gpu.cpp"
  grep -q 'n_gpu_layers = 999' "$WORK/label_gpu.cpp"
  c++ -O2 -std=c++17 -I"$WORK/runtime/include" -I"$WORK/runtime/ggml/include" -I"$WORK/runtime/vendor" \
    "$WORK/label_gpu.cpp" -L"$WORK/build/bin" -Wl,-rpath,"$WORK/build/bin" -lllama -lggml -lggml-base -o "$WORK/score"
fi

cd "$ROOT"
python four_task_models.py dense --root "$WORK/models"
python four_task_models.py reason --root "$WORK/models"

run() { # judge-name model-file
  for prompts in "$OUT"/prompts-*.jsonl; do
    part="$(basename "$prompts" .jsonl)"; part="${part#prompts-}"
    target="$OUT/$1-$part.jsonl"
    if [ -f "$target" ] && [ "$(wc -l < "$target")" -eq "$(wc -l < "$prompts")" ]; then continue; fi
    rm -f "$target.tmp"
    "$WORK/score" "$2" "$prompts" "$target.tmp" 4 4 2> "$WORK/$1.log" || { tail -25 "$WORK/$1.log"; exit 1; }
    test "$(wc -l < "$target.tmp")" -eq "$(wc -l < "$prompts")"
    mv "$target.tmp" "$target"
    echo "$1 $part $(date +%H:%M:%S)"
  done
}
run dense "$WORK/models/dense/Qwen3.6-27B-Q3_K_S.gguf"
run sparse "$WORK/models/reason/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
python teacher_data.py collect --out "$OUT"
