"""Ask a two-way question in both option orders only where direct Qwen judges disagree."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

from four_task_assist import LABELS
from four_task_prompts import nli_prompt
from four_task_reason import engine_scores

DESCRIPTIONS = (
    "前提が真であるどの状況でも、結論は真である",
    "前提が真であるどの状況でも、結論は偽である",
    "前提が真でも、結論が真の場合と偽の場合の両方がありうる",
)


def prompts(premise, hypothesis, labels):
    instruction = (
        "日本語の文章の意味を比較します。主語・修飾の範囲・否定・出来事が実際に成立したかを読み分けてください。"
        "前提にない事柄は勝手に補わず、日本語の通常の語義と文脈を用いてください。"
        "次に示す二つの解釈から妥当な方を一つ選び、AまたはBだけで答えてください。"
    )
    result = []
    for a, b in (labels, labels[::-1]):
        result.append(
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n前提：{premise}\n結論：{hypothesis}\n"
            f"A {DESCRIPTIONS[a]}。\nB {DESCRIPTIONS[b]}。<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    return result


def candidates(dense, sparse):
    return sorted({int(dense.argmax()), int(sparse.argmax())})


def combine(prior, dense, sparse, ordered_logits=None):
    base = 0.1 * np.array(prior) + 0.45 * (dense + sparse)
    labels = candidates(dense, sparse)
    if len(labels) == 1:
        return base
    if ordered_logits is None or np.shape(ordered_logits) != (2, 3):
        raise ValueError("Need two judgments in opposite answer orders")
    a, b = labels
    z = np.asarray(ordered_logits)
    evidence = 0.5 * ((z[0, 0] - z[0, 1]) - (z[1, 0] - z[1, 1]))
    odds = np.log(max(base[a], 1e-15) / max(base[b], 1e-15))
    share = expit(odds + 0.5 * np.clip(evidence, -4.0, 4.0))
    mass = base[a] + base[b]
    base[a], base[b] = mass * share, mass * (1 - share)
    return base


def predict(a):
    cmd = [sys.executable, str(Path(__file__).with_name("predict.py"))]
    if a.model_path:
        cmd += ["--model-path", a.model_path]
    raw = json.loads(
        subprocess.check_output(
            cmd + ["context-nli", a.premise, a.hypothesis], text=True
        )
    )
    prior = [raw["probabilities"][k] for k in LABELS]
    q = nli_prompt(a.premise, a.hypothesis)
    dense, sparse = [
        engine_scores(a.engine, model, [q], 0, a.threads)[0]
        for model in (a.dense, a.sparse)
    ]
    labels = candidates(dense, sparse)
    logits = None
    if len(labels) > 1:
        z = engine_scores(
            a.engine, a.sparse, prompts(a.premise, a.hypothesis, labels), 0, a.threads
        )
        logits = np.log(np.maximum(z, 1e-100))
    p = combine(prior, dense, sparse, logits)
    return {
        "choice": LABELS[p.argmax()],
        "used_pairwise": logits is not None,
        "probabilities": dict(zip(LABELS, p.tolist())),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("premise")
    p.add_argument("hypothesis")
    p.add_argument("--model-path")
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--dense",
        type=Path,
        default=Path(".cache/four-models/dense/Qwen3.6-27B-Q3_K_S.gguf"),
    )
    p.add_argument(
        "--sparse",
        type=Path,
        default=Path(".cache/four-models/reason/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"),
    )
    print(json.dumps(predict(p.parse_args()), ensure_ascii=False, indent=2))
