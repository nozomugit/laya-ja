"""Three-way inference with exact early exit for a fixed mixture of model votes."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax

from countermodel_prompts import prompt as counter_prompt
from four_task_assist import LABELS
from four_task_prompts import nli_prompt
from four_task_reason import engine_scores
from gemma_prompts import prompt as gemma_prompt

# Percent weights. First use the two scorers that do not generate reasoning tokens.
VOTES = (
    ("laya", 10),
    ("direct35", 25),
    ("direct27", 25),
    ("gemma", 30),
    ("counter", 10),
)


def calibrate(probs, theta):
    scale, entail, contradict = theta
    delta = np.array([entail, contradict, -entail - contradict])
    return softmax(np.exp(scale) * np.log(np.maximum(probs, 1e-15)) + delta, axis=-1)


def decide(get_vote, *, early=True, config=None):
    """Return a label and bounds for the final mixture, including any skipped votes."""
    scores = np.zeros(3)
    remaining = 100
    used = []
    votes = VOTES
    if config is not None:
        votes = [(name, 100 * config["weights"][name]) for name, _ in VOTES]
        if any(not math.isfinite(w) or w < 0 for _, w in votes) or not math.isclose(
            sum(w for _, w in votes), 100
        ):
            raise ValueError("Invalid mixture weights")
    for name, weight in votes:
        if weight == 0:
            continue
        p = np.asarray(get_vote(name), dtype=float)
        if p.shape != (3,) or not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError(f"Invalid three-way probabilities from {name}")
        if not math.isclose(float(p.sum()), 1.0, abs_tol=1e-5):
            raise ValueError(f"Unnormalized probabilities from {name}")
        if config is not None:
            p = calibrate(p, config["calibration"][name])
        scores += weight * (p / p.sum())
        remaining = max(0.0, remaining - weight)
        used.append(name)
        leader = int(scores.argmax())
        runner_up = float(np.max(np.delete(scores, leader)))
        if early and scores[leader] - runner_up > remaining + 1e-10:
            break
    return leader, used, scores / 100, (scores + remaining) / 100


def predict(args):
    paths = {
        "direct35": args.reason_model,
        "counter": args.reason_model,
        "direct27": args.dense_model,
        "gemma": args.gemma_model,
    }

    def vote(name):
        if name == "laya":
            cmd = [sys.executable, str(Path(__file__).with_name("predict.py"))]
            if args.model_path:
                cmd += ["--model-path", args.model_path]
            raw = json.loads(
                subprocess.check_output(
                    cmd + ["context-nli", args.premise, args.hypothesis], text=True
                )
            )
            return [raw["probabilities"][label] for label in LABELS]
        prompt, budget, template = nli_prompt, 0, None
        if name == "gemma":
            prompt, budget, template = gemma_prompt, 160, "gemma"
        elif name == "counter":
            prompt, budget = counter_prompt, 256
        return engine_scores(
            args.engine,
            paths[name],
            [prompt(args.premise, args.hypothesis)],
            budget,
            args.threads,
            template,
        )[0]

    config = json.loads(args.calibration.read_text()) if args.calibration else None
    label, used, low, high = decide(vote, early=not args.all_votes, config=config)
    return {
        "choice": LABELS[label],
        "used": used,
        "mixture_bounds": {
            k: [float(a), float(b)] for k, a, b in zip(LABELS, low, high)
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("premise")
    p.add_argument("hypothesis")
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--model-path")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--calibration", type=Path, help="Development-fitted JSON configuration"
    )
    p.add_argument("--all-votes", action="store_true")
    root = Path(".cache/four-models")
    p.add_argument(
        "--reason-model",
        type=Path,
        default=root / "reason/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
    )
    p.add_argument(
        "--dense-model", type=Path, default=root / "dense/Qwen3.6-27B-Q3_K_S.gguf"
    )
    p.add_argument(
        "--gemma-model",
        type=Path,
        default=root / "gemma/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf",
    )
    print(json.dumps(predict(p.parse_args()), ensure_ascii=False, indent=2))
