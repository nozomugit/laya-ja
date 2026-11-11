"""Use necessity probes only when two direct judges disagree."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from cross_family import read_view
from four_task_assist import LABELS
from four_task_prompts import nli_prompt
from four_task_reason import engine_scores
from four_task_report import load, score
from semantic_probes import probe
from semantic_verifier import weights


def resolve(prior, dense, sparse, truth, falsity, factor):
    base = 0.1 * prior + 0.45 * (dense + sparse)
    # Record what the two binary judgments say about the three semantic labels.
    z = np.clip(
        np.stack(
            [
                truth[..., 0] - truth[..., 1],
                falsity[..., 0] - falsity[..., 1],
                np.zeros_like(truth[..., 0]),
            ],
            axis=-1,
        ),
        -4,
        4,
    )
    disagree = dense.argmax(-1) != sparse.argmax(-1)
    return softmax(
        np.log(np.maximum(base, 1e-15)) + factor * z * disagree[..., None], axis=-1
    )


def run(a):
    sources = [read_view(path) for path in [a.dense, a.sparse, a.truth, a.falsity]]
    sizes = {
        "nli-dev": 96,
        "semantic-cal": 144,
        "jsem": 384,
        "jsem-fresh": 96,
        "challenge": 36,
    }
    keys = {(k, i) for k, n in sizes.items() for i in range(n)}
    if any(set(view) != keys for view in sources):
        raise ValueError("Incomplete or unaligned source rows")

    def get(k):
        rows = load(a.inputs / (k + ".jsonl"))
        assert len(rows) == sizes[k]
        cols = [np.array([view[k, i] for i in range(len(rows))]) for view in sources]
        return (
            np.array([r["laya"] for r in rows]),
            softmax(cols[0], axis=1),
            softmax(cols[1], axis=1),
            cols[2],
            cols[3],
        ), np.array([r["gold"] for r in rows])

    dx1, y1 = get("nli-dev")
    dx2, y2 = get("semantic-cal")
    x = [np.concatenate([u, v]) for u, v in zip(dx1, dx2)]
    y = np.r_[y1, y2]
    g = np.array(
        ["jnli"] * 48
        + ["jsick"] * 48
        + [
            "world-" + str(i)
            for i in json.loads((a.inputs / "groups.json").read_text())["semantic-cal"]
        ]
    )
    disagree = x[1].argmax(1) != x[2].argmax(1)

    def fit(keep):
        mask = disagree & keep
        ys = y[mask]
        w = weights(ys, g[mask])

        def loss(t):
            pr = resolve(*[p[mask] for p in x], t)
            return (
                -w @ np.log(np.maximum(pr[np.arange(len(ys)), ys], 1e-15)) + 0.1 * t * t
            )

        opt = minimize_scalar(loss, bounds=(0, 1), method="bounded")
        return float(opt.x) if opt.fun < loss(0) else 0.0

    factor = fit(np.ones(len(y), dtype=bool))
    cfg = {
        "factor": factor,
        "development_rows": len(y),
        "disagreements": int(disagree.sum()),
    }
    a.weights.write_text(json.dumps(cfg, indent=2) + "\n")
    output = {}
    for k in sizes:
        xx, yy = get(k)
        probs = resolve(*xx, factor)
        output[k] = score(yy, probs)
        output[k]["disagreements"] = int((xx[1].argmax(1) != xx[2].argmax(1)).sum())
    a.report.write_text(json.dumps(output, indent=2) + "\n")


def predict(a):
    cmd = [sys.executable, str(Path(__file__).with_name("predict.py"))]
    if a.model_path:
        cmd.extend(["--model-path", a.model_path])
    raw = json.loads(
        subprocess.check_output(
            cmd + ["context-nli", a.premise, a.hypothesis], text=True
        )
    )
    prior = np.array([raw["probabilities"][k] for k in LABELS])
    prompt = nli_prompt(a.premise, a.hypothesis)
    dense, sparse = [
        engine_scores(a.engine, m, [prompt], 0, a.threads)[0]
        for m in (a.dense, a.sparse)
    ]
    factor = json.loads(a.weights.read_text())["factor"]
    needed = bool(factor and dense.argmax() != sparse.argmax())
    z = np.zeros((2, 3))
    if needed:
        prompts = [
            probe(a.premise, a.hypothesis, k) for k in ("must_true", "must_false")
        ]
        z = np.log(
            np.maximum(engine_scores(a.engine, a.sparse, prompts, 0, a.threads), 1e-100)
        )
    probs = resolve(prior, dense, sparse, *z, factor)
    return {
        "choice": LABELS[probs.argmax()],
        "used_probes": needed,
        "probabilities": dict(zip(LABELS, probs.tolist())),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--weights", type=Path, default=Path("weights/semantic-disagreement.json")
    )
    sub = p.add_subparsers(dest="task", required=True)
    f = sub.add_parser("fit")
    for key, default in {
        "dense": "dense",
        "sparse": "semantic-predictions/direct-out",
        "truth": "semantic-predictions/must_true-out",
        "falsity": "semantic-predictions/must_false-out",
    }.items():
        f.add_argument(
            "--" + key, type=Path, default=Path(".cache") / (default + ".jsonl")
        )
    f.add_argument("--inputs", type=Path, default=Path(".cache/four"))
    f.add_argument(
        "--report", type=Path, default=Path("results/semantic-disagreement.json")
    )
    q = sub.add_parser("predict")
    q.add_argument("premise")
    q.add_argument("hypothesis")
    q.add_argument("--model-path")
    q.add_argument("--engine", type=Path, required=True)
    q.add_argument("--threads", type=int, default=4)
    q.add_argument(
        "--dense",
        type=Path,
        default=Path(".cache/four-models/dense/Qwen3.6-27B-Q3_K_S.gguf"),
    )
    q.add_argument(
        "--sparse",
        type=Path,
        default=Path(".cache/four-models/reason/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"),
    )
    args = p.parse_args()
    if args.task == "fit":
        run(args)
    else:
        print(json.dumps(predict(args), ensure_ascii=False, indent=2))
