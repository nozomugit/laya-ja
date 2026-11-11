"""Fit a native Japanese NLI readout to five development groups using encoder and edit scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from four_task_assist import LABELS
from four_task_report import load, score
from native_edit_nli import matrix
from native_semantic_pool import NativePool
from semantic_verifier import weights


def opinions(pool, edit, rows):
    shared, avg = pool.probabilities(rows)
    lex = softmax(matrix(rows) @ edit["coef"].T + edit["bias"], axis=1)
    return np.log(np.maximum(np.stack([shared, avg, lex], axis=1), 1e-15))


def logit(x, c):
    return np.einsum("nvc,v->nc", x, c[:3]) + [c[3], c[4], -c[3] - c[4]]


def fit(x, y, groups, reg):
    anchor = np.array([0.25, 0.25, 0.5, 0.0, 0.0])
    w = weights(y, groups)

    def objective(c):
        z = logit(x, c)
        return (
            w @ (logsumexp(z, axis=1) - z[np.arange(len(y)), y])
            + reg * ((c - anchor) ** 2).sum()
        )

    opt = minimize(
        objective, anchor, method="L-BFGS-B", bounds=[(0, 3)] * 3 + [(-3, 3)] * 2
    )
    if not opt.success:
        raise ValueError(opt.message)
    return opt.x


def experiment(a, pool, edit):
    dev = load(a.inputs / "nli-dev.jsonl") + load(a.inputs / "semantic-cal.jsonl")
    y = np.array([r["gold"] for r in dev])
    g = np.array(
        ["jnli"] * 48
        + ["jsick"] * 48
        + [
            "world-" + str(i)
            for i in json.loads((a.inputs / "groups.json").read_text())["semantic-cal"]
        ]
    )
    x = opinions(pool, edit, dev)
    dev_loss = {}
    for reg in (0.01, 0.1, 1.0):
        lp = np.zeros(len(y))
        for group in np.unique(g):
            mask = g == group
            z = logit(x[mask], fit(x[~mask], y[~mask], g[~mask], reg))
            lp[mask] = logsumexp(z, axis=1) - z[np.arange(mask.sum()), y[mask]]
        dev_loss[reg] = float(weights(y, g) @ lp)
    chosen = min(dev_loss, key=dev_loss.get)
    cfg = fit(x, y, g, chosen)
    a.weights.write_text(json.dumps(cfg.tolist()) + "\n")
    report = {"regularization": chosen, "held_group_nll": dev_loss, "evaluation": {}}
    for label in ("jsem", "jsem-fresh", "challenge"):
        rows = load(a.inputs / (label + ".jsonl"))
        p = softmax(logit(opinions(pool, edit, rows), cfg), axis=1)
        report["evaluation"][label] = score(np.array([r["gold"] for r in rows]), p)
    a.result.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path")
    p.add_argument("--weights", type=Path, default=Path("weights/native-joint.json"))
    p.add_argument("--edit", type=Path, default=Path("weights/native-edit.npz"))
    sub = p.add_subparsers(dest="task", required=True)
    t = sub.add_parser("fit")
    t.add_argument("--inputs", type=Path, default=Path(".cache/four"))
    t.add_argument("--result", type=Path, default=Path("results/native-joint.json"))
    q = sub.add_parser("predict")
    q.add_argument("premise")
    q.add_argument("hypothesis")
    a = p.parse_args()
    torch.set_num_threads(2)
    pool = NativePool(a.model_path)
    with np.load(a.edit) as edit:
        if a.task == "fit":
            experiment(a, pool, edit)
        else:
            row = {"text": a.premise, "options": [a.hypothesis]}
            c = json.loads(a.weights.read_text())
            prob = softmax(logit(opinions(pool, edit, [row]), c), axis=1)[0]
            print(
                json.dumps(
                    {
                        "choice": LABELS[prob.argmax()],
                        "probabilities": dict(zip(LABELS, prob.tolist())),
                    },
                    ensure_ascii=False,
                )
            )
