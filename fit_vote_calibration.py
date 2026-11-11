"""Calibrate voters and learn a convex mixture on five independent development groups."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from cross_family import read_view
from four_task_report import load
from semantic_verifier import weights
from semantic_vote import VOTES, calibrate


def fit_one(probs, gold, groups):
    w = weights(gold, groups)

    def loss(theta):
        p = calibrate(probs, theta)
        nll = -w @ np.log(np.maximum(p[np.arange(len(p)), gold], 1e-15))
        return nll + 0.01 * np.dot(theta, theta)

    opt = minimize(
        loss, np.zeros(3), method="L-BFGS-B", bounds=[(-2.3, 1.5), (-4, 4), (-4, 4)]
    )
    if not opt.success:
        raise ValueError(opt.message)
    return opt.x


def fit(args):
    logits = {name: read_view(getattr(args, name)) for name, _ in VOTES[1:]}
    src, gold, groups = [], [], []
    group_ids = json.loads((args.inputs / "groups.json").read_text())["semantic-cal"]
    for split, grouping in [
        ("nli-dev", ["jnli"] * 48 + ["jsick"] * 48),
        ("semantic-cal", ["world-" + str(i) for i in group_ids]),
    ]:
        rows = load(args.inputs / (split + ".jsonl"))
        if len(rows) != len(grouping):
            raise ValueError("Incorrect development group lengths")
        for i, r in enumerate(rows):
            src.append(
                [r["laya"]] + [softmax(logits[name][split, i]) for name, _ in VOTES[1:]]
            )
            gold.append(r["gold"])
        groups.extend(grouping)
    src, y, g = np.array(src), np.array(gold), np.array(groups)
    if src.shape != (240, 5, 3) or len(np.unique(g)) != 5:
        raise ValueError("Unexpected development split")
    hold = np.empty((len(y), 5))
    models = {}
    for j, (name, _) in enumerate(VOTES):
        p = src[:, j]
        for group in np.unique(g):
            out = g == group
            theta = fit_one(p[~out], y[~out], g[~out])
            hold[out, j] = calibrate(p[out], theta)[np.arange(out.sum()), y[out]]
        models[name] = fit_one(p, y, g).tolist()
    w = weights(y, g)
    prior = np.array([weight / 100 for _, weight in VOTES])

    def loss(v):
        return -w @ np.log(np.maximum(hold @ v, 1e-15)) + 0.02 * np.sum(
            (v - prior) ** 2
        )

    opt = minimize(
        loss,
        prior,
        method="SLSQP",
        bounds=[(0.05, 1)] + [(0, 1)] * 4,
        constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1}],
        options={"ftol": 1e-12},
    )
    if not opt.success:
        raise ValueError(opt.message)
    alpha = np.maximum(opt.x, 0)
    alpha[alpha < 1e-9] = 0
    alpha /= alpha.sum()
    config = {
        "weights": {name: float(a) for (name, _), a in zip(VOTES, alpha)},
        "calibration": models,
        "fit_groups": {k: int((g == k).sum()) for k in np.unique(g)},
    }
    args.output.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=Path(".cache/four"))
    for key, default in (
        ("direct35", "semantic-predictions/direct-out"),
        ("direct27", "dense"),
        ("gemma", "gemma"),
        ("counter", "countermodel"),
    ):
        p.add_argument(
            "--" + key, type=Path, default=Path(".cache") / (default + ".jsonl")
        )
    p.add_argument("--output", type=Path, default=Path("weights/semantic-vote.json"))
    fit(p.parse_args())
