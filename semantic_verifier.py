"""Combine independent yes/no tests of necessary truth and necessary falsity."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from four_task_report import load, score

# Truth test, falsity test, three-way specialist, native Laya and two label intercepts.
ANCHOR = np.array([0.55, 0.55, 0.4, 0.12, 0.0, 0.0])
BOUNDS = [(0.02, 2), (0.02, 2), (0, 2), (0.04, 0.8), (-3, 3), (-3, 3)]


def features(truth, falsity, direct, native):
    lp = np.log(np.maximum(native, 1e-10))
    return np.stack(
        (
            truth[..., 0] - truth[..., 1],
            falsity[..., 0] - falsity[..., 1],
            direct[..., 0] - direct[..., 2],
            direct[..., 1] - direct[..., 2],
            lp[..., 0] - lp[..., 2],
            lp[..., 1] - lp[..., 2],
        ),
        axis=-1,
    ).clip(-20, 20)


def predict(x, coef):
    t, f, d, n, e, c = coef
    z = np.stack(
        [
            t * x[..., 0] + d * x[..., 2] + n * x[..., 4] + e,
            f * x[..., 1] + d * x[..., 3] + n * x[..., 5] + c,
            np.zeros(x.shape[:-1]),
        ],
        axis=-1,
    )
    return softmax(z, axis=-1)


def weights(y, group):
    w = np.zeros(len(y))
    for g in np.unique(group):
        for label in np.unique(y[group == g]):
            idx = (group == g) & (y == label)
            w[idx] = 1 / idx.sum()
    return w / w.sum()


def train(x, y, g, penalty):
    if penalty is None:
        return ANCHOR.copy()
    w = weights(y, g)

    def loss(c):
        p = predict(x, c)
        return -w @ np.log(
            np.maximum(p[np.arange(len(y)), y], 1e-15)
        ) + penalty * np.sum((c - ANCHOR) ** 2)

    opt = minimize(loss, ANCHOR, method="L-BFGS-B", bounds=BOUNDS)
    if not opt.success:
        raise ValueError(opt.message)
    return opt.x


def select(x, y, groups):
    def crossval(penalty):
        p = np.empty((len(y), 3))
        for g in np.unique(groups):
            keep = groups != g
            p[~keep] = predict(x[~keep], train(x[keep], y[keep], groups[keep], penalty))
        w = weights(y, groups)
        return (
            float(w @ (p.argmax(1) == y)),
            float(w @ np.log(p[np.arange(len(y)), y])),
        )

    trials = [(crossval(reg), reg) for reg in (None, 0.02, 0.1, 0.4)]
    best, reg = max(trials, key=lambda r: r[0])
    return train(x, y, groups, reg), reg, best


def run(a):
    args = json.loads((a.inputs / "groups.json").read_text())
    data = {}
    for key in ("must_true", "must_false", "direct"):
        rows = load(a.pred / (key + "-out.jsonl"))
        data[key] = {tuple(r["id"]): r["logits"] for r in rows}
        if len(data[key]) != len(rows):
            raise ValueError("Duplicate row")

    def get(key):
        rows = load(a.inputs / (key + ".jsonl"))
        ids = [(key, i) for i in range(len(rows))]
        z = [
            np.array([data[m][k] for k in ids])
            for m in ("must_true", "must_false", "direct")
        ]
        x = features(*z, np.array([r["laya"] for r in rows]))
        return x, np.array([r["gold"] for r in rows]), z

    nx, ny, _ = get("nli-dev")
    sx, sy, _ = get("semantic-cal")
    g = np.array(
        ["jnli"] * 48
        + ["jsick"] * 48
        + ["world-" + str(i) for i in args["semantic-cal"]]
    )
    coef, reg, quality = select(np.concatenate([nx, sx]), np.concatenate([ny, sy]), g)
    cfg = {
        "coefficients": coef.tolist(),
        "penalty": reg,
        "cv_balanced_accuracy": quality[0],
        "development": {"jnli": 48, "jsick": 48, "finite_worlds": len(sy)},
    }
    a.config.write_text(json.dumps(cfg, indent=2) + "\n")
    result = {}
    records = []
    for key in ("semantic-cal", "nli-dev", "challenge", "jsem", "jsem-fresh"):
        x, y, z = get(key)
        prob = predict(x, coef)
        result[key] = {
            "combined": score(y, prob),
            "anchor": score(y, predict(x, ANCHOR)),
            "direct_only": score(y, softmax(z[2], axis=1)),
        }
        if key in args:
            gg = np.array(args[key])
            result[key]["groups"] = {
                str(k): score(y[gg == k], prob[gg == k]) for k in np.unique(gg)
            }
        for i, p in enumerate(prob):
            records.append(
                {
                    "set": key,
                    "row": i,
                    "gold": int(y[i]),
                    "prediction": int(p.argmax()),
                    "probabilities": p.round(7).tolist(),
                }
            )
    a.result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if a.outputs:
        a.outputs.write_text("".join(json.dumps(v) + "\n" for v in records))
    for k, r in result.items():
        print(k, r["combined"]["correct"], r["combined"]["n"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in {
        "inputs": ".cache/four",
        "pred": ".cache/semantic-predictions",
        "config": "weights/semantic-verifier.json",
        "result": "results/four-semantic.json",
    }.items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    p.add_argument("--outputs", type=Path)
    run(p.parse_args())
