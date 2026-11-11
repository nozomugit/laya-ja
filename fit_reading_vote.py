"""Train a conservatively regularized semantic vote on five independent development groups."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from cross_family import read_view
from four_task_report import load, score
from semantic_verifier import weights

ANCHOR = np.array([0.1, 0.225, 0.225, 0.225, 0.225, 0.0, 0.0])


def predict(native, base, dense, first, second, coef):
    views = np.stack(
        [native, base, dense, first, np.asarray(second)[..., [2, 0, 1]]], axis=-2
    )
    return model(views, np.asarray(coef))


def model(views, coef):
    mixture = np.einsum("...vc,v->...c", views, coef[:5])
    return softmax(np.log(mixture.clip(1e-9)) + [coef[5], coef[6], 0], axis=-1)


def fit(x, y, g, penalty):
    if penalty is None:
        return ANCHOR
    w = weights(y, g)

    def loss(coef):
        p = model(x, coef)
        return -w @ np.log(p[np.arange(len(y)), y].clip(1e-12)) + penalty * np.sum(
            (coef - ANCHOR) ** 2
        )

    opt = minimize(
        loss,
        ANCHOR,
        bounds=[(0.05, 0.35)] + [(0.01, 0.90)] * 4 + [(-1.5, 1.5)] * 2,
        constraints=[{"type": "eq", "fun": lambda c: c[:5].sum() - 1}],
        method="SLSQP",
        options={"maxiter": 300},
    )
    if not opt.success:
        raise RuntimeError(opt.message)
    return opt.x


def run(a):
    by = [read_view(getattr(a, k)) for k in ("base", "dense", "first", "second")]
    if any(set(v) != set(by[0]) for v in by[1:]):
        raise ValueError("Different source rows")
    parts = {}
    groups = json.loads((a.inputs / "groups.json").read_text())
    for k in ("nli-dev", "semantic-cal", "challenge", "jsem", "jsem-fresh"):
        rs = load(a.inputs / (k + ".jsonl"))
        v = [softmax([d[k, i] for i in range(len(rs))], axis=1) for d in by]
        v[-1] = v[-1][:, [2, 0, 1]]
        parts[k] = (
            np.stack([np.array([r["laya"] for r in rs]), *v], axis=1),
            np.array([r["gold"] for r in rs]),
        )
    x = np.concatenate([parts["nli-dev"][0], parts["semantic-cal"][0]])
    y = np.concatenate([parts["nli-dev"][1], parts["semantic-cal"][1]])
    g = np.array(
        ["jnli"] * 48
        + ["jsick"] * 48
        + ["world" + str(v) for v in groups["semantic-cal"]]
    )

    def quality(penalty):
        pred = np.empty((len(y), 3))
        scores = []
        for name in np.unique(g):
            tr = g != name
            pred[~tr] = model(x[~tr], fit(x[tr], y[tr], g[tr], penalty))
            scores.append(weights(y[~tr], g[~tr]) @ (pred[~tr].argmax(1) == y[~tr]))
        return (
            float(min(scores)),
            float(np.mean(scores)),
            float(weights(y, g) @ np.log(pred[np.arange(len(y)), y].clip(1e-12))),
        )

    trials = [(quality(reg), reg) for reg in (None, 0.05, 0.2, 1.0)]
    best, reg = max(trials, key=lambda q: q[0])
    coef = fit(x, y, g, reg)
    a.config.write_text(
        json.dumps(
            {
                "coefficients": coef.tolist(),
                "penalty": reg,
                "worst_domain_cv": best[0],
                "development_items": len(y),
            },
            indent=2,
        )
        + "\n"
    )
    result, out = {}, []
    for k, (xx, yy) in parts.items():
        p = model(xx, coef)
        result[k] = score(yy, p)
        for i, z in enumerate(p):
            out.append(
                {
                    "split": k,
                    "row": i,
                    "gold": int(yy[i]),
                    "prediction": int(z.argmax()),
                    "probabilities": z.round(7).tolist(),
                }
            )
    a.result.write_text(json.dumps(result, indent=2) + "\n")
    a.rows.write_bytes(
        gzip.compress("".join(json.dumps(r) + "\n" for r in out).encode(), mtime=0)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in {
        "inputs": ".cache/four",
        "base": ".cache/semantic-predictions/direct-out.jsonl",
        "dense": ".cache/dense.jsonl",
        "first": ".cache/reading0.jsonl",
        "second": ".cache/reading1.jsonl",
        "result": "results/four-trained-vote.json",
        "rows": "results/four-trained-vote.jsonl.gz",
        "config": "weights/trained-vote.json",
    }.items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    run(p.parse_args())
