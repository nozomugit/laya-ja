"""Calibrate direct native and judge log probabilities with held-domain validation."""

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

ANCHOR = np.array([0.10, 0.45, 0.45, 0.0, 0.0])
BOUNDS = [(0.04, 0.6), (0.05, 2.0), (0.05, 2.0), (-2, 2), (-2, 2)]


def evidence(native, dense, sparse):
    # Class contrasts remove the arbitrary additive logit normalizer of each model.
    z = np.log(np.stack([native, dense, sparse], axis=-2).clip(1e-9))
    return z - z.mean(axis=-1, keepdims=True)


def predict(x, c):
    logits = np.einsum("...mc,m->...c", x, c[:3])
    return softmax(logits + [c[3], c[4], 0.0], axis=-1)


def fit(x, y, g, lam):
    w = weights(y, g)

    def loss(c):
        p = predict(x, c)
        return -w @ np.log(p[np.arange(len(y)), y].clip(1e-15)) + lam * np.sum(
            (c - ANCHOR) ** 2
        )

    out = minimize(loss, ANCHOR, bounds=BOUNDS, method="L-BFGS-B")
    if not out.success:
        raise ValueError(out.message)
    return out.x


def run(a):
    by = [read_view(v) for v in (a.dense, a.sparse)]
    if set(by[0]) != set(by[1]):
        raise ValueError("Views disagree")

    def split(k):
        rs = load(a.inputs / (k + ".jsonl"))
        ext = [softmax([v[k, i] for i in range(len(rs))], axis=1) for v in by]
        return evidence(np.array([r["laya"] for r in rs]), *ext), np.array(
            [r["gold"] for r in rs]
        )

    dev = [split(k) for k in ("nli-dev", "semantic-cal")]
    x, y = (np.concatenate(v) for v in zip(*dev))
    groups = json.loads((a.inputs / "groups.json").read_text())
    g = np.array(
        ["jnli"] * 48
        + ["jsick"] * 48
        + ["world" + str(v) for v in groups["semantic-cal"]]
    )

    def loss(lam):
        pred = np.empty((len(y), 3))
        for k in np.unique(g):
            tr = g != k
            pred[~tr] = predict(x[~tr], fit(x[tr], y[tr], g[tr], lam))
        return -weights(y, g) @ np.log(pred[np.arange(len(y)), y].clip(1e-15))

    lam = min((0.03, 0.1, 0.4, 1.6), key=loss)
    c = fit(x, y, g, lam)
    a.config.write_text(
        json.dumps(
            {
                "coefficients": c.tolist(),
                "regularization": lam,
                "development_items": len(y),
                "cv_log_loss": loss(lam),
            },
            indent=2,
        )
        + "\n"
    )
    report, out = {}, []
    for k in ("nli-dev", "semantic-cal", "jsem", "challenge", "jsem-fresh"):
        xx, yy = split(k)
        p = predict(xx, c)
        report[k] = score(yy, p)
        for i, v in enumerate(p):
            out.append(
                {
                    "split": k,
                    "row": i,
                    "gold": int(yy[i]),
                    "prediction": int(v.argmax()),
                    "probabilities": v.round(7).tolist(),
                }
            )
    a.result.write_text(json.dumps(report, indent=2) + "\n")
    a.rows.write_bytes(
        gzip.compress("".join(json.dumps(r) + "\n" for r in out).encode(), mtime=0)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in {
        "inputs": ".cache/four",
        "dense": ".cache/dense.jsonl",
        "sparse": ".cache/semantic-predictions/direct-out.jsonl",
        "result": "results/four-direct-calibration.json",
        "rows": "results/four-direct-calibration.jsonl.gz",
        "config": "weights/direct-calibration.json",
    }.items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    run(p.parse_args())
