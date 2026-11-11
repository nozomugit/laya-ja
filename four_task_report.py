"""Fix the fusion on separate development questions, then score every test item."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def score(y, p):
    pred = p.argmax(1)
    nclass = p.shape[1]
    confusion = np.zeros((nclass, nclass), dtype=int)
    np.add.at(confusion, (y, pred), 1)
    return {
        "n": len(y),
        "correct": int((y == pred).sum()),
        "accuracy": float(np.mean(y == pred)),
        "confusion": confusion.tolist(),
    }


def shift(p, bias):
    return softmax(np.log(np.maximum(p, 1e-15)) + bias, axis=-1)


def fit_bias(p, y, penalty):
    if penalty is None:
        return np.zeros(3)
    weights = 1 / np.bincount(y, minlength=3)[y]
    weights /= weights.sum()

    def loss(bias):
        q = shift(p, bias)
        cost = -weights @ np.log(q[np.arange(len(y)), y]) + penalty * (bias @ bias)
        q[np.arange(len(y)), y] -= 1
        return cost, weights @ q + 2 * penalty * bias

    return minimize(loss, np.zeros(3), jac=True, method="L-BFGS-B").x


def select_bias(p, y):
    # Each half is a different corpus (JNLI, then JSICK).
    halves = np.array_split(np.arange(len(y)), 2)

    def validation(penalty):
        acc, ce = [], []
        for train, valid in (halves, halves[::-1]):
            probs = shift(p[valid], fit_bias(p[train], y[train], penalty))
            truth = y[valid]
            acc.extend(
                np.mean(probs[truth == k].argmax(1) == k) for k in np.unique(truth)
            )
            ce.extend(np.log(probs[np.arange(len(truth)), truth]))
        return np.mean(acc), np.mean(ce)

    penalty = max((None, 0.05, 0.15, 0.4), key=validation)
    return fit_bias(p, y, penalty), penalty


def run(a):
    result = {}
    group_path = a.inputs / "groups.json"
    groups = json.loads(group_path.read_text()) if group_path.exists() else {}
    for key in ("qa-dev", "jcommonsenseqa", "nli-dev", "challenge", "jsem"):
        rows = load(a.inputs / (key + ".jsonl"))
        prior = np.array([r["laya"] for r in rows])
        gold = np.array([r["gold"] for r in rows])
        if key in ("qa-dev", "jcommonsenseqa"):
            preds = load(a.pred / (key + ".jsonl"))
            if [r["i"] for r in preds] != list(range(len(rows))):
                raise ValueError(f"Incomplete probability file for {key}")
            auxiliary = np.array([r["specialist"] for r in preds])
            # Published Jev trained on the official training split, which includes
            # local QA tune rows. Fix this weight without optimizing on those labels.
            final = 0.1 * prior + 0.9 * auxiliary
            detail = {"specialist_only": score(gold, auxiliary)}
        else:
            views = []
            for style in ("direct", "contrast"):
                if a.http:
                    fn = key + ("-contrast" if style == "contrast" else "") + ".jsonl"
                    preds = load(a.pred / fn)
                    if [r["i"] for r in preds] != list(range(len(rows))):
                        raise ValueError(f"Incomplete {fn}")
                    views.append(np.array([r["specialist"] for r in preds]))
                else:
                    preds = {
                        i: v["logits"]
                        for v in load(a.pred / (style + "-out.jsonl"))
                        for split, i in [v["id"]]
                        if split == key
                    }
                    if sorted(preds) != list(range(len(rows))):
                        raise ValueError(f"Incomplete {style} for {key}")
                    p = softmax([preds[i] for i in range(len(rows))], axis=1)
                    views.append(p[:, [1, 2, 0]] if style == "contrast" else p)

            def mix(w, v, prior=prior, views=views):
                return w * prior + (1 - w) * ((1 - v) * views[0] + v * views[1])

            if key == "nli-dev":

                def merit(pair, gold=gold):
                    p = mix(*pair)
                    by_class = [np.mean(p[gold == c].argmax(1) == c) for c in range(3)]
                    return np.mean(by_class), np.mean(
                        np.log(p[np.arange(len(gold)), gold])
                    )

                native_weight, contrast_weight = max(
                    (
                        (w, v)
                        for w in (0.05, 0.1, 0.2, 0.3)
                        for v in (0, 0.25, 0.5, 0.75, 1)
                    ),
                    key=merit,
                )
            final = mix(native_weight, contrast_weight)
            plain = score(gold, final)
            if key == "nli-dev":
                bias, penalty = select_bias(final, gold)
            final = shift(final, bias)
            detail = {
                "before_bias": plain,
                "direct_only": score(gold, views[0]),
                "direct_laya": score(gold, 0.1 * prior + 0.9 * views[0]),
                "contrast_only": score(gold, views[1]),
            }
        result[key] = {
            "inputs_sha256": hashlib.sha256(
                (a.inputs / (key + ".jsonl")).read_bytes()
            ).hexdigest(),
            "laya_context_only": score(gold, prior),
            **detail,
            "combined": score(gold, final),
        }
        if key in groups:
            g = np.array(groups[key])
            if len(g) != len(gold):
                raise ValueError("Wrong group count")
            result[key]["groups"] = {
                str(k): score(gold[g == k], final[g == k]) for k in np.unique(g)
            }
    config = {
        "qa_laya_weight": 0.1,
        "nli_laya_weight": native_weight,
        "contrast_weight": contrast_weight,
        "label_bias": bias.tolist(),
        "bias_penalty": penalty,
    }
    a.config.write_text(json.dumps(config, indent=2) + "\n")
    a.result.write_text(json.dumps(result, indent=2) + "\n")
    for k, v in result.items():
        print(k, v["combined"]["correct"], v["combined"]["n"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in {
        "inputs": ".cache/four",
        "pred": ".cache/four-predictions",
        "config": "weights/four-fusion.json",
        "result": "results/four-fusion.json",
    }.items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    p.add_argument(
        "--http", action="store_true", help="Use JSONL from the Python HTTP scorer"
    )
    run(p.parse_args())
