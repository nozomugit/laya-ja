"""Fit a Japanese sentence-edit classifier and combine it with the native Laya head."""

import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression

from context_tasks import ContextReader
from corpus_data import normalized, splits
from experiment import partition
from four_task_assist import LABELS
from four_task_report import load, score
from logic_experiment import generate
from semantic_verifier import weights

HASHER = FeatureHasher(n_features=8192, input_type="string")


def grams(text):
    yield text
    for n in (1, 2, 3):
        for i in range(len(text) - n + 1):
            yield text[i : i + n]


def features(p, h):
    def clean(s):
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s)).strip("。")

    p, h = clean(p), clean(h)
    marks = []
    matcher = difflib.SequenceMatcher(None, p, h, autojunk=False)
    for tag, a, b, c, d in matcher.get_opcodes():
        if tag == "equal":
            continue
        left, right = p[a:b], h[c:d]
        marks.extend(tag + "-old:" + g for g in set(grams(left)))
        marks.extend(tag + "-new:" + g for g in set(grams(right)))
        marks.append(tag + "-pair:" + left[-6:] + ">" + right[-6:])
        marks.append(
            tag + "-size:" + str(min(len(left), 5)) + ":" + str(min(len(right), 5))
        )
    marks.extend(
        ["overlap:" + str(int(10 * matcher.ratio())), "contained:" + str(h in p)]
    )
    return marks


def matrix(rows):
    return HASHER.transform(features(r["text"], r["options"][0]) for r in rows)


def predict(rows, coef, bias, weight):
    model = softmax(matrix(rows) @ coef.T + bias, axis=-1)
    native = np.array([r["laya"] for r in rows])
    return native * (1 - weight) + model * weight


def fit(args):
    old = splits(args.corpora, args.data)
    docs = {
        "jnli": partition("jnli", args.data, 42, 4096, 192, 10**9)["train"],
        "jsick": old["jsick-train"],
        "jamp": old["jamp-train"],
        "synthetic": generate("fit", 1800, 941),
    }
    held = {
        normalized(r["text"])
        for key in ("nli-dev", "semantic-cal", "jsem", "jsem-fresh", "challenge")
        for r in load(args.inputs / (key + ".jsonl"))
    }
    rows, y, dom = [], [], []
    for group, items in docs.items():
        for (state, _, _), label, *_ in items:
            p, h = state.removeprefix("前提：").split("\n仮説：", 1)
            if normalized(p) not in held:
                rows.append({"text": p, "options": [h]})
                y.append(label)
                dom.append(group)
    y, dom = np.asarray(y), np.asarray(dom)
    model = LogisticRegression(C=0.35, solver="lbfgs", max_iter=600, tol=1e-5)
    model.fit(matrix(rows), y, sample_weight=len(y) * weights(y, dom))
    if model.n_iter_.max() >= model.max_iter:
        raise RuntimeError("Edit classifier did not converge")
    coef, bias = model.coef_.astype("float32"), model.intercept_.astype("float32")
    groups = json.loads((args.inputs / "groups.json").read_text())["semantic-cal"]
    g = np.array(["jnli"] * 48 + ["jsick"] * 48 + ["world-" + str(i) for i in groups])
    dev = load(args.inputs / "nli-dev.jsonl") + load(args.inputs / "semantic-cal.jsonl")
    gold = np.array([r["gold"] for r in dev])
    dp = weights(gold, g)

    def loss(w):
        prob = predict(dev, coef, bias, w)
        return -dp @ np.log(np.maximum(prob[np.arange(len(dev)), gold], 1e-15))

    grid = (0, 0.1, 0.25, 0.5, 0.75, 0.9)
    mix = min(grid, key=loss)
    np.savez_compressed(args.weights, coef=coef, bias=bias, mix=mix)
    result = {
        "train": {name: int((dom == name).sum()) for name in docs},
        "mix": mix,
        "development_log_loss": {str(w): float(loss(w)) for w in grid},
        "evaluation": {},
    }
    for k in ("jsem", "jsem-fresh", "challenge"):
        test = load(args.inputs / (k + ".jsonl"))
        gold = np.array([r["gold"] for r in test])
        result["evaluation"][k] = score(gold, predict(test, coef, bias, mix))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=Path("weights/native-edit.npz"))
    p.add_argument("--model-path")
    sub = p.add_subparsers(dest="task", required=True)
    t = sub.add_parser("fit")
    for k in ("inputs", "corpora", "data"):
        t.add_argument(
            "--" + k,
            type=Path,
            default=Path(".cache") / ("four" if k == "inputs" else k),
        )
    t.add_argument("--output", type=Path, default=Path("results/native-edit.json"))
    q = sub.add_parser("predict")
    q.add_argument("premise")
    q.add_argument("hypothesis")
    args = p.parse_args()
    if args.task == "fit":
        fit(args)
    else:
        raw = ContextReader("nli", args.model_path).predict(
            args.premise, [args.hypothesis]
        )
        row = {
            "text": args.premise,
            "options": [args.hypothesis],
            "laya": [raw["probabilities"][k] for k in LABELS],
        }
        with np.load(args.weights) as z:
            prob = predict([row], z["coef"], z["bias"], float(z["mix"]))[0]
        print(
            json.dumps(
                {
                    "choice": LABELS[prob.argmax()],
                    "probabilities": dict(zip(LABELS, prob.tolist())),
                },
                ensure_ascii=False,
            )
        )
