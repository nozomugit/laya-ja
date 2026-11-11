"""Fit temporal trees on official training data, select on dev and evaluate once."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from corpus_data import load_tsv, normalized
from temporal_relations import features, probabilities


def load(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def fit(args):
    dev = load(args.inputs / "jamp-dev.jsonl")
    test = load(args.inputs / "jamp.jsonl")
    blocked = {normalized(r["text"]) for r in dev + test}
    blocked_pairs = {
        (normalized(r["text"]), normalized(r["options"][0]))
        for name in ("jsem", "challenge")
        for r in load(args.inputs / (name + ".jsonl"))
    }
    examples = {}
    file = args.corpora / "temporalNLI_dataset/dataset" / args.training_file
    for (state, _, _), gold, _, _ in load_tsv(file, "jamp", "train"):
        p, h = state.removeprefix("前提：").split("\n仮説：", 1)
        if (
            normalized(p) not in blocked
            and (normalized(p), normalized(h)) not in blocked_pairs
        ):
            examples[(p, h)] = gold
    x = np.stack([features(*pair) for pair in examples])
    y = list(examples.values())
    xd = np.stack([features(r["text"], r["options"][0]) for r in dev])
    yd = np.array([r["gold"] for r in dev])
    prior = np.array([r["laya"] for r in dev])
    best = None
    for depth in (10, 16, None):
        clf = RandomForestClassifier(
            n_estimators=18,
            max_depth=depth,
            random_state=918,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=2,
        ).fit(x, y)
        sd = clf.predict_proba(xd)
        for w in (0.1, 0.2, 0.3, 0.4):
            score = (1 - w) * sd + w * prior
            value = (
                int((score.argmax(1) == yd).sum()),
                np.log(score[np.arange(len(yd)), yd]).mean(),
            )
            if best is None or value > best[0]:
                best = (value, clf, w, depth)
    _, clf, w, depth = best
    trees = []
    for est in clf.estimators_:
        t = est.tree_
        trees.append(
            [
                [
                    int(t.feature[i]),
                    float(t.threshold[i]),
                    int(t.children_left[i]),
                    int(t.children_right[i]),
                ]
                if t.children_left[i] >= 0
                else [-1, *(t.value[i, 0] / t.value[i, 0].sum()).round(8).tolist()]
                for i in range(t.node_count)
            ]
        )
    model = {"laya_weight": w, "trees": trees}
    encoded = (json.dumps(model, separators=(",", ":")) + "\n").encode()
    args.weights.write_bytes(gzip.compress(encoded, mtime=0))
    scores = [probabilities(model, r["text"], r["options"][0], r["laya"]) for r in test]
    report = {
        "training_file": args.training_file,
        "training": len(x),
        "dev_correct": best[0][0],
        "dev_total": len(dev),
        "max_depth": depth,
        "laya_weight": w,
        "correct": sum(int(z.argmax() == r["gold"]) for z, r in zip(scores, test)),
        "total": len(test),
    }
    args.result.write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in {
        "inputs": ".cache/four",
        "corpora": ".cache/corpora",
        "weights": "weights/temporal.json.gz",
        "result": "results/four-temporal.json",
    }.items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    p.add_argument(
        "--training-file",
        choices=["train_all.tsv", "train_custom-template-hard.tsv"],
        default="train_all.tsv",
    )
    fit(p.parse_args())
