"""Recount complete-set scores from released per-question decisions, without models."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def read(path):
    with gzip.open(path, "rt") as src:
        return list(map(json.loads, src))


def summarize(folder):
    gold = {}
    methods = defaultdict(lambda: defaultdict(dict))
    expected = {
        "jamp": 348,
        "jcommonsenseqa": 1119,
        "jsem": 384,
        "challenge": 36,
        "jsem-fresh": 96,
    }

    def add(task, idx, label, name, decision):
        if task not in expected:
            return
        if (
            idx not in range(expected[task])
            or (task, idx) in gold
            and gold[task, idx] != label
        ):
            raise ValueError("Inconsistent row or gold label")
        d = methods[name][task]
        if idx in d:
            raise ValueError(f"Duplicate {name}/{task}/{idx}")
        gold[task, idx] = label
        d[idx] = decision

    for r in read(folder / "four-evaluation.jsonl.gz"):
        split = r["split"]
        for key in (
            "laya",
            "combined",
            "template_hard",
            "temporal_graph",
            "direct_laya",
            "sparse_direct_laya",
            "verifier",
            "single_judge",
            "fixed_mix",
            "consensus",
        ):
            if key in r["prediction"]:
                n = (
                    "official_broad"
                    if key == "combined" and split == "jamp"
                    else "qa_mix"
                    if key == "combined" and split == "jcommonsenseqa"
                    else key
                )
                add(split, r["row"], r["gold"], n, r["prediction"][key])
    for file in (
        "gemma",
        "dense",
        "readings",
        "trained-vote",
        "direct-calibration",
        "countermodel",
        "cross-family",
        "temporal-transfer",
    ):
        for r in read(folder / ("four-" + file + ".jsonl.gz")):
            if file in ("gemma", "dense", "readings"):
                for key, pred in r["prediction"].items():
                    add(
                        r["split"],
                        r["row"],
                        r["gold"],
                        file + "_" + key,
                        pred,
                    )
            else:
                y = r.get("prediction")
                if y is None:
                    p = r["probabilities"]
                    y = max(range(3), key=p.__getitem__)
                add(r["split"], r["row"], r["gold"], file, y)
    result = {}
    for method, tasks in methods.items():
        result[method] = {}
        for task, rows in tasks.items():
            if len(rows) != expected[task]:
                raise ValueError(f"Incomplete {method}/{task}")
            result[method][task] = {
                "correct": sum(int(y == gold[task, i]) for i, y in rows.items()),
                "n": len(rows),
            }
    checks = [
        ("countermodel", "four-countermodel.json", "combined"),
        ("direct-calibration", "four-direct-calibration.json", None),
        ("trained-vote", "four-trained-vote.json", None),
        ("readings_single", "four-readings.json", "single"),
        ("readings_dual", "four-readings.json", "dual"),
        ("dense_single", "four-dense.json", "single"),
        ("dense_dual", "four-dense.json", "dual"),
        ("gemma_family", "four-gemma.json", "family"),
        ("gemma_combined", "four-gemma.json", "combined"),
        ("cross-family", "four-cross-family.json", None),
        ("temporal-transfer", "four-temporal-transfer.json", "combined"),
        ("verifier", "four-semantic.json", "combined"),
    ]
    for method, file, key in checks:
        raw = json.loads((folder / file).read_text())
        for task, stat in result[method].items():
            ref = raw[task][key] if key else raw[task]
            if (ref["correct"], ref["n"]) != (stat["correct"], stat["n"]):
                raise ValueError(f"Score mismatch for {method}/{task}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path(__file__).with_name("results"))
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    text = json.dumps(summarize(args.results), indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end="")
