"""Evaluate a fixed counterexample-search instruction on independent splits."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from four_task_report import load, score


def run(a):
    raw = load(a.logits)
    logits = {tuple(v["id"]): v["logits"] for v in raw}
    if len(raw) != len(logits):
        raise ValueError("Repeated row")
    result, records = {}, []
    required = set()
    for split in ("nli-dev", "semantic-cal", "challenge", "jsem", "jsem-fresh"):
        rows = load(a.inputs / (split + ".jsonl"))
        keys = [(split, i) for i in range(len(rows))]
        required.update(keys)
        expert = softmax([logits[key] for key in keys], axis=1)
        prob = 0.9 * expert + 0.1 * np.array([r["laya"] for r in rows])
        gold = np.array([r["gold"] for r in rows])
        result[split] = {"combined": score(gold, prob), "expert": score(gold, expert)}
        for i, p in enumerate(prob):
            records.append(
                {
                    "split": split,
                    "row": i,
                    "gold": int(gold[i]),
                    "prediction": int(p.argmax()),
                    "probabilities": p.round(7).tolist(),
                }
            )
    if required != set(logits):
        raise ValueError("Unexpected inputs")
    a.result.write_text(json.dumps(result, indent=2) + "\n")
    a.rows.write_bytes(
        gzip.compress("".join(json.dumps(r) + "\n" for r in records).encode(), mtime=0)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, value in {
        "inputs": ".cache/four",
        "logits": ".cache/countermodel.jsonl",
        "result": "results/four-countermodel.json",
        "rows": "results/four-countermodel.jsonl.gz",
    }.items():
        p.add_argument("--" + k, type=Path, default=Path(value))
    run(p.parse_args())
