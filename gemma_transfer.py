"""Cross-family Japanese inference with a fixed mixture and no target labels."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from four_task_report import load, score


def mix(prior, gemma, verifier):
    family = 0.1 * prior + 0.9 * gemma
    return family, 0.5 * family + 0.5 * verifier


def run(a):
    items = load(a.logits)
    logits = {tuple(v["id"]): v["logits"] for v in items}
    checks = load(a.verifier)
    verify = {(v["set"], v["row"]): v["probabilities"] for v in checks}
    if set(logits) != set(verify) or len(items) != len(logits):
        raise ValueError("Incomplete or duplicate logits")
    results, records = {}, []
    for split in ("nli-dev", "semantic-cal", "jsem", "challenge", "jsem-fresh"):
        rows = load(a.inputs / (split + ".jsonl"))
        keys = [(split, i) for i in range(len(rows))]
        expert = softmax([logits[k] for k in keys], axis=1)
        family, combined = mix(
            np.array([r["laya"] for r in rows]),
            expert,
            np.array([verify[k] for k in keys]),
        )
        pred = {"expert": expert, "family": family, "combined": combined}
        gold = np.array([r["gold"] for r in rows])
        results[split] = {name: score(gold, p) for name, p in pred.items()}
        for i in range(len(rows)):
            records.append(
                {
                    "split": split,
                    "row": i,
                    "gold": int(gold[i]),
                    "probabilities": {
                        n: p[i].round(7).tolist() for n, p in pred.items()
                    },
                }
            )
    a.result.write_text(json.dumps(results, indent=2) + "\n")
    a.rows.write_bytes(
        gzip.compress("".join(json.dumps(r) + "\n" for r in records).encode(), mtime=0)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key, path in {
        "inputs": ".cache/four",
        "verifier": ".cache/verifier.jsonl",
        "logits": ".cache/gemma.jsonl",
        "result": "results/four-gemma.json",
        "rows": "results/four-gemma.jsonl.gz",
    }.items():
        parser.add_argument("--" + key, type=Path, default=Path(path))
    run(parser.parse_args())
