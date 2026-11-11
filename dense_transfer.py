"""Use two frozen direct semantic judgments, with the same weights on every split."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from cross_family import read_view
from four_task_report import load, score


def mix(native, dense, sparse):
    return 0.1 * native + 0.9 * dense, 0.1 * native + 0.45 * dense + 0.45 * sparse


def run(a):
    dense, sparse = read_view(a.dense), read_view(a.sparse)
    if set(dense) != set(sparse):
        raise ValueError("Different source rows")
    rep, out, expected = {}, [], set()
    for split in ("nli-dev", "semantic-cal", "jsem", "challenge", "jsem-fresh"):
        rows = load(a.inputs / (split + ".jsonl"))
        keys = [(split, i) for i in range(len(rows))]
        expected.update(keys)
        p = softmax([dense[k] for k in keys], axis=1)
        s = softmax([sparse[k] for k in keys], axis=1)
        single, dual = mix(np.array([r["laya"] for r in rows]), p, s)
        methods = {"dense_only": p, "single": single, "dual": dual}
        gold = np.array([r["gold"] for r in rows])
        rep[split] = {k: score(gold, z) for k, z in methods.items()}
        for i, y in enumerate(gold):
            out.append(
                {
                    "split": split,
                    "row": i,
                    "gold": int(y),
                    "prediction": {k: int(z[i].argmax()) for k, z in methods.items()},
                    "probabilities": {
                        k: z[i].round(7).tolist() for k, z in methods.items()
                    },
                }
            )
    if expected != set(dense):
        raise ValueError("Unexpected source rows")
    a.result.write_text(json.dumps(rep, indent=2) + "\n")
    a.rows.write_bytes(
        gzip.compress("".join(json.dumps(r) + "\n" for r in out).encode(), mtime=0)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in {
        "inputs": ".cache/four",
        "dense": ".cache/dense.jsonl",
        "sparse": ".cache/semantic-predictions/direct-out.jsonl",
        "result": "results/four-dense.json",
        "rows": "results/four-dense.jsonl.gz",
    }.items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    run(p.parse_args())
