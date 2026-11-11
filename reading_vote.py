"""Average independently worded direct judgments under fixed model and native weights."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from cross_family import read_view
from four_task_report import load, score


def mix(native, base, dense, first, second):
    # second uses A=contradiction, B=neutral, C=entailment.
    total = base + first + np.asarray(second)[..., [2, 0, 1]]
    return 0.1 * native + 0.3 * total, 0.1 * native + 0.225 * (total + dense)


def run(a):
    views = [read_view(getattr(a, k)) for k in ("base", "dense", "first", "second")]
    if any(set(z) != set(views[0]) for z in views[1:]):
        raise ValueError("Different source rows")
    rep, out, expected = {}, [], set()
    for split in ("nli-dev", "semantic-cal", "jsem", "challenge", "jsem-fresh"):
        rows = load(a.inputs / (split + ".jsonl"))
        keys = [(split, i) for i in range(len(rows))]
        expected.update(keys)
        p, q = mix(
            np.array([r["laya"] for r in rows]),
            *[softmax([z[k] for k in keys], axis=1) for z in views],
        )
        methods = {"single": p, "dual": q}
        y = np.array([r["gold"] for r in rows])
        rep[split] = {k: score(y, x) for k, x in methods.items()}
        for i, v in enumerate(y):
            out.append(
                {
                    "split": split,
                    "row": i,
                    "gold": int(v),
                    "prediction": {k: int(z[i].argmax()) for k, z in methods.items()},
                    "probabilities": {
                        k: z[i].round(7).tolist() for k, z in methods.items()
                    },
                }
            )
    if expected != set(views[0]):
        raise ValueError("Unexpected rows")
    a.result.write_text(json.dumps(rep, indent=2) + "\n")
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
        "result": "results/four-readings.json",
        "rows": "results/four-readings.jsonl.gz",
    }.items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    run(p.parse_args())
