"""Pool independent Japanese inference models without estimating weights on targets."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from four_task_report import load, score


def pool(native, gemma, countermodel, verifier):
    # Give the two independent reasoners equal weight; allow the native and
    # truth/falsity probes to retain a vote where the reasoners disagree.
    return 0.1 * native + 0.4 * gemma + 0.4 * countermodel + 0.1 * verifier


def read_view(path):
    rows = load(path)
    result = {tuple(r["id"]): r["logits"] for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate rows in {path}")
    return result


def run(a):
    gemma, counter = [read_view(v) for v in (a.gemma, a.countermodel)]
    raw = load(a.verifier)
    verify = {(r["set"], r["row"]): r["probabilities"] for r in raw}
    if (
        set(gemma) != set(counter)
        or set(gemma) != set(verify)
        or len(raw) != len(verify)
    ):
        raise ValueError("Sources do not cover the same inputs")
    records, report, required = [], {}, set()
    for split in ("nli-dev", "semantic-cal", "challenge", "jsem", "jsem-fresh"):
        rows = load(a.inputs / (split + ".jsonl"))
        keys = [(split, i) for i in range(len(rows))]
        required.update(keys)
        y = np.array([r["gold"] for r in rows])
        p = pool(
            np.array([r["laya"] for r in rows]),
            softmax([gemma[k] for k in keys], axis=1),
            softmax([counter[k] for k in keys], axis=1),
            np.array([verify[k] for k in keys]),
        )
        report[split] = score(y, p)
        for i, probs in enumerate(p):
            records.append(
                {
                    "split": split,
                    "row": i,
                    "gold": int(y[i]),
                    "prediction": int(probs.argmax()),
                    "probabilities": probs.round(7).tolist(),
                }
            )
    if required != set(gemma):
        raise ValueError("Unexpected rows")
    a.result.write_text(json.dumps(report, indent=2) + "\n")
    a.rows.write_bytes(
        gzip.compress("".join(json.dumps(r) + "\n" for r in records).encode(), mtime=0)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, path in {
        "inputs": ".cache/four",
        "verifier": ".cache/verifier.jsonl",
        "gemma": ".cache/gemma.jsonl",
        "countermodel": ".cache/countermodel.jsonl",
        "result": "results/four-cross-family.json",
        "rows": "results/four-cross-family.jsonl.gz",
    }.items():
        p.add_argument("--" + k, type=Path, default=Path(path))
    run(p.parse_args())
