"""Replay paired predictions, checking complete IDs and equivalence to the full vote."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import softmax

from cross_family import read_view
from four_task_report import load
from semantic_vote import VOTES, decide

SIZES = {
    "nli-dev": 96,
    "semantic-cal": 144,
    "jsem": 384,
    "jsem-fresh": 96,
    "challenge": 36,
}


def evaluate(args):
    views = {name: read_view(getattr(args, name)) for name, _ in VOTES[1:]}
    expected = {(split, i) for split, n in SIZES.items() for i in range(n)}
    for name, view in views.items():
        if set(view) != expected:
            raise ValueError(f"Missing or unexpected IDs in {name}")
    config = json.loads(args.calibration.read_text()) if args.calibration else None
    results, decisions = {}, []
    for split, size in SIZES.items():
        rows = load(args.inputs / (split + ".jsonl"))
        if len(rows) != size:
            raise ValueError(f"Incorrect sample size for {split}")
        used_counts, transitions = Counter(), Counter()
        matrix = np.zeros((3, 3), dtype=int)
        for i, row in enumerate(rows):
            p = {name: softmax(view[split, i]) for name, view in views.items()}
            p["laya"] = row["laya"]
            y = int(row["gold"])
            pred, used, low, high = decide(p.__getitem__, config=config)
            full, _, probs, _ = decide(p.__getitem__, early=False, config=config)
            if (
                pred != full
                or np.any(low > probs + 1e-12)
                or np.any(high < probs - 1e-12)
            ):
                raise ValueError(f"Inconsistent decision for {split}/{i}")
            direct = int((0.1 * np.asarray(p["laya"]) + 0.9 * p["direct35"]).argmax())
            transitions[f"{int(direct == y)}-{int(pred == y)}"] += 1
            used_counts.update(used)
            matrix[y, pred] += 1
            decisions.append(
                {
                    "split": split,
                    "row": i,
                    "gold": y,
                    "choice": pred,
                    "direct35_mix": direct,
                    "used": used,
                    "full_probabilities": probs.round(10).tolist(),
                }
            )
        results[split] = {
            "n": size,
            "correct": int(matrix.trace()),
            "confusion": matrix.tolist(),
            "calls": dict(used_counts),
            "direct_to_vote": dict(transitions),
        }
    args.result.write_text(json.dumps(results, indent=2) + "\n")
    args.rows.write_bytes(
        gzip.compress(
            "".join(json.dumps(r) + "\n" for r in decisions).encode(), mtime=0
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=Path(".cache/four"))
    for name, default in (
        ("direct35", "semantic-predictions/direct-out"),
        ("direct27", "dense"),
        ("gemma", "gemma"),
        ("counter", "countermodel"),
    ):
        p.add_argument(
            "--" + name, type=Path, default=Path(".cache") / (default + ".jsonl")
        )
    p.add_argument("--calibration", type=Path)
    p.add_argument("--result", type=Path, default=Path("results/semantic-vote.json"))
    p.add_argument("--rows", type=Path, default=Path("results/semantic-vote.jsonl.gz"))
    evaluate(p.parse_args())
