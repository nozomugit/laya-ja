"""Use the same direct zero-shot judge for temporal patterns outside the grammar."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from four_task_report import load, score
from temporal_graph import predict as graph
from temporal_graph import resolve
from temporal_relations import probabilities


def combine(base, expert, p, h, supported):
    found = resolve(p, h)
    if found and found[0] in supported:
        return graph(base, p, h, supported)
    return 0.35 * base + 0.65 * expert


def run(a):
    model = json.loads(
        gzip.decompress((a.weights / "temporal-hard.json.gz").read_bytes())
    )
    supported = json.loads((a.weights / "temporal-graph.json").read_text())["supported"]
    items = load(a.logits)
    logits = {tuple(q["id"]): q["logits"] for q in items}
    if len(logits) != len(items):
        raise ValueError("Repeated row")
    rep, out, expected = {}, [], set()
    for split in ("jamp-dev", "jamp"):
        rows = load(a.inputs / (split + ".jsonl"))
        ps, es, ys = [], [], []
        for i, row in enumerate(rows):
            p, h = row["text"], row["options"][0]
            key = (split, i)
            expected.add(key)
            expert = softmax(logits[key])
            base = probabilities(model, p, h, row["laya"])
            z = combine(base, expert, p, h, supported)
            ps.append(z)
            es.append(expert)
            ys.append(row["gold"])
            out.append(
                {
                    "split": split,
                    "row": i,
                    "gold": row["gold"],
                    "prediction": int(z.argmax()),
                    "probabilities": z.round(7).tolist(),
                }
            )
        rep[split] = {
            "combined": score(np.array(ys), np.array(ps)),
            "expert": score(np.array(ys), np.array(es)),
        }
    if expected != set(logits):
        raise ValueError("Unexpected rows")
    a.result.write_text(json.dumps(rep, indent=2) + "\n")
    a.rows.write_bytes(
        gzip.compress("".join(json.dumps(r) + "\n" for r in out).encode(), mtime=0)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in {
        "weights": "weights",
        "inputs": ".cache/four",
        "logits": ".cache/temporal.jsonl",
        "result": "results/four-temporal-transfer.json",
        "rows": "results/four-temporal-transfer.jsonl.gz",
    }.items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    run(p.parse_args())
