"""Check rule reliability by training template before using the temporal graph."""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

from corpus_data import normalized
from temporal_graph import predict, resolve
from temporal_relations import probabilities


def fit(a):
    def rows(name):
        return [
            json.loads(s)
            for s in (a.inputs / (name + ".jsonl")).read_text().splitlines()
        ]

    dev, test = rows("jamp-dev"), rows("jamp")
    blocked = {normalized(r["text"]) for r in dev + test}
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    with a.train.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if normalized(r["premise"]) in blocked:
                continue
            found = resolve(r["premise"], r["hypothesis"])
            if found is None:
                continue
            kind, y = found
            n = counts[kind][r["template_num"]]
            n[0] += int(
                ("entailment", "contradiction", "neutral")[y] == r["gold_label"]
            )
            n[1] += 1
    # Apply only those operations that are accurate across training templates.
    supported = sorted(
        kind
        for kind, cs in counts.items()
        if sum(v[1] for v in cs.values()) >= 20
        and min(v[0] / v[1] for v in cs.values()) >= 0.9
    )
    a.config.write_text(json.dumps({"supported": supported}, indent=2) + "\n")
    model = json.loads(gzip.decompress(a.model.read_bytes()))
    out = {
        "training": a.train.name,
        "by_training_template": counts,
        "supported": supported,
    }
    for split, rs in (("dev", dev), ("test", test)):
        correct = 0
        for r in rs:
            p, h = r["text"], r["options"][0]
            base = probabilities(model, p, h, r["laya"])
            correct += int(predict(base, p, h, supported).argmax() == r["gold"])
        out[split] = {"correct": correct, "n": len(rs)}
    a.result.write_text(json.dumps(out, indent=2) + "\n")
    print(supported, out["dev"], out["test"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for k, v in {
        "inputs": ".cache/four",
        "train": ".cache/corpora/temporalNLI_dataset/dataset/train_custom-template-hard.tsv",
        "model": "weights/temporal-hard.json.gz",
        "result": "results/four-temporal-graph.json",
        "config": "weights/temporal-graph.json",
    }.items():
        parser.add_argument("--" + k, type=Path, default=Path(v))
    fit(parser.parse_args())
