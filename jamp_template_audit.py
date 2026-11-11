"""Report item-level performance separately for seen and unseen Jamp templates."""

import argparse
import csv
import gzip
import json
from pathlib import Path


def run(a):
    with (a.corpora / "train_custom-template-hard.tsv").open() as f:
        trained = {r["template_num"] for r in csv.DictReader(f, delimiter="\t")}
    with (a.corpora / "test_2.tsv").open() as f:
        templates = {
            (r["premise"], r["hypothesis"]): r["template_num"]
            for r in csv.DictReader(f, delimiter="\t")
        }
    questions = list(map(json.loads, a.items.read_text().splitlines()))

    def grab(path, kind):
        with gzip.open(path, "rt") as f:
            return {
                r["row"]: r[kind] for r in map(json.loads, f) if r["split"] == "jamp"
            }

    outputs = grab(a.results / "four-evaluation.jsonl.gz", "prediction")
    transfer = grab(a.results / "four-temporal-transfer.jsonl.gz", "prediction")
    if set(outputs) != set(transfer) or len(questions) != len(outputs):
        raise ValueError("Task rows disagree")
    counts = {}
    for i, r in enumerate(questions):
        bucket = (
            "seen" if templates[r["text"], r["options"][0]] in trained else "unseen"
        )
        d = counts.setdefault(bucket, {"n": 0, "correct": {}})
        d["n"] += 1
        for name, v in {**outputs[i], "transfer": transfer[i]}.items():
            c = d["correct"]
            c[name] = c.get(name, 0) + int(v == r["gold"])
    (a.results / "four-template-slices.json").write_text(
        json.dumps(counts, indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpora",
        type=Path,
        default=Path(".cache/corpora/temporalNLI_dataset/dataset"),
    )
    p.add_argument("--items", type=Path, default=Path(".cache/four/jamp.jsonl"))
    p.add_argument("--results", type=Path, default=Path("results"))
    run(p.parse_args())
