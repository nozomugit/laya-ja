"""Create blind binary prompts for the direct judges' disagreements or score completed outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from audit_direct_vote import digest
from corpus_data import jsem
from cross_family import read_view
from four_task_report import load
from semantic_pairwise import candidates, combine, prompts


def aligned_rows(args):
    dense, sparse = read_view(args.dense), read_view(args.sparse)
    sizes = {
        "nli-dev": 96,
        "semantic-cal": 144,
        "jsem": 384,
        "challenge": 36,
        "jsem-fresh": 96,
    }
    cases = {k: load(args.inputs / (k + ".jsonl")) for k in sizes}
    expected = {(k, i) for k, n in sizes.items() for i in range(n)}
    if (
        set(dense) != expected
        or set(sparse) != expected
        or any(len(cases[k]) != n for k, n in sizes.items())
    ):
        raise ValueError("Misaligned source rows")
    if args.audit:
        corpus = {
            r[3]: r[0][0].removeprefix("前提：").split("\n仮説：", 1)
            for r in jsem(args.corpora)
        }
        sample = json.loads((args.audit / "sample.json").read_text())
        cases["audit36"] = [
            {**r, "text": corpus[r["id"]][0], "options": [corpus[r["id"]][1]]}
            for r in sample
        ]
        if len(sample) != 36 or len({r["id"] for r in sample}) != 36:
            raise ValueError("Invalid extra audit sample")
        for r in sample:
            premise, hypothesis = corpus[r["id"]]
            if digest(premise + "\n" + hypothesis) != r["hash"]:
                raise ValueError(f"Corpus text mismatch for {r['id']}")
        for view, name in ((dense, "direct27"), (sparse, "direct35")):
            rows = load(args.audit / (name + "-out.jsonl"))
            if [r["id"] for r in rows] != list(range(36)):
                raise ValueError("Misaligned extra audit logits")
            view.update({("audit36", r["id"]): r["logits"] for r in rows})
    for split, rows in cases.items():
        for i, row in enumerate(rows):
            yield (split, i), row, softmax(dense[split, i]), softmax(sparse[split, i])


def prepare(args):
    with args.prompts.open("w") as out:
        for (split, i), row, dense, sparse in aligned_rows(args):
            labels = candidates(dense, sparse)
            if len(labels) > 1:
                for j, q in enumerate(prompts(row["text"], row["options"][0], labels)):
                    out.write(
                        json.dumps(
                            {"id": [split, i, j], "prompt": q}, ensure_ascii=False
                        )
                        + "\n"
                    )


def score(args):
    evidence = read_view(args.responses)
    used = set()
    scores = {}
    for (split, i), row, dense, sparse in aligned_rows(args):
        labels = candidates(dense, sparse)
        z = None
        if len(labels) > 1:
            keys = [(split, i, j) for j in (0, 1)]
            z = [evidence[key] for key in keys]
            used.update(keys)
        gold = row["gold"]
        base = int((0.1 * np.asarray(row["laya"]) + 0.45 * (dense + sparse)).argmax())
        choice = int(combine(row["laya"], dense, sparse, z).argmax())
        r = scores.setdefault(
            split,
            {
                "n": 0,
                "correct": 0,
                "fixed": 0,
                "changed": 0,
                "gained": 0,
                "lost": 0,
                "confusion": [[0] * 3 for _ in range(3)],
            },
        )
        r["n"] += 1
        r["correct"] += choice == gold
        r["fixed"] += base == gold
        r["changed"] += choice != base
        r["gained"] += base != gold and choice == gold
        r["lost"] += base == gold and choice != gold
        r["confusion"][gold][choice] += 1
    if used != set(evidence):
        raise ValueError("Unexpected pairwise predictions")
    args.result.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "score"))
    for k, name in (
        ("dense", "dense.jsonl"),
        ("sparse", "semantic-predictions/direct-out.jsonl"),
        ("inputs", "four"),
        ("corpora", "corpora"),
        ("prompts", "pairwise-in.jsonl"),
        ("responses", "pairwise-out.jsonl"),
    ):
        p.add_argument("--" + k, type=Path, default=Path(".cache") / name)
    p.add_argument(
        "--audit",
        type=Path,
        help="Optional directory with the additional 36-case audit",
    )
    p.add_argument(
        "--result", type=Path, default=Path("results/semantic-pairwise.json")
    )
    args = p.parse_args()
    (prepare if args.action == "prepare" else score)(args)
