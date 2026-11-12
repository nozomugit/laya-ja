"""Label Japanese sentence pairs with the two direct external judges, for distillation into Laya.

prepare: sample pairs (by default from the Japanese translation of MultiNLI) and write the same direct
three-way prompts used in the semantic comparison. collect: average the two judges' probabilities.
The gold label of the source corpus is kept only to measure and filter agreement.
"""

import argparse
import hashlib
import json
import random
import unicodedata
from pathlib import Path

import numpy as np

from four_task_prompts import nli_prompt

SOURCE = "cl-nagoya/nu-mnli"
NAMES = {"entailment": 0, "contradiction": 1, "neutral": 2}


def clean(text):
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).split())


def source_pairs():
    # Read the parquet files directly; the corpus stores labels as 0 entailment, 1 neutral, 2 contradiction.
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(
        f
        for f in list_repo_files(SOURCE, repo_type="dataset")
        if f.endswith(".parquet") and "train" in f
    )
    if not files:
        raise ValueError("No training parquet file in the source corpus")
    order = {0: 0, 1: 2, 2: 1}
    index = 0
    for name in files:
        frame = pd.read_parquet(hf_hub_download(SOURCE, name, repo_type="dataset"))
        premise = next(c for c in ("premise_ja", "premise") if c in frame.columns)
        hypothesis = next(
            c for c in ("hypothesis_ja", "hypothesis") if c in frame.columns
        )
        for p, h, label in zip(frame[premise], frame[hypothesis], frame["label"]):
            index += 1
            if isinstance(label, str):
                label = NAMES.get(label.strip().lower())
            elif int(label) in order:
                label = order[int(label)]
            else:
                label = None
            if label is not None:
                yield index - 1, p, h, label


def local_pairs(path):
    for i, line in enumerate(Path(path).read_text().splitlines()):
        r = json.loads(line)
        yield i, r["premise"], r["hypothesis"], int(r["label"])


def prepare(a):
    rows = local_pairs(a.pairs) if a.pairs else source_pairs()
    by_label, seen = {0: [], 1: [], 2: []}, set()
    for i, premise, hypothesis, label in rows:
        p, h = clean(premise), clean(hypothesis)
        key = (p, h)
        if (
            not 8 <= len(p) <= a.max_premise
            or not 4 <= len(h) <= a.max_hypothesis
            or p == h
            or key in seen
        ):
            continue
        seen.add(key)
        by_label[label].append((i, p, h, label))
    rng = random.Random(a.seed)
    chosen = []
    for label, group in by_label.items():
        rng.shuffle(group)
        chosen += group[: a.count // 3]
    rng.shuffle(chosen)
    a.out.mkdir(exist_ok=True, parents=True)
    digest = hashlib.sha256()
    with (a.out / "pairs.jsonl").open("w") as f:
        for i, p, h, label in chosen:
            line = json.dumps(
                {"id": i, "premise": p, "hypothesis": h, "gold": label},
                ensure_ascii=False,
            )
            digest.update(line.encode())
            f.write(line + "\n")
    for start in range(0, len(chosen), a.chunk):
        part = chosen[start : start + a.chunk]
        with (a.out / f"prompts-{start // a.chunk:03d}.jsonl").open("w") as f:
            for i, p, h, _ in part:
                f.write(
                    json.dumps(
                        {"id": i, "prompt": nli_prompt(p, h)}, ensure_ascii=False
                    )
                    + "\n"
                )
    meta = {
        "source": a.pairs and str(a.pairs) or SOURCE,
        "pairs": len(chosen),
        "per_label": {k: min(len(v), a.count // 3) for k, v in by_label.items()},
        "pairs_sha256": digest.hexdigest(),
        "seed": a.seed,
    }
    (a.out / "prepare.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta)


def read_logits(folder, judge):
    out = {}
    for path in sorted(folder.glob(f"{judge}-*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            out[r["id"]] = r["logits"]
    return out


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def collect(a):
    pairs = [json.loads(s) for s in (a.out / "pairs.jsonl").read_text().splitlines()]
    dense, sparse = read_logits(a.out, "dense"), read_logits(a.out, "sparse")
    rows, agree = [], {"dense": 0, "sparse": 0, "teacher": 0, "judges": 0}
    for r in pairs:
        if r["id"] not in dense or r["id"] not in sparse:
            continue
        d, s = softmax(dense[r["id"]]), softmax(sparse[r["id"]])
        t = (d + s) / 2
        agree["dense"] += int(d.argmax() == r["gold"])
        agree["sparse"] += int(s.argmax() == r["gold"])
        agree["teacher"] += int(t.argmax() == r["gold"])
        agree["judges"] += int(d.argmax() == s.argmax())
        rows.append(
            dict(
                r,
                teacher=t.round(6).tolist(),
                dense=d.round(6).tolist(),
                sparse=s.round(6).tolist(),
            )
        )
    if not rows:
        raise ValueError("No pair has both judges' logits yet.")
    with (a.out / "teacher.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kept = sum(int(np.argmax(r["teacher"]) == r["gold"]) for r in rows)
    report = {
        "labeled": len(rows),
        "of_pairs": len(pairs),
        "agreement_with_source_label": {
            k: v / len(rows) for k, v in agree.items() if k != "judges"
        },
        "judges_agree_with_each_other": agree["judges"] / len(rows),
        "kept_when_teacher_matches_source_label": kept,
        "teacher_predictions": np.bincount(
            [int(np.argmax(r["teacher"])) for r in rows], minlength=3
        ).tolist(),
    }
    (a.out / "collect.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["prepare", "collect"])
    ap.add_argument("--out", type=Path, default=Path(".cache/teacher"))
    ap.add_argument(
        "--pairs",
        type=Path,
        help="JSONL of premise, hypothesis, label instead of the default corpus",
    )
    ap.add_argument("--count", type=int, default=12000)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--max-premise", type=int, default=100)
    ap.add_argument("--max-hypothesis", type=int, default=50)
    ap.add_argument("--seed", type=int, default=5005)
    args = ap.parse_args()
    {"prepare": prepare, "collect": collect}[args.mode](args)
