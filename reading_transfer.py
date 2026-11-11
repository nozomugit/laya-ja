"""Test transfer from short Wikipedia passages (JSQuAD) to Japanese five-choice QA."""

import argparse
import hashlib
import json
import os
import random
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from compare_heads import readout, save_selected, select_head
from experiment import (
    DATA_REV,
    bootstrap_difference,
    combine,
    extract,
    fit_weight,
    get_cache,
)
from laya_ja import BASE, REVISION, load, qa_request
from scoring import metrics

DIGEST = {
    "train": "3879e9df049fc4779429e33ef19c0b2685947e211fef538dc85a0561f40e36d0",
    "valid": "fb0a57a35281ff03bbe7f0e878cff5b577985fbe6fcb242e7e8ddb63d1c0ab11",
}


def articles(folder, split):
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / f"jsquad-{split}.json"
    if path.exists():
        data = path.read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/yahoojapan/JGLUE/{DATA_REV}/datasets/jsquad-v1.3/{split}-v1.3.json"
        data = urllib.request.urlopen(url, timeout=90).read()
    if hashlib.sha256(data).hexdigest() != DIGEST[split]:
        raise ValueError("JSQuAD checksum mismatch")
    if not path.exists():
        path.write_bytes(data)
    return json.loads(data)["data"]


def convert(articles_, limit, seed):
    rng = random.Random(seed)
    examples = []
    for article in articles_:
        title = article["title"]
        pool = sorted(
            {
                a["text"].strip()
                for p in article["paragraphs"]
                for q in p["qas"]
                for a in q["answers"]
                if 0 < len(a["text"].strip()) <= 24
            }
        )
        for p in article["paragraphs"]:
            ctx = p["context"]
            if len(ctx) > 180:
                continue
            for q in p["qas"]:
                counts = Counter(a["text"].strip() for a in q["answers"])
                if q.get("is_impossible") or not counts:
                    continue
                gold = counts.most_common(1)[0][0]
                if not gold or gold not in ctx or len(gold) > 24:
                    continue
                others = [
                    a for a in pool if not any(a in g or g in a for g in counts if g)
                ]
                if len(others) < 4:
                    continue
                # Prefer spans from the same passage, then other spans from the same article.
                rng.shuffle(others)
                others.sort(key=lambda a: (a not in ctx, abs(len(a) - len(gold))))
                choices = others[:4] + [gold]
                rng.shuffle(choices)
                req = qa_request(f"本文：{ctx}\n質問：{q['question']}", choices)
                examples.append((req, choices.index(gold), title, q["id"]))
    rng.shuffle(examples)
    seen = set()
    result = []
    for item in examples:
        if item[0][0] not in seen:
            result.append(item)
            seen.add(item[0][0])
        if len(result) == limit:
            break
    return result


def reading_cache(agent, args):
    valid = articles(args.data, "valid")
    reserved = {a["title"] for a in valid}
    train = [a for a in articles(args.data, "train") if a["title"] not in reserved]
    result = {}
    for split, raw, cap in [("train", train, 768), ("eval", valid, 192)]:
        rows = convert(raw, cap, 8301)
        if len(rows) != cap:
            raise ValueError(
                f"Expected {cap} {split} reading examples, found {len(rows)}"
            )
        fingerprint = hashlib.sha256(
            json.dumps(rows, ensure_ascii=False).encode()
        ).hexdigest()
        args.cache.mkdir(exist_ok=True, parents=True)
        path = args.cache / f"reading-{split}-{fingerprint[:16]}.npz"
        if not path.exists():
            x = extract(agent, rows, args.batch_size)
            temp = path.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez(
                temp,
                x=x.numpy(),
                y=[r[1] for r in rows],
                ids=[r[3] for r in rows],
                group=[r[2] for r in rows],
            )
            temp.replace(path)
        with np.load(path) as z:
            result[split] = {k: z[k] for k in z.files}
    return result


def run(args):
    from laya_ja import load_base

    torch.set_num_threads(args.threads)
    base = load_base(args.model_path, "cpu")
    original = base.model.scorer[-1].weight.detach().flatten().clone()
    reader = reading_cache(base, args)
    del base
    torch.set_num_threads(1)
    agent = load("jcommonsenseqa", model_path=args.model_path, device="cpu")
    prior = agent.model.scorer[-1]

    def old(mode):
        return get_cache(args.previous, "jcommonsenseqa", mode)

    target = combine([old("train"), old("train-extra")])
    tune = old("tune")
    tx, ty = torch.from_numpy(tune["x"]), torch.from_numpy(tune["y"])
    report = {
        "source": "JSQuAD v1.3 short passages with constructed distractors",
        "training_counts": {},
        "selected": {},
        "evaluation": {},
    }
    heads = {"v1": prior}
    # Fixed subset and the same seed, 4096 unique examples in each controlled condition.
    permutation = np.random.default_rng(8301).permutation(len(target["y"]))
    for key, target_n, source_n in [
        ("target_only", 4096, 0),
        ("equal_mix", 3328, 768),
        ("full_mix", 4096, 768),
    ]:
        ids = permutation[:target_n]
        target_data = {k: v[ids] for k, v in target.items()}
        dataset = combine([target_data, reader["train"]]) if source_n else target_data
        x, y = torch.from_numpy(dataset["x"]), torch.from_numpy(dataset["y"])
        # Compare the linear readout first, then the release's residual architecture.
        candidates = [original] + [
            fit_weight(x, y, original, s) for s in (0.001, 0.01, 0.1, 1.0, 10.0)
        ]
        best = min(candidates, key=lambda w: metrics(tx @ w, ty)["nll"])
        _, head, title, _ = select_head("jcommonsenseqa", x, y, tx, ty, best)
        heads[key] = head
        report["training_counts"][key] = [target_n, source_n]
        report["selected"][key] = title
        if key == "full_mix":
            folder = args.weights / "reading"
            folder.mkdir(exist_ok=True, parents=True)
            params = {
                "base": BASE,
                "revision": REVISION,
                "task": "jcommonsenseqa",
                "weight": best.reshape(1, -1).tolist(),
            }
            (folder / "jcommonsenseqa.json").write_text(json.dumps(params) + "\n")
            save_selected(folder, "jcommonsenseqa", params, head, title)
    for name, d in [
        ("jcommonsenseqa", combine([old("eval"), old("eval-extra")])),
        ("jsquad_constructed", reader["eval"]),
    ]:
        x, y = torch.from_numpy(d["x"]), torch.from_numpy(d["y"])
        scores = {k: readout(m, x) for k, m in heads.items()}
        report["evaluation"][name] = {k: metrics(v, y) for k, v in scores.items()}
        diff = (scores["full_mix"].argmax(-1) == y).int() - (
            scores["v1"].argmax(-1) == y
        ).int()
        report["evaluation"][name]["paired_difference_95pct"] = bootstrap_difference(
            diff.numpy(), d["group"]
        )
    if report["evaluation"]["jcommonsenseqa"]["v1"]["correct"] != 612:
        raise ValueError("Could not reproduce the released QA score.")
    args.results.mkdir(exist_ok=True, parents=True)
    (args.results / "reading-transfer.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(report["evaluation"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in (
        ("data", ".cache/data"),
        ("previous", ".cache/features"),
        ("cache", ".cache/multi"),
        ("results", "results"),
        ("weights", "weights"),
    ):
        p.add_argument("--" + k, type=Path, default=Path(v))
    p.add_argument("--model-path")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=12)
    run(p.parse_args())
