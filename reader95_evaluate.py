"""Choose span-reader mixture weights on dev only and score the fixed article splits."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from grounded_data import fields
from grounded_experiment import assess, features, scores
from grounded_qa import load_head
from laya_ja import load_base
from reader95 import REV, AuxiliaryQA
from reader95_data import extended

SPLITS = (
    "original",
    "fresh",
    "new",
    "unseen_titles",
    "unseen_passages",
    "fresh_without_question",
)


def aux_scores(a, model, key, rows):
    digest = hashlib.sha256(
        json.dumps([REV, rows], ensure_ascii=False).encode()
    ).hexdigest()[:16]
    file = a.cache / ("aux-" + key + "-" + digest + ".npy")
    if not file.exists():
        if model is None:
            raise FileNotFoundError(file)
        z = model.score_many([fields(r) for r in rows], a.batch).numpy()
        np.save(file, z)
    return torch.from_numpy(np.load(file))


def mixture(logits, c):
    return sum(x * v for x, v in zip(logits, c))


def select(logits, y, native):
    grid = [
        (i / 10, (10 - i - k) / 10, k / 10)
        for k in (range(1) if native else range(8))
        for i in range(11 - k)
    ]

    def objective(c):
        z = mixture(logits, c)
        return int((z.argmax(-1) == y).sum()), -float(
            torch.nn.functional.cross_entropy(z, y)
        )

    return list(max(grid, key=objective))


def run(a):
    torch.set_num_threads(a.threads)
    r = extended(a.data)
    a.cache.mkdir(parents=True, exist_ok=True)
    r["fresh_without_question"] = [
        (
            (row[0][0].split("\n質問：", 1)[0] + "\n質問：？", row[0][1], row[0][2]),
            *row[1:],
        )
        for row in r["fresh"]
    ]
    if a.mode == "aux-cache":
        aux = AuxiliaryQA(a.aux_path)
        for key in ("dev", *SPLITS):
            aux_scores(a, aux, key, r[key])
            print("encoded", key, flush=True)
        return
    aux = None
    for key in ("dev", *SPLITS):
        try:
            aux_scores(a, None, key, r[key])
        except FileNotFoundError:
            if aux is None:
                aux = AuxiliaryQA(a.aux_path)
            aux_scores(a, aux, key, r[key])
    del aux
    base = load_base(a.model_path, "cpu")
    heads = [
        load_head(a.weights / f)
        for f in ("grounded-qa.safetensors", "extended-span.safetensors")
    ]
    cfg = {}
    cutoff = None
    report = {}
    for key in ("dev", *SPLITS):
        rows = r[key]
        d = features(base, a.cache, key, rows, a.batch)
        y = d.tensors[-1]
        az = aux_scores(a, None, key, rows)
        zz = [scores(h, d) for h in heads] + [az]
        if key == "dev":
            cfg = {
                name: select(zz, y, native)
                for name, native in [("native", True), ("combined", False)]
            }
            (a.weights / "reading-mixture.json").write_text(
                json.dumps(cfg, indent=2) + "\n"
            )
        groups = np.array([v[2] for v in rows])
        m = d.tensors[2].any(-1)
        if not torch.all(m.gather(1, y[:, None])):
            raise ValueError("Gold missing from span mask")
        rr = {}
        for name, c in {
            "previous": [1, 0, 0],
            "additional": [0, 1, 0],
            "auxiliary": [0, 0, 1],
            **cfg,
        }.items():
            p = mixture(zz, c).masked_fill(~m, -10000).argmax(-1)
            ok = (p == y).numpy()
            hard = m.all(-1).numpy()
            rr[name] = {
                "all": assess(ok, groups),
                "all_five": assess(ok[hard], groups[hard]) if hard.any() else None,
            }
        # Optional confidence analysis; the primary score above always includes every question.
        logits = mixture(zz, cfg["combined"]).masked_fill(~m, -10000)
        confidence, prediction = logits.softmax(1).max(1)
        ok = (prediction == y).numpy()
        if key == "dev":
            for candidate in np.arange(0.50, 1.0, 0.01):
                keep = (confidence >= candidate).numpy()
                if (
                    keep.sum() >= 50
                    and len(set(groups[keep])) >= 8
                    and assess(ok[keep], groups[keep])["article_bootstrap_95"][0]
                    >= 0.95
                ):
                    cutoff = float(candidate)
                    break
        if cutoff is not None:
            keep = (confidence >= cutoff).numpy()
            rr["combined"]["selective"] = {
                "threshold": cutoff,
                "coverage": float(keep.mean()),
                "accepted": int(keep.sum()),
                "performance": assess(ok[keep], groups[keep]) if keep.any() else None,
            }
        report[key] = rr
        print(
            key,
            [(k, v["all"]["correct"], v["all"]["n"]) for k, v in rr.items()],
            flush=True,
        )
    (a.results / "reading-mixture.json").write_text(
        json.dumps({"weights": cfg, "evaluation": report}, indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in [
        ("data", ".cache/data"),
        ("cache", ".cache/grounded"),
        ("weights", "weights"),
        ("results", "results"),
    ]:
        p.add_argument("--" + k, type=Path, default=Path(v))
    p.add_argument("--model-path")
    p.add_argument("--aux-path")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--mode", choices=["aux-cache", "evaluate"], default="evaluate")
    run(p.parse_args())
