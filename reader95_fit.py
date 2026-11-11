"""Fit an additional native Laya span scorer on 10,596 examples."""

import argparse
import json
from pathlib import Path

import torch

from grounded_experiment import features, fit
from grounded_qa import SpanHead, save_head
from laya_ja import load_base
from reader95_data import extended


def run(a):
    torch.set_num_threads(a.threads)
    rows = extended(a.data)
    print({k: len(v) for k, v in rows.items()}, flush=True)
    agent = load_base(a.model_path, "cpu")
    a.cache.mkdir(parents=True, exist_ok=True)
    parts = [
        (f"additional-{start}", rows["additional"][start : start + 512])
        for start in range(0, len(rows["additional"]), 512)
    ]
    parts.extend((k, rows[k]) for k in ("fit", "dev"))
    for name, examples in parts:
        ds = features(agent, a.cache, name, examples, a.batch)
        del ds
    del agent
    sets = [features(None, a.cache, k, examples, a.batch) for k, examples in parts]
    torch.set_num_threads(2)
    torch.manual_seed(9501)
    model, history = fit(
        SpanHead(), torch.utils.data.ConcatDataset(sets[:-1]), sets[-1]
    )
    a.weights.mkdir(parents=True, exist_ok=True)
    a.results.mkdir(parents=True, exist_ok=True)
    save_head(model, a.weights / "extended-span.safetensors")
    (a.results / "extended-span-training.json").write_text(
        json.dumps(history, indent=2) + "\n"
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
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--model-path")
    run(p.parse_args())
