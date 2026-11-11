"""Train a rank-12 final-layer update on JCoLA; use only training-side labels for selection."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from context_experiment import samples
from grammar_lora import LastLayerGrammar, layers, tokens
from laya_ja import load_base
from scoring import diagnostic


def cache(agent, rows, name, a):
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()[
        :16
    ]
    file = a.cache / (name + "-" + digest + ".npz")
    if not file.exists():
        ts = [tokens(agent, r[0][0]) for r in rows]
        off = np.cumsum([0] + [len(t) - 2 for t in ts])
        base = np.empty((off[-1], 768), np.float16)
        mlp = np.empty((off[-1], 1152), np.float16)
        order = sorted(range(len(rows)), key=lambda i: len(ts[i]))
        for st in range(0, len(rows), a.batch):
            idx = order[st : st + a.batch]
            for i, (h, x) in zip(idx, layers(agent, [ts[k] for k in idx])):
                sl = slice(off[i], off[i + 1])
                base[sl] = h.numpy()
                mlp[sl] = x.numpy()
        tmp = file.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(tmp, base=base, mlp=mlp, off=off)
        tmp.replace(file)
    with np.load(file) as f:
        base, mlp = torch.from_numpy(f["base"]), torch.from_numpy(f["mlp"])
        off = f["off"]
    return [
        (base[lo:hi], mlp[lo:hi], row[1])
        for lo, hi, row in zip(off[:-1], off[1:], rows)
    ]


def collate(rows):
    base, mlp, y = zip(*rows)
    b = torch.nn.utils.rnn.pad_sequence(base, batch_first=True)
    u = torch.nn.utils.rnn.pad_sequence(mlp, batch_first=True)
    mask = (
        torch.arange(b.shape[1])[None, :]
        < torch.tensor([len(t) for t in base])[:, None]
    )
    return b, u, mask, torch.tensor(y)


@torch.no_grad()
def eval_logits(m, d, norm):
    return torch.cat(
        [
            m(b, u, mask, norm)
            for b, u, mask, y in DataLoader(d, batch_size=48, collate_fn=collate)
        ]
    )


def threshold_stats(logits, data, t):
    y = torch.tensor([r[2] for r in data])
    delta = torch.logit(torch.tensor(t))
    z = logits.clone()
    z[:, 1] -= delta
    return diagnostic("jcola", z, y)


def run(a):
    torch.set_num_threads(a.threads)
    agent = load_base(a.model_path, "cpu")
    norm = agent.model.encoder.final_norm.eval().requires_grad_(False)
    a.cache.mkdir(parents=True, exist_ok=True)
    a.weights.mkdir(parents=True, exist_ok=True)
    a.results.mkdir(parents=True, exist_ok=True)
    rows = samples("jcola", a)
    # Use a larger group-disjoint selection set, drawn before fitting this adapter.
    extra = rows["train"][-320:]
    groups = {r[2] for r in extra}
    rows["train"] = [r for r in rows["train"] if r[2] not in groups]
    rows["tune"] = rows["tune"] + extra
    train = cache(agent, rows["train"], "train", a)
    dev = cache(agent, rows["tune"], "tune", a)
    torch.manual_seed(9504)
    torch.set_num_threads(2)
    m = LastLayerGrammar()
    op = torch.optim.AdamW(m.parameters(), lr=0.0005, weight_decay=0.03)
    w = torch.tensor([r[2] for r in train]).bincount().float().rsqrt()
    path = a.weights / "grammar-lora.safetensors"
    best = (-2, -1e8)
    history = []
    for ep in range(10):
        m.train()
        for b, u, mask, y in DataLoader(
            train, batch_size=48, shuffle=True, collate_fn=collate
        ):
            op.zero_grad()
            loss = torch.nn.functional.cross_entropy(m(b, u, mask, norm), y, weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2)
            op.step()
        m.eval()
        z = eval_logits(m, dev, norm)
        chosen = None
        for t in np.arange(0.15, 0.901, 0.025):
            s = threshold_stats(z, dev, float(t))
            if chosen is None or (s["mcc"], -s["nll"]) > chosen[0]:
                chosen = ((s["mcc"], -s["nll"]), float(t), s)
        history.append({"epoch": ep + 1, "threshold": chosen[1], **chosen[2]})
        if chosen[0] > best:
            best = chosen[0]
            cut = chosen[1]
            save_file({k: v.half() for k, v in m.state_dict().items()}, str(path))
        print("grammar dev", ep + 1, chosen[0][0], flush=True)
    (a.weights / "grammar-threshold.json").write_text(
        json.dumps({"threshold": cut}) + "\n"
    )
    m.load_state_dict(load_file(str(path)))
    m.eval()
    torch.set_num_threads(a.threads)
    report = {
        "counts": {k: len(v) for k, v in rows.items()},
        "history": history,
        "threshold": cut,
        "evaluation": {},
    }
    for k, rs in rows.items():
        if k in ("train", "tune"):
            continue
        data = cache(agent, rs, k, a)
        report["evaluation"][k] = threshold_stats(eval_logits(m, data, norm), data, cut)
    (a.results / "grammar-lora.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["evaluation"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in [
        ("data", ".cache/data"),
        ("cache", ".cache/grammar-lora"),
        ("weights", "weights/context"),
        ("results", "results"),
    ]:
        p.add_argument("--" + k, type=Path, default=Path(v))
    p.add_argument("--model-path")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    run(p.parse_args())
