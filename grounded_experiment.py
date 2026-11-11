"""Train a Laya answer-span head. Select on development articles, then open the fixed tests."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from experiment import bootstrap_difference, extract
from grounded_data import fields, partitions
from grounded_qa import SpanHead, encode_input, extract_batch, load_head, save_head
from laya_ja import REVISION, load, load_base
from torch.utils.data import DataLoader, TensorDataset


def features(agent, folder, name, rows, batch):
    key = hashlib.sha256(
        json.dumps([REVISION, rows], ensure_ascii=False).encode()
    ).hexdigest()[:16]
    path = folder / (name + "-" + key + ".npz")
    if not path.exists():
        items = [encode_input(agent.tok, *fields(r)) for r in rows]
        order = sorted(range(len(rows)), key=lambda i: len(items[i][0]))
        store = {}
        for pos in range(0, len(rows), batch):
            idx = order[pos : pos + batch]
            zs = extract_batch(
                agent.model.encoder, agent.tok.pad_token_id, [items[i] for i in idx]
            )
            for k, t in zip(("z", "f", "m"), zs):
                if k not in store:
                    store[k] = np.empty(
                        (len(rows), *t.shape[1:]), dtype=t.numpy().dtype
                    )
                store[k][idx] = t.numpy()
            if pos % (batch * 20) == 0:
                print(name, pos, len(rows), flush=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **store)
        tmp.replace(path)
    with np.load(path) as f:
        return TensorDataset(
            *(torch.from_numpy(f[k]) for k in ("z", "f", "m")),
            torch.tensor([r[1] for r in rows]),
        )


@torch.no_grad()
def scores(model, data):
    return torch.cat([model(z, f, m) for z, f, m, _ in DataLoader(data, batch_size=64)])


def fit(model, train, dev):
    gen = torch.Generator().manual_seed(20261111)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    best = (-1, -1e10)
    from copy import deepcopy

    history = []
    for epoch in range(1, 16):
        model.train()
        for z, f, m, y in DataLoader(train, batch_size=48, shuffle=True, generator=gen):
            opt.zero_grad()
            s = model(z, f, m)
            torch.nn.functional.cross_entropy(s, y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            opt.step()
        model.eval()
        logits = scores(model, dev)
        acc = (logits.argmax(-1) == dev.tensors[-1]).float().mean().item()
        nll = torch.nn.functional.cross_entropy(logits, dev.tensors[-1]).item()
        history.append({"epoch": epoch, "accuracy": acc, "nll": nll})
        print("development", epoch, acc, nll, flush=True)
        if (acc, -nll) > best:
            best = (acc, -nll)
            chosen = deepcopy(model)
    return chosen, history


def assess(correct, groups):
    return {
        "correct": int(correct.sum()),
        "n": len(correct),
        "accuracy": float(correct.mean()),
        "article_bootstrap_95": bootstrap_difference(
            correct.astype(float), np.array(groups)
        ),
    }


def run(a):
    torch.set_num_threads(a.threads)
    torch.manual_seed(20261111)
    rows = partitions(a.data)
    a.cache.mkdir(parents=True, exist_ok=True)
    print({k: (len(v), len({r[2] for r in v})) for k, v in rows.items()}, flush=True)
    if a.stage in ("fit", "all"):
        agent = load_base(a.model_path, "cpu")
        train = features(agent, a.cache, "fit", rows["fit"], a.batch)
        dev = features(agent, a.cache, "dev", rows["dev"], a.batch)
        torch.set_num_threads(min(3, a.threads))
        model, history = fit(SpanHead(), train, dev)
        a.weights.mkdir(parents=True, exist_ok=True)
        save_head(model, a.weights / "grounded-qa.safetensors")
        (a.cache / "history.json").write_text(json.dumps(history))
        del train, dev, agent
    if a.stage == "fit":
        return

    # The frozen encoder is identical in both readers. Reuse the v2 readout for paired scores.
    agent = load(
        "jcommonsenseqa",
        model_path=a.model_path,
        device="cpu",
        weights_dir=Path(__file__).with_name("weights") / "reading",
    )
    model = load_head(a.weights / "grounded-qa.safetensors")
    torch.set_num_threads(a.threads)
    report = {
        "history": json.loads((a.cache / "history.json").read_text()),
        "split_info": {
            k: {
                "n": len(v),
                "articles": len({r[2] for r in v}),
                "ids_sha256": hashlib.sha256(
                    "\n".join(str(r[3]) for r in v).encode()
                ).hexdigest(),
            }
            for k, v in rows.items()
        },
        "evaluation": {},
    }
    for name in ("original", "fresh", "new"):
        rs = rows[name]
        data = features(agent, a.cache, name, rs, a.batch)
        logits = scores(model, data)
        ok = (logits.argmax(-1) == data.tensors[-1]).numpy()
        count = data.tensors[2].any(-1).sum(-1).numpy()
        groups = np.array([r[2] for r in rs])
        with torch.no_grad():
            previous = (
                agent.model.scorer[-1](extract(agent, rs, a.batch))
                .squeeze(-1)
                .argmax(-1)
            )
        prior_ok = (previous == data.tensors[-1]).numpy()
        delta = ok.astype(int) - prior_ok.astype(int)
        report["evaluation"][name] = {
            "previous": assess(prior_ok, groups),
            "gains": int((delta == 1).sum()),
            "losses": int((delta == -1).sum()),
            "difference_article_bootstrap_95": bootstrap_difference(delta, groups),
            "total": assess(ok, groups),
            "by_choices_present": {
                str(k): assess(ok[count == k], groups[count == k])
                for k in sorted(set(count))
            },
            "uniform_among_present": float((1 / count).mean()),
            "predictions": [
                {
                    "id": r[3],
                    "gold": int(r[1]),
                    "pred": int(p),
                    "previous_pred": int(prev),
                    "present": int(c),
                    "article": r[2],
                }
                for r, p, prev, c in zip(rs, logits.argmax(-1), previous, count)
            ],
        }
        if name == "original" and int(prior_ok.sum()) != 61:
            raise ValueError("Previous 61/192 reading result was not reproduced")
        if name == "fresh":
            no_question = [
                (
                    (r[0][0].split("\n質問：", 1)[0] + "\n質問：？", r[0][1], r[0][2]),
                    *r[1:],
                )
                for r in rs
            ]
            blank = features(
                agent, a.cache, "fresh-without-question", no_question, a.batch
            )
            blank_ok = (scores(model, blank).argmax(-1) == blank.tensors[-1]).numpy()
            report["evaluation"][name]["without_question"] = assess(blank_ok, groups)
        print(name, report["evaluation"][name]["total"], flush=True)
    a.results.mkdir(parents=True, exist_ok=True)
    (a.results / "grounded-qa.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in [
        ("data", ".cache/data"),
        ("cache", ".cache/grounded"),
        ("weights", "weights"),
        ("results", "results"),
    ]:
        p.add_argument("--" + k, type=Path, default=Path(v))
    p.add_argument("--model-path")
    p.add_argument("--stage", choices=["all", "fit", "evaluate"], default="all")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--threads", type=int, default=6)
    run(p.parse_args())
