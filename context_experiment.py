"""Retrain Japanese decision modules using the input segments of the frozen Laya encoder."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, TensorDataset

from context_tasks import Readout, batch, item
from corpus_data import pair_key, splits
from corpus_experiment import losses
from experiment import partition
from laya_ja import REVISION, load_base, nli_request
from scoring import diagnostic
from typed import SOURCES, load_split

TASKS = ("jsts", "jcola", "jcommonsenseqa", "nli")


def samples(task, args):
    if task in ("jsts", "jcola"):
        names = list(SOURCES[task])
        train, *tests = [load_split(task, n, args.data) for n in names]
        random.Random(42).shuffle(train)
        groups = {r[2] for v in tests for r in v}
        train = [r for r in train if r[2] not in groups]
        tune = train[:192]
        tg = {r[2] for r in tune}
        train = [r for r in train[192:] if r[2] not in tg]
        res = dict(zip(names[1:], tests))
        old = 2048
    else:
        name = "jnli" if task == "nli" else task
        data = partition(name, args.data, 42, 10**8, 192, 10**8)
        train, tune = data["train"], data["tune"]
        res = {name: data["eval"]}
        old = 4096
    used = {r[2] for r in train[:old]}
    extra = [r for r in train[old:] if r[2] not in used][:256]
    exg = {r[2] for r in extra}
    assert len(extra) > 0
    res["reserved"] = extra
    train = [r for r in train if r[2] not in exg][:6000]
    if task == "nli":
        multi = splits(args.corpora, args.data)
        challenge = json.loads(Path(__file__).with_name("challenge.json").read_text())
        res.update({s: multi[s + "-eval"] for s in ("jsick", "jamp", "janli", "jsem")})
        res["challenge"] = [
            (nli_request(p, h), y, i, str(i))
            for i, (_, p, h, y) in enumerate(challenge)
        ]
        block = {
            pair_key(r[0][0])
            for v in [
                *res.values(),
                tune,
                *[multi[s + "-tune"] for s in ("jsick", "jamp", "janli")],
            ]
            for r in v
        }
        train = [r for r in train if pair_key(r[0][0]) not in block]
        for s in ("jsick", "jamp", "janli"):
            train.extend(
                r for r in multi[s + "-train"] if pair_key(r[0][0]) not in block
            )
            tune = tune + multi[s + "-tune"]
    return {"train": train, "tune": tune, **res}


def parse(task, row):
    state, _, options = row[0]
    if task == "jcola":
        return state, []
    if task == "jcommonsenseqa":
        return state, options
    a, b = state.split("\n仮説：" if task == "nli" else "\n文B：")
    return a.split("：", 1)[1], [b]


def dataset(agent, task, name, rows, a):
    digest = hashlib.sha256(
        json.dumps([REVISION, task, rows], ensure_ascii=False).encode()
    ).hexdigest()[:16]
    file = a.cache / (task + "-" + name + "-" + digest + ".npz")
    if not file.exists():
        if agent is None:
            raise FileNotFoundError(file)
        items = [item(agent.tok, task, *parse(task, r)) for r in rows]
        order = sorted(range(len(rows)), key=lambda i: len(items[i][0]))
        for i in range(0, len(rows), a.batch):
            ids = order[i : i + a.batch]
            z = batch(agent, [items[k] for k in ids]).numpy()
            if i == 0:
                store = np.empty((len(rows), *z.shape[1:]), dtype=z.dtype)
            store[ids] = z
        tmp = file.with_suffix(".tmp.npz")
        np.savez(tmp, x=store)
        tmp.replace(file)
    with np.load(file) as f:
        x = torch.from_numpy(f["x"])
    return TensorDataset(x, torch.tensor([r[1] for r in rows]))


def stats(task, z, y):
    if task == "jsts":
        y = y.double()
        p = z.flatten().double().clamp(0, 5)
        return {
            "n": len(y),
            "pearson": float(torch.corrcoef(torch.stack([p, y]))[1, 0]),
            "mae": float((p - y).abs().mean()),
        }
    if task == "jcola":
        return diagnostic(task, z, y)
    b = z.argmax(1)
    if task == "nli":
        binary = y == 3
        if binary.any() and not (y == 1).any() and not (y == 2).any():
            b = (z[:, 0] < z[:, 1:].logsumexp(1)).long()
            y = (y != 0).long()
        elif binary.any():
            b = torch.where(
                binary, torch.where(z[:, 0] < z[:, 1:].logsumexp(1), 3, 0), b
            )
    return {
        "n": len(y),
        "correct": int((b == y).sum()),
        "accuracy": float((b == y).float().mean()),
    }


@torch.no_grad()
def evaluate(task, model, d, source_blocks=False):
    z = torch.cat([model(x) for x, y in DataLoader(d, batch_size=64)])
    if source_blocks:
        # JaNLI uses both E and non-E probabilities, including on its positive examples.
        parts = [
            stats(task, z[i : i + 192], d.tensors[1][i : i + 192])
            for i in range(0, len(z), 192)
        ]
        return {"n": len(z), "accuracy": sum(v["correct"] for v in parts) / len(z)}
    return stats(task, z, d.tensors[1])


def run(a):
    a.cache.mkdir(exist_ok=True, parents=True)
    a.weights.mkdir(exist_ok=True, parents=True)
    a.results.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(a.threads)
    agent = None if a.cached_only else load_base(a.model_path, "cpu")
    for task in a.tasks:
        rs = samples(task, a)
        print(task, {k: len(v) for k, v in rs.items()}, flush=True)
        torch.set_num_threads(a.threads)
        d = {n: dataset(agent, task, n, rs[n], a) for n in ("train", "tune")}
        torch.manual_seed(941)
        torch.set_num_threads(2)
        m = Readout(task)
        opt = torch.optim.AdamW(m.parameters(), lr=0.00065, weight_decay=0.02)
        key = {"jsts": "pearson", "jcola": "mcc"}.get(task, "accuracy")
        best = -2
        history = []
        path = a.weights / (task + ".safetensors")
        weights = None
        if task == "jcola":
            freq = d["train"].tensors[1].bincount().float()
            weights = (freq.sum() / freq).sqrt()
        for epoch in range(12):
            m.train()
            for x, y in DataLoader(d["train"], shuffle=True, batch_size=80):
                opt.zero_grad()
                z = m(x)
                if task == "nli":
                    loss = losses(z, y).mean()
                elif task == "jsts":
                    loss = torch.nn.functional.smooth_l1_loss(
                        z.flatten(), y.float(), beta=0.5
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(z, y, weight=weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 2)
                opt.step()
            m.eval()
            stat = evaluate(task, m, d["tune"], source_blocks=task == "nli")
            history.append(stat)
            if stat[key] > best:
                best = stat[key]
                save_file({k: v.half() for k, v in m.state_dict().items()}, str(path))
        m.load_state_dict(load_file(str(path)))
        m.eval()
        torch.set_num_threads(a.threads)
        report = {
            "counts": {k: len(v) for k, v in rs.items()},
            "tune": history,
            "evaluation": {},
        }
        for name, rows in rs.items():
            if name in d:
                continue
            test = dataset(agent, task, name, rows, a)
            report["evaluation"][name] = evaluate(task, m, test)
        (a.results / ("context-" + task + ".json")).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(task, report["evaluation"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in [
        ("data", ".cache/data"),
        ("cache", ".cache/context"),
        ("corpora", ".cache/corpora"),
        ("weights", "weights/context"),
        ("results", "results"),
    ]:
        p.add_argument("--" + k, type=Path, default=Path(v))
    p.add_argument("--model-path")
    p.add_argument("--cached-only", action="store_true")
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--batch", type=int, default=24)
    run(p.parse_args())
