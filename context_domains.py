"""Compare shared inference weights with domain-specific residuals using source dev labels."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from context_experiment import samples, stats
from context_tasks import DomainCorrection, Readout
from corpus_data import splits
from corpus_experiment import losses
from laya_ja import REVISION


def cached(a, key, rows):
    digest = hashlib.sha256(
        json.dumps([REVISION, "nli", rows], ensure_ascii=False).encode()
    ).hexdigest()[:16]
    with np.load(a.cache / ("nli-" + key + "-" + digest + ".npz")) as f:
        return torch.from_numpy(f["x"])


def run(a):
    torch.set_num_threads(2)
    r = samples("nli", a)
    src = splits(a.corpora, a.data)
    base = Readout("nli").eval()
    base.load_state_dict(load_file(str(a.weights / "nli.safetensors")))

    def reps(key):
        x = cached(a, key, r[key])
        with torch.no_grad():
            z = base.representation(x)
            prior = base(x)
        y = torch.tensor([v[1] for v in r[key]])
        return z, prior, y

    tr, dev = reps("train"), reps("tune")
    report = {}
    for name in ("jnli", "jsick", "jamp", "janli"):
        trrows = (
            {(row[0][0], row[3]) for row in src[name + "-train"]}
            if name != "jnli"
            else None
        )
        other = {
            (row[0][0], row[3])
            for s in ("jsick", "jamp", "janli")
            for row in src[s + "-train"]
        }
        ids = torch.tensor(
            [
                (str(row[0][0]), row[3]) in trrows
                if trrows is not None
                else (row[0][0], row[3]) not in other
                for row in r["train"]
            ]
        )
        start = ("jnli", "jsick", "jamp", "janli").index(name) * 192
        X, P, y = (v[ids] for v in tr)
        dx, dp, dy = (v[start : start + 192] for v in dev)
        torch.manual_seed(942)
        m = DomainCorrection().eval()
        op = torch.optim.AdamW(m.parameters(), lr=0.0006, weight_decay=0.03)
        path = a.weights / ("domain-" + name + ".safetensors")

        def quality(s, labels=dy):
            return (
                stats("nli", s, labels)["accuracy"],
                -float(losses(s, labels).mean()),
            )

        best = quality(dp)
        chosen = 0
        save_file({k: v.half() for k, v in m.state_dict().items()}, str(path))
        for ep in range(14):
            m.train()
            for idx in torch.randperm(len(y)).split(64):
                op.zero_grad()
                losses(P[idx] + m(X[idx]), y[idx]).mean().backward()
                op.step()
            m.eval()
            with torch.no_grad():
                value = quality(dp + m(dx))
            if value > best:
                best = value
                chosen = ep + 1
                save_file({k: v.half() for k, v in m.state_dict().items()}, str(path))
        m.load_state_dict(load_file(str(path)))
        m.eval()
        v, p, y = reps(name)
        with torch.no_grad():
            stat = stats("nli", p + m(v), y)
        report[name] = {
            "training": int(ids.sum()),
            "epoch": chosen,
            "dev_accuracy": best[0],
            "evaluation": stat,
        }
        if name == "jnli":
            v, p, y = reps("reserved")
            with torch.no_grad():
                report[name]["reserved"] = stats("nli", p + m(v), y)
        print(name, report[name], flush=True)
    (a.results / "context-domains.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in [
        ("data", ".cache/data"),
        ("corpora", ".cache/corpora"),
        ("cache", ".cache/context"),
        ("weights", "weights/context"),
        ("results", "results"),
    ]:
        p.add_argument("--" + k, type=Path, default=Path(v))
    run(p.parse_args())
