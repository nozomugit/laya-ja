"""Test whether controlled Japanese logic examples transfer to held-out semantic benchmarks."""

import argparse
import json
import random
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from context_domains import cached
from context_experiment import dataset, samples, stats
from context_tasks import DomainCorrection, Readout
from corpus_experiment import losses
from laya_ja import load_base, nli_request

NAMES = [
    "山田",
    "川上",
    "清水",
    "坂下",
    "大谷",
    "林",
    "中山",
    "宮本",
    "高橋",
    "金子",
    "大島",
    "西川",
    "田口",
    "佐藤",
    "中田",
    "柴田",
    "水野",
    "井上",
    "赤井",
    "石原",
]
OBJECTS = [
    "封筒",
    "箱",
    "名札",
    "切符",
    "書類",
    "図録",
    "袋",
    "地図",
    "鍵",
    "台帳",
    "券",
    "本",
    "写真",
    "手紙",
    "小包",
]
PLACES = [
    "倉庫",
    "玄関",
    "書庫",
    "受付",
    "資料室",
    "教室",
    "事務室",
    "会議室",
    "展示室",
    "ロビー",
]


def generate(mode, n, seed):
    rng = random.Random(seed)
    out = []
    seen = set()
    while len(out) < n:
        a, b, c = rng.sample(NAMES, 3)
        x, y = rng.sample(OBJECTS, 2)
        u, v = rng.sample(PLACES, 2)
        if mode == "held":
            patterns = [
                (
                    f"{u}の係は{a}一人に決まった。{b}は{v}で待機している。",
                    f"{a}は{u}の係だ。",
                    f"{b}も{u}の係だ。",
                    f"{c}は{v}にいる。",
                ),
                (
                    f"{x}の提出が済んで初めて、{y}の受け取りが始まった。",
                    f"{y}を受け取り始めた時には、{x}はすでに提出されていた。",
                    f"{y}を受け取り始めたのは、{x}の提出より前だ。",
                    f"{x}を提出したのは{a}だ。",
                ),
                (
                    f"{u}にいた三人は順に調べられた。{x}を持っていた人も、持っていなかった人もいた。",
                    f"{u}の三人のうち、少なくとも一人は{x}を持っていた。",
                    f"{u}にいた三人は全員{x}を持っていた。",
                    f"{u}で{x}を持っていたのは二人だ。",
                ),
            ]
        else:
            patterns = [
                (
                    f"今日{u}には{a}と{b}だけが入った。",
                    f"{a}は今日{u}に入った。",
                    f"{c}は今日{u}に入った。",
                    f"{a}は今日{v}に入った。",
                ),
                (
                    f"{a}は{x}を渡してから{y}を受け取った。",
                    f"{a}は{y}を受け取る前に{x}を渡した。",
                    f"{a}が{x}を渡したのは{y}を受け取った後だ。",
                    f"{a}が{x}を渡した相手は{b}だ。",
                ),
                (
                    f"{u}にいる全員が{x}を持っており、{a}もそこにいる。",
                    f"{a}は{x}を持っている。",
                    f"{a}は{x}を持っていない。",
                    f"{b}は{x}を持っていない。",
                ),
                (
                    f"{u}にいる人のうち、{x}を持つ人は一人もいない。{b}はそこにいる。",
                    f"{b}は{x}を持っていない。",
                    f"{b}は{x}を持っている。",
                    f"{a}は{x}を持っている。",
                ),
                (
                    f"{u}に入るには、必ず{x}が必要だ。{a}は{x}を持たない。",
                    f"{a}は{u}に入れない。",
                    f"{a}は{x}を持たずに{u}に入れる。",
                    f"{b}は{u}に入れる。",
                ),
                (
                    f"{x}を受け取ったのは{b}だけだ。{a}は{b}にそれを渡した。",
                    f"{b}は{a}から{x}を受け取った。",
                    f"{a}もその{x}を受け取った。",
                    f"{a}は{b}から{y}を受け取った。",
                ),
                (
                    f"{a}は{u}にはいたが、{v}にはいなかった。",
                    f"{a}は{u}にいた。",
                    f"{a}は{v}にいた。",
                    f"{b}は{v}にいた。",
                ),
                (
                    f"{a}は{x}も{y}も持っていなかった。",
                    f"{a}は{y}を持っていなかった。",
                    f"{a}は{x}を持っていた。",
                    f"{b}は{y}を持っていた。",
                ),
            ]
        k = rng.randrange(len(patterns))
        prem, *hyps = patterns[k]
        if prem in seen:
            continue
        seen.add(prem)
        for label, h in enumerate(hyps):
            out.append((nli_request(prem, h), label, prem, str(len(out))))
    return out[:n]


def run(a):
    torch.set_num_threads(a.threads)
    base = Readout("nli").eval()
    base.load_state_dict(load_file(str(a.weights / "nli.safetensors")))
    r = samples("nli", a)
    agent = load_base(a.model_path, "cpu")
    train = generate("fit", 1800, 941)
    dev = generate("fit", 450, 942)
    used = {v[0][0].split("\n")[0] for v in train}
    dev = [row for row in dev if row[0][0].split("\n")[0] not in used]
    held = generate("held", 300, 943)

    def prepared(key, rows):
        d = dataset(agent, "nli", key, rows, a)
        with torch.no_grad():
            return base.representation(d.tensors[0]), base(d.tensors[0]), d.tensors[1]

    x, p, y = prepared("logic-train", train)
    vx, vp, vy = prepared("logic-tune", dev)
    tx = cached(a, "train", r["train"])
    ty = torch.tensor([v[1] for v in r["train"]])
    idx = torch.randperm(len(ty), generator=torch.Generator().manual_seed(441))[:1800]
    with torch.no_grad():
        x = torch.cat([x, base.representation(tx[idx])])
        p = torch.cat([p, base(tx[idx])])
        y = torch.cat([y, ty[idx]])
    dx = cached(a, "tune", r["tune"])
    dy = torch.tensor([v[1] for v in r["tune"]])
    with torch.no_grad():
        dp = base(dx)
        dx = base.representation(dx)
    torch.manual_seed(944)
    m = DomainCorrection()
    op = torch.optim.AdamW(m.parameters(), lr=0.0006, weight_decay=0.02)
    path = a.weights / "domain-logic.safetensors"
    best = float("inf")
    history = []
    for ep in range(12):
        m.train()
        for idx in torch.randperm(len(y)).split(64):
            op.zero_grad()
            losses(p[idx] + m(x[idx]), y[idx]).mean().backward()
            op.step()
        m.eval()
        with torch.no_grad():
            objective = (
                0.5
                * (losses(vp + m(vx), vy).mean() + losses(dp + m(dx), dy).mean()).item()
            )
        history.append(objective)
        if objective < best:
            best = objective
            save_file({k: v.half() for k, v in m.state_dict().items()}, str(path))
    m.load_state_dict(load_file(str(path)))
    m.eval()
    report = {
        "synthetic_train": len(train),
        "replay": 1800,
        "synthetic_tune": len(dev),
        "tune_mean_nll": history,
        "evaluation": {},
    }
    for k in (
        "held_templates",
        "jnli",
        "jsick",
        "janli",
        "jamp",
        "jsem",
        "challenge",
        "reserved",
    ):
        if k == "held_templates":
            v, p, y = prepared(k, held)
        else:
            raw = cached(a, k, r[k])
            y = torch.tensor([v[1] for v in r[k]])
            with torch.no_grad():
                v, p = base.representation(raw), base(raw)
        with torch.no_grad():
            report["evaluation"][k] = {
                "context": stats("nli", p, y),
                "logic": stats("nli", p + m(v), y),
            }
    (a.results / "logic-transfer.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["evaluation"], flush=True)


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
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--model-path")
    run(p.parse_args())
