"""Compare a linear scorer and small nonlinear residuals using training-side labels only."""

import argparse
import copy
import json
from pathlib import Path

import torch
from experiment import bootstrap_difference, combine, get_cache, metrics
from laya_ja import TASKS
from residual import ResidualScorer
from safetensors.torch import save_file


def linear(weight):
    layer = torch.nn.Linear(len(weight), 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer


def readout(module, x):
    with torch.no_grad():
        return module(x).squeeze(-1)


def select_head(task, x, y, tx, ty, w):
    simple = linear(w).eval()
    chosen, title = simple, "linear"

    def index(scores):
        if task in ("jsts", "jcola"):
            from typed import diagnostic

            return diagnostic(task, scores, ty)
        return metrics(scores, ty)

    def objective(stats):
        if task == "jcola":
            return (-stats["mcc"], stats["nll"])
        if task == "jsts":
            return (stats["mae"],)
        return (stats["nll"],)

    tune = {"linear": index(readout(simple, tx))}
    best = objective(tune["linear"])
    class_weights = None
    targets = y
    if task == "jcola":
        class_weights = len(y) / (2 * torch.bincount(y).float())
    if task == "jsts":
        from typed import interpolate

        targets = interpolate(y)

    for width in (16, 48):
        torch.manual_seed(42)
        head = ResidualScorer(linear(w), width)
        with torch.no_grad():
            head.center.copy_(x.mean((0, 1)))
            head.scale.copy_(x.flatten(0, 1).std(0).clamp_min(0.05))
        optimizer = torch.optim.AdamW(
            head.extra.parameters(), lr=0.001, weight_decay=0.01
        )
        gen = torch.Generator().manual_seed(42)
        for epoch in range(1, 25):
            head.train()
            for batch in torch.randperm(len(x), generator=gen).split(128):
                scores = head(x[batch]).squeeze(-1)
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(
                    scores, targets[batch], weight=class_weights
                )
                loss.backward()
                optimizer.step()
            if epoch in (4, 12, 24):
                head.eval()
                name = f"width{width}-epoch{epoch}"
                tune[name] = index(readout(head, tx))
                if objective(tune[name]) < best:
                    best = objective(tune[name])
                    chosen, title = copy.deepcopy(head), name
    return simple, chosen, title, tune


def save_selected(folder, task, weights, chosen, title):
    outfile = folder / f"{task}.residual.safetensors"
    if title != "linear":
        save_file(
            {
                k: v.detach().contiguous()
                for k, v in chosen.state_dict().items()
                if not k.startswith("linear.")
            },
            str(outfile),
            metadata={
                "base_revision": weights["revision"],
                "task": task,
                "width": str(chosen.extra[0].out_features),
            },
        )
    elif outfile.exists():
        outfile.unlink()


def compare(task, args):
    def read(mode):
        return get_cache(args.cache, task, mode)

    training = (
        combine([read("train"), read("train-extra")]) if args.extra else read("train")
    )
    x, y = torch.from_numpy(training["x"]), torch.from_numpy(training["y"])
    t = read("tune")
    tx, ty = torch.from_numpy(t["x"]), torch.from_numpy(t["y"])
    weights = json.loads((args.weights / f"{task}.json").read_text())
    w = torch.tensor(weights["weight"]).flatten()
    simple, chosen, title, tune = select_head(task, x, y, tx, ty, w)
    # Official validation and challenge labels are first used after the head is selected.
    if args.local_base:
        from safetensors import safe_open

        with safe_open(str(args.local_base / "model.safetensors"), framework="pt") as f:
            original = linear(f.get_tensor("scorer.3.weight").float().flatten()).eval()
    else:
        from laya_ja import load_base

        agent = load_base(device="cpu")
        original = linear(agent.model.scorer[-1].weight.detach().flatten().cpu()).eval()
        del agent
    save_selected(args.weights, task, weights, chosen, title)

    def evaluate(d):
        vx, vy = torch.from_numpy(d["x"]), torch.from_numpy(d["y"])
        scores = {
            k: readout(m, vx)
            for k, m in [("base", original), ("linear", simple), ("selected", chosen)]
        }
        delta = (scores["selected"].argmax(-1) == vy).int() - (
            scores["base"].argmax(-1) == vy
        ).int()
        return {
            **{k: metrics(v, vy) for k, v in scores.items()},
            "gained": int((delta == 1).sum()),
            "lost": int((delta == -1).sum()),
            "difference_95pct": bootstrap_difference(delta.numpy(), d["group"]),
        }

    valid = combine([read("eval"), read("eval-extra")])
    report = {
        "task": task,
        "selected": title,
        "tune_candidates": tune,
        "official_validation": evaluate(valid),
        "reversed": evaluate(read("eval-reversed")),
    }
    vx = torch.from_numpy(valid["x"])
    vy = torch.from_numpy(valid["y"])
    if task == "jnli":
        report["recall"] = {}
        for i, k in enumerate(("entailment", "contradiction", "neutral")):
            subset = vy == i
            report["recall"][k] = {
                n: float((readout(m, vx[subset]).argmax(-1) == i).float().mean())
                for n, m in [("base", original), ("selected", chosen)]
            }
        challenge = read("challenge")
        report["challenge"] = evaluate(challenge)
        probes = torch.from_numpy(challenge["x"])
        report["challenge_items"] = [
            {
                "id": int(idx),
                "category": group,
                "gold": int(gold),
                "base": int(a),
                "selected": int(b),
            }
            for idx, group, gold, a, b in zip(
                challenge["ids"],
                challenge["group"],
                challenge["y"],
                readout(original, probes).argmax(-1),
                readout(chosen, probes).argmax(-1),
            )
        ]
    first, second = [torch.from_numpy(read(k)["x"]) for k in ("eval", "eval-reversed")]
    report["order_agreement"] = {
        n: float(
            (readout(m, first).argmax(-1) == readout(m, second).argmax(-1))
            .float()
            .mean()
        )
        for n, m in [("base", original), ("selected", chosen)]
    }
    args.results.mkdir(exist_ok=True, parents=True)
    (args.results / f"{task}-final.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(task, title, report["official_validation"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path(".cache/features"))
    p.add_argument("--weights", type=Path, default=Path("weights"))
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--local-base", type=Path)
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--extra", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    for task in [args.task] if args.task else TASKS:
        compare(task, args)
