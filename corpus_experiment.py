"""Controlled continuation of the released Laya-JA NLI adapter on multiple corpora."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from experiment import (
    bootstrap_difference,
    combine,
    get_cache,
    partition,
    permute_training,
)
from laya_ja import LABELS, REVISION
from multicorpus import PairResidual
from residual import ResidualScorer
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

SEEDS = (11, 37, 83)
SOURCES = ("jnli", "jsick", "janli", "jamp")
REGIMES = {
    "jnli_only": {"jnli": 4096},
    "equal_4096": {s: 1024 for s in SOURCES},
    "all": {"jnli": 4096, "jsick": 1536, "janli": 1536, "jamp": 1536},
}


def old_head(folder, task="jnli"):
    weights = json.loads((folder / f"{task}.json").read_text())
    linear = nn.Linear(768, 1, bias=False)
    linear.weight.data.copy_(torch.tensor(weights["weight"]))
    with safe_open(str(folder / f"{task}.residual.safetensors"), framework="pt") as f:
        h = ResidualScorer(linear, int(f.metadata()["width"]))
        state = h.state_dict()
        state.update({name: f.get_tensor(name) for name in list(f.keys())})
        h.load_state_dict(state)
    return h.eval().requires_grad_(False)


def training_features(args):
    rows = partition("jnli", args.data, 42, 4096, 192, 10**9)["train"]
    a = get_cache(args.previous, "jnli", "train")
    b = get_cache(args.previous, "jnli", "train-extra")
    ordered = permute_training(rows[1024:], 43)
    if b["ids"].tolist() != [r[3] for r in ordered] or b["y"].tolist() != [
        r[1] for r in ordered
    ]:
        raise ValueError("JNLI cache does not match the release split and ordering.")
    if a["ids"].tolist() != [r[3] for r in rows[:1024]]:
        raise ValueError("Wrong original JNLI training cache")
    for i, (req, _, _, _) in enumerate(ordered):
        order = [req[2].index(label) for label in LABELS]
        b["x"][i] = b["x"][i, order]
    b["y"] = np.array([r[1] for r in rows[1024:]])
    return combine([a, b])


class Data:
    def __init__(self, d, legacy):
        self.x = torch.from_numpy(d["x"])
        self.y = torch.from_numpy(d["y"]).long()
        self.groups = d["group"]
        self.ids = d["ids"]
        with torch.no_grad():
            self.prior = legacy(self.x).squeeze(-1)


def read(args, name, mode, legacy):
    if name == "jnli":
        if mode == "train":
            d = training_features(args)
        elif mode == "eval":
            d = combine(
                [get_cache(args.previous, name, k) for k in ("eval", "eval-extra")]
            )
        else:
            d = get_cache(args.previous, name, mode)
    elif name == "challenge":
        d = get_cache(args.previous, "jnli", "challenge")
    else:
        with np.load(args.cache / f"{name}-{mode}.npz") as z:
            d = {k: z[k] for k in ("x", "y", "ids", "group")}
    return Data(d, legacy)


def losses(logits, y):
    lp = logits.log_softmax(-1)
    singleton = -lp.gather(1, y.clamp_max(2)[:, None]).flatten()
    return torch.where(y == 3, -torch.logsumexp(lp[:, 1:], dim=-1), singleton)


def dev_loss(logits, y):
    terms = losses(logits, y)
    return float(torch.stack([terms[y == k].mean() for k in y.unique()]).mean())


def fit(train, tune, sizes, seed):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    h = PairResidual()
    names = list(sizes)
    # Fixed, nested subsets. All regimes use exactly 76800 training draws per seed.
    ids = {
        s: torch.randperm(
            len(train[s].y), generator=torch.Generator().manual_seed(907)
        )[:n]
        for s, n in sizes.items()
    }
    x = torch.cat([train[s].x[ids[s]] for s in names])
    prior = torch.cat([train[s].prior[ids[s]] for s in names])
    y = torch.cat([train[s].y[ids[s]] for s in names])
    weights = []
    for s in names:
        labels = train[s].y[ids[s]]
        counts = labels.bincount(minlength=4).clamp_min(1)
        p = counts[labels].float().rsqrt()
        weights.append(p / p.sum())
    sampling = torch.cat(weights)
    with torch.no_grad():
        h.center.copy_(x.mean((0, 1)))
        h.scale.copy_(x.flatten(0, 1).std(0).clamp_min(0.05))
    optimizer = torch.optim.AdamW(h.parameters(), lr=0.001, weight_decay=0.02)
    best = None
    history = []
    for epoch in range(1, 21):
        h.train()
        for batch in torch.multinomial(
            sampling, 3840, replacement=True, generator=rng
        ).split(96):
            optimizer.zero_grad()
            loss = losses(prior[batch] + h(x[batch]), y[batch]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(h.parameters(), 3.0)
            optimizer.step()
        if epoch in (4, 8, 12, 16, 20):
            h.eval()
            with torch.no_grad():
                source_loss = {
                    s: dev_loss(d.prior + h(d.x), d.y) for s, d in tune.items()
                }
            score = sum(source_loss.values()) / len(source_loss)
            history.append(
                {"epoch": epoch, "source_balanced_nll": source_loss, "mean": score}
            )
            if best is None or score < best[0]:
                best = (score, epoch, copy.deepcopy(h))
    return best[2], {"seed": seed, "epoch": best[1], "tune": history}


def measures(logits, y, binary):
    lp = logits.log_softmax(-1)
    if binary:
        pred = (lp[:, 0] < torch.logsumexp(lp[:, 1:], -1)).long()
        gold = (y != 0).long()
    else:
        pred = logits.argmax(-1)
        gold = y
    ok = pred == gold
    names = (
        ["entailment", "non_entailment"]
        if binary
        else ["entailment", "contradiction", "neutral"]
    )
    recall = {
        name: float(ok[gold == i].float().mean())
        for i, name in enumerate(names)
        if (gold == i).any()
    }
    return {
        "correct": int(ok.sum()),
        "n": len(y),
        "accuracy": float(ok.float().mean()),
        "balanced_accuracy": sum(recall.values()) / len(recall),
        "recall": recall,
        "nll": float(losses(logits, y).mean()),
    }, ok.numpy()


@torch.no_grad()
def assess(d, name, models):
    result = {}
    indicators = {}
    for regime, heads in {"v1": [], **models}.items():
        members = [d.prior + h(d.x) for h in heads]
        logits = torch.stack(members).mean(0) if heads else d.prior
        score, ok = measures(logits, d.y, name == "janli")
        if heads:
            score["seed_accuracy"] = [
                measures(z, d.y, name == "janli")[0]["accuracy"] for z in members
            ]
        result[regime] = score
        indicators[regime] = ok.astype(int)
    for candidate, reference in (
        ("jnli_only", "v1"),
        ("equal_4096", "jnli_only"),
        ("all", "v1"),
        ("all", "jnli_only"),
        ("all", "equal_4096"),
    ):
        diff = indicators[candidate] - indicators[reference]
        result[candidate]["vs_" + reference] = {
            "gain": int((diff == 1).sum()),
            "loss": int((diff == -1).sum()),
            "difference": float(diff.mean()),
            "paired_item_bootstrap_95pct": bootstrap_difference(
                diff, np.arange(len(diff))
            ),
            "paired_group_bootstrap_95pct": bootstrap_difference(diff, d.groups),
            "num_groups": len(np.unique(d.groups)),
        }
    if name in ("jsem", "jamp"):
        result["diagnostics"] = {
            "group": "JSeM XML phenomenon" if name == "jsem" else "Jamp template",
            "by_group": {
                group: {
                    "n": int((d.groups == group).sum()),
                    "v1_correct": int(indicators["v1"][d.groups == group].sum()),
                    "all_correct": int(indicators["all"][d.groups == group].sum()),
                }
                for group in np.unique(d.groups).tolist()
            },
        }
    return result


def run(args):
    torch.set_num_threads(args.threads)
    legacy = old_head(args.weights)
    train = {s: read(args, s, "train", legacy) for s in SOURCES}
    tune = {s: read(args, s, "tune", legacy) for s in SOURCES}
    models = {}
    selection = {}
    for regime, sizes in REGIMES.items():
        models[regime] = []
        selection[regime] = []
        for seed in SEEDS:
            h, meta = fit(train, tune, sizes, seed)
            models[regime].append(h)
            selection[regime].append(meta)
            print(regime, seed, meta["epoch"], flush=True)
    # The test sets are decoded only after every selection is frozen.
    results = {}
    for name in (*SOURCES, "jsem", "challenge"):
        d = read(args, name, "eval", legacy)
        results[name] = assess(d, name, models)
        print(
            name,
            {
                k: round(v["accuracy"], 4)
                for k, v in results[name].items()
                if "accuracy" in v
            },
            flush=True,
        )
    if (
        results["jnli"]["v1"]["correct"] != 1755
        or results["challenge"]["v1"]["correct"] != 12
    ):
        raise ValueError("Could not reproduce released NLI scores.")
    for regime, filename in (
        ("all", "multicorpus-nli"),
        ("jnli_only", "jnli-specialist"),
    ):
        save_file(
            nn.ModuleList(models[regime]).state_dict(),
            str(args.weights / f"{filename}.safetensors"),
            metadata={
                "base_revision": REVISION,
                "parent_release": "33842a1",
                "seeds": "11,37,83",
                "regime": regime,
            },
        )
    report = {
        "seeds": SEEDS,
        "training_counts": REGIMES,
        "tuning_counts": {s: len(tune[s].y) for s in SOURCES},
        "selection": selection,
        "evaluation": results,
    }
    args.results.mkdir(exist_ok=True, parents=True)
    (args.results / "multicorpus-nli.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg, default in [
        ("data", ".cache/data"),
        ("previous", ".cache/features"),
        ("cache", ".cache/multi"),
        ("weights", "weights"),
        ("results", "results"),
    ]:
        p.add_argument("--" + arg, type=Path, default=Path(default))
    p.add_argument("--threads", type=int, default=1)
    run(p.parse_args())
