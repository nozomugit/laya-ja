"""Train and evaluate Laya's native binary and ordinal decision formats."""

import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path

import numpy as np
import torch
from experiment import DATA_REV, extract, fit_weight, metrics
from laya_ja import BASE, GRAMMAR, REVISION, SIMILARITY, load_base, similarity_state

COLA_REV = "736d9eef3af04bb17e4cf59adf01325973234ff9"
SOURCES = {
    "jcola": {
        "in_domain_train-v1.0": "c665f8a8b371b34c3c8947cde3670637cf0403ac7529ba8bb62370d07f008396",
        "in_domain_valid-v1.0": "677fb021886b6829d6a84e8ca9f62808f6bd479fa829439f49eac4f94f23658c",
        "out_of_domain_valid-v1.0": "029d60eb01a96ef333ffb247468bc3123f7a31a2668fa83757dd57af3ad50237",
    },
    "jsts": {
        "jsts-train": "db6bd653270ac9786759920413a48973164d322998f34dd4ae32202af5e66969",
        "jsts-valid": "7c0bdcb381179f01096c635d058853d96da1e1248d23fe3f5c2beed5dc2d9b1a",
    },
}


def load_split(task, name, folder):
    file = folder / (name + ".json")
    if task == "jcola":
        url = f"https://raw.githubusercontent.com/osekilab/JCoLA/{COLA_REV}/data/jcola-v1.0/{file.name}"
    else:
        split = name.split("-")[-1]
        url = f"https://raw.githubusercontent.com/yahoojapan/JGLUE/{DATA_REV}/datasets/jsts-v1.3/{split}-v1.3.json"
    content = (
        file.read_bytes()
        if file.exists()
        else urllib.request.urlopen(url, timeout=90).read()
    )
    if hashlib.sha256(content).hexdigest() != SOURCES[task][name]:
        raise ValueError(f"Checksum mismatch for {name}")
    if not file.exists():
        folder.mkdir(exist_ok=True, parents=True)
        file.write_bytes(content)
    rows = []
    for line in content.decode().splitlines():
        d = json.loads(line)
        if task == "jcola":
            state = d["sentence"]
            request = (
                state,
                GRAMMAR["instructions"],
                list(GRAMMAR["criteria"].values()),
            )
            group = state.strip()
            idx = str(d["uid"])
        else:
            state = similarity_state(d["sentence1"], d["sentence2"])
            request = (state, SIMILARITY["instructions"], SIMILARITY["criteria"])
            group = d["yjcaptions_id"].split("-")[0]
            idx = str(d["sentence_pair_id"])
        rows.append((request, d["label"], group, idx))
    return rows


def diagnostic(task, scores, y):
    if task == "jcola":
        result = metrics(scores, y)
        pred = scores.argmax(dim=-1)
        tp = ((pred == 1) & (y == 1)).sum().item()
        fp = ((pred == 1) & (y == 0)).sum().item()
        tn = ((pred == 0) & (y == 0)).sum().item()
        fn = ((pred == 0) & (y == 1)).sum().item()
        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        result.update(
            mcc=(tp * tn - fp * fn) / denom if denom else 0.0,
            recall_acceptable=tp / (tp + fn),
            recall_unacceptable=tn / (tn + fp),
            majority_accuracy=max(tp + fn, tn + fp) / len(y),
        )
        return result
    pred = scores.softmax(-1) @ torch.arange(6, dtype=scores.dtype)
    return {
        "n": len(y),
        "pearson": float(torch.corrcoef(torch.stack([pred, y]))[0, 1]),
        "mae": float((pred - y).abs().mean()),
        "rmse": float((pred - y).square().mean().sqrt()),
    }


def interpolate(y):
    low, high = y.floor().long(), y.ceil().long()
    a = torch.nn.functional.one_hot(low, num_classes=6)
    b = torch.nn.functional.one_hot(high, num_classes=6)
    fraction = (y - low).unsqueeze(1)
    return a * (1 - fraction) + b * fraction


def run(task, agent, args):
    files = list(SOURCES[task])
    pool, *eval_sets = [load_split(task, f, args.data) for f in files]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    excluded = {row[2] for ev in eval_sets for row in ev}
    pool = [r for r in pool if r[2] not in excluded]
    tune = pool[:192]
    tune_groups = {r[2] for r in tune}
    train = [r for r in pool[192:] if r[2] not in tune_groups][: args.train_size]
    subsets = {"train": train, "tune": tune, **dict(zip(files[1:], eval_sets))}
    contents = {}
    for split, rows in subsets.items():
        print(task, split, flush=True)
        args.cache.mkdir(exist_ok=True, parents=True)
        digest = hashlib.sha256(
            json.dumps([REVISION, task, rows], ensure_ascii=False).encode()
        ).hexdigest()[:20]
        cache = args.cache / f"{task}-{split}-{digest}.npz"
        if not cache.exists():
            x = extract(
                agent,
                rows,
                args.batch_size,
                kind="noul" if task == "jcola" else "score",
            )
            temp = cache.with_suffix(".tmp.npz")
            np.savez(temp, x=x.numpy(), y=[r[1] for r in rows])
            temp.replace(cache)
        with np.load(cache) as d:
            contents[split] = (torch.from_numpy(d["x"]), torch.from_numpy(d["y"]))
    x, y = contents["train"]
    soft_y = interpolate(y.float()) if task == "jsts" else y
    tx, ty = contents["tune"]
    original = agent.model.scorer[-1].weight.detach().cpu().float().flatten()
    candidates = {
        "base": original,
        **{
            str(s): fit_weight(x, soft_y, original, s)
            for s in (0.001, 0.01, 0.1, 1.0, 10.0)
        },
    }
    tuning = {
        k: diagnostic(task, tx @ w, ty.float() if task == "jsts" else ty)
        for k, w in candidates.items()
    }
    key = "nll" if task == "jcola" else "mae"
    winner = min(tuning, key=lambda k: tuning[k][key])
    fitted = candidates[winner]
    args.weights.mkdir(exist_ok=True, parents=True)
    params = {
        "base": BASE,
        "revision": REVISION,
        "task": task,
        "weight": [fitted.tolist()],
    }
    (args.weights / f"{task}.json").write_text(json.dumps(params) + "\n")
    result = {
        "task": task,
        "seed": args.seed,
        "train_size": len(y),
        "tune_size": len(ty),
        "base_revision": REVISION,
        "dataset_revision": COLA_REV if task == "jcola" else DATA_REV,
        "selected_regularization": winner,
        "tune_candidates": tuning,
    }
    for name in files[1:]:
        vx, vy = contents[name]
        result[name] = {
            k: diagnostic(task, vx @ w, vy.float() if task == "jsts" else vy)
            for k, w in (("base", original), ("adapted", fitted))
        }
    args.results.mkdir(exist_ok=True, parents=True)
    (args.results / f"{task}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    from compare_heads import linear, readout, save_selected, select_head

    target_y = y.float() if task == "jsts" else y
    target_ty = ty.float() if task == "jsts" else ty
    simple, chosen, title, tune_candidates = select_head(
        task, x, target_y, tx, target_ty, fitted
    )
    save_selected(args.weights, task, params, chosen, title)
    final = {"task": task, "selected": title, "tune_candidates": tune_candidates}
    for name in files[1:]:
        vx, vy = contents[name]
        final[name] = {
            k: diagnostic(task, readout(m, vx), vy.float() if task == "jsts" else vy)
            for k, m in (
                ("base", linear(original).eval()),
                ("linear", simple),
                ("selected", chosen),
            )
        }
    (args.results / f"{task}-final.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n"
    )
    print(task, {k: final[k] for k in files[1:]}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, default in [
        ("data", ".cache/data"),
        ("cache", ".cache/features"),
        ("weights", "weights"),
        ("results", "results"),
    ]:
        p.add_argument("--" + k, type=Path, default=Path(default))
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", choices=tuple(SOURCES))
    p.add_argument("--model-path")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    agent = load_base(args.model_path, device="cpu")
    for task in [args.task] if args.task else SOURCES:
        run(task, agent, args)
