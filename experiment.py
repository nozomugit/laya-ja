"""Extract frozen features, fit option scorers and evaluate the held-out official validation split."""

import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path

import numpy as np
import torch
from laya.common import QTYPES, build_sequence, collate_items
from laya_ja import BASE, REVISION, TASKS, load_base, nli_request, qa_request
from scoring import metrics

DATA_REV = "6f071c09316baae89c3d083a90985b4b1cb9968c"
FILES = {
    "jnli-train": "c506354d1310807f7300cfa8ad49482c6359a84fe85eea8d5e667d90d338d1c4",
    "jnli-valid": "ca0353efc7c2eebfb6de4e13f16295053c8b1ee65e7b0849190c90426fbc495f",
    "jcommonsenseqa-train": "9b55fae5ecb3aedd6f8ce5bc09196c3b629864668ec6c18eee4d65c0aa48229e",
    "jcommonsenseqa-valid": "0d8d76f3bfa0d174866939882faccdd01fbc2bcd5a76c43748ba0c40a7b3b8d4",
}


def rows(task, split, folder):
    folder.mkdir(exist_ok=True, parents=True)
    name = f"{task}-{split}"
    file = folder / f"{name}.json"
    if not file.exists():
        url = f"https://raw.githubusercontent.com/yahoojapan/JGLUE/{DATA_REV}/datasets/{task}-v1.3/{split}-v1.3.json"
        content = urllib.request.urlopen(url, timeout=90).read()
    else:
        content = file.read_bytes()
    if hashlib.sha256(content).hexdigest() != FILES[name]:
        raise ValueError(f"Unexpected dataset checksum: {file}")
    if not file.exists():
        file.write_bytes(content)
    result = []
    for line in content.decode().splitlines():
        obj = json.loads(line)
        if task == "jnli":
            req = nli_request(obj["sentence1"], obj["sentence2"])
            gold = ["entailment", "contradiction", "neutral"].index(obj["label"])
            key = obj["yjcaptions_id"].split("-")[0]
            identifier = obj["sentence_pair_id"]
        else:
            req = qa_request(obj["question"], [obj[f"choice{i}"] for i in range(5)])
            gold = obj["label"]
            key = obj["question"].strip()
            identifier = obj["q_id"]
        result.append((req, gold, key, str(identifier)))
    return result


def partition(task, data, seed, train_size, tune_size, eval_size):
    pool, valid = rows(task, "train", data), rows(task, "valid", data)
    rng = random.Random(seed)
    rng.shuffle(pool)
    rng.shuffle(valid)
    official_keys = {row[2] for row in valid}
    pool = [row for row in pool if row[2] not in official_keys]
    tune = pool[:tune_size]
    tune_keys = {row[2] for row in tune}
    train = [r for r in pool[tune_size:] if r[2] not in tune_keys][:train_size]
    return {"train": train, "tune": tune, "eval": valid[:eval_size]}


@torch.inference_mode()
def extract(agent, rows_, batch_size, reverse=False, kind="choice", progress=True):
    """Capture the exact input to Laya's final linear option scorer."""
    items = []
    for idx, ((state, instruction, choices), _, _, _) in enumerate(rows_):
        order = list(reversed(choices)) if reverse else choices
        if kind == "choice":
            q = {"t": kind, "ins": instruction, "crit": dict.fromkeys(order, "")}
        elif kind == "score":
            q = {"t": kind, "ins": instruction, "crit": list(choices)}
        else:
            q = {
                "t": kind,
                "ins": instruction,
                "crit": dict(zip(("false", "true"), choices)),
            }
        ids, markers = build_sequence(
            agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"]
        )
        if len(markers) != len(choices):
            raise ValueError("Truncated choices")
        items.append((idx, {"ids": ids, "markers": markers, "qtype": QTYPES[kind]}))
    items.sort(key=lambda pair: len(pair[1]["ids"]))
    features = [None] * len(items)
    slot = []
    hook = agent.model.scorer[-1].register_forward_pre_hook(
        lambda module, inputs: slot.append(inputs[0].detach())
    )
    try:
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            b = collate_items([[item] for _, item in chunk], agent.tok.pad_token_id)
            slot.clear()
            agent.model(
                **{
                    k: b[k].to(agent.device)
                    for k in (
                        "input_ids",
                        "attention_mask",
                        "marker_pos",
                        "marker_mask",
                        "qtype",
                    )
                }
            )
            vectors = slot.pop().float().cpu()
            for (idx, item), v in zip(chunk, vectors):
                v = v[: len(item["markers"])]
                features[idx] = v.flip(0) if reverse else v
            if progress and start % (batch_size * 16) == 0:
                print(f"  {start + len(chunk)}/{len(items)}", flush=True)
    finally:
        hook.remove()
    return torch.stack(features)


def save_cache(agent, task, splits, folder, batch_size):
    folder.mkdir(exist_ok=True, parents=True)
    for split, examples in splits.items():
        modes = [False, True] if split == "eval" else [False]
        for reverse in modes:
            mode = split + ("-reversed" if reverse else "")
            cache = folder / f"{task}-{mode}.npz"
            ids = [r[3] for r in examples]
            fingerprint = hashlib.sha256(
                json.dumps(
                    [REVISION, task, mode, examples], ensure_ascii=False
                ).encode()
            ).hexdigest()
            if cache.exists():
                with np.load(cache) as old:
                    if (
                        "fingerprint" in old
                        and old["fingerprint"].item() == fingerprint
                    ):
                        continue
            print(f"{task} {mode}", flush=True)
            x = extract(agent, examples, batch_size, reverse)
            temporary = cache.with_suffix(".tmp.npz")
            np.savez(
                temporary,
                x=x.numpy(),
                y=[r[1] for r in examples],
                ids=ids,
                group=[r[2] for r in examples],
                fingerprint=fingerprint,
            )
            temporary.replace(cache)


def permute_training(examples, seed):
    rng = random.Random(seed)
    augmented = []
    for (state, inst, opts), y, group, idx in examples:
        order = list(range(len(opts)))
        rng.shuffle(order)
        augmented.append(
            ((state, inst, [opts[i] for i in order]), order.index(y), group, idx)
        )
    return augmented


def prepare(agent, task, args):
    splits = partition(
        task, args.data, args.seed, args.train_size, args.tune_size, 10**9
    )
    # One view per unique training item. Mix canonical and permuted candidate order.
    train = splits.pop("train")
    valid = splits.pop("eval")
    n_head = min(1024, len(train))
    n_reverse = min(args.order_items, len(valid))
    batches = {
        "train": train[:n_head],
        "tune": splits["tune"],
        "eval": valid[:n_reverse],
    }
    if len(train) > n_head:
        batches["train-extra"] = permute_training(train[n_head:], args.seed + 1)
    if n_reverse < len(valid):
        batches["eval-extra"] = valid[n_reverse:]
    diagnostics = Path(__file__).with_name("challenge.json")
    if task == "jnli":
        probes = json.loads(diagnostics.read_text())
        batches["challenge"] = [
            (nli_request(a, b), y, category, str(i))
            for i, (category, a, b, y) in enumerate(probes)
        ]
    save_cache(agent, task, batches, args.cache, args.batch_size)
    return batches


def get_cache(folder, task, mode):
    with np.load(folder / f"{task}-{mode}.npz") as array:
        return {k: array[k] for k in ("x", "y", "ids", "group")}


def combine(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def fit_weight(x, y, original, strength):
    # The final linear bias cancels under a softmax shared by every option.
    # Optimize a convex multinomial likelihood, shrinking changes toward the base weights.
    delta = torch.zeros_like(original, requires_grad=True)
    opt = torch.optim.LBFGS([delta], max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        scores = x @ (original + delta)
        loss = (
            torch.nn.functional.cross_entropy(scores, y)
            + strength * delta.square().sum() / 2
        )
        loss.backward()
        return loss

    opt.step(closure)
    return (original + delta).detach()


def bootstrap_difference(d, groups):
    _, inv = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inv, weights=d), np.bincount(inv)
    rng = np.random.default_rng(12345)
    diffs = []
    for _ in range(2000):
        draw = rng.integers(0, len(counts), size=len(counts))
        diffs.append(sums[draw].sum() / counts[draw].sum())
    return np.quantile(diffs, [0.025, 0.975]).tolist()


def assess(data, original, fitted):
    x, y = torch.from_numpy(data["x"]), torch.from_numpy(data["y"])
    base, adapted = x @ original, x @ fitted
    diff = (adapted.argmax(-1) == y).int() - (base.argmax(-1) == y).int()
    return {
        "base": metrics(base, y),
        "adapted": metrics(adapted, y),
        "gained": int((diff == 1).sum()),
        "lost": int((diff == -1).sum()),
        "accuracy_difference_95pct": bootstrap_difference(diff.numpy(), data["group"]),
    }


def train(agent, task, batches, args):
    def read(mode):
        return get_cache(args.cache, task, mode)

    train_data = combine([read(s) for s in ("train", "train-extra") if s in batches])
    tune = read("tune")
    original = agent.model.scorer[-1].weight.detach().cpu().float().flatten()
    x, y = torch.from_numpy(train_data["x"]), torch.from_numpy(train_data["y"])
    tx, ty = torch.from_numpy(tune["x"]), torch.from_numpy(tune["y"])
    candidates = {"base": original}
    for strength in (0.001, 0.01, 0.1, 1.0, 10.0):
        candidates[str(strength)] = fit_weight(x, y, original, strength)
    tuning = {key: metrics(tx @ w, ty) for key, w in candidates.items()}
    selected = min(tuning, key=lambda k: tuning[k]["nll"])
    fitted = candidates[selected]
    args.weights.mkdir(exist_ok=True, parents=True)
    adapter = {
        "base": BASE,
        "revision": REVISION,
        "task": task,
        "weight": [fitted.tolist()],
    }
    (args.weights / f"{task}.json").write_text(json.dumps(adapter) + "\n")
    normal = combine([read(s) for s in ("eval", "eval-extra") if s in batches])
    reversed_data = read("eval-reversed")
    normal_subset = read("eval")
    report = {
        "task": task,
        "seed": args.seed,
        "dataset_revision": DATA_REV,
        "base_revision": REVISION,
        "train_size": len(y),
        "tune_size": len(ty),
        "selected_regularization": selected,
        "tune_candidates": tuning,
        "official_validation": assess(normal, original, fitted),
        "order_reversed": assess(reversed_data, original, fitted),
    }
    for name, w in (("base", original), ("adapted", fitted)):
        first = (torch.from_numpy(normal_subset["x"]) @ w).argmax(-1)
        second = (torch.from_numpy(reversed_data["x"]) @ w).argmax(-1)
        report["order_reversed"][name]["agreement_with_original_order"] = (
            (first == second).float().mean().item()
        )
    if "challenge" in batches:
        c = read("challenge")
        report["challenge"] = {
            name: metrics(torch.from_numpy(c["x"]) @ w, torch.from_numpy(c["y"]))
            for name, w in (("base", original), ("adapted", fitted))
        }
        report["challenge"]["items"] = [
            {
                "id": idx,
                "category": group,
                "gold": int(label),
                "base": int((torch.from_numpy(v) @ original).argmax()),
                "adapted": int((torch.from_numpy(v) @ fitted).argmax()),
            }
            for v, label, idx, group in zip(c["x"], c["y"], c["ids"], c["group"])
        ]
    if task == "jnli":
        report["label_recall"] = {}
        for label, name in enumerate(("entailment", "contradiction", "neutral")):
            subset = normal["y"] == label
            report["label_recall"][name] = {
                k: metrics(
                    torch.from_numpy(normal["x"][subset]) @ w,
                    torch.from_numpy(normal["y"][subset]),
                )["accuracy"]
                for k, w in (("base", original), ("adapted", fitted))
            }
    args.results.mkdir(exist_ok=True, parents=True)
    (args.results / f"{task}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(task, report["official_validation"], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag, default in [
        ("data", ".cache/data"),
        ("cache", ".cache/features"),
        ("weights", "weights"),
        ("results", "results"),
    ]:
        ap.add_argument("--" + flag, type=Path, default=Path(default))
    ap.add_argument("--model-path")
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--train-size", type=int, default=4096)
    ap.add_argument("--tune-size", type=int, default=192)
    ap.add_argument("--order-items", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if (
        min(
            args.train_size,
            args.tune_size,
            args.order_items,
            args.batch_size,
            args.threads,
        )
        < 1
    ):
        ap.error("sizes must be positive")
    torch.set_num_threads(args.threads)
    agent = load_base(args.model_path, device="cpu")
    for task in [args.task] if args.task else TASKS:
        batches = prepare(agent, task, args)
        train(agent, task, batches, args)


if __name__ == "__main__":
    main()
