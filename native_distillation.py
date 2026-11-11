"""Transfer soft semantic decisions into a small Laya-only residual readout."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

from context_tasks import ContextReader, batch, item
from four_task_report import load, score
from semantic_verifier import weights


@torch.inference_mode()
def encode(reader, rows):
    zs, ps = [], []
    for start in range(0, len(rows), 12):
        chunk = rows[start : start + 12]
        x = batch(
            reader.agent,
            [item(reader.agent.tok, "nli", r["text"], r["options"]) for r in chunk],
        )
        zs.append(reader.head.representation(x).squeeze(1).cpu().numpy())
        ps.append(reader.head(x).cpu().numpy())
    return np.vstack(zs), np.vstack(ps)


def fit(z, prior, targets, w, penalty):
    mean = w @ z
    sd = np.sqrt(w @ ((z - mean) ** 2)).clip(0.15, None)
    x = (z - mean) / sd
    x = np.column_stack([x, np.ones(len(x))])
    target = np.log(np.maximum(targets, 1e-7))
    target -= target.mean(axis=1, keepdims=True)
    residual = target - (prior - prior.mean(axis=1, keepdims=True))
    weighted = w[:, None] * x
    mat = weighted.T @ x + penalty * np.eye(x.shape[1])
    coef = np.linalg.solve(mat, weighted.T @ residual)
    return {"mean": mean.tolist(), "sd": sd.tolist(), "coef": coef.tolist()}


def predict(z, prior, cfg):
    x = (z - cfg["mean"]) / cfg["sd"]
    x = np.column_stack([x, np.ones(len(x))])
    return softmax(prior + x @ cfg["coef"], axis=1)


def run(a):
    torch.set_num_threads(a.threads)
    reader = ContextReader("nli", a.model_path)
    teach = load(a.teacher)
    by_id = {tuple(r["id"]): r["logits"] for r in teach}
    parts = ["nli-dev", "semantic-cal"]
    dev = [load(a.inputs / (k + ".jsonl")) for k in parts]
    rows = dev[0] + dev[1]
    z, prior = encode(reader, rows)
    y = np.array([r["gold"] for r in rows])
    groups = json.loads((a.inputs / "groups.json").read_text())
    g = np.array(
        ["jnli"] * 48
        + ["jsick"] * 48
        + ["world" + str(n) for n in groups["semantic-cal"]]
    )
    teacher = softmax(
        [by_id[k, i] for k, rs in zip(parts, dev) for i in range(len(rs))], axis=1
    )
    target = 0.65 * np.eye(3)[y] + 0.35 * teacher
    allw = weights(y, g)
    candidates = []
    for reg in (0.03, 0.3, 3, 30):
        held = np.empty((len(y), 3))
        for name in np.unique(g):
            train = g != name
            cfg = fit(
                z[train], prior[train], target[train], weights(y[train], g[train]), reg
            )
            held[~train] = predict(z[~train], prior[~train], cfg)
        acc = float(allw @ (held.argmax(1) == y))
        loss = float(-allw @ np.log(held[np.arange(len(y)), y].clip(1e-10)))
        candidates.append((acc, -loss, reg))
    _, _, reg = max(candidates)
    cfg = fit(z, prior, target, allw, reg)
    a.config.write_text(json.dumps(cfg, separators=(",", ":")) + "\n")
    report = {
        "regularization": reg,
        "group_held_out_validation": candidates,
        "training": len(rows),
    }
    for k in ("jsem", "challenge", "jsem-fresh"):
        rs = load(a.inputs / (k + ".jsonl"))
        x, p = encode(reader, rs)
        pred = predict(x, p, cfg)
        gold = np.array([r["gold"] for r in rs])
        report[k] = {
            "student": score(gold, pred),
            "base": score(gold, softmax(p, axis=1)),
        }
    a.result.write_text(json.dumps(report, indent=2) + "\n")
    print(
        {
            s: report[s]["student"]["correct"]
            for s in ("jsem", "challenge", "jsem-fresh")
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path")
    p.add_argument("--threads", type=int, default=3)
    for k, v in {
        "inputs": ".cache/four",
        "teacher": ".cache/semantic-predictions/direct-out.jsonl",
        "result": "results/four-native-distill.json",
        "config": "weights/native-distill.json",
    }.items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    run(p.parse_args())
