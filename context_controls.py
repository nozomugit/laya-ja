"""Order and symmetry controls, secondary metrics, and class recalls for the fixed readouts."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from context_experiment import dataset, samples, stats
from context_tasks import Readout
from laya_ja import load_base
from scoring import metrics


@torch.no_grad()
def run(a):
    torch.set_num_threads(a.threads)
    agent = load_base(a.model_path, "cpu")
    out = {}
    for task in ("jcommonsenseqa", "jsts", "nli"):
        rows = samples(task, a)
        head = Readout(task).eval()
        head.load_state_dict(load_file(str(a.weights / (task + ".safetensors"))))

        def predict(key, rs, task=task, head=head):
            d = dataset(agent, task, key, rs, a)
            return torch.cat([head(v) for v in d.tensors[0].split(64)]), d.tensors[1]

        if task == "nli":
            out[task] = {}
            for key in (
                "jnli",
                "jsick",
                "jamp",
                "janli",
                "jsem",
                "challenge",
                "reserved",
            ):
                z, y = predict(key, rows[key])
                if key == "janli":
                    z = torch.stack([z[:, 0], z[:, 1:].logsumexp(-1)], dim=1)
                    y = (y != 0).long()
                pred = z.argmax(-1)
                out[task][key] = {
                    **metrics(z, y),
                    "recall": {
                        str(i): float((pred[y == i] == i).float().mean())
                        for i in torch.unique(y).tolist()
                    },
                }
            continue
        name = task if task != "jsts" else "jsts-valid"
        raw = rows[name]
        z, y = predict(name, raw)
        if task == "jcommonsenseqa":
            from corpus_experiment import old_head
            from experiment import combine, extract, get_cache

            legacy = old_head(a.weights.parent, "jcommonsenseqa")

            def prev(key, rs, reverse=False, legacy=legacy):
                mode = {
                    "jcommonsenseqa": "eval",
                    "tune": "tune",
                    "reversed": "eval-reversed",
                }.get(key)
                if mode and (a.previous / f"jcommonsenseqa-{mode}.npz").exists():
                    old = get_cache(a.previous, "jcommonsenseqa", mode)
                    if key == "jcommonsenseqa":
                        old = combine(
                            [old, get_cache(a.previous, "jcommonsenseqa", "eval-extra")]
                        )
                    if list(old["ids"]) != [r[3] for r in rs]:
                        raise ValueError("Legacy QA cache has different rows")
                    feat = torch.from_numpy(old["x"])
                else:
                    feat = extract(agent, rs, a.batch, reverse=reverse)
                return legacy(feat).squeeze(-1)

            tz, ty = predict("tune", rows["tune"])
            pz = prev("tune", rows["tune"])

            def quality(weight, tz=tz, pz=pz, ty=ty):
                s = weight * tz + (1 - weight) * pz
                return int((s.argmax(1) == ty).sum()), -float(
                    torch.nn.functional.cross_entropy(s, ty)
                )

            weight = max((i / 10 for i in range(11)), key=quality)
            (a.weights / "common-mixture.json").write_text(
                json.dumps({"context_weight": weight}) + "\n"
            )
            mix = {}
            for split in ("jcommonsenseqa", "reserved"):
                cz, cy = predict(split, rows[split])
                prior = prev(split, rows[split])
                mix[split] = metrics(weight * cz + (1 - weight) * prior, cy)
            raw = raw[:512]
            z = z[:512]
            y = y[:512]
            flip = [
                ((state, inst, list(reversed(opts))), len(opts) - 1 - g, gp, idx)
                for (state, inst, opts), g, gp, idx in raw
            ]
            f, _ = predict("order-reversed-512", flip)
            f = f.flip(1)
            out[task] = {
                "full": metrics(*predict(name, rows[name])),
                "original_512": metrics(z, y),
                "reversed_512": metrics(f, y),
                "order_agreement": float((z.argmax(1) == f.argmax(1)).float().mean()),
            }
            before = prev("jcommonsenseqa", rows["jcommonsenseqa"])[:512]
            after = prev("reversed", raw, reverse=True)
            ori = z * weight + before * (1 - weight)
            rev = f * weight + after * (1 - weight)
            mix["order_agreement_512"] = float(
                (ori.argmax(1) == rev.argmax(1)).float().mean()
            )
            mix["reversed_512"] = metrics(rev, y)
            out[task]["mixture"] = mix
        else:
            from context_experiment import parse
            from laya_ja import similarity_state

            flip = []
            for row in raw:
                atext, bs = parse(task, row)
                flip.append(((similarity_state(bs[0], atext), *row[0][1:]), *row[1:]))
            f, _ = predict("order-reversed-all", flip)
            out[task] = {
                "original": {
                    **stats(task, z, y),
                    "rmse": float((z.flatten().clamp(0, 5) - y).square().mean().sqrt()),
                },
                "reversed": stats(task, f, y),
                "average": stats(task, (f + z) / 2, y),
                "mean_abs_order_difference": float(
                    (f.clip(0, 5) - z.clip(0, 5)).abs().mean()
                ),
            }
    (a.results / "context-controls.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in [
        ("data", ".cache/data"),
        ("previous", ".cache/features"),
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
