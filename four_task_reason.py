"""Fit the optional deliberating judge using only independently reserved development labels."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import softmax

from four_task_prompts import nli_prompt
from four_task_report import load, score, select_bias, shift


def reason_prompt(p, h):
    prompt = nli_prompt(p, h)
    suffix = "<think>\n\n</think>\n\n"
    if not prompt.endswith(suffix):
        raise ValueError("Unexpected model template")
    return prompt[: -len(suffix)] + "<think>\n"


def engine_scores(engine, model, prompts, budget, threads):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "inputs.jsonl"
        out = Path(tmp) / "logits.jsonl"
        src.write_text(
            "".join(
                json.dumps({"id": i, "prompt": p}, ensure_ascii=False) + "\n"
                for i, p in enumerate(prompts)
            )
        )
        subprocess.run(
            [
                str(engine.resolve()),
                str(model),
                str(src),
                str(out),
                str(min(4, len(prompts))),
                str(threads),
                str(budget),
            ],
            check=True,
        )
        records = load(out)
        if [r["id"] for r in records] != list(range(len(prompts))):
            raise ValueError("Unexpected response rows")
        return softmax(np.asarray([r["logits"] for r in records]), axis=1)


def combine(prior, quick, deep, cfg):
    specialist = (1 - cfg["deep_weight"]) * quick + cfg["deep_weight"] * deep
    mix = cfg["native_weight"] * prior + (1 - cfg["native_weight"]) * specialist
    return shift(mix, cfg.get("label_bias", [0, 0, 0]))


def run(a):
    baseline = json.loads(a.fast_config.read_text())
    groups = json.loads((a.inputs / "groups.json").read_text())
    outputs = {}
    raw = []
    for name in ("nli-dev", "challenge", "jsem"):
        rows = load(a.inputs / (name + ".jsonl"))
        y = np.array([r["gold"] for r in rows])
        prior = np.array([r["laya"] for r in rows])
        views = {}
        for key in ["direct", "contrast", "reason"]:
            r = {
                v["id"][1]: v["logits"]
                for v in load(a.pred / (key + "-out.jsonl"))
                if v["id"][0] == name
            }
            if sorted(r) != list(range(len(rows))):
                raise ValueError(f"Incomplete {key}/{name}")
            p = softmax([r[i] for i in range(len(rows))], axis=1)
            views[key] = p[:, [1, 2, 0]] if key == "contrast" else p
        v = baseline["contrast_weight"]
        quick = (1 - v) * views["direct"] + v * views["contrast"]
        deep = views["reason"]
        fixed = {"native_weight": 0.1, "deep_weight": 0.75}
        if name == "nli-dev":

            def dev(weights, prior=prior, quick=quick, deep=deep, y=y):
                w, u = weights
                probs = combine(
                    prior, quick, deep, {"native_weight": w, "deep_weight": u}
                )
                accuracy = [np.mean(probs[y == c].argmax(1) == c) for c in range(3)]
                return np.mean(accuracy), np.log(probs[np.arange(len(y)), y]).mean()

            w, u = max(
                (
                    (w, u)
                    for w in (0.05, 0.1, 0.2, 0.3)
                    for u in (0, 0.25, 0.5, 0.75, 1)
                ),
                key=dev,
            )
            cfg = {"native_weight": w, "deep_weight": u, "contrast_weight": v}
            b, penalty = select_bias(combine(prior, quick, deep, cfg), y)
            cfg.update(label_bias=b.tolist(), bias_penalty=penalty, reasoning_tokens=96)
        p = combine(prior, quick, deep, cfg)
        result = {
            "combined": score(y, p),
            "reason_only": score(y, deep),
            "fixed_mix": score(y, combine(prior, quick, deep, fixed)),
            "single_judge": score(y, 0.1 * prior + 0.9 * deep),
        }
        if name in groups:
            g = np.array(groups[name])
            result["groups"] = {
                str(k): score(y[g == k], p[g == k]) for k in np.unique(g)
            }
        outputs[name] = result
        raw.extend(
            {"set": name, "row": i, "prediction": int(c), "gold": int(y[i])}
            for i, c in enumerate(p.argmax(1))
        )
    a.config.write_text(json.dumps(cfg, indent=2) + "\n")
    a.result.write_text(json.dumps(outputs, indent=2, ensure_ascii=False) + "\n")
    if a.predictions:
        a.predictions.write_text("".join(json.dumps(v) + "\n" for v in raw))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in {
        "inputs": ".cache/four",
        "pred": ".cache/four-predictions",
        "fast-config": "weights/four-fusion.json",
        "config": "weights/four-reason.json",
        "result": "results/four-reason.json",
    }.items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    p.add_argument("--predictions", type=Path)
    run(p.parse_args())
