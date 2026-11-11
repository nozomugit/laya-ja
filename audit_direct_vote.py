"""Verify an additional premise-disjoint JSeM sample with zero-generation judges."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import softmax

from corpus_data import jsem, normalized
from four_task_assist import LABELS
from four_task_prompts import nli_prompt
from four_task_report import load
from semantic_vote import decide


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def prepare(args):
    from context_tasks import ContextReader

    sizes = {
        "jsem": 384,
        "jsem-fresh": 96,
        "challenge": 36,
        "nli-dev": 96,
        "semantic-cal": 144,
        "qa-dev": 64,
        "jcommonsenseqa": 1119,
        "jamp": 348,
        "jamp-dev": 192,
    }
    blocked = set()
    for split, n in sizes.items():
        rows = load(args.inputs / (split + ".jsonl"))
        if len(rows) != n:
            raise ValueError(f"Incomplete prior data for {split}")
        blocked.update(normalized(row["text"]) for row in rows)
    pool = sorted(
        jsem(args.corpora), key=lambda r: digest("Laya audit 36 2026-11-12 " + r[3])
    )
    count = Counter()
    samples = []
    reader = ContextReader("nli", args.model_path)
    prompts = []
    for (s, _, _), gold, _, ident in pool:
        premise, h = s.removeprefix("前提：").split("\n仮説：", 1)
        if normalized(premise) in blocked or count[gold] >= 12:
            continue
        blocked.add(normalized(premise))
        count[gold] += 1
        probs = reader.predict(premise, [h])["probabilities"]
        prompts.append({"id": len(samples), "prompt": nli_prompt(premise, h)})
        samples.append(
            {
                "id": ident,
                "hash": digest(premise + "\n" + h),
                "laya": [probs[k] for k in LABELS],
                "gold": gold,
            }
        )
    if sorted(count.values()) != [12, 12, 12]:
        raise ValueError("Not enough new premises")
    args.work.mkdir(exist_ok=True, parents=True)
    (args.work / "sample.json").write_text(json.dumps(samples))
    (args.work / "direct.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prompts)
    )


def score(args):
    rows = json.loads((args.work / "sample.json").read_text())
    if len(rows) != 36 or len({r["id"] for r in rows}) != 36:
        raise ValueError("Incomplete independent sample")
    cfg = json.loads(args.calibration.read_text())
    if cfg["weights"]["counter"] or cfg["weights"]["gemma"]:
        raise ValueError("This test requires the direct-only calibration")
    views = {}
    for name in ("direct35", "direct27"):
        values = load(args.work / (name + "-out.jsonl"))
        if [r["id"] for r in values] != list(range(len(rows))):
            raise ValueError(f"Misaligned predictions from {name}")
        views[name] = softmax([r["logits"] for r in values], axis=1)
    result = []
    for i, r in enumerate(rows):
        p = {name: v[i] for name, v in views.items()}
        p["laya"] = np.asarray(r["laya"])
        choice, used, _, _ = decide(p.__getitem__, config=cfg)
        full, _, _, _ = decide(p.__getitem__, config=cfg, early=False)
        if choice != full:
            raise ValueError("Early exit disagrees with full mixture")
        fixed = 0.1 * p["laya"] + 0.45 * p["direct35"] + 0.45 * p["direct27"]
        direct35 = 0.1 * p["laya"] + 0.9 * p["direct35"]
        direct27 = 0.1 * p["laya"] + 0.9 * p["direct27"]
        result.append(
            {
                **r,
                "calibrated": choice,
                "used": used,
                "fixed": int(fixed.argmax()),
                "baseline35": int(direct35.argmax()),
                "baseline27": int(direct27.argmax()),
                "native": int(p["laya"].argmax()),
            }
        )
    methods = ("native", "baseline35", "baseline27", "fixed", "calibrated")
    args.result.write_text(
        json.dumps(
            {
                "n": len(rows),
                "correct": {
                    key: sum(r[key] == r["gold"] for r in result) for key in methods
                },
                "rows": result,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "score"])
    p.add_argument("--corpora", type=Path, default=Path(".cache/corpora"))
    p.add_argument("--inputs", type=Path, default=Path(".cache/four"))
    p.add_argument("--model-path")
    p.add_argument("--work", type=Path, default=Path(".cache/vote-audit"))
    p.add_argument(
        "--calibration", type=Path, default=Path("weights/semantic-vote-direct.json")
    )
    p.add_argument(
        "--result", type=Path, default=Path("results/semantic-vote-audit36.json")
    )
    a = p.parse_args()
    (prepare if a.action == "prepare" else score)(a)
