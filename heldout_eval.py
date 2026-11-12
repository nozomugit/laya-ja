"""Score a Laya checkpoint and hosted Jev on the held-out problem files (heldout/dev.jsonl and the sealed test file).

Each problem carries its type: nli (three labels), choice (1-based option number), noul (true or false) or
score (0 to 5). Both models get the same state, question and options. Only totals per type and per category are
printed and written; with --sealed nothing about a single problem is shown, and the stored Jev records hold
probabilities and line numbers, never text.
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch

import jev_eval
import lora_nli
import lora_tasks

LABELS = ["含意", "矛盾", "判断できない"]
KIND = {"choice": "jcommonsenseqa", "score": "jsts", "noul": "jcola"}
YES_NO = ["いいえ", "はい"]


def load(path):
    """Rows in the shape the scoring code uses, grouped by problem type."""
    out = defaultdict(list)
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip() or line.startswith("#"):
            continue
        r = json.loads(line)
        t, key = r["type"], (r["category"], r["group"], n)
        if t == "nli":
            out[t].append((lora_nli.nli_request(r["state"], r["question"]), LABELS.index(r["answer"]), key))
        elif t == "choice":
            out[t].append(((r["state"], r["question"], list(r["options"])), int(r["answer"]) - 1, key))
        elif t == "noul":
            out[t].append(((r["state"], r["question"], YES_NO), int(bool(r["answer"])), key))
        elif t == "score":
            out[t].append(((r["state"], r["question"], list(r["options"])), float(r["answer"]), key))
        else:
            raise ValueError(f"line {n + 1}: unknown type {t}")
    return out


def summarize(t, probs, rows):
    """Totals for one type, and per category."""

    def measure(idx):
        p = probs[idx]
        gold = [rows[i][1] for i in idx]
        if t == "score":
            pred = p @ torch.arange(p.shape[1], dtype=p.dtype)
            y = torch.tensor(gold, dtype=p.dtype)
            m = {"n": len(idx), "mae": round(float((pred - y).abs().mean()), 3)}
            if len(idx) > 2 and float(y.std()) > 0 and float(pred.std()) > 0:
                m["pearson"] = round(float(torch.corrcoef(torch.stack([pred, y]))[0, 1]), 3)
            return m
        hit = int((p.argmax(-1) == torch.tensor(gold)).sum())
        return {"n": len(idx), "correct": hit, "accuracy": round(hit / len(idx), 4)}

    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r[2][0]].append(i)
    return {"all": measure(list(range(len(rows)))), "by_category": {c: measure(idx) for c, idx in sorted(by.items())}}


def laya(args, data):
    bench = lora_nli.Bench(args)
    model = bench.model(args.checkpoint)
    out = {}
    for t, rows in data.items():
        shaped = [(r[0], r[1], "", "") for r in rows]
        if t == "nli":
            logits = bench.score(model, bench.items(shaped))
        else:
            logits = lora_tasks.score(bench, model, lora_tasks.task_items(bench, KIND[t], shaped))
        out[t] = logits.softmax(-1)
    return out


def jev(args, data, folder):
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        try:
            from google.colab import userdata

            key = userdata.get(args.secret)
        except Exception as error:
            raise SystemExit(
                f"Set TYPESAFE_API_KEY, or the Colab secret {args.secret}: {type(error).__name__}"
            ) from None
    client = jev_eval.Client(args.url, args.model, key)
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for t, rows in data.items():
        shaped = [(r[0], r[1], "", str(r[2][2])) for r in rows]
        logits = jev_eval.collect(client, KIND.get(t, "nli"), t, shaped, folder, args.workers)
        out[t] = logits.exp()
    return out, sorted(str(v) for v in client.resolved)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problems", type=Path, default=Path(__file__).with_name("heldout") / "dev.jsonl")
    ap.add_argument("--checkpoint", type=Path, nargs="*", default=[], help="folders laya.load reads")
    ap.add_argument("--upstream", action="store_true", help="also score the upstream base")
    ap.add_argument("--jev", action="store_true")
    ap.add_argument("--records", type=Path, default=Path(".cache/heldout-jev"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sealed", action="store_true", help="print totals only")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--url", default=jev_eval.URL)
    ap.add_argument("--secret", default="JEV_API_KEY")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data", type=Path, default=Path(".cache/data"))
    ap.add_argument("--model-path")
    ap.add_argument("--device")
    args = ap.parse_args()
    data = load(args.problems)
    result = {"problems": args.problems.name, "counts": {t: len(v) for t, v in data.items()}, "models": {}}
    runs = [(str(p), p) for p in args.checkpoint] + ([("upstream", None)] if args.upstream else [])
    for name, path in runs:
        args.checkpoint_path = path
        probs = laya(argparse.Namespace(**{**vars(args), "checkpoint": path}), data)
        result["models"][name] = {t: summarize(t, probs[t], data[t]) for t in data}
    if args.jev:
        probs, answered = jev(args, data, args.records / args.problems.stem / args.model)
        result["models"]["jev"] = {t: summarize(t, probs[t], data[t]) for t in data}
        result["jev_answered_by"] = answered
    for name, m in result["models"].items():
        for t, s in m.items():
            print(name, t, s["all"], flush=True)
            if not args.sealed:
                for c, v in s["by_category"].items():
                    print("   ", c, v)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
