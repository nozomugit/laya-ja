"""Measure candidate teachers on the public held-out problems before they label any training data.

prepare  writes heldout/dev.jsonl in the layout score_teacher.py reads (same wording as the pilot questions).
report   scores one or more score_teacher.py outputs, and their average, against the answers.

Only the public dev file is accepted; the sealed test file is refused by its checksum.
"""

import argparse
import hashlib
import json
from pathlib import Path

LABELS = ["含意", "矛盾", "判断できない"]
NLI_QUESTION = "前提が正しいとき、仮説はどう判断できますか？"
# Jev 1.14 on the same 200 problems, and the interim bar from the distillation plan.
JEV = {"nli": 0.931, "choice": 1.0, "noul": 1.0, "score": 0.03}
INTERIM = {"nli": 0.85, "choice": 0.85, "noul": 0.90, "score": 0.5}


def dev_rows(path):
    sums = dict(line.split()[::-1] for line in (path.parent / "SHA256SUMS").read_text().splitlines())
    if hashlib.sha256(path.read_bytes()).hexdigest() != sums.get("dev.jsonl"):
        raise SystemExit("Only the public dev file may be used to choose teachers.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shaped(r):
    t = r["type"]
    if t == "nli":
        return f"前提：{r['state']}\n仮説：{r['question']}", NLI_QUESTION, LABELS, LABELS.index(r["answer"])
    if t == "choice":
        return r["state"], r["question"], list(r["options"]), int(r["answer"]) - 1
    if t == "noul":
        return r["state"], r["question"], ["いいえ", "はい"], int(bool(r["answer"]))
    return r["state"], r["question"], list(r["options"]), float(r["answer"])


def prepare(args):
    with args.out.open("x", encoding="utf-8") as f:
        for r in dev_rows(args.dev):
            state, question, options, _ = shaped(r)
            row = dict(id=r["id"], doc_id=r["group"], type=r["type"], state=state, question=question, options=options)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def measure(rows, probs):
    out = {}
    for t in ("nli", "choice", "noul", "score"):
        picked = [(shaped(r)[3], probs[r["id"]]) for r in rows if r["type"] == t]
        if t == "score":
            err = [abs(sum(i * p for i, p in enumerate(pr)) - gold) for gold, pr in picked]
            top = [abs(max(range(len(pr)), key=pr.__getitem__) - gold) for gold, pr in picked]
            out[t] = {"n": len(picked), "mae": round(sum(err) / len(err), 3), "mae_top": round(sum(top) / len(top), 3)}
        else:
            hit = sum(max(range(len(pr)), key=pr.__getitem__) == gold for gold, pr in picked)
            out[t] = {"n": len(picked), "correct": hit, "accuracy": round(hit / len(picked), 4)}
    return out


def verdict(m):
    def meets(bar):
        return all(m[t]["accuracy"] >= bar[t] for t in ("nli", "choice", "noul")) and m["score"]["mae"] <= bar["score"]

    return "jev" if meets(JEV) else "interim" if meets(INTERIM) else "below"


def report(args):
    rows = dev_rows(args.dev)
    teachers = {}
    for path in args.outputs:
        recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        probs = {r["id"]: r["probabilities"] for r in recs}
        if probs.keys() != {r["id"] for r in rows}:
            raise SystemExit(f"{path}: ids do not match the dev file")
        teachers[f"{recs[0]['model']}@{recs[0]['revision'][:8]}"] = probs
    names = list(teachers)
    combos = {n: teachers[n] for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            combos[f"mean({a}, {b})"] = {k: [(x + y) / 2 for x, y in zip(teachers[a][k], teachers[b][k])] for k in teachers[a]}
    result = {}
    for name, probs in combos.items():
        m = measure(rows, probs)
        result[name] = dict(m, verdict=verdict(m))
        print(f"{name}\n  nli {m['nli']['accuracy']:.3f}  choice {m['choice']['accuracy']:.3f}  noul {m['noul']['accuracy']:.3f}"
              f"  score MAE {m['score']['mae']:.2f} (top {m['score']['mae_top']:.2f})  -> {result[name]['verdict']}")
    if args.json:
        args.json.write_text(json.dumps(dict(jev=JEV, interim=INTERIM, results=result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--dev", type=Path, default=Path("heldout/dev.jsonl"))
    a.add_argument("--out", type=Path, required=True)
    a.set_defaults(fn=prepare)
    b = sub.add_parser("report")
    b.add_argument("outputs", type=Path, nargs="+")
    b.add_argument("--dev", type=Path, default=Path("heldout/dev.jsonl"))
    b.add_argument("--json", type=Path)
    b.set_defaults(fn=report)
    args = p.parse_args()
    args.fn(args)
