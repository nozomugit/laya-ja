"""Where the unified LoRA model and hosted Jev differ, problem by problem.

Scores every evaluation set with a saved Laya checkpoint, reads the per-problem probabilities jev_eval.py stored,
and writes a compact text report: confusion tables, who is right on which problems, the phenomena with the largest
gaps, threshold-free measures for acceptability, and a few problems Jev answers and the checkpoint does not.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

import lora_nli
import lora_tasks

NAMES = ["含意", "矛盾", "判断できない"]


def jev_scores(folder, source, name, n):
    rows = {}
    for line in (folder / f"{source}-{name}.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["index"]] = r["probabilities"]
    return torch.tensor([rows[i] for i in range(n)])


def auc(score, positive):
    """Probability that a positive item outranks a negative one."""
    order = sorted(range(len(score)), key=lambda i: score[i])
    rank, total, pos = 0.0, 0.0, 0
    for r, i in enumerate(order, 1):
        if positive[i]:
            total += r
            pos += 1
    neg = len(score) - pos
    return (total - pos * (pos + 1) / 2) / (pos * neg) if pos and neg else float("nan")


def mcc(pred, gold):
    tp = sum(p and g for p, g in zip(pred, gold))
    tn = sum((not p) and (not g) for p, g in zip(pred, gold))
    fp = sum(p and (not g) for p, g in zip(pred, gold))
    fn = sum((not p) and g for p, g in zip(pred, gold))
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


def confusion(pred, gold, k):
    table = [[0] * k for _ in range(k)]
    for p, g in zip(pred, gold):
        table[g][p] += 1
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path(".cache/data"))
    ap.add_argument("--corpora", type=Path, default=Path(".cache/corpora"))
    ap.add_argument("--checkpoint", type=Path, required=True, help="folder laya.load reads, e.g. tasks/unified-final")
    ap.add_argument("--jev", type=Path, required=True, help="folder of jev_eval.py records for one model")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--model-path")
    ap.add_argument("--device")
    args = ap.parse_args()
    args.teacher, args.max_train = None, None
    args.extra_cap, args.logic_cap, args.teacher_cap, args.teacher_all = 4000, 6000, 8000, False
    bench = lora_nli.Bench(args)
    model = bench.model(args.checkpoint)
    out = []

    def say(text=""):
        out.append(text)
        print(text, flush=True)

    for name, rows in lora_nli.corpora(args)[2].items():
        binary = name == "janli"
        ours = bench.score(model, bench.items(rows)).softmax(-1)
        jev = jev_scores(args.jev, "nli", name, len(rows))
        gold = [0 if r[1] == 0 else (1 if binary else r[1]) for r in rows]

        def decide(p):
            if binary:
                return [int(v[0] < v[1] + v[2]) for v in p.tolist()]
            return p.argmax(-1).tolist()

        po, pj = decide(ours), decide(jev)
        k = 2 if binary else 3
        both = sum(a == g and b == g for a, b, g in zip(po, pj, gold))
        only_j = [i for i in range(len(rows)) if pj[i] == gold[i] and po[i] != gold[i]]
        only_o = [i for i in range(len(rows)) if po[i] == gold[i] and pj[i] != gold[i]]
        neither = len(rows) - both - len(only_j) - len(only_o)
        say(f"## nli/{name} n={len(rows)} ours={both + len(only_o)} jev={both + len(only_j)}")
        say(f"both={both} only_jev={len(only_j)} only_ours={len(only_o)} neither={neither}")
        say(f"confusion gold->pred ours={confusion(po, gold, k)} jev={confusion(pj, gold, k)}")
        if not binary:
            lost = Counter((gold[i], po[i]) for i in only_j)
            say(
                "ours wrong where jev right, gold->ours: "
                + " ".join(f"{NAMES[g]}->{NAMES[p]}:{c}" for (g, p), c in lost.most_common())
            )
            conf = [float(ours[i].max()) for i in only_j]
            if conf:
                say(
                    f"ours mean top probability on those={sum(conf) / len(conf):.3f}; share above 0.9={sum(c > 0.9 for c in conf) / len(conf):.2f}"
                )
        if name in ("jsem", "jsem-fresh", "jsem-audit", "challenge", "janli"):
            groups = defaultdict(lambda: [0, 0, 0])
            for i, r in enumerate(rows):
                g = groups[r[2]]
                g[0] += 1
                g[1] += po[i] == gold[i]
                g[2] += pj[i] == gold[i]
            ranked = sorted(groups.items(), key=lambda kv: kv[1][1] - kv[1][2])
            say(
                "groups n/ours/jev, largest deficit first: "
                + "; ".join(f"{g} {n}/{o}/{j}" for g, (n, o, j) in ranked[:14])
            )
        if not binary:
            for i in only_j[: args.examples]:
                state = rows[i][0][0].replace("\n", " / ")
                say(f"  ex gold={NAMES[gold[i]]} ours={NAMES[po[i]]}({float(ours[i].max()):.2f}) | {state[:170]}")
        say()

    for task in lora_tasks.TASKS:
        for name, rows in lora_tasks.task_rows(task, args.data)[2].items():
            ours = lora_tasks.score(bench, model, lora_tasks.task_items(bench, task, rows)).softmax(-1)
            jev = jev_scores(args.jev, task, name, len(rows))
            gold = [r[1] for r in rows]
            say(f"## {task}/{name} n={len(rows)}")
            if task == "jcommonsenseqa":
                po, pj = ours.argmax(-1).tolist(), jev.argmax(-1).tolist()
                both = sum(a == g and b == g for a, b, g in zip(po, pj, gold))
                only_j = [i for i in range(len(rows)) if pj[i] == gold[i] and po[i] != gold[i]]
                only_o = sum(po[i] == gold[i] and pj[i] != gold[i] for i in range(len(rows)))
                say(
                    f"ours={both + only_o} jev={both + len(only_j)} both={both} only_jev={len(only_j)} only_ours={only_o}"
                )
                ranks = Counter(int((ours[i] > ours[i][gold[i]]).sum()) + 1 for i in range(len(rows)))
                say("ours rank of the gold option: " + " ".join(f"{k}:{ranks[k]}" for k in sorted(ranks)))
                wrong = [float(ours[i].max()) for i in range(len(rows)) if po[i] != gold[i]]
                right = [float(ours[i].max()) for i in range(len(rows)) if po[i] == gold[i]]
                say(
                    f"ours mean top probability right={sum(right) / len(right):.3f} wrong={sum(wrong) / len(wrong):.3f}"
                )
                for i in only_j[: args.examples * 2]:
                    options = rows[i][0][2]
                    say(
                        f"  ex {rows[i][0][0]} | gold={options[gold[i]]} ours={options[po[i]]}({float(ours[i].max()):.2f}) | {'・'.join(options)}"
                    )
            elif task == "jcola":
                g = [int(v) for v in gold]
                for label, p in (("ours", ours[:, 1].tolist()), ("jev", jev[:, 1].tolist())):
                    at_half = mcc([v > 0.5 for v in p], g)
                    best = max((mcc([v > t / 100 for v in p], g), t / 100) for t in range(2, 99, 2))
                    table = confusion([int(v > 0.5) for v in p], g, 2)
                    say(
                        f"{label} auc={auc(p, g):.3f} mcc@0.5={at_half:.3f} best_mcc={best[0]:.3f}@{best[1]:.2f} (threshold picked on these labels) confusion={table} mean_p={sum(p) / len(p):.3f}"
                    )
                say(f"share labeled acceptable={sum(g) / len(g):.3f}")
            else:
                grid = torch.arange(6, dtype=ours.dtype)
                eo, ej, y = ours @ grid, jev @ grid.to(jev.dtype), torch.tensor(gold)
                for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5.01)):
                    m = (y >= lo) & (y < hi)
                    say(
                        f"gold {lo}-{int(hi)} n={int(m.sum())} mae ours={float((eo[m] - y[m]).abs().mean()):.3f} jev={float((ej[m] - y[m]).abs().mean()):.3f} bias ours={float((eo[m] - y[m]).mean()):+.3f} jev={float((ej[m] - y[m]).mean()):+.3f}"
                    )
            say()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
