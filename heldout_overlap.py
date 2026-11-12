"""Check held-out problems against everything the LoRA models train on, and print counts and problem ids only.

A held-out problem is flagged when the character-bigram Jaccard similarity between its text (state and question)
and a training example's text reaches the threshold, the same measure used to keep JSeM training apart from the
JSeM test sets. No problem text is printed.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import lora_nli
import lora_tasks


def grams(text):
    t = "".join(text.split())
    return {t[i : i + 2] for i in range(len(t) - 1)}


def training_texts(args):
    train = lora_nli.corpora(args)[0]
    for source, rows in train.items():
        for r in rows:
            yield source, r[0][0]
    for task in lora_tasks.TASKS:
        for r in lora_tasks.task_rows(task, args.data)[0]:
            yield task, r[0][0] + " " + " ".join(map(str, r[0][2])) if task == "jcommonsenseqa" else r[0][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problems", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path(".cache/data"))
    ap.add_argument("--corpora", type=Path, default=Path(".cache/corpora"))
    ap.add_argument("--teacher", type=Path)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    args.max_train, args.extra_cap, args.logic_cap, args.teacher_cap, args.teacher_all = None, 4000, 6000, 8000, False
    raw = args.problems.read_bytes()
    held = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            r = json.loads(line)
            body = (
                f"前提：{r['state']}\n仮説：{r['question']}" if r["type"] == "nli" else r["state"] + " " + r["question"]
            )
            held.append((r["id"], grams(body)))
    index = defaultdict(list)
    for i, (_, g) in enumerate(held):
        for x in g:
            index[x].append(i)
    flagged, best, seen = defaultdict(set), [0.0] * len(held), Counter()
    for source, text in training_texts(args):
        seen[source] += 1
        g = grams(text)
        shared = Counter(i for x in g for i in index.get(x, ()))
        for i, n in shared.items():
            score = n / (len(g) + len(held[i][1]) - n)
            best[i] = max(best[i], score)
            if score >= args.threshold:
                flagged[held[i][0]].add(source)
    result = {
        "problems": args.problems.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "held_out": len(held),
        "training_texts": dict(seen),
        "threshold": args.threshold,
        "flagged": {k: sorted(v) for k, v in sorted(flagged.items())},
        "highest_similarity": round(max(best), 3),
        "problems_above_0.3": sum(b >= 0.3 for b in best),
    }
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
