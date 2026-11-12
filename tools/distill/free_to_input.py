"""Turn generated replies into teacher input, dropping malformed items and documents close to held-out problems.

Only counts are printed. Held-out files (the public dev file, and the sealed file when given) are read solely to
compare character bigrams; nothing from them is written anywhere.
"""

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

LABELS = ["含意", "矛盾", "判断できない"]
NLI_QUESTION = "前提が正しいとき、仮説はどう判断できますか？"


def grams(text):
    t = "".join(unicodedata.normalize("NFKC", text).split())
    return {t[i : i + 2] for i in range(len(t) - 1)}


def parse(reply):
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def strings(value, low, high):
    ok = isinstance(value, list) and low <= len(value) <= high and all(isinstance(v, str) and v.strip() for v in value)
    return ok and len(set(value)) == len(value)


def shaped(text, item):
    """(type, state, question, options) or None."""
    if not isinstance(item, dict):
        return None
    t, q = item.get("type"), item.get("question")
    if t == "nli":
        h = item.get("hypothesis")
        if isinstance(h, str) and h.strip():
            return t, f"前提：{text}\n仮説：{h.strip()}", NLI_QUESTION, LABELS
        return None
    if not isinstance(q, str) or not q.strip():
        return None
    if t == "choice" and strings(item.get("options"), 2, 6):
        return t, text, q.strip(), item["options"]
    if t == "noul":
        return t, text, q.strip(), ["いいえ", "はい"]
    if t == "score" and strings(item.get("options"), 6, 6):
        return t, text, q.strip(), item["options"]
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("replies", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--heldout", type=Path, nargs="*", default=[])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--prefix", default="F")
    a = p.parse_args()
    held = []
    for path in a.heldout:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                held.append(grams(json.loads(line)["state"]))
    a.out.mkdir(parents=True)
    stats, top = Counter(), 0.0
    with (a.out / "teacher_input.jsonl").open("w", encoding="utf-8") as qs, (a.out / "documents.jsonl").open(
        "w", encoding="utf-8"
    ) as ds:
        for line in a.replies.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            obj = parse(r["reply"])
            text = obj.get("text") if obj else None
            if not isinstance(text, str) or len(text.strip()) < 30 or not isinstance(obj.get("items"), list):
                stats["document_unusable"] += 1
                continue
            text = text.strip()
            g = grams(text)
            near = max((len(g & h) / len(g | h) for h in held if g | h), default=0.0)
            top = max(top, near)
            if near >= a.threshold:
                stats["document_near_heldout"] += 1
                continue
            doc = a.prefix + r["doc_id"]
            kept = 0
            for i, item in enumerate(obj["items"]):
                s = shaped(text, item)
                if s is None:
                    stats["item_malformed"] += 1
                    continue
                row = dict(id=f"{doc}-{i:02d}", doc_id=doc, type=s[0], state=s[1], question=s[2], options=s[3])
                qs.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats[s[0]] += 1
                kept += 1
            if kept:
                ds.write(json.dumps(dict(doc_id=doc, split=r["split"]), ensure_ascii=False) + "\n")
                stats["documents_" + r["split"]] += 1
    report = dict(counts=dict(stats), highest_similarity_to_heldout=round(top, 3), threshold=a.threshold)
    (a.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
