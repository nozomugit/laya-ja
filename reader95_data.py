"""Additional answer-span training and article/paragraph-disjoint JSQuAD controls."""

from grounded_data import paragraph, partitions
from reading_transfer import articles, convert


def extended(data):
    r = partitions(data)
    va = articles(data, "valid")
    valid_titles = {v["title"] for v in va}
    tr = convert(
        [a for a in articles(data, "train") if a["title"] not in valid_titles],
        10**7,
        8301,
    )
    all_va = convert(va, 10**7, 8301)
    titles = {x[2] for x in r["fit"]}
    used = {paragraph(x) for v in r.values() for x in v}
    old_titles = {x[2] for v in r.values() for x in v} | {x[2] for x in tr[:768]}
    r["unseen_titles"] = [
        x for x in tr if x[2] not in old_titles and paragraph(x) not in used
    ]
    r["unseen_passages"] = [x for x in all_va if paragraph(x) not in used][:800]
    block = (
        used
        | {paragraph(x) for k in ("unseen_titles", "unseen_passages") for x in r[k]}
        | {paragraph(x) for x in tr[:768]}
    )
    r["additional"] = [x for x in tr if x[2] in titles and paragraph(x) not in block][
        :4096
    ]
    return r


def audit(data, output):
    import json
    import unicodedata
    from pathlib import Path

    from huggingface_hub import snapshot_download

    revision = "05600ff310a0970823e70f82f428893b85c71ffe"
    source = Path(
        snapshot_download(
            "SkelterLabsInc/JaQuAD",
            repo_type="dataset",
            revision=revision,
            allow_patterns=["data/train/*.json"],
        )
    )
    titles = set()
    for path in sorted((source / "data/train").glob("*.json")):
        for article in json.loads(path.read_text())["data"]:
            titles.add(unicodedata.normalize("NFKC", article["title"]).strip())
    if len(titles) != 691:
        raise ValueError("Unexpected JaQuAD training title count")
    r = extended(data)
    fit = r["fit"] + r["additional"]
    fit_titles = {v[2] for v in fit}
    tune_titles = {v[2] for v in r["dev"]}
    paras = {paragraph(v) for v in fit + r["dev"]}
    results = {}
    for k in ("original", "fresh", "new", "unseen_titles", "unseen_passages"):
        rows = r[k]
        ts = {row[2] for row in rows}
        normalized = {unicodedata.normalize("NFKC", t).strip() for t in ts}
        results[k] = {
            "n": len(rows),
            "titles": len(ts),
            "native_fit_title_overlap": len(ts & fit_titles),
            "tune_title_overlap": len(ts & tune_titles),
            "train_or_tune_paragraph_overlap": sum(
                paragraph(row) in paras for row in rows
            ),
            "jaquad_training_title_overlap": len(normalized & titles),
        }
    out = {
        "jaquad_revision": revision,
        "jaquad_training_titles": len(titles),
        "native_training_articles": len(fit_titles),
        "partitions": results,
    }
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(
        description="Audit training/evaluation article and paragraph intersections"
    )
    p.add_argument("--data", type=Path, default=Path(".cache/data"))
    p.add_argument("--output", type=Path, default=Path("results/reading-overlap.json"))
    args = p.parse_args()
    audit(args.data, args.output)
