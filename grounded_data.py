"""Fixed article-level partitions for the JSQuAD-derived five-choice task."""

import hashlib
import unicodedata

from reading_transfer import articles, convert


def fields(row):
    state, _, choices = row[0]
    context, question = state.removeprefix("本文：").split("\n質問：", 1)
    return context, question, choices


def paragraph(row):
    text = fields(row)[0].split("[SEP]", 1)[-1]
    return "".join(unicodedata.normalize("NFKC", text).split())


def partitions(data):
    valid = articles(data, "valid")
    reserved = {a["title"] for a in valid}
    train = [a for a in articles(data, "train") if a["title"] not in reserved]
    tr = convert(train, 10**7, 8301)
    va = convert(valid, 10**7, 8301)
    original = va[:192]
    old_titles = {r[2] for r in original}
    fresh = [r for r in va if r[2] not in old_titles]
    prior = tr[:768]
    prior_titles = {r[2] for r in prior}
    eligible = sorted(
        {r[2] for r in tr} - prior_titles,
        key=lambda t: hashlib.sha256(("laya-read-v3/" + t).encode()).digest(),
    )
    test_titles, tune_titles = set(eligible[:70]), set(eligible[70:125])
    excluded = {paragraph(r) for r in va + prior}
    new = [r for r in tr if r[2] in test_titles and paragraph(r) not in excluded][:1200]
    excluded.update(paragraph(r) for r in new)
    dev = [r for r in tr if r[2] in tune_titles and paragraph(r) not in excluded][:900]
    excluded.update(paragraph(r) for r in dev)
    fit = [
        r
        for r in tr
        if r[2] not in test_titles | tune_titles and paragraph(r) not in excluded
    ][:6500]
    assert len(fit) == 6500 and len(dev) > 500 and len(new) > 850 and len(fresh) == 198
    assert not {r[2] for r in fit} & {r[2] for r in original + fresh + new + dev}
    assert not {paragraph(r) for r in fit} & {
        paragraph(r) for r in original + fresh + new + dev
    }
    return {"fit": fit, "dev": dev, "original": original, "fresh": fresh, "new": new}
