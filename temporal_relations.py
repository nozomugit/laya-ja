"""Jamp date comparison features, blended with the Japanese Laya inference head."""

import datetime
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np

TIME = re.compile(r"(?:[0-9]+(?:年間|か月間|ヶ月間|月間|日間|時間|年|月|日|時))+")
FIELD = re.compile(r"([0-9]+)(年間|か月間|ヶ月間|月間|日間|時間|年|月|日|時)")
UNITS = ("年", "月", "日", "時")
KEYS = (
    "現在",
    "以前",
    "以降",
    "以来",
    "以上",
    "以内",
    "より",
    "前",
    "後",
    "先",
    "まで",
    "から",
    "ちょうど",
    "費や",
    "かけ",
    "である",
    "始め",
    "終え",
    "毎",
    "いつも",
    "決して",
    "遅れ",
    "ず",
    "なかった",
    "ない",
    "存在",
    "多く",
    "少なく",
    "全",
    "いる",
    "いた",
    "それ",
    "た。",
    "。",
)


def stamps(text):
    result = []
    for match in TIME.finditer(text):
        values = {
            unit.replace("か", "").replace("ヶ", "")[0]: int(n)
            for n, unit in FIELD.findall(match[0])
        }
        y, m, d, h = [values.get(u, 0) for u in UNITS]
        try:
            ordinal = datetime.date(y or 2000, m or 1, d or 1).toordinal() * 24 + h
        except ValueError:
            ordinal = 0
        result.append(([y, m, d, h], ordinal))
    return result


def features(premise, hypothesis):
    a, b = (unicodedata.normalize("NFKC", s) for s in (premise, hypothesis))
    p, h = stamps(a), stamps(b)
    vals = [len(p), len(h), a == b]
    for s in (a, b):
        vals.extend(s.count(k) for k in KEYS)
    vals.extend(
        [
            SequenceMatcher(None, a, b, autojunk=False).ratio(),
            len(set(a) & set(b)) / max(1, len(set(b))),
        ]
    )
    missing = ([0] * 4, 0)
    x, y, z, w = (p + [missing] * 2)[:2] + (h + [missing] * 2)[:2]
    for lhs, rhs in ((x, z), (y, z), (x, y), (z, w), (y, w)):
        vals.extend(np.sign(np.array(lhs[0]) - rhs[0]).tolist())
        vals.extend([np.sign(lhs[1] - rhs[1]), lhs == rhs])
    # Compare elapsed calendar units with the stated duration.
    for i in range(4):
        a0, a1, b0, b1 = (v[0][i] for v in (x, y, z, w))
        vals.extend(
            (np.sign(a1 - a0 - b0), np.sign(b0 - a1 - a0), np.sign(b1 - b0 - a0))
        )
    return np.asarray(vals, dtype=float)


def probabilities(model, premise, hypothesis, laya):
    v = features(premise, hypothesis)
    total = np.zeros(3)
    for tree in model["trees"]:
        i = 0
        while tree[i][0] >= 0:
            j, pivot, left, right = tree[i]
            i = left if v[j] <= pivot else right
        total += tree[i][1:]
    total /= len(model["trees"])
    return (1 - model["laya_weight"]) * total + model["laya_weight"] * np.array(laya)
