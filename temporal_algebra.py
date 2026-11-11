"""Derive interval containment and contradiction from Japanese temporal expressions."""

import calendar
import datetime
import math
import re
import unicodedata

from temporal_relations import FIELD, TIME, UNITS

DURATION = re.compile(r"(\d+)(年(?:間)|(?:か|ヶ)?月間|日間|時間)")
NAMES = re.compile(r"[ァ-ヴー]{2,}")
LABEL = {"yes": 0, "no": 1, "unknown": 2}


def bound(n, suffix, subset=False):
    if "以内" in suffix or "まで" in suffix or subset:
        return 0, n
    if "以上" in suffix or "以降" in suffix or "以後" in suffix:
        return n, math.inf
    return n, n


def compare(left, right):
    lo, hi = left
    a, b = right
    if a <= lo and hi <= b:
        return LABEL["yes"]
    if hi < a or b < lo:
        return LABEL["no"]
    return LABEL["unknown"]


def moment(text):
    vals = {u: int(v) for v, u in FIELD.findall(text) if u in UNITS}
    return datetime.datetime(
        vals.get("年", 2000), vals.get("月", 1), vals.get("日", 1), vals.get("時", 0)
    )


def elapsed(start, end, unit):
    a, b = moment(start), moment(end)
    if unit == "年":
        return b.year - a.year
    if unit == "月":
        return (b.year - a.year) * 12 + b.month - a.month
    return (b - a).total_seconds() / (86400 if unit == "日" else 3600)


def decide(premise, hypothesis):
    p, h = (unicodedata.normalize("NFKC", s) for s in (premise, hypothesis))
    # This parser only resolves relations about the same actor.
    names_p, names_h = NAMES.findall(p), NAMES.findall(h)
    if not names_p or not names_h or names_p[0] != names_h[0]:
        return None
    dp, dh = DURATION.search(p), DURATION.search(h)
    if dp and dh:
        if (
            dp[2].replace("か", "").replace("ヶ", "")[0]
            != dh[2].replace("か", "").replace("ヶ", "")[0]
        ):
            return None
        a = bound(
            int(dp[1]), p[dp.end() : dp.end() + 3], "多く" in p and "多く" not in h
        )
        b = bound(int(dh[1]), h[dh.end() : dh.end() + 3])
        return compare(a, b)
    times_p, times_h = list(TIME.finditer(p)), list(TIME.finditer(h))
    if dp and len(times_p) == 2 and len(times_h) == 1 and ("始め" in p or "終え" in p):
        unit = dp[2].replace("か", "").replace("ヶ", "")[0]
        date = next((x for x in times_p if x.start() != dp.start()), None)
        if date and unit in UNITS:
            n = elapsed(date[0], times_h[0][0], unit)
            if "終え" in p and "始め" in h:
                n = -n
            return compare(
                bound(int(dp[1]), p[dp.end() : dp.end() + 3]),
                bound(n, h[times_h[0].end() : times_h[0].end() + 4]),
            )
    if len(times_p) == len(times_h) == 1 and any(
        x in p
        for x in (
            "昨日",
            "一昨日",
            "明日",
            "翌日",
            "前日",
            "先月",
            "来月",
            "去年",
            "昨年",
            "来年",
        )
    ):
        base, goal = moment(times_p[0][0]), moment(times_h[0][0])
        offs = {"一昨日": -2, "昨日": -1, "前日": -1, "明日": 1, "翌日": 1}
        for word, d in offs.items():
            if word in p:
                return (
                    LABEL["yes"]
                    if base + datetime.timedelta(days=d) == goal
                    else LABEL["unknown"]
                )
        months = (
            -12
            if "昨年" in p or "去年" in p
            else 12
            if "来年" in p
            else -1
            if "先月" in p
            else 1
        )
        ym = base.year * 12 + base.month - 1 + months
        y, m = divmod(ym, 12)
        target = base.replace(
            year=y, month=m + 1, day=min(base.day, calendar.monthrange(y, m + 1)[1])
        )
        return LABEL["yes"] if target == goal else LABEL["unknown"]
    return None


def with_algebra(base, p, h):
    choice = decide(p, h)
    if choice is None:
        return base
    raw = 0.2 * base.copy()
    raw[choice] += 0.8
    return raw


if __name__ == "__main__":
    import argparse
    import gzip
    import json
    from pathlib import Path

    from temporal_relations import probabilities

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path(".cache/four"))
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument(
        "--result", type=Path, default=Path("results/four-intervals.json")
    )
    args = parser.parse_args()
    reports = {}
    for variant in ("temporal", "temporal-hard"):
        tree = json.loads(
            gzip.decompress((args.weights / (variant + ".json.gz")).read_bytes())
        )
        for name in ("jamp-dev", "jamp"):
            rows = list(
                map(
                    json.loads,
                    (args.inputs / (name + ".jsonl")).read_text().splitlines(),
                )
            )
            base = parser_hits = cover = 0
            for r in rows:
                p, h = r["text"], r["options"][0]
                z = probabilities(tree, p, h, r["laya"])
                base += int(z.argmax() == r["gold"])
                parser_hits += int(with_algebra(z, p, h).argmax() == r["gold"])
                cover += int(decide(p, h) is not None)
            reports[variant + "/" + name] = {
                "total": len(rows),
                "base": base,
                "with_interval_parser": parser_hits,
                "covered": cover,
            }
    args.result.write_text(json.dumps(reports, indent=2) + "\n")
