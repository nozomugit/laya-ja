"""Resolve time relations with an event-order graph and explicit calendar bounds."""

import re
import unicodedata

import numpy as np

from temporal_algebra import DURATION, decide, moment
from temporal_relations import TIME

ACTOR = re.compile(r"([ァ-ヴー]{2,})(?:が|は|より)")
ORDER = re.compile(r"(?:より(先|後)に|(前|後)(?:に|で))")


def edge(sentence):
    actors = ACTOR.findall(sentence)
    order = ORDER.search(sentence)
    if len(actors) != 2 or actors[0] == actors[1] or not order:
        return None
    return tuple(actors if (order[1] or order[2]) in ("先", "前") else reversed(actors))


def reachable(edges, source, target):
    seen = {source}
    pending = [source]
    while pending:
        current = pending.pop()
        for a, b in edges:
            if a == current and b not in seen:
                if b == target:
                    return True
                seen.add(b)
                pending.append(b)
    return False


def _resolve(premise, hypothesis):
    p, h = (unicodedata.normalize("NFKC", t) for t in (premise, hypothesis))
    want = edge(h)
    if want:
        graph = [e for s in p.split("。") if (e := edge(s))]
        if graph:
            f = reachable(graph, *want)
            r = reachable(graph, *reversed(want))
            if f != r:
                return "order", 0 if f else 1
            if not f and not r:
                return "order", 2
    actors_p, actors_h = ACTOR.findall(p), ACTOR.findall(h)
    if not actors_p or not actors_h or actors_p[0] != actors_h[0]:
        return None
    dp, dh = DURATION.search(p), DURATION.search(h)
    tp, th = TIME.findall(p), TIME.findall(h)
    if (
        len(tp) == 2
        and len(th) == 1
        and "以来" in p
        and "現在" in p
        and not dp
        and not dh
    ):
        start, now, at = map(moment, [*tp, th[0]])
        return "since", 1 if at < start else 2 if at > now else 0
    if len(tp) == len(th) == 1 and not dp and not dh:
        if ("いつも" in p or "決して" in p) and "遅れ" in h and "遅れ" in p:
            same = ("遅れず" in h or "遅れな" in h) == ("決して" in p)
            return "habit", (0 if same else 1) if moment(tp[0]) == moment(th[0]) else 2
        if not any(
            w in p + h
            for w in (
                "昨日",
                "明日",
                "先月",
                "来月",
                "昨年",
                "来年",
                "毎",
                "以",
                "前",
                "後",
            )
        ):
            if moment(tp[0]) != moment(th[0]):
                return "unmentioned", 2
            if "それ" in h or p == h:
                return "coreference", 0
    if "存在する" in h and any(w in p for w in ("毎日", "毎年", "毎月", "毎週")):
        return "existence", 0
    if dp or dh:
        typ = "duration-pair" if dp and dh else "duration-date"
        decision = decide(p, h)
        if decision is not None:
            return typ, decision
    if any(w in p for w in ("昨日", "明日", "昨年", "去年", "来年", "先月", "来月")):
        decision = decide(p, h)
        if decision is not None:
            return "relative-date", decision
    return None


def resolve(premise, hypothesis):
    try:
        return _resolve(premise, hypothesis)
    except (ValueError, OverflowError):
        # Unparseable calendar dates leave the statistical readout in charge.
        return None


def predict(base, premise, hypothesis, supported):
    finding = resolve(premise, hypothesis)
    if finding is None or finding[0] not in supported:
        return base
    scores = np.asarray(base) * 0.25
    scores[finding[1]] += 0.75
    return scores
