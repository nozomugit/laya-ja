"""Japanese inference problems at scale, each labeled by checking all 512 states of three people and three properties.

The sentence frames extend semantic_worlds.py. The names and properties are disjoint from its calibration
problems, and the three splits here (train, tune, held) use disjoint vocabularies.
"""

import hashlib
import itertools
import random

WORLDS = 512
ALL = (1 << WORLDS) - 1

# (affirmative, negative, quoted label)
PROPERTIES = {
    "train": [
        ("会員である", "会員ではない", "会員である"),
        ("出席している", "出席していない", "出席している"),
        ("名簿に載っている", "名簿に載っていない", "名簿に載っている"),
        ("傘を持っている", "傘を持っていない", "傘を持っている"),
        ("切符を買った", "切符を買わなかった", "切符を買った"),
        ("試験に合格した", "試験に合格しなかった", "試験に合格した"),
        ("昼食を済ませた", "昼食を済ませていない", "昼食を済ませた"),
        ("帽子をかぶっている", "帽子をかぶっていない", "帽子をかぶっている"),
        ("申込書を出した", "申込書を出していない", "申込書を出した"),
        ("眼鏡をかけている", "眼鏡をかけていない", "眼鏡をかけている"),
        ("委員である", "委員ではない", "委員である"),
        ("手袋をしている", "手袋をしていない", "手袋をしている"),
        ("会場に着いた", "会場に着いていない", "会場に着いた"),
        ("返事を書いた", "返事を書いていない", "返事を書いた"),
        ("東京に住んでいる", "東京に住んでいない", "東京に住んでいる"),
        ("自転車で来た", "自転車で来なかった", "自転車で来た"),
        ("朝に走った", "朝に走らなかった", "朝に走った"),
        ("部屋を予約した", "部屋を予約していない", "部屋を予約した"),
    ],
    "tune": [
        ("係員である", "係員ではない", "係員である"),
        ("名刺を持っている", "名刺を持っていない", "名刺を持っている"),
        ("料金を払った", "料金を払っていない", "料金を払った"),
        ("講習を受けた", "講習を受けていない", "講習を受けた"),
        ("席に座っている", "席に座っていない", "席に座っている"),
        ("荷物を預けた", "荷物を預けていない", "荷物を預けた"),
    ],
    "held": [
        ("司会である", "司会ではない", "司会である"),
        ("時計をしている", "時計をしていない", "時計をしている"),
        ("署名を済ませた", "署名を済ませていない", "署名を済ませた"),
        ("写真を撮った", "写真を撮らなかった", "写真を撮った"),
        ("列に並んでいる", "列に並んでいない", "列に並んでいる"),
        ("資料を読んだ", "資料を読んでいない", "資料を読んだ"),
    ],
}
PEOPLE = {
    "train": "石田 松本 藤井 岡田 中島 前田 小川 後藤 長谷川 村上 近藤 坂本 遠藤 青木 藤田 西村 福田 太田 三浦 岡本 松田 中川 中野 原田".split(),
    "tune": "安藤 河野 丸山 今井 高木 武田".split(),
    "held": "上野 杉山 千葉 増田 小島 桑原".split(),
}
OPENINGS = (
    "対象は{0}、{1}、{2}の三人だけである。",
    "ここでは{0}、{1}、{2}の三人について考える。ほかの人は関係しない。",
    "{0}、{1}、{2}の三人だけが調べられた。",
)


def _table(fn):
    return sum(1 << w for w in range(WORLDS) if fn(w))


def _has(w, i, j):
    return bool(w & (1 << (3 * i + j)))


def _count(w, j):
    return sum(_has(w, i, j) for i in range(3))


def frames():
    """Statement frames: (renderers, truth table over the 512 states, properties mentioned)."""
    out = []

    def add(texts, fn, symbols):
        out.append((texts, _table(fn), frozenset(symbols)))

    for i in range(3):
        for j in range(3):
            add(
                [lambda n, p, i=i, j=j: f"{n[i]}は{p[j][0]}。"],
                lambda w, i=i, j=j: _has(w, i, j),
                [j],
            )
            add(
                [
                    lambda n, p, i=i, j=j: f"{n[i]}は{p[j][1]}。",
                    lambda n, p, i=i, j=j: f"{n[i]}が「{p[j][2]}」というのは事実ではない。",
                ],
                lambda w, i=i, j=j: not _has(w, i, j),
                [j],
            )
    for j in range(3):
        for k in (0, 1, 2, 3):
            add(
                [
                    lambda n, p, j=j, k=k: f"「{p[j][2]}」に当てはまるのはちょうど{k}人である。",
                    lambda n, p, j=j, k=k: f"三人のうち「{p[j][2]}」人の数は{k}である。",
                ],
                lambda w, j=j, k=k: _count(w, j) == k,
                [j],
            )
        for k in (1, 2):
            add(
                [lambda n, p, j=j, k=k: f"少なくとも{k}人は{p[j][0]}。"],
                lambda w, j=j, k=k: _count(w, j) >= k,
                [j],
            )
            add(
                [
                    lambda n, p, j=j, k=k: f"「{p[j][2]}」に当てはまるのは多くても{k}人である。"
                ],
                lambda w, j=j, k=k: _count(w, j) <= k,
                [j],
            )
        add(
            [
                lambda n, p, j=j: f"三人とも{p[j][0]}。",
                lambda n, p, j=j: f"三人の全員が「{p[j][2]}」に当てはまる。",
            ],
            lambda w, j=j: _count(w, j) == 3,
            [j],
        )
        add(
            [lambda n, p, j=j: f"三人の中に「{p[j][2]}」に当てはまる人はいない。"],
            lambda w, j=j: _count(w, j) == 0,
            [j],
        )
        for a, b in itertools.combinations(range(3), 2):
            add(
                [
                    lambda n, p, j=j, a=a, b=b: f"{n[a]}と{n[b]}の少なくとも一方は{p[j][0]}。"
                ],
                lambda w, j=j, a=a, b=b: _has(w, a, j) or _has(w, b, j),
                [j],
            )
            add(
                [lambda n, p, j=j, a=a, b=b: f"{n[a]}も{n[b]}も{p[j][0]}。"],
                lambda w, j=j, a=a, b=b: _has(w, a, j) and _has(w, b, j),
                [j],
            )
            add(
                [lambda n, p, j=j, a=a, b=b: f"{n[a]}も{n[b]}も{p[j][1]}。"],
                lambda w, j=j, a=a, b=b: not _has(w, a, j) and not _has(w, b, j),
                [j],
            )
        for i in range(3):
            add(
                [
                    lambda n, p, j=j, i=i: f"「{p[j][2]}」に当てはまるのは{n[i]}だけである。"
                ],
                lambda w, j=j, i=i: all(_has(w, x, j) == (x == i) for x in range(3)),
                [j],
            )
    for j, k in itertools.permutations(range(3), 2):
        add(
            [
                lambda n, p, j=j, k=k: f"「{p[j][2]}」に当てはまる人は全員「{p[k][2]}」にも当てはまる。",
                lambda n, p, j=j, k=k: f"「{p[j][2]}」人は、必ず「{p[k][2]}」人でもある。",
            ],
            lambda w, j=j, k=k: all(
                not _has(w, i, j) or _has(w, i, k) for i in range(3)
            ),
            [j, k],
        )
        add(
            [
                lambda n, p, j=j, k=k: f"「{p[j][2]}」と「{p[k][2]}」の両方に当てはまる人がいる。"
            ],
            lambda w, j=j, k=k: any(_has(w, i, j) and _has(w, i, k) for i in range(3)),
            [j, k],
        )
        add(
            [
                lambda n, p, j=j, k=k: f"「{p[j][2]}」と「{p[k][2]}」の両方に当てはまる人はいない。"
            ],
            lambda w, j=j, k=k: not any(
                _has(w, i, j) and _has(w, i, k) for i in range(3)
            ),
            [j, k],
        )
        for i in range(3):
            add(
                [
                    lambda n, p, j=j, k=k, i=i: f"{n[i]}が「{p[j][2]}」に当てはまるなら、{n[i]}は「{p[k][2]}」にも当てはまる。"
                ],
                lambda w, j=j, k=k, i=i: not _has(w, i, j) or _has(w, i, k),
                [j, k],
            )
    return out


def generate(split, count, seed):
    """Balanced (premise, hypothesis, label, identifier) rows; label 0 entailed, 1 contradicted, 2 undetermined."""
    if count % 3:
        raise ValueError("count must be divisible by three")
    rng = random.Random(seed)
    statements = frames()
    quota = [count // 3] * 3
    seen, out = set(), []
    while any(quota):
        size = rng.choice((1, 2, 2, 3))
        *premises, hypothesis = rng.sample(statements, size + 1)
        if not any(v[2] & hypothesis[2] for v in premises):
            continue
        bits = ALL
        for v in premises:
            bits &= v[1]
        if not bits:
            continue
        label = (
            0 if bits & ~hypothesis[1] == 0 else 1 if bits & hypothesis[1] == 0 else 2
        )
        if not quota[label]:
            continue
        names = rng.sample(PEOPLE[split], 3)
        props = rng.sample(PROPERTIES[split], 3)
        body = "".join(rng.choice(v[0])(names, props) for v in premises)
        premise = rng.choice(OPENINGS).format(*names) + body
        text = rng.choice(hypothesis[0])(names, props)
        ident = hashlib.sha256((premise + "\n" + text).encode()).hexdigest()[:24]
        if ident in seen:
            continue
        seen.add(ident)
        quota[label] -= 1
        out.append((premise, text, label, ident))
    return out
