"""Japanese inference problems in contrast sets: each sentence pattern yields entailed, contradicted and undetermined cases.

A model trained on corpora of minimally edited pairs can learn that a small difference means contradiction. Here
the same patterns are paired with hypotheses of all three labels, so the wording alone does not give the label away.
Labels of the counting, membership, ordering and conditional families come from checking every situation; the
others follow a fixed rule of the construction (a stated fact, a report, a mistaken belief; an attempt, a failure,
a completion; a paraphrase, an opposite, an unrelated property; a correction). Vocabulary differs between the
training, tuning and held-out splits. No sentence or name is taken from an evaluation corpus.
"""

import hashlib
import itertools
import random

PEOPLE = {
    "train": [
        "田中さん",
        "鈴木さん",
        "高橋さん",
        "渡辺さん",
        "伊藤さん",
        "山本さん",
        "中村さん",
        "小林さん",
        "加藤さん",
        "吉川さん",
    ],
    "tune": ["吉田さん", "山田さん", "佐々木さん", "山口さん"],
    "held": ["松本さん", "井上さん", "木村さん", "林さん", "清水さん"],
}
# (did, did not)
DEEDS = {
    "train": [
        ("来た", "来なかった"),
        ("合格した", "合格しなかった"),
        ("発言した", "発言しなかった"),
        ("欠席した", "欠席しなかった"),
        ("賛成した", "賛成しなかった"),
        ("残業した", "残業しなかった"),
        ("入賞した", "入賞しなかった"),
        ("遅刻した", "遅刻しなかった"),
    ],
    "tune": [("当選した", "当選しなかった"), ("帰宅した", "帰宅しなかった"), ("出場した", "出場しなかった")],
    "held": [
        ("到着した", "到着しなかった"),
        ("辞退した", "辞退しなかった"),
        ("登壇した", "登壇しなかった"),
        ("完走した", "完走しなかった"),
    ],
}
# (what is counted, counter)
COUNTED = {
    "train": [
        ("参加者", "人"),
        ("応募", "件"),
        ("欠席者", "人"),
        ("注文", "件"),
        ("空席", "席"),
        ("返品", "件"),
        ("来場者", "人"),
        ("届いた手紙", "通"),
    ],
    "tune": [("合格者", "人"), ("問い合わせ", "件"), ("貸し出した本", "冊")],
    "held": [("立候補者", "人"), ("苦情", "件"), ("売れた切符", "枚"), ("当選者", "人")],
}
EVENTS = {
    "train": ["開会式", "講演", "休憩", "昼食", "表彰", "見学", "討論", "演奏", "抽選", "挨拶"],
    "tune": ["点呼", "説明会", "試食", "記念撮影"],
    "held": ["乾杯", "上映", "質疑", "閉会式", "朗読"],
}
# (state holds, state does not hold); plain forms, usable before こと and 場合
STATES = {
    "train": [
        ("店は閉まっている", "店は閉まっていない"),
        ("ランプが点いている", "ランプが点いていない"),
        ("扉に鍵がかかっている", "扉に鍵がかかっていない"),
        ("道路は凍っている", "道路は凍っていない"),
        ("会議は延期になった", "会議は延期にならなかった"),
        ("荷物は届いている", "荷物は届いていない"),
        ("窓が開いている", "窓が開いていない"),
        ("電車は遅れている", "電車は遅れていない"),
        ("在庫は残っている", "在庫は残っていない"),
        ("橋は通行止めになっている", "橋は通行止めになっていない"),
    ],
    "tune": [
        ("暖房が入っている", "暖房が入っていない"),
        ("受付は始まっている", "受付は始まっていない"),
        ("警報が鳴っている", "警報が鳴っていない"),
    ],
    "held": [
        ("エレベーターは動いている", "エレベーターは動いていない"),
        ("駐車場は空いている", "駐車場は空いていない"),
        ("試合は中止になった", "試合は中止にならなかった"),
        ("水道は止まっている", "水道は止まっていない"),
    ],
}
# did, did not, meant to, tried to, had to, failed to, finished, managed to, could not
EFFORTS = {
    "train": [
        (
            "書類を提出した",
            "書類を提出しなかった",
            "書類を提出するつもりだった",
            "書類を提出しようとした",
            "書類を提出しなければならなかった",
            "書類を提出しそこねた",
            "書類を提出し終えた",
            "書類を提出することができた",
            "書類を提出できなかった",
        ),
        (
            "手紙を書いた",
            "手紙を書かなかった",
            "手紙を書くつもりだった",
            "手紙を書こうとした",
            "手紙を書かなければならなかった",
            "手紙を書きそこねた",
            "手紙を書き終えた",
            "手紙を書くことができた",
            "手紙を書けなかった",
        ),
        (
            "薬を飲んだ",
            "薬を飲まなかった",
            "薬を飲むつもりだった",
            "薬を飲もうとした",
            "薬を飲まなければならなかった",
            "薬を飲み忘れた",
            "薬を飲み終えた",
            "薬を飲むことができた",
            "薬を飲めなかった",
        ),
        (
            "部屋を片づけた",
            "部屋を片づけなかった",
            "部屋を片づけるつもりだった",
            "部屋を片づけようとした",
            "部屋を片づけなければならなかった",
            "部屋を片づけそこねた",
            "部屋を片づけ終えた",
            "部屋を片づけることができた",
            "部屋を片づけられなかった",
        ),
        (
            "料金を払った",
            "料金を払わなかった",
            "料金を払うつもりだった",
            "料金を払おうとした",
            "料金を払わなければならなかった",
            "料金を払い忘れた",
            "料金を払い終えた",
            "料金を払うことができた",
            "料金を払えなかった",
        ),
        (
            "本を返した",
            "本を返さなかった",
            "本を返すつもりだった",
            "本を返そうとした",
            "本を返さなければならなかった",
            "本を返し忘れた",
            "本を返し終えた",
            "本を返すことができた",
            "本を返せなかった",
        ),
        (
            "窓を閉めた",
            "窓を閉めなかった",
            "窓を閉めるつもりだった",
            "窓を閉めようとした",
            "窓を閉めなければならなかった",
            "窓を閉め忘れた",
            "窓を閉め終えた",
            "窓を閉めることができた",
            "窓を閉められなかった",
        ),
        (
            "山に登った",
            "山に登らなかった",
            "山に登るつもりだった",
            "山に登ろうとした",
            "山に登らなければならなかった",
            "山に登りそこねた",
            "山に登り終えた",
            "山に登ることができた",
            "山に登れなかった",
        ),
    ],
    "tune": [
        (
            "皿を洗った",
            "皿を洗わなかった",
            "皿を洗うつもりだった",
            "皿を洗おうとした",
            "皿を洗わなければならなかった",
            "皿を洗い忘れた",
            "皿を洗い終えた",
            "皿を洗うことができた",
            "皿を洗えなかった",
        ),
        (
            "切符を買った",
            "切符を買わなかった",
            "切符を買うつもりだった",
            "切符を買おうとした",
            "切符を買わなければならなかった",
            "切符を買いそこねた",
            "切符を買い終えた",
            "切符を買うことができた",
            "切符を買えなかった",
        ),
    ],
    "held": [
        (
            "原稿を送った",
            "原稿を送らなかった",
            "原稿を送るつもりだった",
            "原稿を送ろうとした",
            "原稿を送らなければならなかった",
            "原稿を送り忘れた",
            "原稿を送り終えた",
            "原稿を送ることができた",
            "原稿を送れなかった",
        ),
        (
            "川を渡った",
            "川を渡らなかった",
            "川を渡るつもりだった",
            "川を渡ろうとした",
            "川を渡らなければならなかった",
            "川を渡りそこねた",
            "川を渡り終えた",
            "川を渡ることができた",
            "川を渡れなかった",
        ),
        (
            "花に水をやった",
            "花に水をやらなかった",
            "花に水をやるつもりだった",
            "花に水をやろうとした",
            "花に水をやらなければならなかった",
            "花に水をやり忘れた",
            "花に水をやり終えた",
            "花に水をやることができた",
            "花に水をやれなかった",
        ),
    ],
}
# topic, stated, same meaning, opposite, unrelated
WORDS = {
    "train": [
        ("この部屋は", "広い", "広々としている", "狭い", "明るい"),
        ("その映画は", "退屈だった", "つまらなかった", "面白かった", "長かった"),
        ("今朝の気温は", "低かった", "高くなかった", "高かった", "昨日と同じだった"),
        ("その店は", "混んでいた", "混雑していた", "空いていた", "新しかった"),
        ("この道は", "安全だ", "危険ではない", "危険だ", "まっすぐだ"),
        ("その説明は", "正しかった", "間違っていなかった", "間違っていた", "短かった"),
        ("会場は", "静かだった", "騒がしくなかった", "騒がしかった", "広かった"),
        ("その荷物は", "重い", "軽くない", "軽い", "大きい"),
        ("彼の返事は", "早かった", "遅くなかった", "遅かった", "丁寧だった"),
        ("この問題は", "簡単だ", "難しくない", "難しい", "有名だ"),
        ("そのスープは", "熱かった", "冷たくなかった", "冷たかった", "辛かった"),
        ("駅までの道は", "遠い", "近くない", "近い", "暗い"),
        ("その計画は", "成功した", "失敗しなかった", "失敗した", "先月始まった"),
        ("図書館は今日", "開いている", "閉まっていない", "閉まっている", "混んでいる"),
        ("その箱は", "空だった", "何も入っていなかった", "物でいっぱいだった", "木でできていた"),
        ("選手は全員", "無事だった", "けがをしなかった", "けがをした", "若かった"),
    ],
    "tune": [
        ("その川は", "深い", "浅くない", "浅い", "長い"),
        ("昨日の会議は", "短かった", "長くなかった", "長かった", "有意義だった"),
        ("この靴は", "新しい", "古くない", "古い", "黒い"),
    ],
    "held": [
        ("その橋は", "丈夫だ", "壊れやすくない", "壊れやすい", "赤い"),
        ("このパンは", "硬い", "やわらかくない", "やわらかい", "甘い"),
        ("その試合は", "中止になった", "行われなかった", "予定どおり行われた", "雨の日に予定されていた"),
        ("廊下は", "明るかった", "暗くなかった", "暗かった", "長かった"),
        ("その機械は", "正常に動いている", "故障していない", "故障している", "去年買われた"),
    ],
}
ITEMS = {
    "train": [
        ["紅茶", "コーヒー", "緑茶", "ココア"],
        ["青い傘", "黒い傘", "赤い傘"],
        ["窓側の席", "通路側の席", "最前列の席"],
        ["魚料理", "肉料理", "野菜料理"],
        ["朝の便", "昼の便", "夜の便"],
    ],
    "tune": [["白い皿", "青い皿", "木の皿"], ["大きい箱", "小さい箱", "中くらいの箱"]],
    "held": [
        ["麦茶", "牛乳", "炭酸水"],
        ["一階の部屋", "二階の部屋", "三階の部屋"],
        ["銀のペン", "金のペン", "木のペン"],
    ],
}
CHOSE = [("選んだ", "選ばなかった"), ("注文した", "注文しなかった"), ("予約した", "予約しなかった")]


def _inner(clause):
    """The clause as it reads inside a larger sentence: its topic particle becomes a subject particle."""
    return clause.replace("は", "が", 1)


def _label(worlds, premises, hypothesis):
    kept = [w for w in worlds if all(p(w) for p in premises)]
    if not kept:
        return None
    hits = sum(bool(hypothesis(w)) for w in kept)
    return 0 if hits == len(kept) else 1 if hits == 0 else 2


def counting(split, rng):
    thing, unit = rng.choice(COUNTED[split])

    def statement():
        kind, k = rng.choice(["exactly", "at_least", "at_most", "more", "fewer"]), rng.randint(2, 12)
        text = {
            "exactly": f"{thing}はちょうど{k}{unit}だった。",
            "at_least": rng.choice([f"{thing}は{k}{unit}以上だった。", f"{thing}は少なくとも{k}{unit}だった。"]),
            "at_most": rng.choice([f"{thing}は{k}{unit}以下だった。", f"{thing}は多くても{k}{unit}だった。"]),
            "more": f"{thing}は{k}{unit}より多かった。",
            "fewer": f"{thing}は{k}{unit}未満だった。",
        }[kind]
        test = {
            "exactly": lambda n: n == k,
            "at_least": lambda n: n >= k,
            "at_most": lambda n: n <= k,
            "more": lambda n: n > k,
            "fewer": lambda n: n < k,
        }[kind]
        return text, test

    (p, pt), (h, ht) = statement(), statement()
    return p, h, None if p == h else _label(range(40), [pt], ht)


def membership(split, rng):
    a, b = rng.sample(PEOPLE[split], 2)
    did, not_did = rng.choice(DEEDS[split])
    worlds = list(itertools.product((False, True), repeat=2))
    premises = [
        (f"{a}と{b}のうち、{did}のは{a}だけだ。", lambda w: w[0] and not w[1]),
        (f"{a}も{b}も{did}。", lambda w: w[0] and w[1]),
        (f"{a}も{b}も{not_did}。", lambda w: not w[0] and not w[1]),
        (f"{a}は{did}。", lambda w: w[0]),
        (f"{a}は{not_did}。", lambda w: not w[0]),
        (f"{a}と{b}の少なくとも一方は{did}。", lambda w: w[0] or w[1]),
        (f"{a}と{b}のどちらか一人だけが{did}。", lambda w: w[0] != w[1]),
        (f"{a}と{b}のどちらか一人だけが{did}。{a}は{not_did}。", lambda w: w[0] != w[1] and not w[0]),
        (f"{a}と{b}の少なくとも一方は{did}。{a}は{not_did}。", lambda w: (w[0] or w[1]) and not w[0]),
    ]
    hypotheses = [
        (f"{b}は{did}。", lambda w: w[1]),
        (f"{b}は{not_did}。", lambda w: not w[1]),
        (f"{a}は{did}。", lambda w: w[0]),
        (f"{a}は{not_did}。", lambda w: not w[0]),
        (f"{a}と{b}は二人とも{did}。", lambda w: w[0] and w[1]),
        (f"{a}と{b}の少なくとも一方は{did}。", lambda w: w[0] or w[1]),
        (f"{a}と{b}のうち{did}人はいない。", lambda w: not w[0] and not w[1]),
    ]
    (p, pt), (h, ht) = rng.choice(premises), rng.choice(hypotheses)
    return p, h, None if h in p else _label(worlds, [pt], ht)


def ordering(split, rng):
    e = rng.sample(EVENTS[split], 3)
    worlds = list(itertools.permutations(range(3)))  # w[i] = position of event i

    def statement():
        i, j = rng.sample(range(3), 2)
        kind = rng.choice(["before", "after", "first", "last", "between"])
        if kind == "before":
            return (
                rng.choice([f"{e[i]}は{e[j]}より前に行われた。", f"{e[i]}のあとで{e[j]}が行われた。"]),
                lambda w: w[i] < w[j],
            )
        if kind == "after":
            return (
                rng.choice([f"{e[i]}は{e[j]}より後に行われた。", f"{e[i]}の前に{e[j]}が行われた。"]),
                lambda w: w[i] > w[j],
            )
        if kind == "first":
            return f"三つのうち最初に行われたのは{e[i]}だ。", lambda w: w[i] == 0
        if kind == "last":
            return f"三つのうち最後に行われたのは{e[i]}だ。", lambda w: w[i] == 2
        return f"{e[i]}は三つのうち二番目に行われた。", lambda w: w[i] == 1

    parts = [statement() for _ in range(rng.choice((1, 2)))]
    h, ht = statement()
    opening = f"{e[0]}、{e[1]}、{e[2]}は一つずつ順に行われた。"
    premise = opening + "".join(t for t, _ in parts)
    return premise, h, None if h in premise else _label(worlds, [t for _, t in parts], ht)


def report(split, rng):
    yes, no = rng.choice(STATES[split])
    name = rng.choice(PEOPLE[split])
    kind = rng.choice(["said", "fact", "false"])
    y = _inner(yes)
    premise = {
        "said": [
            f"{name}は「{yes}」と言った。",
            f"{name}は{y}と思っている。",
            f"{y}という噂がある。",
            f"{name}は{y}かどうかを尋ねた。",
        ],
        "fact": [
            f"{name}は{y}ことを知っている。",
            f"{name}は{y}ことに気づいた。",
            f"{name}は{y}ことを確かめた。",
        ],
        "false": [
            f"{y}というのは誤りだった。",
            f"{name}は{y}と勘違いしていた。",
            f"{name}は{y}と思っていたが、実際は違った。",
        ],
    }[kind]
    positive = rng.random() < 0.5
    label = {"said": 2, "fact": 0 if positive else 1, "false": 1 if positive else 0}[kind]
    return rng.choice(premise), (yes if positive else no) + "。", label


def conditional(split, rng):
    (p, np_), (q, nq) = rng.sample(STATES[split], 2)
    worlds = list(itertools.product((False, True), repeat=2))
    ip, iq = _inner(p), _inner(q)
    rule = rng.choice([f"{ip}場合は、必ず{q}。", f"{ip}のは、{iq}場合だけだ。", f"{iq}のでなければ、{ip}ことはない。"])
    facts = [
        (f"{p}。", lambda w: w[0]),
        (f"{np_}。", lambda w: not w[0]),
        (f"{q}。", lambda w: w[1]),
        (f"{nq}。", lambda w: not w[1]),
    ]
    fact = rng.choice([None, *facts, *facts])
    h, ht = rng.choice(facts)
    tests = [lambda w: (not w[0]) or w[1]] + ([fact[1]] if fact else [])
    premise = rule + (fact[0] if fact else "")
    return premise, h, None if h in premise else _label(worlds, tests, ht)


def effort(split, rng):
    did, not_did, meant, tried, had_to, failed, finished, managed, could_not = rng.choice(EFFORTS[split])
    name = rng.choice(PEOPLE[split])
    kind = rng.choice(["open", "open", "done", "undone"])
    verb = {
        "open": rng.choice([meant, tried, had_to]),
        "done": rng.choice([finished, managed]),
        "undone": rng.choice([failed, could_not]),
    }[kind]
    positive = rng.random() < 0.5
    label = {"open": 2, "done": 0 if positive else 1, "undone": 1 if positive else 0}[kind]
    return f"{name}は{verb}。", f"{name}は{did if positive else not_did}。", label


def wording(split, rng):
    topic, stated, same, opposite, other = rng.choice(WORDS[split])
    label = rng.randrange(3)
    opening = rng.choice(
        ["", "調べたところ、", "予想に反して、", "見たかぎりでは、", "記録によると、", "案内のとおり、"]
    )
    return f"{opening}{topic}{stated}。", f"{topic}{(same, opposite, other)[label]}。", label


def correction(split, rng):
    a, b, c = rng.sample(rng.choice(ITEMS[split]), 3)  # like for like
    name = rng.choice(PEOPLE[split])
    did, not_did = rng.choice(CHOSE)
    if rng.random() < 0.5:
        premise = f"{name}は{a}ではなく{b}を{did}。"
        options = [
            (f"{name}は{b}を{did}。", 0),
            (f"{name}は{a}を{did}。", 1),
            (f"{name}は{a}を{not_did}。", 0),
            (f"{name}は{b}を{not_did}。", 1),
            (f"{name}は以前に{c}を{did}ことがある。", 2),
            (f"{name}は{a}が嫌いだ。", 2),
        ]
    else:
        premise = f"{name}は最初は{a}を{did}が、あとで{b}に変えた。"
        options = [
            (f"{name}が最終的に{did}のは{b}だ。", 0),
            (f"{name}が最終的に{did}のは{a}だ。", 1),
            (f"{name}が最初に{did}のは{a}だ。", 0),
            (f"{name}が最初に{did}のは{b}だ。", 1),
            (f"{name}は{c}も検討した。", 2),
            (f"{name}は{b}のほうが安いと考えた。", 2),
        ]
    h, label = rng.choice(options)
    return premise, h, label


FAMILIES = {
    "counting": counting,
    "membership": membership,
    "ordering": ordering,
    "report": report,
    "conditional": conditional,
    "effort": effort,
    "wording": wording,
    "correction": correction,
}


def generate(split, count, seed):
    """Rows (premise, hypothesis, label, identifier, family); labels are balanced within every family.

    A family whose vocabulary cannot supply count / 24 problems per label contributes as many as it has.
    """
    names = list(FAMILIES)
    if count % (3 * len(names)):
        raise ValueError(f"count must be divisible by {3 * len(names)}")
    rng = random.Random(seed)
    each = count // (3 * len(names))
    seen, out = set(), []
    for family in names:
        quota, tries, rows = [each] * 3, 0, []
        while any(quota):
            tries += 1
            if tries > 60 * each + 3000:  # a small vocabulary runs out; keep what the three labels share
                break
            premise, hypothesis, label = FAMILIES[family](split, rng)
            if label is None or not quota[label]:
                continue
            ident = hashlib.sha256((premise + "\n" + hypothesis).encode()).hexdigest()[:24]
            if ident in seen:
                continue
            seen.add(ident)
            quota[label] -= 1
            rows.append((premise, hypothesis, label, ident, family))
        kept = each - max(quota)
        for label in range(3):
            out += [r for r in rows if r[2] == label][:kept]
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    for row in generate("train", 72, 1):
        print(row[4], ["含意", "矛盾", "判断できない"][row[2]], "|", row[0], "→", row[1])
