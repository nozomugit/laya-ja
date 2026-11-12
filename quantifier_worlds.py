"""Japanese inference problems about quantifiers, labeled by checking every small situation.

A situation is up to ten people, each either outside the group, in the group, or in the modified subgroup, and
each either not doing the action, doing it, or doing it in the narrower way; one of them is the named person. A statement is true or false in every
situation; a hypothesis is entailed when it holds in every situation where the premises hold, contradicted when it
holds in none, and undetermined otherwise. Statements with "ほとんど" are read both as more than half and as more
than three quarters, and a problem is kept only when both readings give the same label. A premise that speaks of
all or most members of a group is read as saying the group has members. Vocabulary differs between the training,
tuning and held-out splits, and none of it is taken from an evaluation corpus.
"""

import hashlib
import itertools
import random

import numpy as np

# (group, modified subgroup)
GROUPS = {
    "train": [
        ("委員", "若手の委員"),
        ("社員", "新人の社員"),
        ("参加者", "県外からの参加者"),
        ("医師", "外科の医師"),
        ("店員", "夜勤の店員"),
        ("研究者", "化学の研究者"),
        ("会員", "学生の会員"),
        ("職員", "窓口の職員"),
        ("記者", "地元紙の記者"),
        ("生徒", "二年生の生徒"),
    ],
    "tune": [("技師", "電気の技師"), ("係員", "受付の係員"), ("部員", "一年生の部員")],
    "held": [
        ("作業員", "夜間の作業員"),
        ("講師", "非常勤の講師"),
        ("利用者", "初回の利用者"),
        ("審査員", "外部の審査員"),
    ],
}
# (action, narrower action, not the action, not the narrower action)
ACTIONS = {
    "train": [
        (
            "報告書を読んだ",
            "報告書を最後まで読んだ",
            "報告書を読まなかった",
            "報告書を最後までは読まなかった",
        ),
        (
            "会議に出席した",
            "会議に最初から出席した",
            "会議に出席しなかった",
            "会議に最初からは出席しなかった",
        ),
        (
            "質問に答えた",
            "質問に正しく答えた",
            "質問に答えなかった",
            "質問に正しくは答えなかった",
        ),
        (
            "申込書を出した",
            "申込書を期限までに出した",
            "申込書を出さなかった",
            "申込書を期限までには出さなかった",
        ),
        (
            "会場に着いた",
            "会場に時間どおりに着いた",
            "会場に着かなかった",
            "会場に時間どおりには着かなかった",
        ),
        (
            "料理を作った",
            "料理を一人で作った",
            "料理を作らなかった",
            "料理を一人では作らなかった",
        ),
        (
            "研修を受けた",
            "研修を二日とも受けた",
            "研修を受けなかった",
            "研修を二日ともは受けなかった",
        ),
        (
            "荷物を運んだ",
            "荷物を倉庫まで運んだ",
            "荷物を運ばなかった",
            "荷物を倉庫までは運ばなかった",
        ),
        (
            "意見を述べた",
            "意見を文書で述べた",
            "意見を述べなかった",
            "意見を文書では述べなかった",
        ),
        (
            "試験に合格した",
            "試験に満点で合格した",
            "試験に合格しなかった",
            "試験に満点では合格しなかった",
        ),
    ],
    "tune": [
        (
            "資料を配った",
            "資料を全員に配った",
            "資料を配らなかった",
            "資料を全員には配らなかった",
        ),
        (
            "写真を撮った",
            "写真を屋外で撮った",
            "写真を撮らなかった",
            "写真を屋外では撮らなかった",
        ),
        (
            "電話をかけた",
            "電話を午前中にかけた",
            "電話をかけなかった",
            "電話を午前中にはかけなかった",
        ),
    ],
    "held": [
        (
            "点検を行った",
            "点検を二回行った",
            "点検を行わなかった",
            "点検を二回は行わなかった",
        ),
        (
            "感想を書いた",
            "感想を英語で書いた",
            "感想を書かなかった",
            "感想を英語では書かなかった",
        ),
        (
            "名簿に署名した",
            "名簿に自筆で署名した",
            "名簿に署名しなかった",
            "名簿に自筆では署名しなかった",
        ),
        (
            "説明を聞いた",
            "説明を最後まで聞いた",
            "説明を聞かなかった",
            "説明を最後までは聞かなかった",
        ),
    ],
}
NAMES = {
    "train": [
        "田中さん",
        "鈴木さん",
        "高橋さん",
        "渡辺さん",
        "伊藤さん",
        "山本さん",
        "中村さん",
        "小林さん",
    ],
    "tune": ["吉田さん", "山田さん", "佐々木さん"],
    "held": ["松本さん", "井上さん", "木村さん", "林さん"],
}
MAX_PEOPLE = 10
# Person types: membership (0 outside, 1 group, 2 subgroup) x action (0 none, 1 action, 2 narrower action).
TYPES = list(itertools.product(range(3), range(3)))


def _situations():
    """Every way to have one to MAX_PEOPLE people of the nine types, times the type of the named person."""
    rows = []

    def fill(prefix, left):
        if len(prefix) == len(TYPES) - 1:
            for last in range(left + 1):
                if sum(prefix) + last:
                    rows.append(prefix + [last])
            return
        for v in range(left + 1):
            fill(prefix + [v], left - v)

    fill([], MAX_PEOPLE)
    counts = np.array(rows, dtype=np.int16)
    person = np.repeat(np.arange(len(TYPES)), len(counts))
    counts = np.tile(counts, (len(TYPES), 1))
    present = counts[np.arange(len(counts)), person] > 0
    return counts[present], person[present]


COUNTS, PERSON = _situations()
_MEMBER = np.array([m for m, _ in TYPES])
_ACT = np.array([a for _, a in TYPES])
# People in the (sub)group, and those of them doing the (narrower) action, per situation.
_TABLE = {
    (sub, narrow): (
        COUNTS[:, _MEMBER >= (2 if sub else 1)].sum(1),
        COUNTS[:, (_MEMBER >= (2 if sub else 1)) & (_ACT >= (2 if narrow else 1))].sum(1),
    )
    for sub in (False, True)
    for narrow in (False, True)
}


def _counts(sub, narrow):
    return _TABLE[(sub, narrow)]


def _truth(kind, sub, narrow, k=0, strict=False):
    n, d = _counts(sub, narrow)
    if kind == "all":
        return d == n
    if kind == "some":
        return d >= 1
    if kind == "none":
        return d == 0
    if kind == "not_all":
        return d < n
    if kind == "most":
        return (4 * d > 3 * n) if strict else (2 * d > n)
    if kind == "at_least":
        return d >= k
    if kind == "at_most":
        return d <= k
    if kind == "exactly":
        return d == k
    raise ValueError(kind)


def _text(kind, group, action, negative, k, rng):
    if kind == "all":
        return rng.choice(
            [
                f"すべての{group}が{action}。",
                f"{group}は全員{action}。",
                f"{group}はみな{action}。",
            ]
        )
    if kind == "some":
        return rng.choice(
            [
                f"{action}{group}がいる。",
                f"少なくとも一人の{group}が{action}。",
                f"ある{group}が{action}。",
            ]
        )
    if kind == "none":
        return rng.choice([f"{action}{group}は一人もいない。", f"{action}{group}はいない。"])
    if kind == "not_all":
        return rng.choice([f"{group}の全員が{action}わけではない。", f"{negative}{group}がいる。"])
    if kind == "most":
        return rng.choice([f"ほとんどの{group}が{action}。", f"{group}のほとんどが{action}。"])
    if kind == "at_least":
        return rng.choice(
            [
                f"少なくとも{k}人の{group}が{action}。",
                f"{action}{group}は{k}人以上いる。",
            ]
        )
    if kind == "at_most":
        return rng.choice([f"{action}{group}は多くても{k}人だ。", f"{action}{group}は{k}人以下だ。"])
    return rng.choice([f"{action}{group}はちょうど{k}人だ。", f"ちょうど{k}人の{group}が{action}。"])


KINDS = ["all", "some", "none", "not_all", "most", "at_least", "at_most", "exactly"]


def _statement(rng):
    kind = rng.choice(KINDS)
    k = rng.choice((2, 3)) if kind in ("at_least", "at_most", "exactly") else 0
    return kind, rng.random() < 0.4, rng.random() < 0.4, k


def _person(rng):
    """A statement about the first person: membership, or what that person did."""
    return rng.choice(["is_member", "is_sub", "did", "did_narrow", "did_not", "did_not_narrow"])


def _person_truth(kind):
    m, a = _MEMBER[PERSON], _ACT[PERSON]
    return {
        "is_member": m >= 1,
        "is_sub": m >= 2,
        "did": a >= 1,
        "did_narrow": a >= 2,
        "did_not": a == 0,
        "did_not_narrow": a <= 1,
    }[kind]


def _person_text(kind, name, groups, actions):
    return {
        "is_member": f"{name}は{groups[0]}だ。",
        "is_sub": f"{name}は{groups[1]}だ。",
        "did": f"{name}は{actions[0]}。",
        "did_narrow": f"{name}は{actions[1]}。",
        "did_not": f"{name}は{actions[2]}。",
        "did_not_narrow": f"{name}は{actions[3]}。",
    }[kind]


def generate(split, count, seed):
    """Balanced (premise, hypothesis, label, identifier) rows; label 0 entailed, 1 contradicted, 2 undetermined."""
    if count % 3:
        raise ValueError("count must be divisible by three")
    rng = random.Random(seed)
    quota = [count // 3] * 3
    seen, out = set(), []
    while any(quota):
        groups, actions = rng.choice(GROUPS[split]), rng.choice(ACTIONS[split])
        name = rng.choice(NAMES[split])
        with_person = rng.random() < 0.3
        parts = [("q", _statement(rng)) for _ in range(rng.choice((1, 1, 2)))]
        if with_person:
            parts.append(("p", rng.choice(["is_member", "is_sub"])))
            target = (
                "p",
                rng.choice(["did", "did_narrow", "did_not", "did_not_narrow"]),
            )
        else:
            target = ("q", _statement(rng))
            if target in parts:
                continue
        labels = []
        for strict in (False, True):
            holds = np.ones(len(PERSON), dtype=bool)
            for tag, s in parts:
                if tag == "p":
                    holds &= _person_truth(s)
                else:
                    kind, sub, narrow, k = s
                    holds &= _truth(kind, sub, narrow, k, strict)
                    if kind in ("all", "most"):
                        holds &= _counts(sub, narrow)[0] >= 1
            if not holds.any():
                labels = []
                break
            if target[0] == "p":
                t = _person_truth(target[1])
            else:
                t = _truth(*target[1][:3], target[1][3], strict)
            labels.append(0 if t[holds].all() else 1 if not t[holds].any() else 2)
        if len(set(labels)) != 1 or not quota[labels[0]]:
            continue

        def render(tag, s):
            if tag == "p":
                return _person_text(s, name, groups, actions)
            kind, sub, narrow, k = s
            return _text(kind, groups[sub], actions[narrow], actions[2 + narrow], k, rng)

        order = parts[:]
        rng.shuffle(order)
        premise = "".join(render(*p) for p in order)
        hypothesis = render(*target)
        if hypothesis in premise:
            continue
        ident = hashlib.sha256((premise + "\n" + hypothesis).encode()).hexdigest()[:24]
        if ident in seen:
            continue
        seen.add(ident)
        quota[labels[0]] -= 1
        out.append((premise, hypothesis, labels[0], ident))
    return out


if __name__ == "__main__":
    for row in generate("train", 30, 1):
        print(["含意", "矛盾", "判断できない"][row[2]], "|", row[0], "→", row[1])
