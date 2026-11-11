"""Japanese proof calibration cases labeled by enumeration of a finite set of states."""

import hashlib
import itertools
import random

NAMES = (("荒木", "久保", "村瀬"), ("橋本", "長野", "大江"), ("平山", "谷川", "小野"))
ATTR = (
    ("当番である", "登録済みである", "研修済みである"),
    ("鍵を持つ", "腕章を持つ", "券を持つ"),
    ("白い札を持つ", "青い札を持つ", "黒い札を持つ"),
)
WORLD_COUNT = 512
ALL = (1 << WORLD_COUNT) - 1


def generate():
    rng = random.Random(131740)
    picked = [{} for _ in range(3)]
    for domain, (names, attrs) in enumerate(zip(NAMES, ATTR)):
        statements = []

        def add(sentence, fn, symbols, statements=statements):
            bits = sum(1 << w for w in range(WORLD_COUNT) if fn(w))
            statements.append((sentence, bits, set(symbols)))

        def f(w, i, j):
            return bool(w & (1 << (3 * i + j)))

        for i, name in enumerate(names):
            for j, attr in enumerate(attrs):
                add(f"{name}は{attr}。", lambda w, i=i, j=j: f(w, i, j), [j])
                add(
                    f"{name}について「{attr}」という記録は誤りである。",
                    lambda w, i=i, j=j: not f(w, i, j),
                    [j],
                )
        for j, attr in enumerate(attrs):
            for n in (0, 1, 2, 3):
                add(
                    f"「{attr}」に該当するのはちょうど{n}人である。",
                    lambda w, j=j, n=n: sum(f(w, i, j) for i in range(3)) == n,
                    [j],
                )
                if n in (1, 2):
                    add(
                        f"少なくとも{n}人について「{attr}」が成り立つ。",
                        lambda w, j=j, n=n: sum(f(w, i, j) for i in range(3)) >= n,
                        [j],
                    )
                    add(
                        f"「{attr}」を満たすのは多くても{n}人だ。",
                        lambda w, j=j, n=n: sum(f(w, i, j) for i in range(3)) <= n,
                        [j],
                    )
        for j, k in itertools.permutations(range(3), 2):
            add(
                f"「{attrs[j]}」を満たす人は全員「{attrs[k]}」も満たす。",
                lambda w, j=j, k=k: all(not f(w, i, j) or f(w, i, k) for i in range(3)),
                [j, k],
            )
            add(
                f"「{attrs[j]}」と「{attrs[k]}」を同時に満たす人がいる。",
                lambda w, j=j, k=k: any(f(w, i, j) and f(w, i, k) for i in range(3)),
                [j, k],
            )
            add(
                f"「{attrs[j]}」と「{attrs[k]}」の両方を満たす人はいない。",
                lambda w, j=j, k=k: (
                    not any(f(w, i, j) and f(w, i, k) for i in range(3))
                ),
                [j, k],
            )
        groups = [0, 0, 0]
        while min(groups) < 16:
            a, b, h = rng.sample(statements, 3)
            premises = [a, b] if rng.random() < 0.7 else [a]
            if not any(v[2] & h[2] for v in premises) or h[0] in [
                v[0] for v in premises
            ]:
                continue
            bits = ALL
            for v in premises:
                bits &= v[1]
            if not bits:
                continue
            label = 0 if bits & ~h[1] == 0 else 1 if bits & h[1] == 0 else 2
            if groups[label] >= 16:
                continue
            p = f"調査の対象は{'、'.join(names)}の三人だけである。" + "".join(
                v[0] for v in premises
            )
            ident = hashlib.sha256((p + "\n" + h[0]).encode()).hexdigest()[:24]
            if ident in picked[label]:
                continue
            picked[label][ident] = (p, h[0], label, domain)
            groups[label] += 1
    return dict(sorted({k: v for group in picked for k, v in group.items()}.items()))
