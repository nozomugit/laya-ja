"""Independent truth and falsity probes for Japanese open-world inference."""

RULES = """前提に書かれたことだけを真として、日本語の論理問題を判定する。前提にない事実を補ってはならない。問は「常にそう言えるか」である。A は「はい」、B は「いいえ」。"""
EXAMPLES = {
    "must_true": """例：前提「緑の布を少なくとも二枚買った」から、仮説「布を買った」は必ず真なのでA。「三枚買った」は二枚だけでも前提と両立し、必ず真とは言えないのでB。""",
    "must_false": """例：前提「緑の布を少なくとも二枚買った」から、仮説「布を一枚も買っていない」は必ず偽なのでA。「三枚買った」は三枚でも前提と両立し、必ず偽とは言えないのでB。""",
}


def probe(p, h, kind):
    question = (
        "仮説が偽である場合を前提が排除していて、仮説は必ず真か？"
        if kind == "must_true"
        else "仮説が真である場合を前提が排除していて、仮説は必ず偽か？"
    )
    return (
        f"<|im_start|>system\n{RULES}\n{EXAMPLES[kind]}<|im_end|>\n"
        f"<|im_start|>user\n前提：{p}\n仮説：{h}\n{question}\nA はい／B いいえ。"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
