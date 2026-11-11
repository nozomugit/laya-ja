"""Models receive only the Japanese premise and hypothesis."""

INSTRUCTION = """日本語の文章の論理関係を判定します。前提のすべての文を真とします。
A 含意：前提が真なら結論は必ず真。
B 矛盾：前提が真なら結論は必ず偽。
C 判断できない：真の場合も偽の場合も前提と両立する。
記述がないだけならCです。量化、否定の範囲、時制、主語、照応に注意してください。"""

CONTRAST = """あなたは言語学の推論問題を解く。第一の文章だけが証拠である。第二の文章がそれと両立する場合と、成り立たない場合の双方を想像して、次の三つから選ぶ。
A　両方の解釈が可能（不明）
B　第二の文章は第一の文章から導かれる（真）
C　第二の文章は第一の文章と両立しない（偽）
量化語の作用域、否定の対象、比較の方向、時点、複数の前提の組み合わせを吟味する。たとえば「丸い石を三つ拾った」から「石を拾った」はB、「石を一つも拾わなかった」はC、「全ての石が丸かった」はA。"""


def nli_prompt(premise, hypothesis, contrast=False):
    if contrast:
        system = CONTRAST
        user = f"第一の文章：{premise}\n第二の文章：{hypothesis}\n答え（A・B・C）："
    else:
        system = INSTRUCTION
        user = f"【前提】{premise}\n【結論】{hypothesis}\n判定をA・B・Cの一文字で答えてください。"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
