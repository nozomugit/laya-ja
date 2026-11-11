"""Prompt for an explicit two-world check using only the premise and hypothesis."""


def prompt(p, h):
    system = """You are a semanticist working with Japanese. Infer the relation under the ordinary intended reading of the Japanese sentences, respecting lexical entailments and conventional presuppositions. Never fill in missing facts merely because they sound plausible.
Build a situation where every premise is true and the hypothesis is FALSE. Also try one where every premise is true and the hypothesis is TRUE. A situation must respect the exact scope of quantifiers and negation, identity of participants, collective vs individual readings, and event order. If possible, give specific witnesses to yourself. A model in which the hypothesis is false rules out entailment; a model in which it is true rules out contradiction.
Output A if the hypothesis must be true, B if it must be false, C if both possibilities remain. The answer letter is required."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n前提：{p}\n仮説：{h}\n判定：<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
