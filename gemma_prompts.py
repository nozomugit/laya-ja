"""Standard Gemma chat format for Japanese entailment."""

from four_task_prompts import INSTRUCTION


def prompt(p, h):
    return (
        f"<bos><|turn>system\n<|think|>\n{INSTRUCTION}<turn|>\n"
        f"<|turn>user\n【前提】{p}\n【結論】{h}\n"
        "主語・数量・否定・時間を確かめ、必ず成り立つかを検討してください。"
        "最後はA・B・Cの一文字で答えてください。<turn|>\n"
        "<|turn>model\n<|channel>thought\n"
    )
