"""Try one Japanese decision with a fitted Laya-JA adapter."""

import argparse
import json
from pathlib import Path

from laya_ja import (
    GRAMMAR,
    SIMILARITY,
    load,
    nli_request,
    qa_request,
    query,
    similarity_state,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", help="Use a local copy of the pinned base checkpoint"
    )
    parser.add_argument("--device", default="cpu")
    tasks = parser.add_subparsers(dest="task", required=True)
    for name, description in (
        ("jnli", "前提から仮説を判断する"),
        ("jnli-specialist", "JNLI 単独で追加学習した推論版を試す"),
        ("multicorpus-nli", "複数コーパスで追加学習した推論版を試す"),
    ):
        nli = tasks.add_parser(name, help=description)
        nli.add_argument("premise")
        nli.add_argument("hypothesis")
    for name, help_text in (
        ("jcommonsenseqa", "質問に対し選択肢から答える"),
        ("reading-qa", "日本語Wikipedia読解を混ぜた実験版で選択肢から答える"),
    ):
        qa = tasks.add_parser(name, help=help_text)
        qa.add_argument("question")
        qa.add_argument("choices", nargs="+")
    grammar = tasks.add_parser("jcola", help="文の容認可能性を推定する")
    grammar.add_argument("sentence")
    sts = tasks.add_parser("jsts", help="二文の意味の近さを0〜5で見積もる")
    sts.add_argument("first")
    sts.add_argument("second")
    for name in ("grounded-qa", "grounded-qa-plus", "grounded-qa-native-plus"):
        grounded = tasks.add_parser(name, help="本文の根拠に基づき五択で答える")
        grounded.add_argument("passage")
        grounded.add_argument("question")
        grounded.add_argument("choices", nargs=5)
        if name == "grounded-qa-plus":
            grounded.add_argument(
                "--aux-path", help="Local copy of the pinned Japanese QA checkpoint"
            )
    for task, fields in (
        ("nli", ("premise", "hypothesis")),
        ("jsts", ("first", "second")),
        ("jcola", ("sentence",)),
        ("jcommonsenseqa", ("question", "choices")),
    ):
        cmd = tasks.add_parser("context-" + task)
        if task == "jsts":
            cmd.add_argument(
                "--symmetric", action="store_true", help="両方向の類似度を平均する"
            )
        if task == "nli":
            cmd.add_argument(
                "--domain", choices=["jnli", "jsick", "janli", "jamp", "logic"]
            )
        for field in fields:
            if field == "choices":
                cmd.add_argument(field, nargs=5)
            else:
                cmd.add_argument(field)
    syntax = tasks.add_parser("grammar-lora")
    syntax.add_argument("sentence")
    common = tasks.add_parser("common-qa-plus")
    common.add_argument("question")
    common.add_argument("choices", nargs=5)
    args = parser.parse_args()
    if args.task == "common-qa-plus":
        from context_tasks import CommonSenseMixture

        print(
            json.dumps(
                CommonSenseMixture(args.model_path, args.device).predict(
                    args.question, args.choices
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.task == "grammar-lora":
        from grammar_lora import GrammarReader

        print(
            json.dumps(
                GrammarReader(args.model_path, args.device).predict(args.sentence),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.task.startswith("context-"):
        from context_tasks import ContextReader

        task = args.task.removeprefix("context-")
        if task == "nli":
            a, options = args.premise, [args.hypothesis]
        elif task == "jsts":
            a, options = args.first, [args.second]
        elif task == "jcola":
            a, options = args.sentence, []
        else:
            a, options = args.question, args.choices
        print(
            json.dumps(
                ContextReader(
                    task,
                    args.model_path,
                    args.device,
                    domain=getattr(args, "domain", None),
                ).predict(a, options, symmetric=getattr(args, "symmetric", False)),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.task in ("grounded-qa-plus", "grounded-qa-native-plus"):
        from reader95 import ExtendedReader

        reader = ExtendedReader(
            args.model_path,
            getattr(args, "aux_path", None),
            args.device,
            native=args.task == "grounded-qa-native-plus",
        )
        print(
            json.dumps(
                reader.predict(args.passage, args.question, args.choices),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.task == "grounded-qa":
        from grounded_qa import GroundedReader

        result = GroundedReader(args.model_path, args.device).predict(
            args.passage, args.question, args.choices
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.task in ("multicorpus-nli", "jnli-specialist"):
        from multicorpus import MultiCorpusNLI

        model = MultiCorpusNLI(
            model_path=args.model_path, device=args.device, variant=args.task
        )
        print(
            json.dumps(
                model.predict(args.premise, args.hypothesis),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    reading = args.task == "reading-qa"
    agent = load(
        "jcommonsenseqa" if reading else args.task,
        model_path=args.model_path,
        device=args.device,
        weights_dir=Path(__file__).with_name("weights") / "reading"
        if reading
        else None,
    )
    if args.task == "jnli":
        result = query(agent, *nli_request(args.premise, args.hypothesis))
    elif args.task in ("jcommonsenseqa", "reading-qa"):
        result = query(agent, *qa_request(args.question, args.choices))
    else:
        if args.task == "jcola":
            state, question = args.sentence, GRAMMAR
        else:
            state, question = similarity_state(args.first, args.second), SIMILARITY
        result = agent.predict(state, {"decision": question})["answers"]["decision"]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
