"""Try one Japanese decision with a fitted Laya-JA adapter."""

import argparse
import json

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
    nli = tasks.add_parser("jnli", help="前提から仮説を判断する")
    nli.add_argument("premise")
    nli.add_argument("hypothesis")
    qa = tasks.add_parser("jcommonsenseqa", help="質問に対し選択肢から答える")
    qa.add_argument("question")
    qa.add_argument("choices", nargs="+")
    grammar = tasks.add_parser("jcola", help="文の容認可能性を推定する")
    grammar.add_argument("sentence")
    sts = tasks.add_parser("jsts", help="二文の意味の近さを0〜5で見積もる")
    sts.add_argument("first")
    sts.add_argument("second")
    args = parser.parse_args()
    agent = load(args.task, model_path=args.model_path, device=args.device)
    if args.task == "jnli":
        result = query(agent, *nli_request(args.premise, args.hypothesis))
    elif args.task == "jcommonsenseqa":
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
