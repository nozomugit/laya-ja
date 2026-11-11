"""Predict one new example using the same four-task settings as the benchmark."""

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from four_task_assist import LABELS, LocalJudges, blend
from four_task_report import shift
from temporal_relations import probabilities


def predict(a):
    weights = Path(__file__).with_name("weights")
    if a.task == "inference-native":
        from context_tasks import ContextReader
        from native_distillation import encode
        from native_distillation import predict as student

        reader = ContextReader("nli", a.model_path)
        cfg = json.loads((weights / "native-distill.json").read_text())
        x, prior = encode(reader, [{"text": a.premise, "options": [a.hypothesis]}])
        result = student(x, prior, cfg)[0]
        labels = LABELS
    elif a.task == "commonsense":
        from context_tasks import ContextReader

        if len(set(a.choices)) != 5:
            raise ValueError("Supply five distinct choices")
        native = ContextReader("jcommonsenseqa", a.model_path).predict(
            a.question, a.choices
        )
        prior = [native["probabilities"][c] for c in a.choices]
        judge = LocalJudges(a.url, "qa", a.head)
        ext = judge.commonsense(a.question, a.choices)
        config = json.loads((weights / "four-fusion.json").read_text())
        result = blend(ext, prior, config["qa_laya_weight"])
        labels = a.choices
    else:
        # Releasing the encoder process also releases its CPU allocator before
        # the quantized auxiliary model starts.
        command = [sys.executable, str(Path(__file__).with_name("predict.py"))]
        if a.model_path:
            command += ["--model-path", a.model_path]
        command += ["context-nli", a.premise, a.hypothesis]
        if a.task in ("temporal", "temporal-transfer"):
            command += ["--domain", "jamp"]
        raw = json.loads(subprocess.check_output(command, text=True))
        prior = np.array([raw["probabilities"][c] for c in LABELS])
        labels = LABELS
        if a.task == "inference-readings":
            if a.calibrated and a.single_judge:
                raise ValueError("Calibration needs both external judges")
            from four_task_prompts import nli_prompt
            from four_task_reason import engine_scores
            from reading_vote import mix as reading_mix
            from semantic_readings import prompt as reading_prompt

            direct = nli_prompt(a.premise, a.hypothesis)
            prompts = [direct] + [
                reading_prompt(a.premise, a.hypothesis, v) for v in (0, 1)
            ]
            base, first, second = engine_scores(
                a.engine, a.reason_model, prompts, 0, a.threads
            )
            dense = (
                base
                if a.single_judge
                else engine_scores(a.engine, a.dense_model, [direct], 0, a.threads)[0]
            )
            single, dual = reading_mix(prior, base, dense, first, second)
            result = single if a.single_judge else dual
            if a.calibrated:
                from fit_reading_vote import predict as calibrated

                cfg = json.loads((weights / "trained-vote.json").read_text())
                result = calibrated(
                    prior, base, dense, first, second, cfg["coefficients"]
                )
        elif a.task == "inference-dense":
            if a.calibrated and a.single_judge:
                raise ValueError("Calibration needs both external judges")
            from dense_transfer import mix
            from four_task_prompts import nli_prompt
            from four_task_reason import engine_scores

            prompts = [nli_prompt(a.premise, a.hypothesis)]
            dense = engine_scores(a.engine, a.dense_model, prompts, 0, a.threads)[0]
            sparse = (
                dense
                if a.single_judge
                else engine_scores(a.engine, a.reason_model, prompts, 0, a.threads)[0]
            )
            single, dual = mix(prior, dense, sparse)
            result = single if a.single_judge else dual
            if a.calibrated:
                from calibrate_direct import evidence
                from calibrate_direct import predict as calibrated

                cfg = json.loads((weights / "direct-calibration.json").read_text())
                result = calibrated(evidence(prior, dense, sparse), cfg["coefficients"])
        elif a.task == "inference-countermodel":
            from countermodel_prompts import prompt
            from four_task_reason import engine_scores

            expert = engine_scores(
                a.engine,
                a.reason_model,
                [prompt(a.premise, a.hypothesis)],
                256,
                a.threads,
            )[0]
            result = 0.1 * prior + 0.9 * expert
        elif a.task in (
            "inference-verify",
            "inference-gemma",
            "inference-cross-family",
        ):
            from four_task_prompts import nli_prompt
            from four_task_reason import engine_scores
            from semantic_probes import probe
            from semantic_verifier import features
            from semantic_verifier import predict as verify

            if a.task in ("inference-gemma", "inference-cross-family"):
                from gemma_prompts import prompt as gemma_prompt
                from gemma_transfer import mix

                gemma = engine_scores(
                    a.engine,
                    a.gemma_model,
                    [gemma_prompt(a.premise, a.hypothesis)],
                    160,
                    a.threads,
                    "gemma",
                )[0]
                if a.task == "inference-gemma" and a.single_judge:
                    result = 0.1 * prior + 0.9 * gemma
                    return {
                        "choice": labels[int(result.argmax())],
                        "probabilities": dict(zip(labels, result.tolist())),
                    }
            prompts = [
                probe(a.premise, a.hypothesis, k) for k in ("must_true", "must_false")
            ]
            prompts.append(nli_prompt(a.premise, a.hypothesis))
            scores = np.log(
                np.maximum(
                    engine_scores(a.engine, a.reason_model, prompts, 0, a.threads),
                    1e-100,
                )
            )
            cfg = json.loads((weights / "semantic-verifier.json").read_text())
            result = verify(features(*scores, prior), cfg["coefficients"])
            if a.task == "inference-gemma":
                _, result = mix(prior, gemma, result)
            elif a.task == "inference-cross-family":
                from countermodel_prompts import prompt as counter_prompt
                from cross_family import pool

                counter = engine_scores(
                    a.engine,
                    a.reason_model,
                    [counter_prompt(a.premise, a.hypothesis)],
                    256,
                    a.threads,
                )[0]
                result = pool(prior, gemma, counter, result)
        elif a.task == "inference-deep":
            from four_task_prompts import nli_prompt
            from four_task_reason import combine, engine_scores, reason_prompt

            cfg = json.loads((weights / "four-reason.json").read_text())
            if a.single_judge:
                cfg = {"native_weight": 0.1, "deep_weight": 1.0, "contrast_weight": 0.0}
            prompts = [nli_prompt(a.premise, a.hypothesis, t) for t in (False, True)]
            quick = np.zeros(3)
            deep = np.zeros(3)
            if cfg["deep_weight"] < 1:
                z = engine_scores(a.engine, a.direct_model, prompts, 0, a.threads)
                quick = blend(z[0], z[1][[1, 2, 0]], cfg["contrast_weight"])
            if cfg["deep_weight"] > 0:
                deep = engine_scores(
                    a.engine,
                    a.reason_model,
                    [reason_prompt(a.premise, a.hypothesis)],
                    96,
                    a.threads,
                )[0]
            result = combine(prior, quick, deep, cfg)
        elif a.task in ("temporal", "temporal-transfer"):
            hard = a.task == "temporal-transfer" or a.hard or a.graph
            name = "temporal-hard.json.gz" if hard else "temporal.json.gz"
            model = json.loads(gzip.decompress((weights / name).read_bytes()))
            result = probabilities(model, a.premise, a.hypothesis, prior)
            if a.task == "temporal-transfer" or a.graph:
                from temporal_graph import predict as graph_predict
                from temporal_graph import resolve

                supported = json.loads((weights / "temporal-graph.json").read_text())[
                    "supported"
                ]
                found = resolve(a.premise, a.hypothesis)
                if found and found[0] in supported:
                    result = graph_predict(result, a.premise, a.hypothesis, supported)
                elif a.task == "temporal-transfer":
                    from four_task_prompts import nli_prompt
                    from four_task_reason import engine_scores
                    from temporal_transfer import combine as temporal_combine

                    expert = engine_scores(
                        a.engine,
                        a.reason_model,
                        [nli_prompt(a.premise, a.hypothesis)],
                        0,
                        a.threads,
                    )[0]
                    result = temporal_combine(
                        result, expert, a.premise, a.hypothesis, supported
                    )
        else:
            config = json.loads((weights / "four-fusion.json").read_text())
            if a.direct:
                config = {
                    "contrast_weight": 0,
                    "nli_laya_weight": 0.1,
                    "label_bias": [0, 0, 0],
                }
            weight = config["contrast_weight"]
            if a.engine:
                from four_task_prompts import nli_prompt
                from four_task_reason import engine_scores

                styles = (
                    [False] if weight == 0 else [True] if weight == 1 else [False, True]
                )
                z = engine_scores(
                    a.engine,
                    a.direct_model,
                    [nli_prompt(a.premise, a.hypothesis, t) for t in styles],
                    0,
                    a.threads,
                )
                views = [p[[1, 2, 0]] if t else p for p, t in zip(z, styles)]
                ext = views[0] if len(views) == 1 else blend(*views, weight)
            else:
                judge = LocalJudges(a.url, "nli")
                if weight == 0 or weight == 1:
                    ext = judge.infer(a.premise, a.hypothesis, contrast=bool(weight))
                else:
                    ext = blend(
                        judge.infer(a.premise, a.hypothesis),
                        judge.infer(a.premise, a.hypothesis, contrast=True),
                        weight,
                    )
            result = shift(
                blend(ext, prior, config["nli_laya_weight"]), config["label_bias"]
            )
    return {
        "choice": labels[int(result.argmax())],
        "probabilities": dict(zip(labels, result.tolist())),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    tasks = parser.add_subparsers(dest="task", required=True)
    qa = tasks.add_parser("commonsense")
    qa.add_argument("question")
    qa.add_argument("choices", nargs=5)
    qa.add_argument(
        "--head", type=Path, default=Path(".cache/four-models/qa/gguf/head.npz")
    )
    qa.add_argument("--url", default="http://127.0.0.1:18346")
    for name in (
        "temporal",
        "temporal-transfer",
        "inference",
        "inference-native",
        "inference-deep",
        "inference-verify",
        "inference-gemma",
        "inference-countermodel",
        "inference-dense",
        "inference-readings",
        "inference-cross-family",
    ):
        cmd = tasks.add_parser(name)
        cmd.add_argument("premise")
        cmd.add_argument("hypothesis")
        if name in (
            "temporal-transfer",
            "inference-deep",
            "inference-verify",
            "inference-gemma",
            "inference-countermodel",
            "inference-dense",
            "inference-readings",
            "inference-cross-family",
        ):
            cmd.add_argument("--engine", type=Path, required=True)
            if name in ("inference-readings", "inference-dense"):
                cmd.add_argument(
                    "--calibrated",
                    action="store_true",
                    help="Use weights fitted to separate development corpora",
                )
            if name in ("inference-dense", "inference-readings"):
                cmd.add_argument(
                    "--single-judge",
                    action="store_true",
                    help="Use only Laya and the primary judge for this mode",
                )
                cmd.add_argument(
                    "--dense-model",
                    type=Path,
                    default=Path(".cache/four-models/dense/Qwen3.6-27B-Q3_K_S.gguf"),
                )
            if name == "inference-gemma":
                cmd.add_argument(
                    "--single-judge", action="store_true", help="Only Laya and Gemma"
                )
            if name in ("inference-gemma", "inference-cross-family"):
                cmd.add_argument(
                    "--gemma-model",
                    type=Path,
                    default=Path(
                        ".cache/four-models/gemma/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf"
                    ),
                )
            if name == "inference-deep":
                cmd.add_argument(
                    "--single-judge",
                    action="store_true",
                    help="Use Laya with the 35B judge only",
                )
                cmd.add_argument(
                    "--direct-model",
                    type=Path,
                    default=Path(".cache/four-models/nli/Qwen3.5-9B-Q4_K_M.gguf"),
                )
            cmd.add_argument(
                "--reason-model",
                type=Path,
                default=Path(
                    ".cache/four-models/reason/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
                ),
            )
            cmd.add_argument("--threads", type=int, default=4)
        elif name == "inference":
            cmd.add_argument(
                "--engine", type=Path, help="Load the GGUF after Laya exits"
            )
            cmd.add_argument(
                "--direct-model",
                type=Path,
                default=Path(".cache/four-models/nli/Qwen3.5-9B-Q4_K_M.gguf"),
            )
            cmd.add_argument("--threads", type=int, default=4)
            cmd.add_argument(
                "--direct",
                action="store_true",
                help="Fixed 10% Laya, 90% direct three-way judge",
            )
            cmd.add_argument("--url", default="http://127.0.0.1:18345")
        elif name == "temporal":
            cmd.add_argument(
                "--graph",
                action="store_true",
                help="Use template-hard training and supported temporal operations",
            )
            cmd.add_argument(
                "--hard", action="store_true", help="Use template-hard training only"
            )
    print(json.dumps(predict(parser.parse_args()), ensure_ascii=False, indent=2))
