"""Japanese typed decision adapters for the official Laya multilingual model."""

import json
from pathlib import Path

import laya
import torch

BASE = "convaiinnovations/laya-multilingual"
REVISION = "1720e3e3357cfe1e281542e223f8273b0890ca34"
LABELS = ["含意", "矛盾", "判断できない"]
NLI_QUESTION = "前提が正しいとき、仮説はどう判断できますか？"
QA_QUESTION = "質問に最もよく答える選択肢を選んでください。"
TASKS = ("jnli", "jcommonsenseqa")


def nli_request(premise, hypothesis):
    return f"前提：{premise}\n仮説：{hypothesis}", NLI_QUESTION, LABELS.copy()


def qa_request(question, choices):
    if len(choices) < 2 or len(set(choices)) != len(choices):
        raise ValueError("Provide at least two distinct choices.")
    return question, QA_QUESTION, list(choices)


def query(agent, state, instruction, choices):
    question = {
        "type": "choice",
        "instructions": instruction,
        "criteria": dict.fromkeys(choices, ""),
    }
    return agent.predict(state, {"decision": question})["answers"]["decision"]


def load_base(model_path=None, device=None):
    return laya.load(
        model_path or BASE,
        revision=None if model_path else REVISION,
        device=device,
        expected_sha256={
            "model.safetensors": "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204"
        },
    )


def load(task, *, model_path=None, device=None, weights_dir=None):
    """Load the base agent and install the selected task-specific option scorer."""
    if task not in (*TASKS, "jcola", "jsts"):
        raise ValueError(f"Unknown task: {task}")
    weights = Path(weights_dir or Path(__file__).with_name("weights")) / f"{task}.json"
    params = json.loads(weights.read_text())
    if (params["base"], params["revision"], params["task"]) != (BASE, REVISION, task):
        raise ValueError("Adapter metadata mismatch.")
    agent = load_base(model_path, device)
    layer = agent.model.scorer[-1]
    with torch.no_grad():
        weight = torch.as_tensor(
            params["weight"], dtype=layer.weight.dtype, device=layer.weight.device
        )
        if weight.shape != layer.weight.shape or not torch.isfinite(weight).all():
            raise ValueError("Invalid scorer tensor.")
        layer.weight.copy_(weight)
    residual_path = weights.with_name(f"{task}.residual.safetensors")
    if residual_path.exists():
        from residual import ResidualScorer
        from safetensors import safe_open

        with safe_open(str(residual_path), framework="pt") as f:
            meta = f.metadata()
            if meta["task"] != task or meta["base_revision"] != REVISION:
                raise ValueError("Residual metadata mismatch.")
            scorer = ResidualScorer(layer, int(meta["width"]))
            state = scorer.state_dict()
            expected = {k for k in state if not k.startswith("linear.")}
            if set(f.keys()) != expected:
                raise ValueError("Residual tensor names do not match architecture.")
            state.update({k: f.get_tensor(k) for k in expected})
            scorer.load_state_dict(state)
        agent.model.scorer[-1] = scorer.to(
            device=layer.weight.device, dtype=layer.weight.dtype
        ).eval()
    return agent


GRAMMAR = {
    "type": "noul",
    "instructions": "この文は日本語の文法上、容認できますか？",
    "criteria": {"false": "容認しにくい文", "true": "容認できる文"},
}
SIMILARITY = {
    "type": "score",
    "instructions": "二つの文の意味の近さを0から5で評価してください。",
    "criteria": [
        "無関係",
        "ほぼ異なる",
        "一部が共通",
        "かなり共通",
        "ほぼ同じ",
        "同じ意味",
    ],
}


def similarity_state(first, second):
    return f"文A：{first}\n文B：{second}"
