"""Combine Laya span scores with an optional, independently trained Japanese QA reader."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import torch

from grounded_qa import GroundedReader, encode_input, extract_batch, load_head

REPO = "SkelterLabsInc/bert-base-japanese-jaquad"
REV = "7bd5402cab37743c5c989e0773e92cf8089a8c5a"
SHA = "cd12bf57107db2646279587c7830bb461e312611b5e72d08835f7a6f2b903ff8"


class AuxiliaryQA:
    def __init__(self, path=None, device="cpu"):
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        folder = Path(
            path
            or snapshot_download(
                REPO,
                revision=REV,
                allow_patterns=[
                    "pytorch_model.bin",
                    "config.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.txt",
                ],
            )
        )
        with (folder / "pytorch_model.bin").open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != SHA:
                raise ValueError("Unexpected JaQuAD checkpoint")
        self.tok = AutoTokenizer.from_pretrained(
            folder, mecab_kwargs={"mecab_dic": "ipadic"}
        )
        self.net = (
            AutoModelForQuestionAnswering.from_pretrained(folder, use_safetensors=False)
            .eval()
            .to(device)
        )
        self.device = device

    def encode(self, ctx, q, options):
        ctx = unicodedata.normalize("NFKC", ctx)
        ids = []
        offsets = []
        pos = 0
        for word in self.tok.word_tokenizer.tokenize(ctx):
            a = ctx.find(word, pos)
            if a < 0:
                return None
            pos = a + len(word)
            for part in self.tok.subword_tokenizer.tokenize(word):
                b = (
                    pos
                    if part == self.tok.unk_token
                    else min(pos, a + len(part.removeprefix("##")))
                )
                ids.append(self.tok.convert_tokens_to_ids(part))
                offsets.append((a, b))
                a = b
        question = self.tok.encode(q, add_special_tokens=False)
        shift = len(question) + 2
        tokens = [
            self.tok.cls_token_id,
            *question,
            self.tok.sep_token_id,
            *ids,
            self.tok.sep_token_id,
        ]
        if len(tokens) > self.net.config.max_position_embeddings:
            return None
        spans = []
        for option in options:
            pairs = []
            for match in re.finditer(
                re.escape(unicodedata.normalize("NFKC", option)), ctx
            ):
                idx = [
                    j + shift
                    for j, (a, b) in enumerate(offsets)
                    if a < match.end() and b > match.start() and b > a
                ]
                if idx:
                    pairs.append((idx[0], idx[-1]))
            spans.append(pairs)
        return tokens, shift, spans

    @torch.inference_mode()
    def score_many(self, inputs, batch_size=16):
        items = [self.encode(*v) for v in inputs]
        result = torch.zeros(len(inputs), 5)
        order = sorted(
            (i for i, x in enumerate(items) if x is not None),
            key=lambda i: len(items[i][0]),
        )
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            tokens = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(items[i][0]) for i in idx],
                batch_first=True,
                padding_value=self.tok.pad_token_id,
            )
            types = torch.zeros_like(tokens)
            for k, i in enumerate(idx):
                types[k, items[i][1] : len(items[i][0])] = 1
            output = self.net(
                input_ids=tokens.to(self.device),
                attention_mask=(tokens != self.tok.pad_token_id).long().to(self.device),
                token_type_ids=types.to(self.device),
            )
            starts, ends = output.start_logits.cpu(), output.end_logits.cpu()
            for k, i in enumerate(idx):
                choices = items[i][2]
                if not any(choices):
                    continue
                result[i] = torch.tensor(
                    [
                        max(
                            (float(starts[k, a] + ends[k, b]) for a, b in matches),
                            default=-100.0,
                        )
                        for matches in choices
                    ]
                )
        return result


class ExtendedReader:
    def __init__(self, model_path=None, aux_path=None, device="cpu", native=False):
        root = Path(__file__).with_name("weights")
        self.base = GroundedReader(model_path, device)
        self.additional = load_head(root / "extended-span.safetensors", device)
        setting = json.loads((root / "reading-mixture.json").read_text())
        self.coef = setting["native" if native else "combined"]
        self.aux = AuxiliaryQA(aux_path, device) if self.coef[2] else None

    @torch.inference_mode()
    def predict(self, ctx, q, choices):
        if (
            not ctx.strip()
            or not q.strip()
            or len(choices) != 5
            or len(set(choices)) != 5
            or any(not c.strip() for c in choices)
        ):
            raise ValueError(
                "Provide a passage, question and five distinct nonempty choices"
            )
        a = self.base.agent
        it = encode_input(a.tok, ctx, q, choices)
        z, f, m = extract_batch(a.model.encoder, a.tok.pad_token_id, [it])
        if not m.any():
            raise ValueError("No choice occurs in the passage")
        z, f, m = [v.to(a.device) for v in (z, f, m)]
        u = (
            self.base.head(z, f, m).cpu() * self.coef[0]
            + self.additional(z, f, m).cpu() * self.coef[1]
        )
        if self.aux:
            u += self.aux.score_many([(ctx, q, choices)]) * self.coef[2]
        p = u.masked_fill(~m.any(-1).cpu(), -10000)[0].softmax(-1)
        return {
            "choice": choices[p.argmax().item()],
            "probabilities": dict(zip(choices, p.tolist())),
        }
