"""Write new documents and questions with an open model from the pilot kit's generation requests.

Each request asks for one JSON object (text and items). Raw replies are stored as they come; free_to_input.py
turns the usable ones into teacher input. Finished rows are kept, so the script can be rerun after a stop.
"""

import argparse
import json
from pathlib import Path


def main():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=1400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=20261113)
    a = p.parse_args()
    rows = [json.loads(line) for line in a.requests.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = 0
    if a.out.exists():
        kept = [line for line in a.out.read_text(encoding="utf-8").splitlines() if line.strip()]
        done = len(kept)
        if [json.loads(line)["doc_id"] for line in kept] != [r["doc_id"] for r in rows[:done]]:
            raise ValueError("Existing output does not match the requests")
    config = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if type(config) in AutoModelForCausalLM._model_mapping:
        loader = AutoModelForCausalLM
    else:
        from transformers import AutoModelForImageTextToText as loader
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = loader.from_pretrained(
        a.model, revision=a.revision, device_map={"": 0}, torch_dtype="auto", quantization_config=quant
    ).eval()
    with a.out.open("a", encoding="utf-8") as out, torch.inference_mode():
        for start in range(done, len(rows), a.batch_size):
            batch = rows[start : start + a.batch_size]
            torch.manual_seed(a.seed + start)
            prompts = [
                tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for r in batch
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            ids = model.generate(
                **enc, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=a.temperature, top_p=0.95
            )
            texts = tok.batch_decode(ids[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            for r, text in zip(batch, texts):
                row = dict(doc_id=r["doc_id"], split=r["split"], model=a.model, revision=a.revision, reply=text)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(start + len(batch), "/", len(rows), flush=True)


if __name__ == "__main__":
    main()
