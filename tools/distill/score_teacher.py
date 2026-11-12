"""Score the supplied candidates with a local or Hugging Face causal teacher."""
import argparse
import hashlib
import json
import os
import string
from pathlib import Path

def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def model_prompt(row):
    choices = [f"{string.ascii_uppercase[i]}  {desc}" for i, desc in enumerate(row["options"])]
    return ("次の日本語の記録に基づいて設問に回答してください。採点尺度がある場合はその定義を使います。"
            "選択肢から一つを選び、答えの英大文字を一文字だけ返してください。\n\n"
            + row["state"] + "\n\n" + row["question"] + "\n" + "\n".join(choices))

def main():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=Path("teacher_input.jsonl"))
    p.add_argument("--model",required=True)
    p.add_argument("--revision",help="Hugging Face commit or tag (prefer a fixed commit)")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--four-bit",action="store_true")
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    if a.batch_size<1: raise ValueError("batch-size must be positive")
    rows=load_rows(a.input)
    digest=hashlib.sha256(a.input.read_bytes()).hexdigest()
    config=AutoConfig.from_pretrained(a.model,revision=a.revision)
    tokenizer=AutoTokenizer.from_pretrained(a.model,revision=a.revision,padding_side="left")
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    config_args=dict(revision=a.revision,device_map="auto",torch_dtype="auto")
    if a.four_bit:
        config_args["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)
    if type(config) in AutoModelForCausalLM._model_mapping:
        loader=AutoModelForCausalLM
    else:
        from transformers import AutoModelForImageTextToText
        loader=AutoModelForImageTextToText
    model=loader.from_pretrained(a.model,**config_args).eval()
    letters=string.ascii_uppercase[:6]
    tokens=[tokenizer.encode(letter,add_special_tokens=False) for letter in letters]
    if any(len(t)!=1 for t in tokens) or len(set(t[0] for t in tokens))!=len(tokens):
        raise ValueError("Teacher needs distinct single-token answer letters A to F.")
    def prompt(row):
        msgs=[{"role":"user","content":model_prompt(row)}]
        return tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    tmp=a.out.with_suffix(a.out.suffix+".partial")
    revision=getattr(config,"_commit_hash",None) or a.revision or "local-unpinned"
    signature={"model":a.model,"revision":revision,"input_sha256":digest,"four_bit":a.four_bit,"method":"letter-logits-v1"}
    # Only consume whole records previously flushed for this exact input and model.
    done=0
    if tmp.exists():
        with tmp.open("rb+") as f:
            while True:
                offset=f.tell();line=f.readline()
                if not line:break
                if not line.endswith(b"\n"):
                    f.truncate(offset);break
                old=json.loads(line)
                if done>=len(rows) or old["id"]!=rows[done]["id"] or any(old.get(k)!=v for k,v in signature.items()):
                    raise ValueError("Resume metadata or row order changed")
                done+=1
    with tmp.open("a",encoding="utf-8") as out,torch.inference_mode():
        for start in range(done,len(rows),a.batch_size):
            batch=rows[start:start+a.batch_size]
            enc=tokenizer([prompt(r) for r in batch],return_tensors="pt",padding=True,add_special_tokens=False)
            enc=enc.to(model.get_input_embeddings().weight.device)
            logits=model(**enc,use_cache=False).logits[:,-1,:].float()
            for i,r in enumerate(batch):
                scores=logits[i,[t[0] for t in tokens[:len(r["options"])]]]
                probs=torch.softmax(scores,dim=-1).cpu().tolist()
                out.write(json.dumps(dict(id=r["id"],probabilities=probs,**signature),ensure_ascii=False)+"\n")
            out.flush()
    os.replace(tmp,a.out)

if __name__=="__main__":main()
