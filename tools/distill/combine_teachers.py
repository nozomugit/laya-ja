"""Validate two complete model outputs and prepare soft-target training data."""
import argparse
import hashlib
import json
import math
import random
import tempfile
from pathlib import Path

def read(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
def index(rows):
    idx={r["id"]:r for r in rows}
    if len(idx)!=len(rows):raise ValueError("Duplicate question id")
    return idx
def dump(path,rows):
    with path.open("w",encoding="utf-8") as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("teacher_a",type=Path);p.add_argument("teacher_b",type=Path)
    p.add_argument("--input",type=Path,default=Path("teacher_input.jsonl"))
    p.add_argument("--documents",type=Path,default=Path("documents.jsonl"))
    p.add_argument("--out",type=Path,default=Path("scored"))
    p.add_argument("--confidence",type=float,default=.9)
    p.add_argument("--score-teacher",choices=["a","b"],help="use only this teacher's distribution for score questions")
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    if not .5<args.confidence<=1:raise ValueError("confidence must be above .5 and at most 1")
    qs=index(read(args.input)); ds={d["doc_id"]:d["split"] for d in read(args.documents)}
    if set(ds.values())!={"train","tune"}:raise ValueError("Invalid splits")
    teachers=[index(read(path)) for path in (args.teacher_a,args.teacher_b)]
    sha=hashlib.sha256(args.input.read_bytes()).hexdigest()
    identity=[]
    for data in teachers:
        if data.keys()!=qs.keys():raise ValueError("Missing or unexpected ids")
        ids={(r["model"],r["revision"]) for r in data.values()}
        if len(ids)!=1 or any(r["input_sha256"]!=sha for r in data.values()): raise ValueError("Input or teacher changed")
        identity.append(ids.pop())
    if identity[0][0]==identity[1][0]:raise ValueError("Supply two distinct teacher models")
    sets={"train":[],"tune":[],"conflicts":[]}
    for qid,r in qs.items():
        k=len(r["options"]); pp=[t[qid]["probabilities"] for t in teachers]
        if any(len(v)!=k or any(not math.isfinite(x) or x<0 or x>1 for x in v) or abs(sum(v)-1)>1e-5 for v in pp):raise ValueError(qid)
        p,q=pp
        ai=max(range(k),key=p.__getitem__);bi=max(range(k),key=q.__getitem__)
        if r["type"]=="score":opposed=abs(ai-bi)>=3
        elif r["type"]=="nli":opposed={ai,bi}=={0,1}
        else:opposed=ai!=bi
        if opposed and min(max(p),max(q))>=args.confidence:
            sets["conflicts"].append(dict(id=qid,teacher_probabilities=pp))
            continue
        probs=[(x+y)/2 for x,y in zip(p,q)]
        if r["type"]=="score" and args.score_teacher:probs=list(p if args.score_teacher=="a" else q)
        typ={"nli":"choice","choice":"choice","noul":"noul","score":"score"}[r["type"]]
        if typ=="choice":criteria=dict.fromkeys(r["options"],"")
        elif typ=="noul":criteria=dict(zip(["false","true"],r["options"]))
        else:criteria=r["options"]
        row=dict(id=qid,doc_id=r["doc_id"],state=r["state"],question=dict(type=typ,instructions=r["question"],criteria=criteria),target=probs)
        if typ=="noul":row["p_yes"]=probs[1]
        if typ=="score":row["expected_score"]=sum(i*v for i,v in enumerate(probs))
        sets[ds[r["doc_id"]]].append(row)
    kept=sets["train"]+sets["tune"]
    audit=random.Random(20261112).sample(kept,min(300,len(kept)))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temp:
        folder=Path(temp)/"deliver";folder.mkdir()
        for name,rows in sets.items():dump(folder/(name+".jsonl"),rows)
        dump(folder/"human_review.jsonl",audit)
        (folder/"provenance.json").write_text(json.dumps(dict(models=identity,input_sha256=sha,confidence=args.confidence,score_teacher=args.score_teacher,counts={k:len(v) for k,v in sets.items()}),indent=2))
        folder.rename(args.out)
if __name__=="__main__":main()
