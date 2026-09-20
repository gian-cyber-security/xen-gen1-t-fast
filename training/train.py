import argparse, json, random
from pathlib import Path
import yaml
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from model.config import XENConfig
from model.model import XENModel
from model.tokenizer import XENTokenizer

class DS(Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.rows, self.t, self.m = rows, tokenizer, max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r=self.rows[i]
        ids=self.t.encode(str(r.get("instruction",""))+"\n"+str(r.get("response","")), self.m)
        ids=ids+[0]*(self.m-len(ids))
        x=torch.tensor(ids[:-1],dtype=torch.long)
        y=torch.tensor(ids[1:],dtype=torch.long)
        y[y==0]=-100
        return x,y

def load_rows(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            r=json.loads(line)
            if "instruction" not in r or "response" not in r:
                raise ValueError(f"Missing instruction/response at line {n}")
            rows.append(r)
    if not rows: raise ValueError("Dataset is empty")
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="datasets/train.jsonl")
    p.add_argument("--output",default="outputs/xen")
    p.add_argument("--steps",type=int,default=10000)
    p.add_argument("--resume",default=None)
    p.add_argument("--config",default="configs/train.yaml")
    a=p.parse_args()
    with open(a.config,encoding="utf-8") as f: tc=yaml.safe_load(f) or {}
    seed=int(tc.get("seed",42)); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    t=XENTokenizer(); t.fit()
    cfg=XENConfig(vocab_size=int(tc.get("vocab_size",260)),max_seq_len=int(tc.get("max_seq_len",512)),
        d_model=int(tc.get("d_model",512)),n_heads=int(tc.get("n_heads",8)),n_layers=int(tc.get("n_layers",12)),
        ffn_mult=float(tc.get("ffn_mult",2.6666666667)),dropout=float(tc.get("dropout",0.0)),
        rope_theta=float(tc.get("rope_theta",10000.0)))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=XENModel(cfg).to(device); rows=load_rows(a.data)
    val_fraction=float(tc.get("validation_split",0.05))
    val_size=max(1,int(len(rows)*val_fraction)) if len(rows)>20 else 0
    if val_size:
        tr,va=random_split(rows,[len(rows)-val_size,val_size],generator=torch.Generator().manual_seed(seed))
        tr,va=list(tr),list(va)
    else: tr,va=rows,[]
    bs=int(tc.get("batch_size",2)); accum=max(1,int(tc.get("gradient_accumulation",4)))
    loader=DataLoader(DS(tr,t,cfg.max_seq_len),batch_size=bs,shuffle=True,drop_last=False)
    val_loader=DataLoader(DS(va,t,cfg.max_seq_len),batch_size=bs) if va else None
    lr=float(tc.get("learning_rate",3e-4)); wd=float(tc.get("weight_decay",0.1))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd,betas=(0.9,0.95))
    warmup=int(tc.get("warmup_steps",500)); save_every=int(tc.get("save_every",500)); log_every=int(tc.get("log_every",20))
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); step=0; data_iter=iter(loader)
    if a.resume:
        ck=torch.load(a.resume,map_location=device,weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); step=int(ck.get("step",0))
        print(f"Resumed from step={step}")
    model.train()
    while step<a.steps:
        opt.zero_grad(set_to_none=True); total=0.0
        for _ in range(accum):
            try: x,y=next(data_iter)
            except StopIteration: data_iter=iter(loader); x,y=next(data_iter)
            x,y=x.to(device),y.to(device)
            _,loss=model(x,y); (loss/accum).backward(); total+=loss.item()
        if warmup and step<warmup:
            cur=lr*(step+1)/warmup
            for g in opt.param_groups: g["lr"]=cur
        else:
            for g in opt.param_groups: g["lr"]=lr
        torch.nn.utils.clip_grad_norm_(model.parameters(),float(tc.get("max_grad_norm",1.0)))
        opt.step(); step+=1
        if step%log_every==0: print(f"step={step} loss={total/accum:.5f} lr={opt.param_groups[0]['lr']:.2e}")
        if step%save_every==0:
            torch.save({"config":cfg.__dict__,"model":model.state_dict(),"optimizer":opt.state_dict(),"step":step},out/f"checkpoint-{step}.pt")
            print(f"checkpoint saved: step={step}")
    model.eval()
    if val_loader:
        total=n=0
        with torch.no_grad():
            for x,y in val_loader:
                _,loss=model(x.to(device),y.to(device)); total+=loss.item(); n+=1
        print(f"validation_loss={total/max(1,n):.5f}")
    torch.save({"config":cfg.__dict__,"model":model.state_dict()},out/"model.pt"); t.save(out/"tokenizer.json")
    print(f"model saved: {out/'model.pt'}")

if __name__=="__main__": main()
