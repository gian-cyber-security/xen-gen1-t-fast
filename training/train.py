import argparse,json,random,yaml,torch
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from model.config import XENConfig
from model.model import XENModel
from model.tokenizer import XENTokenizer
class DS(Dataset):
 def __init__(self,p,t,maxlen): self.rows=[json.loads(x) for x in open(p,encoding='utf-8') if x.strip()]; self.t=t; self.m=maxlen
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  x=self.rows[i]; ids=self.t.encode(str(x.get('instruction',''))+'\n'+str(x.get('response','')),self.m); ids=ids+[0]*(self.m-len(ids)); return torch.tensor(ids[:-1]),torch.tensor(ids[1:])
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',default='datasets/train.jsonl'); p.add_argument('--output',default='outputs/xen'); p.add_argument('--steps',type=int,default=1000); a=p.parse_args(); t=XENTokenizer(); t.fit(); cfg=XENConfig(); d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=XENModel(cfg).to(d); dl=DataLoader(DS(a.data,t,cfg.max_seq_len),batch_size=2,shuffle=True); o=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.1); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); step=0
 while step<a.steps:
  for x,y in dl:
   x,y=x.to(d),y.to(d); _,loss=m(x,y); o.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); o.step(); step+=1
   if step%20==0: print(f'step={step} loss={loss.item():.5f}')
   if step>=a.steps: break
 torch.save({'config':cfg.__dict__,'model':m.state_dict()},out/'model.pt'); t.save(out/'tokenizer.json')
if __name__=='__main__': main()
