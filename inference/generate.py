import argparse
import torch
from pathlib import Path
from model.config import XENConfig
from model.model import XENModel
from model.tokenizer import XENTokenizer
from model.skills import load_skills, build_context
from tools.agent_runtime import build_runtime_context, remember_turn

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",default="outputs/xen")
    p.add_argument("--prompt",required=True)
    p.add_argument("--max-new-tokens",type=int,default=128)
    p.add_argument("--skills-dir",default="skills")
    p.add_argument("--no-skills",action="store_true")
    p.add_argument("--web-search",action="store_true")
    p.add_argument("--no-web-search",action="store_true")
    p.add_argument("--memory-file",default="outputs/xen_memory.json")
    p.add_argument("--no-memory",action="store_true")
    a=p.parse_args()
    d=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z=torch.load(Path(a.model_dir)/"model.pt",map_location=d,weights_only=False)
    c=XENConfig(**z["config"]); m=XENModel(c).to(d); m.load_state_dict(z["model"]); m.eval()
    t=XENTokenizer.load(Path(a.model_dir)/"tokenizer.json")
    prompt=a.prompt.strip()
    sys=Path("configs/system_prompt.txt").read_text(encoding="utf-8")
    sk=[] if a.no_skills else load_skills(a.skills_dir)
    base=build_context(sys,sk)
    runtime,_=build_runtime_context(prompt,None if a.no_memory else a.memory_file,force_web=a.web_search,no_web=a.no_web_search)
    full_prompt=base+(("\n\n"+runtime) if runtime else "")+"\n\nUSER:\n"+prompt+"\nASSISTANT:\n"
    ids_list=t.encode(full_prompt,c.max_seq_len)
    if ids_list and ids_list[-1]==2: ids_list=ids_list[:-1]
    ids=torch.tensor([ids_list],device=d)
    for _ in range(a.max_new_tokens):
        logits,_=m(ids[:,-c.max_seq_len:]); nxt=logits[:,-1].argmax(-1,keepdim=True); ids=torch.cat((ids,nxt),1)
        if nxt.item()==2: break
    answer=t.decode(ids[0].tolist()[len(ids_list):])
    print(answer)
    if not a.no_memory: remember_turn(prompt,answer,a.memory_file)

if __name__=="__main__": main()
