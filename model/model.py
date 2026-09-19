from __future__ import annotations
import torch
from torch import nn
from .config import XENConfig
class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=eps
    def forward(self,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.weight
def rotate_half(x):
    x1,x2=x[...,:x.shape[-1]//2],x[...,x.shape[-1]//2:]; return torch.cat((-x2,x1),dim=-1)
def apply_rope(q,k,cos,sin):
    return q*cos+rotate_half(q)*sin,k*cos+rotate_half(k)*sin
class CausalSelfAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.n_heads=cfg.n_heads; self.head_dim=cfg.d_model//cfg.n_heads
        if cfg.d_model%cfg.n_heads: raise ValueError("d_model must be divisible by n_heads")
        self.qkv=nn.Linear(cfg.d_model,3*cfg.d_model,bias=False); self.out=nn.Linear(cfg.d_model,cfg.d_model,bias=False)
        half=self.head_dim//2; inv=1.0/(cfg.rope_theta**(torch.arange(0,half,dtype=torch.float32)/half))
        freqs=torch.outer(torch.arange(cfg.max_seq_len,dtype=torch.float32),inv); emb=torch.cat((freqs,freqs),dim=-1)
        self.register_buffer("rope_cos",emb.cos()[None,None],persistent=False); self.register_buffer("rope_sin",emb.sin()[None,None],persistent=False)
    def forward(self,x):
        b,t,c=x.shape; q,k,v=self.qkv(x).chunk(3,dim=-1)
        q=q.view(b,t,self.n_heads,self.head_dim).transpose(1,2); k=k.view(b,t,self.n_heads,self.head_dim).transpose(1,2); v=v.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        q,k=apply_rope(q,k,self.rope_cos[:,:,:t],self.rope_sin[:,:,:t]); y=nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.out(y.transpose(1,2).contiguous().view(b,t,c))
class SwiGLU(nn.Module):
    def __init__(self,cfg):
        super().__init__(); hidden=((int(cfg.d_model*cfg.ffn_mult)+63)//64)*64
        self.gate=nn.Linear(cfg.d_model,hidden,bias=False); self.up=nn.Linear(cfg.d_model,hidden,bias=False); self.down=nn.Linear(hidden,cfg.d_model,bias=False)
    def forward(self,x): return self.down(nn.functional.silu(self.gate(x))*self.up(x))
class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.attn_norm=RMSNorm(cfg.d_model); self.attn=CausalSelfAttention(cfg); self.ffn_norm=RMSNorm(cfg.d_model); self.ffn=SwiGLU(cfg)
    def forward(self,x): x=x+self.attn(self.attn_norm(x)); return x+self.ffn(self.ffn_norm(x))
class XENModel(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; self.token_embedding=nn.Embedding(cfg.vocab_size,cfg.d_model); self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)]); self.norm=RMSNorm(cfg.d_model); self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False); self.lm_head.weight=self.token_embedding.weight; self.apply(self._init_weights); self.token_embedding.weight.data.normal_(0,.02)
    @staticmethod
    def _init_weights(m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,.02); m.bias is not None and nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,.02)
    def forward(self,input_ids,targets=None):
        b,t=input_ids.shape
        if t>self.cfg.max_seq_len: raise ValueError("Sequence exceeds XEN max_seq_len")
        x=self.token_embedding(input_ids)
        for block in self.blocks: x=block(x)
        logits=self.lm_head(self.norm(x)); loss=None
        if targets is not None: loss=nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1),ignore_index=-100)
        return logits,loss
