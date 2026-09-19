#!/usr/bin/env python3
"""Synthetic TEST/SMOKE only: checks model forward, backward, and checkpoint I/O."""
from pathlib import Path
import sys, tempfile, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.config import XENConfig
from model.model import XENModel
from model.tokenizer import XENTokenizer

def main():
    torch.manual_seed(42)
    cfg = XENConfig()
    model = XENModel(cfg)
    tok = XENTokenizer(); tok.fit()
    ids = torch.tensor([tok.encode("TEST/SMOKE synthetic sample", cfg.max_seq_len)])
    _, loss = model(ids[:, :-1], ids[:, 1:])
    loss.backward(); opt = torch.optim.AdamW(model.parameters(), lr=1e-4); opt.step()
    with tempfile.TemporaryDirectory(prefix="xen-smoke-") as d:
        p = Path(d); torch.save({"config": cfg.__dict__, "model": model.state_dict(), "optimizer": opt.state_dict(), "step": 1, "smoke_test": True}, p / "model.pt"); tok.save(p / "tokenizer.json")
        z = torch.load(p / "model.pt", map_location="cpu", weights_only=False)
        check = XENModel(XENConfig(**z["config"])); check.load_state_dict(z["model"]); XENTokenizer.load(p / "tokenizer.json")
    print("SMOKE TEST ONLY: PASS; no real dataset or training claim")
if __name__ == "__main__": main()
