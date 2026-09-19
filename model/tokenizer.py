from pathlib import Path
import json
class XENTokenizer:
 def __init__(self): self.vocab_size=260
 def fit(self,texts=None,vocab_size=260): pass
 def encode(self,text,max_length=None):
  ids=[1]+[4+b for b in text.encode("utf-8")]+[2]
  if max_length is not None: ids=ids[:max_length]; ids[-1]=2
  return ids
 def decode(self,ids): return bytes(i-4 for i in ids if 4<=i<=259).decode("utf-8",errors="replace")
 def save(self,path): Path(path).write_text(json.dumps({"type":"byte","vocab_size":260}),encoding="utf-8")
 @classmethod
 def load(cls,path): return cls()
