"""XEN runtime tools: lightweight tool calling + persistent local memory.

The model remains a pure local model. This layer orchestrates tools and injects
relevant memory/context into prompts before inference.
"""
from __future__ import annotations
import ast,json,operator,os,re
from pathlib import Path
from typing import Any

try:
    from .web_search import should_search, search_web, format_results
except ImportError:
    from tools.web_search import should_search, search_web, format_results

_MEMORY_ENV="XEN_MEMORY_FILE"
_DEFAULT_MEMORY="outputs/xen_memory.json"

def _memory_path(path=None):
    return Path(path or os.getenv(_MEMORY_ENV, _DEFAULT_MEMORY))

def load_memory(path=None):
    p=_memory_path(path)
    if not p.exists(): return {"facts":[],"recent":[]}
    try:
        x=json.loads(p.read_text(encoding="utf-8"))
        return {"facts":list(x.get("facts",[])), "recent":list(x.get("recent",[]))}
    except Exception:
        return {"facts":[],"recent":[]}

def save_memory(memory,path=None):
    p=_memory_path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(memory,ensure_ascii=False,indent=2),encoding="utf-8")

def remember(text,path=None):
    text=" ".join(str(text).split()).strip()
    if not text: return
    m=load_memory(path)
    if text not in m["facts"]: m["facts"].append(text)
    m["facts"]=m["facts"][-100:]
    save_memory(m,path)

def remember_turn(user,assistant,path=None):
    m=load_memory(path)
    m["recent"].append({"user":str(user)[-2000:],"assistant":str(assistant)[-2000:]})
    m["recent"]=m["recent"][-20:]
    save_memory(m,path)

def memory_context(path=None):
    m=load_memory(path); parts=[]
    if m["facts"]: parts.append("MEMORY FACTS:\n"+"\n".join("- "+x for x in m["facts"][-40:]))
    if m["recent"]:
        parts.append("RECENT MEMORY:\n" + "\n".join(
            f"User: {x['user']}\nAssistant: {x['assistant']}" for x in m["recent"][-5:]))
    return "\n\n".join(parts)

_BIN={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,
      ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.Pow:operator.pow,
      ast.USub:operator.neg,ast.UAdd:operator.pos}
def _calc(node):
    if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)): return node.value
    if isinstance(node,ast.UnaryOp) and type(node.op) in _BIN: return _BIN[type(node.op)](_calc(node.operand))
    if isinstance(node,ast.BinOp) and type(node.op) in _BIN:
        a,b=_calc(node.left),_calc(node.right)
        if isinstance(node.op,ast.Pow) and abs(b)>100: raise ValueError("exponent too large")
        return _BIN[type(node.op)](a,b)
    raise ValueError("unsupported expression")
def calculate(expr):
    return _calc(ast.parse(expr,mode="eval").body)

def tool_call(prompt, max_results=3):
    """Return tool results selected by a lightweight runtime router."""
    p=str(prompt).strip()
    results=[]
    # Calculator tool: explicit arithmetic/math expressions.
    if re.search(r"\b(calculate|compute|berapa|hitung|hasil dari)\b",p,re.I):
        candidates=re.findall(r"(?<![A-Za-z])[-+*/(). 0-9]{3,}(?![A-Za-z])",p)
        for expr in candidates[:3]:
            try:
                expr=expr.strip()
                if re.search(r"[0-9]",expr) and re.search(r"[+*/-]",expr):
                    results.append(("calculator",expr,str(calculate(expr))))
            except Exception: pass
    # Web search tool: reuse XEN's automatic trigger.
    if should_search(p):
        try:
            web=search_web(p,max_results=max_results)
            if web: results.append(("web_search",p,format_results(web)))
        except Exception as e:
            results.append(("web_search",p,f"Search unavailable: {e}"))
    return results

def build_runtime_context(prompt,memory_path=None,force_web=False,no_web=False):
    parts=[]
    mem=memory_context(memory_path)
    if mem: parts.append(mem)
    calls=tool_call(prompt)
    if force_web and not no_web:
        try:
            web=search_web(prompt,max_results=3)
            if web: calls.append(("web_search",prompt,format_results(web)))
        except Exception: pass
    if no_web: calls=[x for x in calls if x[0]!="web_search"]
    for kind,query,result in calls:
        parts.append(f"TOOL RESULT [{kind}]\n{result}")
    return "\n\n".join(parts), calls
