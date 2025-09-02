import os, time, threading, requests, re, math
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

# Upstreams
SOLVER_BASE   = os.getenv("SOLVER_BASE", "http://localhost:8001")
ESCALATE_BASE = os.getenv("ESCALATE_BASE", "http://localhost:8002")
PRM_BASE      = os.getenv("PRM_BASE", "http://localhost:8010")
SANDBOX_BASE  = os.getenv("SANDBOX_BASE", "http://localhost:8020")

# Carbon & policy
EM_TOKEN = os.getenv("EM_TOKEN", "")
EM_ZONE  = os.getenv("EM_ZONE", "TW")
DEFAULT_KGCO2_PER_KWH = float(os.getenv("DEFAULT_KGCO2_PER_KWH", "0.45"))
CRE_THRESHOLD_PCT_PER_G = float(os.getenv("CRE_THRESHOLD_PCT_PER_G", "0.05"))
UPLIFT_7B  = float(os.getenv("UPLIFT_7B",  "0.25"))
UPLIFT_GPT5= float(os.getenv("UPLIFT_GPT5","0.45"))
DEFAULT_7B_WH = float(os.getenv("DEFAULT_7B_WH", "3.0"))
GPT5_G_CO2_PER_1K_TOK = float(os.getenv("GPT5_G_CO2_PER_1K_TOK", "120.0"))
SAMPLE_INTERVAL_S = float(os.getenv("SAMPLE_INTERVAL_S", "0.2"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")

# NVML sampler
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
    NVH = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    NVML_OK = False
def read_power():
    if not NVML_OK: return 0.0
    try: return pynvml.nvmlDeviceGetPowerUsage(NVH)/1000.0
    except Exception: return 0.0

class Sampler:
    def __init__(self, interval=SAMPLE_INTERVAL_S):
        self.i = interval; self.s=[]; self._e=threading.Event(); self._t=None
    def start(self):
        self.s.clear(); self._e.clear()
        def loop():
            while not self._e.is_set():
                self.s.append((time.time(), read_power()))
                self._e.wait(self.i)
        self._t=threading.Thread(target=loop, daemon=True); self._t.start()
    def stop(self):
        self._e.set()
        if self._t: self._t.join(timeout=1.5)
    def wh(self):
        s=self.s
        if len(s)<2: return 0.0
        tot=0.0
        for i in range(1,len(s)):
            t0,p0=s[i-1]; t1,p1=s[i]
            tot += 0.5*(p0+p1)*((t1-t0)/3600.0)
        return max(0.0, tot)

# Intensity
def kgco2_per_kwh():
    if not EM_TOKEN: return DEFAULT_KGCO2_PER_KWH
    try:
        r=requests.get(f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={EM_ZONE}",
                       headers={"auth-token":EM_TOKEN}, timeout=3)
        r.raise_for_status()
        g=r.json().get("carbonIntensity", 1000.0)
        return max(0.0, float(g)/1000.0)
    except Exception:
        return DEFAULT_KGCO2_PER_KWH

# OpenAI clients for local vLLM servers
solver = OpenAI(base_url=f"{SOLVER_BASE}/v1", api_key="not-needed")
escal  = OpenAI(base_url=f"{ESCALATE_BASE}/v1", api_key="not-needed")

# Optional GPT-5 escalator
remote = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SYSTEM_PoT = "You are a concise math solver. Write minimal Python to compute the answer, then output 'Answer: <value>'."
SYSTEM_CoT = "You are a concise math solver. Think step-by-step briefly, then end with 'Answer: <value>'."

def parse_final(text:str)->str:
    m=re.search(r"(?i)Answer:\s*([^\n]+)", text)
    return m.group(1).strip() if m else text.strip().splitlines()[-1]

def ask(client: OpenAI, system: str, user: str, n=1, temperature=0.2, max_tokens=512):
    return client.chat.completions.create(
        model="local", temperature=temperature, max_tokens=max_tokens, n=n,
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
    )

# Sandbox + PRM
def sandbox_exec(code: str, final: Optional[str], tests: Optional[List[str]]=None, extra_timeout_s: float=0.0):
    try:
        r = requests.post(f"{SANDBOX_BASE}/exec",
                          json={"program":code,"final":final,"tests":tests or [],"extra_timeout_s":extra_timeout_s},
                          timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"ok": False, "stderr": f"sandbox_error: {e}"}

def prm_score(question:str, steps: List[str]) -> Optional[float]:
    try:
        r = requests.post(f"{PRM_BASE}/score", json={"question":question,"steps":steps[:10]}, timeout=3)
        r.raise_for_status()
        return float(r.json()["score"])
    except Exception:
        return None

# Rolling energy stats
ENERGY = {"7b": {"n":0,"sum":0.0,"avg":None}}
def upd_energy(key:str, wh:float):
    st=ENERGY.setdefault(key,{"n":0,"sum":0.0,"avg":None})
    st["n"]+=1; st["sum"]+=max(0.0,wh); st["avg"]=st["sum"]/st["n"]

# Difficulty & budgets
ADV_KEYS = ["proof","lemma","theorem","diophantine","induction","inequality","integral","limit","series","geometry","triangle","cylinder","combinatorics","congruence","prime"]
def lexical_difficulty(q:str)->str:
    L=len(q)
    adv=sum(k in q.lower() for k in ADV_KEYS)
    if adv>=2 or L>400: return "hard"
    if adv==1 or L>200: return "medium"
    return "easy"

BUDGETS = {
    "easy":   {"mode":"pot","k":1,"temp":0.0,"max_tokens":384},
    "medium": {"mode":"pot","k":2,"temp":0.2,"max_tokens":512},
    "hard":   {"mode":"pot","k":4,"temp":0.3,"max_tokens":768},
}

# Confidence aggregation
def combine_conf(ok: bool, prm: Optional[float], maj_ratio: float)->float:
    # ok (binary), prm in [0..1] (None→0.5), maj_ratio in [0..1]
    p = 1.0 if ok else 0.0
    s = prm if prm is not None else 0.5
    return 0.55*p + 0.30*s + 0.15*maj_ratio

# Request schema
app = FastAPI()
class SolveReq(BaseModel):
    question: str
    mode: str = "auto"     # "auto"|"pot"|"cot"
    k: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    repair_rounds: int = 2

@app.post("/solve")
def solve(req: SolveReq):
    q = req.question.strip()
    # 0) Plan budgets
    dif = lexical_difficulty(q)
    plan = BUDGETS[dif].copy()
    if req.mode!="auto": plan["mode"]=req.mode
    if req.k is not None: plan["k"]=req.k
    if req.temperature is not None: plan["temp"]=req.temperature
    if req.max_tokens is not None: plan["max_tokens"]=req.max_tokens
    system = SYSTEM_PoT if plan["mode"]=="pot" else SYSTEM_CoT

    # 1) 1.5B generation with self-consistency (k)
    samp = Sampler(); samp.start()
    resp = ask(solver, system, q, n=plan["k"], temperature=plan["temp"], max_tokens=plan["max_tokens"])
    cand_texts = [c.message.content for c in resp.choices]
    samp.stop()
    wh_15b = samp.wh()

    # score candidates
    finals = [parse_final(t) for t in cand_texts]
    steps_list = [[ln for ln in t.splitlines() if ln.strip()] for t in cand_texts]
    verifs = [sandbox_exec(t, f) for t,f in zip(cand_texts, finals)]
    prms   = [prm_score(q, st) for st in steps_list]

    # majority vote on finals
    freq = {}
    for f in finals: freq[f]=freq.get(f,0)+1
    top_final, top_count = max(freq.items(), key=lambda kv: kv[1])
    maj_ratio = top_count/max(1,len(finals))
    # pick best by score
    def cand_score(i):
        ok = bool(verifs[i].get("ok", False))
        return combine_conf(ok, prms[i], maj_ratio)
    best_i = max(range(len(cand_texts)), key=cand_score)
    best = {
        "text": cand_texts[best_i], "final": finals[best_i],
        "ok": bool(verifs[best_i].get("ok", False)),
        "prm": prms[best_i], "maj_ratio": maj_ratio
    }
    conf = combine_conf(best["ok"], best["prm"], best["maj_ratio"])

    # 2) Iterative repair loop (StepCo-style)
    rounds = max(0, int(req.repair_rounds))
    for r in range(rounds):
        if conf >= 0.80: break
        repair_prompt = (
          f"The following solution seems incorrect or low-confidence. "
          f"Fix ONLY the wrong step(s) and keep it concise.\n\nQuestion:\n{q}\n\n"
          f"Current solution:\n{best['text']}\n"
        )
        # use 1.5B first repair, then 7B for the last repair round if needed
        client = solver if r < rounds-1 else escal
        samp_r = Sampler(); samp_r.start()
        rep = ask(client, system, repair_prompt, n=1, temperature=0.0, max_tokens=plan["max_tokens"])
        samp_r.stop()
        wh_15b += samp_r.wh()
        txt = rep.choices[0].message.content
        fin = parse_final(txt)
        ver = sandbox_exec(txt, fin)
        prm = prm_score(q, [ln for ln in txt.splitlines() if ln.strip()])
        conf2 = combine_conf(bool(ver.get("ok",False)), prm, maj_ratio)
        if conf2 > conf:
            best = {"text":txt,"final":fin,"ok":bool(ver.get("ok",False)),"prm":prm,"maj_ratio":maj_ratio}
            conf = conf2

    # If confident enough, return
    kg = kgco2_per_kwh()
    base = {
        "route": "1.5B" + ("+repair" if rounds>0 else ""),
        "confidence": conf, "verified": best["ok"], "prm": best["prm"],
        "answer": best["final"], "text": best["text"],
        "energy_wh_1p5b": wh_15b, "gco2_1p5b": wh_15b * kg * 1000.0, "kgco2_per_kwh": kg,
        "difficulty": dif, "plan": plan
    }
    if conf >= 0.80:
        return base

    # 3) CRE gate for 7B escalation
    # Estimate delta accuracy and grams
    p1 = conf # proxy for P(correct) with 1.5B
    p2 = min(0.98, p1 + UPLIFT_7B)
    delta_pct = max(0.0, (p2 - p1)*100.0)
    avg7 = ENERGY["7b"]["avg"] if ENERGY["7b"]["avg"] else DEFAULT_7B_WH
    pred_g_7b = avg7 * kg * 1000.0
    cre_7b = delta_pct / pred_g_7b if pred_g_7b>0 else float("inf")

    if cre_7b < CRE_THRESHOLD_PCT_PER_G:
        base["escalation_blocked"] = {
            "stage":"7B", "cre":cre_7b, "delta_acc_pct":delta_pct, "pred_grams":pred_g_7b
        }
        return base

    # 4) Run 7B with energy metering
    samp7 = Sampler(); samp7.start()
    rep7 = ask(escal, system, q, n=1, temperature=0.0, max_tokens=plan["max_tokens"])
    samp7.stop()
    wh7 = samp7.wh(); upd_energy("7b", wh7)
    txt7 = rep7.choices[0].message.content
    fin7 = parse_final(txt7)
    ver7 = sandbox_exec(txt7, fin7)
    prm7 = prm_score(q, [ln for ln in txt7.splitlines() if ln.strip()])
    conf7 = combine_conf(bool(ver7.get("ok",False)), prm7, 1.0)  # single sample at 7B

    if conf7 >= 0.88:
        return {
            **base,
            "route": "7B(spec)",
            "answer": fin7, "text": txt7,
            "verified": bool(ver7.get("ok",False)), "prm": prm7, "confidence": conf7,
            "energy_wh_total": wh_15b + wh7,
            "gco2_total": (wh_15b + wh7) * kg * 1000.0,
            "rolling_avg_7b_wh": ENERGY["7b"]["avg"]
        }

    # 5) CRE gate for GPT-5 escalation (optional)
    if remote is None:
        # No API key -> return best 7B/1.5B
        return {
            **base,
            "route": "7B(spec)-lowconf-no-gpt5",
            "answer": fin7, "text": txt7,
            "verified": bool(ver7.get("ok",False)), "prm": prm7, "confidence": conf7,
            "energy_wh_total": wh_15b + wh7,
            "gco2_total": (wh_15b + wh7) * kg * 1000.0
        }

    # Predict GPT-5 grams by a policy (tokens unknown -> assume 1k)
    delta_pct_5 = max(0.0, (min(0.995, conf7 + UPLIFT_GPT5) - conf7) * 100.0)
    pred_g_5 = GPT5_G_CO2_PER_1K_TOK  # rough policy
    cre_5 = delta_pct_5 / pred_g_5 if pred_g_5>0 else float("inf")
    if cre_5 < CRE_THRESHOLD_PCT_PER_G:
        return {
            **base,
            "route": "7B(spec)-lowconf-blocked-gpt5",
            "note": "GPT-5 escalation blocked by CRE",
            "confidence": conf7,
            "energy_wh_total": wh_15b + wh7,
            "gco2_total": (wh_15b + wh7) * kg * 1000.0
        }

    # 6) GPT-5 escalation
    r5 = remote.chat.completions.create(
        model="gpt-5",
        temperature=0.0, max_tokens=plan["max_tokens"],
        messages=[{"role":"system","content":system},{"role":"user","content":q}]
    )
    txt5 = r5.choices[0].message.content
    fin5 = parse_final(txt5)
    ver5 = sandbox_exec(txt5, fin5)
    return {
        **base,
        "route": "GPT-5",
        "answer": fin5, "text": txt5,
        "verified": bool(ver5.get("ok",False)),
        "confidence": 0.98 if bool(ver5.get("ok",False)) else 0.90,
        "energy_wh_total": wh_15b + wh7,           # API energy not metered locally
        "gco2_total": (wh_15b + wh7) * kg * 1000.0 + pred_g_5
    }