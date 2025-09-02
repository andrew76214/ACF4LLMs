from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, os

MODEL = os.getenv("PRM_MODEL", "")  # mount your trained PRM or leave empty to run neutral
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
tok = None
mdl = None

if MODEL and os.path.isdir(MODEL):
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32).to(DEVICE).eval()

class PRMReq(BaseModel):
    question: str
    steps: list[str]

@app.post("/score")
def score(req: PRMReq):
    if mdl is None or tok is None:
        return {"score": 0.5}
    texts = [f"{req.question}\n\n# step\n{st}" for st in req.steps]
    toks = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        probs = torch.softmax(mdl(**toks).logits, dim=-1)[:,1]
    return {"score": float(probs.mean().item())}