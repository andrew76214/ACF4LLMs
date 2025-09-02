from fastapi import FastAPI
from pydantic import BaseModel
import subprocess, tempfile, os, resource, textwrap, json, sys

app = FastAPI()

class ExecReq(BaseModel):
    program: str                   # code emitted by the model
    final: str | None = None       # parsed 'Answer: <x>' if available
    tests: list[str] | None = []   # optional unit tests (assertions)
    extra_timeout_s: float | None = 0.0

TEMPLATE = """\
import sys, math, fractions, decimal, itertools
import sympy as sp

{code}

def _ci_entry():
    # run tests first
    {tests}
    # if solve() exists, call it
    val = None
    try:
        val = solve()
    except Exception:
        pass
    if val is not None:
        print("Solve:", val)

if __name__ == "__main__":
    _ci_entry()
"""

def run_isolated(py_code: str, timeout=2.5):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(py_code)
        fp = f.name
    def set_limits():
        resource.setrlimit(resource.RLIMIT_CPU, (3,3))
        resource.setrlimit(resource.RLIMIT_DATA, (512*1024*1024, 512*1024*1024))
        os.setsid()
    try:
        p = subprocess.run(
            ["python3", fp],
            preexec_fn=set_limits,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False, text=True, env={"PYTHONIOENCODING":"utf-8","PYTHONHASHSEED":"0"}
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    finally:
        try: os.remove(fp)
        except: pass

def sympy_equal(a: str, b: str) -> bool:
    try:
        return sp.simplify(sp.nsimplify(a) - sp.nsimplify(b)) == 0  # type: ignore
    except Exception:
        return a.strip() == b.strip()

@app.post("/exec")
def exec_code(req: ExecReq):
    tests = ""
    for t in (req.tests or []):
        tests += f"\n    {t}"
    code = TEMPLATE.format(code=req.program, tests=tests or "pass")
    rc, out, err = run_isolated(code, timeout=2.5 + (req.extra_timeout_s or 0.0))
    # derive final candidates
    final_line = None
    for ln in out.splitlines():
        if "Answer:" in ln:
            final_line = ln.split("Answer:",1)[1].strip()
            break
    ok = rc == 0 and ("Traceback" not in err)
    eq = None
    if req.final and final_line:
        eq = sympy_equal(final_line, req.final)
        ok = ok and bool(eq)
    return {"ok": ok, "rc": rc, "stdout": out[-800:], "stderr": err[-800:], "final_runtime": final_line, "equal_to_hint": eq}