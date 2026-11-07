"""Hidden, predeclared checks for the objective items (frozen before any model ran).

Code items run in a throwaway directory, isolated Python (-I), no network (unshare -rn when available), 20 s timeout.
Each check returns {"pass": int, "total": int, "notes": [...]} so results read as "tests passed / total".
"""
import json, re, shutil, sqlite3, subprocess, sys, tempfile
from pathlib import Path

PY_TESTS = {
    "C01": ("stable_unique", r'''
xs = ["b", "a", "b", "", "a"]; orig = list(xs)
T("example", lambda: stable_unique(xs) == ["b", "a", ""])
T("no mutation", lambda: (stable_unique(xs), xs == orig)[1])
T("empty", lambda: stable_unique([]) == [])
T("repeated empty strings", lambda: stable_unique(["", "", ""]) == [""])
T("returns list", lambda: isinstance(stable_unique(["x"]), list))
'''),
    "C02": ("merge_intervals", r'''
iv = [(5, 8), (1, 3), (3, 4), (10, 10)]; orig = list(iv)
T("example", lambda: [tuple(x) for x in merge_intervals(iv)] == [(1, 4), (5, 8), (10, 10)])
T("no mutation", lambda: (merge_intervals(iv), iv == orig)[1])
T("empty", lambda: list(merge_intervals([])) == [])
T("nested", lambda: [tuple(x) for x in merge_intervals([(1, 10), (2, 3)])] == [(1, 10)])
T("tuples", lambda: all(isinstance(x, tuple) for x in merge_intervals([(2, 3), (1, 1)])))
def _rev():
    try:
        merge_intervals([(3, 1)])
    except ValueError:
        return True
    return False
T("reversed raises ValueError", _rev)
'''),
    "C03": ("totals", r'''
T("example", lambda: totals('item,qty\n"red,blue",2\nred,3\n"red,blue",-1\n') == {"red,blue": 1, "red": 3})
T("header only", lambda: totals("item,qty\n") == {})
T("header only, no newline", lambda: totals("item,qty") == {})
def _bad():
    try:
        totals("item,qty\nx,abc\n")
    except ValueError:
        return True
    return False
T("invalid int raises ValueError", _bad)
T("negative", lambda: totals("item,qty\na,-5\na,2\n") == {"a": -3})
'''),
    "C04": ("append_one", r'''
T("fresh default 1", lambda: append_one(1) == [1])
T("fresh default 2 (no shared state)", lambda: append_one(2) == [2])
def _explicit():
    b = [0]
    r = append_one(5, b)
    return r is b and b == [0, 5]
T("explicit list mutated and returned", _explicit)
'''),
    "C06": ("first_unique", r'''
xs = ["b", "a", "b", "c", "a"]; orig = list(xs)
T("example", lambda: first_unique(xs) == "c")
T("no mutation", lambda: (first_unique(xs), xs == orig)[1])
T("empty", lambda: first_unique([]) is None)
T("none unique", lambda: first_unique(["x", "x"]) is None)
import time as _t
big = [str(i) for i in range(200000)] * 2 + ["z"]
_t0 = _t.time(); _r = first_unique(big); _dt = _t.time() - _t0
T("linear-ish (400k items < 2 s)", lambda: _r == "z" and _dt < 2.0)
'''),
}

HARNESS = r'''
import json, sys
RES = []
def T(name, fn):
    try:
        ok = bool(fn())
    except Exception as e:  # a crash is a fail
        ok = False; name += f" [{type(e).__name__}: {str(e)[:80]}]"
    RES.append([name, ok])
'''


def extract_code(text, lang="python"):
    blocks = re.findall(rf"```(?:{lang}|py)?[^\n]*\n(.*?)```", text, re.S | re.I)
    if not blocks:
        return text
    return "\n\n".join(blocks)


def _run(src, timeout=20):
    d = tempfile.mkdtemp(prefix="arete-check-")
    try:
        f = Path(d) / "t.py"
        f.write_text(src)
        cmd = [sys.executable, "-I", str(f)]
        if shutil.which("unshare"):
            probe = subprocess.run(["unshare", "-rn", "true"], capture_output=True)
            if probe.returncode == 0:
                cmd = ["unshare", "-rn"] + cmd
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout)
        return r.stdout, r.stderr[-400:]
    except subprocess.TimeoutExpired:
        return "", "timeout"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def check_py(item_id, text):
    entry, tests = PY_TESTS[item_id]
    code = extract_code(text)
    # drop the model's own top-level asserts/tests so a failing self-test doesn't mask the function
    lines, keep = code.split("\n"), []
    for ln in lines:
        if re.match(r"^(assert |if __name__|test_\w+\(\)|print\()", ln):
            break
        keep.append(ln)
    code = "\n".join(keep)
    if f"def {entry}" not in code:
        return {"pass": 0, "total": tests.count("T("), "notes": [f"no def {entry} found"]}
    src = HARNESS + "\n" + code + "\n" + tests + "\nprint('@@RES' + json.dumps(RES))\n"
    out, err = _run(src)
    m = re.search(r"@@RES(.*)", out)
    if not m:
        return {"pass": 0, "total": tests.count("T("), "notes": ["did not run: " + err.strip()[-200:]]}
    res = json.loads(m.group(1))
    return {"pass": sum(ok for _, ok in res), "total": len(res), "notes": [n for n, ok in res if not ok]}


def check_sql(text):
    """C05: run the answer's SQL on a fixed fixture; expected rows (1,2),(2,0),(3,0) in id order."""
    blocks = re.findall(r"```(?:sql|sqlite)?[^\n]*\n(.*?)```", text, re.S | re.I)
    sql = max(blocks, key=len) if blocks else text
    stmts = [s.strip() for s in sql.split(";") if s.strip().upper().startswith(("SELECT", "WITH"))]
    if not stmts:
        return {"pass": 0, "total": 3, "notes": ["no SELECT found"]}
    con = sqlite3.connect(":memory:")
    con.executescript("""create table customers(id integer primary key, name text);
        create table orders(id integer primary key, customer_id integer, status text);
        insert into customers values (1,'a'),(2,'b'),(3,'c');
        insert into orders values (1,1,'paid'),(2,1,'paid'),(3,1,'pending'),(4,2,'pending');""")
    try:
        rows = [tuple(r[:2]) for r in con.execute(stmts[-1]).fetchall()]
    except Exception as e:  # noqa: BLE001
        return {"pass": 0, "total": 3, "notes": [f"SQL error: {e}"]}
    notes, p = [], 0
    if rows == [(1, 2), (2, 0), (3, 0)]:
        p = 3
    else:
        notes.append(f"rows {rows}")
        p += int(len(rows) == 3)
        p += int(sorted(rows) == rows)
    return {"pass": p, "total": 3, "notes": notes}


def words(text):
    return len(text.split())


def check_text(item_id, text):
    t = text.strip()
    if item_id == "R01":
        return {"pass": int(bool(re.search(r"\b96\b", t))), "total": 1, "notes": []}
    if item_id == "R02":
        return {"pass": int(bool(re.search(r"\b1\s*/\s*3\b|one[- ]third|0\.333", t, re.I))), "total": 1, "notes": []}
    if item_id == "R03":
        return {"pass": int(bool(re.search(r"\b9\b|nine", t, re.I))), "total": 1,
                "notes": ["schedule validity scored by the rubric"]}
    if item_id == "R04":
        return {"pass": int(bool(re.match(r"\W*(\*\*)?no\b", t, re.I))), "total": 1,
                "notes": ["counterexample scored by the rubric"]}
    if item_id == "W01":
        n = words(t); return {"pass": int(80 <= n <= 100), "total": 1, "notes": [f"{n} words"]}
    if item_id == "W02":
        n = words(t); return {"pass": int(60 <= n <= 80) + int(bool(re.search(r"friday", t, re.I) and re.search(r"noon|12", t, re.I))),
                              "total": 2, "notes": [f"{n} words"]}
    if item_id == "W03":
        bullets = [ln for ln in t.split("\n") if re.match(r"\s*([-*•]|\d+[.)])\s+", ln)]
        facts = all(re.search(p, t, re.I) for p in (r"\b12\b|twelve", r"\b9\b|nine", r"\b2\b|two", r"\b1\b|one"))
        return {"pass": int(len(bullets) == 3) + int(facts), "total": 2, "notes": [f"{len(bullets)} bullets"]}
    if item_id == "W04":
        js = t
        m = re.search(r"```(?:json)?\s*\n(.*?)```", t, re.S)
        if m:
            js = m.group(1)
        try:
            d = json.loads(js)
        except Exception:  # noqa: BLE001
            return {"pass": 0, "total": 3, "notes": ["not valid JSON"]}
        k = isinstance(d, dict) and set(d) == {"title", "summary", "risks"}
        ty = k and isinstance(d["title"], str) and isinstance(d["summary"], str)
        r3 = k and isinstance(d["risks"], list) and len(d["risks"]) == 3 and all(isinstance(x, str) for x in d["risks"])
        notes = [] if m is None else ["wrapped in a code fence (prompt said 'Return only valid JSON')"]
        return {"pass": int(k) + int(bool(ty)) + int(bool(r3)), "total": 3, "notes": notes}
    return None


def check(item_id, text):
    if item_id in PY_TESTS:
        return check_py(item_id, text)
    if item_id == "C05":
        return check_sql(text)
    return check_text(item_id, text)
