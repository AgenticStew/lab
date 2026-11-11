"""Copy an NCM config with a few keys changed (TESTPLAN step 4).

    python -I make_configs.py SRC.yaml OUTDIR --set key=value [--set ...] --name OUT.yaml

A key is a dotted path (generation.frequency_penalty) or a bare leaf name (batch_size), which is
accepted only when exactly one leaf in the config has that name. Unknown or ambiguous keys are refused.
Each matching line is edited in place so `diff SRC OUT` shows only the changed values; the result is
reloaded and must equal SRC merged with the overrides, or the script falls back to OmegaConf.save and
says so.
"""
import argparse
import os
import re
import sys

import yaml
from omegaconf import OmegaConf

KEY_RE = re.compile(r"^(\s*)([A-Za-z0-9_\-]+):(\s*)(.*?)\s*$")


def leaf_paths(node, prefix=()):
    if isinstance(node, dict):
        for k, v in node.items():
            p = prefix + (str(k),)
            if isinstance(v, dict) and v:
                yield from leaf_paths(v, p)
            else:
                yield p


def resolve_key(key, plain):
    paths = list(leaf_paths(plain))
    want = tuple(key.split("."))
    if len(want) > 1:
        # a dotted path may also name a key that is not in the file yet, if its parent exists
        if want in paths:
            return want
        parent = plain
        for k in want[:-1]:
            if not isinstance(parent, dict) or k not in parent:
                sys.exit(f"refused: {key}: no such key in the config")
            parent = parent[k]
        sys.exit(f"refused: {key}: not in the config (only existing keys are changed)")
    hits = [p for p in paths if p[-1] == key]
    if len(hits) != 1:
        sys.exit(f"refused: bare key {key!r} matches {len(hits)} leaves: {['.'.join(h) for h in hits]}; give the full path")
    return hits[0]


def find_line(lines, path):
    """Index of the line holding `path` in a block-style YAML file (None if not found)."""
    stack = []  # (indent, key)
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#") or line.lstrip().startswith("- "):
            continue
        m = KEY_RE.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, m.group(2)))
        if tuple(k for _, k in stack) == path:
            return i
    return None


def dump_value(v):
    s = yaml.safe_dump(v, default_flow_style=True, width=10**6).strip()
    if s.endswith("\n..."):
        s = s[:-4].strip()
    if s.endswith("..."):
        s = s[:-3].strip()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("outdir")
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    base = OmegaConf.load(a.src)
    plain = OmegaConf.to_container(base, resolve=False)
    dot = []
    for s in a.sets:
        if "=" not in s:
            sys.exit(f"refused: --set {s!r} is not key=value")
        k, v = s.split("=", 1)
        path = resolve_key(k.strip(), plain)
        full = ".".join(path)
        val = OmegaConf.to_container(OmegaConf.from_dotlist([f"{full}={v}"]))
        for p in path:
            val = val[p]
        print(f"resolved {k!r} -> {full} = {val!r}")
        dot.append((path, full, v, val))

    expected = OmegaConf.merge(base, OmegaConf.from_dotlist([f"{full}={v}" for _, full, v, _ in dot]))
    lines = open(a.src, encoding="utf-8").read().splitlines(keepends=True)
    ok = True
    for path, full, _, val in dot:
        i = find_line(lines, path)
        if i is None:
            ok = False
            break
        m = KEY_RE.match(lines[i].rstrip("\n"))
        comment = ""
        rest = m.group(4)
        if " #" in rest:
            comment = "  #" + rest.split(" #", 1)[1]
        nl = "\n" if lines[i].endswith("\n") else ""
        lines[i] = f"{m.group(1)}{m.group(2)}: {dump_value(val)}{comment}{nl}"
        # a block value (list or mapping on the following lines) must go with it
        j = i + 1
        indent = len(m.group(1))
        while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > indent
                                  or (lines[j].lstrip().startswith("- ") and len(lines[j]) - len(lines[j].lstrip()) >= indent)):
            if not lines[j].strip():
                break
            j += 1
        del lines[i + 1:j]

    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, a.name)
    if ok:
        with open(out, "w", encoding="utf-8") as f:
            f.writelines(lines)
        got = OmegaConf.load(out)
        if OmegaConf.to_container(got, resolve=False) != OmegaConf.to_container(expected, resolve=False):
            ok = False
    if not ok:
        OmegaConf.save(expected, out)
        print("note: in-place edit failed its check; wrote the merged config with OmegaConf.save (the diff shows reformatting)")
    else:
        print("in-place edit verified: reloaded copy equals the source merged with the overrides")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
