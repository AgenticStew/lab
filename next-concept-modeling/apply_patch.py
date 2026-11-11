#!/usr/bin/env python3
"""Strict unified-diff applier (git-style a/ b/ paths, -p1), for images without git or patch.

Usage: python -I apply_patch.py <patch> <tree> [--check]
Every hunk must match its old lines exactly (line endings included) at the stated position: no fuzz, no offset
search. Every file is checked before anything is written. Handles new files (--- /dev/null) and
"\\ No newline at end of file".
"""
import os
import re
import sys

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse(text):
    lines = text.splitlines(keepends=True)
    files, i = [], 0
    while i < len(lines):
        if lines[i].startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            old = lines[i][4:].rstrip("\r\n").split("\t")[0].strip()
            new = lines[i + 1][4:].rstrip("\r\n").split("\t")[0].strip()
            i += 2
            hunks = []
            while i < len(lines) and lines[i].startswith("@@"):
                m = HUNK.match(lines[i])
                if not m:
                    sys.exit(f"STOP: bad hunk header: {lines[i]!r}")
                os_, oc = int(m[1]), int(m[2] if m[2] is not None else 1)
                ns, nc = int(m[3]), int(m[4] if m[4] is not None else 1)
                i += 1
                body, seen_old, seen_new = [], 0, 0
                while seen_old < oc or seen_new < nc:
                    if i >= len(lines):
                        sys.exit(f"STOP: truncated hunk in {new}")
                    ln = lines[i]
                    if ln.startswith("\\"):
                        if body:
                            body[-1] = (body[-1][0], body[-1][1].rstrip("\n"))
                        i += 1
                        continue
                    tag, content = (ln[0], ln[1:]) if ln not in ("\n", "\r\n") else (" ", ln)
                    if tag not in " +-":
                        sys.exit(f"STOP: unexpected line in a hunk of {new}: {ln!r}")
                    body.append((tag, content))
                    seen_old += tag != "+"
                    seen_new += tag != "-"
                    i += 1
                if i < len(lines) and lines[i].startswith("\\"):
                    body[-1] = (body[-1][0], body[-1][1].rstrip("\n"))
                    i += 1
                hunks.append((os_, oc, body))
            files.append((old, new, hunks))
        else:
            i += 1
    return files


def strip1(p):
    return None if p == "/dev/null" else p.split("/", 1)[1]


def apply_file(tree, old, new, hunks):
    src_rel, dst_rel = strip1(old), strip1(new)
    if dst_rel is None:
        sys.exit("STOP: file deletions are not expected in #5")
    if src_rel is None:
        if os.path.exists(os.path.join(tree, dst_rel)):
            sys.exit(f"STOP: {dst_rel} should be new but exists")
        cur = []
    else:
        with open(os.path.join(tree, src_rel), encoding="utf-8", newline="") as f:
            cur = f.read().splitlines(keepends=True)
    out, pos = [], 0
    for os_, oc, body in hunks:
        start = os_ - 1 if oc else os_
        if start < pos:
            sys.exit(f"STOP: overlapping hunks in {dst_rel}")
        out.extend(cur[pos:start])
        olds = [c for t, c in body if t != "+"]
        if cur[start:start + oc] != olds:
            sys.exit(f"STOP: hunk @@ -{os_},{oc} in {dst_rel} does not match the file exactly")
        out.extend(c for t, c in body if t != "-")
        pos = start + oc
    out.extend(cur[pos:])
    return dst_rel, "".join(out)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    patch, tree, check = sys.argv[1], sys.argv[2], "--check" in sys.argv[3:]
    with open(patch, encoding="utf-8", newline="") as f:
        files = parse(f.read())
    if not files:
        sys.exit("STOP: no file diffs found in the patch")
    results = [apply_file(tree, o, n, h) for o, n, h in files]  # every file checked before any write
    for dst, text in results:
        print(("would patch " if check else "patched ") + dst)
        if not check:
            path = os.path.join(tree, dst)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text)
    print(f"{len(results)} files {'check out' if check else 'patched'}")


if __name__ == "__main__":
    main()
