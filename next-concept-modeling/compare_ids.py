#!/usr/bin/env python3
"""Compare one model's output ids between two tables of results.json (TESTPLAN step 5, H3).

    python -I compare_ids.py /results/results.json --table-a A --table-b B --model ncm

Both rows must have been run with --save-ids. The ids files are checked against the sha256 their rows
recorded. identical_pct is the share of sequences that came out identical token for token (the script's
"percent of NCM's passages came out identical"); token_identical_pct is the share of positions that match.
The result is written to results.json under compare_ids.<model> and printed.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_ids(row, table):
    path = row.get("ids_file")
    if not path or not os.path.exists(path):
        sys.exit(f"STOP: table {table} has no ids file (run it with --save-ids)")
    got = sha256_file(path)
    if got != row.get("ids_sha256"):
        sys.exit(f"STOP: {path} sha256 {got} differs from the one table {table} recorded")
    with open(path, encoding="utf-8") as f:
        return json.load(f), path, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--table-a", required=True)
    ap.add_argument("--table-b", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    with open(a.results, encoding="utf-8") as f:
        doc = json.load(f)
    try:
        ra = doc["tables"][a.table_a][a.model]
        rb = doc["tables"][a.table_b][a.model]
    except KeyError as e:
        sys.exit(f"STOP: results.json has no row {e}")
    ia, pa, sa = load_ids(ra, a.table_a)
    ib, pb, sb = load_ids(rb, a.table_b)
    if len(ia) != len(ib):
        sys.exit(f"STOP: {len(ia)} sequences in {a.table_a} vs {len(ib)} in {a.table_b}")
    same_seq = sum(1 for x, y in zip(ia, ib) if x == y)
    pos = sum(max(len(x), len(y)) for x, y in zip(ia, ib))
    same_pos = sum(sum(1 for p, q in zip(x, y) if p == q) for x, y in zip(ia, ib))
    first_diff = []
    for x, y in zip(ia, ib):
        if x != y:
            first_diff.append(next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y))))
    rec = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "table_a": a.table_a, "table_b": a.table_b, "model": a.model,
        "ids_a": {"file": pa, "sha256": sa, "precision": ra.get("precision"), "kv_cache": ra.get("kv_cache")},
        "ids_b": {"file": pb, "sha256": sb, "precision": rb.get("precision"), "kv_cache": rb.get("kv_cache")},
        "sequences": len(ia),
        "identical_sequences": same_seq,
        "identical_pct": 100.0 * same_seq / len(ia) if ia else None,
        "positions": pos,
        "identical_positions": same_pos,
        "token_identical_pct": 100.0 * same_pos / pos if pos else None,
        "first_difference_position_median": sorted(first_diff)[len(first_diff) // 2] if first_diff else None,
        "note": "both tables seeded with the same --seed; ids are the decoder's final token picks",
    }
    doc.setdefault("compare_ids", {})[a.model] = rec
    tmp = a.results + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    os.replace(tmp, a.results)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
