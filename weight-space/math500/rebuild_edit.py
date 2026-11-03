"""Rebuild GPT-6 Astra's p50c edit of Qwen3.5-0.8B from the public weights and the authors' public logs.

This is my own reimplementation of the logs repo's apply_edit.py (I read it but didn't run it). The base file is copied
byte for byte and only the edited tensors are overwritten in place, so the header and every other tensor are identical
to the public release.

The edit (lm_understand-logs/exp8_math/runs/<run>/edit) is data only: manifest.json (names, shapes, sha256 of the base
and the edited tensor) and edit.npz (for sparse edits: flat indices and new bfloat16 bit patterns). Every tensor is
checked against both hashes, so a successful run means you have exactly the model the authors evaluated.

  python rebuild_edit.py <base .safetensors> <edit dir> <out .safetensors>
"""
import hashlib
import json
import shutil
import struct
import sys

import numpy as np


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def main(base, edit_dir, out):
    man = json.load(open(f"{edit_dir}/manifest.json"))
    z = np.load(f"{edit_dir}/edit.npz", allow_pickle=False)
    head, off = header(base)
    shutil.copyfile(base, out)
    with open(out, "r+b") as f:
        for i, t in enumerate(man["tensors"]):
            info = head[t["name"]]
            assert info["dtype"] == "BF16" and info["shape"] == t["shape"], (t["name"], info)
            if t["form"] != "sparse":
                raise SystemExit(f"{t['name']}: form {t['form']} not supported here (p50c and p25c are sparse)")
            a, b = info["data_offsets"]
            f.seek(off + a)
            bits = np.frombuffer(f.read(b - a), dtype=np.uint16).copy()
            assert hashlib.sha256(bits.tobytes()).hexdigest() == t["base_sha256"], f"{t['name']}: base differs"
            idx, val = z[f"t{i}_idx"], z[f"t{i}_val"]
            changed = int((bits[idx] != val).sum())
            bits[idx] = val
            assert hashlib.sha256(bits.tobytes()).hexdigest() == t["edited_sha256"], f"{t['name']}: edited hash mismatch"
            f.seek(off + a)
            f.write(bits.tobytes())
            print(f"{t['name']}: {len(idx):,} entries written, {changed:,} bits differ from base, sha256 OK")
    print(f"-> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:4])
