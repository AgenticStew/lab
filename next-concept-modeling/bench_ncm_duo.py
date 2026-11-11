#!/usr/bin/env python3
"""NCM (HazyResearch/ncm @ c702742, #5 applied) against DUO (duo @ 7c9b498): generation speed and quality on
one GPU. TESTPLAN steps 4-6. Runs only inside the throwaway container, from /bench (read-only).

Speed rows (tables A, B, C) generate --n sequences in batches of --batch, one warm-up batch discarded, then
--repeats full runs, each seeded with --seed. Every model forward pass is timed with CUDA events; the wall
clock of each run covers everything (level-0 draw, passes, sampling, the decoder, token pick, the copy to the
host). Quality rows (table Q) generate DUO samples per seed and score them with NCM's own eval_merge.py
(genppl_flm, the gpt2-large scorer), so both models are scored by the same code.

NCM rows replicate scripts/eval_shard.py's unconditional loop line for line; its pick_tokens and
load_allowance functions are taken from that file's source (AST), not re-typed. DUO rows call DUO's own
generate_samples on DUO_BASE, loaded by hand (no Lightning Trainer). Nothing here is typed in by hand: every
number lands in --out by code.

Exit code 3 means a stop condition from TESTPLAN (the message says which); read it before anything else runs.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import pickle
import platform
import re
import statistics
import subprocess
import sys
import textwrap
import time
import types
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = "bench_ncm_duo/1"


# ---------------------------------------------------------------- small helpers

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg):
    print(msg, flush=True)


def die(msg):
    print(f"STOP: {msg}", flush=True)
    sys.exit(3)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def median(xs):
    return float(statistics.median(xs)) if xs else None


def update_results(out, fn):
    """Read-modify-write --out atomically (tmp file + os.replace)."""
    doc = {}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
    fn(doc)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    os.replace(tmp, out)


def put_row(doc, table, key, row):
    t = doc.setdefault("tables", {}).setdefault(table, {})
    if key in t:
        doc.setdefault("superseded", []).append({"table": table, "key": key, "replaced_at": now(), "row": t[key]})
    t[key] = row


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def hf_home_files():
    root = os.environ.get("HF_HOME", "")
    if not root or not os.path.isdir(root):
        return set()
    out = set()
    for d, _, files in os.walk(root):
        for fn in files:
            out.add(os.path.relpath(os.path.join(d, fn), root))
    return out


def pkg_versions():
    out = {}
    for name in ("torch", "transformers", "tokenizers", "lightning", "lightning-utilities", "hydra-core", "omegaconf",
                 "torchmetrics", "huggingface_hub", "numpy", "einops", "triton", "antlr4-python3-runtime"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def git_head(root):
    if not os.path.isdir(os.path.join(root, ".git")):
        return "no .git (tree copied from the sha-checked tarball)"
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"git rev-parse failed: {e}"


def env_meta(torch, a, sources):
    m = {"time_utc": now(), "harness": HARNESS, "harness_sha256": sha256_file(os.path.abspath(__file__)),
         "argv": sys.argv[1:], "python": sys.version.split()[0], "platform": platform.platform(),
         "packages": pkg_versions(), "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
         "arch_list": torch.cuda.get_arch_list()}
    d = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(d)
    free, total = torch.cuda.mem_get_info()
    m["device"] = {"name": p.name, "capability": list(torch.cuda.get_device_capability(d)),
                   "total_gb": round(total / 1e9, 2), "free_gb_at_start": round(free / 1e9, 2)}
    flags = {}
    for k, f in (("matmul_allow_tf32", lambda: torch.backends.cuda.matmul.allow_tf32),
                 ("cudnn_allow_tf32", lambda: torch.backends.cudnn.allow_tf32),
                 ("float32_matmul_precision", torch.get_float32_matmul_precision),
                 ("cudnn_benchmark", lambda: torch.backends.cudnn.benchmark),
                 ("sdp_flash", torch.backends.cuda.flash_sdp_enabled),
                 ("sdp_mem_efficient", torch.backends.cuda.mem_efficient_sdp_enabled),
                 ("sdp_math", torch.backends.cuda.math_sdp_enabled),
                 ("sdp_cudnn", lambda: torch.backends.cuda.cudnn_sdp_enabled())):
        try:
            flags[k] = f()
        except Exception as e:  # noqa: BLE001
            flags[k] = f"unavailable: {type(e).__name__}"
    m["torch_flags"] = flags
    m["sources"] = sources
    return m


# ---------------------------------------------------------------- CUDA graphs

def _flatten(obj):
    leaves = []

    def go(o):
        if isinstance(o, tuple) and not hasattr(o, "_fields"):
            return ("tuple", [go(x) for x in o])
        if isinstance(o, list):
            return ("list", [go(x) for x in o])
        if isinstance(o, dict):
            return ("dict", [(k, go(v)) for k, v in o.items()])
        leaves.append(o)
        return ("leaf", len(leaves) - 1)

    return leaves, go(obj)


def _unflatten(spec, leaves):
    kind, body = spec
    if kind == "leaf":
        return leaves[body]
    if kind == "dict":
        return {k: _unflatten(v, leaves) for k, v in body}
    seq = [_unflatten(x, leaves) for x in body]
    return tuple(seq) if kind == "tuple" else seq


class GraphCache:
    """Capture a side-effect-free module call once per input signature and replay it.

    Inputs are copied into the graph's static tensors before each replay; the returned tensors are the
    graph's static outputs, valid until the same graph replays again (every caller here consumes them first).
    """

    def __init__(self, torch):
        self.torch = torch
        self.graphs = {}
        self.capture_log = []
        self.timed = False
        self.captures_while_timed = 0
        self.bypass = False

    def wrap(self, fn, label):
        torch = self.torch

        def run(*args, **kwargs):
            if self.bypass:
                return fn(*args, **kwargs)
            flat, spec = _flatten((args, kwargs))
            sig = tuple(("T", tuple(v.shape), str(v.dtype), str(v.device)) if torch.is_tensor(v) else ("V", repr(v))
                        for v in flat)
            key = (label, repr(spec), sig)
            g = self.graphs.get(key)
            if g is None:
                g = self._capture(fn, flat, spec, label, sig)
                self.graphs[key] = g
            for dst, src in zip(g["static_in"], flat):
                if dst is not None and dst.data_ptr() != src.data_ptr():
                    dst.copy_(src)
            g["graph"].replay()
            return g["static_out"]

        return run

    def _capture(self, fn, flat, spec, label, sig):
        torch = self.torch
        static_in = [x.clone() if torch.is_tensor(x) else None for x in flat]
        call = _unflatten(spec, [s if s is not None else x for s, x in zip(static_in, flat)])
        args, kwargs = call
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch.clear_autocast_cache()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        if self.timed:
            self.captures_while_timed += 1
        self.capture_log.append(f"{label} {[s[1] for s in sig if s[0] == 'T']}")
        return {"graph": graph, "static_in": static_in, "static_out": out}


# ---------------------------------------------------------------- per-pass timing

class PassMeter:
    def __init__(self, torch):
        self.torch = torch
        self.active = False
        self.cur = None
        self.batches = []  # per timed batch: list of (name, start_event, end_event)

    def wrap(self, fn, name):
        torch = self.torch

        def run(*args, **kwargs):
            if not self.active:
                return fn(*args, **kwargs)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*args, **kwargs)
            e.record()
            self.cur.append((name, s, e))
            return out

        return run

    def begin(self):
        self.cur = []

    def end(self):
        self.batches.append(self.cur)
        self.cur = None

    def resolve(self):
        """Call after torch.cuda.synchronize(): event pairs to milliseconds."""
        return [[(n, s.elapsed_time(e)) for n, s, e in b] for b in self.batches]


def install_attr_wrapper(obj, name, wrapper_factory):
    """Shadow a bound method with an instance attribute (module __call__ looks up self.forward / self.<name>)."""
    bound = getattr(obj, name)
    setattr(obj, name, wrapper_factory(bound))


# ---------------------------------------------------------------- the timed loop

def timed_runs(torch, a, make_batch, meter, graphs):
    """make_batch(B) -> (ids list[list[int]], level0_ms or None). Returns the row's timing fields."""
    nb = -(-a.n // a.batch)
    sizes = [min(a.batch, a.n - i * a.batch) for i in range(nb)]

    torch.manual_seed(a.seed)
    for _ in range(a.warmup):
        make_batch(sizes[0])
    torch.cuda.synchronize()
    log(f"  warm-up done ({a.warmup} batch); graphs captured so far: {len(graphs.capture_log) if graphs else 0}")

    walls, sums_ms, level0, id_hashes, all_ids = [], [], [], [], None
    if graphs:
        graphs.timed = True
    meter.active = a.count_passes
    pass_rows = []
    for r in range(a.repeats):
        meter.batches = []
        torch.manual_seed(a.seed)
        ids_r, l0_r = [], []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for B in sizes:
            meter.begin()
            ids, l0ms = make_batch(B)
            meter.end()
            ids_r.extend(ids)
            if l0ms is not None:
                l0_r.append(l0ms)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        res = meter.resolve() if a.count_passes else []
        pass_rows.append(res)
        walls.append(wall)
        sums_ms.append(sum(ms for b in res for _, ms in b))
        level0.append(sum(l0_r))
        id_hashes.append(sha256_bytes(json.dumps(ids_r).encode()))
        if all_ids is None:
            all_ids = ids_r
        log(f"  repeat {r + 1}/{a.repeats}: {wall:.3f} s wall, {len(ids_r)} sequences")
    meter.active = False
    if graphs:
        graphs.timed = False

    row = {"n": a.n, "batch": a.batch, "batches_per_run": nb, "warmup_batches": a.warmup, "repeats": a.repeats,
           "seed": a.seed, "wall_s": walls, "median_wall_s": median(walls),
           "level0_s_per_run": level0, "median_level0_s": median(level0) if any(level0) else 0.0,
           "repeat_ids_sha256": id_hashes, "repeat_ids_identical": len(set(id_hashes)) == 1}
    if a.count_passes:
        counts = [len(b) for rep in pass_rows for b in rep]
        if len(set(counts)) != 1:
            die(f"model passes per batch were not constant: {sorted(set(counts))}")
        times = [ms for rep in pass_rows for b in rep for _, ms in b]
        by_name, by_index = {}, {}
        for rep in pass_rows:
            for b in rep:
                for i, (name, ms) in enumerate(b):
                    by_name.setdefault(name, []).append(ms)
                    by_index.setdefault(i, []).append(ms)
        share = [(w - s / 1000.0) / w * 100.0 for w, s in zip(walls, sums_ms)]
        row.update({
            "passes_per_batch": counts[0],
            "median_ms_per_pass": median(times),
            "mean_ms_per_pass": float(statistics.fmean(times)),
            "pass_ms_by_name": {k: {"calls": len(v), "median_ms": median(v)} for k, v in by_name.items()},
            "pass_ms_by_index": [median(by_index[i]) for i in sorted(by_index)],
            "pass_ms_sum_per_run_s": [s / 1000.0 for s in sums_ms],
            "non_pass_share_pct_per_run": share,
            "non_pass_share_pct": median(share),
            "pass_timing": "CUDA events recorded around each model forward call on the current stream; with graphs "
                           "on, the time includes copying the inputs into the graph's static tensors",
        })
    if graphs:
        row["graphs"] = {"captured": len(graphs.capture_log), "captured_during_timed_runs": graphs.captures_while_timed,
                         "captures": graphs.capture_log}
        if graphs.captures_while_timed:
            row["graphs"]["warning"] = "a graph was captured during a timed run; that run includes capture time"
    return row, all_ids


def save_ids(a, model, ids, row):
    path = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"ids-{a.table}-{model}.json")
    write_json_atomic(path, ids)
    row["ids_file"] = path
    row["ids_sha256"] = sha256_file(path)
    log(f"  ids -> {path}")


def precision_ctx(torch, a, graphs_on):
    if a.precision == "bf16":
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=not graphs_on)
    return contextlib.nullcontext


# ================================================================ NCM

NCM_RUN_LINES = [
    'temp = pick(a.temp, "temperature", float, 1.0)',
    'top_k = pick(a.top_k, "top_k", int, 1)',
    'top_p = pick(a.top_p, "top_p", float, 0.0)',
    'cfg = pick(a.cfg, "cfg_scale", float, 0.0)',
    'freq = pick(a.freq, "frequency_penalty", float, 0.0)',
    'mu = pick(a.mu, "frequency_deficit", float, 0.0)',
    'adj = str(a.adj if a.adj is not None else G.get("adjacency_penalty", "") or "")',
    'CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)',
    'tk.vqvae = tk.vqvae.cuda().eval()',
    'net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).cuda()',
    'sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)',
    'net.load_state_dict(sd["model"], strict=True)',
    'ALLOW = load_allowance(str(G.frequency_allowance)) if G.get("frequency_allowance") else None',
    'L0_P = load_allowance(str(G.level0_unigram)).float() if G.get("level0_unigram") else None',
    'UNCOND = not CFG.data or not CFG.data.get(split)',
    'ADJ = [float(v) for v in adj.split(",")] if adj else None',
    'g = net.generate(l0, x, temperature=temp, top_k=top_k, top_p=top_p, cfg_scale=cfg, cfg_null_eos=EOS, use_kv_cache=a.kv_cache)',
    '_, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)',
    'ids = pick_tokens(tk.vqvae.decoder(zl[-1])).tolist()',
    'L0 = tk.vqvae.vq.levels[0]',
    'torch.manual_seed(seed)',
    'l0 = torch.multinomial(L0_P.cpu(), B * L0, replacement=True).view(B, L0).cuda() if L0_P is not None else None',
    'hyps += run(torch.full((B, 1), EOS, device="cuda"), l0)',
]


def ncm_eval_shard_functions(torch, path, ck):
    src = open(path, encoding="utf-8").read()
    missing = [ln for ln in NCM_RUN_LINES if ln not in src]
    if missing:
        die(f"{path} no longer matches what this harness replicates; missing lines: {missing}")
    tree = ast.parse(src)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("pick_tokens", "load_allowance")}
    if set(fns) != {"pick_tokens", "load_allowance"}:
        die(f"{path}: pick_tokens/load_allowance not found at module level")
    code = "\n\n".join(ast.get_source_segment(src, fns[k]) for k in ("load_allowance", "pick_tokens"))
    ns = {"torch": torch, "os": os, "ck": ck}
    exec(compile(code, path, "exec"), ns)  # the released functions, verbatim
    return ns, {"eval_shard_sha256": sha256_bytes(src.encode()), "functions_sha256": sha256_bytes(code.encode())}


def pr5_check(a, net, root):
    """TESTPLAN step 5: a KV row must load the cached path step 3 confirmed or applied."""
    rec = {"kv_source": "#5"}
    if not os.path.exists(a.pr5_record):
        die(f"{a.pr5_record} (step 3's #5 record) is missing")
    lines = [ln.rstrip("\n") for ln in open(a.pr5_record, encoding="utf-8") if ln.strip()]
    rec["pr5_record_lines"] = lines
    model_file = inspect.getsourcefile(type(net))
    if not os.path.realpath(model_file).startswith(os.path.realpath(root) + os.sep):
        die(f"NextConceptTransformer loads from {model_file}, outside {root}")
    rec["model_file"] = model_file
    rec["model_file_sha256"] = sha256_file(model_file)
    for m in ("_kv_prefix", "_kv_level"):
        if not hasattr(net, m):
            die(f"the loaded model has no {m}: #5's cached path is not in this tree")
    gsrc = inspect.getsource(type(net).generate)
    if "self._kv_prefix(" not in gsrc or "self._kv_level(" not in gsrc or "use_kv_cache" not in gsrc:
        die("generate() does not call #5's _kv_prefix/_kv_level")
    shas = [m.group(0) for ln in lines for m in [re.search(r"\b[0-9a-f]{64}\b", ln)] if m]
    patch = os.path.join(os.path.dirname(root.rstrip("/")), "ncm-pr5.patch")
    if shas:
        if not os.path.exists(patch):
            die(f"{a.pr5_record} records a patch sha but {patch} is missing")
        got = sha256_file(patch)
        if got not in shas:
            die(f"{patch} sha256 {got} is not the one step 3 recorded ({shas})")
        rec["patch"] = patch
        rec["patch_sha256"] = got
        # every added line of the patch must be present in the file it targets
        target, missing = None, []
        for ln in open(patch, encoding="utf-8", errors="replace"):
            if ln.startswith("+++ "):
                p = ln[4:].strip()
                target = None if p == "/dev/null" else os.path.join(root, p[2:] if p.startswith("b/") else p)
                tsrc = open(target, encoding="utf-8").read() if target and os.path.exists(target) else ""
            elif ln.startswith("+") and not ln.startswith("+++") and target and ln[1:].strip():
                if ln[1:].rstrip("\n") not in tsrc:
                    missing.append((os.path.relpath(target, root), ln[1:80].rstrip()))
        if missing:
            die(f"#5's patch is not fully applied in {root}: {missing[:5]} ...")
        rec["patch_applied_check"] = "every added line present"
    else:
        rec["patch"] = "none: step 3 found #5 in c702742's history"
    return rec


def ncm_main(a, torch):
    root = os.path.abspath(a.ncm_root)
    sys.path.insert(0, root)
    from omegaconf import OmegaConf

    CFG = OmegaConf.load(a.cfg)
    G = CFG.generation
    ck = a.ckpt_dir
    shard_py = os.path.join(root, "scripts", "eval_shard.py")
    ns, src_meta = ncm_eval_shard_functions(torch, shard_py, ck)

    def pick(key, cast, default):
        return cast(G.get(key, default))

    temp = pick("temperature", float, 1.0)
    top_k = pick("top_k", int, 1)
    top_p = pick("top_p", float, 0.0)
    cfg_scale = pick("cfg_scale", float, 0.0)
    freq = pick("frequency_penalty", float, 0.0)
    mu = pick("frequency_deficit", float, 0.0)
    adj = str(G.get("adjacency_penalty", "") or "")
    ADJ = [float(v) for v in adj.split(",")] if adj else None
    if a.penalties == "off":
        freq, mu, ADJ = 0.0, 0.0, None
    split = CFG.eval.get("split", "test")
    if not (not CFG.data or not CFG.data.get(split)):
        die("this config is conditional (CFG.data has the split); the harness times the unconditional LM1B loop only")
    if a.l0 != "unigram":
        die("--l0 other than unigram is not in the test plan")

    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if CFG.checkpoints.get(name):
            CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])

    hf_before = hf_home_files() if a.dry_run else set()
    if a.dry_run:
        # random weights: no checkpoint is unpickled before the network is cut (TESTPLAN safety)
        import hydra
        from models.vqvae import VQVAE
        from utils.registry import tokenizers as TOKREG
        pre_cls = hydra.utils.get_class(TOKREG[CFG.tokenizer.pre_tokenizer.name])
        pre = pre_cls.from_config(CFG.tokenizer.pre_tokenizer)
        tk = VQVAETokenizer(pre, VQVAE(**CFG.tokenizer.vqvae))
    else:
        tk = VQVAETokenizer.from_config(CFG.tokenizer)
    tk.vqvae = tk.vqvae.cuda().eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).cuda()
    ckpt_meta = {}
    for fn in (CFG.checkpoints.generator, CFG.checkpoints.tokenizer):
        p = os.path.join(ck, fn)
        ckpt_meta[fn] = {"path": p, "bytes": os.path.getsize(p) if os.path.exists(p) else None}
    if not a.dry_run:
        sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
        net.load_state_dict(sd["model"], strict=True)
        del sd
    net.eval()
    EOS = tk.pre_tokenizer.eos_token_id
    L0 = tk.vqvae.vq.levels[0]
    V = int(net.config.vocab_size[-1]) if hasattr(net, "config") and hasattr(net.config, "vocab_size") else None

    if a.dry_run:
        ALLOW = None
        if G.get("frequency_allowance"):
            nv = int(tk.vqvae.decoder.out.out_features) if hasattr(getattr(tk.vqvae.decoder, "out", None), "out_features") else 30528
            ALLOW = torch.ones(nv, dtype=torch.float64, device="cuda") / nv
        L0_P = None
        if G.get("level0_unigram"):
            nl = int(CFG.model.vocab_size[0])
            L0_P = torch.ones(nl, dtype=torch.float32, device="cuda") / nl
    else:
        ALLOW = ns["load_allowance"](str(G.frequency_allowance)) if G.get("frequency_allowance") else None
        L0_P = ns["load_allowance"](str(G.level0_unigram)).float() if G.get("level0_unigram") else None
    ns.update({"freq": freq, "mu": mu, "ADJ": ADJ, "ALLOW": ALLOW})
    pick_tokens = ns["pick_tokens"]

    sources = {"ncm_root": root, "ncm_head": git_head(root), "config": a.cfg, "config_sha256": sha256_file(a.cfg),
               **src_meta, "checkpoints": ckpt_meta}
    meta = env_meta(torch, a, sources)
    meta["generation"] = {"temperature": temp, "top_k": top_k, "top_p": top_p, "cfg_scale": cfg_scale,
                          "frequency_penalty": freq, "frequency_deficit": mu, "adjacency_penalty": ADJ,
                          "frequency_allowance": str(G.get("frequency_allowance", "")) or None,
                          "level0": "unigram draw from " + str(G.get("level0_unigram")) if L0_P is not None else "model-predicted",
                          "penalties": a.penalties,
                          "level_sampling": "multinomial per level at the temperature above",
                          "token_pick": "pick_tokens from eval_shard.py (argmax with the penalties above)"}
    meta["levels"] = list(tk.vqvae.vq.levels)

    def build_modes():
        if a.dry_run:
            return [("fp32", "off", "off"), ("bf16", "on", "on")]
        return [(a.precision, a.kv_cache, a.graphs)]

    results = []
    for precision, kv, gr in build_modes():
        if gr == "on" and kv != "on":
            die("--graphs on needs --kv-cache on for NCM (the eager path is not graph-safe as written)")
        if kv == "on" and a.kv_source != "pr5" and not a.dry_run:
            die("--kv-cache on needs --kv-source pr5")
        sub = argparse.Namespace(**vars(a))
        sub.precision, sub.kv_cache, sub.graphs = precision, kv, gr
        if a.dry_run:
            sub.n, sub.warmup, sub.repeats = a.n, 1, 1
        kvrec = pr5_check(a, net, root) if kv == "on" else {"kv_source": "none", "note": "eager path, unchanged by #5"}

        # fresh wrappers per mode
        for m in ("_kv_prefix", "_kv_level", "_decode_level_logits"):
            if m in net.__dict__:
                del net.__dict__[m]
        graphs = GraphCache(torch) if gr == "on" else None
        meter = PassMeter(torch)
        for m in ("_kv_prefix", "_kv_level", "_decode_level_logits"):
            install_attr_wrapper(net, m, lambda f, m=m: meter.wrap(graphs.wrap(f, m) if graphs else f, m))
        ctx = precision_ctx(torch, sub, gr == "on")
        seq_lens = set()

        def make_batch(B):
            t0 = time.perf_counter()
            l0 = torch.multinomial(L0_P.cpu(), B * L0, replacement=True).view(B, L0).cuda() if L0_P is not None else None
            l0ms = (time.perf_counter() - t0) * 1000.0 if L0_P is not None else None
            x = torch.full((B, 1), EOS, device="cuda")
            with torch.no_grad():
                with ctx():
                    g = net.generate(l0, x, temperature=temp, top_k=top_k, top_p=top_p, cfg_scale=cfg_scale,
                                     cfg_null_eos=EOS, use_kv_cache=(kv == "on"))
                    _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
                    logits = tk.vqvae.decoder(zl[-1])
                ids = pick_tokens(logits).tolist()
            for r in ids:
                seq_lens.add(len(r))
            return ids, (l0ms / 1000.0 if l0ms is not None else None)

        log(f"NCM {precision} kv={kv} graphs={gr} penalties={a.penalties}: {sub.n} sequences, batch {a.batch}")
        row, ids = timed_runs(torch, sub, make_batch, meter, graphs)
        if seq_lens != {a.seq_len}:
            die(f"NCM produced sequence lengths {sorted(seq_lens)}, expected {a.seq_len}")
        row.update({"model": "ncm", "label": "NCM, KV cache via #5" if kv == "on" else "NCM, eager (as released)",
                    "precision": precision, "kv_cache": kv, "graphs": row.get("graphs", "off") if gr == "on" else "off",
                    "seq_len": a.seq_len, "kv": kvrec, "meta": meta})
        if precision == "bf16":
            row["precision_note"] = ("torch.autocast bf16 around generate, idx_to_zhats and the decoder; pick_tokens "
                                     "runs in float64 as released")
        results.append((precision, kv, gr, row, ids))

    if a.dry_run:
        new = sorted(hf_home_files() - hf_before)
        log(f"dry run OK on random weights ({len(results)} modes). Files new in HF_HOME: {new or 'none'}")
        for p, k, g_, row, _ in results:
            log(f"  {p} kv={k} graphs={g_}: passes/batch {row.get('passes_per_batch')}, "
                f"median {row.get('median_ms_per_pass')} ms/pass, wall {row['median_wall_s']:.2f} s")
        log("eval_shard.py --help (the real option names step 6 uses):")
        h = subprocess.run([sys.executable, "-I", shard_py, "--help"], cwd=root, capture_output=True, text=True,
                           env={**os.environ, "NCM_ROOT": root})
        log(h.stdout or h.stderr)
        log(f"checkpoint sizes (not opened): {json.dumps(ckpt_meta)}")
        return

    _, _, _, row, ids = results[0]
    if a.save_ids:
        save_ids(a, "ncm", ids, row)
    update_results(a.out, lambda doc: put_row(doc, a.table, "ncm", row))
    log(f"table {a.table} ncm -> {a.out}: median wall {row['median_wall_s']:.3f} s")


# ================================================================ DUO

DUO_FORWARD_UPSTREAM = '''\
  def forward(self, x, sigma, class_cond=None, weights=None):
    assert class_cond is None, 'Not implemented for DiT'
    x = self.vocab_embed(x, weights)
    if self.causal:
      t_cond = None
    else:
      t_cond = F.silu(self.sigma_map(sigma))

    rotary_cos_sin = self.rotary_emb(x)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
      x = self.output_layer(x, c=t_cond)

    return x
'''


def restricted_eval(expr):
    """Stand-in for DUO main.py's `eval` resolver: integer arithmetic only, plus the one worker-count expression."""
    expr = str(expr).strip()
    if expr in ("len(__import__('os').sched_getaffinity(0))", 'len(__import__("os").sched_getaffinity(0))'):
        return len(os.sched_getaffinity(0))
    tree = ast.parse(expr, mode="eval")
    ok = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
          ast.Mod, ast.Pow, ast.USub, ast.UAdd)
    for n in ast.walk(tree):
        if not isinstance(n, ok) or (isinstance(n, ast.Constant) and not isinstance(n.value, (int, float))):
            raise ValueError(f"restricted eval resolver refuses: {expr!r}")
    return eval(compile(tree, "<resolver>", "eval"), {"__builtins__": {}}, {})  # arithmetic only, checked above


def load_patch_module():
    p = os.path.join(HERE, "duo_sdpa_patch.py")
    spec = importlib.util.spec_from_file_location("duo_sdpa_patch", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sha256_file(p)


def stub_module(name):
    m = types.ModuleType(name)

    def __getattr__(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        raise RuntimeError(f"bench stub: DUO's {name}.{attr} is training-only and not loaded here")

    m.__getattr__ = __getattr__
    m.__file__ = f"<bench stub {name}>"
    sys.modules[name] = m


def ckpt_pickle_module(stubbed: set):
    """A pickle module for torch.load whose Unpickler stands in an empty holder for a transformers class that no
    longer exists (lm1b_Duo.ckpt pickles transformers 4.x tokenizer objects, e.g. tokenization_bert.BasicTokenizer,
    which 5.x removed). Only transformers names fall back; anything else missing still raises."""
    import types

    class _Holder:
        def __setstate__(self, state):
            self.__dict__["_pickled_state"] = state

    class Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            try:
                return super().find_class(mod, name)
            except (AttributeError, ImportError):
                if mod.split(".")[0] != "transformers":
                    raise
                stubbed.add(f"{mod}.{name}")
                return type(name, (_Holder,), {"__module__": mod})

    m = types.ModuleType("ckpt_pickle")
    m.Unpickler = Unpickler
    m.load = pickle.load
    m.loads = pickle.loads
    m.dump, m.dumps = pickle.dump, pickle.dumps
    return m


def duo_main(a, torch):
    root = os.path.abspath(a.duo_root)
    patch, patch_sha = load_patch_module()
    patch.install()
    stub_module("dataloader")
    stub_module("utils")
    sys.path.insert(0, root)

    import omegaconf
    from omegaconf import OmegaConf
    import hydra
    import transformers
    compat = []
    if not hasattr(transformers, "PreTrainedTokenizer"):
        transformers.PreTrainedTokenizer = transformers.PreTrainedTokenizerBase  # annotation in trainer_base only
        compat.append("transformers.PreTrainedTokenizer aliased to PreTrainedTokenizerBase (type annotation only)")

    OmegaConf.register_new_resolver("cwd", os.getcwd)
    OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
    OmegaConf.register_new_resolver("eval", restricted_eval)
    OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)

    import models.dit as dit
    import algo

    if not patch.is_installed(dit.flash_attn):
        die("DUO's dit.py did not load the SDPA patch module as flash_attn")
    up = inspect.getsource(dit.DIT.forward)
    if up != DUO_FORWARD_UPSTREAM:
        die("DUO's DIT.forward is not the 7c9b498 text this harness replaces")

    def _stub_ckpt_path():
        return a.ckpt

    ov = ["data=lm1b-wrap", "algo=duo_base", "model=small", f"model.length={a.seq_len}", f"sampling.steps={a.steps}",
          f"sampling.noise_removal={a.noise_removal}", f"loader.eval_batch_size={a.batch}",
          f"eval.gen_ppl_eval_model_name_or_path={a.ref}", f"eval.checkpoint_path={_stub_ckpt_path()}"]
    with hydra.initialize_config_dir(config_dir=os.path.join(root, "configs"), version_base=None):
        config = hydra.compose(config_name="config", overrides=ov)
    if config.sampling.predictor != "ancestral":
        die(f"DUO sampling.predictor is {config.sampling.predictor}, the plan times ancestral")

    hf_before = hf_home_files() if a.dry_run else set()
    tok = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    if tok.bos_token is None:
        tok.bos_token = tok.cls_token
    if tok.eos_token is None:
        tok.eos_token = tok.sep_token
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})

    if a.dry_run:
        # eval_merge.py's genppl_flm loads the generation tokenizer with AutoTokenizer: cache what it needs now,
        # while the network is still on (step 4), so the offline quality rows find it
        transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    model = algo.DUO_BASE(config, tok)
    load = {}
    if a.dry_run:
        load["checkpoint"] = {"path": a.ckpt, "bytes": os.path.getsize(a.ckpt) if os.path.exists(a.ckpt) else None,
                              "note": "not opened in the dry run (random weights)"}
    else:
        stubbed = set()
        ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False, pickle_module=ckpt_pickle_module(stubbed))
        load["checkpoint"] = {"path": a.ckpt, "bytes": os.path.getsize(a.ckpt)}
        if stubbed:
            load["pickle_stubs"] = {"classes": sorted(stubbed), "note": "transformers 4.x classes gone in 5.x, pickled "
                                    "in the checkpoint's hyper_parameters; stood in by empty holders, never used (the "
                                    "bench builds its own BertTokenizer)"}
        hp = ckpt.get("hyper_parameters", {}).get("config")
        if hp is not None:
            hp = OmegaConf.create(hp) if not isinstance(hp, omegaconf.DictConfig) else hp
            diffs, mine_keys = {}, {}
            for sect in ("model", "algo", "noise"):
                mine = OmegaConf.to_container(config[sect], resolve=True)
                mine_keys[sect] = set(mine)
                theirs = OmegaConf.to_container(hp[sect], resolve=False) if sect in hp else None
                if theirs is None:
                    continue
                for k in sorted(set(mine) | set(theirs)):
                    if mine.get(k) != theirs.get(k):
                        diffs[f"{sect}.{k}"] = {"bench": mine.get(k), "checkpoint": theirs.get(k)}
            load["config_vs_checkpoint"] = diffs
            # A key the pinned code doesn't have (bench value None) is harmless only when the checkpoint has it
            # switched off (False/None/0): the pinned code then does what the checkpoint was trained with. Any
            # switched-on feature the pinned code lacks, or any differing shared key, still stops the run.
            off = {k for k, v in diffs.items() if k.startswith("model.") and k not in mine_keys.get("model", set())
                   and (v["checkpoint"] in (False, None, 0, "none", "None")
                        or (isinstance(v["checkpoint"], (int, float)) and not isinstance(v["checkpoint"], bool)
                            and v["checkpoint"] <= 0))}   # softcap: -1 = disabled (a cap of -1 could not generate)
            if off:
                load["checkpoint_only_keys_off"] = sorted(off)
            bad = {k: v for k, v in diffs.items() if k.startswith("model.") and k != "model.length" and k not in off}
            if bad:
                die(f"DUO model config differs from the checkpoint's: {json.dumps(bad, default=str)}")
        try:
            model.on_load_checkpoint(ckpt)
            load["on_load_checkpoint"] = "ok"
        except KeyError as e:
            if e.args and e.args[0] == "loops":
                load["on_load_checkpoint"] = "ema loaded; checkpoint has no Lightning 'loops' (training resume only)"
            else:
                raise
        model.load_state_dict(ckpt["state_dict"], strict=True)
        load["state_dict"] = "strict=True ok"
        del ckpt
    model = model.to("cuda")
    if model.ema is not None:
        model.ema.move_shadow_params_to_device(model.device)
    model._eval_mode()
    load["ema"] = "EMA weights copied in (_eval_mode)" if model.ema is not None else "no EMA"

    # attention record
    attn = {"calls": 0, "q_dtypes": set(), "q_shapes": set()}
    orig_attn = dit.regular_attention_multi_headed

    def attn_wrap(*args, **kwargs):
        q = args[0] if args else next(iter(kwargs.values()))
        attn["calls"] += 1
        attn["q_dtypes"].add(str(q.dtype))
        attn["q_shapes"].add(tuple(q.shape))
        return orig_attn(*args, **kwargs)

    def set_forward(precision, graphs_on):
        """DIT.forward as released, with only its inner bf16 autocast made switchable."""
        if precision == "bf16":
            inner = lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=not graphs_on)  # noqa: E731
        else:
            inner = contextlib.nullcontext
        F = torch.nn.functional

        def forward(self, x, sigma, class_cond=None, weights=None):
            assert class_cond is None, 'Not implemented for DiT'
            x = self.vocab_embed(x, weights)
            if self.causal:
                t_cond = None
            else:
                t_cond = F.silu(self.sigma_map(sigma))
            rotary_cos_sin = self.rotary_emb(x)
            with inner():
                for i in range(len(self.blocks)):
                    x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
                x = self.output_layer(x, c=t_cond)
            return x

        dit.DIT.forward = forward

    def attention_probe(precision):
        """One eager forward, outside every timed row: the attention dtype and the SDPA kernel torch picks."""
        from torch.profiler import profile, ProfilerActivity
        dit.regular_attention_multi_headed = attn_wrap
        x = model.prior_sample(a.batch, model.num_tokens)
        t = torch.full((a.batch, 1), 0.5, device="cuda")
        sigma = model._sigma_from_alphat(model.noise(t)[1])
        with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
            model.forward(xt=x, sigma=sigma)
            torch.cuda.synchronize()
        dit.regular_attention_multi_headed = orig_attn
        names = sorted({e.key for e in prof.key_averages()
                        if any(s in e.key.lower() for s in ("attention", "fmha", "flash", "sdpa", "cudnn", "efficient"))})
        rec = {"backend": "SDPA patch" if patch.is_installed(dit.flash_attn) else "OTHER",
               "attention_fn": "models.dit.regular_attention_multi_headed (F.scaled_dot_product_attention)",
               "rotary": "duo_sdpa_patch.apply_rotary_emb_torch (flash-attn 2.x pure-torch rotary)",
               "patch_sha256": patch_sha, "probe_precision": precision,
               "q_dtypes": sorted(attn["q_dtypes"]), "q_shapes": sorted(attn["q_shapes"]),
               "attention_calls_in_probe": attn["calls"], "cuda_kernels": names}
        attn["calls"], attn["q_dtypes"], attn["q_shapes"] = 0, set(), set()
        if rec["backend"] != "SDPA patch":
            die("a DUO row recorded an attention backend other than the SDPA patch")
        return rec

    sources = {"duo_root": root, "duo_head": git_head(root), "dit_py_sha256": sha256_file(dit.__file__),
               "algo_py_sha256": sha256_file(algo.__file__), "sdpa_patch_sha256": patch_sha, "compat": compat}
    meta = env_meta(torch, a, sources)
    meta["load"] = load
    meta["sampling"] = {"predictor": config.sampling.predictor, "steps": a.steps, "noise_removal": a.noise_removal,
                        "use_float64": bool(config.sampling.use_float64), "length": a.seq_len,
                        "loop": "DUO_BASE.generate_samples as released"}

    meter = PassMeter(torch)

    def install_backbone(graphs):
        if "forward" in model.backbone.__dict__:
            del model.backbone.__dict__["forward"]
        install_attr_wrapper(model.backbone, "forward",
                             lambda f: meter.wrap(graphs.wrap(f, "backbone") if graphs else f, "backbone"))

    if a.quality:
        return duo_quality(a, torch, model, tok, meta, set_forward, attention_probe, install_backbone)

    modes = [("fp32", "off"), ("bf16", "on")] if a.dry_run else [(a.precision, a.graphs)]
    rows = []
    for precision, gr in modes:
        if a.kv_cache != "off":
            die("DUO has no KV cache path; use --kv-cache off")
        set_forward(precision, gr == "on")
        graphs = GraphCache(torch) if gr == "on" else None
        install_backbone(graphs)
        attn_rec = None
        if a.log_attn or a.dry_run:
            if graphs:
                graphs.bypass = True
            attn_rec = attention_probe(precision)
            if graphs:
                graphs.bypass = False
        sub = argparse.Namespace(**vars(a))
        sub.precision, sub.graphs = precision, gr
        if a.dry_run:
            sub.warmup, sub.repeats = 1, 1
        seq_lens = set()

        def make_batch(B):
            with torch.no_grad():
                s = model.generate_samples(num_samples=B, num_steps=a.steps)
            ids = s.tolist()
            for r in ids:
                seq_lens.add(len(r))
            return ids, None

        log(f"DUO {precision} graphs={gr} steps={a.steps} noise_removal={a.noise_removal}: {sub.n} sequences, batch {a.batch}")
        row, ids = timed_runs(torch, sub, make_batch, meter, graphs)
        if seq_lens != {a.seq_len}:
            die(f"DUO produced sequence lengths {sorted(seq_lens)}, expected {a.seq_len}")
        row.update({"model": "duo", "label": "DUO, SDPA attention", "precision": precision, "kv_cache": "off",
                    "graphs": row.get("graphs", "off") if gr == "on" else "off", "seq_len": a.seq_len,
                    "steps": a.steps, "noise_removal": a.noise_removal, "attention": attn_rec, "meta": meta,
                    "precision_note": ("fp32: DIT's hard-coded bf16 block autocast switched off (true fp32)"
                                       if precision == "fp32" else
                                       "bf16: DIT's own bf16 block autocast, as released")})
        rows.append((precision, gr, row))

    if a.dry_run:
        new = sorted(hf_home_files() - hf_before)
        log(f"dry run OK on random weights ({len(rows)} modes). Files new in HF_HOME: {new or 'none'}")
        for p, g_, row in rows:
            log(f"  {p} graphs={g_}: passes/batch {row.get('passes_per_batch')}, "
                f"median {row.get('median_ms_per_pass')} ms/pass, wall {row['median_wall_s']:.2f} s; "
                f"attention {json.dumps(row['attention'], default=list)}")
        log(f"checkpoint (not opened): {json.dumps(load['checkpoint'])}")
        return
    row = rows[0][2]
    update_results(a.out, lambda doc: put_row(doc, a.table, "duo", row))
    log(f"table {a.table} duo -> {a.out}: median wall {row['median_wall_s']:.3f} s")


def duo_quality(a, torch, model, tok, meta, set_forward, attention_probe, install_backbone):
    if a.scorer != "genppl_flm":
        die("only the genppl_flm scorer (NCM's eval_merge.py) is in the plan")
    if a.graphs != "off" or a.precision not in ("fp32", "bf16"):
        die("quality rows run without graphs")
    set_forward(a.precision, False)
    install_backbone(None)
    attn_rec = attention_probe(a.precision) if a.log_attn else None
    special = set(tok.all_special_ids)
    raw_ids, hyps, seeds_col = [], [], []
    t0 = time.perf_counter()
    for seed in a.seeds:
        torch.manual_seed(seed)
        for s in range(0, a.samples, a.batch):
            B = min(a.batch, a.samples - s)
            with torch.no_grad():
                ids = model.generate_samples(num_samples=B, num_steps=a.steps).tolist()
            raw_ids.extend(ids)
            hyps.extend(tok.decode([t for t in r if t not in special], skip_special_tokens=True) for r in ids)
            seeds_col.extend([seed] * B)
        log(f"  seed {seed}: {a.samples} samples")
    gen_s = time.perf_counter() - t0
    mode = a.noise_removal
    rdir = os.path.dirname(os.path.abspath(a.out))
    pkl = os.path.join(rdir, f"duo-{mode}.pkl")
    out = {"idxs": list(range(len(hyps))), "hyps": hyps, "refs": [""] * len(hyps), "seeds": seeds_col,
           "toks": [tok.encode(h, add_special_tokens=False) for h in hyps], "raw_ids": raw_ids}
    with open(pkl + ".tmp", "wb") as f:
        pickle.dump(out, f)
    os.replace(pkl + ".tmp", pkl)

    lens = [len(r) for r in raw_ids]
    trunc = {}
    if a.count_truncated:
        from transformers import AutoTokenizer
        g2 = AutoTokenizer.from_pretrained(a.ref)
        texts = tok.batch_decode(raw_ids)  # specials kept, as eval_merge.py decodes raw_ids
        over = sum(1 for t in texts if len(g2(t)["input_ids"]) > 128)
        trunc = {"generation_len_not_seq_len": sum(1 for n in lens if n != a.seq_len),
                 "scorer_inputs_over_128_gpt2_tokens": over,
                 "note": "eval_merge.py scores the first 128 gpt2-large tokens of each sample"}

    # free the GPU before the scorer loads gpt2-large
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    ncm_root = os.path.abspath(a.ncm_root)
    cfg_path = os.path.join(rdir, f"eval-merge-duo-{mode}.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""\
            task: lm1b-duo-{mode}
            tokenizer:
              pre_tokenizer:
                model_id: bert-base-uncased
            eval:
              ref_model: {a.ref}
              metric: genppl_flm
            """))
    txt = os.path.join(rdir, f"quality-duo-{mode}.txt")
    proc = subprocess.run([sys.executable, "-I", os.path.join(ncm_root, "scripts", "eval_merge.py"), "--config",
                           cfg_path, "--glob", pkl], cwd=ncm_root, capture_output=True, text=True,
                          env={**os.environ, "NCM_ROOT": ncm_root})
    with open(txt, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n--- stderr ---\n" + proc.stderr)
    log(proc.stdout)
    if proc.returncode != 0:
        log(proc.stderr)
        die(f"eval_merge.py failed (exit {proc.returncode}); see {txt}")
    parsed = parse_eval_merge(proc.stdout)
    row = {"model": "duo", "label": "DUO, SDPA attention", "noise_removal": mode, "precision": a.precision,
           "steps": a.steps, "seq_len": a.seq_len, "samples_per_seed": a.samples, "seeds": a.seeds,
           "scorer": "genppl_flm via NCM scripts/eval_merge.py, ref " + a.ref, "eval_merge_txt": txt,
           "eval_merge_config": cfg_path, "pkl": pkl, "pkl_sha256": sha256_file(pkl), "generation_s": gen_s,
           **parsed, "truncation": trunc, "attention": attn_rec, "meta": meta,
           "precision_note": ("fp32: DIT's hard-coded bf16 block autocast switched off (true fp32)"
                              if a.precision == "fp32" else "bf16: DIT's own bf16 block autocast, as released")}
    key = f"duo_{mode}" if a.precision == "fp32" else f"duo_{mode}_bf16"
    update_results(a.out, lambda doc: put_row(doc, a.table, key, row))
    log(f"table {a.table} {key} -> {a.out}: gen ppl {row.get('gen_ppl_mean')}, entropy {row.get('entropy_mean')}")


def parse_eval_merge(text):
    per_seed = []
    mean = None
    for ln in text.splitlines():
        m = re.match(r"^\s*(-?\d+)\s+([0-9.]+|nan|inf)\s+([0-9.]+|nan)\s*$", ln)
        if m:
            per_seed.append({"seed": int(m.group(1)), "gen_ppl": float(m.group(2)), "entropy": float(m.group(3))})
            continue
        m = re.match(r"^\s*mean\s+([0-9.]+|nan|inf)\s+([0-9.]+|nan)\s+\(\+/-\s*([0-9.]+|nan) over (\d+) seeds, (\d+) samples\)", ln)
        if m:
            mean = {"gen_ppl_mean": float(m.group(1)), "entropy_mean": float(m.group(2)), "gen_ppl_sd": float(m.group(3)),
                    "n_seeds": int(m.group(4)), "n_samples": int(m.group(5))}
    if mean is None:
        die("could not find eval_merge.py's mean line")
    return {**mean, "per_seed": per_seed}


# ================================================================ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=["ncm", "duo"], required=True)
    ap.add_argument("--cfg", help="NCM config (from make_configs.py)")
    ap.add_argument("--ckpt", help="DUO checkpoint")
    ap.add_argument("--ckpt-dir", default="/dl/ncm/lm1b/small", help="NCM checkpoint dir (eval_shard's --checkpoint-dir)")
    ap.add_argument("--ncm-root", default="/work/ncm")
    ap.add_argument("--duo-root", default="/work/duo")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--kv-cache", choices=["off", "on"], default="off")
    ap.add_argument("--kv-source", choices=["none", "pr5"], default="none")
    ap.add_argument("--graphs", choices=["off", "on"], default="off")
    ap.add_argument("--penalties", choices=["config", "off"], default="config")
    ap.add_argument("--l0", default="unigram")
    ap.add_argument("--noise-removal", choices=["ancestral", "greedy"], default="ancestral")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--count-passes", action="store_true")
    ap.add_argument("--save-ids", action="store_true")
    ap.add_argument("--log-attn", action="store_true")
    ap.add_argument("--count-truncated", action="store_true")
    ap.add_argument("--table", default="X")
    ap.add_argument("--out", default="/results/results.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--scorer", default="genppl_flm")
    ap.add_argument("--ref", default="/dl/gpt2-large")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    ap.add_argument("--pr5-record", default="/results/ncm-pr5.txt")
    a = ap.parse_args()

    if a.dry_run:
        a.count_passes = True
    if a.model == "ncm" and not a.cfg:
        ap.error("--cfg is required for --model ncm")
    if a.model == "duo" and not a.ckpt:
        ap.error("--ckpt is required for --model duo")
    if a.quality and a.model != "duo":
        ap.error("--quality is for DUO; NCM quality comes from its own eval_shard.py + eval_merge.py")

    import torch
    if not torch.cuda.is_available():
        die("no CUDA device")
    if (12, 0) != torch.cuda.get_device_capability(0) or "sm_120" not in torch.cuda.get_arch_list():
        die(f"expected an sm_120 GPU and torch built for it; got {torch.cuda.get_device_capability(0)} {torch.cuda.get_arch_list()}")
    if not a.dry_run and os.environ.get("HF_HUB_OFFLINE") != "1":
        die("timed and quality rows run with HF_HUB_OFFLINE=1 (step 5)")
    (ncm_main if a.model == "ncm" else duo_main)(a, torch)


if __name__ == "__main__":
    main()
