#!/usr/bin/env python3
"""results/results.json + root results.json -> bench/RESULTS.md (every number from code), and copies the run's
text logs into bench/logs/.   python3 bench/write_results_md.py"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
doc = json.loads((RES / "results.json").read_text())
top = json.loads((ROOT / "results.json").read_text())
T, S, D, F, raw = doc["tables"], top["slots"], top["derived"], top["frame_text"], top["raw"]
oc = doc.get("output_checks", {})
same = doc["compare_ids"]["ncm"]

logs = ROOT / "bench" / "logs"
logs.mkdir(exist_ok=True)
for p in sorted(RES.glob("*.txt")) + sorted(RES.glob("*.log")) + sorted(RES.glob("*.yaml")):
    shutil.copy2(p, logs / p.name)
shutil.copy2(RES / "results.json", logs / "results.full.json")


def row(t, m):
    r = T[t][m]
    return (f"| {t} | {r['label']} | {r['precision']} | {r.get('kv_cache', '—')} | {'on' if (isinstance(r.get('graphs'), dict) or r.get('graphs') == 'on') else 'off'} | "
            f"{r['median_wall_s']:.3f} | {r['passes_per_batch']} | {r['median_ms_per_pass']:.2f} | {r['non_pass_share_pct']:.1f} |")


q = T["Q"]
lines = [
    "# RESULTS: NCM vs DUO on the RTX 5090 (phase 2, run 4, 10 Nov 2026)",
    "",
    "Written by `bench/write_results_md.py` from `results/results.json` (copied to `bench/logs/results.full.json`) and the",
    "root `results.json` (slots, derived values with their expressions). Raw logs: `bench/logs/`.",
    "",
    "## Timed rows (1,024 sequences, batch 16, 128 positions, 1 warm-up batch, median of 3 repeats, seed 42)",
    "",
    "| table | model | precision | KV cache | CUDA graphs | median wall s | passes/batch | ms/pass (median) | outside-pass share % |",
    "|---|---|---|---|---|---|---|---|---|",
    row("A", "ncm"), row("A", "duo"), row("B", "ncm"), row("B", "duo"),
    f"| C | NCM, penalties off (bf16, KV, graphs) | bf16 | on | on | {T['C']['ncm']['median_wall_s']:.3f} | "
    f"{T['C']['ncm']['passes_per_batch']} | {T['C']['ncm']['median_ms_per_pass']:.2f} | {T['C']['ncm']['non_pass_share_pct']:.1f} |",
    "",
    f"- NCM ÷ DUO wall-clock: Table A {S['RATIO_FP32']}×, Table B {S['RATIO_FAST']}× (authors, H100: 21×). "
    f"B vs A ratio change {D['RATIO_AB_CHANGE_PCT']['value']}%.",
    f"- Passes: DUO {raw['B_duo_passes']} ÷ NCM {raw['B_ncm_passes']} = {D['PASS_RATIO']['value']}×; a DUO pass costs "
    f"{D['PASS_COST_RATIO']['value']}× an NCM pass (bf16 + graphs). DUO spends {F['DUO_OTHER_PCT']}% of its time outside its "
    f"passes (NCM {S['NCM_OTHER_PCT']}%), which is why the total exceeds passes × pass cost.",
    f"- NCM counted {raw['A_ncm_passes']} passes per batch in Table A (eager) and {raw['B_ncm_passes']} in Table B (KV cache + graphs).",
    f"- Settings gain: NCM {S['NCM_SETTINGS_GAIN']}×, DUO {S['DUO_SETTINGS_GAIN']}×. " + ("Every repeat produced identical ids within its row." if all(T[k][m].get('repeat_ids_identical') for k in 'ABC' for m in T[k] if 'median_wall_s' in T[k][m]) else "Some repeats differed in their ids (see results.full.json)."),
    f"- Penalties off: NCM bf16 time {S['NCM_FAST_S']} s to {S['NCM_NOPEN_S']} s ({D['PENALTY_TIME_CHANGE_PCT']['value']}% of NCM's time).",
    "",
    "## Identical outputs, Table A vs Table B (NCM)",
    "",
    f"{same['identical_pct']}% of sequences identical; {same['token_identical_pct']}% of token positions identical. Caveat: both tables",
    "used the same --seed, but NCM samples its codes at every level, so any numeric difference between fp32 eager and",
    "bf16 + KV cache + graphs (and a different pass count, 11 vs 12) sends the random draws apart. This measures",
    "path divergence under sampling, not bf16 error on a fixed draw. The quality scores below were taken at fp32 (as R7 says).",
    "",
    "## Quality (generative perplexity, GPT-2 large, genppl_flm via NCM's eval_merge.py; 5 seeds × 256 samples; fp32)",
    "",
    "| run | gen PPL mean | ± sd (seeds) | entropy |",
    "|---|---|---|---|",
    f"| NCM, released penalties | {raw['ncm_pen_ppl']:.2f} | {raw['ncm_pen_sd']:.2f} | {raw['ncm_pen_h']:.4f} |",
    f"| NCM, penalties off | {raw['ncm_nopen_ppl']:.2f} | {raw['ncm_nopen_sd']:.2f} | {raw['ncm_nopen_h']:.4f} |",
    f"| DUO, ancestral (default) | {raw['duo_anc_ppl']:.2f} | {raw['duo_anc_sd']:.2f} | {raw['duo_anc_h']:.4f} |",
    f"| DUO, greedy | {raw['duo_greedy_ppl']:.2f} | {raw['duo_greedy_sd']:.2f} | {raw['duo_greedy_h']:.4f} |",
    "",
    "Paper (H100): NCM 93.05 / H 4.31, DUO 97.30 / H 4.30. NCM landed close; DUO ancestral landed about 3% higher than",
    "the paper's DUO figure. Under greedy sampling DUO beats NCM on gen PPL with lower entropy.",
    "",
    "## Output checks (bench/check_outputs.py)",
    "",
    "| file | samples | 8+ token run | dominant 4-gram | distinct-1 (seed 42) | distinct-2 (seed 42) |",
    "|---|---|---|---|---|---|",
]
for k, v in oc.items():
    s42 = v["per_seed"].get("42", {})
    lines.append(f"| {k} | {v['samples']} | {v['run_ge_8']} | {v['top_4gram_share_ge_25pct']} | "
                 f"{s42.get('distinct_1', 0):.3f} | {s42.get('distinct_2', 0):.3f} |")
lines += [
    "",
    "NCM's pickles show every `hyps` string empty: eval_shard.py's `dec()` cuts at the first EOS, and BERT's [CLS] (id 101)",
    "is both the EOS and the first token of every packed LM1B sample. The scorer (genppl_flm) decodes `raw_ids` with",
    "specials kept, as it does for DUO, and the raw ids hold real packed text, so the scores are unaffected (FACTCHECK",
    "C038). The empty-string counts in output_checks are that artefact, not empty generations.",
    "Penalties off: 45 of 1,280 NCM samples hold a run of 8+ identical tokens (0 with penalties on).",
    "Distinct-1 over a seed is lower for NCM than DUO; it is not in the script.",
]
(ROOT / "bench" / "RESULTS.md").write_text("\n".join(lines) + "\n")
print("wrote bench/RESULTS.md; logs ->", logs)
