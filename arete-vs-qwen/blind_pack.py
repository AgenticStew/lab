#!/usr/bin/env python3
"""Build the blind answer pack from the two quality runs.

Stops if any pair differs in rendered prompt, input token IDs or sampling settings.
Runs the hidden objective checks on every answer. Shuffles LEFT/RIGHT independently per item (SystemRandom).
Writes:
- blind/answers.md: what the scorer reads (no model names);
- results/_key/mapping.json: the key, kept away from the scorer;
- results/objective.json: per-model objective results.
"""
import json, random, sys
from pathlib import Path

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B / "tests"))
from hidden import check  # noqa: E402


def load(tag):
    rows = [json.loads(x) for x in (B / "results" / f"{tag}-quality.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if not r.get("meta")]
    assert len(rows) == 32, (tag, len(rows))
    return {r["id"]: r for r in rows}


def main():
    suf = sys.argv[1] if len(sys.argv) > 1 else ""
    fixture = {x["id"]: x for x in map(json.loads, (B / "prompts.jsonl").read_text().splitlines())}
    runs = {"qwen": load("qwen"), "arete": load("arete")}
    for i in fixture:
        q, a = runs["qwen"][i], runs["arete"][i]
        assert q["prompt_hash"] == a["prompt_hash"], (i, "template mismatch")
        assert q["prompt_token_ids"] == a["prompt_token_ids"], (i, "token mismatch")
        assert q["settings"] == a["settings"], (i, "sampling mismatch")
    objective = {t: {i: check(i, runs[t][i]["text"]) for i in fixture} for t in runs}
    (B / "results" / "objective.json").write_text(json.dumps(objective, indent=1))
    rng = random.SystemRandom()
    key, parts = {}, ["# Blind answer pack\n\nLEFT/RIGHT are shuffled independently for every item. "
                      "Score with bench/RUBRIC.md. Do not open results/_key.\n"]
    for i in sorted(fixture):
        pair = ["qwen", "arete"]
        rng.shuffle(pair)
        key[i] = {"LEFT": pair[0], "RIGHT": pair[1]}
        it = fixture[i]
        parts.append(f"\n---\n\n## {i} ({it['category']})\n\n**Prompt:** {it['prompt']}\n\n**Predeclared checks:** {it['checks']}\n")
        for side, tag in zip(("LEFT", "RIGHT"), pair):
            r = runs[tag][i]
            obj = objective[tag][i]
            meta = f"{r['tokens_predicted']} tokens, stop={r['stop_type']}" + (", TRUNCATED at cap" if r["stop_type"] == "limit" else "")
            parts.append(f"\n### {side}  ({meta})\n")
            if obj:
                parts.append(f"\n*Hidden checks: {obj['pass']}/{obj['total']}"
                             + (f"; failed: {'; '.join(obj['notes'])}" if obj["notes"] else "") + "*\n")
            parts.append("\n````text\n" + r["text"].strip() + "\n````\n")
    (B / "blind").mkdir(exist_ok=True)
    (B / "blind" / f"answers{suf}.md").write_text("".join(parts), encoding="utf-8")
    kd = B / "results" / "_key"
    kd.mkdir(exist_ok=True)
    (kd / f"mapping{suf}.json").write_text(json.dumps(key, indent=1))
    print("All 32 pairs matched on rendered prompt, token IDs and settings. Pack: blind/answers.md")
    for t in runs:
        tot = [objective[t][i] for i in fixture if objective[t][i]]
        print(t, "objective", sum(o["pass"] for o in tot), "/", sum(o["total"] for o in tot))


if __name__ == "__main__":
    main()
