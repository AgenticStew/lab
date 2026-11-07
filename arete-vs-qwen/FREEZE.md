# Freeze record

Frozen 2026-11-07T12:31:34+00:00, before any Arete or Qwen generation. Any later change to these files is logged below.

```
cdcb56040e0010910736ccba211cc85a7a022b84f42304937e74c43bbbbe6f43  prompts.jsonl
b68bf092bca7632e66e6be469172a5a554c507f3add5b777c27510baac1a7098  tests/hidden.py
3996014dcd502552f549ef56794e921939eea0e22b4fe1f4df284c47451c8f77  RUBRIC.md
1ee5d03fe5de30e2046d52e6b1c65238fe6156da8e08de80d1272554e7c7e81c  run_quality.py
dbec6835b5a35346a9f6f3ec17163b35812d8df0c905d75c67d8a6bda7c59ed6  prompts.json
1a9d9c0d52e01bd14a4de7449b3d14d414a70e7dc355db78d370c0b03b43a9f3  harness.py
```

## Post-run harness fix (logged 7 Nov, after the quality runs, before any scoring)

`tests/hidden.py` (frozen copy kept as `tests/hidden.frozen.py`):
- **Unclosed final code block:** `extract_code` now also reads a final block that was cut off by the token cap.
- **Model's own tests:** `check_py` now also drops a model's own unittest/pytest tests (`class Test…`,
  `import unittest`, `def test_`), as its frozen comment already intended.

Applied to both models. Before the fix, Arete's C02 scored 0/6: its answer hit the 1,536-token cap inside a long
unittest class, and the parser crashed on the unclosed block. After the fix it scores 6/6. Qwen's results are
unchanged (38/39 objective before and after). Two blind packs (answers_A.md, answers_B.md) with independent shuffles
go to two independent scorers.

## Published copies

The hashes above are of the files as frozen. Before publishing, a few comment and reference lines were edited: harness.py's timing-flag path (a local temp folder) became /tmp/bench_timing.flag; run_quality.py's docstring, RUBRIC.md's title and the P01/P04 reference notes in prompts.jsonl lost notes pointing at private planning documents; local model paths became ~ paths. No prompt, answer-key fact or rubric criterion changed. SHA-256 of the published copies:

```
3dc515bb2e99953fa12aa1e4b7750f8f077bb333e299af597db588a40de110b7  prompts.jsonl
de2c359538dd7063819d9fd76f5b47c8a7554f3243a0c824a6622b2243cb4589  tests/hidden.py
ed74a360cb074dc7752c2293721e3752dd567bc662fe227f5bfa222b6f6d1ae6  RUBRIC.md
6c205cf9a39411e1d44066dedd90edcfd14a465d97f0a0b5575c7ae5b9633c54  run_quality.py
dbec6835b5a35346a9f6f3ec17163b35812d8df0c905d75c67d8a6bda7c59ed6  prompts.json
fc0ef94b312a6c169731f966eef5f69c4a4642f21e6a4fe873fb8f276515a2db  harness.py
```
