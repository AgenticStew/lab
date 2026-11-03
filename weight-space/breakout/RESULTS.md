# Rerun of GPT-6 Astra's one-shot Breakout network (Experiment 5)

Run on 3 Nov 2026 on my test PC: Ryzen 9 9950X3D, 64 GB DDR5-6000 (2 × 32 GB), Windows 11 + WSL2 Ubuntu 24.04. CPU only,
no GPU needed. Software: Python 3.12, numpy 2.5, MinAtar 1.0.15 (installed with `--no-deps`).

## What was run
- **The network:** `lm_understand-logs/exp5_write_from_scratch/networks/breakout/hand.npz`, exactly as the authors read it
  from Astra's reply.
- **The protocol (`run.py`):** the authors' own scoring protocol (exp5 README):
  - 1,000 episodes of MinAtar Breakout;
  - actions sampled from the softmax of the logits;
  - sticky actions 0.1;
  - episodes cut at 10,000 steps.
- **The input mapping:** from the prompt (`message.txt`, "THE INPUT"): paddle → channel 0, brick → 2, ball → 3, trail → 4.

## Results

| | Mean score | SE | Games that hit the 10,000-step cap | Mean length |
|---|---|---|---|---|
| **My rerun** (seeds 0–999, `results/summary.json`) | **502.9** | 7.7 | 517 / 1,000 | 7,188 |
| Extra batches (seeds 1000–3999, `results/summary_from*.json`) | 507.8 / 494.0 / 508.8 | 7.7 / 7.9 / 7.7 | 528 / 491 / 516 | — |
| Authors (`eval/minatar_eval.json`, `breakout.hand`) | 517.4 | 7.4 | — | 7,417 |
| Authors, PPO 500M steps, best of 3 seeds | 25.3 | — | — | — |
| Authors, PPO 100M steps, best of 10 seeds | 15.8 | — | — | — |

**Reading the gap:** my mean is within the spread of my own batches (494–509), so the gap to the authors' 517.4 is seed
noise.

**Never missing:** MinAtar Breakout has one life, so a game ends at the first miss. About half the games never missed and
were stopped at 10,000 steps.

## A scripted baseline: the strategy is simple
**What it is:** `scripted_rule.py keepdir`, a plain-Python rule:
- simulate the ball with the game's own physics until it reaches the paddle row;
- move the paddle there.

The core is about 30 lines.

**Results,** same protocol:

| Rule | Mean | SE | Capped games |
|---|---|---|---|
| The scripted rule (`keepdir`) | **533.9** | 9.5 | 408 / 1,000 |
| Wall reflections only (`walls`), no lookahead past the walls | 11.8 | 0.3 | 0 |

**What it means:** the PPO baseline is weak at this game, and a simple program does as well as Astra's network. The feat is
that Astra wrote such a program directly as network weights, in one reply, without running anything.

## Recorded games
`results/rec_<seed>.json` hold step-by-step states: paddle, ball, trail, bricks, score, action, and the network's own
landing target. The target is read from first-layer units 0–3 (≈ 0.02 × (target − paddle) before the tanh).
These are the games replayed in the video.

| Seed | Score | Steps | Why this one |
|---|---|---|---|
| 784 | 506 | 6,930 | closest to the mean |
| 450 | 694 | 10,000 | median of the capped games (`--record 450`) |
| 125 | 821 | 10,000 | best game |

## Reproduce

```bash
python run.py --record 450                          # results/summary.json, episodes.json, rec_*.json (a few seconds on 32 threads)
for s in 1000 2000 3000; do python run.py --first-seed $s; done
python scripted_rule.py keepdir
python scripted_rule.py walls
```

Each game's randomness is seeded from its seed number, so these reproduce the files in `results/` exactly (apart from
`wall_s`), whatever the number of worker processes.
