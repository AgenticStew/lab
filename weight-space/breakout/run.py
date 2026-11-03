"""Rerun GPT-6 Astra's one-shot Breakout network (Experiment 5) on the real MinAtar 1.0.15 game, with the authors' scoring
protocol: 1,000 episodes, actions sampled from the softmax of the logits, sticky actions 0.1, episodes cut at 10,000 steps.
It also records a few episodes step by step (paddle, ball, trail, bricks, score, the network's own landing prediction).

  python run.py                         -> results/summary.json, results/episodes.json, results/rec_<seed>.json
  python run.py --first-seed 1000       -> another batch of 1,000 games (seeds 1000-1999)
  python run.py --record 450            -> also record seed 450

The network is read from a clone of the authors' logs repo (default: ../lm_understand-logs, see README.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from minatar import Environment

HERE = Path(__file__).resolve().parent
NETWORK = "exp5_write_from_scratch/networks/breakout/hand.npz"
# game channels (breakout.py): paddle 0, ball 1, trail 2, brick 3 -> the prompt's 6 channels: player 0, reward 2, projectile 3, trail 4
GAME2NET = {0: 0, 1: 3, 2: 4, 3: 2}
MAX_STEPS = 10_000
W1 = b1 = W2 = b2 = W3 = b3 = None


def load(path):
    global W1, b1, W2, b2, W3, b3
    D = np.load(path)
    W1, b1 = D["actor/dense1/kernel"], D["actor/dense1/bias"]
    W2, b2 = D["actor/dense2/kernel"], D["actor/dense2/bias"]
    W3, b3 = D["actor/logits/kernel"], D["actor/logits/bias"]


def observe(env) -> np.ndarray:
    s = env.state()
    x = np.zeros((10, 10, 6), np.float32)
    for g, n in GAME2NET.items():
        x[..., n] = s[..., g]
    return x.reshape(-1)


def episode(seed: int, record: bool = False):
    env = Environment("breakout", sticky_action_prob=0.1)
    env.seed(seed)
    env.reset()
    rng = np.random.default_rng(1_000_003 + seed)
    score, steps, rec = 0, 0, []
    g = env.env
    while steps < MAX_STEPS:
        x = observe(env)
        z1 = x @ W1 + b1
        h2 = np.tanh(np.tanh(z1) @ W2 + b2)
        lg = (h2 @ W3 + b3).astype(np.float64)
        p = np.exp(lg - lg.max())
        p /= p.sum()
        a = int(rng.choice(6, p=p))
        if record:
            # the network's own landing target: units 0..3 = (descending-right, descending-left, ascending-right, ascending-left),
            # each ~0.02 x (target column - paddle column) before the tanh (answer.md, "Subtract paddle position from each target")
            dx, dy = g.ball_x - g.last_x, g.ball_y - g.last_y
            unit = (0 if dx >= 0 else 1) if dy >= 0 else (2 if dx >= 0 else 3)
            target = g.pos + float(z1[unit]) / 0.02
            rec.append([int(g.pos), int(g.ball_x), int(g.ball_y), int(g.last_x), int(g.last_y),
                        int("".join(str(int(v)) for v in g.brick_map[1:4].reshape(-1)), 2), score, a, round(target, 2)])
        r, term = env.act(a)
        score += r
        steps += 1
        if term:
            break
    if record:
        rec.append([int(g.pos), int(g.ball_x), int(g.ball_y), int(g.last_x), int(g.last_y),
                    int("".join(str(int(v)) for v in g.brick_map[1:4].reshape(-1)), 2), score, -1, None])
    return seed, score, steps, bool(g.terminal), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(HERE.parent / "lm_understand-logs"), help="clone of RikiyaT/lm_understand-logs")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--record", type=int, nargs="*", default=[], help="extra seeds to record step by step")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    hand = Path(a.logs) / NETWORK
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    load(hand)
    t0 = time.time()
    seeds = range(a.first_seed, a.first_seed + a.episodes)
    with Pool(a.workers, initializer=load, initargs=(hand,)) as pool:
        res = pool.map(episode, seeds, chunksize=4)
    scores = np.array([r[1] for r in res], float)
    lens = np.array([r[2] for r in res], float)
    summary = {
        "network": f"lm_understand-logs/{NETWORK}",
        "minatar": "1.0.15", "episodes": a.episodes, "first_seed": a.first_seed, "sticky_action_prob": 0.1, "max_steps": MAX_STEPS,
        "mean": round(float(scores.mean()), 3), "se": round(float(scores.std(ddof=1) / np.sqrt(len(scores))), 3),
        "median": float(np.median(scores)), "max": float(scores.max()), "min": float(scores.min()),
        "mean_length": round(float(lens.mean()), 1), "hit_step_cap": int(sum(1 for r in res if not r[3])),
        "authors_mean": 517.411, "authors_se": 7.417, "authors_mean_length": 7417.202,
        "ppo500_best_seed": 25.3, "ppo100_best_seed": 15.8,
        "wall_s": round(time.time() - t0, 1),
    }
    tag = "" if a.first_seed == 0 else f"_from{a.first_seed}"
    (out / f"summary{tag}.json").write_text(json.dumps(summary, indent=1))
    (out / f"episodes{tag}.json").write_text(json.dumps([{"seed": r[0], "score": r[1], "steps": r[2], "terminal": r[3]} for r in res]))
    print(json.dumps(summary, indent=1))
    # record: the episode closest to the mean, the best one, and any seeds asked for
    order = sorted(res, key=lambda r: abs(r[1] - scores.mean()))
    picks = {order[0][0], max(res, key=lambda r: r[1])[0], *a.record}
    for s in sorted(picks):
        _, sc, st, _, rec = episode(s, record=True)
        (out / f"rec_{s}.json").write_text(json.dumps({"seed": s, "score": sc, "steps": st, "cols": ["pos", "ball_x", "ball_y", "last_x", "last_y",
                                                       "bricks_rows1to3_bits", "score", "action", "net_target_col"], "steps_data": rec}))
        print(f"recorded seed {s}: score {sc}, {st} steps")


if __name__ == "__main__":
    sys.exit(main())
