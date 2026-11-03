# SPDX-License-Identifier: GPL-3.0-only
"""A scripted Breakout player to compare with GPT-6 Astra's network: a plain-Python rule that predicts where the ball
reaches the paddle row and moves the paddle there. Same protocol as run.py (1,000 episodes, sticky actions 0.1,
10,000-step cap).

sim_full() follows the ball update in MinAtar's breakout.py (Kenny Young and Tian Tian, GPL-3.0), so this file is
GPL-3.0 too: see LICENSE-MinAtar.

  python scripted_rule.py keepdir   # simulate the ball with the game's own update rule (MinAtar breakout.py), bricks included
  python scripted_rule.py walls     # wall reflections only, no lookahead past the bricks
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
from minatar import Environment

MODE = "keepdir"


def land_walls(x, y, dx):
    # descend with wall reflections only, until row 9
    while y < 9:
        nx = x + dx
        if nx < 0 or nx > 9:
            dx = -dx
            nx = min(9, max(0, nx))
        x, y = nx, y + 1
    return x


def sim_full(g):
    # the game's own ball update (no paddle), stepped until the ball reaches row 9; returns (column before, column at row 9)
    x, y, d = g.ball_x, g.ball_y, g.ball_dir
    bm = g.brick_map.copy()
    strike = g.strike
    for _ in range(200):
        lx, ly = x, y
        nx, ny = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)][d]
        st = False
        if nx < 0 or nx > 9:
            nx = min(9, max(0, nx))
            d = [1, 0, 3, 2][d]
        if ny < 0:
            ny = 0
            d = [3, 2, 1, 0][d]
        elif bm[ny, nx] == 1:
            st = True
            if not strike:
                strike = True
                bm[ny, nx] = 0
                ny = ly
                d = [3, 2, 1, 0][d]
        elif ny == 9:
            return (x, nx)
        if not st:
            strike = False
        x, y = nx, ny
    return (x, x)


def episode(seed):
    env = Environment("breakout", sticky_action_prob=0.1)
    env.seed(seed)
    env.reset()
    g = env.env
    score = steps = 0
    while steps < 10000:
        dx, dy = g.ball_x - g.last_x, g.ball_y - g.last_y
        if MODE == "walls":
            tx = land_walls(g.ball_x, g.ball_y, 1 if dx > 0 else -1) if dy > 0 else g.ball_x
        else:
            tx = sim_full(g)[0]
        a = 1 if tx < g.pos else (3 if tx > g.pos else 0)
        r, t = env.act(a)
        score += r
        steps += 1
        if t:
            break
    return score, steps


def set_mode(m):
    global MODE
    MODE = m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["keepdir", "walls"])
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    with Pool(a.workers, initializer=set_mode, initargs=(a.mode,)) as p:
        res = p.map(episode, range(a.episodes), chunksize=4)
    s = np.array([r[0] for r in res], float)
    L = np.array([r[1] for r in res])
    print(a.mode, "mean %.1f se %.1f capped %d mean_len %.0f max %d"
          % (s.mean(), s.std(ddof=1) / np.sqrt(len(s)), (L >= 10000).sum(), L.mean(), s.max()))


if __name__ == "__main__":
    main()
