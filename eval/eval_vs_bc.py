"""
eval_vs_bc.py  ―  モデル A vs モデル B の直接対戦（混在卓 A×2 + B×2）

  python3 eval_vs_bc.py --model-a checkpoints1/grpo/model_grpo_final.pt \
                        --model-b checkpoints1/model_bc_best.pt --games 400

[今回の改修]
  - 平均順位差に標準誤差と 95% 信頼区間を付ける。旧版は 200半荘で差 0.01 を「改善」と判定して
    いたが、この規模の標準誤差は 0.05〜0.08 あり、0.01 は誤差の中。
    同卓の A 2人 / B 2人 は順位が負に相関するので、誤差は「半荘ごとの (A平均順位 − B平均順位)」
    から計算する。
  - モデル構成は checkpoint から自動復元（--d-model 等は不要）。
  - run_match() を関数化（train_grpo.py の定期評価からも使う）。
"""
from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
import argparse
import math
import random
from pathlib import Path

import torch

from mahjong_model import load_policy
from self_play import MahjongSimulator

PATTERNS = [{0: "A", 1: "B", 2: "A", 3: "B"}, {0: "B", 1: "A", 2: "B", 3: "A"},
            {0: "A", 1: "A", 2: "B", 3: "B"}, {0: "B", 1: "B", 2: "A", 3: "A"},
            {0: "A", 1: "B", 2: "B", 3: "A"}, {0: "B", 1: "A", 2: "A", 3: "B"}]


def _new_stat():
    return {"seats": 0, "rank_sum": 0, "ranks": [0] * 4, "kyoku": 0, "agari": 0, "houju": 0,
            "agari_pts": 0, "houju_pts": 0, "riichi": 0}


def run_match(model_a, model_b, device, games: int, seed: int | None = None, progress: bool = False) -> dict:
    if seed is not None:
        random.seed(seed); torch.manual_seed(seed)
    sim = MahjongSimulator(model_a, device, watch=False)
    S = {"A": _new_stat(), "B": _new_stat()}
    diffs = []
    it = range(games)
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, desc="games")
        except ImportError:
            pass
    for gi in it:
        seats = PATTERNS[gi % len(PATTERNS)]
        sim.player_models = {p: (model_a if seats[p] == "A" else model_b) for p in range(4)}
        ev = sim.simulate_game()
        n_kyoku = sum(e["type"] == "start_kyoku" for e in ev)
        for e in ev:
            if e["type"] == "hora":
                s = S[seats[e["actor"]]]; s["agari"] += 1; s["agari_pts"] += max(0, e.get("score", 0))
                if e.get("target") is not None:
                    t = S[seats[e["target"]]]; t["houju"] += 1
                    t["houju_pts"] += -e["deltas"][e["target"]]
            elif e["type"] == "reach":
                S[seats[e["actor"]]]["riichi"] += 1
        scores = ev[-1]["scores"]
        order = sorted(range(4), key=lambda i: (-scores[i], i))
        rk = {p: r + 1 for r, p in enumerate(order)}
        for p in range(4):
            s = S[seats[p]]; s["seats"] += 1; s["rank_sum"] += rk[p]; s["ranks"][rk[p] - 1] += 1
            s["kyoku"] += n_kyoku
        ra = [rk[p] for p in range(4) if seats[p] == "A"]; rb = [rk[p] for p in range(4) if seats[p] == "B"]
        diffs.append(sum(ra) / 2 - sum(rb) / 2)
    n = max(len(diffs), 1)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)
    se = math.sqrt(var / n)
    return {"A": S["A"], "B": S["B"], "rank_diff": mean, "se": se, "games": len(diffs)}


def report(res: dict):
    for name in ("A", "B"):
        s = res[name]; k = max(s["kyoku"], 1); n = max(s["seats"], 1)
        print(f"[{name}] 平均順位 {s['rank_sum']/n:.3f}  (1位{s['ranks'][0]} 2位{s['ranks'][1]} "
              f"3位{s['ranks'][2]} 4位{s['ranks'][3]})  和了率 {s['agari']/k*100:.2f}%  "
              f"放銃率 {s['houju']/k*100:.2f}%  リーチ率 {s['riichi']/k*100:.2f}%  "
              f"平均和了 {s['agari_pts']/max(s['agari'],1):.0f}  平均放銃 {s['houju_pts']/max(s['houju'],1):.0f}")
    d, se = res["rank_diff"], res["se"]
    print(f"平均順位差 (A−B) = {d:+.4f} ± {se:.4f} (SE)   95%CI [{d-1.96*se:+.4f}, {d+1.96*se:+.4f}]"
          f"   n={res['games']}半荘")
    if d + 1.96 * se < 0:   print("→ A が有意に強い")
    elif d - 1.96 * se > 0: print("→ B が有意に強い")
    else:                   print("→ 有意差なし（この半荘数では判別できない差）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", type=Path, required=True)
    ap.add_argument("--model-b", type=Path, required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mA, mB = load_policy(args.model_a, device), load_policy(args.model_b, device)
    report(run_match(mA, mB, device, args.games, args.seed, progress=True))


if __name__ == "__main__":
    main()
