"""
calib_kyoku_value.py  ―  打ち切り推定 vs 完全プレイアウト の一致度検証

  python3 debug/calib_kyoku_value.py --model train/checkpoints1/model_bc_best.pt \
      --branches 100 --horizon 6 --n-truth 24 --n-est 3

何を測るか:
  同じ分岐点の同じ枝について、
    truth : 局終了まで N本(--n-truth)回した報酬の平均（分散は大きいが偏りはない基準）
    est   : horizon 巡で打ち切り、kyoku_value の期待局収支で評価した報酬（--n-est 本の平均）
  を両方求め、
    - 枝の順位がどれだけ一致するか（グループ内 Spearman、GRPO が実際に使うのはこれ）
    - 最良枝が一致した割合
    - 値そのものの相関と系統的なズレ
  を出す。

判断の目安:
  グループ内順位相関の平均が 0.5 を超え、最良枝一致が 0.5 以上なら、打ち切りに
  置き換えてよい。低い場合は kyoku_value.py 先頭のパラメータを調整する。
  なお GRPO はグループ内の相対値しか使わないので、全枝に共通して乗るズレ
  （下の "平均バイアス"）は問題にならない。効くのは順位の一致。
"""

from __future__ import annotations
import _pathsetup  # noqa: F401
import argparse
import math
import random
from pathlib import Path

import torch

from mahjong_model import load_policy, disable_dropout
from self_play import MahjongSimulator, RLContext, resample_wall
from rollout import RolloutConfig, _pick_branch_actions, _reward_of
from mahjong_tokenizer import PADDING_ACTION_ID


def _spearman(a: list, b: list) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    def rank(xs):
        order = sorted(range(n), key=lambda i: xs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da > 0 and db > 0 else float("nan")


def _pearson(a: list, b: list) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else float("nan")


def eval_branch(sim, snap, rec, a_idx, cfg, worlds, horizon):
    """
    1枝を、完全プレイアウトと打ち切り推定の両方で評価する（同じ山を共有）。
    戻り値は (truth平均, est平均, truth前半平均, truth後半平均)。
    後ろ2つは「基準側自身のばらつき」を測るためのもの（信頼性の天井）。
    """
    truth, est = [], []
    for W in worlds:
        rl = RLContext(rl_player=rec.player, temperature=cfg.temperature,
                       forced_first_idx=a_idx, continuation=cfg.continuation,
                       take_snapshots=False)
        res, ev = sim.playout_from(W, rl)
        truth.append(_reward_of(cfg, snap, res, ev, rec.player))
    h = len(truth) // 2
    t_a = sum(truth[:h]) / max(h, 1)
    t_b = sum(truth[h:]) / max(len(truth) - h, 1)
    for W in worlds[:cfg.n_playout]:
        rl = RLContext(rl_player=rec.player, temperature=cfg.temperature,
                       forced_first_idx=a_idx, continuation=cfg.continuation,
                       take_snapshots=False)
        res, ev = sim.playout_from(W, rl, max_turns=horizon)
        est.append(_reward_of(cfg, snap, res, ev, rec.player))
    return sum(truth) / len(truth), sum(est) / len(est), t_a, t_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--branches", type=int, default=100, help="評価する分岐点の数")
    ap.add_argument("--horizon", type=int, default=6, help="打ち切りまでの巡数")
    ap.add_argument("--n-truth", type=int, default=24, help="基準側のプレイアウト本数")
    ap.add_argument("--n-est", type=int, default=3, help="打ち切り側の本数")
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = disable_dropout(load_policy(args.model, device))
    sim = MahjongSimulator(model, device, watch=False)
    sim.player_models = {s: model for s in range(4)}
    cfg = RolloutConfig(G=args.G, n_playout=args.n_est)
    rng = random.Random(args.seed)
    random.seed(args.seed); torch.manual_seed(args.seed)

    rhos, tops, all_t, all_e, biases, ceilings = [], [], [], [], [], []
    done = 0
    while done < args.branches:
        rl = RLContext(rl_player=rng.randrange(4), temperature=cfg.temperature)
        sim.capture_game(rl)
        recs = [r for r in rl.records if r.snapshot is not None and r.n_valid >= 2]
        rng.shuffle(recs)
        for rec in recs[:4]:
            if done >= args.branches:
                break
            idxs = _pick_branch_actions(rec, cfg.G, cfg.p_floor, rng)
            if len(idxs) < 2:
                continue
            snap = rec.snapshot
            worlds = [resample_wall(snap, rng) for _ in range(args.n_truth)]
            t_vals, e_vals, a_vals, b_vals = [], [], [], []
            for a in idxs:
                t, e, ta, tb = eval_branch(sim, snap, rec, a, cfg, worlds, args.horizon)
                t_vals.append(t); e_vals.append(e); a_vals.append(ta); b_vals.append(tb)
            if max(t_vals) - min(t_vals) < cfg.tie_eps:
                continue
            ceilings.append(_spearman(a_vals, b_vals))
            rhos.append(_spearman(t_vals, e_vals))
            tops.append(1.0 if t_vals.index(max(t_vals)) == e_vals.index(max(e_vals)) else 0.0)
            biases.append(sum(e_vals) / len(e_vals) - sum(t_vals) / len(t_vals))
            all_t += t_vals; all_e += e_vals
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{args.branches} 分岐点  "
                      f"順位相関MA {sum(rhos[-10:])/len(rhos[-10:]):+.3f}", flush=True)

    rhos_ok = [r for r in rhos if r == r]      # NaN 除去
    ceil_ok = [r for r in ceilings if r == r]
    ceil = sum(ceil_ok) / max(len(ceil_ok), 1)
    print("=" * 60)
    print(f"分岐点 {len(rhos)}  (horizon={args.horizon}巡, truth N={args.n_truth}, est N={args.n_est})")
    rho = sum(rhos_ok) / max(len(rhos_ok), 1)
    print(f"  グループ内 順位相関の平均 : {rho:+.3f}")
    print(f"  基準側じたいの再現性(天井): {ceil:+.3f}   ← truth を半分ずつに割った同士の相関")
    print(f"  天井に対する達成率        : {rho/ceil if ceil > 0 else float('nan'):+.3f}   ← 0.7 以上なら置き換え可")
    print(f"  最良枝の一致率            : {sum(tops)/max(len(tops),1):.3f}   ← ランダムなら 1/G")
    print(f"  全枝での相関 (Pearson)    : {_pearson(all_t, all_e):+.3f}")
    print(f"  平均バイアス (est − truth): {sum(biases)/max(len(biases),1):+.4f}   ← 相対比較には影響しない")
    print("=" * 60)


if __name__ == "__main__":
    main()
