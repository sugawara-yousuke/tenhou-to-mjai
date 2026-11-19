"""
eval_paired.py  ―  配牌を揃えた対戦（同卓複式 / 共通乱数）でモデル A と B を比べる

  python3 eval/eval_paired.py --model-a weights/model_bc_best.pt --model-b rule --pairs 100

  --model-a / --model-b には .pt のパスか、ルールベースの打ち手を表す "rule" を渡せる。

なぜ必要か:
  eval_vs_bc.py は A と B で別々の対局を打つので、差のほとんどが配牌の運で埋まる。
  1000半荘でも平均順位差の標準誤差は ±0.04 で、1段ぶん（約0.05）の差すら判別できない。
  そこで競技麻雀の同卓複式と同じことをする。

  同じ山・同じ席で2回打つ:
    run1: 席集合 S に A、残り T に B
    run2: 席集合 S に B、残り T に A      ← 入れ替え。山は run1 と同一
    差 = (run1 で S が取った平均順位) − (run2 で S が取った平均順位)

  どちらの測定も「この配牌で席 S に座った打ち手の成績」なので、配牌の運が相殺され、
  残るのは打ち手の差だけになる。

仕組み上の前提（self_play.py の実装に依存）:
  - 評価時のモデルは model_choose（argmax）で決定論的に打つので乱数を消費しない。
  - 評価経路で Python の random を消費するのは _new_kyoku の random.shuffle(deck) だけ。
  したがって random.seed を揃えれば、打牌がどう分岐しても局ごとの山は一致する。

  ※鳴きで打順がずれると以降の自摸は分岐する。配牌は完全に一致するが、相殺は部分的。
    それでも分散はかなり下がる。

動作確認:
  --model-a と --model-b に同じファイルを渡すと、2回の対局は完全に同一になるので
  ペア差は全て 0.0000、SE も 0 になる。ならなければ対応付けが壊れている。

出力:
  - ペア方式の平均順位差（本命）
  - 従来方式（各対局内で A と B を比べる）の平均順位差。同じ対局数で比較するので、
    SE がどれだけ下がったかがそのまま分散削減の効果になる。
  - 放銃率・和了率・リーチ率。局単位なので半荘単位の順位よりサンプルが約9倍多く、収束が速い。
"""
from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
import argparse
import math
import random
from pathlib import Path

import torch

from mahjong_model import load_policy
from rule_based import MixedSimulator, RULE_BASED

# 4席を2人ずつに分ける3通り。ペアごとに回して席の偏りを消す。
SPLITS = [((0, 2), (1, 3)), ((0, 1), (2, 3)), ((0, 3), (1, 2))]


def _mean_se(xs: list) -> tuple:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def play_one(sim: MahjongSimulator, seat_models: dict, seed: int) -> dict:
    """1半荘打って、席ごとの成績を返す。seed を揃えれば山は再現される。"""
    random.seed(seed)
    torch.manual_seed(seed)
    sim.reseed_rule(seed)
    sim.player_models = dict(seat_models)
    ev = sim.simulate_game()

    n_kyoku = sum(e["type"] == "start_kyoku" for e in ev)
    agari = [0] * 4
    houju = [0] * 4
    riichi = [0] * 4
    agari_pts = [0] * 4
    houju_pts = [0] * 4
    for e in ev:
        t = e["type"]
        if t == "hora":
            a = e["actor"]
            agari[a] += 1
            agari_pts[a] += max(0, e.get("score", 0))
            tg = e.get("target")
            if tg is not None:
                houju[tg] += 1
                houju_pts[tg] += -e["deltas"][tg]
        elif t == "reach":
            riichi[e["actor"]] += 1

    scores = ev[-1]["scores"]
    order = sorted(range(4), key=lambda i: (-scores[i], i))     # 同点は起家に近い席が上位
    rank = [0] * 4
    for r, p in enumerate(order):
        rank[p] = r + 1
    return {"rank": rank, "kyoku": n_kyoku, "agari": agari, "houju": houju,
            "riichi": riichi, "agari_pts": agari_pts, "houju_pts": houju_pts}


def _mean_rank(info: dict, seats: tuple) -> float:
    return sum(info["rank"][p] for p in seats) / len(seats)


def _rate(info: dict, seats: tuple, key: str) -> float:
    """その席集合の 1局あたりの発生率。"""
    k = max(info["kyoku"], 1) * len(seats)
    return sum(info[key][p] for p in seats) / k


def run_paired(model_a, model_b, device, pairs: int, seed: int = 20261022,
               progress: bool = True) -> dict:
    sim = MixedSimulator(model_a, device, watch=False)
    paired, unpaired = [], []
    d_houju, d_agari, d_riichi = [], [], []
    acc = {"A": {k: 0 for k in ("agari", "houju", "riichi", "kyoku", "seats",
                                "rank_sum", "agari_pts", "houju_pts")},
           "B": {k: 0 for k in ("agari", "houju", "riichi", "kyoku", "seats",
                                "rank_sum", "agari_pts", "houju_pts")}}
    acc["A"]["ranks"] = [0] * 4
    acc["B"]["ranks"] = [0] * 4

    def absorb(info, seats, who):
        s = acc[who]
        s["seats"] += len(seats)
        s["kyoku"] += info["kyoku"] * len(seats)
        for p in seats:
            s["rank_sum"] += info["rank"][p]
            s["ranks"][info["rank"][p] - 1] += 1
            for k in ("agari", "houju", "riichi", "agari_pts", "houju_pts"):
                s[k] += info[k][p]

    it = range(pairs)
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, desc="pairs")
        except ImportError:
            pass

    for i in it:
        S, T = SPLITS[i % len(SPLITS)]
        s_i = seed + i
        r1 = play_one(sim, {**{p: model_a for p in S}, **{p: model_b for p in T}}, s_i)
        r2 = play_one(sim, {**{p: model_b for p in S}, **{p: model_a for p in T}}, s_i)

        # 本命: 同じ席・同じ配牌での A と B の差
        paired.append(_mean_rank(r1, S) - _mean_rank(r2, S))
        # 比較用: 従来方式（対局内で A 側と B 側を比べる）。1ペアから2点取れる
        unpaired.append(_mean_rank(r1, S) - _mean_rank(r1, T))
        unpaired.append(_mean_rank(r2, T) - _mean_rank(r2, S))
        # 局単位の指標もペアで取る（順位より収束が速い）
        d_houju.append(_rate(r1, S, "houju") - _rate(r2, S, "houju"))
        d_agari.append(_rate(r1, S, "agari") - _rate(r2, S, "agari"))
        d_riichi.append(_rate(r1, S, "riichi") - _rate(r2, S, "riichi"))

        absorb(r1, S, "A"); absorb(r1, T, "B")
        absorb(r2, T, "A"); absorb(r2, S, "B")

    return {"paired": _mean_se(paired), "unpaired": _mean_se(unpaired),
            "d_houju": _mean_se(d_houju), "d_agari": _mean_se(d_agari),
            "d_riichi": _mean_se(d_riichi),
            "A": acc["A"], "B": acc["B"], "pairs": pairs}


def report(res: dict):
    for name in ("A", "B"):
        s = res[name]
        k = max(s["kyoku"], 1); n = max(s["seats"], 1)
        print(f"[{name}] 平均順位 {s['rank_sum']/n:.3f}  "
              f"(1位{s['ranks'][0]} 2位{s['ranks'][1]} 3位{s['ranks'][2]} 4位{s['ranks'][3]})  "
              f"和了率 {s['agari']/k*100:.2f}%  放銃率 {s['houju']/k*100:.2f}%  "
              f"リーチ率 {s['riichi']/k*100:.2f}%  "
              f"平均和了 {s['agari_pts']/max(s['agari'],1):.0f}  "
              f"平均放銃 {s['houju_pts']/max(s['houju'],1):.0f}")

    print("=" * 72)
    pm, pse = res["paired"]
    um, use = res["unpaired"]
    print(f"【ペア方式】平均順位差 (A−B) = {pm:+.4f} ± {pse:.4f} (SE)"
          f"   95%CI [{pm-1.96*pse:+.4f}, {pm+1.96*pse:+.4f}]   n={res['pairs']}ペア")
    print(f"【従来方式】平均順位差 (A−B) = {um:+.4f} ± {use:.4f} (SE)"
          f"   95%CI [{um-1.96*use:+.4f}, {um+1.96*use:+.4f}]   n={res['pairs']*2}半荘")
    if pse > 0 and use > 0:
        print(f"   → SE の比（従来方式 / ペア方式）= {use/pse:.2f}"
              f"（1 より大きければペア方式のほうが精度が高い）")
    elif pse == 0:
        print("   → ペア差が全て 0。A と B が同一モデルなら正常（対応付けの動作確認）")

    for label, key in (("放銃率", "d_houju"), ("和了率", "d_agari"), ("リーチ率", "d_riichi")):
        m, se = res[key]
        mark = ""
        if se > 0:
            if m + 1.96 * se < 0:   mark = "  ← A が有意に低い"
            elif m - 1.96 * se > 0: mark = "  ← A が有意に高い"
        print(f"   {label}差 (A−B) = {m*100:+.3f} ± {se*100:.3f} pt/局{mark}")

    print("=" * 72)
    if pse > 0:
        if pm + 1.96 * pse < 0:   print("→ A が有意に強い")
        elif pm - 1.96 * pse > 0: print("→ B が有意に強い")
        else:                     print("→ 順位では有意差なし（局単位の指標も見てください）")


def _load(spec: str, device):
    if spec == "rule":
        return RULE_BASED
    return load_policy(Path(spec), device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", type=str, required=True, help=".pt のパス、または rule")
    ap.add_argument("--model-b", type=str, required=True, help=".pt のパス、または rule")
    ap.add_argument("--pairs", type=int, default=300, help="1ペア = 同じ山で2半荘")
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mA = _load(args.model_a, device)
    mB = _load(args.model_b, device)
    print(f"A: {args.model_a}\nB: {args.model_b}\n{args.pairs}ペア = {args.pairs*2}半荘")
    report(run_paired(mA, mB, device, args.pairs, args.seed, progress=True))


if __name__ == "__main__":
    main()
