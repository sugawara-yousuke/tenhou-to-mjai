"""
aggregate_pushfold.py  ―  被リーチ押し引きの定量評価

  python3 aggregate_pushfold.py --model checkpoints1/model_bc_best.pt --games 200
  python3 aggregate_pushfold.py --model checkpoints1/grpo/model_grpo_final.pt --games 200

被リーチ（他家リーチ・自分は非リーチ）の打牌で、
  - 手牌に現物（リーチ者全員に通る牌）があったか
  - 実際に切ったのは現物か、非現物か（向聴別）
  - 切った牌の危険度（danger_features の対リーチ者最大値）の平均
を集計する。

[今回の改修]
  - 現物判定を tile_id の整数で行う。旧版は {v:k for k,v in TILE_STR_TO_ID.items()} で逆引き
    しており、別名が後勝ちして 1z→'E'、0m→'5mr' になっていた。その結果、字牌と赤5の現物が
    現物として数えられず、「現物の字牌を切ったのに危険牌押し扱い」になっていた。
  - 現物は state.safe_tiles（リーチ者の河 ＋ リーチ後に通った牌）。モデルの特徴量と同じ定義。
  - テンパイ時の押しは分けて表示（テンパイからの押しは悪手とは限らない）。
  - 対戦相手モデルを指定可能（--opponent）。既定は同じモデル同士。
  - genbutsu 版スクリプトは不要になった（tokenizer が1本化されたため）。
"""
from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
import argparse
import random
from pathlib import Path

import torch

from mahjong_model import load_policy
from mahjong_tokenizer import MahjongTokenizer, TILE_STR_TO_ID, _tid_norm
from danger_features import build_danger_context
from self_play import MahjongSimulator


def run_pushfold(model, device, games: int, seed: int = 12345, opponent=None, progress=False) -> dict:
    sim = MahjongSimulator(model, device, watch=False)
    if opponent is not None:
        sim.player_models = {1: opponent, 2: opponent, 3: opponent}
    tok = MahjongTokenizer()
    S = {"under_riichi": 0, "had_genbutsu": 0, "chose_safe": 0, "pushed": 0,
         "push_by_shanten": {}, "danger_sum": 0.0, "kyoku": 0, "houju": 0}
    target_seats = {0} if opponent is not None else {0, 1, 2, 3}

    def hook(state, p, riichi_flags, pai):
        if p not in target_seats:
            return
        rp = [q for q in range(4) if riichi_flags[q] and q != p]
        if not rp or riichi_flags[p]:
            return
        S["under_riichi"] += 1
        ctx = build_danger_context(state, p)
        k = _tid_norm(TILE_STR_TO_ID[pai])
        S["danger_sum"] += max(ctx["matrix"][(q - p) % 4 - 1][k] for q in rp)
        common = set.intersection(*[set(state.safe_tiles[q]) for q in rp])
        hand = {_tid_norm(TILE_STR_TO_ID[t]) for t in state.tehais[p]}
        if not (hand & common):
            return
        S["had_genbutsu"] += 1
        if k in common:
            S["chose_safe"] += 1
        else:
            S["pushed"] += 1
            rest = list(state.tehais[p]); rest.remove(pai)
            sh = tok._shanten_of(rest, state.meld_counts[p])
            S["push_by_shanten"][sh] = S["push_by_shanten"].get(sh, 0) + 1

    sim._eval_hook = hook
    random.seed(seed); torch.manual_seed(seed)
    it = range(games)
    if progress:
        try:
            from tqdm import tqdm; it = tqdm(it, desc="eval games")
        except ImportError:
            pass
    for _ in it:
        for e in sim.simulate_game():
            if e["type"] == "start_kyoku":
                S["kyoku"] += len(target_seats)
            elif e["type"] == "hora" and e.get("target") in target_seats:
                S["houju"] += 1
    return S


def report(S: dict):
    n, hg = S["under_riichi"], S["had_genbutsu"]
    print("=" * 60)
    print(f"被リーチ打牌局面: {n}   放銃率: {S['houju']/max(S['kyoku'],1)*100:.2f}% (/局)")
    if n == 0:
        print("局面が集まりませんでした。--games を増やしてください。"); return
    print(f"  切った牌の平均危険度: {S['danger_sum']/n:.3f}")
    print(f"  現物が手牌にあった: {hg} ({hg/n*100:.1f}%)")
    if hg:
        tenpai_push = S["push_by_shanten"].get(0, 0)
        print(f"    現物を切った       : {S['chose_safe']} ({S['chose_safe']/hg*100:.1f}%)")
        print(f"    非現物を切った     : {S['pushed']} ({S['pushed']/hg*100:.1f}%)")
        print(f"      うちテンパイ維持  : {tenpai_push}")
        print(f"      ★ノーテンからの押し: {S['pushed']-tenpai_push} "
              f"({(S['pushed']-tenpai_push)/hg*100:.1f}%)")
        for sh in sorted(S["push_by_shanten"]):
            print(f"        打牌後 {'テンパイ' if sh == 0 else f'{sh}向聴'}: {S['push_by_shanten'][sh]}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--opponent", type=Path, default=None, help="他3席のモデル（省略時は同じモデル）")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy(args.model, device)
    opp = load_policy(args.opponent, device) if args.opponent else None
    print(f"model: {args.model}" + (f"  vs {args.opponent}" if opp else ""))
    report(run_pushfold(model, device, args.games, args.seed, opp, progress=True))


if __name__ == "__main__":
    main()
