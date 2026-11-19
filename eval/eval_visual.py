"""
eval_visual.py ― 学習済みモデルの予測を牌姿つきで表示する（牌譜データ上の BC 精度の目視確認）。

  python3 eval_visual.py --data 2024.zip --model checkpoints1/model_bc_best.pt
  python3 eval_visual.py --data 2024.zip --model ... --riichi-enemy          # 被リーチ局面だけ
  python3 eval_visual.py --data 2024.zip --model ... --event-type meld_decision
  python3 eval_visual.py --data 2024.zip --model ... --riichi-turn           # リーチ/ダマを選べた手番だけ

[今回の改修] 新しい sparse レイアウトに対応。モデル構成は checkpoint から自動復元。
             候補ごとの危険度(cand_feats)も併記するので「危険牌だと分かった上で押したか」が見える。
"""
import _pathsetup  # noqa: F401
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from mahjong_model import load_policy, samples_to_batch
from mahjong_tokenizer import (
    TILE_ID_TO_STR, SPARSE_TEHAI_OFFSET, SPARSE_DORA_OFFSET, SPARSE_SUTEHAI_SELF,
    SPARSE_BAKAZE_OFFSET, SPARSE_KYOKU_OFFSET, SPARSE_HONBA_OFFSET, SPARSE_JIKAZE_OFFSET,
    ACT_DAHAI_BASE, ACT_RIICHI_BASE, ACT_PON_BASE, ACT_CHI_BASE, ACT_DAIMINKAN_BASE,
    ACT_ANKAN_BASE, ACT_KAKAN_BASE, ACT_TSUMO_BASE, ACT_PASS_BASE, PADDING_ACTION_ID, decode_act_chi,
)
from load_dataset import TenhouDatasetLoader

_Z = {"1z": "東", "2z": "南", "3z": "西", "4z": "北", "5z": "白", "6z": "発", "7z": "中",
      "0m": "赤5m", "0p": "赤5p", "0s": "赤5s"}
disp = lambda tid: _Z.get(TILE_ID_TO_STR[tid], TILE_ID_TO_STR[tid]) if 0 <= tid < 37 else f"?{tid}"


def action_str(tok: int) -> str:
    if tok == PADDING_ACTION_ID: return "PAD"
    if tok == ACT_PASS_BASE: return "パス"
    if tok < ACT_RIICHI_BASE: return f"打{disp(tok - ACT_DAHAI_BASE)}"
    if tok < ACT_PON_BASE: return f"リーチ打{disp(tok - ACT_RIICHI_BASE)}"
    if tok < ACT_CHI_BASE: return f"ポン{disp(tok - ACT_PON_BASE)}"
    if tok < ACT_DAIMINKAN_BASE:
        s, start, red = decode_act_chi(tok)
        return f"チー{start}{start+1}{start+2}{'mps'[s]}{'(赤)' if red else ''}"
    if tok < ACT_ANKAN_BASE: return f"明槓{disp(tok - ACT_DAIMINKAN_BASE)}"
    if tok < ACT_KAKAN_BASE: return f"暗槓{disp(tok - ACT_ANKAN_BASE)}"
    if tok < ACT_TSUMO_BASE: return f"加槓{disp(tok - ACT_KAKAN_BASE)}"
    return f"?{tok}"


def decode_sparse(tokens):
    hand, dora, info = [], [], {}
    for t in tokens:
        if SPARSE_TEHAI_OFFSET <= t < SPARSE_DORA_OFFSET: hand.append(t - SPARSE_TEHAI_OFFSET)
        elif SPARSE_DORA_OFFSET <= t < SPARSE_SUTEHAI_SELF: dora.append(t - SPARSE_DORA_OFFSET)
        elif SPARSE_BAKAZE_OFFSET <= t < SPARSE_KYOKU_OFFSET: info["場"] = "東南"[t - SPARSE_BAKAZE_OFFSET]
        elif SPARSE_KYOKU_OFFSET <= t < SPARSE_HONBA_OFFSET: info["局"] = t - SPARSE_KYOKU_OFFSET + 1
        elif SPARSE_HONBA_OFFSET <= t < SPARSE_JIKAZE_OFFSET: info["本場"] = t - SPARSE_HONBA_OFFSET
        elif t >= SPARSE_JIKAZE_OFFSET: info["自風"] = "東南西北"[t - SPARSE_JIKAZE_OFFSET]
    key = lambda tid: ("mpsz".index(TILE_ID_TO_STR[tid][-1]), int(TILE_ID_TO_STR[tid][0]) or 5)
    return sorted(hand, key=key), dora, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--max-games", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--event-type", default="dahai", choices=["dahai", "meld_decision", "kan_decision", "all"])
    ap.add_argument("--min-dahai", type=int, default=0)
    ap.add_argument("--max-dahai", type=int, default=9999)
    ap.add_argument("--riichi-enemy", action="store_true", help="他家リーチ中の局面のみ")
    ap.add_argument("--riichi-turn", action="store_true", help="リーチ可能だった手番のみ")
    ap.add_argument("--only-miss", action="store_true", help="予測が外れた局面のみ表示")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_policy(args.model, device)

    shown = total = correct = 0
    for s in TenhouDatasetLoader(args.data).iter_samples(max_games=args.max_games, offset=args.offset):
        if shown >= args.n_samples: break
        if args.event_type != "all" and s["event_type"] != args.event_type: continue
        dc = s.get("dahai_count", -1)
        if dc >= 0 and not (args.min_dahai <= dc <= args.max_dahai): continue
        if args.riichi_enemy and not any(s["numeric"][6:9]): continue
        if args.riichi_turn and "riichi_label" not in s: continue
        b = samples_to_batch([s], device)
        with torch.no_grad():
            logits = model(b["sparse"], b["numeric"], b["progression"], b["candidates"], b["cand_feats"])[0]
        probs = F.softmax(logits, -1).cpu().tolist()
        poss, label = s["possible_actions"], s["action_label"]
        order = sorted((i for i, t in enumerate(poss) if t != PADDING_ACTION_ID), key=lambda i: -probs[i])
        total += 1; hit = order[0] == label; correct += int(hit)
        if args.only_miss and hit: continue
        hand, dora, info = decode_sparse(s["sparse"])
        n = s["numeric"]
        print("─" * 60)
        print(f"#{shown+1} {info.get('場','?')}{info.get('局','?')}局{info.get('本場',0)}本場 {info.get('自風','?')}家"
              f"  順位{int(round(n[19]*3))+1}  点数(自,下,対,上)={[int(x*25000) for x in n[:4]]}"
              f"  他家リーチ={[int(x) for x in n[6:9]]}  残り山{int(round(n[9]*70))}")
        print(f"  ドラ表示: {' '.join(disp(d) for d in dora)}   手牌: {' '.join(disp(t) for t in hand)}")
        for rank, i in enumerate(order[:5]):
            f = s["cand_feats"][i]
            dz = f"  危険度(下/対/上)={f[0]:.2f}/{f[1]:.2f}/{f[2]:.2f}" if any(f[:3]) else ""
            print(f"   {rank+1}. {action_str(poss[i]):<12} {probs[i]:6.1%}{dz}{'  ← 実際' if i == label else ''}")
        if label not in order[:5]:
            print(f"   実際: {action_str(poss[label])} ({probs[label]:.1%})")
        shown += 1
    print("═" * 60)
    print(f"対象 {total} 局面  Top-1 = {correct/max(total,1):.1%}")


if __name__ == "__main__":
    main()
