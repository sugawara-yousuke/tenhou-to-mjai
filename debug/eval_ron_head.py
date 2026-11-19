"""
eval_ron_head.py  ―  当たり牌ヘッドの精度を、手計算の危険度と同じ土俵で比べる

  python3 debug/eval_ron_head.py --data ../data/2024.zip --model train/checkpoints1/model_bc_best.pt \
      --max-games 200 --offset 50000

何を測るか:
  各打牌サンプルの「打牌候補にある牌 × 相手3人」を1件として、
    正解      : その牌をいま切ったら、その相手にロンされるか（aux_ron）
    モデル    : ron ヘッドの確率
    手計算    : cand_feats[i][rel] の危険度（danger_features のヒューリスティック）
  の AUC を比べる。AUC はランダムなら 0.5、完全なら 1.0。

  モデルが手計算を上回れば、danger_features.py の値を学習された確率に置き換える
  意味がある（置き換えの可否はこの数字で判断する）。
"""
from __future__ import annotations
import _pathsetup  # noqa: F401
import argparse
from pathlib import Path

import torch

from mahjong_model import load_policy, samples_to_batch
from mahjong_tokenizer import (ACT_DAHAI_BASE, ACT_RIICHI_BASE, PADDING_ACTION_ID, _tid_norm)
from load_dataset import TenhouDatasetLoader
from aux_labels import attach_ron_labels


def _auc(pairs: list) -> float:
    """pairs: [(score, label0/1), ...] の AUC（順位和で計算）。"""
    if not pairs:
        return float("nan")
    pos = sum(1 for _s, y in pairs if y)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rsum = sum(ranks[i] for i, (_s, y) in enumerate(pairs) if y)
    return (rsum - pos * (pos + 1) / 2.0) / (pos * neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--max-games", type=int, default=200)
    ap.add_argument("--offset", type=int, default=50000, help="学習に使っていない範囲を指定する")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy(args.model, device); model.eval()

    loader = TenhouDatasetLoader(args.data)
    loader.tokenizer._emit_aux_fields = True

    m_pairs, h_pairs, n_pos, n_tot, n_smp = [], [], 0, 0, 0
    for _gid, events in loader.iter_games(max_games=args.max_games, offset=args.offset):
        samples = list(loader.tokenizer.process_events(events))
        attach_ron_labels(samples)
        batch_samples = [s for s in samples if s.get("event_type") == "dahai" and s.get("aux_ron_valid")]
        for i in range(0, len(batch_samples), 64):
            chunk = batch_samples[i:i + 64]
            b = samples_to_batch(chunk, device)
            with torch.no_grad():
                _logits, aux = model(b["sparse"], b["numeric"], b["progression"],
                                     b["candidates"], b["cand_feats"], return_aux=True)
            ron = torch.sigmoid(aux["ron"]).cpu()          # (B,3,34)
            for bi, s in enumerate(chunk):
                n_smp += 1
                for ci, tok in enumerate(s["possible_actions"]):
                    if tok == PADDING_ACTION_ID or tok >= ACT_RIICHI_BASE:
                        continue
                    k = _tid_norm(tok - ACT_DAHAI_BASE)
                    if not (0 <= k < 34):
                        continue
                    for rel in range(3):
                        y = s["aux_ron"][rel][k]
                        m_pairs.append((float(ron[bi, rel, k]), y))
                        h_pairs.append((float(s["cand_feats"][ci][rel]), y))
                        n_pos += y; n_tot += 1

    print("=" * 60)
    print(f"サンプル {n_smp}  判定 {n_tot} 件  うち正例(ロンされる) {n_pos} ({n_pos/max(n_tot,1)*100:.2f}%)")
    print(f"  AUC  ron ヘッド       : {_auc(m_pairs):.4f}")
    print(f"  AUC  手計算の危険度   : {_auc(h_pairs):.4f}   ← danger_features.py")
    print("  （0.5 がランダム。ヘッドが上回れば cand_feats の置き換えに進める）")
    print("=" * 60)


if __name__ == "__main__":
    main()
