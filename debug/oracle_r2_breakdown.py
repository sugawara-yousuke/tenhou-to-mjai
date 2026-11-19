"""
oracle_r2_breakdown.py  ―  オラクル Q の R² が小さい理由を切り分ける

  python3 debug/oracle_r2_breakdown.py --data data/2024.zip --qmodel train/oracle_q/oracle_q_best.pt --qmodel-noor train/oracle_q_noor/oracle_q_best.pt --offset 148000 --max-games 2000

（--offset 148000 --max-games 2000 は train_oracle_q.py の val と同じ範囲）

出すもの（局面の種類ごとに、オラクルあり／なし／オラクルあり＋入れ替え の R²）:
  ・巡目（0-5 / 6-11 / 12+）
  ・相手の誰かが聴牌しているか（aux_ron に当たり牌があるか）
  ・被リーチか
  ・全体

読み方:
  ・「入れ替え」は、オラクル入力をバッチ内の別の局面のものとシャッフルして予測したもの。
    オラクルありの R² が入れ替えでほとんど落ちなければ、モデルはオラクル入力を使っていない（学習の問題）。
  ・終盤・相手聴牌の局面で オラクルあり ≫ なし なら、全体の差が小さいのは序盤の運のせい（本質的な限界）。
    そこでも差が小さいなら、オラクル情報の入れ方か学習量に問題がある。
"""
from __future__ import annotations
import _pathsetup  # noqa: F401
import argparse
from collections import defaultdict
from pathlib import Path

import torch

from mahjong_model import samples_to_batch
from load_dataset import TenhouDatasetLoader
from aux_labels import attach_ron_labels, attach_return_labels
from oracle_q import load_oracle_q, oracle_feat


class R2:
    __slots__ = ("n", "sr", "sr2", "se")

    def __init__(self):
        self.n = 0; self.sr = 0.0; self.sr2 = 0.0; self.se = {}

    def add(self, r, preds: dict):
        self.n += 1; self.sr += r; self.sr2 += r * r
        for k, q in preds.items():
            self.se[k] = self.se.get(k, 0.0) + (q - r) ** 2

    def r2(self, k):
        if self.n < 50:
            return float("nan")
        var = self.sr2 / self.n - (self.sr / self.n) ** 2
        return 1 - (self.se[k] / self.n) / max(var, 1e-12)

    def var(self):
        return self.sr2 / self.n - (self.sr / self.n) ** 2 if self.n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--qmodel", type=Path, required=True, help="オラクルあり")
    ap.add_argument("--qmodel-noor", type=Path, required=True, help="オラクルなし（対照）")
    ap.add_argument("--offset", type=int, default=148000)
    ap.add_argument("--max-games", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    qo = load_oracle_q(args.qmodel, device)
    qn = load_oracle_q(args.qmodel_noor, device)
    loader = TenhouDatasetLoader(args.data)
    loader.tokenizer._emit_aux_fields = True
    G = defaultdict(R2)
    pending = []

    def flush():
        if not pending:
            return
        b = samples_to_batch(pending, device)
        orc = torch.tensor([oracle_feat(s) for s in pending], dtype=torch.float32, device=device)
        perm = orc[torch.randperm(orc.size(0), device=device)]
        lab = torch.tensor([s["action_label"] for s in pending], device=device).unsqueeze(1)
        args_ = (b["sparse"], b["numeric"], b["progression"], b["candidates"], b["cand_feats"])
        with torch.no_grad():
            q_o = qo(*args_, orc)[1].gather(1, lab).squeeze(1).cpu()
            q_p = qo(*args_, perm)[1].gather(1, lab).squeeze(1).cpu()
            q_n = qn(*args_, orc)[1].gather(1, lab).squeeze(1).cpu()
        for i, s in enumerate(pending):
            r = float(s["aux_ret"])
            preds = {"o": float(q_o[i]), "p": float(q_p[i]), "n": float(q_n[i])}
            turn = int(round(s["numeric"][15] * 30))
            tb = "巡目 0-5" if turn < 6 else ("巡目 6-11" if turn < 12 else "巡目 12+")
            ten = "相手聴牌あり" if any(any(row) for row in s["aux_ron"]) else "相手聴牌なし"
            ri = "被リーチ" if any(s["numeric"][6:9]) else "平場"
            for g in ("全体", tb, ten, ri, f"{tb}・{ten}"):
                G[g].add(r, preds)
        pending.clear()

    for gi, (_gid, events) in enumerate(loader.iter_games(max_games=args.max_games, offset=args.offset)):
        samples = list(loader.tokenizer.process_events(events))
        attach_ron_labels(samples)
        attach_return_labels(events, samples)
        for s in samples:
            if s.get("event_type") == "dahai" and s.get("aux_ron_valid") and s.get("aux_ret_valid"):
                pending.append(s)
                if len(pending) >= args.batch:
                    flush()
        if (gi + 1) % 500 == 0:
            print(f"  {gi+1:,} / {args.max_games:,} 半荘")
    flush()

    order = ["全体", "平場", "被リーチ", "相手聴牌なし", "相手聴牌あり", "巡目 0-5", "巡目 6-11", "巡目 12+"]
    order += [f"{t}・{a}" for t in ("巡目 0-5", "巡目 6-11", "巡目 12+") for a in ("相手聴牌なし", "相手聴牌あり")]
    W = 92
    print("=" * W)
    print(f"  {'':<22}{'打牌数':>10}{'Var(R)':>9}{'なし':>9}{'あり':>9}{'差':>9}{'あり・入替':>12}{'入替で失う':>11}")
    for g in order:
        x = G.get(g)
        if x is None or x.n == 0:
            continue
        ro, rn, rp = x.r2("o"), x.r2("n"), x.r2("p")
        print(f"  {g:<22}{x.n:>10,}{x.var():>9.4f}{rn:>9.4f}{ro:>9.4f}{ro-rn:>+9.4f}{rp:>12.4f}{ro-rp:>+11.4f}")
    print("=" * W)
    print("  入替で失う ≒ 0  → モデルはオラクル入力をほぼ使っていない（学習・入力の問題）")
    print("  終盤・相手聴牌ありで 差 が大きく、序盤で小さい → 全体の差が小さいのは序盤の運のせい")


if __name__ == "__main__":
    main()
