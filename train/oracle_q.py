"""
oracle_q.py  ―  オラクル Q 関数（全員の手牌を見たうえでの、打牌ごとの局収支の期待値）

考え方:
  牌譜には全員の手牌が入っている。そこで「相手3人の手牌と当たり牌が見えている状態で、
  この候補を切ったら、その局の順位期待値の変化（R_局 = aux_ret）がどうなるか」を候補ごとに
  予測する Q(s, a) を学習する。

  ・入力: 通常の BC と同じ入力 ＋ 相手3人ぶんのオラクルトークン（手牌の牌種34枚数 / 4 と
          当たり牌34の 0/1、計68次元）を encoder の memory の末尾に3トークン足す。
          ツモ山（まだ誰も見ていない牌）は入れない。山は運として残し、相手の手牌だけを開ける。
  ・出力: decoder の候補ごとの出力から Q(s, a)（64候補ぶん）。
  ・教師: 実際に選ばれた候補の位置にだけ R_局 を当てる（回帰）。他の候補の Q は、似た局面で
          別の打牌を選んだ数千万サンプルからの汎化で決まる。
          → 人間がほとんど選ばない候補の Q は信用できない。評価側では参照方策の確率で重み付けする。

  use_oracle=False にするとオラクルトークンを入れない（対照実験。同じ構造で R² の差を測る）。

使い道:
  1. 打牌の良し悪しの連続値化: A(s, a) = Q(s, a) − Σ_b π_ref(b|s) Q(s, b)
  2. プレイヤーごとに A を平均して実力を推定（順位より前後半の再現性が高いかを検証）
  3. 教師分布（方向1）
"""
from __future__ import annotations
import _pathsetup  # noqa: F401
import random

import torch
import torch.nn as nn

from mahjong_model import MahjongTransformer, PADDING_ACTION_ID, load_policy

ORACLE_DIM = 68          # 手牌34 + 当たり牌34（相手1人あたり）


# ---------------------------------------------------------------------------
# 特徴量
# ---------------------------------------------------------------------------
def oracle_feat(smp: dict) -> list:
    """相手 rel=1,2,3 ぶんの (3, 68)。ラベルが無いサンプルは全部 0（＋ oracle_valid=0）。"""
    from aux_labels import _to34_str
    out = [[0.0] * ORACLE_DIM for _ in range(3)]
    if not smp.get("aux_ron_valid") or "_all_hands" not in smp:
        return out
    player = smp.get("player", 0)
    ron = smp.get("aux_ron") or [[0] * 34 for _ in range(3)]
    for rel in range(1, 4):
        q = (player + rel) % 4
        row = out[rel - 1]
        for t in smp["_all_hands"][q]:
            k = _to34_str(t)
            if k >= 0:
                row[k] += 0.25
        for k in range(34):
            row[34 + k] = float(ron[rel - 1][k])
    return out


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------
class OracleQ(nn.Module):
    def __init__(self, base: MahjongTransformer, use_oracle: bool = True):
        super().__init__()
        self.base = base
        self.use_oracle = use_oracle
        d = base.encoder.d_model
        self.oracle_proj = nn.Sequential(nn.Linear(ORACLE_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.opp_emb = nn.Embedding(3, d)
        self.q_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    def forward(self, sparse, numeric, progression, candidates, cand_feats, oracle):
        memory, pad = self.base.encoder(sparse, numeric, progression)
        if self.use_oracle:
            tok = self.oracle_proj(oracle) + self.opp_emb.weight.unsqueeze(0)      # (B,3,d)
            memory = torch.cat([memory, tok], dim=1)
            pad = torch.cat([pad, torch.zeros(pad.size(0), 3, dtype=torch.bool, device=pad.device)], dim=1)
        logits, hidden = self.base.decoder(candidates, memory, pad, cand_feats, return_hidden=True)
        q = self.q_head(hidden).squeeze(-1)
        q = q.masked_fill(candidates == PADDING_ACTION_ID, 0.0)
        return logits, q


def build_oracle_q(base_path, device, use_oracle=True) -> OracleQ:
    base = load_policy(base_path, device, eval_mode=False)
    return OracleQ(base, use_oracle=use_oracle).to(device)


def save_oracle_q(path, model: OracleQ, base_path, extra: dict | None = None):
    torch.save({"model": model.state_dict(), "base_path": str(base_path),
                "use_oracle": model.use_oracle, **(extra or {})}, path)


def load_oracle_q(path, device, base_path=None) -> OracleQ:
    """base_path を省略するとチェックポイントに記録された BC のパスから構造を作る（重みは上書きされる）。"""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_oracle_q(base_path or ck["base_path"], device, use_oracle=ck.get("use_oracle", True))
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def oracle_batch(samples: list, device) -> torch.Tensor:
    return torch.tensor([s.get("oracle") or oracle_feat(s) for s in samples],
                        dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# データセット（打牌サンプルのみ。aux_ron / aux_ret とオラクル特徴を付ける）
# ---------------------------------------------------------------------------
try:
    from load_dataset import TenhouAuxDataset, _to_item
    from torch.nn.utils.rnn import pad_sequence  # noqa: F401
    from torch.utils.data import get_worker_info

    class OracleQDataset(TenhouAuxDataset):
        WITH_AUX = True

        def __init__(self, path, max_games=None, offset=0, shuffle_buffer=0, seed=0, use_cache=False):
            super().__init__(path, max_games=max_games, offset=offset, event_filter="dahai",
                             shuffle_buffer=shuffle_buffer, seed=seed, use_cache=use_cache)

        @staticmethod
        def _item(s):
            it = _to_item(s, True)
            it["oracle"] = torch.tensor(oracle_feat(s), dtype=torch.float32)
            return it

        def __iter__(self):
            wi = get_worker_info()
            shard = (wi.id, wi.num_workers) if wi is not None else (0, 1)
            self._epoch += 1
            rng = random.Random(self.seed * 1000003 + self._epoch * 101 + shard[0])
            src = (s for s in self._source(shard) if s.get("aux_ron_valid") and s.get("aux_ret_valid"))
            if self.shuffle_buffer <= 1:
                for s in src:
                    yield self._item(s)
                return
            buf = []
            for s in src:
                if len(buf) < self.shuffle_buffer:
                    buf.append(s); continue
                j = rng.randrange(len(buf))
                buf[j], s = s, buf[j]
                yield self._item(s)
            rng.shuffle(buf)
            for s in buf:
                yield self._item(s)

        @staticmethod
        def collate_fn(batch):
            out = TenhouAuxDataset.collate_fn(batch)
            out["oracle"] = torch.stack([b["oracle"] for b in batch])
            return out
except ImportError:
    pass
