"""
mahjong_model.py
================
  SharedMahjongEncoder  - sparse / numeric / progression → memory
  MahjongDecoder        - DETR ライク Decoder（候補スコアリング）
  MahjongTransformer    - 統合モデル（打牌・リーチ・鳴き・槓を候補スコアリングで一括）
  MeldDecisionModel / RiichiDecisionModel - 旧・分離モデル（互換のため残置。encoder のみ使用）

[今回の改修]
  - 語彙サイズ・numeric 次元・sparse のオフセット境界を mahjong_tokenizer から import する。
    旧版はモデル側に同じ数字を手書きしており、tokenizer と食い違っても黙って動いていた
    （danger_features の import に失敗すると NUMERIC_DIM が静かに 19 に落ちる等）。
  - Decoder のクエリに「候補ごとの特徴 cand_feats」を加算。
      query_i = action_emb(候補_i) + cand_proj(その候補牌の対3家危険度・現物・ドラ・見え枚数)
    「どの牌が危険か」を候補単位で直接与えるので、押し引きを河トークンの照合だけに頼らない。
    cand_feats は必須引数（渡し忘れ＝学習時と推論時の食い違い、を黙って通さないため）。
  - 補助タスクの放銃ヘッドを「局面プール」から「候補ごと（decoder 出力）」に変更。
    放銃するかは切る牌で決まるので、実際に切った候補の位置で教師を当てる。
  - MahjongPipeline は削除（統合モデル化で未使用。encoder 共有時に重みを上書きするバグもあった）。
  - samples_to_batch(): サンプル dict → テンソルの変換を1か所に集約（self_play / rollout /
    grpo / server / eval が全部これを使う）。
"""

from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import mahjong_tokenizer as _T

SPARSE_VOCAB_SIZE     = _T.SPARSE_VOCAB_SIZE
PROG_VOCAB_SIZE       = _T.PROG_VOCAB_SIZE
NUM_ACTION_TYPES      = _T.NUM_ACTION_TYPES + 1     # +1 = PAD 用スロット
PADDING_ACTION_ID     = _T.PADDING_ACTION_ID
MAX_ACTION_CANDIDATES = _T.MAX_ACTION_CANDIDATES
NUMERIC_DIM           = _T.NUMERIC_DIM
CAND_FEAT_DIM         = _T.CAND_FEAT_DIM
NUM_MELD_LABELS       = 4
NUM_TILE_TYPES        = 37

_HAND_LO, _HAND_HI = _T.SPARSE_HAND_RANGE
_KAWA_LO, _KAWA_HI = _T.SPARSE_KAWA_RANGE
_MELD_LO, _MELD_HI = _T.SPARSE_MELD_RANGE


def samples_to_batch(samples: list[dict], device) -> dict:
    """tokenizer.make_sample() の dict 群 → モデル入力テンソル（0 パディング）。"""
    from torch.nn.utils.rnn import pad_sequence
    sp = pad_sequence([torch.tensor(s["sparse"], dtype=torch.long) for s in samples], batch_first=True)
    pr = pad_sequence([torch.tensor(s["progression"] or [0], dtype=torch.long) for s in samples],
                      batch_first=True)
    return {
        "sparse":      sp.to(device),
        "numeric":     torch.tensor([s["numeric"] for s in samples], dtype=torch.float32, device=device),
        "progression": pr.to(device),
        "candidates":  torch.tensor([s["possible_actions"] for s in samples], dtype=torch.long, device=device),
        "cand_feats":  torch.tensor([s["cand_feats"] for s in samples], dtype=torch.float32, device=device),
    }


def disable_dropout(model: nn.Module) -> nn.Module:
    """
    RL 用: dropout を恒久的に 0 にする。train()/eval() のどちらで呼ばれても出力が同じになり、
    「ロールアウト時と更新時で確率が違う」「greedy 再生が再現しない」を原理的に防ぐ。
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        if isinstance(m, nn.MultiheadAttention):
            m.dropout = 0.0
    return model


def load_policy(path, device="cpu", eval_mode: bool = True) -> "MahjongTransformer":
    """
    チェックポイントから MahjongTransformer を復元する。
    層数・次元は state_dict の形から推定するので --d-model 等の指定ミスが起きない。
    critic_head.* など余分なキーは無視。入力表現が旧版のチェックポイントは明示的に弾く。
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {k: v for k, v in sd.items() if not k.startswith("critic_head")}
    emb = sd["encoder.sparse_emb.weight"]
    num_w = sd["encoder.numeric_proj.0.weight"]
    if (emb.shape[0] != SPARSE_VOCAB_SIZE or num_w.shape[1] != NUMERIC_DIM
            or "decoder.cand_proj.0.weight" not in sd):
        raise RuntimeError(
            f"{path}: 旧入力表現のチェックポイントです "
            f"(sparse_vocab={emb.shape[0]}≠{SPARSE_VOCAB_SIZE} / numeric={num_w.shape[1]}≠{NUMERIC_DIM} "
            f"/ cand_proj 無し)。tokenizer を改修したので BC から学び直してください。")
    d_model = emb.shape[1]
    n_enc = 1 + max(int(k.split(".")[3]) for k in sd if k.startswith("encoder.encoder.layers."))
    n_dec = 1 + max(int(k.split(".")[3]) for k in sd if k.startswith("decoder.decoder.layers."))
    d_ff = sd["encoder.encoder.layers.0.linear1.weight"].shape[0]
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    nhead = int(cfg.get("nhead", max(1, d_model // 32)))
    model = MahjongTransformer(
        d_model=d_model, nhead=nhead, enc_layers=n_enc, dec_layers=n_dec, dim_feedforward=d_ff,
        num_action_types=sd["decoder.action_emb.weight"].shape[0],
        use_cross_attn=any(k.startswith("encoder.cross_attn") for k in sd))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # 学習時だけ使うヘッドは、推論用チェックポイントに無くても／余分にあってもよい。
    # value_head は AWR 以降に追加したので、それ以前のチェックポイントには入っていない。
    _TRAIN_ONLY = ("aux_", "value_head")
    bad = ([k for k in missing if not k.startswith(_TRAIN_ONLY)]
           + [k for k in unexpected if not k.startswith(_TRAIN_ONLY)])
    if bad:
        raise RuntimeError(f"{path}: state_dict が合いません: {bad[:6]}")
    model.to(device)
    if eval_mode:
        model.eval()
    return model


# ---------------------------------------------------------------------------
# 1. Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 2. SharedMahjongEncoder（旧 MahjongEncoder を独立クラスに）
# ---------------------------------------------------------------------------
class SharedMahjongEncoder(nn.Module):
    """
    sparse / numeric / progression → memory (B, seq_len, d_model)

    全サブモデル（Dahai / MeldDecision / RiichiDecision）で共有できる。
    プレトレイン済み MahjongTransformer の encoder 重みをそのまま移植可能。
    """
    def __init__(
        self,
        d_model:         int   = 384,
        nhead:           int   = 12,
        num_layers:      int   = 6,
        dim_feedforward: int   = 1536,
        dropout:         float = 0.1,
        use_cross_attn:  bool  = True,   # [ADD 提案2] 手牌×場 のクロスアテンション
    ):
        super().__init__()
        self.d_model = d_model
        self.use_cross_attn = use_cross_attn

        self.sparse_emb   = nn.Embedding(SPARSE_VOCAB_SIZE, d_model, padding_idx=0)
        self.prog_emb     = nn.Embedding(PROG_VOCAB_SIZE,   d_model, padding_idx=0)
        self.numeric_proj = nn.Sequential(
            nn.Linear(NUMERIC_DIM, d_model),
            nn.LayerNorm(d_model),
        )
        self.seg_emb = nn.Embedding(3, d_model)  # 0=sparse 1=prog 2=numeric
        self.pos_enc = PositionalEncoding(d_model, max_len=1024, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ---- [ADD 提案2] クロスアテンション層 ----
        # self-attention 後の memory に対し、手牌トークンを query、
        # 場トークン（他家の河・副露・リーチ情報）を key/value として
        # 1層のクロスアテンションを掛け、手牌トークン表現を「場に対する危険度」で
        # 変調する。残差接続で self-only の表現も保持するので、
        # use_cross_attn=False の旧チェックポイントからのファインチューンも安全。
        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=nhead,
                dropout=dropout, batch_first=True,
            )
            self.cross_norm_q  = nn.LayerNorm(d_model)   # query(手牌)側 pre-norm
            self.cross_norm_kv = nn.LayerNorm(d_model)   # key/value(場)側 pre-norm
            self.cross_ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
            )
            self.cross_ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        sparse:      Tensor,   # (B, s_len)  Long
        numeric:     Tensor,   # (B, NUMERIC_DIM) Float
        progression: Tensor,   # (B, p_len)  Long
    ) -> tuple[Tensor, Tensor]:
        B = sparse.size(0)
        sparse      = sparse.clamp(0, SPARSE_VOCAB_SIZE - 1)
        progression = progression.clamp(0, PROG_VOCAB_SIZE - 1)

        s = self.sparse_emb(sparse)
        p = self.prog_emb(progression)
        n = self.numeric_proj(numeric).unsqueeze(1)

        seg_s = self.seg_emb(torch.zeros(B, s.size(1), dtype=torch.long, device=sparse.device))
        seg_p = self.seg_emb(torch.ones( B, p.size(1), dtype=torch.long, device=sparse.device))
        seg_n = self.seg_emb(torch.full((B, 1), 2,     dtype=torch.long, device=sparse.device))
        s = s + seg_s
        p = p + seg_p
        n = n + seg_n

        x = torch.cat([s, p, n], dim=1)
        x = self.pos_enc(x)

        s_pad = (sparse == 0)
        p_pad = (progression == 0)
        n_pad = torch.zeros(B, 1, dtype=torch.bool, device=sparse.device)
        pad_mask = torch.cat([s_pad, p_pad, n_pad], dim=1)

        memory = self.encoder(x, src_key_padding_mask=pad_mask)

        # ---- [ADD 提案2] 手牌 × 場 クロスアテンション ----
        # sparse トークンのオフセット境界で「自分の手牌」と「場（他家の河・副露）」を
        # 識別する。手牌トークン位置の memory を query、場トークン位置を key/value とし、
        # 「この手牌牌は“あの河/副露”に対して危険か」という関連を構造的に注入する。
        if self.use_cross_attn:
            s_len = s.size(1)
            # sparse 内の各トークンが手牌か場か（tokenizer のオフセットに一致）
            #   手牌: SPARSE_TEHAI_OFFSET(0) .. +37   → id in [1, 37)  (0=pad)
            #   場  : 捨て牌 74.. と 副露 3034.. の範囲
            is_hand  = (sparse >= _HAND_LO) & (sparse < _HAND_HI)          # (B, s_len)
            is_board = ((sparse >= _KAWA_LO) & (sparse < _KAWA_HI)) | \
                       ((sparse >= _MELD_LO) & (sparse < _MELD_HI))        # (B, s_len)

            mem_s = memory[:, :s_len, :]                    # sparse 部分の memory
            q_in  = self.cross_norm_q(mem_s)
            kv_in = self.cross_norm_kv(mem_s)

            # key padding: 場トークン以外（手牌・pad）は attend 対象外
            kv_pad = ~is_board                              # True=無視
            # 場トークンが1つも無い行は全maskになり NaN を生むので、その行は
            # ダミーで自己参照可能にしておき、後段で cross 出力をゼロ寄与にする。
            no_board = kv_pad.all(dim=1)                    # (B,)
            if no_board.any():
                kv_pad = kv_pad.clone()
                kv_pad[no_board, 0] = False                 # 便宜的に先頭を開ける

            attn_out, _ = self.cross_attn(
                q_in, kv_in, kv_in,
                key_padding_mask=kv_pad, need_weights=False,
            )
            # 手牌トークン位置だけ更新（場・その他は self-only 表現を維持）
            hand_gate = is_hand.unsqueeze(-1).float()       # (B, s_len, 1)
            if no_board.any():
                hand_gate = hand_gate * (~no_board).view(B, 1, 1).float()

            mem_s = mem_s + hand_gate * attn_out            # 残差
            mem_s = mem_s + hand_gate * self.cross_ffn(self.cross_ffn_norm(mem_s))

            memory = torch.cat([mem_s, memory[:, s_len:, :]], dim=1)

        return memory, pad_mask


# 後方互換エイリアス
MahjongEncoder = SharedMahjongEncoder


# ---------------------------------------------------------------------------
# 3. MahjongDecoder（打牌/鳴き後打牌 共用）
# ---------------------------------------------------------------------------
class MahjongDecoder(nn.Module):
    def __init__(
        self,
        d_model:          int   = 256,
        nhead:            int   = 8,
        num_layers:       int   = 2,
        dim_feedforward:  int   = 1024,
        dropout:          float = 0.1,
        num_action_types: int   = NUM_ACTION_TYPES,
    ):
        super().__init__()
        self.d_model = d_model
        self._num_action_types = num_action_types

        self.action_emb = nn.Embedding(num_action_types, d_model)
        self.cand_proj  = nn.Sequential(
            nn.Linear(CAND_FEAT_DIM, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        candidates:  Tensor,   # (B, 64)
        memory:      Tensor,   # (B, src_len, d)
        memory_mask: Tensor,   # (B, src_len) Bool
        cand_feats:  Tensor,   # (B, 64, CAND_FEAT_DIM)
        return_hidden: bool = False,
    ):
        pad_mask           = (candidates == PADDING_ACTION_ID)
        candidates_clamped = candidates.clamp(0, self._num_action_types - 1)
        query  = self.action_emb(candidates_clamped) + self.cand_proj(cand_feats)
        # PAD 候補同士が self-attention で混ざらないよう tgt 側もマスクする
        out    = self.decoder(tgt=query, memory=memory,
                              tgt_key_padding_mask=pad_mask,
                              memory_key_padding_mask=memory_mask)
        logits = self.score_head(out).squeeze(-1)
        logits = logits.masked_fill(pad_mask, float('-inf'))
        if return_hidden:
            return logits, out
        return logits


# ---------------------------------------------------------------------------
# 4. MahjongTransformer（打牌モデル・既存互換）
# ---------------------------------------------------------------------------
class MahjongTransformer(nn.Module):
    """
    打牌選択モデル。SharedMahjongEncoder を外から渡せる。
    encoder=None の場合は内部で生成（既存コードとの後方互換）。
    """
    def __init__(
        self,
        d_model:          int   = 384,
        nhead:            int   = 12,
        enc_layers:       int   = 6,
        dec_layers:       int   = 2,
        dim_feedforward:  int   = 1536,
        dropout:          float = 0.1,
        num_action_types: int | None = None,
        encoder: SharedMahjongEncoder | None = None,  # [ADD] 共有 Encoder
        use_cross_attn:   bool  = True,
    ):
        super().__init__()
        _num_actions = num_action_types if num_action_types is not None else NUM_ACTION_TYPES

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = SharedMahjongEncoder(
                d_model=d_model, nhead=nhead, num_layers=enc_layers,
                dim_feedforward=dim_feedforward, dropout=dropout,
                use_cross_attn=use_cross_attn,
            )

        self.decoder = MahjongDecoder(
            d_model=d_model, nhead=nhead, num_layers=dec_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            num_action_types=_num_actions,
        )

        # ---- [ADD 提案3] 補助ヘッド（マルチタスク学習） ----
        # encoder の pooled 表現から、押し引きに直結する量を陽に予測させる。
        #   houjuu_head : 候補ごと。その牌を切ったら放銃になるか（decoder 出力から）
        #   tenpai_head : 他家3人が最終的にテンパイか（3値マルチラベル）
        #   ukeire_head : 自分の受け入れ枚数（回帰）
        # これらは推論では使わず、学習時の補助損失としてのみ用いる。
        # 内部表現に「相手の待ち／自分の危険度」を持たせるのが狙い。
        self.aux_houjuu_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.aux_tenpai_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        self.aux_ukeire_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        #   ron_head : 「いまこの牌を切ったら相手 rel=1,2,3 にロンされるか」を
        #              牌種34すべてについて予測する (3 x 34)。教師は牌譜から正確に作れる
        #              （aux_labels.attach_ron_labels）。
        #   ※ 補助損失としてのみ使う。return_aux=True のときだけ計算するので推論は重くならない。
        #     出力を cand_feats に連結する配線は試したが、精度ゲインが無く推論が4.2倍遅く
        #     なったので採用していない（2026-10 の実測）。
        self.aux_ron_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 3 * 34),
        )
        #   value_head : この局面から見た「その局の順位期待値の変化」V(s) を予測する。
        #                AWR（Advantage Weighted Regression）で
        #                  A = R_局 − V(s),  w = exp(A/β)
        #                として方策損失を重み付けするのに使う。R を手書きの価値関数では
        #                なく学習した V で引くことで、A の平均がゼロになり、関数の誤差を
        #                増幅しない。return_aux=True のときだけ計算するので推論は重くならない。
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    @staticmethod
    def _masked_mean_pool(memory: Tensor, pad_mask: Tensor) -> Tensor:
        valid = (~pad_mask).unsqueeze(-1).float()
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (memory * valid).sum(dim=1) / denom

    def forward(
        self,
        sparse:      Tensor,
        numeric:     Tensor,
        progression: Tensor,
        candidates:  Tensor,
        cand_feats:  Tensor,
        return_aux:  bool = False,
    ):
        memory, pad_mask = self.encoder(sparse, numeric, progression)
        if not return_aux:
            return self.decoder(candidates, memory, pad_mask, cand_feats)
        logits, hidden = self.decoder(candidates, memory, pad_mask, cand_feats, return_hidden=True)
        pooled = self._masked_mean_pool(memory, pad_mask)
        aux = {
            "houjuu": self.aux_houjuu_head(hidden).squeeze(-1),   # (B, 64) 候補ごと
            "tenpai": self.aux_tenpai_head(pooled),                # (B,3)
            "ukeire": self.aux_ukeire_head(pooled).squeeze(-1),   # (B,)
            "ron":    self.aux_ron_head(pooled).view(-1, 3, 34),   # (B,3,34) ロジット
            "value":  self.value_head(pooled).squeeze(-1),         # (B,) V(s)
        }
        return logits, aux

    def predict(
        self,
        sparse: Tensor, numeric: Tensor,
        progression: Tensor, candidates: Tensor, cand_feats: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            logits = self.forward(sparse, numeric, progression, candidates, cand_feats)
        best_idx = logits.argmax(dim=-1)
        B = candidates.size(0)
        return candidates[torch.arange(B, device=candidates.device), best_idx]


# ---------------------------------------------------------------------------
# 5. MeldDecisionModel（鳴くか否か 4クラス分類）
# ---------------------------------------------------------------------------
class MeldDecisionModel(nn.Module):
    """
    捨て牌が出た時、「スルー / ポン / チー / 明槓」の4クラスを出力する。

    入力:
      sparse / numeric / progression  ← SharedEncoder に渡す
      called_tile_ids (B,)            ← 鳴き対象の牌ID（埋め込みで局面に追加）

    出力:
      logits (B, 4)  ← cross_entropy で学習

    学習データ:
      event_type == "meld_decision" のサンプルのみ使用
      meld_label フィールドが正解クラス
    """
    def __init__(
        self,
        d_model:          int   = 256,
        nhead:            int   = 8,
        enc_layers:       int   = 4,
        dim_feedforward:  int   = 1024,
        dropout:          float = 0.1,
        num_meld_labels:  int   = NUM_MELD_LABELS,
        encoder: SharedMahjongEncoder | None = None,
    ):
        super().__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = SharedMahjongEncoder(
                d_model=d_model, nhead=nhead, num_layers=enc_layers,
                dim_feedforward=dim_feedforward, dropout=dropout,
            )

        # 鳴き対象の牌を別途 Embedding で局面に足す
        self.called_tile_emb = nn.Embedding(NUM_ACTION_TYPES, d_model)

        # memory の [CLS] 相当として mean pooling → 分類ヘッド
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_meld_labels),
        )

    def forward(
        self,
        sparse:          Tensor,   # (B, s_len)
        numeric:         Tensor,   # (B, NUMERIC_DIM)
        progression:     Tensor,   # (B, p_len)
        called_tile_ids: Tensor,   # (B,)  鳴き対象牌ID（ACT_* スペース）
        pad_mask_ext:    Tensor | None = None,  # 外部から渡す場合
    ) -> Tensor:
        memory, pad_mask = self.encoder(sparse, numeric, progression)

        # 鳴き対象牌の埋め込みを memory の末尾に追加
        tile_emb = self.called_tile_emb(
            called_tile_ids.clamp(0, NUM_ACTION_TYPES - 1)
        ).unsqueeze(1)  # (B, 1, d)
        memory_ext  = torch.cat([memory, tile_emb], dim=1)
        tile_pad    = torch.zeros(memory.size(0), 1, dtype=torch.bool,
                                  device=memory.device)
        pad_mask_ext = torch.cat([pad_mask, tile_pad], dim=1)

        # 有効トークンの mean pooling → 分類
        valid = ~pad_mask_ext                               # (B, seq+1)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1).float()
        pooled = (memory_ext * valid.unsqueeze(-1).float()).sum(dim=1) / denom
        return self.classifier(pooled)  # (B, 4)

    def predict(
        self,
        sparse: Tensor, numeric: Tensor,
        progression: Tensor, called_tile_ids: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            logits = self.forward(sparse, numeric, progression, called_tile_ids)
        return logits.argmax(dim=-1)  # (B,) : 0=skip 1=pon 2=chi 3=daiminkan


# ---------------------------------------------------------------------------
# 6. RiichiDecisionModel（立直するか + どの牌で立直するか）
# ---------------------------------------------------------------------------
class RiichiDecisionModel(nn.Module):
    """
    立直の2段階判定モデル。

    出力1: riichi_logit  (B, 1)  → sigmoid → 立直するか否か
    出力2: tile_logits   (B, 37) → 立直打牌選択（どの牌で立直するか）

    学習データ:
      event_type == "riichi" のサンプル   → riichi_label=1
      event_type == "dahai" で立直可能局面 → riichi_label=0 （要別途生成 or 推論時のみ）

    NOTE: 現状 riichi_label=0（立直しなかった）サンプルはデータセットに含まれない。
          学習は riichi_label=1 の tile_logits（どの牌で立直するか）のみでも有効。
          立直するか否かの binary 判定は今後 dahai サンプルの拡張で対応。
    """
    def __init__(
        self,
        d_model:         int   = 256,
        nhead:           int   = 8,
        enc_layers:      int   = 4,
        dim_feedforward: int   = 1024,
        dropout:         float = 0.1,
        encoder: SharedMahjongEncoder | None = None,
    ):
        super().__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = SharedMahjongEncoder(
                d_model=d_model, nhead=nhead, num_layers=enc_layers,
                dim_feedforward=dim_feedforward, dropout=dropout,
            )

        # mean pooling → 2つのヘッド
        self.riichi_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        self.tile_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, NUM_TILE_TYPES),  # 37種
        )

    def forward(
        self,
        sparse:      Tensor,
        numeric:     Tensor,
        progression: Tensor,
    ) -> tuple[Tensor, Tensor]:
        memory, pad_mask = self.encoder(sparse, numeric, progression)
        valid  = ~pad_mask
        denom  = valid.sum(dim=1, keepdim=True).clamp(min=1).float()
        pooled = (memory * valid.unsqueeze(-1).float()).sum(dim=1) / denom

        riichi_logit = self.riichi_head(pooled)   # (B, 1)
        tile_logits  = self.tile_head(pooled)     # (B, 37)
        return riichi_logit, tile_logits

    def predict(
        self,
        sparse: Tensor, numeric: Tensor, progression: Tensor,
        riichi_threshold: float = 0.5,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            do_riichi (B,) Bool  : 立直するか
            tile_id   (B,) Long  : 立直打牌の tile_id（0〜36）
        """
        with torch.no_grad():
            riichi_logit, tile_logits = self.forward(sparse, numeric, progression)
        do_riichi = (riichi_logit.squeeze(-1).sigmoid() >= riichi_threshold)
        tile_id   = tile_logits.argmax(dim=-1)
        return do_riichi, tile_id




# ---------------------------------------------------------------------------
# 8. 動作確認
# ---------------------------------------------------------------------------
def _make_dummy_batch(B: int = 4, s_len: int = 19, p_len: int = 10):
    sparse      = torch.randint(1, SPARSE_VOCAB_SIZE,    (B, s_len))
    numeric     = torch.randn(B, NUMERIC_DIM)
    progression = torch.randint(1, PROG_VOCAB_SIZE,      (B, p_len))
    candidates  = torch.full((B, MAX_ACTION_CANDIDATES), PADDING_ACTION_ID)
    candidates[:, :12] = torch.randint(0, NUM_ACTION_TYPES - 1, (B, 12))
    label       = torch.randint(0, 12, (B,))
    called_tile = torch.randint(0, 37, (B,))
    meld_label  = torch.randint(0, NUM_MELD_LABELS, (B,))
    cand_feats  = torch.rand(B, MAX_ACTION_CANDIDATES, CAND_FEAT_DIM)
    return sparse, numeric, progression, candidates, label, called_tile, meld_label, cand_feats


if __name__ == "__main__":
    import torch.nn.functional as F

    print("=== 分離モデル 動作確認 ===\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  NUMERIC_DIM: {NUMERIC_DIM}\n")

    B = 4
    sparse, numeric, progression, candidates, label, called_tile, meld_label, cand_feats = \
        _make_dummy_batch(B=B)
    cand_feats = cand_feats.to(device)
    sparse, numeric, progression = (
        sparse.to(device), numeric.to(device), progression.to(device))
    candidates, label = candidates.to(device), label.to(device)
    called_tile = called_tile.to(device)
    meld_label  = meld_label.to(device)

    # ---- SharedEncoder ----
    enc = SharedMahjongEncoder(d_model=128, nhead=4, num_layers=2,
                                dim_feedforward=256).to(device)
    mem, pmask = enc(sparse, numeric, progression)
    print(f"SharedEncoder memory: {tuple(mem.shape)}  → OK")

    # ---- MahjongTransformer (shared enc) ----
    dahai = MahjongTransformer(d_model=128, nhead=4, enc_layers=2,
                                dec_layers=1, dim_feedforward=256,
                                encoder=enc).to(device)
    logits = dahai(sparse, numeric, progression, candidates, cand_feats)
    assert torch.isfinite(logits[candidates != PADDING_ACTION_ID]).all()
    loss_d = F.cross_entropy(logits, label)
    print(f"MahjongTransformer  logits: {tuple(logits.shape)}  loss: {loss_d.item():.4f}  → OK")

    # ---- MeldDecisionModel ----
    meld = MeldDecisionModel(d_model=128, nhead=4, enc_layers=2,
                              dim_feedforward=256, encoder=enc).to(device)
    meld_logits = meld(sparse, numeric, progression, called_tile)
    loss_m = F.cross_entropy(meld_logits, meld_label)
    print(f"MeldDecisionModel   logits: {tuple(meld_logits.shape)}  loss: {loss_m.item():.4f}  → OK")

    # ---- RiichiDecisionModel ----
    riichi = RiichiDecisionModel(d_model=128, nhead=4, enc_layers=2,
                                  dim_feedforward=256, encoder=enc).to(device)
    r_logit, t_logits = riichi(sparse, numeric, progression)
    riichi_label = torch.ones(B, 1, device=device)
    tile_label   = torch.randint(0, 37, (B,), device=device)
    loss_r = F.binary_cross_entropy_with_logits(r_logit, riichi_label) \
           + F.cross_entropy(t_logits, tile_label)
    print(f"RiichiDecisionModel r:{tuple(r_logit.shape)} t:{tuple(t_logits.shape)}  loss: {loss_r.item():.4f}  → OK")

    total = sum(p.numel() for p in enc.parameters()) \
          + sum(p.numel() for p in dahai.decoder.parameters()) \
          + sum(p.numel() for p in meld.parameters() if p.requires_grad) \
          + sum(p.numel() for p in riichi.parameters() if p.requires_grad)
    print(f"\n合計パラメータ（Encoder共有時）: {total:,}  ({total/1e6:.1f}M)")
    print("\n→ 全モデル 動作確認 OK")
