"""
rule_based.py  ―  学習済みモデルを使わない簡易ルールベースの打ち手（評価の基準用）

  - 打牌: 切った後の向聴数が最小になる牌を選ぶ（同点ならリーチ宣言を優先、残りは乱数）
  - 鳴き: 25% の確率でランダムに鳴き、それ以外は見送る
  - 槓:   候補があれば槓する

席ごとにモデルとルールベースを混ぜられるよう、MahjongSimulator を拡張する。
player_models の値に RULE_BASED を入れた席がルールベースで打つ。

山の再現性について:
  eval_paired.py は random.seed で山を揃える。ルールベースが Python の global random を
  消費すると山がずれるので、ここでは席ごとに独立した random.Random を使う。
"""
from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
import random

from mahjong_tokenizer import (shanten_number, TILE_STR_TO_ID, _tid_norm,
                               ACT_RIICHI_BASE, ACT_PASS_BASE, PADDING_ACTION_ID)
from self_play import MahjongSimulator


class _RuleBased:
    """player_models に入れる目印。モデルの代わりに使う。"""
    is_rule_based = True

    def __repr__(self):
        return "RULE_BASED"


RULE_BASED = _RuleBased()


def is_rule_based(m) -> bool:
    return getattr(m, "is_rule_based", False)


class MixedSimulator(MahjongSimulator):
    """席ごとに学習済みモデルとルールベースを混在させられるシミュレータ。"""

    def __init__(self, model, device, watch=False, rule_seed: int = 0):
        super().__init__(model, device, watch=watch)
        self._rng = random.Random(rule_seed)

    def reseed_rule(self, seed: int):
        self._rng.seed(seed)

    def _decide_dahai(self, K, a, sample):
        if not is_rule_based(self._model_for(a)) or self.rl is not None or self._is_human(a):
            return super()._decide_dahai(K, a, sample)
        if self._sample_log is not None:
            self._sample_log.append(sample)
        st = K.state
        ids = [TILE_STR_TO_ID[t] for t in st.tehais[a]]
        best = []
        for i, tok in enumerate(sample["possible_actions"]):
            if tok == PADDING_ACTION_ID:
                continue
            k = tok % 37
            j = next(x for x, t in enumerate(ids) if _tid_norm(t) == k)
            sh = shanten_number(ids[:j] + ids[j + 1:], st.meld_counts[a])
            best.append((sh - (0.5 if tok >= ACT_RIICHI_BASE else 0), self._rng.random(), tok, i))
        _, _, tok, i = min(best)
        poss = sample["possible_actions"]
        return tok, i, {"pred_idx": i, "pred_token": tok, "probs": None, "poss": poss}

    def _choose(self, player, sample, kind, context=None):
        if not is_rule_based(self._model_for(player)) or self._is_human(player):
            return super()._choose(player, sample, kind, context)
        if self._sample_log is not None:
            self._sample_log.append(sample)
        valid = [t for t in sample["possible_actions"] if t != PADDING_ACTION_ID]
        if kind == "kan":
            return valid[-1]
        if len(valid) > 1 and self._rng.random() < 0.25:
            return self._rng.choice(valid[1:])
        return ACT_PASS_BASE
