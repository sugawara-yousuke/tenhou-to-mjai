"""
hand_quality.py  ―  GRPO 報酬シェーピング用の手牌品質シグナル
================================================================

背景（前セッションの結論）:
  固定山で8分岐しても、局収支(near報酬)はグループ内で平均1.58種類しか値が
  出ず、半数のグループが range=0.000。tie-zeroing で約8割のサンプルがゼロ化
  され、GRPO の学習信号がほぼ消えていた。
  → 打牌ごとに必ず差がつく「即時の手牌品質」を報酬に混ぜて信号を復活させる。

このモジュールが提供するのは打牌ごとに値が変わる3シグナルの計算基盤:
  - hand_shanten : 打牌後13枚手牌の向聴数（副露数を考慮）
  - hand_ukeire  : 向聴を進める有効牌の「場に残っている実枚数」合計
  （危険度 danger は被リーチ状況・現物/スジ情報を要するため self_play 側で算出）

牌IDは self_play/tokenizer と同じ 0-33 の34種 + 赤ドラ(34/35/36)。
正規化は shanten_number 内の _normalize_hand が吸収するのでそのまま渡してよい。
"""

from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
from mahjong_tokenizer import shanten_number, _normalize_hand


# 各牌種は山+他家手牌+自分に最大4枚。ここでは「観測できない残り枚数」を
# 厳密に数えず、seen（見えている枚数）を差し引く簡易版と、
# 手牌のみから計算する素朴版の両方を用意する。


def hand_shanten(hand_ids: list[int], num_melds: int = 0) -> int:
    """
    打牌後の手牌（13 - 3*num_melds 枚を想定）の向聴数を返す。
    0=テンパイ、-1=和了、n=n向聴。tokenizer の shanten_number をそのまま利用。
    """
    return shanten_number(hand_ids, num_melds)


def _counts34(hand_ids: list[int]) -> list[int]:
    counts = [0] * 34
    for t in _normalize_hand(hand_ids):
        if 0 <= t <= 33:
            counts[t] += 1
    return counts


def hand_ukeire(
    hand_ids: list[int],
    num_melds: int = 0,
    seen_counts: list[int] | None = None,
) -> tuple[int, int]:
    """
    有効牌（引くと向聴が1つ進む牌種）の枚数を数える。

    Returns:
        (kinds, tiles)
          kinds : 有効牌の種類数
          tiles : 有効牌の場に残っている実枚数の合計
                  （seen_counts が与えられれば 4 - seen を各種で加算、
                    無ければ 4 - 自手の枚数 で近似）

    seen_counts: 長さ34の「見えている枚数」（自手＋捨て牌＋ドラ表示＋副露）。
                 与えると残り枚数がより正確になる。
    """
    base = hand_shanten(hand_ids, num_melds)
    if base < 0:
        # すでに和了形。有効牌の概念なし。
        return 0, 0

    own = _counts34(hand_ids)
    kinds = 0
    tiles = 0
    for t in range(34):
        # その牌種はもう4枚見えている/持っているなら引けない
        already = seen_counts[t] if seen_counts is not None else own[t]
        remain = 4 - already
        if remain <= 0:
            continue
        # t を1枚加えて向聴が下がるか
        trial = hand_ids + [t]
        if shanten_number(trial, num_melds) < base:
            kinds += 1
            tiles += remain
    return kinds, tiles


def shaping_signal(
    hand_ids: list[int],
    num_melds: int = 0,
    seen_counts: list[int] | None = None,
) -> dict:
    """
    打牌後手牌のシェーピング用シグナルをまとめて返す。
    self_play 側で danger を足して最終報酬に合成する想定。
    """
    s = hand_shanten(hand_ids, num_melds)
    kinds, tiles = hand_ukeire(hand_ids, num_melds, seen_counts)
    return {"shanten": s, "ukeire_kinds": kinds, "ukeire_tiles": tiles}
