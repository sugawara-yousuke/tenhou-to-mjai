"""
danger_features.py
==================
「場の状況」と「自分の手牌」の関連（＝各牌の放銃リスク）を明示的な特徴量にする。

[改修点]
  - 現物の定義を実戦どおりに拡張:
      state.safe_tiles[opp] = opp 自身の河 ＋ opp のリーチ後に誰かが通した牌
    （tokenizer / self_play 共通の st_dahai が保守する）。無い旧 state では河のみ。
  - ノーチャンス（壁）判定を追加。従来の「その牌自体が3枚見え」は単騎/シャボ
    にしか効かないので、両面待ちが物理的に成立しない牌を別途割り引く。
  - build_danger_context() で 3x34 行列を1回だけ計算し、numeric 用サマリ
    (build_danger_vector) と候補ごとの特徴 (tokenizer.build_cand_feats) の両方で
    使い回す（二重計算の排除）。

危険度は「厳密な放銃確率」ではなく、人間の押し引き判断に対応するヒューリスティック。
牌種は 0..33 の 34種（赤5は通常5に正規化）。
"""

from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)

NUM_TILE_KINDS = 34
NUM_OPP        = 3

DANGER_PER_OPP = 4
DANGER_SUMMARY = 4
DANGER_DIM     = DANGER_PER_OPP * NUM_OPP + DANGER_SUMMARY   # = 16


def _norm34(tid: int) -> int:
    if tid == 34: return 4
    if tid == 35: return 13
    if tid == 36: return 22
    return tid if 0 <= tid <= 33 else 0


def _is_honor(k34: int) -> bool:
    return k34 >= 27


def _suit_num(k34: int):
    if k34 >= 27:
        return None
    return (k34 // 9, k34 % 9)


def _tid_of(pai_str: str) -> int:
    from mahjong_tokenizer import TILE_STR_TO_ID   # 遅延importで循環回避
    return TILE_STR_TO_ID.get(pai_str, 0)


def count_visible(state, player) -> list[int]:
    """player から見えている枚数（自手牌＋全員の河＋全副露＋ドラ表示）。"""
    vis = [0] * NUM_TILE_KINDS
    for t in state.tehais[player]:
        vis[_norm34(_tid_of(t))] += 1
    for pi in range(4):
        for tid in state.sutehais[pi]:
            vis[_norm34(tid)] += 1
    # 副露: meld_info があれば構成牌を正確に数える（鳴いた牌は河と二重になるので除く）
    meld_info = getattr(state, "meld_info", None)
    if meld_info is not None:
        for pi in range(4):
            for m in meld_info[pi]:
                for c in (m.get("consumed") or []):
                    vis[_norm34(_tid_of(c))] += 1
                if m.get("type") == "kakan" and m.get("added"):
                    vis[_norm34(_tid_of(m["added"]))] += 1
    else:
        for pi in range(4):
            for tid, _mt in state.melds[pi]:
                vis[_norm34(tid)] += 1
    for dm in state.dora_markers:
        vis[_norm34(_tid_of(dm))] += 1
    return [min(v, 4) for v in vis]


_count_visible = count_visible   # 後方互換


def genbutsu_set(state, opp: int) -> set[int]:
    """opp に対して絶対安全な牌種34の集合。"""
    safe = getattr(state, "safe_tiles", None)
    if safe is not None:
        return set(safe[opp])
    return {_norm34(tid) for tid in state.sutehais[opp]}


_genbutsu_set = genbutsu_set


def _suji_safe(genbutsu: set[int], k34: int) -> bool:
    sn = _suit_num(k34)
    if sn is None:
        return False
    suit, num = sn
    g_nums = {g % 9 for g in genbutsu if g < 27 and g // 9 == suit}
    if num <= 2:
        return (num + 3) in g_nums
    if num >= 6:
        return (num - 3) in g_nums
    # 4,5,6 は両スジが通って初めて両面に当たらない（片スジは半分だけ安全）
    return (num - 3) in g_nums and (num + 3) in g_nums


def _half_suji(genbutsu: set[int], k34: int) -> bool:
    sn = _suit_num(k34)
    if sn is None:
        return False
    suit, num = sn
    if not (3 <= num <= 5):
        return False
    g_nums = {g % 9 for g in genbutsu if g < 27 and g // 9 == suit}
    return ((num - 3) in g_nums) != ((num + 3) in g_nums)


def _no_chance(visible: list[int], k34: int) -> bool:
    """k34 を待ちに含む両面搭子がすべて壁で消えているか（ノーチャンス）。"""
    sn = _suit_num(k34)
    if sn is None:
        return False
    suit, num = sn
    base = suit * 9
    alive = False
    # 下側の搭子 (num-2, num-1) で num 待ち ※ (num-2,num-1) が 1-2 のペンチャンは除外しない
    if num >= 2:
        if visible[base + num - 2] < 4 and visible[base + num - 1] < 4:
            alive = True
    if num <= 6:
        if visible[base + num + 1] < 4 and visible[base + num + 2] < 4:
            alive = True
    return not alive


def _base_danger(k34: int, opp_riichi: bool) -> float:
    if _is_honor(k34):
        return 0.35 if opp_riichi else 0.25
    _, num = _suit_num(k34)
    center = 1.0 - abs(num - 4) / 4.0
    return 0.30 + 0.45 * center


def _threat(state, opp: int) -> float:
    if state.riichi_flags[opp]:
        return 1.0
    open_melds = state.meld_counts[opp]
    ank = getattr(state, "ankan_counts", None)
    if ank is not None:
        open_melds -= ank[opp]
    if open_melds <= 0:
        return 0.10
    return min(0.15 + 0.20 * open_melds, 0.9)


def tile_danger_against(state, player: int, opp: int, k34: int,
                        visible: list[int], genb: set[int] | None = None,
                        threat: float | None = None) -> float:
    if genb is None:
        genb = genbutsu_set(state, opp)
    if k34 in genb:
        return 0.0
    if threat is None:
        threat = _threat(state, opp)
    d = _base_danger(k34, state.riichi_flags[opp])
    if _suji_safe(genb, k34):
        d *= 0.45
    elif _half_suji(genb, k34):
        d *= 0.75
    seen = visible[k34]
    if _is_honor(k34):
        if seen >= 3:   d *= 0.2
        elif seen == 2: d *= 0.5
    else:
        if _no_chance(visible, k34):
            d *= 0.35
        if seen >= 3:   d *= 0.6
        elif seen == 2: d *= 0.85
    return max(0.0, min(1.0, d * threat))


def build_danger_context(state, player: int, all_kinds: bool = False) -> dict:
    """
    1サンプルにつき1回だけ計算する共有コンテキスト。
      matrix[rel-1][k34] : 危険度 0..1   (rel=1:下家, 2:対面, 3:上家)
                           ※既定では自分の手牌にある牌種だけ計算する（それ以外は 0）。
                             numeric サマリも候補特徴も手牌の牌しか参照しないため。
      genb[rel-1]        : その相手への現物集合
      threat[rel-1]      : 脅威度
      riichi[rel-1]      : リーチ中か
      visible            : 見えている枚数[34]
    """
    visible = count_visible(state, player)
    kinds = range(NUM_TILE_KINDS) if all_kinds else \
        sorted({_norm34(_tid_of(t)) for t in state.tehais[player]})
    matrix, genbs, threats, riichis = [], [], [], []
    for rel in range(1, 4):
        opp = (player + rel) % 4
        g = genbutsu_set(state, opp)
        th = _threat(state, opp)
        row = [0.0] * NUM_TILE_KINDS
        for k in kinds:
            row[k] = tile_danger_against(state, player, opp, k, visible, g, th)
        matrix.append(row)
        genbs.append(g); threats.append(th); riichis.append(bool(state.riichi_flags[opp]))
    return {"matrix": matrix, "genb": genbs, "threat": threats,
            "riichi": riichis, "visible": visible}


def build_tile_danger_matrix(state, player: int) -> list[list[float]]:
    return build_danger_context(state, player, all_kinds=True)["matrix"]


def build_danger_vector(state, player: int, ctx: dict | None = None) -> list[float]:
    """
    numeric に連結する危険度サマリ（長さ DANGER_DIM=16）。
      各相手 rel=1,2,3 について [脅威度, 手牌の現物率, 手牌の最大リスク, 平均リスク]
      + [最脅威相手への最大リスク, 同 平均リスク, 全体最大リスク, 安全牌保有率]
    """
    if ctx is None:
        ctx = build_danger_context(state, player)
    hand = [_norm34(_tid_of(t)) for t in state.tehais[player]]
    vec: list[float] = []
    per_opp = []
    safe_any = [False] * len(hand)
    for r in range(3):
        row, genb, threat = ctx["matrix"][r], ctx["genb"][r], ctx["threat"][r]
        if hand:
            risks = [row[k] for k in hand]
            max_r, mean_r = max(risks), sum(risks) / len(risks)
            genb_rate = sum(1 for k in hand if k in genb) / len(hand)
            for i, k in enumerate(hand):
                if k in genb:
                    safe_any[i] = True
        else:
            max_r = mean_r = genb_rate = 0.0
        vec.extend([threat, genb_rate, max_r, mean_r])
        per_opp.append((threat, max_r, mean_r))
    top = max(per_opp, key=lambda s: s[0])
    all_max = max(s[1] for s in per_opp)
    safe_rate = (sum(safe_any) / len(safe_any)) if safe_any else 0.0
    vec.extend([top[1], top[2], all_max, safe_rate])
    assert len(vec) == DANGER_DIM
    return vec
