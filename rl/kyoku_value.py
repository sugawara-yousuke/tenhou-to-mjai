"""
kyoku_value.py  ―  打ち切り地点の局面価値（期待局収支）
================================================================
GRPO のプレイアウトを局終了まで回さず、分岐から horizon 巡で打ち切り、
その時点の局面を「この局がこの先どう決着しそうか」の期待値に置き換える。

なぜ必要か:
  局終了まで回すと、報酬は「和了できたか/放銃したか」という低頻度イベントに
  支配される。降りは結果が決定的（放銃しないことが確定）なのに対し、押しの
  利益は確率的にしか出ないので、グループ内比較では押しが系統的に過小評価され、
  方策がベタオリ側に潰れた（grpo4/grpo6 で再現）。
  期待値なら押しの価値が決定的な値として出るので、この非対称が消える。

設計の原則:
  - 全枝を同じ地平・同じ関数で評価する。価値関数の誤差のうち枝に共通する分は
    GRPO のグループ内中央値で相殺されるので、精度より一貫性が重要。
  - 「いま切った牌が当たるか」は地平の中で実際に打つので、ここでは扱わない。
    この関数が担うのは地平より先に残る分だけ。
  - self_play は他家の手牌が見える。報酬計算は学習の外側なので、見える情報は
    そのまま使ってよい（モデルの入力には一切入らない）。

公開関数:
  estimate_score_delta(state, tokenizer) -> list[float]
      4人ぶんの期待局収支（点）。合計はおおむね 0。
  estimate_scores_after(state, tokenizer) -> list[float]
      state.scores に上の期待値を足したもの。reward.placement_reward にそのまま渡せる。

パラメータはすべてこのファイルの先頭に集約した。調整するときはここだけ見ればよい。
"""

from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)

from mahjong_tokenizer import TILE_STR_TO_ID, shanten_from_counts
from danger_features import count_visible, genbutsu_set, _norm34

# --- パラメータ（すべて目安値。calibrate_kyoku_value.py で検証する） -------
RON_RATIO       = 0.65    # 和了のうちロンの割合（残りはツモ）
TENPAI_RON_MULT = 3.0     # ロンを含めた実効的な和了機会の倍率（ツモのみなら 1.0）
TURNS_TO_TENPAI = 3.0     # 1向聴あたり、テンパイまでに要する平均ツモ数
RYUKYOKU_PAY    = 3000.0  # 流局時のテンパイ料（総額）
OYA_MULT        = 1.5     # 親の和了打点の倍率

# 打点の目安（子の和了。親は OYA_MULT 倍）
PTS_BASE_RIICHI = 5800.0  # 立直していれば裏/一発/ツモが乗るので高め
PTS_BASE_MENZEN = 3900.0  # 門前ダマ
PTS_BASE_OPEN   = 3000.0  # 副露手
PTS_PER_DORA    = 1300.0  # ドラ1枚あたりの上乗せ
# -------------------------------------------------------------------------


def _hand_counts34(tehai: list) -> list:
    counts = [0] * 34
    for t in tehai:
        k = _norm34(TILE_STR_TO_ID.get(t, 0))
        counts[k] += 1
    return counts


def _shanten_and_ukeire(counts: list, num_melds: int, visible: list):
    """向聴数と、場に残っている有効牌の枚数を返す。"""
    base = shanten_from_counts(tuple(counts), num_melds)
    if base < 0:
        return -1, 0
    tiles = 0
    cand = set()
    for k in range(34):
        if not counts[k]:
            continue
        cand.add(k)
        if k < 27:
            lo, hi = (k // 9) * 9, (k // 9) * 9 + 8
            for d in (-2, -1, 1, 2):
                if lo <= k + d <= hi:
                    cand.add(k + d)
    for k in cand:
        remain = 4 - visible[k]
        if remain <= 0:
            continue
        counts[k] += 1
        s = shanten_from_counts(tuple(counts), num_melds)
        counts[k] -= 1
        if s < base:
            tiles += remain
    return base, tiles


def _win_prob(shanten: int, ukeire: int, draws: float, unseen: int) -> float:
    """
    その局で和了する確率の近似。
      - テンパイ: 1手ごとに ukeire/unseen で和了機会。ロンぶんを TENPAI_RON_MULT 倍。
      - n向聴: テンパイまでに TURNS_TO_TENPAI × n ツモかかるとみなし、残りをテンパイ扱い。
    厳密さは求めない。向聴・受け入れ・残り巡目に対して単調であればよい。
    """
    if shanten < 0:
        return 1.0
    if draws <= 0 or unseen <= 0:
        return 0.0
    eff_draws = draws - TURNS_TO_TENPAI * max(shanten, 0)
    if eff_draws <= 0:
        return 0.0
    if shanten > 0 and ukeire <= 0:
        return 0.0
    # テンパイ時の待ち枚数。n向聴の手は、テンパイしたとき平均的な待ち(6枚)になるとみなす
    waits = ukeire if shanten == 0 else 6.0
    p_miss = max(0.0, 1.0 - TENPAI_RON_MULT * waits / unseen)
    return 1.0 - p_miss ** eff_draws


def _hand_value(state, p: int, counts: list) -> float:
    """和了したときの打点の目安。"""
    if state.riichi_flags[p]:
        base = PTS_BASE_RIICHI
    elif state.meld_counts[p] - getattr(state, "ankan_counts", [0] * 4)[p] <= 0:
        base = PTS_BASE_MENZEN
    else:
        base = PTS_BASE_OPEN
    n_dora = 0
    for dm in state.dora_markers:
        d = TILE_STR_TO_ID.get(dm, 0)
        k = _norm34(d)
        nxt = (k + 1) % 9 + (k // 9) * 9 if k < 27 else (27 + (k - 27 + 1) % 4 if k < 31 else 31 + (k - 31 + 1) % 3)
        n_dora += counts[nxt]
    for t in state.tehais[p]:
        if t in ("0m", "0p", "0s"):
            n_dora += 1
    pts = base + PTS_PER_DORA * n_dora
    return pts * (OYA_MULT if p == state.oya else 1.0)


def _deal_in_weights(state, losers: list) -> list:
    """
    ロンの放銃者の割り振り。安全牌を持っていない人ほど放銃しやすいとみなす。
    ベタオリ（現物を抱える）が報われ、安全牌が尽きた手は不利に評価される。
    """
    w = []
    for p in losers:
        hand = [_norm34(TILE_STR_TO_ID.get(t, 0)) for t in state.tehais[p]]
        if not hand:
            w.append(1.0); continue
        safe = set()
        for q in range(4):
            if q != p and state.riichi_flags[q]:
                safe |= genbutsu_set(state, q)
        safe_rate = sum(1 for k in hand if k in safe) / len(hand)
        w.append(0.2 + (1.0 - safe_rate))
    tot = sum(w) or 1.0
    return [x / tot for x in w]


def estimate_score_delta(state, tokenizer=None) -> list:
    """
    打ち切り地点から見た、この局の期待収支（4人ぶん、点）。

    各プレイヤーの和了確率と打点を見積もり、
      和了者の収入 = 打点
      支払い       = ツモなら他3人で分担、ロンなら「安全牌の少ない人」に寄せて分担
    とする。誰も和了しないぶんは流局として、テンパイ料をやりとりする。
    """
    visible_by = [count_visible(state, p) for p in range(4)]
    draws = max(state.wall_count, 0) / 4.0

    win_p, pts, tenpai = [], [], []
    for p in range(4):
        counts = _hand_counts34(state.tehais[p])
        n_melds = state.meld_counts[p]
        sh, uke = _shanten_and_ukeire(counts, n_melds, visible_by[p])
        unseen = max(136 - sum(visible_by[p]) - 13 * 3, 1)
        win_p.append(_win_prob(sh, uke, draws, unseen))
        pts.append(_hand_value(state, p, counts))
        tenpai.append(sh <= 0)

    # 和了は1人だけ起きるとみなして正規化（合計が1を超えたら按分）
    tot = sum(win_p)
    if tot > 1.0:
        win_p = [x / tot for x in win_p]
        tot = 1.0

    delta = [0.0] * 4
    for w in range(4):
        if win_p[w] <= 0:
            continue
        gain = pts[w] * win_p[w]
        delta[w] += gain
        losers = [p for p in range(4) if p != w]
        # ツモ: 3人で等分（親の支払いが多い点は無視する近似）
        tsumo_share = gain * (1.0 - RON_RATIO) / 3.0
        for p in losers:
            delta[p] -= tsumo_share
        # ロン: 安全牌の少ない人に寄せる
        ron_total = gain * RON_RATIO
        for p, wt in zip(losers, _deal_in_weights(state, losers)):
            delta[p] -= ron_total * wt

    # 流局ぶん
    p_draw = max(0.0, 1.0 - tot)
    if p_draw > 0:
        n = sum(tenpai)
        if 0 < n < 4:
            for p in range(4):
                delta[p] += p_draw * ((RYUKYOKU_PAY / n) if tenpai[p] else -(RYUKYOKU_PAY / (4 - n)))
    return delta


def estimate_scores_after(state, tokenizer=None) -> list:
    """state.scores に期待局収支を足したもの。placement_reward にそのまま渡せる。"""
    d = estimate_score_delta(state, tokenizer)
    return [state.scores[p] + d[p] for p in range(4)]
