"""
reward.py  ―  GRPO の報酬
================================================================
既定は「順位期待値の変化」(placement)。

  r = PV(局後の点数, 残り局数) − PV(局前の点数, 残り局数)  ＋ w_score × 局収支/8000

  PV = 最終順位点の期待値。各ペアの逆転確率を正規分布で近似し（点差 / (σ√残り局数)）、
       そこから順位分布を作って順位点 (既定: 天鳳 +90/+45/0/−135) の期待値を取る。
       残り0局（終局）では実際の順位点そのもの。

  なぜ変えたか:
    旧 near 報酬は局収支を ±8000 でクリップし、和了 +0.5 / 放銃 −0.5 / リーチ +0.1 /
    流局テンパイ +0.2 を足していた。これは点数状況を一切見ない（東1の 1000点 とオーラスの
    順位が入れ替わる 1000点 が同じ）うえ、「リーチしただけで加点」「テンパイを取るだけで加点」
    なので、押し引きや状況判断とは逆向きの癖を強化し得る。
    PV の差分はポテンシャル型なので、半荘を通して足すと「最終順位点 − 開始時の期待値」に
    一致する。つまり最終目的(順位)をそのまま最適化しつつ、毎局フィードバックが得られる。
    序盤は PV がほぼ点数に比例するので局収支を取りに行き、終盤ほど順位の入れ替わりに敏感になる。

旧 compute_near_reward / KyokuResult は比較実験用に残してある（--reward near）。
"""

from __future__ import annotations
import math
from dataclasses import dataclass

TENHOU_RANK_PTS = (90.0, 45.0, 0.0, -135.0)
SIGMA_PAIR = 8000.0      # 1局あたりの「2者間の点差」の変動幅（標準偏差）の目安


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def rank_distribution(scores: list, player: int, kyoku_left: float, sigma_pair: float = SIGMA_PAIR) -> list:
    """player の最終順位分布 [P(1位), ..., P(4位)]。同点は起家に近い席が上位。"""
    p_beaten = []
    for q in range(4):
        if q == player:
            continue
        d = scores[q] - scores[player]
        if kyoku_left <= 0:
            p = 1.0 if (d > 0 or (d == 0 and q < player)) else 0.0
        else:
            p = _phi(d / (sigma_pair * math.sqrt(kyoku_left)))
        p_beaten.append(p)
    dist = [1.0, 0.0, 0.0, 0.0]          # 上にいる人数の分布（独立近似）
    for p in p_beaten:
        nxt = [0.0] * 4
        for k in range(4):
            if dist[k] == 0.0:
                continue
            nxt[k] += dist[k] * (1 - p)
            if k + 1 < 4:
                nxt[k + 1] += dist[k] * p
        dist = nxt
    return dist


def placement_value(scores, player, kyoku_left, rank_pts=TENHOU_RANK_PTS, sigma_pair=SIGMA_PAIR) -> float:
    """順位点の期待値を |最下位の順位点| で割ったもの（概ね −1 〜 +0.67）。"""
    dist = rank_distribution(scores, player, kyoku_left, sigma_pair)
    scale = max(abs(x) for x in rank_pts) or 1.0
    return sum(d * r for d, r in zip(dist, rank_pts)) / scale


def kyoku_left_before(bakaze_idx: int, kyoku: int) -> float:
    """その局を含めた残り局数の目安（連荘は数えない。西入後は 1）。"""
    return float(max(8 - (bakaze_idx * 4 + kyoku - 1), 1))


def placement_reward(scores_before, scores_after, player, left_before: float, left_after: float,
                     w_score: float = 0.1, rank_pts=TENHOU_RANK_PTS, sigma_pair=SIGMA_PAIR) -> float:
    dv = (placement_value(scores_after, player, left_after, rank_pts, sigma_pair)
          - placement_value(scores_before, player, left_before, rank_pts, sigma_pair))
    ds = (scores_after[player] - scores_before[player]) / 8000.0
    return dv + w_score * max(-3.0, min(3.0, ds))


# ---------------------------------------------------------------------------
# 旧 near 報酬（比較用）
# ---------------------------------------------------------------------------
@dataclass
class KyokuResult:
    player_id: int
    score_delta: int
    agari: bool
    houju: bool
    tsumo_agari: bool
    riichi: bool
    tenpai_ryukyoku: bool


def compute_near_reward(result: KyokuResult) -> float:
    r = max(-1.0, min(1.0, result.score_delta / 8000.0))
    r += 0.5 if result.agari else 0.0
    r -= 0.5 if result.houju else 0.0
    r += 0.1 if result.tsumo_agari else 0.0
    r += 0.1 if result.riichi else 0.0
    r += 0.2 if result.tenpai_ryukyoku else 0.0
    return r


def near_reward_from_events(player: int, score_delta: list, events: list) -> float:
    agari = houju = tsumo = riichi = tenpai = False
    for e in events:
        t = e.get("type")
        if t == "hora":
            if e.get("actor") == player:
                agari = True; tsumo = e.get("target") is None
            elif e.get("target") == player:
                houju = True
        elif t == "reach" and e.get("actor") == player:
            riichi = True
        elif t == "ryukyoku":
            tnp = e.get("tenpai") or []
            tenpai = bool(tnp[player]) if player < len(tnp) else False
    return compute_near_reward(KyokuResult(player, int(score_delta[player]), agari, houju, tsumo, riichi, tenpai))


if __name__ == "__main__":
    s = [25000] * 4
    print("東1で +8000 :", round(placement_reward(s, [33000, 25000, 25000, 17000], 0, 8, 7), 3))
    print("東1で 8000放銃:", round(placement_reward(s, [33000, 25000, 25000, 17000], 3, 8, 7), 3))
    b = [30000, 29000, 21000, 20000]
    print("オーラス2着目が 1000点の出和了でトップ :",
          round(placement_reward(b, [29000, 30000 + 0, 21000, 20000], 1, 1, 0), 3))
    print("オーラストップ目が 1000点の横移動を見送る:",
          round(placement_reward(b, [30000, 29000, 22000, 19000], 0, 1, 0), 3))
    print("オーラス ラス目がトップから 1000点和了してもラスのまま終局:",
          round(placement_reward(b, [29000, 29000, 21000, 21000], 3, 1, 0), 3))
