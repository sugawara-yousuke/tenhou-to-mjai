"""
mahjong_tokenizer.py
====================
mjai形式の牌譜を Transformer 用トークン列に変換する。

[今回の改修（入力表現のバグ修正。旧チェックポイントとは非互換 → BC 学び直し必須）]
  1. PAD 衝突の解消
       sparse / progression とも 0 を PAD 専用にし、全トークンを +1 シフト。
       旧版は「手牌の1m」「P0の1m手出し」が token 0 = PAD になりモデルから見えなかった。
  2. 隠れ情報リークの解消
       progression は視点プレイヤーごとに描画する（build_progression）。
       他家のツモ牌は入れない。打牌・リーチの actor は相対席（0=自分,1=下家,2=対面,3=上家）。
  3. 点数を実際に追跡
       start_kyoku.scores を読み、リーチ棒(reach_accepted)も反映。旧版は全局 25000 固定だった。
  4. numeric を相対席順に統一し、点数状況（順位・トップ差・ラス差・親・局進行・オーラス）を追加。
       危険度サマリ(16) と現物特徴(4) も同じ numeric に統合（次元の食い違いを根絶）。
  5. 状態更新を st_* 関数に一本化
       process_events(BC) と self_play(RL/評価) が同じ関数で KyokuState を更新する。
       → 槓で手牌を抜き忘れる / self_play だけ progression が欠ける、等の食い違いが原理的に起きない。
  6. 自分の手番の判断を「1サンプル＝実際の選択肢」に統一
       候補 = 打牌(食い替え禁止牌を除く) ＋ リーチ可能ならリーチ宣言牌。
       ラベル = 実際にリーチしたなら RIICHI+牌、しなければ DAHAI+牌。
       旧版はリーチ判断を混在候補で一度も学習しておらず、self_play では未学習の比較をしていた。
       リーチ後のツモ切りは判断ではないのでサンプル化しない。
  7. 暗槓/加槓の判断サンプル(kan_decision)を追加。旧版は未学習トークンで槓を決めていた。
  8. 鳴き判断の skip ラベルは「確実に見送った」場合だけ出す
       （他家のポン/ロンに優先された場合は意図不明なので捨てる）。
       赤5絡みの _can_pon / _can_chi の取りこぼしも修正。
  9. 候補ごとの特徴 cand_feats（各打牌候補の対3家危険度・現物・ドラ・見え枚数）を追加。
       デコーダのクエリに足すことで「どの牌が危険か」を候補単位で直接与える。
"""

from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
import json
from dataclasses import dataclass, field
from typing import Iterator
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. 牌の定数定義
# ---------------------------------------------------------------------------
TILE_STR_TO_ID: dict[str, int] = {}
_CANON = ['1m','2m','3m','4m','5m','6m','7m','8m','9m',
          '1p','2p','3p','4p','5p','6p','7p','8p','9p',
          '1s','2s','3s','4s','5s','6s','7s','8s','9s',
          '1z','2z','3z','4z','5z','6z','7z']
for _i, _t in enumerate(_CANON):
    TILE_STR_TO_ID[_t] = _i
TILE_STR_TO_ID['0m'] = 34
TILE_STR_TO_ID['0p'] = 35
TILE_STR_TO_ID['0s'] = 36
TILE_STR_TO_ID['5mr'] = 34
TILE_STR_TO_ID['5pr'] = 35
TILE_STR_TO_ID['5sr'] = 36
for _alias, _base in [('E','1z'),('S','2z'),('W','3z'),('N','4z'),
                       ('P','5z'),('F','6z'),('C','7z')]:
    TILE_STR_TO_ID[_alias] = TILE_STR_TO_ID[_base]

# tile_id → 正準文字列（逆引きは必ずこれを使う。dict 反転は別名が後勝ちして壊れる）
TILE_ID_TO_STR: list[str] = _CANON + ['0m', '0p', '0s']

NUM_TILE_TYPES = 37
NUM_PLAYERS    = 4
RED_STRS       = ("0m", "0p", "0s")

# ---------------------------------------------------------------------------
# 2. Sparse トークンID（0 = PAD 専用）
# ---------------------------------------------------------------------------
SPARSE_PAD                = 0
SUTEHAI_MAX_TURN          = 20
SUTEHAI_SLOT_SIZE         = NUM_TILE_TYPES * SUTEHAI_MAX_TURN          # 740

SPARSE_TEHAI_OFFSET       = 1
SPARSE_DORA_OFFSET        = SPARSE_TEHAI_OFFSET    + NUM_TILE_TYPES     # 38
SPARSE_SUTEHAI_SELF       = SPARSE_DORA_OFFSET     + NUM_TILE_TYPES     # 75
SPARSE_SUTEHAI_REL1       = SPARSE_SUTEHAI_SELF    + SUTEHAI_SLOT_SIZE
SPARSE_SUTEHAI_REL2       = SPARSE_SUTEHAI_REL1    + SUTEHAI_SLOT_SIZE
SPARSE_SUTEHAI_REL3       = SPARSE_SUTEHAI_REL2    + SUTEHAI_SLOT_SIZE
MELD_SLOT_SIZE            = NUM_TILE_TYPES * 2
SPARSE_MELD_SELF          = SPARSE_SUTEHAI_REL3    + SUTEHAI_SLOT_SIZE
SPARSE_MELD_REL1          = SPARSE_MELD_SELF       + MELD_SLOT_SIZE
SPARSE_MELD_REL2          = SPARSE_MELD_REL1       + MELD_SLOT_SIZE
SPARSE_MELD_REL3          = SPARSE_MELD_REL2       + MELD_SLOT_SIZE
SPARSE_BAKAZE_OFFSET      = SPARSE_MELD_REL3       + MELD_SLOT_SIZE
SPARSE_KYOKU_OFFSET       = SPARSE_BAKAZE_OFFSET   + 2
SPARSE_HONBA_OFFSET       = SPARSE_KYOKU_OFFSET    + 4
SPARSE_JIKAZE_OFFSET      = SPARSE_HONBA_OFFSET    + 9
SPARSE_VOCAB_SIZE         = SPARSE_JIKAZE_OFFSET   + 4                  # 3350

_MELD_OFFSETS    = [SPARSE_MELD_SELF, SPARSE_MELD_REL1, SPARSE_MELD_REL2, SPARSE_MELD_REL3]
_SUTEHAI_OFFSETS = [SPARSE_SUTEHAI_SELF, SPARSE_SUTEHAI_REL1,
                    SPARSE_SUTEHAI_REL2, SPARSE_SUTEHAI_REL3]

# モデルのクロスアテンションが使う範囲（モデル側はここから import する）
SPARSE_HAND_RANGE  = (SPARSE_TEHAI_OFFSET, SPARSE_DORA_OFFSET)
SPARSE_KAWA_RANGE  = (SPARSE_SUTEHAI_SELF, SPARSE_MELD_SELF)
SPARSE_MELD_RANGE  = (SPARSE_MELD_SELF, SPARSE_BAKAZE_OFFSET)

# ---------------------------------------------------------------------------
# 3. Progression トークンID（0 = PAD 専用）
#    打牌・リーチの player は「視点プレイヤーから見た相対席」。
# ---------------------------------------------------------------------------
PROG_PAD            = 0
PROG_DAHAI_BASE     = 1
PROG_DAHAI_SIZE     = NUM_TILE_TYPES * NUM_PLAYERS * 2 * 2  # 592
PROG_MELD_BASE      = PROG_DAHAI_BASE     + PROG_DAHAI_SIZE
PROG_CHI_BASE       = PROG_MELD_BASE
PROG_PON_BASE       = PROG_CHI_BASE       + 90
PROG_DAIMINKAN_BASE = PROG_PON_BASE       + NUM_TILE_TYPES
PROG_ANKAN_BASE     = PROG_DAIMINKAN_BASE + NUM_TILE_TYPES
PROG_KAKAN_BASE     = PROG_ANKAN_BASE     + 34
PROG_RIICHI_BASE    = PROG_KAKAN_BASE     + NUM_TILE_TYPES
PROG_TSUMO_BASE     = PROG_RIICHI_BASE    + NUM_PLAYERS
PROG_VOCAB_SIZE     = PROG_TSUMO_BASE     + NUM_TILE_TYPES              # 869


def dahai_token(tile_id, rel_player, tsumogiri, riichi):
    return (PROG_DAHAI_BASE + tile_id
            + NUM_TILE_TYPES * rel_player
            + NUM_TILE_TYPES * NUM_PLAYERS * int(tsumogiri)
            + NUM_TILE_TYPES * NUM_PLAYERS * 2 * int(riichi))


def _tid_norm(tid: int) -> int:
    """赤ドラtid(34/35/36)を通常牌tid(4/13/22)へ畳む。"""
    return {34: 4, 35: 13, 36: 22}.get(tid, tid)


def _tid_is_red(tid: int) -> bool:
    return tid in (34, 35, 36)


_CANON_STR = {"5mr": "0m", "5pr": "0p", "5sr": "0s", "E": "1z", "S": "2z", "W": "3z", "N": "4z",
              "P": "5z", "F": "6z", "C": "7z"}


def canon_str(pai: str) -> str:
    """牌文字列を内部表記へ（赤は 0m/0p/0s、字牌は 1z..7z）。牌譜の表記ゆれを入口で吸収する。"""
    return _CANON_STR.get(pai, pai)


def _canon_event(ev: dict) -> dict:
    out = dict(ev)
    for k in ("pai", "dora_marker"):
        if isinstance(out.get(k), str):
            out[k] = canon_str(out[k])
    if isinstance(out.get("consumed"), list):
        out["consumed"] = [canon_str(t) for t in out["consumed"]]
    if isinstance(out.get("tehais"), list):
        out["tehais"] = [[canon_str(t) for t in h] for h in out["tehais"]]
    return out


def norm_str(pai: str) -> str:
    return {"0m": "5m", "0p": "5p", "0s": "5s",
            "5mr": "5m", "5pr": "5p", "5sr": "5s"}.get(pai, pai)


def chi_token(called_tile, consumed):
    ct = _tid_norm(called_tile)
    cons = [_tid_norm(c) for c in consumed]
    suit = ct // 9; number = ct % 9
    min_num = min(number, min(c % 9 for c in cons))
    return PROG_CHI_BASE + suit * 30 + min_num * 3 + (number - min_num)


def pon_token(tile_id):       return PROG_PON_BASE       + min(_tid_norm(tile_id), 33)
def daiminkan_token(tile_id): return PROG_DAIMINKAN_BASE + min(_tid_norm(tile_id), 33)
def ankan_token(tile_id):     return PROG_ANKAN_BASE     + min(_tid_norm(tile_id), 33)
def kakan_token(tile_id):     return PROG_KAKAN_BASE     + min(_tid_norm(tile_id), 33)
def riichi_token(rel_player): return PROG_RIICHI_BASE    + rel_player
def tsumo_token(tile_id):     return PROG_TSUMO_BASE     + min(tile_id, 36)


# ---------------------------------------------------------------------------
# 4. Action トークンID
# ---------------------------------------------------------------------------
ACT_DAHAI_BASE     = 0
ACT_RIICHI_BASE    = ACT_DAHAI_BASE     + NUM_TILE_TYPES   # 37
ACT_PON_BASE       = ACT_RIICHI_BASE    + NUM_TILE_TYPES   # 74
ACT_CHI_BASE       = ACT_PON_BASE       + NUM_TILE_TYPES   # 111
ACT_DAIMINKAN_BASE = ACT_CHI_BASE       + 90               # 201
ACT_ANKAN_BASE     = ACT_DAIMINKAN_BASE + NUM_TILE_TYPES   # 238
ACT_KAKAN_BASE     = ACT_ANKAN_BASE     + 34               # 272
ACT_TSUMO_BASE     = ACT_KAKAN_BASE     + NUM_TILE_TYPES   # 309
ACT_RON_BASE       = ACT_TSUMO_BASE     + 1                # 310
ACT_PASS_BASE      = ACT_RON_BASE       + 1                # 311
NUM_ACTION_TYPES   = ACT_PASS_BASE      + 1                # 312

MAX_ACTION_CANDIDATES = 64
PADDING_ACTION_ID     = NUM_ACTION_TYPES  # 312

MELD_LABEL_SKIP      = 0
MELD_LABEL_PON       = 1
MELD_LABEL_CHI       = 2
MELD_LABEL_DAIMINKAN = 3
NUM_MELD_LABELS      = 4


def act_chi_token(called_tile, consumed):
    """チー行動トークン = ACT_CHI_BASE + suit*14 + (start-1)*2 + red_flag"""
    ids = [_tid_norm(called_tile)] + [_tid_norm(c) for c in consumed]
    suit = ids[0] // 9
    nums = sorted(i % 9 for i in ids)
    start = nums[0] + 1
    has_red = _tid_is_red(called_tile) or any(_tid_is_red(c) for c in consumed)
    return ACT_CHI_BASE + suit * 14 + (start - 1) * 2 + (1 if has_red else 0)


def decode_act_chi(token):
    off = token - ACT_CHI_BASE
    if off < 0 or off >= 42:
        return None
    suit = off // 14
    rem = off % 14
    return suit, rem // 2 + 1, bool(rem % 2)


def act_pon_token(tile_id, has_red=False):
    nt = _tid_norm(tile_id)
    if has_red:
        return ACT_PON_BASE + {4: 34, 13: 35, 22: 36}.get(nt, min(nt, 33))
    return ACT_PON_BASE + min(nt, 33)


def decode_act_pon(token):
    off = token - ACT_PON_BASE
    if off in (34, 35, 36):
        return {34: 4, 35: 13, 36: 22}[off], True
    return off, False


# ---------------------------------------------------------------------------
# 5. numeric / 候補特徴 の次元
# ---------------------------------------------------------------------------
from danger_features import (DANGER_DIM, build_danger_context, build_danger_vector)

NUMERIC_BASE_DIM      = 19
NUMERIC_SITUATION_DIM = 7
NUMERIC_GENBUTSU_DIM  = 4
NUMERIC_DIM = NUMERIC_BASE_DIM + NUMERIC_SITUATION_DIM + DANGER_DIM + NUMERIC_GENBUTSU_DIM  # 46
USE_DANGER_FEATURES = True   # 後方互換のための名前だけ残す（常に True）

# 候補ごとの特徴: [危険度 対下家/対面/上家, 現物(リーチ者に対して)×3, ドラ価値, 見え枚数]
CAND_FEAT_DIM = 8


# ---------------------------------------------------------------------------
# 6. 向聴数計算
# ---------------------------------------------------------------------------
def _normalize_hand(hand_ids: list[int]) -> list[int]:
    norm = {34: 4, 35: 13, 36: 22}
    return [norm.get(t, t) for t in hand_ids]


from functools import lru_cache

_NEG = (-1, -1, -1, -1, -1)


@lru_cache(maxsize=1 << 18)
def _suit_table(c: tuple, honor: bool = False) -> tuple:
    """
    1色ぶん(9枚種 or 字牌7枚種)の枚数タプル → (頭なし, 頭あり) の2表。
    各表は t[m] = 「面子をちょうど m 個取ったときの搭子(対子含む)の最大数」、不可能なら -1。
    色ごとに独立に求めてメモ化する（同じ並びは何度も出てくるので非常によく当たる）。
    """
    n = len(c)
    i = next((j for j in range(n) if c[j]), -1)
    if i < 0:
        return ((0, -1, -1, -1, -1), _NEG)
    nh = [-1] * 5; wh = [-1] * 5

    def dec(*idx):
        l = list(c)
        for j in idx:
            l[j] -= 1
        return tuple(l)

    def merge(res, dm, dt):
        ra, rb = res
        for m in range(5 - dm):
            if ra[m] >= 0 and ra[m] + dt > nh[m + dm]: nh[m + dm] = ra[m] + dt
            if rb[m] >= 0 and rb[m] + dt > wh[m + dm]: wh[m + dm] = rb[m] + dt

    merge(_suit_table(dec(i), honor), 0, 0)                          # 浮き牌として捨てる
    if c[i] >= 3: merge(_suit_table(dec(i, i, i), honor), 1, 0)       # 刻子
    if c[i] >= 2:
        sub = _suit_table(dec(i, i), honor)
        merge(sub, 0, 1)                                              # 対子(搭子扱い)
        for m in range(5):                                            # 雀頭
            if sub[0][m] > wh[m]: wh[m] = sub[0][m]
    if not honor:
        if i <= 6 and c[i + 1] and c[i + 2]: merge(_suit_table(dec(i, i + 1, i + 2)), 1, 0)   # 順子
        if i <= 7 and c[i + 1]:              merge(_suit_table(dec(i, i + 1)), 0, 1)          # 両面/辺張
        if i <= 6 and c[i + 2]:              merge(_suit_table(dec(i, i + 2)), 0, 1)          # 嵌張
    return (tuple(nh), tuple(wh))


def _conv(a: tuple, b: tuple) -> tuple:
    out = [-1] * 5
    for i in range(5):
        if a[i] < 0: continue
        for j in range(5 - i):
            if b[j] >= 0 and a[i] + b[j] > out[i + j]:
                out[i + j] = a[i] + b[j]
    return tuple(out)


def _shanten_regular(counts: list[int], num_melds: int = 0) -> int:
    """
    一般形の向聴数。8 − 2×面子 − 搭子 − 雀頭、ただし 面子＋搭子 ≤ 4。
    [修正] 旧実装は搭子を先に数えてから面子を足すと 面子＋搭子 が 5 になり得て、
    「3面子＋2搭子・雀頭なし」(本当は1向聴)をテンパイと誤判定していた。
    リーチ可否の判定にも使うので、ノーテンリーチの候補が出る原因になっていた。
    """
    A = _suit_table(tuple(counts[0:9]));   B = _suit_table(tuple(counts[9:18]))
    C = _suit_table(tuple(counts[18:27])); H = _suit_table(tuple(counts[27:34]), True)
    ab = _conv(A[0], B[0]); ch = _conv(C[0], H[0])
    best = 8 - 2 * num_melds

    def scan(tab, head):
        nonlocal best
        for m in range(5 - num_melds):
            t = tab[m]
            if t < 0: continue
            mm = m + num_melds
            v = 8 - 2 * mm - min(t, 4 - mm) - head
            if v < best: best = v

    scan(_conv(ab, ch), 0)
    scan(_conv(_conv(A[1], B[0]), ch), 1); scan(_conv(_conv(A[0], B[1]), ch), 1)
    scan(_conv(ab, _conv(C[1], H[0])), 1); scan(_conv(ab, _conv(C[0], H[1])), 1)
    return best


def _shanten_chiitoi(counts: list[int]) -> int:
    pairs = sum(1 for c in counts if c >= 2)
    kinds = sum(1 for c in counts if c > 0)
    return 6 - pairs + max(0, 7 - kinds)


def _shanten_kokushi(counts: list[int]) -> int:
    yaochuuhai = [0,8,9,17,18,26,27,28,29,30,31,32,33]
    has = sum(1 for i in yaochuuhai if counts[i] > 0)
    has_pair = any(counts[i] >= 2 for i in yaochuuhai)
    return 13 - has - (1 if has_pair else 0)


@lru_cache(maxsize=1 << 17)
def _shanten_cached(counts: tuple, num_melds: int) -> int:
    c = list(counts)
    s_regular = _shanten_regular(c, num_melds)
    if num_melds > 0:
        return s_regular
    return min(s_regular, _shanten_chiitoi(c), _shanten_kokushi(c))


def shanten_from_counts(counts, num_melds: int = 0) -> int:
    """34種の枚数から直接（受け入れ計算など、枚数表を持っている呼び出し側用）。"""
    return _shanten_cached(tuple(counts), num_melds)


def shanten_number(hand_ids: list[int], num_melds: int = 0) -> int:
    """0 = テンパイ、-1 = 和了、1以上 = n向聴。num_melds は副露(暗槓含む)面子数。
    同じ手牌は numeric・リーチ判定・プレイアウトで何度も評価されるのでキャッシュする。"""
    counts = [0] * 34
    for t in hand_ids:
        t = _tid_norm(t)
        if 0 <= t <= 33:
            counts[t] += 1
    return _shanten_cached(tuple(counts), num_melds)


# ---------------------------------------------------------------------------
# 7. 局面状態
# ---------------------------------------------------------------------------
@dataclass
class KyokuState:
    bakaze:      str   = "E"
    kyoku:       int   = 1
    honba:       int   = 0
    kyotaku:     int   = 0
    oya:         int   = 0
    dahai_count: int   = 0
    dora_markers: list = field(default_factory=list)
    tehais:      list  = field(default_factory=lambda: [[], [], [], []])
    scores:      list  = field(default_factory=lambda: [25000]*4)
    riichi_flags: list = field(default_factory=lambda: [False]*4)
    # 進行履歴（視点非依存の生イベント）。トークン化は build_progression(state, player)。
    prog_events: list  = field(default_factory=list)
    last_dahai_player: int = -1
    last_dahai_tile:   int = -1
    last_dahai_str:    str = ""
    pending_call_candidates: dict = field(default_factory=dict)
    wall_count:  int   = 70
    meld_counts: list  = field(default_factory=lambda: [0,0,0,0])   # 暗槓含む面子数
    ankan_counts: list = field(default_factory=lambda: [0,0,0,0])
    pending_riichi_actor: int = -1
    sutehais: list = field(default_factory=lambda: [[], [], [], []])
    riichi_sutehai_idx: list = field(default_factory=lambda: [-1, -1, -1, -1])
    # sparse 用の副露表現 [(tile_id, meld_type)]  meld_type: 0=ポン/槓, 1=チー
    melds: list = field(default_factory=lambda: [[], [], [], []])
    # 副露の詳細 [{"type","pai","consumed",("added")}]（役・符・食い替え・加槓判定用）
    meld_info: list = field(default_factory=lambda: [[], [], [], []])
    # opp に対して絶対安全な牌種34（opp の河 ＋ opp のリーチ後に通った牌）
    safe_tiles: list = field(default_factory=lambda: [set(), set(), set(), set()])
    # 直前に鳴いた内容（食い替え判定用。次の打牌で消える）
    last_call: list = field(default_factory=lambda: [None, None, None, None])
    # その手番でツモった牌（鳴き後は None）
    last_tsumo: list = field(default_factory=lambda: [None, None, None, None])

    def is_menzen(self, p: int) -> bool:
        return self.meld_counts[p] - self.ankan_counts[p] == 0


def _remove_from_hand(hand: list, pai: str) -> bool:
    """手牌から1枚除去。赤5と通常5のゆらぎを吸収する。"""
    if pai in hand:
        hand.remove(pai); return True
    n = norm_str(pai)
    for alt in (n, {"5m": "0m", "5p": "0p", "5s": "0s"}.get(n)):
        if alt and alt in hand:
            hand.remove(alt); return True
    return False


# ---- 状態更新（BC の process_events と self_play の両方がこれだけを使う） ----

def st_tsumo(state: KyokuState, actor: int, pai: str):
    state.tehais[actor].append(pai)
    state.last_tsumo[actor] = pai
    state.prog_events.append(("tsumo", actor, TILE_STR_TO_ID.get(pai, 0)))
    state.wall_count = max(state.wall_count - 1, 0)


def st_riichi(state: KyokuState, actor: int):
    """リーチ宣言（宣言牌の打牌の直前に呼ぶ）。"""
    state.riichi_flags[actor] = True
    state.prog_events.append(("riichi", actor))
    state.riichi_sutehai_idx[actor] = len(state.sutehais[actor])
    state.pending_riichi_actor = -1


def st_riichi_accepted(state: KyokuState, actor: int):
    state.scores[actor] -= 1000
    state.kyotaku += 1


def st_dahai(state: KyokuState, actor: int, pai: str, tsumogiri: bool):
    _remove_from_hand(state.tehais[actor], pai)
    tid = TILE_STR_TO_ID.get(pai, 0)
    state.prog_events.append(("dahai", actor, tid, bool(tsumogiri),
                              bool(state.riichi_flags[actor])))
    state.sutehais[actor].append(tid)
    k34 = _tid_norm(tid)
    state.safe_tiles[actor].add(k34)
    for q in range(NUM_PLAYERS):
        if q != actor and state.riichi_flags[q]:
            state.safe_tiles[q].add(k34)    # リーチ者が見逃した牌は以後その人に通る
    state.last_dahai_player = actor
    state.last_dahai_tile   = tid
    state.last_dahai_str    = pai
    state.dahai_count += 1
    state.last_call[actor]  = None
    state.last_tsumo[actor] = None


def _st_open_meld(state, actor, mtype, pai, consumed, sparse_type):
    for c in consumed:
        _remove_from_hand(state.tehais[actor], c)
    state.meld_counts[actor] += 1
    tid = TILE_STR_TO_ID.get(pai, 0)
    state.melds[actor].append((tid, sparse_type))
    state.meld_info[actor].append({"type": mtype, "pai": pai, "consumed": list(consumed)})
    state.last_tsumo[actor] = None


def st_chi(state: KyokuState, actor: int, pai: str, consumed: list):
    _st_open_meld(state, actor, "chi", pai, consumed, 1)
    cons_ids = [TILE_STR_TO_ID.get(c, 0) for c in consumed]
    state.prog_events.append(("chi", actor, TILE_STR_TO_ID.get(pai, 0), tuple(cons_ids)))
    state.last_call[actor] = ("chi", TILE_STR_TO_ID.get(pai, 0), tuple(cons_ids))


def st_pon(state: KyokuState, actor: int, pai: str, consumed: list):
    _st_open_meld(state, actor, "pon", pai, consumed, 0)
    state.prog_events.append(("pon", actor, TILE_STR_TO_ID.get(pai, 0)))
    state.last_call[actor] = ("pon", TILE_STR_TO_ID.get(pai, 0), ())


def st_daiminkan(state: KyokuState, actor: int, pai: str, consumed: list):
    _st_open_meld(state, actor, "daiminkan", pai, consumed, 0)
    state.prog_events.append(("daiminkan", actor, TILE_STR_TO_ID.get(pai, 0)))


def st_ankan(state: KyokuState, actor: int, consumed: list):
    for c in consumed:
        _remove_from_hand(state.tehais[actor], c)
    state.meld_counts[actor] += 1
    state.ankan_counts[actor] += 1
    tid = TILE_STR_TO_ID.get(consumed[0], 0) if consumed else 0
    state.melds[actor].append((_tid_norm(tid), 0))
    state.meld_info[actor].append({"type": "ankan", "pai": None, "consumed": list(consumed)})
    state.prog_events.append(("ankan", actor, tid))
    state.last_tsumo[actor] = None


def st_kakan(state: KyokuState, actor: int, pai: str):
    _remove_from_hand(state.tehais[actor], pai)
    n = norm_str(pai)
    for m in state.meld_info[actor]:
        if m["type"] == "pon" and norm_str(m["pai"]) == n:
            m["type"] = "kakan"; m["added"] = pai
            break
    state.prog_events.append(("kakan", actor, TILE_STR_TO_ID.get(pai, 0)))
    state.last_tsumo[actor] = None


def st_dora(state: KyokuState, marker: str):
    state.dora_markers.append(marker)


def dora_kind_of(marker_tid: int) -> int:
    """ドラ表示牌(tile_id) → ドラの牌種34。"""
    k = _tid_norm(marker_tid)
    if k < 27:
        return (k // 9) * 9 + (k % 9 + 1) % 9
    if k <= 30:
        return 27 + (k - 27 + 1) % 4
    return 31 + (k - 31 + 1) % 3


def kuikae_forbidden(state: KyokuState, actor: int) -> set[int]:
    """直前の鳴きによる食い替え禁止牌（牌種34）。現物食い替え＋スジ食い替え。"""
    lc = state.last_call[actor]
    if not lc:
        return set()
    mtype, called, cons = lc
    c = _tid_norm(called)
    forb = {c}
    if mtype == "chi" and len(cons) == 2:
        nums = sorted(_tid_norm(x) % 9 for x in cons)
        n = c % 9; base = (c // 9) * 9
        if n < nums[0] and n + 3 <= 8:       # 下端を鳴いた (n, n+1, n+2) → n+3 も禁止
            forb.add(base + n + 3)
        elif n > nums[1] and n - 3 >= 0:     # 上端を鳴いた → n-3 も禁止
            forb.add(base + n - 3)
    return forb


# ---------------------------------------------------------------------------
# 8. トークナイザー
# ---------------------------------------------------------------------------
class MahjongTokenizer:

    def __init__(self):
        self._emit_aux_fields = True
        self._cur_event_index = -1

    def tile_id(self, pai: str) -> int:
        return TILE_STR_TO_ID.get(pai, 0)

    # ---- 鳴き可否（赤5を正規化して数える） ----
    def _can_pon(self, hand, pai):
        n = norm_str(pai)
        return sum(1 for t in hand if norm_str(t) == n) >= 2

    def _can_daiminkan(self, hand, pai):
        n = norm_str(pai)
        return sum(1 for t in hand if norm_str(t) == n) >= 3

    def _can_chi(self, hand, pai, discard_player, my_player):
        if (my_player - discard_player) % 4 != 1:
            return False
        tid = _tid_norm(self.tile_id(pai))
        if tid >= 27:
            return False
        suit = tid // 9; num = tid % 9
        hand_nums = set()
        for t in hand:
            hid = _tid_norm(self.tile_id(t))
            if hid < 27 and hid // 9 == suit:
                hand_nums.add(hid % 9)
        for a_off, b_off in [(-2,-1), (-1,1), (1,2)]:
            a, b = num + a_off, num + b_off
            if 0 <= a <= 8 and 0 <= b <= 8 and a in hand_nums and b in hand_nums:
                return True
        return False

    def call_options(self, state: KyokuState, player: int, pai: str, discard_player: int) -> dict:
        """player が pai を鳴けるか。リーチ者・河底牌は不可。"""
        if player == discard_player or state.riichi_flags[player] or state.wall_count <= 0:
            return {"pon": False, "chi": False, "daiminkan": False}
        hand = state.tehais[player]
        return {"pon": self._can_pon(hand, pai),
                "chi": self._can_chi(hand, pai, discard_player, player),
                "daiminkan": self._can_daiminkan(hand, pai)}

    def _shanten_of(self, hand: list[str], num_melds: int = 0) -> int:
        hand_ids = [self.tile_id(t) for t in hand]
        if not hand_ids:
            return max(8 - 2 * num_melds, 0)
        return shanten_number(hand_ids, num_melds)

    # ---- 入力特徴 ----
    def build_sparse(self, state: KyokuState, player: int) -> list[int]:
        tokens = [
            SPARSE_BAKAZE_OFFSET + (0 if state.bakaze == "E" else 1),
            SPARSE_KYOKU_OFFSET  + min(max(state.kyoku - 1, 0), 3),
            SPARSE_HONBA_OFFSET  + min(state.honba, 8),
            SPARSE_JIKAZE_OFFSET + (player - state.oya) % 4,
        ]
        for dm in state.dora_markers:
            tokens.append(SPARSE_DORA_OFFSET + min(self.tile_id(dm), NUM_TILE_TYPES - 1))
        for t in state.tehais[player]:
            tokens.append(SPARSE_TEHAI_OFFSET + min(self.tile_id(t), NUM_TILE_TYPES - 1))
        for rel in range(NUM_PLAYERS):
            pi = (player + rel) % NUM_PLAYERS
            offset = _SUTEHAI_OFFSETS[rel]
            for turn, tid in enumerate(state.sutehais[pi]):
                t = min(turn, SUTEHAI_MAX_TURN - 1)
                tokens.append(offset + t * NUM_TILE_TYPES + min(tid, NUM_TILE_TYPES - 1))
        for rel in range(NUM_PLAYERS):
            pi = (player + rel) % NUM_PLAYERS
            offset = _MELD_OFFSETS[rel]
            for tid, meld_type in state.melds[pi]:
                tokens.append(offset + meld_type * NUM_TILE_TYPES + min(tid, NUM_TILE_TYPES - 1))
        return tokens

    def build_progression(self, state: KyokuState, player: int) -> list[int]:
        """player 視点の進行トークン列。他家のツモ牌（隠れ情報）は含めない。"""
        out = []
        for ev in state.prog_events:
            kind, actor = ev[0], ev[1]
            rel = (actor - player) % NUM_PLAYERS
            if kind == "tsumo":
                if actor == player:
                    out.append(tsumo_token(ev[2]))
            elif kind == "dahai":
                out.append(dahai_token(ev[2], rel, ev[3], ev[4]))
            elif kind == "riichi":
                out.append(riichi_token(rel))
            elif kind == "chi":
                out.append(chi_token(ev[2], list(ev[3])))
            elif kind == "pon":
                out.append(pon_token(ev[2]))
            elif kind == "daiminkan":
                out.append(daiminkan_token(ev[2]))
            elif kind == "ankan":
                out.append(ankan_token(ev[2]))
            elif kind == "kakan":
                out.append(kakan_token(ev[2]))
        return out

    @staticmethod
    def _rank_of(scores: list, p: int) -> int:
        """順位 0..3。同点は起家に近い席(絶対index小)が上位。"""
        return sum(1 for q in range(NUM_PLAYERS)
                   if q != p and (scores[q] > scores[p] or (scores[q] == scores[p] and q < p)))

    def build_numeric(self, state: KyokuState, player: int, ctx: dict | None = None) -> list[float]:
        """
        [0-3]   点数 (自分, 下家, 対面, 上家) / 25000
        [4]     供託 / 4
        [5]     自分リーチ
        [6-8]   他家リーチ (下家, 対面, 上家)
        [9]     残り山 / 70
        [10]    自分の副露数 / 4        [11-13] 他家の副露数 / 4
        [14]    自分の向聴数 / 8
        [15]    自分の捨て牌数 / 30     [16-18] 他家の捨て牌数 / 30
        [19]    自分の順位 / 3
        [20]    トップとの点差 / 25000 (0..2)   [21] ラスとの点差 / 25000 (0..2)
        [22]    直上の相手との点差 / 25000      [23] 直下の相手との点差 / 25000
        [24]    親か
        [25]    局進行 (東1=0 … 南4=1)  ※オーラスは 1.0
        [26-41] 危険度サマリ (danger_features)
        [42-45] 現物特徴: 現物あり / 現物種率 / 被リーチ / リーチ者数
        """
        order = [(player + r) % NUM_PLAYERS for r in range(NUM_PLAYERS)]
        sc = state.scores
        n = [sc[p] / 25000.0 for p in order]
        n.append(min(max(state.kyotaku, 0), 4) / 4.0)
        n.append(1.0 if state.riichi_flags[player] else 0.0)
        n.extend(1.0 if state.riichi_flags[p] else 0.0 for p in order[1:])
        n.append(max(state.wall_count, 0) / 70.0)
        n.append(min(state.meld_counts[player], 4) / 4.0)
        n.extend(min(state.meld_counts[p], 4) / 4.0 for p in order[1:])
        if state.riichi_flags[player]:
            n.append(0.0)
        else:
            sht = self._shanten_of(state.tehais[player], state.meld_counts[player])
            n.append(min(max(sht, 0), 8) / 8.0)
        n.append(min(len(state.sutehais[player]), 30) / 30.0)
        n.extend(min(len(state.sutehais[p]), 30) / 30.0 for p in order[1:])

        # ---- 点数状況 ----
        rank = self._rank_of(sc, player)
        others = [sc[p] for p in order[1:]]
        above = [s for s in others if s > sc[player]]
        below = [s for s in others if s < sc[player]]
        clip = lambda x: max(0.0, min(2.0, x / 25000.0))
        n.append(rank / 3.0)
        n.append(clip(max(sc) - sc[player]))
        n.append(clip(sc[player] - min(sc)))
        n.append(clip(min(above) - sc[player]) if above else 0.0)
        n.append(clip(sc[player] - max(below)) if below else 0.0)
        n.append(1.0 if player == state.oya else 0.0)
        bk = {"E": 0, "S": 1, "W": 2, "N": 3}.get(state.bakaze, 0)
        n.append(min(bk * 4 + max(state.kyoku - 1, 0), 7) / 7.0)

        # ---- 危険度サマリ ----
        if ctx is None:
            ctx = build_danger_context(state, player)
        n.extend(build_danger_vector(state, player, ctx))

        # ---- 現物特徴（リーチ者全員に共通して通る牌） ----
        riichi_rel = [r for r in range(3) if ctx["riichi"][r]]
        under = bool(riichi_rel) and not state.riichi_flags[player]
        hand_kinds = {_tid_norm(self.tile_id(t)) for t in state.tehais[player]}
        if riichi_rel:
            common = set(ctx["genb"][riichi_rel[0]])
            for r in riichi_rel[1:]:
                common &= ctx["genb"][r]
            gb = hand_kinds & common
        else:
            gb = set()
        n.append(1.0 if gb else 0.0)
        n.append(len(gb) / max(len(hand_kinds), 1))
        n.append(1.0 if under else 0.0)
        n.append(len(riichi_rel) / 3.0)
        assert len(n) == NUMERIC_DIM, f"numeric dim {len(n)} != {NUMERIC_DIM}"
        return n

    def build_cand_feats(self, state: KyokuState, player: int, acts: list[int],
                         ctx: dict | None = None) -> list[list[float]]:
        """
        候補ごとの特徴 (MAX_ACTION_CANDIDATES x CAND_FEAT_DIM)。
        打牌/リーチ宣言の候補にだけ値が入り、鳴き・槓・PASS・PAD は 0。
          [0-2] その牌を切ったときの対 下家/対面/上家 危険度
          [3-5] その牌がリーチ中の 下家/対面/上家 の現物か
          [6]   ドラ価値 / 4（表ドラ枚数。手牌のその牌種が赤しか無ければ +1）
          [7]   見えている枚数 / 4
        """
        if ctx is None:
            ctx = build_danger_context(state, player)
        dora_cnt = [0] * 34
        for dm in state.dora_markers:
            dora_cnt[dora_kind_of(self.tile_id(dm))] += 1
        hand = state.tehais[player]
        feats = []
        zero = [0.0] * CAND_FEAT_DIM
        for a in acts[:MAX_ACTION_CANDIDATES]:
            if 0 <= a < ACT_PON_BASE:            # DAHAI or RIICHI
                k = _tid_norm(a % NUM_TILE_TYPES)
                if k > 33:
                    feats.append(zero); continue
                copies = [t for t in hand if _tid_norm(self.tile_id(t)) == k]
                only_red = bool(copies) and all(t in RED_STRS for t in copies)
                f = [ctx["matrix"][r][k] for r in range(3)]
                f += [1.0 if (ctx["riichi"][r] and k in ctx["genb"][r]) else 0.0 for r in range(3)]
                f.append(min(dora_cnt[k] + (1 if only_red else 0), 4) / 4.0)
                f.append(ctx["visible"][k] / 4.0)
                feats.append(f)
            else:
                feats.append(zero)
        while len(feats) < MAX_ACTION_CANDIDATES:
            feats.append(zero)
        return feats

    def make_sample(self, state: KyokuState, player: int, acts: list[int],
                    label_idx: int, event_type: str) -> dict:
        """BC・self_play 共通のサンプル生成（特徴量の作り方はここ1か所だけ）。"""
        acts = list(acts)[:MAX_ACTION_CANDIDATES]
        ctx = build_danger_context(state, player)
        padded = acts + [PADDING_ACTION_ID] * (MAX_ACTION_CANDIDATES - len(acts))
        return {
            "player":           player,
            "sparse":           self.build_sparse(state, player),
            "numeric":          self.build_numeric(state, player, ctx),
            "progression":      self.build_progression(state, player),
            "possible_actions": padded,
            "cand_feats":       self.build_cand_feats(state, player, acts, ctx),
            "action_label":     label_idx,
            "event_type":       event_type,
        }

    # ---- 候補列挙（BC・self_play 共通） ----
    def riichi_discards(self, state: KyokuState, player: int) -> list[int]:
        """リーチ宣言できる打牌（牌種34）。不可なら空。"""
        if (state.riichi_flags[player] or not state.is_menzen(player)
                or state.scores[player] < 1000 or state.wall_count < 4):
            return []
        nm = state.meld_counts[player]
        ids = [self.tile_id(t) for t in state.tehais[player]]
        if len(ids) % 3 != 2 or shanten_number(ids, nm) > 0:
            return []
        out, seen = [], set()
        for i, t in enumerate(ids):
            k = _tid_norm(t)
            if k in seen:
                continue
            seen.add(k)
            if shanten_number(ids[:i] + ids[i+1:], nm) == 0:
                out.append(k)
        return out

    def enumerate_dahai_actions(self, state: KyokuState, player: int) -> list[int]:
        """
        自分の手番の選択肢: 打牌(食い替え禁止牌を除く) ＋ リーチ宣言牌。
        リーチ中は呼ばないこと（ツモ切り固定で判断が無い）。
        """
        forb = kuikae_forbidden(state, player)
        kinds, seen = [], set()
        for t in state.tehais[player]:
            k = _tid_norm(self.tile_id(t))
            if k not in seen:
                seen.add(k); kinds.append(k)
        legal = [k for k in kinds if k not in forb] or kinds
        acts = [ACT_DAHAI_BASE + k for k in legal]
        acts += [ACT_RIICHI_BASE + k for k in self.riichi_discards(state, player)]
        return acts

    def enumerate_kan_candidates(self, state: KyokuState, player: int) -> list:
        """ツモ後の暗槓/加槓。[(token, type, tile_str)]、先頭は PASS。リーチ中・山切れは不可。"""
        cands = [(ACT_PASS_BASE, "pass", None)]
        if state.riichi_flags[player] or state.wall_count <= 0 or state.last_tsumo[player] is None:
            return cands
        hand = state.tehais[player]
        cnt: dict[str, int] = {}
        for t in hand:
            cnt[norm_str(t)] = cnt.get(norm_str(t), 0) + 1
        for n, c in cnt.items():
            if c == 4:
                cands.append((ACT_ANKAN_BASE + min(self.tile_id(n), 33), "ankan", n))
        for m in state.meld_info[player]:
            if m["type"] == "pon":
                n = norm_str(m["pai"])
                if cnt.get(n, 0) >= 1:
                    cands.append((ACT_KAKAN_BASE + min(self.tile_id(n), 33), "kakan", n))
        return cands

    def enumerate_meld_candidates(self, hand, pai, can_pon, can_chi, can_daiminkan,
                                   discard_player=None, my_player=None):
        """[(token, mtype, consumed_list), ...]  先頭は常に skip。"""
        cands = [(ACT_PASS_BASE, "skip", [])]
        pai_norm = norm_str(pai)
        pai_tid = self.tile_id(pai)
        pai_tid_norm = _tid_norm(pai_tid)
        same = [t for t in hand if norm_str(t) == pai_norm]
        normals = [t for t in same if t not in RED_STRS]
        reds    = [t for t in same if t in RED_STRS]

        if can_pon:
            if len(normals) >= 2:
                cands.append((act_pon_token(pai_tid_norm, has_red=False), "pon", normals[:2]))
            if reds and normals:
                cands.append((act_pon_token(pai_tid_norm, has_red=True), "pon", [reds[0], normals[0]]))
            if not any(c[1] == "pon" for c in cands) and len(same) >= 2:
                cands.append((act_pon_token(pai_tid_norm, has_red=any(x in RED_STRS for x in same[:2])),
                              "pon", same[:2]))

        if can_chi and pai_tid_norm < 27:
            suit = pai_tid_norm // 9; num = pai_tid_norm % 9
            by_num: dict[int, list] = {}
            for t in hand:
                hn = _tid_norm(self.tile_id(t))
                if hn < 27 and hn // 9 == suit:
                    by_num.setdefault(hn % 9, []).append(t)

            def pick(tiles, prefer_red):
                r = [x for x in tiles if x in RED_STRS]
                nn = [x for x in tiles if x not in RED_STRS]
                if prefer_red and r: return r[0]
                return nn[0] if nn else tiles[0]

            for a_off, b_off in [(-2, -1), (-1, 1), (1, 2)]:
                a, b = num + a_off, num + b_off
                if not (0 <= a <= 8 and 0 <= b <= 8 and a in by_num and b in by_num):
                    continue
                for prefer_red in (False, True):
                    ca, cb = pick(by_num[a], prefer_red and a == 4), pick(by_num[b], prefer_red and b == 4)
                    tok = act_chi_token(pai_tid, [self.tile_id(ca), self.tile_id(cb)])
                    if not any(c[0] == tok for c in cands):
                        cands.append((tok, "chi", [ca, cb]))

        if can_daiminkan and len(same) >= 3:
            cands.append((ACT_DAIMINKAN_BASE + min(pai_tid_norm, 33), "daiminkan", same[:3]))
        return cands

    def _meld_label_index(self, cands, taken_mtype, taken_consumed):
        if taken_mtype == "skip":
            return 0
        taken_norm = sorted(norm_str(c) for c in (taken_consumed or []))
        taken_red = any(c in RED_STRS for c in (taken_consumed or []))
        for idx, (_tok, mtype, consumed) in enumerate(cands):
            if mtype == taken_mtype and sorted(norm_str(c) for c in consumed) == taken_norm \
                    and any(c in RED_STRS for c in consumed) == taken_red:
                return idx
        for idx, (_tok, mtype, consumed) in enumerate(cands):
            if mtype == taken_mtype and sorted(norm_str(c) for c in consumed) == taken_norm:
                return idx
        return None

    def build_meld_decision_sample(self, state, player, pai, opts,
                                   taken_mtype="skip", taken_consumed=None):
        if not (opts.get("pon") or opts.get("chi") or opts.get("daiminkan")):
            return None
        cands = self.enumerate_meld_candidates(
            state.tehais[player], pai, opts.get("pon"), opts.get("chi"), opts.get("daiminkan"))
        if len(cands) <= 1:
            return None
        lbl = self._meld_label_index(cands, taken_mtype, taken_consumed)
        if lbl is None:
            return None
        s = self.make_sample(state, player, [c[0] for c in cands], lbl, "meld_decision")
        s["meld_label"] = {"skip": MELD_LABEL_SKIP, "pon": MELD_LABEL_PON, "chi": MELD_LABEL_CHI,
                           "daiminkan": MELD_LABEL_DAIMINKAN}[taken_mtype]
        s["called_tile_id"] = self.tile_id(pai)
        return s

    # ---- メイン処理 ----
    def _flush_pending(self, state, mode: str, caller: int | None = None):
        """
        保留中の鳴き候補を処理する。
          mode="all_skipped" : 次のツモが来た＝全員が確実に見送った → skip サンプル
          mode="chi_by"      : caller がチー。他家(ポン/槓可)は優先権があるのに鳴かなかった → skip 確定
          mode="drop"        : ポン/槓/ロン/流局。他家の意図は牌譜から分からない → 捨てる
        """
        if mode != "drop":
            for p, opts in state.pending_call_candidates.items():
                if p == caller:
                    continue
                s = self.build_meld_decision_sample(state, p, state.last_dahai_str, opts, "skip", [])
                if s:
                    yield s
        state.pending_call_candidates = {}

    def process_events(self, events: list[dict]) -> Iterator[dict]:
        state = KyokuState()
        game_scores = [25000] * 4

        for _ev_i, event in enumerate(events):
            etype = event.get("type", "")
            self._cur_event_index = _ev_i
            if etype in ("start_kyoku", "tsumo", "dahai", "chi", "pon", "daiminkan", "ankan", "kakan", "dora"):
                event = _canon_event(event)

            if etype == "start_kyoku":
                state = KyokuState()
                state.bakaze       = event.get("bakaze", "E")
                state.kyoku        = event.get("kyoku", 1)
                state.honba        = event.get("honba", 0)
                state.kyotaku      = event.get("kyotaku", 0)
                state.oya          = event.get("oya", 0)
                state.dora_markers = [event.get("dora_marker", "1m")]
                state.tehais       = [list(h) for h in event.get("tehais", [[], [], [], []])]
                sc = event.get("scores")
                state.scores = list(sc) if isinstance(sc, list) and len(sc) == 4 else list(game_scores)

            elif etype == "tsumo":
                yield from self._flush_pending(state, "all_skipped")
                st_tsumo(state, event["actor"], event["pai"])

            elif etype in ("riichi", "reach"):
                state.pending_riichi_actor = event["actor"]

            elif etype == "reach_accepted":
                a = event.get("actor")
                if a is not None:
                    st_riichi_accepted(state, a)

            elif etype == "dahai":
                actor = event["actor"]; pai = event["pai"]
                tg = bool(event.get("tsumogiri", False))
                declaring = (state.pending_riichi_actor == actor)

                if not state.riichi_flags[actor]:
                    # 槓できたのにしなかった
                    kc = self.enumerate_kan_candidates(state, actor)
                    if len(kc) > 1:
                        yield self.make_sample(state, actor, [c[0] for c in kc], 0, "kan_decision")
                    acts = self.enumerate_dahai_actions(state, actor)
                    k = _tid_norm(self.tile_id(pai))
                    target = (ACT_RIICHI_BASE if declaring else ACT_DAHAI_BASE) + k
                    if target in acts:
                        s = self.make_sample(state, actor, acts, acts.index(target), "dahai")
                        s["dahai_count"] = state.dahai_count
                        s["is_riichi_dahai"] = declaring
                        if any(a >= ACT_RIICHI_BASE for a in acts):
                            s["riichi_label"] = 1 if declaring else 0
                        if self._emit_aux_fields:
                            s["_event_index"] = _ev_i
                            s["_hand_strs"]   = list(state.tehais[actor])
                            s["_num_melds"]   = state.meld_counts[actor]
                            s["_dahai_str"]   = pai
                            # 当たり牌ラベル用（相手の手牌は報酬/ラベル計算にのみ使い、入力には入れない）
                            s["_all_hands"]   = [list(state.tehais[p]) for p in range(NUM_PLAYERS)]
                            s["_all_melds"]   = [list(state.meld_info[p]) for p in range(NUM_PLAYERS)]
                            s["_riichi_flags"] = list(state.riichi_flags)
                            s["_discards34"]  = [sorted(state.safe_tiles[p]) for p in range(NUM_PLAYERS)]
                            s["_bakaze"]      = state.bakaze
                            s["_oya"]         = state.oya
                        yield s

                if declaring:
                    st_riichi(state, actor)
                st_dahai(state, actor, pai, tg)
                state.pending_call_candidates = {}
                for p in range(NUM_PLAYERS):
                    opts = self.call_options(state, p, pai, actor)
                    if opts["pon"] or opts["chi"] or opts["daiminkan"]:
                        state.pending_call_candidates[p] = opts

            elif etype in ("chi", "pon", "daiminkan"):
                actor = event["actor"]
                consumed = list(event.get("consumed", []))
                pai = event.get("pai") or (consumed[0] if consumed else "1m")
                opts = state.pending_call_candidates.get(actor) or \
                    self.call_options(state, actor, pai, state.last_dahai_player)
                opts = dict(opts); opts[etype] = True
                s = self.build_meld_decision_sample(state, actor, pai, opts, etype, consumed)
                if s:
                    yield s
                yield from self._flush_pending(state, "chi_by" if etype == "chi" else "drop", caller=actor)
                {"chi": st_chi, "pon": st_pon, "daiminkan": st_daiminkan}[etype](state, actor, pai, consumed)

            elif etype in ("ankan", "kakan"):
                actor = event["actor"]
                consumed = list(event.get("consumed", []))
                pai = event.get("pai") or (consumed[0] if consumed else "1m")
                if not state.riichi_flags[actor]:
                    kc = self.enumerate_kan_candidates(state, actor)
                    tok = (ACT_ANKAN_BASE if etype == "ankan" else ACT_KAKAN_BASE) \
                        + min(_tid_norm(self.tile_id(pai)), 33)
                    toks = [c[0] for c in kc]
                    if tok in toks:
                        yield self.make_sample(state, actor, toks, toks.index(tok), "kan_decision")
                if etype == "ankan":
                    n = norm_str(pai)
                    cons = [t for t in state.tehais[actor] if norm_str(t) == n][:4] or consumed
                    st_ankan(state, actor, cons)
                else:
                    st_kakan(state, actor, pai)

            elif etype == "dora":
                st_dora(state, event.get("dora_marker", "1m"))

            elif etype in ("hora", "ryukyoku", "end_kyoku"):
                for _ in self._flush_pending(state, "drop"):
                    pass
                # 次局の start_kyoku に scores が無い牌譜でも点数を引き継げるようにしておく
                sc = event.get("scores")
                if isinstance(sc, list) and len(sc) == 4:
                    game_scores = list(sc)
                    state.scores = list(sc)
                elif isinstance(event.get("deltas"), list) and len(event["deltas"]) == 4:
                    state.scores = [state.scores[i] + event["deltas"][i] for i in range(4)]
                    game_scores = list(state.scores)

            elif etype == "end_game":
                sc = event.get("scores")
                if isinstance(sc, list) and len(sc) == 4:
                    game_scores = list(sc)

    def process_file(self, path):
        path = Path(path); events = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        yield from self.process_events(events)

    def process_string(self, text):
        events = [json.loads(l) for l in text.strip().splitlines() if l.strip()]
        yield from self.process_events(events)


# ---------------------------------------------------------------------------
# 9. 動作確認
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ([0,1,2,3,4,5,6,7,8,9,10,11,12], 0, 0),
        ([0,1,2,9,10,11,18,19,20,27,28,29,30], 0, 2),
        ([0,3,6,9,12,15,18,21,24,27,28,29,30], 0, 6),
        ([0,1,2,3,4,5,6,7,8,13], 1, 0),
        ([0,1,13,13], 3, 0),
        ([13], 4, 0),
    ]
    for ids, nm, exp in tests:
        s = shanten_number(ids, nm)
        print(f"shanten({len(ids)}枚, melds={nm}) = {s}  {'OK' if s == exp else 'NG'}")
    print(f"SPARSE_VOCAB_SIZE={SPARSE_VOCAB_SIZE} PROG_VOCAB_SIZE={PROG_VOCAB_SIZE} "
          f"NUMERIC_DIM={NUMERIC_DIM} CAND_FEAT_DIM={CAND_FEAT_DIM}")
