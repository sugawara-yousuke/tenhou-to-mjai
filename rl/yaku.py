"""
yaku.py  ―  面子分解ベースの役判定・符計算
================================================================
互換実装（リーチ/タンヤオ/役牌/チャンタ/混一/清一/対々/七対子/国士/大三元 だけ）を置き換える。
互換実装では三色・一通・一盃口・三暗刻などを役として認識できず、self_play では
「それらしか役の無い手は和了できない」麻雀を打っていた（＝RL の報酬が実戦と食い違う）。

方針:
  門前部分の「全ての面子分解 × 和了牌がどの部分を完成させたか」を総当たりし、
  (翻, 符) が最大になる解釈を採用する（高点法）。

公開関数:
  evaluate_hand(hand, melds, win_tile, is_tsumo, ...) -> dict
      {"yaku": [(name, han)], "han": int(ドラ抜き), "fu": int, "yakuman": int(倍数, 0=役満でない)}
  detect_yaku(...) -> (yaku_list, is_yakuman)        # 旧シグネチャ互換
  has_yaku(...) -> bool

引数:
  hand     : 門前の牌（文字列, 赤は "0m" 等）。win_tile を渡す場合は和了牌を含めない 13-3n 枚。
             win_tile=None の場合は hand の最後の1枚を和了牌とみなす（旧呼び出し互換）。
  melds    : [{"type": "chi"|"pon"|"daiminkan"|"ankan"|"kakan", "pai": 鳴いた牌 or None,
               "consumed": [...], ("added": 加槓牌)}]
  ルール   : 天鳳準拠（喰いタン有り・連風牌の雀頭4符・単独役のダブル役満なし・役満の複合あり）
"""

from __future__ import annotations

_ID = {}
for _i, _t in enumerate([f"{n}{s}" for s in "mps" for n in range(1, 10)] + [f"{n}z" for n in range(1, 8)]):
    _ID[_t] = _i
_ID.update({"0m": 4, "0p": 13, "0s": 22, "5mr": 4, "5pr": 13, "5sr": 22,
            "E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33})
WIND = {"E": 27, "S": 28, "W": 29, "N": 30}
DRAGONS = (31, 32, 33)
TERMINALS = {0, 8, 9, 17, 18, 26}
YAOCHU = TERMINALS | set(range(27, 34))
GREEN = {19, 20, 21, 23, 25, 32}


def _k(t: str) -> int:
    return _ID[t]


def _is_yaochu_set(kind: str, x: int) -> bool:
    if kind == "shuntsu":
        return x % 9 == 0 or x % 9 == 6
    return x in YAOCHU


def _decompose(counts: list[int]) -> list[tuple[int, list[tuple[str, int]]]]:
    """34種の枚数 → [(雀頭, [("shuntsu", start) | ("kotsu", kind), ...]), ...] 全解。"""
    out = []

    def sets(c, i, acc):
        while i < 34 and c[i] == 0:
            i += 1
        if i >= 34:
            out_sets.append(list(acc)); return
        if c[i] >= 3:
            c[i] -= 3; acc.append(("kotsu", i)); sets(c, i, acc); acc.pop(); c[i] += 3
        if i < 27 and i % 9 <= 6 and c[i + 1] and c[i + 2]:
            c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
            acc.append(("shuntsu", i)); sets(c, i, acc); acc.pop()
            c[i] += 1; c[i + 1] += 1; c[i + 2] += 1

    for p in range(34):
        if counts[p] >= 2:
            c = counts[:]; c[p] -= 2
            out_sets: list = []
            sets(c, 0, [])
            for s in out_sets:
                out.append((p, s))
    return out


def _meld_sets(melds):
    """副露 → [(kind, x, is_open, is_kan)]"""
    res = []
    for m in melds or []:
        mt = m.get("type", "")
        tiles = [_k(t) for t in (m.get("consumed") or [])]
        if m.get("pai"):
            tiles.append(_k(m["pai"]))
        if m.get("added"):
            tiles.append(_k(m["added"]))
        if not tiles:
            continue
        if mt == "chi":
            res.append(("shuntsu", min(tiles), True, False))
        elif mt == "pon":
            res.append(("kotsu", tiles[0], True, False))
        elif mt in ("daiminkan", "kakan"):
            res.append(("kotsu", tiles[0], True, True))
        elif mt == "ankan":
            res.append(("kotsu", tiles[0], False, True))
    return res


def _round_up10(x):
    return ((x + 9) // 10) * 10


def evaluate_hand(hand, melds=None, win_tile=None, is_tsumo=True, is_riichi=False,
                  bakaze="E", jikaze="E", is_ippatsu=False, is_double_riichi=False,
                  is_haitei=False, is_houtei=False, is_rinshan=False, is_chankan=False,
                  is_tenhou=False, is_chiihou=False) -> dict:
    melds = melds or []
    hand = list(hand)
    if win_tile is None:
        if not hand:
            return {"yaku": [], "han": 0, "fu": 0, "yakuman": 0}
        win_tile = hand[-1]; hand = hand[:-1]
    w = _k(win_tile)
    counts = [0] * 34
    for t in hand:
        counts[_k(t)] += 1
    counts[w] += 1
    msets = _meld_sets(melds)
    is_menzen = all(not s[2] for s in msets)
    n_kan = sum(1 for s in msets if s[3])
    bw, jw = WIND.get(bakaze, 27), WIND.get(jikaze, 27)

    all_kinds = [i for i in range(34) if counts[i]]
    for kind, x, _o, _kn in msets:
        all_kinds += [x, x + 1, x + 2] if kind == "shuntsu" else [x]
    kinds_set = set(all_kinds)
    suits = {k // 9 for k in kinds_set if k < 27}
    has_honor = any(k >= 27 for k in kinds_set)

    situational = []
    if is_double_riichi:   situational.append(("double_riichi", 2))
    elif is_riichi:        situational.append(("riichi", 1))
    if (is_riichi or is_double_riichi) and is_ippatsu: situational.append(("ippatsu", 1))
    if is_menzen and is_tsumo: situational.append(("menzen_tsumo", 1))
    if is_haitei and is_tsumo: situational.append(("haitei", 1))
    if is_houtei and not is_tsumo: situational.append(("houtei", 1))
    if is_rinshan: situational.append(("rinshan", 1))
    if is_chankan: situational.append(("chankan", 1))

    def color_yaku():
        y = []
        if len(suits) == 1:
            y.append(("honitsu", 3 if is_menzen else 2) if has_honor else ("chinitsu", 6 if is_menzen else 5))
        return y

    best = None

    def consider(yaku, fu, yakuman):
        nonlocal best
        han = sum(h for _, h in yaku)
        key = (yakuman, han, fu)
        if best is None or key > best[0]:
            best = (key, yaku, fu, yakuman)

    base_ym = []
    if is_tenhou:  base_ym.append("tenhou")
    if is_chiihou: base_ym.append("chiihou")
    if kinds_set and kinds_set <= set(range(27, 34)): base_ym.append("tsuuiisou")
    if kinds_set and kinds_set <= TERMINALS:          base_ym.append("chinroutou")
    if kinds_set and kinds_set <= GREEN:              base_ym.append("ryuuiisou")

    # ---- 国士無双 ----
    if is_menzen and not msets and all(counts[k] >= 1 for k in YAOCHU) \
            and sum(counts[k] for k in YAOCHU) == 14:
        consider([("kokushi", 13)] + [(n, 13) for n in base_ym if n in ("tenhou", "chiihou")], 0,
                 1 + sum(1 for n in base_ym if n in ("tenhou", "chiihou")))

    # ---- 七対子 ----
    if is_menzen and not msets and sum(1 for c in counts if c == 2) == 7:
        if base_ym:
            consider([(n, 13) for n in base_ym], 0, len(base_ym))
        else:
            y = list(situational) + [("chiitoitsu", 2)] + color_yaku()
            if not (kinds_set & YAOCHU): y.append(("tanyao", 1))
            if kinds_set <= YAOCHU:      y.append(("honroutou", 2))
            consider(y, 25, 0)

    # ---- 一般形 ----
    for pair, csets in _decompose(counts):
        if len(csets) + len(msets) != 4:
            continue
        roles = set()
        if pair == w:
            roles.add(("tanki", w))
        for kind, x in csets:
            if kind == "kotsu" and x == w:
                roles.add(("shanpon", x))
            elif kind == "shuntsu" and x <= w <= x + 2:
                if w == x + 1:                       roles.add(("kanchan", x))
                elif w == x + 2 and x % 9 == 0:      roles.add(("penchan", x))
                elif w == x and x % 9 == 6:          roles.add(("penchan", x))
                else:                                roles.add(("ryanmen", x))
        for wait, wx in roles:
            # 各面子: (kind, x, is_open, is_kan)。ロンで完成した刻子は明刻扱い。
            sets_all = list(msets)
            ron_kotsu_used = False
            for kind, x in csets:
                opened = False
                if kind == "kotsu" and wait == "shanpon" and x == wx and not is_tsumo and not ron_kotsu_used:
                    opened = True; ron_kotsu_used = True
                sets_all.append((kind, x, opened, False))
            shuntsu = [x for k, x, _o, _kn in sets_all if k == "shuntsu"]
            kotsu = [(x, o, kn) for k, x, o, kn in sets_all if k == "kotsu"]
            n_anko = sum(1 for x, o, kn in kotsu if not o)

            ym = list(base_ym)
            if n_anko == 4: ym.append("suuankou")
            if sum(1 for x, *_ in kotsu if x in DRAGONS) == 3: ym.append("daisangen")
            n_wind = sum(1 for x, *_ in kotsu if 27 <= x <= 30)
            if n_wind == 4: ym.append("daisuushii")
            elif n_wind == 3 and 27 <= pair <= 30: ym.append("shousuushii")
            if n_kan == 4: ym.append("suukantsu")
            if is_menzen and not msets and len(suits) == 1 and not has_honor:
                s0 = next(iter(suits)) * 9
                c9 = counts[s0:s0 + 9]
                if c9[0] >= 3 and c9[8] >= 3 and all(c >= 1 for c in c9):
                    ym.append("chuuren")
            if ym:
                consider([(n, 13) for n in ym], 0, len(ym)); continue

            y = list(situational)
            pair_yakuhai_fu = (2 if pair in DRAGONS else 0) + (2 if pair == bw else 0) + (2 if pair == jw else 0)
            is_pinfu = (is_menzen and len(shuntsu) == 4 and pair_yakuhai_fu == 0 and wait == "ryanmen")
            if is_pinfu: y.append(("pinfu", 1))
            if not (kinds_set & YAOCHU): y.append(("tanyao", 1))
            if is_menzen:
                dup = sum(shuntsu.count(x) // 2 for x in set(shuntsu))
                if dup >= 2:   y.append(("ryanpeikou", 3))
                elif dup == 1: y.append(("iipeikou", 1))
            for x, *_ in kotsu:
                if x == 31: y.append(("haku", 1))
                if x == 32: y.append(("hatsu", 1))
                if x == 33: y.append(("chun", 1))
                if x == bw: y.append(("bakaze", 1))
                if x == jw: y.append(("jikaze", 1))
            for n in range(7):
                if all((s * 9 + n) in shuntsu for s in range(3)):
                    y.append(("sanshoku", 2 if is_menzen else 1)); break
            for s in range(3):
                if all((s * 9 + n) in shuntsu for n in (0, 3, 6)):
                    y.append(("ittsu", 2 if is_menzen else 1)); break
            ksets = {x for x, *_ in kotsu}
            if any(all((s * 9 + n) in ksets for s in range(3)) for n in range(9)):
                y.append(("sanshoku_doukou", 2))
            if len(kotsu) == 4: y.append(("toitoi", 2))
            if n_anko == 3:     y.append(("sanankou", 2))
            if n_kan == 3:      y.append(("sankantsu", 2))
            if sum(1 for x, *_ in kotsu if x in DRAGONS) == 2 and pair in DRAGONS:
                y.append(("shousangen", 2))
            if kinds_set <= YAOCHU:
                y.append(("honroutou", 2))
            elif pair in YAOCHU and all(_is_yaochu_set(k, x) for k, x, *_ in sets_all):
                y.append(("chanta", 2 if is_menzen else 1) if has_honor else ("junchan", 3 if is_menzen else 2))
            y += color_yaku()

            # ---- 符 ----
            if is_pinfu:
                fu = 20 if is_tsumo else 30
            else:
                fu = 20 + (10 if (is_menzen and not is_tsumo) else 0) + (2 if is_tsumo else 0)
                fu += pair_yakuhai_fu
                if wait in ("tanki", "kanchan", "penchan"): fu += 2
                for x, o, kn in kotsu:
                    f = 2 * (2 if x in YAOCHU else 1) * (1 if o else 2) * (4 if kn else 1)
                    fu += f
                fu = _round_up10(fu)
                if fu == 20: fu = 30       # 喰い平和形
            consider(y, fu, 0)

    if best is None:
        return {"yaku": [], "han": 0, "fu": 0, "yakuman": 0}
    (_key, yaku, fu, yakuman) = best
    # 状況役だけ（門前ツモ等）でも役。何も無ければ役なし。
    return {"yaku": yaku, "han": 0 if yakuman else sum(h for _, h in yaku), "fu": fu, "yakuman": yakuman}


def detect_yaku(hand, melds, win_tile=None, is_tsumo=True, is_riichi=False,
                bakaze="E", jikaze="E", **flags):
    r = evaluate_hand(hand, melds, win_tile, is_tsumo, is_riichi, bakaze, jikaze, **flags)
    return r["yaku"], r["yakuman"] > 0


def has_yaku(hand, melds, win_tile=None, is_tsumo=True, is_riichi=False,
             bakaze="E", jikaze="E", **flags) -> bool:
    r = evaluate_hand(hand, melds, win_tile, is_tsumo, is_riichi, bakaze, jikaze, **flags)
    return r["yakuman"] > 0 or len(r["yaku"]) > 0


def is_agari_shape(hand, melds=None, win_tile=None) -> bool:
    """役の有無を問わず和了形か。"""
    tiles = list(hand) + ([win_tile] if win_tile else [])
    counts = [0] * 34
    for t in tiles:
        counts[_k(t)] += 1
    n_m = len(melds or [])
    if n_m == 0:
        if sum(1 for c in counts if c == 2) == 7:
            return True
        if all(counts[k] >= 1 for k in YAOCHU) and sum(counts[k] for k in YAOCHU) == 14:
            return True
    return any(len(s) + n_m == 4 for _p, s in _decompose(counts))
