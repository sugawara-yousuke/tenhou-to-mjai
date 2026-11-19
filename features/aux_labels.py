"""
aux_labels.py
=============
提案3: BCの主タスク（打牌予測）に、押し引きに直結する補助タスクを
マルチタスクで同時学習させるための「教師ラベル」を生成する。

方式A（2パス）:
  process_events は頭からストリーミングでサンプルを吐くが、
  「この打牌で放銃したか」「その局で誰がテンパイだったか」は未来のイベントを
  見ないと分からない。そこで局単位でイベントを事前走査して
  局の決着情報（和了者/放銃者/放銃牌/流局テンパイ）を作り、
  各サンプルにラベルを後付けする。

付与する補助ラベル（サンプル dict に追加されるキー）:
  aux_houjuu       : int(0/1)   この dahai サンプルの打牌が放銃になったか
                     （＝直後に他家がこの牌でロン和了した）
  aux_tenpai_opp   : list[int]  他家3人(rel=1,2,3順)が「その局で最終的に
                     テンパイしていたか」の 0/1。和了者=テンパイ、
                     流局テンパイ者=テンパイ。※局全体の帰結を弱ラベルとして使う。
  aux_ukeire       : float      現局面での自分の受け入れ枚数（正規化前の生値）。
                     未来不要・その場で正確に計算できる。
  aux_valid        : int(0/1)   補助ラベルが有効か（放銃/テンパイが判定できたか）

設計上の注意:
  - aux_tenpai_opp は「打牌時点の相手テンパイ」ではなく「局終了時点でテンパイか」
    の弱ラベル。局面時点の相手テンパイの真値は牌譜に無いため、
    「その相手はこの局でアガれる手だったか」を近似シグナルとして使う。
    序盤の局面にも同じ帰結が付くのでノイズはあるが、平均すれば
    「今テンパイの気配がある相手か」を表現する内部特徴を促す方向に効く。
  - aux_houjuu は局面時点で正確（その打牌が実際に放銃したかは牌譜で確定）。
  - 相手プレイヤーの並びは build_sparse/build_numeric と同じ相対 rel=1,2,3。
"""

from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
from typing import Iterator


def _split_kyoku(events: list[dict]) -> list[tuple[int, int]]:
    """events を start_kyoku..(end_kyoku or 次のstart_kyoku) 区間に分割。
    返り値: [(start_idx, end_idx_exclusive), ...]"""
    spans = []
    cur = None
    for i, e in enumerate(events):
        t = e.get("type", "")
        if t == "start_kyoku":
            if cur is not None:
                spans.append((cur, i))
            cur = i
        elif t == "end_kyoku":
            if cur is not None:
                spans.append((cur, i + 1))
                cur = None
    if cur is not None:
        spans.append((cur, len(events)))
    return spans


def summarize_kyoku(events: list[dict], span: tuple[int, int]) -> dict:
    """
    1局分のイベント区間から決着情報を抽出。

    Returns dict:
      winners      : set[int]   和了したプレイヤー
      loser        : int|None   放銃したプレイヤー（ロン和了の target）
      ron_tile_id  : int|None   放銃牌の tile_id（37種）— 使わないが保持
      ron_dahai_ev : int|None   放銃となった打牌イベントの events index
      tenpai       : list[bool] 流局時テンパイ（ryukyoku の tenpai フィールド）
      any_win      : bool
    """
    from mahjong_tokenizer import TILE_STR_TO_ID
    s, e = span
    winners: set[int] = set()
    loser = None
    ron_tile_id = None
    ron_dahai_ev = None
    tenpai = [False, False, False, False]

    # 直近の打牌イベントindexとactorを追いながら、hora(ロン)に紐づける
    last_dahai_idx = None
    last_dahai_actor = None
    last_dahai_tile = None

    for i in range(s, e):
        ev = events[i]
        t = ev.get("type", "")
        if t == "dahai":
            last_dahai_idx = i
            last_dahai_actor = ev.get("actor")
            last_dahai_tile = ev.get("pai")
        elif t == "hora":
            actor = ev.get("actor")
            target = ev.get("target")
            if actor is not None:
                winners.add(actor)
            # ロン和了（target != actor かつ target 指定あり）なら放銃
            if target is not None and target != actor:
                loser = target
                # 放銃牌 = 直前の打牌（target が打った牌）
                if last_dahai_actor == target and last_dahai_tile is not None:
                    ron_tile_id = TILE_STR_TO_ID.get(last_dahai_tile, None)
                    ron_dahai_ev = last_dahai_idx
        elif t == "ryukyoku":
            tnp = ev.get("tenpai")
            if isinstance(tnp, list):
                for k in range(min(4, len(tnp))):
                    tenpai[k] = bool(tnp[k])

    # 和了者はテンパイ扱い
    for w in winners:
        if 0 <= w < 4:
            tenpai[w] = True

    return {
        "winners": winners,
        "loser": loser,
        "ron_tile_id": ron_tile_id,
        "ron_dahai_ev": ron_dahai_ev,
        "tenpai": tenpai,
        "any_win": len(winners) > 0,
    }


def _to34(tokenizer, tile_str: str) -> int:
    """牌文字列 → 0..33 の34種index（赤牌 0m/0p/0s は 5 の位置へ）。範囲外は -1。"""
    x = tokenizer.tile_id(tile_str)
    x = {34: 4, 35: 13, 36: 22}.get(x, x)  # 赤5 → 通常5
    return x if 0 <= x <= 33 else -1


from functools import lru_cache

_YAOCHU34 = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)


@lru_cache(maxsize=1 << 16)
def _ukeire_cached(counts_t: tuple, num_melds: int) -> int:
    from mahjong_tokenizer import shanten_from_counts, _shanten_kokushi
    counts = list(counts_t)
    base = shanten_from_counts(counts_t, num_melds)
    # 向聴を下げ得るのは「手牌にある牌」「同色で±2以内の牌」だけ（国士が絡むときは么九牌も）。
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
    if num_melds == 0 and _shanten_kokushi(counts) <= base:
        cand.update(_YAOCHU34)
    total = 0
    for k in cand:
        held = counts[k]
        if held >= 4:
            continue
        counts[k] += 1
        s = shanten_from_counts(tuple(counts), num_melds)
        counts[k] -= 1
        if s < base:
            total += 4 - held
    return total


def _ukeire_from_counts(counts: list[int], num_melds: int) -> int:
    """受け入れ枚数。同じ13枚形（ツモ切りが続く巡目など）は結果ごとキャッシュする。"""
    return _ukeire_cached(tuple(counts), num_melds)


def compute_ukeire(tokenizer, hand_strs: list[str], num_melds: int = 0,
                   dahai_str: str | None = None) -> int:
    """
    受け入れ枚数（有効牌の種類数ではなく「枚数」）を数える。正確な即時計算。

    [高速化・定義] dahai_str が渡された場合、「その牌を1枚切った後の
    13枚」の受け入れを計算する。BC の補助ラベルとしては
    「教師がこの牌を切った結果どれだけ受け入れが残るか」が自然な信号であり、
    14枚全切り牌の最良探索（旧実装）と違い shanten 呼び出しが 1/14 で済む。

    dahai_str が None のとき:
      - 13枚形ならそのまま計算
      - 14枚形は後方互換のため最初の1枚を切った13枚で計算（近似）
    """
    ids34 = [t for t in (_to34(tokenizer, s) for s in hand_strs) if t >= 0]
    # 打牌後の枚数は 13 - 3*副露数（旧版は 13 枚固定で、副露手の受け入れが常に 0 になっていた）
    target = 13 - 3 * num_melds

    if dahai_str is not None and len(ids34) == target + 1:
        d = _to34(tokenizer, dahai_str)
        if d >= 0 and d in ids34:
            ids34 = ids34[:]
            ids34.remove(d)
    if len(ids34) == target + 1:
        ids34 = ids34[1:]          # dahai 情報が無い場合の後方互換
    if len(ids34) != target or target < 1:
        return 0

    counts = [0] * 34
    for x in ids34:
        counts[x] += 1
    return _ukeire_from_counts(counts, num_melds)


def attach_aux_labels(
    tokenizer,
    events: list[dict],
    samples: list[dict],
) -> list[dict]:
    """
    process_events が生成した samples（1ゲーム分）に補助ラベルを後付けする。

    samples の各要素は tokenizer.process_events の yield 物（dict）を想定。
    dahai サンプルには sample["_event_index"] が必要（下の instrumented 版が付ける）。
    ここでは _event_index を使って所属する局を特定し、放銃/テンパイを引く。
    """
    spans = _split_kyoku(events)
    # event index -> (span, summary) の対応表
    summaries = [(sp, summarize_kyoku(events, sp)) for sp in spans]

    def find_summary(ev_idx: int):
        for sp, summ in summaries:
            if sp[0] <= ev_idx < sp[1]:
                return sp, summ
        return None, None

    for smp in samples:
        et = smp.get("event_type", "")
        player = smp.get("player", 0)
        ev_idx = smp.get("_event_index", None)

        # デフォルト（無効）
        smp["aux_valid"] = 0
        smp["aux_houjuu"] = 0
        smp["aux_tenpai_opp"] = [0, 0, 0]
        smp["aux_ukeire"] = 0.0

        if et != "dahai" or ev_idx is None:
            continue

        sp, summ = find_summary(ev_idx)
        if summ is None:
            continue

        # --- 放銃ラベル: この打牌イベントが放銃牌そのものか ---
        smp["aux_houjuu"] = 1 if (summ["ron_dahai_ev"] == ev_idx) else 0

        # --- 他家テンパイ弱ラベル（rel=1,2,3 順） ---
        tnp = summ["tenpai"]
        opp = []
        for rel in range(1, 4):
            pi = (player + rel) % 4
            opp.append(1 if tnp[pi] else 0)
        smp["aux_tenpai_opp"] = opp

        # --- 受け入れ枚数（打牌前の手牌から。sparse 再構成せず sample から取れないので
        #     tokenizer 状態は無い→ここでは近似として progression 非依存に skip し、
        #     instrumented 版が sample["_hand_strs"] を渡してくれた時のみ計算） ---
        hand_strs = smp.get("_hand_strs", None)
        num_melds = smp.get("_num_melds", 0)
        dahai_str = smp.get("_dahai_str", None)
        if hand_strs is not None:
            smp["aux_ukeire"] = float(
                compute_ukeire(tokenizer, hand_strs, num_melds, dahai_str))

        smp["aux_valid"] = 1

    return samples


# ===========================================================================
# 当たり牌ラベル（提案: 危険度ヘッドの教師）
# ===========================================================================
"""
各打牌サンプルについて「いまこの牌を切ったら、相手 rel=1,2,3 にロンされるか」を
牌譜から正確に作る。self_play/報酬と同じく、相手の手牌はラベル計算にのみ使い、
モデルの入力には一切入らない。

付与するキー:
  aux_ron        : list[3][34]  rel=1,2,3 × 牌種34 の 0/1。1 ならその牌はロンされる
  aux_ron_valid  : int(0/1)     ラベルが計算できたか

danger_features.py の手計算ヒューリスティック（スジ・現物・ノーチャンス）を
学習された確率に置き換えるための教師。染め手・対々・副露者のテンパイ推定なども、
ラベルが正解を含んでいるのでモデル側が勝手に学ぶ。
"""

_ID34_TO_STR = None


def _id34_to_str(k: int) -> str:
    global _ID34_TO_STR
    if _ID34_TO_STR is None:
        from mahjong_tokenizer import TILE_ID_TO_STR
        _ID34_TO_STR = [TILE_ID_TO_STR[i] for i in range(34)]
    return _ID34_TO_STR[k]


def _meld_sig(melds) -> tuple:
    """副露をキャッシュキーにできる形へ。"""
    out = []
    for m in melds or []:
        out.append((m.get("type", ""), m.get("pai") or "",
                    tuple(m.get("consumed") or []), m.get("added") or ""))
    return tuple(out)


@lru_cache(maxsize=1 << 17)
def _ron_tiles_cached(counts_t: tuple, meld_sig: tuple, bakaze: str, jikaze: str,
                      is_riichi: bool, discards_t: tuple) -> frozenset:
    """
    その手牌でロン和了できる牌種34の集合。役なし・フリテンは除く。
    counts_t: 34種の枚数（手牌のみ、13-3n 枚）
    """
    from yaku import is_agari_shape, has_yaku
    hand = []
    for k, c in enumerate(counts_t):
        hand += [_id34_to_str(k)] * c
    melds = [{"type": t, "pai": (p or None), "consumed": list(c),
              **({"added": a} if a else {})} for (t, p, c, a) in meld_sig]
    waits = set()
    for k in range(34):
        if counts_t[k] >= 4:
            continue
        w = _id34_to_str(k)
        if not is_agari_shape(hand, melds, w):
            continue
        waits.add(k)
    if not waits:
        return frozenset()
    # フリテン: 待ちのどれか1つでも自分の河にあれば、どの牌でもロンできない
    if waits & set(discards_t):
        return frozenset()
    out = set()
    for k in waits:
        w = _id34_to_str(k)
        if has_yaku(hand, melds, w, is_tsumo=False, is_riichi=is_riichi,
                    bakaze=bakaze, jikaze=jikaze):
            out.add(k)
    return frozenset(out)


def attach_ron_labels(samples: list) -> list:
    """
    process_events が出した samples（_emit_aux_fields=True で生成したもの）に
    aux_ron / aux_ron_valid を付ける。
    """
    _WINDS = ("E", "S", "W", "N")
    for smp in samples:
        smp["aux_ron"] = [[0] * 34 for _ in range(3)]
        smp["aux_ron_valid"] = 0
        if smp.get("event_type") != "dahai" or "_all_hands" not in smp:
            continue
        player = smp.get("player", 0)
        oya = smp.get("_oya", 0)
        bakaze = smp.get("_bakaze", "E")
        for rel in range(1, 4):
            q = (player + rel) % 4
            counts = [0] * 34
            for t in smp["_all_hands"][q]:
                x = _to34_str(t)
                if x >= 0:
                    counts[x] += 1
            jikaze = _WINDS[(q - oya) % 4]
            tiles = _ron_tiles_cached(tuple(counts), _meld_sig(smp["_all_melds"][q]),
                                      bakaze, jikaze, bool(smp["_riichi_flags"][q]),
                                      tuple(smp["_discards34"][q]))
            for k in tiles:
                smp["aux_ron"][rel - 1][k] = 1
        smp["aux_ron_valid"] = 1
    return samples


def _to34_str(tile_str: str) -> int:
    from mahjong_tokenizer import TILE_STR_TO_ID
    x = TILE_STR_TO_ID.get(tile_str, -1)
    x = {34: 4, 35: 13, 36: 22}.get(x, x)
    return x if 0 <= x <= 33 else -1


# ===========================================================================
# 局リターンラベル（AWR 用）
# ===========================================================================
"""
各サンプルに「その局の順位期待値の変化」R_局 を付ける。

  R_局 = placement_reward(局前点数, 局後点数, player, 局前残り局数, 局後残り局数)

AWR（Advantage Weighted Regression）で
  A = R_局 − V(s),   w = exp(A/β)
として方策損失を重み付けするのに使う。「鳳凰卓の打ち手の選択のうち、
結果が良かったものを重く学ぶ」形になる。

点数の取り方:
  局前 = その局の start_kyoku の scores
  局後 = 次の局の start_kyoku の scores（最終局は end_game の scores）
hora / ryukyoku イベントのキー名に依存しないので、牌譜の方言に強い。

計算は点数の四則演算だけで軽いので .aux_cache には入れない（毎回計算する）。
"""

_BK_IDX = {"E": 0, "S": 1, "W": 2, "N": 3}


def attach_return_labels(events: list[dict], samples: list[dict]) -> list[dict]:
    from reward import placement_reward, kyoku_left_before

    spans = _split_kyoku(events)
    final_scores = None
    for ev in reversed(events):
        if ev.get("type") == "end_game" and ev.get("scores"):
            final_scores = list(ev["scores"]); break

    # 局ごとに (開始index, 終了index, 局前点数, 局後点数, 残り局数前, 残り局数後)
    info = []
    for i, (s, e) in enumerate(spans):
        head = events[s]
        sc_before = head.get("scores")
        if not sc_before or len(sc_before) != 4:
            info.append(None); continue
        bk = _BK_IDX.get(head.get("bakaze", "E"), 0)
        kyoku = int(head.get("kyoku", 1))
        left_before = kyoku_left_before(bk, kyoku)
        if i + 1 < len(spans):
            nxt = events[spans[i + 1][0]]
            sc_after = nxt.get("scores")
            bk2 = _BK_IDX.get(nxt.get("bakaze", "E"), 0)
            left_after = kyoku_left_before(bk2, int(nxt.get("kyoku", 1)))
        else:
            sc_after = final_scores
            left_after = 0.0
        if not sc_after or len(sc_after) != 4:
            info.append(None); continue
        info.append((s, e, list(sc_before), list(sc_after), left_before, left_after))

    def find(ev_idx):
        for rec in info:
            if rec is not None and rec[0] <= ev_idx < rec[1]:
                return rec
        return None

    # 鳴き/槓の判断サンプルには _event_index が付かないので、直前の打牌サンプルの局を継ぐ
    last_idx = None
    for smp in samples:
        smp["aux_ret"] = 0.0
        smp["aux_ret_valid"] = 0
        ev_idx = smp.get("_event_index")
        if ev_idx is not None:
            last_idx = ev_idx
        else:
            ev_idx = last_idx
        if ev_idx is None:
            continue
        rec = find(ev_idx)
        if rec is None:
            continue
        _s, _e, sc_b, sc_a, lb, la = rec
        player = smp.get("player", 0)
        smp["aux_ret"] = float(placement_reward(sc_b, sc_a, player, lb, la))
        smp["aux_ret_valid"] = 1
    return samples
