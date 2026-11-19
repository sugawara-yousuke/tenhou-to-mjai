"""
self_play.py  ―  自己対戦シミュレータ（BC と同じ状態更新・同じ特徴量で打つ）
================================================================

[今回の改修]
  1. 状態更新は mahjong_tokenizer の st_* だけを使う。特徴量も tokenizer.make_sample だけで作る。
     旧版は self_play 側で state を直接いじっており、progression にツモしか積まれない・点数の
     扱いが BC と違う等、「BC で見た入力」と「対戦中の入力」が食い違っていた。
  2. 1局の進行を明示的な状態機械(KyokuCtx.phase = draw / after_draw / discard)にした。
     → 判断点で KyokuCtx を deepcopy すれば、そこから何度でも再開できる（playout_from）。
     旧・方式A は「同じ山を頭から greedy で打ち直して分岐点を再現」していたが、dropout が
     有効だと再現できず、plan3 のフォールバックでは分岐前のツモごと変わっていた。
     スナップショット再開なら再現性の問題そのものが無くなり、前半の再生コストも消える。
  3. ルール修正
     - リーチ後はツモ切り固定（モデルに聞かない）。旧版は手牌全部が候補だった。
     - 食い替えは候補から除外（現物＋スジ）。旧版は「最後の副露がチーの間ずっと」誤った
       禁止牌で打牌を差し替え、RL の記録には差し替え前の行動が残っていた。
     - 槓→嶺上開花の点数が加算されない不具合、槓でツモ山が減らない不具合を修正。搶槓を実装。
     - リーチ棒は宣言牌が通ってから供託（mjai 順: reach → dahai → reach_accepted）。
     - 一発・ダブルリーチ・海底/河底・嶺上・天和/地和を判定。役と符は yaku.evaluate_hand。
     - 流局時の本場、西入、オーラスの和了止め/聴牌止めを simulate_game / RL で共通化。
  4. 推論は必ず eval モード・no_grad。モデルごとの使い分け(player_models)は従来どおり。
"""

from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
import argparse
import copy
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

import yaku as _yaku
from mahjong_model import samples_to_batch, load_policy
from mahjong_tokenizer import (
    MahjongTokenizer, KyokuState, TILE_STR_TO_ID, TILE_ID_TO_STR, RED_STRS,
    PADDING_ACTION_ID, ACT_DAHAI_BASE, ACT_RIICHI_BASE, ACT_PON_BASE,
    ACT_ANKAN_BASE, ACT_KAKAN_BASE, ACT_PASS_BASE, NUM_TILE_TYPES,
    norm_str, _tid_norm, dora_kind_of,
    st_tsumo, st_dahai, st_riichi, st_riichi_accepted,
    st_chi, st_pon, st_daiminkan, st_ankan, st_kakan, st_dora,
)

WIND_STR = ["E", "S", "W", "N"]


def make_full_deck() -> list[str]:
    base = [f"{n}{s}" for s in "mps" for n in range(1, 10)] * 4 + [f"{n}z" for n in range(1, 8)] * 4
    for src, dst in (("5m", "0m"), ("5p", "0p"), ("5s", "0s")):
        base[base.index(src)] = dst
    return base


def tile_id(pai: str) -> int:
    return TILE_STR_TO_ID.get(pai, 0)


normalize = norm_str


# ---------------------------------------------------------------------------
# 和了形判定（門前部分のみ。副露ぶんは枚数が減っているだけで同じロジックで通る）
# ---------------------------------------------------------------------------
def _count(hand):
    cnt = {}
    for t in hand:
        n = normalize(t); cnt[n] = cnt.get(n, 0) + 1
    return cnt


def _ext(cnt):
    if not cnt:
        return True
    t = sorted(cnt.keys())[0]
    if cnt[t] >= 3:
        c2 = dict(cnt); c2[t] -= 3
        if c2[t] == 0: del c2[t]
        if _ext(c2): return True
    s = t[-1]
    if s in "mps":
        n = int(t[0])
        t2, t3 = f"{n+1}{s}", f"{n+2}{s}"
        if n <= 7 and cnt.get(t2, 0) and cnt.get(t3, 0):
            c2 = dict(cnt)
            for tx in (t, t2, t3):
                c2[tx] -= 1
                if c2[tx] == 0: del c2[tx]
            if _ext(c2): return True
    return False


def _can_tsumo(hand) -> bool:
    cnt = _count(hand)
    if len(hand) == 14:
        ks = {"1m","9m","1p","9p","1s","9s","1z","2z","3z","4z","5z","6z","7z"}
        if ks.issubset(cnt) and any(cnt[t] >= 2 for t in ks): return True
        if len(cnt) == 7 and all(v == 2 for v in cnt.values()): return True
    if len(hand) % 3 != 2:
        return False
    for p in sorted(cnt):
        if cnt[p] >= 2:
            c2 = dict(cnt); c2[p] -= 2
            if c2[p] == 0: del c2[p]
            if _ext(c2): return True
    return False


_ALL34 = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + [f"{n}z" for n in range(1, 8)]


def _waits(hand13) -> set:
    return {t for t in _ALL34 if _can_tsumo(list(hand13) + [t])}


# ---------------------------------------------------------------------------
# モデル推論
# ---------------------------------------------------------------------------
@torch.no_grad()
def _logits_for(model, sample, device) -> torch.Tensor:
    was_training = model.training
    if was_training:
        model.eval()
    try:
        b = samples_to_batch([sample], device)
        logits = model(b["sparse"], b["numeric"], b["progression"], b["candidates"], b["cand_feats"])[0]
        return logits.masked_fill(b["candidates"][0] == PADDING_ACTION_ID, float("-inf"))
    finally:
        if was_training:
            model.train()


def model_choose(model, sample, device):
    """greedy。戻り値: (token, idx, probs(list), possible_actions)"""
    masked = _logits_for(model, sample, device)
    idx = int(masked.argmax().item())
    probs = F.softmax(masked, dim=-1)
    return sample["possible_actions"][idx], idx, probs.cpu().tolist(), sample["possible_actions"]


def model_choose_rl(model, sample, device, temperature=1.0, forced_idx=None, greedy=False):
    """戻り値: (token, idx, log_prob, log_probs(list, 温度適用後), entropy)"""
    masked = _logits_for(model, sample, device)
    log_probs = F.log_softmax(masked / max(temperature, 1e-6), dim=-1)
    probs = log_probs.exp()
    if forced_idx is not None and forced_idx >= 0:
        idx = int(forced_idx)
    elif greedy:
        idx = int(masked.argmax().item())
    else:
        idx = int(torch.multinomial(probs, 1).item())
    ent = float(-(probs * torch.nan_to_num(log_probs, neginf=0.0)).sum().item())
    return sample["possible_actions"][idx], idx, float(log_probs[idx].item()), log_probs.cpu().tolist(), ent


def load_model(path, device):
    return load_policy(path, device)


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------
@dataclass
class KyokuCtx:
    """1局ぶんのシミュレータ状態。deepcopy すればその時点から再開できる。"""
    state: KyokuState
    yama: list
    rinshan: list
    extra_dora: list
    ura_markers: list
    extra_ura: list
    actor: int
    phase: str = "draw"              # draw / after_draw / discard
    cur_tsumo: str | None = None
    is_rinshan: bool = False
    turn: int = 0
    kan_count: int = 0
    no_calls: bool = True            # まだ誰も鳴いていない（ダブリー/天和/地和の条件）
    perm_furiten: list = field(default_factory=lambda: [False] * 4)
    temp_furiten: list = field(default_factory=lambda: [False] * 4)
    ippatsu: list = field(default_factory=lambda: [False] * 4)
    double_riichi: list = field(default_factory=lambda: [False] * 4)
    start_scores: list = field(default_factory=lambda: [25000] * 4)
    kyoku_id: int = 0
    result: dict | None = None


@dataclass
class StepRecord:
    """rl_player の打牌判断1回ぶん。snapshot は「その判断の直前」の KyokuCtx。"""
    sample: dict
    action_idx: int
    log_prob: float
    log_probs: list
    entropy: float
    n_valid: int
    player: int
    kyoku_id: int
    under_threat: bool = False
    snapshot: KyokuCtx | None = None


@dataclass
class RLContext:
    rl_player: int = 0
    mode: str = "capture"            # capture: 記録しながら1半荘 / playout: スナップショットから1局
    temperature: float = 1.0
    capture_greedy: bool = False     # capture 中の rl_player を greedy にするか（既定はサンプリング）
    forced_first_idx: int = -1       # playout: 最初の判断で強制する候補 index
    continuation: str = "greedy"     # playout: 2手目以降  greedy / sample
    take_snapshots: bool = True
    records: list = field(default_factory=list)
    branch_record: StepRecord | None = None
    _forced_done: bool = False


def kyoku_index(bakaze: str, kyoku: int) -> int:
    return WIND_STR.index(bakaze) * 4 + (kyoku - 1)


def next_game_state(bk: int, kyoku: int, honba: int, oya: int, scores: list, res: dict):
    """
    1局の結果から次局を決める。戻り値 (finished, bk, kyoku, honba, oya)。
    天鳳の東南戦: 飛び終了 / 南4(以降)で 30000 以上がいれば終了・いなければ西入(サドンデス) /
    オーラスの親がトップで連荘権を得たら和了止め・聴牌止め。
    """
    renchan = bool(res["renchan"])
    honba2 = honba + 1 if (renchan or res.get("is_draw")) else 0
    if any(s < 0 for s in scores):
        return True, bk, kyoku, honba2, oya
    last = (bk * 4 + kyoku - 1) >= 7
    top = max(range(4), key=lambda i: (scores[i], -i))
    if last and max(scores) >= 30000:
        if not renchan or top == oya:
            return True, bk, kyoku, honba2, oya
    if renchan:
        return False, bk, kyoku, honba2, oya
    oya2 = (oya + 1) % 4; kyoku2 = kyoku + 1; bk2 = bk
    if kyoku2 > 4:
        kyoku2 = 1; bk2 += 1
    if bk2 >= 3:                       # 西4 まで打っても決着しなければ終了
        return True, bk, kyoku, honba2, oya
    return False, bk2, kyoku2, honba2, oya2


# ---------------------------------------------------------------------------
# シミュレータ
# ---------------------------------------------------------------------------
class MahjongSimulator:

    def __init__(self, model, device, watch=False, meld_model=None, riichi_model=None):
        self.model = model
        self.player_models = None       # {seat: model}。None の席は self.model
        self.device = device
        self.watch = watch
        self.tokenizer = MahjongTokenizer()
        self.names = ["AI-0", "AI-1", "AI-2", "AI-3"]
        self.rl: RLContext | None = None
        self._kyoku_counter = 0
        self._eval_hook = None          # (state, player, riichi_flags, 打牌) を受け取る評価用フック
        self._sample_log = None         # list を入れると、モデルに見せた全サンプルを記録（整合性テスト用）
        # 対人プレイ用フック: decision_provider(player, sample, context) -> token or None
        self.decision_provider = None
        self.human_player = None

    # ---- モデル選択 ----
    def _model_for(self, player):
        if self.player_models is not None and self.player_models.get(player) is not None:
            return self.player_models[player]
        return self.model

    def _is_human(self, player):
        return (self.decision_provider is not None and self.human_player is not None
                and player == self.human_player)

    def _choose(self, player, sample, kind, context=None):
        if self._sample_log is not None:
            self._sample_log.append(sample)
        if self._is_human(player):
            ctx = dict(context or {}); ctx["kind"] = kind
            tok = self.decision_provider(player, sample, ctx)
            if tok is not None:
                return tok
        return model_choose(self._model_for(player), sample, self.device)[0]

    def _ask_agari(self, player, kind, state, win_pai=None, target=None):
        if self._is_human(player):
            ctx = {"kind": "agari", "agari_kind": kind, "state": state, "win_pai": win_pai, "target": target}
            return bool(self.decision_provider(player, {"agari": True}, ctx))
        return True

    # ---- 半荘 ----
    def _play_game(self, fixed_decks=None):
        events = [{"type": "start_game", "names": self.names, "kyoku_first": 0, "aka_flag": True}]
        scores = [25000] * 4; oya = 0; bk = 0; kyoku = 1; honba = 0; kyotaku = 0
        self._kyoku_counter = 0
        per_kyoku = []
        while True:
            deck = fixed_decks.get(self._kyoku_counter) if fixed_decks else None
            K = self._new_kyoku(WIND_STR[min(bk, 3)], kyoku, honba, kyotaku, oya, scores, events, deck)
            res = self._run_kyoku(K, events)
            scores = list(res["scores"]); kyotaku = res["kyotaku_next"]
            per_kyoku.append((K.kyoku_id, list(res["score_delta"])))
            finished, bk, kyoku, honba, oya = next_game_state(bk, kyoku, honba, oya, scores, res)
            if finished:
                break
        if kyotaku:                                   # 残った供託はトップへ
            top = max(range(4), key=lambda i: (scores[i], -i))
            scores[top] += kyotaku * 1000
        events.append({"type": "end_game", "scores": scores})
        return events, per_kyoku

    def simulate_game(self) -> list[dict]:
        return self._play_game()[0]

    def capture_game(self, rl: RLContext, seed=None):
        """rl_player の全打牌判断を StepRecord（スナップショット付き）として記録しながら1半荘打つ。"""
        if seed is not None:
            random.seed(seed)
        rl.mode = "capture"; rl.records = []
        self.rl = rl
        try:
            events, per_kyoku = self._play_game()
        finally:
            self.rl = None
        return events, per_kyoku

    def playout_from(self, snapshot: KyokuCtx, rl: RLContext, max_turns: int | None = None):
        """
        スナップショット（rl_player の判断直前）から打つ。

        max_turns を渡すと、そこから max_turns 巡で打ち切り、局面価値（kyoku_value の
        期待局収支）で決着を推定した疑似 result を返す。res["truncated"] が True になる。
        打ち切り前に局が終われば、通常どおり実際の結果を返す（truncated は False）。
        """
        K = copy.deepcopy(snapshot)
        rl.mode = "playout"; rl._forced_done = False; rl.branch_record = None
        self.rl = rl
        events: list = []
        try:
            res = self._run_kyoku(K, events, stop_turn=(None if max_turns is None
                                                        else K.turn + max_turns))
        finally:
            self.rl = None
        return res, events

    # ---- 局の準備 ----
    def _new_kyoku(self, bakaze, kyoku, honba, kyotaku, oya, scores, events, deck=None) -> KyokuCtx:
        if deck is None:
            deck = make_full_deck(); random.shuffle(deck)
        deck = list(deck)
        tehais = [deck[i * 13:(i + 1) * 13] for i in range(4)]
        dead = deck[122:]
        state = KyokuState(bakaze=bakaze, kyoku=kyoku, honba=honba, kyotaku=kyotaku, oya=oya,
                           dora_markers=[dead[4]], tehais=[list(h) for h in tehais], scores=list(scores))
        K = KyokuCtx(state=state, yama=list(deck[52:122]), rinshan=list(dead[0:4]),
                     extra_dora=list(dead[5:9]), ura_markers=[dead[9]], extra_ura=list(dead[10:14]),
                     actor=oya, start_scores=list(scores), kyoku_id=self._kyoku_counter)
        self._kyoku_counter += 1
        events.append({"type": "start_kyoku", "bakaze": bakaze, "kyoku": kyoku, "honba": honba,
                       "kyotaku": kyotaku, "oya": oya, "dora_marker": dead[4],
                       "scores": list(scores), "tehais": [list(h) for h in tehais]})
        if self.watch:
            print(f"\n{'='*55}\n  {bakaze}{kyoku}局 {honba}本場  ドラ表示:{dead[4]}  点数:{scores}")
        return K

    # ---- 局の進行（状態機械） ----
    def _run_kyoku(self, K: KyokuCtx, events: list, stop_turn: int | None = None) -> dict:
        st = K.state
        while K.result is None:
            if stop_turn is not None and K.turn >= stop_turn:
                return self._truncated_result(K, events)
            if K.phase == "draw":
                if not K.yama:
                    self._ryukyoku(K, events); break
                pai = K.yama.pop(0)
                st_tsumo(st, K.actor, pai)
                events.append({"type": "tsumo", "actor": K.actor, "pai": pai})
                K.temp_furiten[K.actor] = False
                K.cur_tsumo = pai; K.is_rinshan = False; K.phase = "after_draw"

            elif K.phase == "after_draw":
                a = K.actor
                if _can_tsumo(st.tehais[a]):
                    flags = self._win_flags(K, a, True)
                    if self._has_yaku(K, a, K.cur_tsumo, True, flags) and \
                            self._ask_agari(a, "tsumo", st, win_pai=K.cur_tsumo):
                        self._hora(K, events, a, None, K.cur_tsumo, flags); break
                if self._try_kan(K, events):
                    continue
                K.phase = "discard"

            elif K.phase == "discard":
                self._discard_and_calls(K, events)

        events.append({"type": "end_kyoku"})
        res = K.result
        res["scores"] = list(st.scores)
        res["score_delta"] = [st.scores[i] - K.start_scores[i] for i in range(4)]
        return res

    # ---- 槓 ----
    def _rinshan_draw(self, K, events, actor):
        st = K.state
        K.kan_count += 1; K.no_calls = False; K.ippatsu = [False] * 4
        if K.extra_dora:
            nd = K.extra_dora.pop(0); st_dora(st, nd)
            events.append({"type": "dora", "dora_marker": nd})
            if K.extra_ura:
                K.ura_markers.append(K.extra_ura.pop(0))
        rin = K.rinshan.pop(0)
        if K.yama:
            K.yama.pop()                       # 海底側の1枚が王牌に回る（ツモ山が1枚減る）
        st_tsumo(st, actor, rin)
        events.append({"type": "tsumo", "actor": actor, "pai": rin})
        K.actor = actor; K.cur_tsumo = rin; K.is_rinshan = True; K.phase = "after_draw"

    def _try_kan(self, K, events) -> bool:
        st, a = K.state, K.actor
        if not K.rinshan or K.kan_count >= 4 or not K.yama:
            return False
        cands = self.tokenizer.enumerate_kan_candidates(st, a)
        if len(cands) <= 1:
            return False
        sample = self.tokenizer.make_sample(st, a, [c[0] for c in cands], 0, "kan_decision")
        chosen = self._choose(a, sample, "kan", context={
            "candidates": sample["possible_actions"], "state": st,
            "ankan_cands": [c[2] for c in cands if c[1] == "ankan"],
            "kakan_cands": [c[2] for c in cands if c[1] == "kakan"]})
        pick = next((c for c in cands if c[0] == chosen), None)
        if pick is None or pick[1] == "pass":
            return False
        _tok, ktype, n = pick
        hand = st.tehais[a]
        if ktype == "ankan":
            consumed = [t for t in hand if normalize(t) == n][:4]
            st_ankan(st, a, consumed)
            events.append({"type": "ankan", "actor": a, "consumed": consumed})
        else:
            added = next(t for t in hand if normalize(t) == n)
            pon_m = next(m for m in st.meld_info[a] if m["type"] == "pon" and normalize(m["pai"]) == n)
            events.append({"type": "kakan", "actor": a, "pai": added,
                           "consumed": list(pon_m["consumed"]) + [pon_m["pai"]]})
            if self._check_ron(K, events, a, added, is_chankan=True):   # 搶槓
                return True
            st_kakan(st, a, added)
        if self.watch: print(f"  P{a} {ktype}:{n}")
        self._rinshan_draw(K, events, a)
        return True

    # ---- 打牌と鳴き ----
    def _decide_dahai(self, K, a, sample):
        """戻り値: (token, idx, infer_dict)"""
        rl = self.rl
        poss = sample["possible_actions"]
        if self._sample_log is not None:
            self._sample_log.append(sample)
        if rl is not None and a == rl.rl_player:
            n_valid = sum(1 for t in poss if t != PADDING_ACTION_ID)
            forced, greedy, is_branch = None, False, False
            if rl.mode == "capture":
                greedy = rl.capture_greedy
            elif not rl._forced_done:
                forced = rl.forced_first_idx; rl._forced_done = True; is_branch = True
            else:
                greedy = (rl.continuation == "greedy")
            token, idx, lp, lps, ent = model_choose_rl(
                self.model, sample, self.device, temperature=rl.temperature,
                forced_idx=forced, greedy=greedy)
            if rl.mode == "capture" or is_branch:
                st = K.state
                rec = StepRecord(sample=sample, action_idx=idx, log_prob=lp, log_probs=lps, entropy=ent,
                                 n_valid=n_valid, player=a, kyoku_id=K.kyoku_id,
                                 under_threat=any(st.riichi_flags[p] for p in range(4) if p != a))
                if rl.mode == "capture":
                    if rl.take_snapshots and n_valid >= 2:
                        rec.snapshot = copy.deepcopy(K)
                    rl.records.append(rec)
                else:
                    rl.branch_record = rec
            return token, idx, {"pred_idx": idx, "pred_token": token, "probs": None, "poss": poss}
        if self._is_human(a):
            st = K.state
            ctx = {"kind": "dahai", "candidates": poss, "state": st, "tsumo_pai": K.cur_tsumo,
                   "can_riichi": any(ACT_RIICHI_BASE <= t < ACT_PON_BASE for t in poss),
                   "already_riichi": False}
            token = self.decision_provider(a, sample, ctx)
            if token is not None and token in poss:
                idx = poss.index(token)
                return token, idx, {"pred_idx": idx, "pred_token": token, "probs": None, "poss": poss}
        token, idx, probs, _ = model_choose(self._model_for(a), sample, self.device)
        return token, idx, {"pred_idx": idx, "pred_token": token, "probs": probs, "poss": poss}

    @staticmethod
    def _pick_tile(hand, kind34: int) -> str:
        """牌種 → 手牌の実牌。赤と通常が両方あれば通常牌を切る。"""
        same = [t for t in hand if _tid_norm(tile_id(t)) == kind34]
        normal = [t for t in same if t not in RED_STRS]
        return (normal or same)[0]

    def _discard_and_calls(self, K, events):
        st, a = K.state, K.actor
        do_riichi = False
        if st.riichi_flags[a]:
            pai = K.cur_tsumo                                   # リーチ後はツモ切り固定
            infer = None
        else:
            acts = self.tokenizer.enumerate_dahai_actions(st, a)
            sample = self.tokenizer.make_sample(st, a, acts, 0, "dahai")
            token, _idx, infer = self._decide_dahai(K, a, sample)
            do_riichi = ACT_RIICHI_BASE <= token < ACT_PON_BASE
            pai = self._pick_tile(st.tehais[a], token % NUM_TILE_TYPES)
            if self._eval_hook is not None:
                try:
                    self._eval_hook(st, a, st.riichi_flags, pai)
                except Exception:
                    pass
        tsumogiri = K.cur_tsumo is not None and pai == K.cur_tsumo
        if do_riichi:
            K.double_riichi[a] = K.no_calls and not st.sutehais[a]
            events.append({"type": "reach", "actor": a})
            st_riichi(st, a)
            if self.watch: print(f"  P{a} リーチ")
        else:
            K.ippatsu[a] = False
        ev = {"type": "dahai", "actor": a, "pai": pai, "tsumogiri": tsumogiri}
        if infer is not None:
            infer.update({"decoded_pai": pai, "actual_pai": pai, "kuikae_overridden": False})
            ev["_infer"] = infer
        st_dahai(st, a, pai, tsumogiri)
        events.append(ev)
        if self.watch: print(f"  P{a} 打:{pai}" + (" (ツモ切り)" if tsumogiri else ""))

        if self._check_ron(K, events, a, pai):
            return
        if do_riichi:
            st_riichi_accepted(st, a)
            K.ippatsu[a] = True
            events.append({"type": "reach_accepted", "actor": a, "scores": list(st.scores)})

        call = self._check_meld(K, events, a, pai)
        K.cur_tsumo = None
        if call is None:
            K.actor = (a + 1) % 4; K.phase = "draw"; K.turn += 1
            return
        pi, mtype = call
        K.no_calls = False; K.ippatsu = [False] * 4
        if mtype == "daiminkan":
            self._rinshan_draw(K, events, pi)
        else:
            K.actor = pi; K.phase = "discard"

    def _check_ron(self, K, events, discarder, pai, is_chankan=False) -> bool:
        st = K.state
        for off in (1, 2, 3):
            pi = (discarder + off) % 4
            if not _can_tsumo(st.tehais[pi] + [pai]):
                continue
            wait_norms = {normalize(w) for w in _waits(st.tehais[pi])}
            flags = self._win_flags(K, pi, False, is_chankan=is_chankan)
            own_disc = {_tid_norm(t) for t in st.sutehais[pi]}
            furiten = (any(_tid_norm(tile_id(w)) in own_disc for w in wait_norms)
                       or K.perm_furiten[pi] or K.temp_furiten[pi])
            can = (not furiten) and self._has_yaku(K, pi, pai, False, flags)
            if can and self._ask_agari(pi, "ron", st, win_pai=pai, target=discarder):
                self._hora(K, events, pi, discarder, pai, flags)
                return True
            # 和了牌を見逃した（フリテン・役なし・任意の見逃し）→ 同巡フリテン。リーチ中は永久。
            K.temp_furiten[pi] = True
            if st.riichi_flags[pi]:
                K.perm_furiten[pi] = True
        return False

    def _check_meld(self, K, events, discarder, pai):
        st = K.state
        elig = []
        for off in (1, 2, 3):
            pi = (discarder + off) % 4
            opts = self.tokenizer.call_options(st, pi, pai, discarder)
            if opts["pon"] or opts["chi"] or opts["daiminkan"]:
                elig.append((pi, opts))
        elig.sort(key=lambda e: 0 if (e[1]["daiminkan"] or e[1]["pon"]) else 1)   # ポン/槓が優先
        for pi, opts in elig:
            if opts["daiminkan"] and (not K.rinshan or K.kan_count >= 4):
                opts = dict(opts); opts["daiminkan"] = False
            cands = self.tokenizer.enumerate_meld_candidates(
                st.tehais[pi], pai, opts["pon"], opts["chi"], opts["daiminkan"])
            if len(cands) <= 1:
                continue
            sample = self.tokenizer.make_sample(st, pi, [c[0] for c in cands], 0, "meld_decision")
            chosen = self._choose(pi, sample, "meld", context={
                "candidates": cands, "state": st, "dahai_pai": pai, "dahai_player": discarder})
            pick = next((c for c in cands if c[0] == chosen), None)
            if pick is None or pick[1] == "skip":
                continue
            _tok, mtype, consumed = pick
            events.append({"type": mtype, "actor": pi, "target": discarder, "pai": pai,
                           "consumed": list(consumed)})
            {"pon": st_pon, "chi": st_chi, "daiminkan": st_daiminkan}[mtype](st, pi, pai, consumed)
            if self.watch: print(f"  P{pi} {mtype}:{pai} {consumed}")
            return pi, mtype
        return None

    # ---- 和了・流局 ----
    def _win_flags(self, K, p, is_tsumo, is_chankan=False) -> dict:
        st = K.state
        first = K.no_calls and not st.sutehais[p]
        return {
            "is_ippatsu": bool(K.ippatsu[p]), "is_double_riichi": bool(K.double_riichi[p]),
            "is_haitei": is_tsumo and not K.yama and not K.is_rinshan,
            "is_houtei": (not is_tsumo) and not K.yama and not is_chankan,
            "is_rinshan": is_tsumo and K.is_rinshan, "is_chankan": is_chankan,
            "is_tenhou": is_tsumo and first and p == st.oya and not K.is_rinshan,
            "is_chiihou": is_tsumo and first and p != st.oya and not K.is_rinshan,
        }

    def _evaluate(self, K, p, win_tile, is_tsumo, flags) -> dict:
        st = K.state
        hand = list(st.tehais[p])
        if is_tsumo and win_tile in hand:
            hand.remove(win_tile)
        return _yaku.evaluate_hand(
            hand, st.meld_info[p], win_tile, is_tsumo, is_riichi=st.riichi_flags[p],
            bakaze=st.bakaze, jikaze=WIND_STR[(p - st.oya) % 4], **flags)

    def _has_yaku(self, K, p, win_tile, is_tsumo, flags) -> bool:
        r = self._evaluate(K, p, win_tile, is_tsumo, flags)
        return r["yakuman"] > 0 or len(r["yaku"]) > 0

    @staticmethod
    def _count_dora(tiles, markers) -> int:
        kinds = [dora_kind_of(tile_id(m)) for m in markers]
        return sum(kinds.count(_tid_norm(tile_id(t))) for t in tiles)

    @staticmethod
    def _base_points(fu, han, yakuman) -> int:
        if yakuman:  return 8000 * yakuman
        if han >= 13: return 8000
        if han >= 11: return 6000
        if han >= 8:  return 4000
        if han >= 6:  return 3000
        return min(fu * (2 ** (han + 2)), 2000)

    def _hora(self, K, events, winner, loser, win_tile, flags):
        st = K.state
        is_tsumo = loser is None
        r = self._evaluate(K, winner, win_tile, is_tsumo, flags)
        tiles = list(st.tehais[winner]) + ([] if is_tsumo else [win_tile])
        for m in st.meld_info[winner]:
            tiles += list(m.get("consumed") or [])
            if m.get("pai"):   tiles.append(m["pai"])
            if m.get("added"): tiles.append(m["added"])
        ura = K.ura_markers[:1 + K.kan_count] if st.riichi_flags[winner] else []
        n_dora = (self._count_dora(tiles, st.dora_markers) + self._count_dora(tiles, ura)
                  + sum(1 for t in tiles if t in RED_STRS))
        han = r["han"] + (0 if r["yakuman"] else n_dora)
        base = self._base_points(r["fu"], han, r["yakuman"])
        up = lambda x: math.ceil(x / 100) * 100
        delta = [0] * 4
        if is_tsumo:
            for i in range(4):
                if i == winner: continue
                pay = up(base * (2 if (i == st.oya or winner == st.oya) else 1)) + st.honba * 100
                delta[i] -= pay; delta[winner] += pay
        else:
            pay = up(base * (6 if winner == st.oya else 4)) + st.honba * 300
            delta[loser] -= pay; delta[winner] += pay
        delta[winner] += st.kyotaku * 1000
        st.scores = [st.scores[i] + delta[i] for i in range(4)]
        st.kyotaku = 0
        events.append({"type": "hora", "actor": winner, "target": loser, "pai": win_tile,
                       "score": delta[winner], "deltas": delta, "scores": list(st.scores),
                       "ura_markers": list(ura), "yaku": [n for n, _ in r["yaku"]],
                       "han": han, "fu": r["fu"], "yakuman": r["yakuman"]})
        if self.watch:
            print(f"  🎉 P{winner} {'ツモ' if is_tsumo else f'ロン←P{loser}'} {r['yaku']} "
                  f"{han}翻{r['fu']}符 +{delta[winner]}")
        K.result = {"renchan": winner == st.oya, "kyotaku_next": 0, "is_draw": False,
                    "winner": winner, "loser": loser}

    def _check_tenpai(self, hand) -> bool:
        return any(_can_tsumo(list(hand) + [t]) for t in _ALL34)

    def _truncated_result(self, K, events) -> dict:
        """打ち切り地点の局面を期待局収支に置き換えた疑似 result。点棒は動かさない。"""
        from kyoku_value import estimate_score_delta
        st = K.state
        delta = estimate_score_delta(st)
        est = [st.scores[i] + delta[i] for i in range(4)]
        events.append({"type": "truncated", "turn": K.turn})
        return {"renchan": False, "kyotaku_next": st.kyotaku, "is_draw": False,
                "winner": None, "loser": None, "truncated": True,
                "scores": est,
                "score_delta": [est[i] - K.start_scores[i] for i in range(4)]}

    def _ryukyoku(self, K, events):
        st = K.state
        tenpai = [self._check_tenpai(st.tehais[i]) for i in range(4)]
        n = sum(tenpai)
        if 0 < n < 4:
            for i in range(4):
                st.scores[i] += (3000 // n) if tenpai[i] else -(3000 // (4 - n))
        events.append({"type": "ryukyoku", "tenpai": tenpai, "scores": list(st.scores)})
        if self.watch: print(f"  💨 流局 テンパイ:{[i for i, t in enumerate(tenpai) if t]}")
        K.result = {"renchan": bool(tenpai[st.oya]), "kyotaku_next": st.kyotaku, "is_draw": True,
                    "winner": None, "loser": None, "tenpai": tenpai}


def resample_wall(K: KyokuCtx, rng: random.Random) -> KyokuCtx:
    """
    まだ誰にも見えていない山（ツモ山＋嶺上＋未公開のドラ表示/裏ドラ）を引き直した複製を返す。
    他家の手牌は真の手牌のまま固定する。他家の手牌まで引き直すとリーチ者がテンパイでなくなり
    （放銃が起きなくなる）、「危険牌を押す」行動が不当に高く評価されるため。
    """
    K2 = copy.deepcopy(K)
    sizes = [len(K2.yama), len(K2.rinshan), len(K2.extra_dora), len(K2.ura_markers), len(K2.extra_ura)]
    pool = K2.yama + K2.rinshan + K2.extra_dora + K2.ura_markers + K2.extra_ura
    rng.shuffle(pool)
    parts, i = [], 0
    for s in sizes:
        parts.append(pool[i:i + s]); i += s
    K2.yama, K2.rinshan, K2.extra_dora, K2.ura_markers, K2.extra_ura = parts
    return K2


# ---------------------------------------------------------------------------
# エントリポイント（牌譜生成）
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("replays"))
    p.add_argument("--watch", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    if args.seed is not None:
        random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sim = MahjongSimulator(load_policy(args.model, device), device, watch=args.watch)
    args.out.mkdir(parents=True, exist_ok=True)
    for g in range(args.games):
        events = sim.simulate_game()
        path = args.out / f"replay_{g:04d}.mjson"
        path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events), encoding="utf-8")
        print(f"{path}  最終点数: {events[-1]['scores']}")


if __name__ == "__main__":
    main()
