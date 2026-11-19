"""
human_game.py ― 人間 vs AI の対局セッション管理

設計の要点
-----------
self_play.MahjongSimulator の局進行（_play_game / _run_kyoku）は「最後まで一気に回す」
実装で、途中で止めて入力を挟む構造になっていない。これをジェネレータへ分解すると
槓・鳴き・ロンの複雑な分岐を壊すリスクが高い。

そこで simulator を **バックグラウンドスレッド** で走らせ、人間席の意思決定
（decision_provider）に来たら **Queue でブロック** して HTTP からの入力を待つ、
という方式を採る。これにより既存の局進行ロジックには一切手を入れず、
「人間の手番だけモデルの代わりに人間が選ぶ」を実現する。

公開する状態は server.py が infer_game で使うのと同じ「イベント列」で、
UI は既存の牌譜ビューアーのレンダリングをそのまま流用できる。

人間が選ぶもの
---------------
- 打牌・リーチ宣言・ポン/チー/明槓・暗槓/加槓
- ツモ和了・ロンをするか見逃すか（見逃すとフリテンになる）

局の進行と半荘の終了条件は AI 同士の対局と同じ（self_play._play_game をそのまま使う）。
"""

from __future__ import annotations
import sys
import threading
import queue
import uuid
import time
from typing import Optional

import torch

from self_play import MahjongSimulator
from mahjong_tokenizer import (
    PADDING_ACTION_ID,
    ACT_DAHAI_BASE, ACT_RIICHI_BASE, ACT_PON_BASE, ACT_CHI_BASE,
    ACT_DAIMINKAN_BASE, ACT_ANKAN_BASE, ACT_KAKAN_BASE,
    ACT_TSUMO_BASE, ACT_PASS_BASE,
)


class _GameOver(Exception):
    """セッション破棄時にワーカースレッドを速やかに畳むための内部シグナル。"""


class HumanGameSession:
    """
    1 つの人間 vs AI 対局を保持するセッション。

    フロー:
      1. start() でワーカースレッドが simulate_game を回し始める。
      2. AI だけで進む間はイベントがどんどん積まれる。
      3. 人間席の意思決定に来ると decision_provider がブロックし、
         pending（人間が今選ぶべき候補）をセットして待つ。
      4. server が /human/state で pending を返し、UI が選択肢を提示。
      5. UI が /human/action でトークンを送ると submit() が解放し、
         ワーカーが続行する。
    """

    def __init__(self, model, device, human_player: int = 0,
                 names: Optional[list[str]] = None):
        self.id = uuid.uuid4().hex[:12]
        self.model = model
        self.device = device
        self.human_player = human_player

        self.sim = MahjongSimulator(model, device, watch=False)
        self.sim.human_player = human_player
        self.sim.decision_provider = self._on_decision
        print(f"[human_game] セッション作成 id={self.id} human_player={human_player}", flush=True)
        if names:
            self.sim.names = list(names)
        else:
            nm = ["AI-0", "AI-1", "AI-2", "AI-3"]
            nm[human_player] = "あなた"
            self.sim.names = nm

        self._events: list[dict] = []          # 生成済みイベント（mjai形式）
        self._events_lock = threading.Lock()
        self._pending: Optional[dict] = None   # 人間が今選ぶべき候補（None=待ちなし）
        self._answer_q: "queue.Queue[int]" = queue.Queue(maxsize=1)
        self._finished = False
        self._error: Optional[str] = None
        self._killed = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ 起動
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        # self_play._play_game にライブ更新される list を渡し、生成と同時にセッションへ反映する。
        # 半荘の進め方（連荘・西入・飛び終了など）は AI 同士の対局と完全に同じ。
        try:
            live = _LiveList(self._on_event_appended)
            self.sim._play_game(events=live)
            self._finished = True
        except _GameOver:
            pass
        except Exception as e:  # noqa
            import traceback
            tb = traceback.format_exc()
            self._error = f"{e}\n{tb}"
            self._finished = True
            # サーバーのターミナルにも必ず出す（原因追跡用）
            print(f"[human_game] ワーカー例外 session={self.id}:\n{tb}", flush=True)

    def _on_event_appended(self, ev):
        # イベントが 1 つ積まれるたびに呼ばれる。ロックして自分のリストへ複製。
        with self._events_lock:
            self._events.append(ev)

    # ------------------------------------------- 人間の意思決定（ワーカー側）
    def _on_decision(self, player, sample, ctx) -> Optional[int]:
        """
        decision_provider 本体。人間席の手番でワーカースレッドから呼ばれる。
        pending をセットして人間の入力を待ち、返ってきたトークンを返す。
        """
        if self._killed:
            raise _GameOver()

        kind = ctx.get("kind")
        pending = self._build_pending(player, sample, ctx)
        if pending is None:
            # 選択肢が実質1つ→自動でその候補（＝スキップ等）。人間に聞かない。
            return None

        self._pending = pending
        # 人間の回答を待つ（ブロッキング）
        while True:
            try:
                tok = self._answer_q.get(timeout=0.2)
                break
            except queue.Empty:
                if self._killed:
                    raise _GameOver()
        self._pending = None
        return tok

    def _build_pending(self, player, sample, ctx) -> Optional[dict]:
        """UI に見せる「人間が今選ぶべき候補」を組み立てる。"""
        kind = ctx["kind"]

        if kind == "agari":
            # 和了する(1) / 見逃す(0) の二択。ツモ/ロンで文言を変える。
            ak = ctx.get("agari_kind")
            label_win = "ツモ和了する" if ak == "tsumo" else "ロン和了する"
            return {"kind": "agari", "player": player, "agari_kind": ak,
                    "win_pai": ctx.get("win_pai"), "target": ctx.get("target"),
                    "options": [
                        {"token": 1, "action": "agari", "label": f"🎉 {label_win}"},
                        {"token": 0, "action": "skip",  "label": "見逃す"},
                    ]}

        if kind == "dahai":
            # 既にリーチ済みなら打牌は強制（ツモ切り）。人間に選ばせず自動処理。
            if ctx.get("already_riichi"):
                return None
            poss = [t for t in ctx["candidates"] if t != PADDING_ACTION_ID]
            options = []
            for t in poss:
                options.append(self._decode_token_for_ui(t))
            return {"kind": "dahai", "player": player, "options": options,
                    "can_riichi": bool(ctx.get("can_riichi"))}

        if kind == "meld":
            # cands: [(token, mtype, consumed_list), ...] 先頭は skip
            cands = ctx["candidates"]
            if len(cands) <= 1:
                return None  # skip のみ→自動スキップ
            options = []
            for tok, mtype, consumed in cands:
                if mtype == "skip":
                    options.append({"token": tok, "action": "skip",
                                    "label": "スキップ", "consumed": []})
                else:
                    label = {"pon": "ポン", "chi": "チー",
                             "daiminkan": "明槓"}.get(mtype, mtype)
                    options.append({"token": tok, "action": mtype,
                                    "label": f"{label}",
                                    "pai": ctx.get("dahai_pai"),
                                    "consumed": list(consumed)})
            return {"kind": "meld", "player": player, "options": options,
                    "pai": ctx.get("dahai_pai")}

        if kind == "kan":
            poss = [t for t in ctx["candidates"] if t != PADDING_ACTION_ID]
            if len(poss) <= 1:
                return None
            options = []
            for t in poss:
                if t == ACT_PASS_BASE:
                    options.append({"token": t, "action": "skip",
                                    "label": "スキップ"})
                else:
                    options.append(self._decode_token_for_ui(t))
            return {"kind": "kan", "player": player, "options": options}

        return None

    def _decode_token_for_ui(self, token: int) -> dict:
        """トークンを UI 用ラベル・牌・アクション種別に変換。"""
        label = _token_label(token)
        pai = None
        action = "other"
        if ACT_DAHAI_BASE <= token < ACT_RIICHI_BASE:
            action = "dahai"; pai = _tile_from_tid(token - ACT_DAHAI_BASE)
        elif ACT_RIICHI_BASE <= token < ACT_PON_BASE:
            action = "riichi"; pai = _tile_from_tid(token - ACT_RIICHI_BASE)
        elif ACT_ANKAN_BASE <= token < ACT_KAKAN_BASE:
            action = "ankan"; pai = _tile_from_tid(token - ACT_ANKAN_BASE)
        elif ACT_KAKAN_BASE <= token < ACT_TSUMO_BASE:
            action = "kakan"; pai = _tile_from_tid(token - ACT_KAKAN_BASE)
        return {"token": token, "action": action, "label": label, "pai": pai}

    # ------------------------------------------------- 人間の回答（server側）
    def submit(self, token: int) -> bool:
        """UI から選択トークンを受け取る。pending が無ければ False。"""
        if self._pending is None:
            return False
        valid = {o["token"] for o in self._pending["options"]}
        if token not in valid:
            return False
        try:
            self._answer_q.put_nowait(int(token))
            return True
        except queue.Full:
            return False

    # -------------------------------------------------------------- 状態取得
    def snapshot(self) -> dict:
        with self._events_lock:
            events = list(self._events)
        return {
            "id": self.id,
            "human_player": self.human_player,
            "names": self.sim.names,
            "events": events,
            "pending": self._pending,
            "finished": self._finished,
            "error": self._error,
            "awaiting_human": self._pending is not None,
        }

    def kill(self):
        self._killed = True
        try:
            self._answer_q.put_nowait(-1)
        except Exception:
            pass


class _LiveList(list):
    """append されるたびにコールバックを呼ぶ list。"""
    def __init__(self, on_append):
        super().__init__()
        self._on_append = on_append

    def append(self, item):
        super().append(item)
        try:
            self._on_append(item)
        except Exception:
            pass


_AKA = {34: "5mr", 35: "5pr", 36: "5sr"}


def _tile_from_tid(tid: int) -> str:
    """牌 ID（0-33 が通常牌、34-36 が赤5）を mjai 表記に。"""
    if tid in _AKA:
        return _AKA[tid]
    for suit, count in [("m", 9), ("p", 9), ("s", 9), ("z", 7)]:
        if tid < count:
            return f"{tid+1}{suit}"
        tid -= count
    return "?"


def _token_label(token: int) -> str:
    """server.py の token_to_label を使う。server.py をスクリプトとして起動すると
    モジュール名が __main__ になるので、import し直して二重に初期化しないよう両方を探す。"""
    for name in ("server", "__main__"):
        m = sys.modules.get(name)
        if m is not None and hasattr(m, "token_to_label"):
            return m.token_to_label(token)
    return str(token)


# =========================================================================
# セッションレジストリ（server.py から使う）
# =========================================================================
class SessionRegistry:
    def __init__(self):
        self._sessions: dict[str, HumanGameSession] = {}
        self._lock = threading.Lock()

    def create(self, model, device, human_player=0, names=None) -> HumanGameSession:
        s = HumanGameSession(model, device, human_player=human_player, names=names)
        with self._lock:
            # 古いセッションが溜まりすぎないよう、終了済みを掃除
            dead = [k for k, v in self._sessions.items()
                    if v._finished and (time.time() - getattr(v, "_born", 0)) > 3600]
            for k in dead:
                self._sessions.pop(k, None)
            self._sessions[s.id] = s
        s._born = time.time()
        s.start()
        return s

    def get(self, sid: str) -> Optional[HumanGameSession]:
        with self._lock:
            return self._sessions.get(sid)

    def remove(self, sid: str):
        with self._lock:
            s = self._sessions.pop(sid, None)
        if s:
            s.kill()
