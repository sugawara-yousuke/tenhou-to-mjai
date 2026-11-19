"""
server.py  ― 改修版
各イベントに全局面状態（state）を付加して返す。
UIはstateをそのまま表示するだけで手牌計算不要。

[統合モデル化]
  打牌・鳴き・立直を単一の候補スコアリングモデル（MahjongTransformer）で扱う。
  旧 MeldDecisionModel / RiichiDecisionModel は廃止し、--model 1本で動作する。
"""

from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mahjong_model import load_policy, samples_to_batch
from mahjong_tokenizer import (
    MahjongTokenizer,
    PADDING_ACTION_ID,
    ACT_DAHAI_BASE, ACT_RIICHI_BASE, ACT_PON_BASE, ACT_CHI_BASE,
    ACT_DAIMINKAN_BASE, ACT_ANKAN_BASE, ACT_KAKAN_BASE,
    ACT_TSUMO_BASE, ACT_RON_BASE, ACT_PASS_BASE,
    NUM_ACTION_TYPES,
)

TILE_NAMES = [
    "1m","2m","3m","4m","5m","6m","7m","8m","9m",
    "1p","2p","3p","4p","5p","6p","7p","8p","9p",
    "1s","2s","3s","4s","5s","6s","7s","8s","9s",
    "東","南","西","北","白","発","中",
    "赤5m","赤5p","赤5s",
]

def token_to_label(token: int) -> str:
    if token == PADDING_ACTION_ID:        return "[PAD]"
    if token == ACT_PASS_BASE:            return "パス"
    if token == ACT_TSUMO_BASE:           return "ツモ和了"
    if token == ACT_RON_BASE:             return "ロン"
    if ACT_DAHAI_BASE     <= token < ACT_RIICHI_BASE:
        t = token - ACT_DAHAI_BASE
        return f"打牌 {TILE_NAMES[t] if t < len(TILE_NAMES) else t}"
    if ACT_RIICHI_BASE    <= token < ACT_PON_BASE:
        t = token - ACT_RIICHI_BASE
        return f"リーチ {TILE_NAMES[t] if t < len(TILE_NAMES) else t}"
    if ACT_PON_BASE       <= token < ACT_CHI_BASE:
        t = token - ACT_PON_BASE
        return f"ポン {TILE_NAMES[t] if t < len(TILE_NAMES) else t}"
    if ACT_CHI_BASE       <= token < ACT_DAIMINKAN_BASE:
        return f"チー({token - ACT_CHI_BASE})"
    if ACT_DAIMINKAN_BASE <= token < ACT_ANKAN_BASE:
        t = token - ACT_DAIMINKAN_BASE
        return f"明槓 {TILE_NAMES[t] if t < len(TILE_NAMES) else t}"
    if ACT_ANKAN_BASE     <= token < ACT_KAKAN_BASE:
        t = token - ACT_ANKAN_BASE
        return f"暗槓 {TILE_NAMES[t] if t < len(TILE_NAMES) else t}"
    if ACT_KAKAN_BASE     <= token < ACT_TSUMO_BASE:
        t = token - ACT_KAKAN_BASE
        return f"加槓 {TILE_NAMES[t] if t < len(TILE_NAMES) else t}"
    return f"?(token={token})"


app = FastAPI(title="MahjongTransformer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_model:  Optional[MahjongTransformer]   = None
_device: str = "cpu"
_tokenizer = MahjongTokenizer()

# 対人対局セッション管理（human_game.py）。無くても牌譜ビューア/自己対戦生成は動く。
try:
    from human_game import SessionRegistry
    _sessions = SessionRegistry()
except ImportError:
    _sessions = None
    print("[INFO] human_game.py が無いため /human/* エンドポイントは無効です。")

# ---- 牌譜自動保存の設定（main()で上書きされる）----
_REPLAY_SAVE: bool = True
_REPLAY_DIR:  Path = Path("replays")


def load_model_global(model_path: Path, device: str):
    """統合モデル（打牌・リーチ・鳴き・槓を担う候補スコアリングモデル）をロード。"""
    global _model, _device
    _device = device
    print(f"統合モデルロード中: {model_path}  device={device}")
    _model = load_policy(model_path, device)
    print(f"  完了  params={sum(p.numel() for p in _model.parameters()):,}")


def _sort_key(pai: str):
    so = {"m": 0, "p": 1, "s": 2, "z": 3}
    if not pai: return (9, 9)
    s = pai[-1] if pai[-1] in "mpsz" else "z"
    n = int(pai[0]) if pai[0].isdigit() else 8
    return (so.get(s, 9), n)

def _sorted(hand):
    return sorted(hand, key=_sort_key)


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------

class GameState:
    def __init__(self, tehais, oya):
        self.hands       = [list(h) for h in tehais]
        self.tsumo_pai   = [None] * 4
        self.melds       = [[] for _ in range(4)]
        self.riichi      = [False] * 4
        self.discards    = [[] for _ in range(4)]
        self.dora_list   = []
        self.tsumo_count = 0
        self.oya         = oya
        self.honba       = 0     # 本場（表示用）
        self.kyotaku     = 0     # 供託リーチ棒本数（表示用・リーチで増加）

    @staticmethod
    def _remove_tile(hand, tile):
        """手牌から1枚除去。赤5(0x)と通常5(5x)の相互ゆらぎも吸収する。
        裸単騎などで赤5が絡むと従来は除去に失敗し手牌枚数がずれて表示が崩れた。"""
        if tile in hand:
            hand.remove(tile); return True
        # 赤→通常, 通常→赤 の両方向を試す
        alt = None
        if tile and tile[0] == "0":
            alt = "5" + tile[1:]          # 0m → 5m
        elif tile and tile[0] == "5":
            alt = "0" + tile[1:]          # 5m → 0m
        if alt and alt in hand:
            hand.remove(alt); return True
        return False

    def on_tsumo(self, actor, pai):
        self.hands[actor].append(pai)
        self.tsumo_pai[actor] = pai
        self.tsumo_count += 1

    def on_dahai(self, actor, pai):
        self._remove_tile(self.hands[actor], pai)
        self.discards[actor].append({"pai": pai, "riichi": self.riichi[actor]})
        self.tsumo_pai[actor] = None

    def on_reach(self, actor):
        self.riichi[actor] = True
        self.kyotaku += 1     # リーチ棒が供託に1本増える（表示用）

    def on_chi(self, actor, pai, consumed):
        for c in consumed:
            self._remove_tile(self.hands[actor], c)
        self.melds[actor].append({"type": "chi", "pai": pai, "consumed": consumed})
        self.tsumo_pai[actor] = None

    def on_pon(self, actor, pai, consumed):
        for c in consumed:
            self._remove_tile(self.hands[actor], c)
        self.melds[actor].append({"type": "pon", "pai": pai, "consumed": consumed})
        self.tsumo_pai[actor] = None

    def on_daiminkan(self, actor, pai, consumed):
        for c in consumed:
            self._remove_tile(self.hands[actor], c)
        self.melds[actor].append({"type": "daiminkan", "pai": pai, "consumed": consumed})
        self.tsumo_pai[actor] = None

    def on_ankan(self, actor, consumed):
        for c in consumed:
            self._remove_tile(self.hands[actor], c)
        self.melds[actor].append({"type": "ankan", "consumed": consumed})
        self.tsumo_pai[actor] = None

    def on_kakan(self, actor, pai):
        self._remove_tile(self.hands[actor], pai)
        norm = pai.replace("0", "5")
        for m in self.melds[actor]:
            if m["type"] == "pon" and (m["pai"] == pai or m["pai"].replace("0","5") == norm):
                m["type"] = "kakan"; break
        self.tsumo_pai[actor] = None

    def on_dora(self, dora_marker):
        self.dora_list.append(dora_marker)

    def snapshot(self):
        def hand_without_tsumo(pi):
            h = list(self.hands[pi]); t = self.tsumo_pai[pi]
            if t and t in h:
                h2 = list(h); h2.remove(t); return _sorted(h2)
            return _sorted(h)
        return {
            "hands":     [hand_without_tsumo(i) for i in range(4)],
            "tsumo_pai": list(self.tsumo_pai),
            "melds":     [[dict(m) for m in ms] for ms in self.melds],
            "riichi":    list(self.riichi),
            "discards":  [[dict(d) for d in ds] for ds in self.discards],
            "dora_list": list(self.dora_list),
            "yama_rest": max(0, 70 - self.tsumo_count),
            "honba":     self.honba,
            "kyotaku":   self.kyotaku,
        }


# ---------------------------------------------------------------------------
# 推論
# ---------------------------------------------------------------------------

def build_infer_from_event(infer_data: dict) -> dict:
    poss = infer_data["poss"]; probs = infer_data["probs"]
    pred_idx = infer_data["pred_idx"]; pred_token = infer_data["pred_token"]
    actual_pai = infer_data.get("actual_pai")
    decoded_pai = infer_data.get("decoded_pai")
    kuikae = bool(infer_data.get("kuikae_overridden", False))

    # [対人対局] 人間の打牌は probs=None（確率を出さない）。
    #   その場合は確率依存の top3/candidates を空にして安全に返す。
    if probs is None:
        return {
            "pred_idx":   pred_idx, "pred_token": pred_token,
            "pred_label": token_to_label(pred_token) if pred_token is not None else "",
            "actual_pai": actual_pai, "decoded_pai": decoded_pai,
            "kuikae_overridden": kuikae,
            "top3_labels": [], "candidates": [],
            "label_idx": pred_idx, "label_token": pred_token,
            "label_label": token_to_label(pred_token) if pred_token is not None else "",
            "is_top1": True, "is_top3": True, "is_human": True,
        }

    valid        = [(i, p) for i, (t, p) in enumerate(zip(poss, probs)) if t != PADDING_ACTION_ID]
    valid_sorted = sorted(valid, key=lambda x: -x[1])[:3]
    # [FIX-⑤] モデル推論牌と実打牌の食い違いをUIに渡す。
    #   self_play が記録する actual_pai（実際に切った牌）/ kuikae_overridden を中継。
    return {
        "pred_idx":   pred_idx, "pred_token": pred_token,
        "pred_label": token_to_label(pred_token),
        "actual_pai": actual_pai, "decoded_pai": decoded_pai,
        "kuikae_overridden": kuikae,
        "top3_labels": [{"rank": r+1, "token": poss[i], "label": token_to_label(poss[i]), "prob": round(p, 4)}
                        for r, (i, p) in enumerate(valid_sorted)],
        "candidates":  [{"idx": i, "token": t, "label": token_to_label(t), "prob": round(probs[i], 4)}
                        for i, t in enumerate(poss) if t != PADDING_ACTION_ID],
        "label_idx": pred_idx, "label_token": pred_token,
        "label_label": token_to_label(pred_token), "is_top1": True, "is_top3": True,
    }


def run_infer(sample: dict) -> dict:
    if _model is None:
        raise RuntimeError("モデル未ロード")
    b = samples_to_batch([sample], _device)
    ca = b["candidates"]
    with torch.no_grad():
        logits = _model(b["sparse"], b["numeric"], b["progression"], ca, b["cand_feats"])[0]
    vm = ca[0] != PADDING_ACTION_ID; masked = logits.clone(); masked[~vm] = float("-inf")
    n_valid      = vm.sum().item()
    pred_idx     = masked.argmax().item()
    top3_indices = masked.topk(min(3, n_valid)).indices.tolist()
    label_idx    = sample["action_label"]; poss = sample["possible_actions"]
    probs        = torch.softmax(masked, dim=-1)
    return {
        "pred_idx": pred_idx, "label_idx": label_idx, "top3_indices": top3_indices,
        "pred_token": poss[pred_idx], "label_token": poss[label_idx],
        "pred_label": token_to_label(poss[pred_idx]), "label_label": token_to_label(poss[label_idx]),
        "is_top1": pred_idx == label_idx, "is_top3": label_idx in top3_indices,
        "top3_labels": [{"rank": i+1, "token": poss[ix], "label": token_to_label(poss[ix]),
                         "prob": round(probs[ix].item(), 4)} for i, ix in enumerate(top3_indices)],
        "candidates":  [{"idx": i, "token": t, "label": token_to_label(t),
                         "prob": round(probs[i].item(), 4)} for i, t in enumerate(poss) if t != PADDING_ACTION_ID],
    }


def run_meld_infer(sample: dict) -> Optional[dict]:
    """
    [統合モデル化] meld_decision サンプルは打牌と同じ候補スコアリング形式
    （possible_actions + action_label）なので、統合モデルで run_infer に委譲する。
    旧 MeldDecisionModel（4クラス分類）は廃止。
    """
    if _model is None:
        return None
    res = run_infer(sample)
    res["kind"] = "meld"
    return res


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class InferGameRequest(BaseModel):
    mjson:  str
    player: Optional[int] = None


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_loaded":  _model is not None,
        "device":        _device,
    }


@app.post("/infer_game")
def infer_game(req: InferGameRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="モデル未ロード")
    lines = [l.strip() for l in req.mjson.strip().splitlines() if l.strip()]
    events = []
    for line in lines:
        try:   events.append(json.loads(line))
        except json.JSONDecodeError: continue
    if not events:
        raise HTTPException(status_code=400, detail="イベントが空です")

    player_names = [f"P{i}" for i in range(4)]; end_scores = None
    for ev in events:
        if ev.get("type") == "start_game" and ev.get("names"): player_names = list(ev["names"])
        if ev.get("type") == "end_game": end_scores = ev.get("scores")

    blocks = []; cur = []
    for ev in events:
        if ev.get("type") == "start_kyoku":   cur = [ev]
        elif ev.get("type") == "end_kyoku":
            if cur: blocks.append(cur); cur = []
        elif cur: cur.append(ev)
    if cur: blocks.append(cur)

    WIND_JP = {"E": "東", "S": "南", "W": "西", "N": "北"}
    kyokus_out = []

    for block in blocks:
        s0 = block[0]
        bakaze = WIND_JP.get(s0.get("bakaze", "E"), "東"); kyoku_n = s0.get("kyoku", 1)
        honba  = s0.get("honba", 0); oya = s0.get("oya", 0)
        kyotaku0 = s0.get("kyotaku", 0)   # 前局から繰り越された供託本数
        scores = s0.get("scores", [25000]*4); dora_m = s0.get("dora_marker", "?")
        tehais = s0.get("tehais", [[], [], [], []])
        gs = GameState(tehais, oya); gs.on_dora(dora_m)
        gs.honba = honba; gs.kyotaku = kyotaku0
        samples = list(_tokenizer.process_events(block))
        sq_idx = 0; events_out = []; result_ev = None
        kyoku_stats = {"total": 0, "top1": 0, "top3": 0}

        for ev in block[1:]:
            etype = ev.get("type", ""); actor = ev.get("actor")
            if etype == "reach_accepted": continue
            dahai_pre_state = None
            if   etype == "tsumo"      and actor is not None: gs.on_tsumo(actor, ev.get("pai","?"))
            elif etype == "dahai"      and actor is not None: dahai_pre_state = gs.snapshot(); gs.on_dahai(actor, ev.get("pai","?"))
            elif etype == "reach"      and actor is not None: gs.on_reach(actor)
            elif etype == "chi"        and actor is not None: gs.on_chi(actor, ev.get("pai","?"), ev.get("consumed",[]))
            elif etype == "pon"        and actor is not None: gs.on_pon(actor, ev.get("pai","?"), ev.get("consumed",[]))
            elif etype == "daiminkan"  and actor is not None: gs.on_daiminkan(actor, ev.get("pai","?"), ev.get("consumed",[]))
            elif etype == "ankan"      and actor is not None: gs.on_ankan(actor, ev.get("consumed",[]))
            elif etype == "kakan"      and actor is not None: gs.on_kakan(actor, ev.get("pai","?"))
            elif etype == "dora":                             gs.on_dora(ev.get("dora_marker","?"))
            elif etype in ("hora","ryukyoku"):                result_ev = ev

            ev_out = {"type": etype, "actor": actor, "raw": ev,
                      "state": dahai_pre_state if dahai_pre_state is not None else gs.snapshot(),
                      "infer": None}

            # [FIX-D] どの実イベントの前にも meld_decision サンプルが挟まりうる。
            # 先頭に溜まっている meld_decision を統合モデルで推論・消費する。
            # meld_decision は打牌と同じ候補スコアリング形式なので run_infer 系で扱える。
            meld_infers = []
            while sq_idx < len(samples) and samples[sq_idx].get("event_type") in ("meld_decision", "kan_decision"):
                ms = samples[sq_idx]
                mres = run_meld_infer(ms)
                if mres is not None and ms.get("event_type") == "kan_decision":
                    mres["kind"] = "kan"
                if mres is not None and (req.player is None or ms["player"] == req.player):
                    meld_infers.append({"player": ms["player"], **mres})
                sq_idx += 1
            if meld_infers:
                ev_out["meld_infer"] = meld_infers

            if etype == "dahai" and ev.get("_infer"):
                infer_res = build_infer_from_event(ev["_infer"])
                if req.player is None or actor == req.player: ev_out["infer"] = infer_res
                if sq_idx < len(samples):
                    s = samples[sq_idx]
                    if s["event_type"] == etype and s["player"] == actor: sq_idx += 1
            else:
                if sq_idx < len(samples):
                    s = samples[sq_idx]
                    if s["event_type"] == etype and s["player"] == actor:
                        infer_res = run_infer(s)
                        if req.player is None or s["player"] == req.player: ev_out["infer"] = infer_res
                        kyoku_stats["total"] += 1
                        kyoku_stats["top1"]  += int(infer_res["is_top1"])
                        kyoku_stats["top3"]  += int(infer_res["is_top3"])
                        sq_idx += 1
            events_out.append(ev_out)

        kyokus_out.append({
            "title": f"{bakaze}{kyoku_n}局 {honba}本場", "bakaze": bakaze,
            "kyoku": kyoku_n, "honba": honba, "kyotaku": kyotaku0, "oya": oya, "start_scores": scores,
            "haipai": [_sorted(h) for h in tehais], "events": events_out,
            "result": result_ev, "stats": kyoku_stats,
        })

    total = sum(k["stats"]["total"] for k in kyokus_out)
    top1  = sum(k["stats"]["top1"]  for k in kyokus_out)
    top3  = sum(k["stats"]["top3"]  for k in kyokus_out)
    return {
        "player_names": player_names, "end_scores": end_scores, "kyokus": kyokus_out,
        "stats": {"total": total, "top1": top1, "top3": top3,
                  "top1_rate": round(top1/max(total,1), 4), "top3_rate": round(top3/max(total,1), 4)},
    }


# ---------------------------------------------------------------------------
# 対人対局 API（人間 vs AI）
# ---------------------------------------------------------------------------

class HumanNewRequest(BaseModel):
    human_player: int = 0          # 人間の席（0-3）
    names: Optional[list] = None


def _events_to_kyokus(events: list) -> dict:
    """セッションのイベント列を、UIが使う kyokus 構造へ変換する。
    infer_game を player=human で通し、モデル推論パネルも人間席基準で付与する。"""
    mjson_text = "\n".join(json.dumps(ev, ensure_ascii=False) for ev in events)
    if not mjson_text.strip():
        return {"player_names": [], "kyokus": [], "stats": {}}
    return infer_game(InferGameRequest(mjson=mjson_text, player=None))


def _require_sessions():
    if _sessions is None:
        raise HTTPException(status_code=501, detail="human_game.py がありません（対人対局は無効）")


@app.post("/human/new")
def human_new(req: HumanNewRequest):
    _require_sessions()
    if _model is None:
        raise HTTPException(status_code=503, detail="モデル未ロード")
    if not (0 <= req.human_player <= 3):
        raise HTTPException(status_code=400, detail="human_player は 0-3")
    s = _sessions.create(_model, _device,
                         human_player=req.human_player, names=req.names)
    return {"session_id": s.id, "human_player": s.human_player}


@app.get("/human/state")
def human_state(session_id: str):
    _require_sessions()
    s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    snap = s.snapshot()
    # 人間の入力待ちの間は同じイベント列で何度も呼ばれるので、変換結果を使い回す
    # （変換では全手のモデル推論をやり直すので、CPU だと 1 回数秒かかる）。
    n_ev = len(snap["events"])
    cache = getattr(s, "_view_cache", None)
    if cache is not None and cache[0] == n_ev:
        out = dict(cache[1])
    else:
        out = _events_to_kyokus(snap["events"])
        s._view_cache = (n_ev, dict(out))
    out["session_id"]     = snap["id"]
    out["human_player"]   = snap["human_player"]
    out["player_names"]   = snap["names"]
    out["pending"]        = snap["pending"]
    out["awaiting_human"] = snap["awaiting_human"]
    out["finished"]       = snap["finished"]
    out["error"]          = snap["error"]
    out["is_self_play"]   = True   # 正解ラベル比較を出さない（人間の手はラベル無し）
    out["is_human_game"]  = True
    return out


class HumanActionRequest(BaseModel):
    session_id: str
    token: int


@app.post("/human/action")
def human_action(req: HumanActionRequest):
    _require_sessions()
    s = _sessions.get(req.session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    ok = s.submit(req.token)
    if not ok:
        raise HTTPException(status_code=409,
                            detail="今は入力待ちでないか、無効な選択です")
    return {"ok": True}


@app.post("/human/quit")
def human_quit(session_id: str):
    _require_sessions()
    _sessions.remove(session_id)
    return {"ok": True}


@app.get("/", response_class=None)
def serve_ui():
    from fastapi.responses import HTMLResponse
    ui_path = Path(__file__).with_name("ui.html")
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="ui.html が見つかりません")
    return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))


class GenerateGameRequest(BaseModel):
    seed: Optional[int] = None


@app.post("/generate_game")
def generate_game(req: GenerateGameRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="モデル未ロード")
    import random as _random
    from self_play import MahjongSimulator
    if req.seed is not None:
        _random.seed(req.seed); torch.manual_seed(req.seed)
    # 統合モデル1本で打牌・鳴き・立直すべてを処理する
    sim    = MahjongSimulator(_model, _device, watch=False)
    events = sim.simulate_game()
    mjson_text = "\n".join(json.dumps(ev, ensure_ascii=False) for ev in events)

    # ---- 牌譜の自動保存 ----
    # 生成した対局を mjai 形式(.mjson)で replays/ に保存する。
    # これにより後から「この局面でモデルが何を見て何を選んだか」を
    # 正確に再現・検証できる。保存先・有効/無効は環境変数で制御。
    saved_path = None
    if _REPLAY_SAVE:
        try:
            import datetime
            _REPLAY_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            seed_tag = f"_seed{req.seed}" if req.seed is not None else ""
            fname = f"replay_{ts}{seed_tag}.mjson"
            saved_path = _REPLAY_DIR / fname
            saved_path.write_text(mjson_text, encoding="utf-8")
            print(f"[牌譜保存] {saved_path}")
        except Exception as e:
            print(f"[WARN] 牌譜保存に失敗: {e}")
            saved_path = None

    result = infer_game(InferGameRequest(mjson=mjson_text, player=None))
    result["is_self_play"] = True
    if saved_path is not None:
        result["replay_file"] = str(saved_path)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        type=Path, default=None,
                   help="統合モデル .pt（打牌・鳴き・立直すべてを担う）")
    p.add_argument("--ckpt-dir",     type=Path, default=Path("checkpoints1"))
    p.add_argument("--port",         type=int,  default=8000)
    p.add_argument("--host",         type=str,  default="127.0.0.1")
    p.add_argument("--device",       type=str,  default=None)
    p.add_argument("--replay-dir",   type=Path, default=Path("replays"),
                   help="生成した対局の牌譜(.mjson)を保存するディレクトリ")
    p.add_argument("--no-save-replay", action="store_true",
                   help="牌譜の自動保存を無効にする")
    args   = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 牌譜保存設定をグローバルへ反映
    global _REPLAY_SAVE, _REPLAY_DIR
    _REPLAY_SAVE = not args.no_save_replay
    _REPLAY_DIR  = args.replay_dir
    if _REPLAY_SAVE:
        print(f"牌譜自動保存: 有効  保存先={_REPLAY_DIR}/")
    else:
        print("牌譜自動保存: 無効")

    default_w = Path(__file__).resolve().parent.parent / "weights" / "model_bc_best.pt"
    if args.model:
        model_path = args.model
    elif default_w.exists():
        model_path = default_w
    else:
        pts = sorted(args.ckpt_dir.glob("model_bc_epoch*.pt")) or sorted(args.ckpt_dir.glob("*.pt"))
        if not pts: raise FileNotFoundError(f"{args.ckpt_dir} に .pt が見つかりません")
        model_path = pts[-1]

    load_model_global(model_path=model_path, device=device)
    print(f"\nサーバー起動: http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
