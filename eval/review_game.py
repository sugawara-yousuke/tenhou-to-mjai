"""
review_game.py
==============
1半荘を通してモデルの推論結果を牌譜＋手牌と一緒に表示するスクリプト。

使い方:
  python review_game.py --data 2024.zip
  python review_game.py --data 2024.zip --game-index 3
  python review_game.py --data 2024.zip --player 0
  python review_game.py --data 2024.zip --model checkpoints1/model_bc_epoch5.pt
"""

from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
import argparse
import gzip
import json
import zipfile
from pathlib import Path

import torch

from mahjong_model import load_policy, samples_to_batch
from mahjong_tokenizer import (
    MahjongTokenizer,
    PADDING_ACTION_ID,
    ACT_DAHAI_BASE, ACT_RIICHI_BASE, ACT_PON_BASE, ACT_CHI_BASE,
    ACT_DAIMINKAN_BASE, ACT_ANKAN_BASE, ACT_KAKAN_BASE,
    ACT_TSUMO_BASE, ACT_RON_BASE, ACT_PASS_BASE,
)


# ---------------------------------------------------------------------------
# 定数・色
# ---------------------------------------------------------------------------

TILE_NAMES = [
    "1m","2m","3m","4m","5m","6m","7m","8m","9m",
    "1p","2p","3p","4p","5p","6p","7p","8p","9p",
    "1s","2s","3s","4s","5s","6s","7s","8s","9s",
    "東","南","西","北","白","発","中",
    "赤5m","赤5p","赤5s",
]
WIND_JP = {"E":"東","S":"南","W":"西","N":"北"}

SUIT_COLOR = {"m":"\033[91m","p":"\033[92m","s":"\033[94m","z":"\033[95m"}
C_RESET="\033[0m"; C_BOLD="\033[1m"; C_CYAN="\033[96m"; C_YELLOW="\033[93m"
C_GREEN="\033[92m"; C_RED="\033[91m"; C_GRAY="\033[90m"
C_BLUE="\033[94m"; C_MAG="\033[95m"; C_UL="\033[4m"

def col(text, *codes): return "".join(codes)+str(text)+C_RESET


# ---------------------------------------------------------------------------
# 牌表示ヘルパー
# ---------------------------------------------------------------------------

def tile_colored(pai: str) -> str:
    if not pai or pai=="?": return col("?",C_GRAY)
    alias={"E":"東","S":"南","W":"西","N":"北","P":"白","F":"発","C":"中"}
    if pai in alias: return col(alias[pai],SUIT_COLOR["z"])
    z_map={"1z":"東","2z":"南","3z":"西","4z":"北","5z":"白","6z":"発","7z":"中"}
    if pai in z_map: return col(z_map[pai],SUIT_COLOR["z"])
    suit=pai[-1] if pai[-1] in "mpsz" else "z"
    num=pai[0]
    disp=("赤" if num=="0" else num)+{"m":"萬","p":"筒","s":"索","z":""}[suit]
    return col(disp,SUIT_COLOR.get(suit,C_RESET))


def token_to_str(token: int) -> str:
    if token==PADDING_ACTION_ID: return "[PAD]"
    if token==ACT_PASS_BASE:     return "パス"
    if token==ACT_TSUMO_BASE:    return "ツモ和了"
    if token==ACT_RON_BASE:      return "ロン"
    if ACT_DAHAI_BASE<=token<ACT_RIICHI_BASE:
        t=token-ACT_DAHAI_BASE; return f"打牌 {TILE_NAMES[t] if t<len(TILE_NAMES) else t}"
    if ACT_RIICHI_BASE<=token<ACT_PON_BASE:
        t=token-ACT_RIICHI_BASE; return f"リーチ {TILE_NAMES[t] if t<len(TILE_NAMES) else t}"
    if ACT_PON_BASE<=token<ACT_CHI_BASE:
        t=token-ACT_PON_BASE; return f"ポン {TILE_NAMES[t] if t<len(TILE_NAMES) else t}"
    if ACT_CHI_BASE<=token<ACT_DAIMINKAN_BASE: return f"チー({token-ACT_CHI_BASE})"
    if ACT_DAIMINKAN_BASE<=token<ACT_ANKAN_BASE:
        t=token-ACT_DAIMINKAN_BASE; return f"明槓 {TILE_NAMES[t] if t<len(TILE_NAMES) else t}"
    if ACT_ANKAN_BASE<=token<ACT_KAKAN_BASE:
        t=token-ACT_ANKAN_BASE; return f"暗槓 {TILE_NAMES[t] if t<len(TILE_NAMES) else t}"
    if ACT_KAKAN_BASE<=token<ACT_TSUMO_BASE:
        t=token-ACT_KAKAN_BASE; return f"加槓 {TILE_NAMES[t] if t<len(TILE_NAMES) else t}"
    return f"?(token={token})"


# ---------------------------------------------------------------------------
# 手牌トラッカー
# ---------------------------------------------------------------------------

def _sort_key(pai):
    so={"m":0,"p":1,"s":2,"z":3}
    if not pai: return(9,9)
    s=pai[-1] if pai[-1] in "mpsz" else "z"
    n=int(pai[0]) if pai[0].isdigit() else 8
    return(so.get(s,9),n)


class HandTracker:
    def __init__(self, tehais):
        self.hands=[list(h) for h in tehais]
        self.tsumo_pai=[None]*4

    def on_tsumo(self,actor,pai):
        self.tsumo_pai[actor]=pai; self.hands[actor].append(pai)

    def on_dahai(self,actor,pai):
        self.tsumo_pai[actor]=None
        h=self.hands[actor]
        if pai in h: h.remove(pai)
        elif pai.replace("0","5") in h: h.remove(pai.replace("0","5"))

    def on_chi(self,actor,consumed):
        for c in consumed:
            if c in self.hands[actor]: self.hands[actor].remove(c)

    def on_pon(self,actor,consumed):
        for c in consumed:
            if c in self.hands[actor]: self.hands[actor].remove(c)

    def on_ankan(self,actor,consumed):
        for c in consumed:
            if c in self.hands[actor]: self.hands[actor].remove(c)

    def on_kakan(self,actor,pai):
        if pai in self.hands[actor]: self.hands[actor].remove(pai)

    def get_hand(self,player): return sorted(self.hands[player],key=_sort_key)
    def get_tsumo(self,player): return self.tsumo_pai[player]


# ---------------------------------------------------------------------------
# モデルロード・推論
# ---------------------------------------------------------------------------

def load_model(model_path,device):
    print(f"モデルロード中: {model_path}")
    m=load_policy(model_path,device)
    print(f"  params={sum(p.numel() for p in m.parameters()):,}\n")
    return m


def infer(model,sample,device):
    b=samples_to_batch([sample],device); ca=b["candidates"]
    with torch.no_grad():
        logits=model(b["sparse"],b["numeric"],b["progression"],ca,b["cand_feats"])[0]
    vm=ca[0]!=PADDING_ACTION_ID
    mk=logits.clone(); mk[~vm]=float("-inf")
    pred=mk.argmax().item()
    top3=mk.topk(min(3,int(vm.sum().item()))).indices.tolist()
    return pred,sample["action_label"],top3


def read_one_game(data_path,game_index):
    if data_path.suffix==".zip":
        with zipfile.ZipFile(data_path) as zf:
            names=sorted(n for n in zf.namelist() if n.endswith(".mjson"))
            if game_index>=len(names):
                raise IndexError(f"game_index={game_index} 範囲外（総試合数:{len(names)}）")
            raw=zf.read(names[game_index]); fname=names[game_index]
    elif data_path.is_dir():
        files=sorted(data_path.glob("**/*.mjson"))
        raw=files[game_index].read_bytes(); fname=str(files[game_index])
    else:
        raw=data_path.read_bytes(); fname=str(data_path)
    text=gzip.decompress(raw).decode() if raw[:2]==b'\x1f\x8b' else raw.decode()
    events=[json.loads(l) for l in text.splitlines() if l.strip()]
    return events,fname


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def review_game(args):
    device=args.device or("cuda" if torch.cuda.is_available() else "cpu")

    if args.model:
        model_path=args.model
    else:
        pts=sorted(args.ckpt_dir.glob("model_bc_epoch*.pt")) or sorted(args.ckpt_dir.glob("*.pt"))
        if not pts: raise FileNotFoundError(f"{args.ckpt_dir} に .pt が見つかりません")
        model_path=pts[-1]

    model=load_model(model_path,device)
    events,fname=read_one_game(args.data,args.game_index)
    tokenizer=MahjongTokenizer()
    target=args.player

    # ゲームメタ
    pnames=[f"P{i}" for i in range(4)]; end_scores=None
    for ev in events:
        if ev.get("type")=="start_game" and ev.get("names"): pnames=list(ev["names"])
        if ev.get("type")=="end_game": end_scores=ev.get("scores")

    # ヘッダー
    print()
    print(col("="*68,C_BOLD,C_CYAN))
    print(col("  🀄  1半荘 モデル推論レビュー",C_BOLD,C_CYAN))
    print(col(f"  {fname}",C_GRAY))
    print(col("  "+"  ".join(f"P{i}:{n}" for i,n in enumerate(pnames)),C_GRAY))
    print(col("="*68,C_BOLD,C_CYAN))

    # 局単位に分割
    blocks=[]; cur=[]
    for ev in events:
        if ev.get("type")=="start_kyoku": cur=[ev]
        elif ev.get("type")=="end_kyoku":
            if cur: blocks.append(cur); cur=[]
        elif cur: cur.append(ev)
    if cur: blocks.append(cur)

    game_stats={"total":0,"top1":0,"top3":0}

    for block in blocks:
        s0=block[0]
        bakaze=WIND_JP.get(s0.get("bakaze","E"),"東")
        kn=s0.get("kyoku",1); honba=s0.get("honba",0); oya=s0.get("oya",0)
        scores=s0.get("scores",[25000]*4)
        dora_m=s0.get("dora_marker","?")
        tehais=s0.get("tehais",[[]]*4)
        dora_list=[dora_m]

        print()
        print(col(f"  ╔══ {bakaze}{kn}局 {honba}本場  親:{pnames[oya]}  ドラ表示:{dora_m} ══╗",C_BOLD,C_BLUE))
        print(col("  点数: "+"  ".join(f"P{i}:{s:,}" for i,s in enumerate(scores)),C_GRAY))

        # ── 配牌表示 ──
        print()
        print(col("  ┌─ 配牌 ─┐",C_BOLD))
        for pi,hand in enumerate(tehais):
            if target is not None and pi!=target: continue
            sh=" ".join(tile_colored(t) for t in sorted(hand,key=_sort_key))
            print(f"  │ P{pi} {col(pnames[pi],C_BOLD)} : {sh}")
        print(col("  └────────┘",C_BOLD))

        tracker=HandTracker(tehais)
        samples=list(tokenizer.process_events(block+[{"type":"end_kyoku"}]))
        sq_idx=0
        kyoku_stats={"total":0,"top1":0,"top3":0}
        result_ev=None

        for ev in block[1:]:
            etype=ev.get("type",""); actor=ev.get("actor")

            # ── ツモ ──
            if etype=="tsumo" and actor is not None:
                pai=ev.get("pai","?")
                tracker.on_tsumo(actor,pai)
                if target is None or actor==target:
                    hw=tracker.get_hand(actor)
                    # ツモ牌を除いた手牌 + | + ツモ牌
                    base=[t for t in hw if t!=pai]
                    h_str=" ".join(tile_colored(t) for t in base)
                    print(f"\n  P{actor} {col(pnames[actor],C_BOLD)}  "
                          f"ツモ {col(tile_colored(pai),C_BOLD)}")
                    print(f"  手牌: {h_str}  {col('|',C_GRAY)}  {col(tile_colored(pai),C_BOLD)}")

            # ── 打牌 ──
            elif etype=="dahai" and actor is not None:
                pai=ev.get("pai","?"); tg=ev.get("tsumogiri",False)
                tracker.on_dahai(actor,pai)
                if target is None or actor==target:
                    hw=tracker.get_hand(actor)
                    h_str=" ".join(tile_colored(t) for t in hw)
                    tg_mark=col(" (ツモ切り)",C_GRAY) if tg else ""
                    print(f"  打牌: {col(tile_colored(pai),C_BOLD)}{tg_mark}")
                    print(f"  手牌: {h_str}")

            # ── 副露 ──
            elif etype=="chi":
                consumed=ev.get("consumed",[]); pai=ev.get("pai","?")
                tracker.on_chi(actor,consumed)
                if target is None or actor==target:
                    cs=" ".join(tile_colored(c) for c in consumed)
                    print(f"\n  P{actor} {col(pnames[actor],C_BOLD)}  "
                          f"{col('チー',C_MAG,C_BOLD)}: {tile_colored(pai)} [{cs}]")
                    hw=tracker.get_hand(actor)
                    print(f"  手牌: {' '.join(tile_colored(t) for t in hw)}")

            elif etype=="pon":
                consumed=ev.get("consumed",[]); pai=ev.get("pai","?")
                tracker.on_pon(actor,consumed)
                if target is None or actor==target:
                    cs=" ".join(tile_colored(c) for c in consumed)
                    print(f"\n  P{actor} {col(pnames[actor],C_BOLD)}  "
                          f"{col('ポン',C_MAG,C_BOLD)}: {tile_colored(pai)} [{cs}]")
                    hw=tracker.get_hand(actor)
                    print(f"  手牌: {' '.join(tile_colored(t) for t in hw)}")

            elif etype=="ankan":
                consumed=ev.get("consumed",[])
                tracker.on_ankan(actor,consumed)
                if target is None or actor==target:
                    cs=" ".join(tile_colored(c) for c in consumed)
                    print(f"\n  P{actor} {col(pnames[actor],C_BOLD)}  "
                          f"{col('暗槓',C_MAG,C_BOLD)}: [{cs}]")

            elif etype=="kakan":
                pai=ev.get("pai","?"); tracker.on_kakan(actor,pai)
                if target is None or actor==target:
                    print(f"\n  P{actor} {col(pnames[actor],C_BOLD)}  "
                          f"{col('加槓',C_MAG,C_BOLD)}: {tile_colored(pai)}")

            elif etype=="reach":
                if target is None or actor==target:
                    print(col(f"\n  P{actor} {pnames[actor]}  🔴 リーチ！",C_RED,C_BOLD))

            elif etype=="dora":
                dm=ev.get("dora_marker","?"); dora_list.append(dm)
                print(col(f"\n  ── 新ドラ表示: {dm} ──",C_YELLOW))

            elif etype in("hora","ryukyoku"):
                result_ev=ev

            # ── モデル推論結果 ──
            # 鳴き判断/槓判断のサンプルはどのイベントの前にも挟まりうるので先に消費する
            while sq_idx<len(samples) and samples[sq_idx]["event_type"] in ("meld_decision","kan_decision"):
                sk=samples[sq_idx]; sq_idx+=1
                p2,l2,t2=infer(model,sk,device)
                if target is None or sk["player"]==target:
                    poss=sk["possible_actions"]
                    tag="鳴き判断" if sk["event_type"]=="meld_decision" else "槓判断"
                    mark=col("✅",C_GREEN) if p2==l2 else col("❌",C_RED)
                    print(col(f"  [{tag} P{sk['player']}] {mark} モデル→{token_to_str(poss[p2])}  実際→{token_to_str(poss[l2])}",C_GRAY))
                for st_ in (kyoku_stats,game_stats):
                    st_["total"]+=1; st_["top1"]+=int(p2==l2); st_["top3"]+=int(l2 in t2)
            if etype=="dahai" and sq_idx<len(samples):
                s=samples[sq_idx]
                if s["event_type"]=="dahai" and s["player"]==actor:
                    pred,label,top3=infer(model,s,device)
                    is1=(pred==label); is3=(label in top3)
                    if target is None or s["player"]==target:
                        poss=s["possible_actions"]
                        pt=token_to_str(poss[pred]); lt=token_to_str(poss[label])
                        mark=col("✅ TOP1",C_GREEN,C_BOLD) if is1 else \
                             col("△  TOP3",C_YELLOW,C_BOLD) if is3 else \
                             col("❌ 圏外",C_RED,C_BOLD)
                        print(f"  {mark}  モデル→{col(pt,C_CYAN)}"
                              +("" if is1 else f"  正解→{col(lt,C_YELLOW)}"))
                        if not is1:
                            t3s="  ".join(f"{i+1}:{col(token_to_str(poss[ix]),C_GRAY)}"
                                          for i,ix in enumerate(top3))
                            print(f"  Top3: {t3s}")
                    for st_ in (kyoku_stats,game_stats):
                        st_["total"]+=1; st_["top1"]+=int(is1); st_["top3"]+=int(is3)
                    sq_idx+=1

        # 局結果
        print()
        if result_ev:
            if result_ev.get("type")=="hora":
                w=result_ev.get("actor","?"); lo=result_ev.get("target"); sc=result_ev.get("score","?")
                wn=pnames[w] if isinstance(w,int) and w<4 else w
                if lo is not None:
                    ln=pnames[lo] if isinstance(lo,int) and lo<4 else lo
                    print(col(f"  🎉 ロン和了: P{w}({wn}) ← P{lo}({ln})  +{sc}",C_GREEN,C_BOLD))
                else:
                    print(col(f"  🎉 ツモ和了: P{w}({wn})  +{sc}",C_GREEN,C_BOLD))
            elif result_ev.get("type")=="ryukyoku":
                print(col("  💨 流局",C_GRAY,C_BOLD))

        t=kyoku_stats["total"]
        if t:
            t1,t3=kyoku_stats["top1"],kyoku_stats["top3"]
            print(col(f"  この局: {t}局面  Top1={t1}/{t}({t1/t*100:.0f}%)  Top3={t3}/{t}({t3/t*100:.0f}%)",C_GRAY))

    # 最終サマリ
    total=game_stats["total"]; top1=game_stats["top1"]; top3=game_stats["top3"]
    print()
    print(col("="*68,C_BOLD,C_CYAN))
    print(col("  1半荘 合計",C_BOLD,C_CYAN))
    print(col("="*68,C_BOLD,C_CYAN))
    print(f"  総局面数  : {total:,}")
    if total:
        print(f"  Top-1精度 : {col(f'{top1/total*100:.1f}%  ({top1}/{total})',C_GREEN,C_BOLD)}")
        print(f"  Top-3精度 : {col(f'{top3/total*100:.1f}%  ({top3}/{total})',C_YELLOW,C_BOLD)}")
    if end_scores:
        print()
        print(col("  最終スコア:",C_BOLD))
        ranked=sorted(enumerate(end_scores),key=lambda x:-x[1])
        medals=["🥇","🥈","🥉","　"]
        for rank,(pi,score) in enumerate(ranked):
            diff=score-25000; ds=f"+{diff:,}" if diff>=0 else f"{diff:,}"
            print(f"  {medals[rank]} P{pi} {pnames[pi]:<10} {score:>8,}  ({ds})")
    print(col("="*68,C_BOLD,C_CYAN))
    print()


def main():
    p=argparse.ArgumentParser(description="1半荘通し モデル推論レビュー（手牌表示付き）")
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--model",type=Path,default=None)
    p.add_argument("--ckpt-dir",type=Path,default=Path("checkpoints1"))
    p.add_argument("--game-index",type=int,default=0,help="何番目の試合か（0始まり）")
    p.add_argument("--player",type=int,default=None,help="特定プレイヤーだけ表示（例: 0）")
    p.add_argument("--device",type=str,default=None)
    review_game(p.parse_args())

if __name__=="__main__":
    main()
