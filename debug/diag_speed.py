"""
diag_speed.py  ―  「半荘1回が遅い」のがどこかを切り分ける

  python3 debug/diag_speed.py --model train/checkpoints1_ronhead_off/model_bc_best.pt

測るもの:
  1. 環境（GPU名・CPU負荷・クロック）
  2. 半荘1回の実時間と、その間の forward 回数 → 1 forward あたりの実測 ms
  3. forward 単体の時間（同じサンプルを繰り返す）
  4. シミュレータだけの時間（モデルを使わず、合法手の先頭を選ぶダミーで1半荘）

読み方:
  - forward 単体が遅い（2 ms 超）     → 推論側。GPU の取り合い／クロック低下
  - forward 単体は速いが半荘が遅い    → Python 側（向聴計算・役判定・tokenizer）
  - ダミー打ちだけで遅い              → シミュレータ本体。モデルは無関係

基準値（2026-10-10 時点の実測）:
  半荘1回 = 約 2.7 秒。これが 12 秒なら約 4.5 倍遅い。
"""
from __future__ import annotations
import _pathsetup  # noqa: F401
import argparse
import os
import random
import time
from pathlib import Path

import torch

import self_play as SP
from mahjong_model import load_policy
from mahjong_tokenizer import PADDING_ACTION_ID


def show_env():
    print("=" * 64)
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        print(f"GPU: {torch.cuda.get_device_name(i)}")
        try:
            free, total = torch.cuda.mem_get_info(i)
            print(f"  VRAM 空き {free/2**30:.1f} / {total/2**30:.1f} GiB"
                  f"   ← 空きが極端に少なければ他プロセスが使用中")
        except Exception:
            pass
    try:
        la = os.getloadavg()
        print(f"CPU load average: {la[0]:.2f} / {la[1]:.2f} / {la[2]:.2f}"
              f"   (コア数 {os.cpu_count()})   ← 1分値がコア数に近ければ取り合い")
    except Exception:
        pass
    try:
        mhz = [float(l.split(":")[1]) for l in open("/proc/cpuinfo")
               if l.startswith("cpu MHz")]
        if mhz:
            print(f"CPU クロック: 平均 {sum(mhz)/len(mhz):.0f} MHz  "
                  f"最低 {min(mhz):.0f}  最高 {max(mhz):.0f}   ← 極端に低ければ省電力モード")
    except Exception:
        pass
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--fwd-reps", type=int, default=300)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    show_env()

    model = load_policy(args.model, device)

    # ---- 2) 半荘1回 ＋ forward 回数 ----
    orig_choose = SP.model_choose
    state = {"n": 0, "sample": None, "t_in": 0.0}

    def counting(m, sample, dev):
        state["n"] += 1
        state["sample"] = sample
        t = time.perf_counter()
        r = orig_choose(m, sample, dev)
        state["t_in"] += time.perf_counter() - t
        return r

    SP.model_choose = counting
    sim = SP.MahjongSimulator(model, device, watch=False)
    sim.player_models = {p: model for p in range(4)}
    random.seed(1); torch.manual_seed(1)
    t0 = time.perf_counter()
    ev = sim.simulate_game()
    t_game = time.perf_counter() - t0
    SP.model_choose = orig_choose

    n_kyoku = sum(e["type"] == "start_kyoku" for e in ev)
    n = max(state["n"], 1)
    print(f"[2] 半荘1回: {t_game:.2f} 秒   局数 {n_kyoku}   forward {state['n']} 回")
    print(f"    うち forward の合計: {state['t_in']:.2f} 秒 "
          f"({state['t_in']/t_game*100:.0f}%)   → {state['t_in']/n*1000:.2f} ms/forward")
    print(f"    それ以外(Python側): {t_game-state['t_in']:.2f} 秒 "
          f"({(t_game-state['t_in'])/t_game*100:.0f}%)")

    # ---- 3) forward 単体 ----
    smp = state["sample"]
    if smp is not None:
        for _ in range(10):
            orig_choose(model, smp, device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.fwd_reps):
            orig_choose(model, smp, device)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.fwd_reps
        print(f"[3] forward 単体: {dt*1000:.2f} ms  "
              f"(候補 {sum(1 for t in smp['possible_actions'] if t != PADDING_ACTION_ID)} 個, "
              f"sparse {len(smp['sparse'])}, prog {len(smp['progression'])})")

    # ---- 4) シミュレータだけ（モデルを使わない） ----
    def dummy(m, sample, dev):
        # model_choose と同じ 4-tuple (token, idx, probs, possible_actions) を返す
        poss = sample["possible_actions"]
        idx = next((i for i, t in enumerate(poss) if t != PADDING_ACTION_ID), 0)
        probs = [1.0 if i == idx else 0.0 for i in range(len(poss))]
        return poss[idx], idx, probs, poss

    SP.model_choose = dummy
    sim2 = SP.MahjongSimulator(model, device, watch=False)
    sim2.player_models = {p: model for p in range(4)}
    random.seed(1); torch.manual_seed(1)
    t0 = time.perf_counter()
    try:
        sim2.simulate_game()
        t_dummy = time.perf_counter() - t0
        print(f"[4] シミュレータのみ(ダミー打ち): {t_dummy:.2f} 秒"
              f"   ← ここが数秒なら Python 側が主犯")
    except Exception as e:
        print(f"[4] ダミー打ちは失敗（打牌が不正になるため正常）: {type(e).__name__}")
    finally:
        SP.model_choose = orig_choose

    print("=" * 64)
    print(f"目安: 2026-10-10 の実測は 半荘1回 ≒ 2.7 秒。今回 {t_game:.1f} 秒 "
          f"→ {t_game/2.7:.1f} 倍")


if __name__ == "__main__":
    main()
