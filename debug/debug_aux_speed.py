#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_aux_speed.py
==================
--aux-coef を足すと 1it=49秒まで激遅になる問題の切り分け専用スクリプト。
学習は一切回さず、以下を測る:

  [1] 通常 Dataset（aux 無し）の DataLoader スループット   sec/batch
  [2] Aux  Dataset（aux 有り）の DataLoader スループット   sec/batch
  [3] compute_ukeire 単体のコスト（1コールあたり μs、shanten 呼び出し回数）
  [4] IterableDataset の worker 二重計算チェック
      （num_workers を変えて総サンプル数が増える＝重複するか）

使い方（train/ ディレクトリから）:
    python3 debug_aux_speed.py --data ../data/2024.zip --max-games 3
    # 速さだけ見たいなら max-games は 2〜5 で十分（全数流さない）

    # 個別に実行:
    python3 debug_aux_speed.py --data ../data/2024.zip --only 3   # ukeire単体だけ
"""

from __future__ import annotations
import _pathsetup  # noqa: F401
import argparse
import time
import sys

import torch
from torch.utils.data import DataLoader

from load_dataset import (
    TenhouTorchDataset,
    TenhouAuxDataset,
    TenhouDatasetLoader,
)


# ─────────────────────────────────────────────────────────────
# [1][2] DataLoader スループット
# ─────────────────────────────────────────────────────────────
def bench_loader(name, dataset, collate, batch_size, num_workers,
                 max_batches=30, pin=False):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin,
    )
    # 最初の1バッチは worker 起動込みなので別計測
    it = iter(loader)
    t0 = time.perf_counter()
    first = next(it)
    t_first = time.perf_counter() - t0

    n = 0
    n_samples = 0
    t0 = time.perf_counter()
    for batch in it:
        n += 1
        # バッチ内サンプル数（label が基準）
        n_samples += batch["label"].shape[0]
        if n >= max_batches:
            break
    dt = time.perf_counter() - t0

    per = dt / max(1, n)
    print(f"\n[{name}]  workers={num_workers}  batch_size={batch_size}")
    print(f"  first-batch(起動込): {t_first:6.2f} s")
    print(f"  定常   : {per*1000:8.1f} ms/batch   ({n} batches, {n_samples} samples)")
    print(f"  換算   : {1.0/per:6.2f} it/s")
    return per, n_samples


# ─────────────────────────────────────────────────────────────
# [3] compute_ukeire 単体
# ─────────────────────────────────────────────────────────────
def bench_ukeire(data_path, max_games=2, sample_limit=300):
    """
    aux 付きサンプルを回して、compute_ukeire に渡る手牌で
    1コールあたりのコストと shanten_number 呼び出し回数を測る。
    """
    import aux_labels
    from load_dataset import TenhouDatasetLoader

    # shanten_number をラップして呼び出し回数を数える
    import mahjong_tokenizer as tok
    orig_shanten = tok.shanten_number
    counter = {"n": 0}

    def counting_shanten(*a, **k):
        counter["n"] += 1
        return orig_shanten(*a, **k)

    tok.shanten_number = counting_shanten
    # aux_labels 側が from mahjong_tokenizer import shanten_number を
    # 関数内で行っているのでモジュール差し替えで捕捉できるはず。
    # 念のため aux_labels 名前空間にも反映:
    if hasattr(aux_labels, "shanten_number"):
        aux_labels.shanten_number = counting_shanten

    loader = TenhouDatasetLoader(data_path)

    # instrumented サンプルを取得（_hand_strs 付き）
    tokenizer = loader.tokenizer
    tokenizer._emit_aux_fields = True

    total_calls = 0
    total_time = 0.0
    n_uke = 0
    hands_seen = []

    for game_id, events in loader.iter_games(max_games=max_games):
        game_samples = []
        for s in tokenizer.process_events(events):
            game_samples.append(s)

        # attach 前に、_hand_strs を持つ dahai サンプルだけ compute_ukeire を直接叩く
        for smp in game_samples:
            if smp.get("event_type") != "dahai":
                continue
            hand = smp.get("_hand_strs", None)
            if hand is None:
                continue
            nm = smp.get("_num_melds", 0)
            dh = smp.get("_dahai_str", None)
            counter["n"] = 0
            t0 = time.perf_counter()
            _ = aux_labels.compute_ukeire(tokenizer, hand, nm, dh)
            dt = time.perf_counter() - t0
            total_time += dt
            total_calls += counter["n"]
            n_uke += 1
            if len(hands_seen) < 5:
                hands_seen.append((len(hand), counter["n"], dt * 1e6))
            if n_uke >= sample_limit:
                break
        if n_uke >= sample_limit:
            break

    tok.shanten_number = orig_shanten  # 復元

    print("\n[3] compute_ukeire 単体プロファイル")
    if n_uke == 0:
        print("  !! _hand_strs 付き dahai サンプルが取れませんでした。")
        print("     tokenizer が _emit_aux_fields で _hand_strs を出しているか要確認。")
        return
    print(f"  計測サンプル数         : {n_uke}")
    print(f"  平均コスト             : {total_time / n_uke * 1e6:8.1f} μs / call")
    print(f"  平均 shanten 呼び出し  : {total_calls / n_uke:8.1f} 回 / call")
    print(f"  → 1サンプルあたり shanten を {total_calls / n_uke:.0f} 回呼んでいる")
    print(f"  最初の5件 (手牌枚数, shanten呼び出し, μs):")
    for h in hands_seen:
        print(f"     len={h[0]:2d}  calls={h[1]:4d}  {h[2]:8.1f} μs")


# ─────────────────────────────────────────────────────────────
# [4] worker 二重計算チェック
# ─────────────────────────────────────────────────────────────
def check_worker_dup(data_path, max_games=2, batch_size=256):
    """
    同じ max_games で num_workers=0 と 2 の総サンプル数を比較。
    IterableDataset がシャーディングしていないと、workers 分だけ
    総数が (ほぼ) 増える。増えたら「二重計算 & データ重複」確定。
    """
    collate = TenhouAuxDataset.__dict__  # placeholder
    from load_dataset import TenhouTorchDataset as TT
    _collate = TT.collate_fn_aux

    def count_samples(nw):
        ds = TenhouAuxDataset(data_path, max_games=max_games,
                              offset=0, event_filter="dahai")
        loader = DataLoader(ds, batch_size=batch_size,
                            collate_fn=_collate, num_workers=nw)
        n = 0
        for b in loader:
            n += b["label"].shape[0]
        return n

    n0 = count_samples(0)
    n2 = count_samples(2)
    print("\n[4] IterableDataset worker 二重計算チェック")
    print(f"  num_workers=0 → 総サンプル {n0}")
    print(f"  num_workers=2 → 総サンプル {n2}")
    if n2 > n0 * 1.5:
        ratio = n2 / max(1, n0)
        print(f"  !! 重複あり: workers=2 で約 {ratio:.1f}倍。"
              f"シャーディング未実装。num_workers は 0 にするか、")
        print(f"     __iter__ に get_worker_info() 分配を実装する必要あり。")
    else:
        print("  重複なし（シャーディング済みか、単一worker相当）。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-games", type=int, default=3,
                    help="全数流さない。速さ計測は2〜5で十分。")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--only", type=int, default=0,
                    help="1/2/3/4 のいずれかだけ実行。0=全部")
    args = ap.parse_args()

    dev_pin = torch.cuda.is_available()

    print("=" * 60)
    print(f"data={args.data}  max_games={args.max_games}  bs={args.batch_size}")
    print("=" * 60)

    # [1] 通常 Dataset
    if args.only in (0, 1):
        ds = TenhouTorchDataset(args.data, max_games=args.max_games,
                                offset=0, event_filter="dahai")
        bench_loader("1) 通常Dataset(aux無)", ds,
                     TenhouTorchDataset.collate_fn,
                     args.batch_size, num_workers=0, pin=dev_pin)

    # [2] Aux Dataset
    if args.only in (0, 2):
        ds = TenhouAuxDataset(args.data, max_games=args.max_games,
                              offset=0, event_filter="dahai")
        bench_loader("2) AuxDataset(aux有)", ds,
                     TenhouTorchDataset.collate_fn_aux,
                     args.batch_size, num_workers=0, pin=dev_pin)

    # [3] ukeire 単体
    if args.only in (0, 3):
        bench_ukeire(args.data, max_games=max(2, args.max_games))

    # [4] worker 重複
    if args.only in (0, 4):
        check_worker_dup(args.data, max_games=args.max_games,
                         batch_size=args.batch_size)

    print("\n" + "=" * 60)
    print("読み方:")
    print("  [2]が[1]より桁違いに遅い → aux計算がボトルネック（想定通り）")
    print("  [3]の shanten 呼び出し回数が数百 → compute_ukeire が主犯")
    print("  [4]で総数が増える → IterableのworkerシャーディングNG。num_workers=0必須")
    print("=" * 60)


if __name__ == "__main__":
    main()
