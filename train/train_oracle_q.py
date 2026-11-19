"""
train_oracle_q.py  ―  オラクル Q 関数の学習

  # 本命（オラクルあり）
  python3 train/train_oracle_q.py --data data/2024.zip --bc-model train/checkpoints1_ronhead_off/model_bc_best.pt --max-games 150000 --save-dir train/oracle_q --save-every 5000

  # 対照（オラクルなし。同じ構造・同じデータで R² の差を見る）
  python3 train/train_oracle_q.py --data data/2024.zip --bc-model train/checkpoints1_ronhead_off/model_bc_best.pt --max-games 150000 --save-dir train/oracle_q_noor --no-oracle --save-every 5000

損失:
  L = smooth_L1( Q(s, a_実際) , R_局 )  ＋  bc_coef × CE(π(s), a_実際)
  R_局 は aux_labels.attach_return_labels の局リターン（AWR で使ったものと同じ）。
  CE は decoder の表現を崩さないための正則化（このモデルの π はオラクル情報を見ているので、
  方策としては使わない）。

検証:
  val の R² = 1 − MSE(Q(s,a), R) / Var(R)。オラクルあり／なしの差が「相手の手牌が見えると
  局の結果がどれだけ説明できるか」。TanyaoDojo の報告では全員の手牌を見る critic は価値の
  当てはまりが約 100 倍良くなった。
"""
from __future__ import annotations
import _pathsetup  # noqa: F401
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mahjong_tokenizer import PADDING_ACTION_ID
from oracle_q import OracleQDataset, build_oracle_q, save_oracle_q


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--bc-model", type=Path, required=True, help="初期値にする BC（encoder/decoder を流用）")
    p.add_argument("--max-games", type=int, default=150000)
    p.add_argument("--val-games", type=int, default=2000, help="--max-games の末尾から取る検証用半荘数")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--bc-coef", type=float, default=0.1)
    p.add_argument("--no-oracle", action="store_true")
    p.add_argument("--save-dir", type=Path, default=Path("train/oracle_q"))
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=20000, help="N ステップごとに val の R² を出す（0で無効）")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--shuffle-buffer", type=int, default=30000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--gpu-mem-fraction", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", type=Path, default=None, help="このスクリプトが保存した oracle_q_*.pt から再開")
    return p.parse_args()


def _fwd(model, b, device):
    return model(b["sparse"].to(device), b["numeric"].to(device), b["progression"].to(device),
                 b["candidates"].to(device), b["cand_feats"].to(device), b["oracle"].to(device))


def losses(model, b, device, bc_coef):
    label = b["label"].to(device)
    cands = b["candidates"].to(device)
    logits, q = _fwd(model, b, device)
    R = b["aux_ret"].to(device).float()
    ok = (b["aux_ret_valid"].to(device) > 0).float()
    q_a = q.gather(1, label.unsqueeze(1)).squeeze(1)
    lq = (F.smooth_l1_loss(q_a, R, reduction="none") * ok).sum() / ok.sum().clamp(min=1.0)
    valid = cands != PADDING_ACTION_ID
    logp = F.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)
    lbc = -logp.gather(1, label.unsqueeze(1)).mean()
    return lq + bc_coef * lbc, lq, lbc, q_a.detach(), R, ok


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = 0; s_r = s_r2 = s_e2 = 0.0; s_q = s_q2 = 0.0
    for b in loader:
        _, _, _, q_a, R, ok = losses(model, b, device, 0.0)
        m = ok > 0
        r = R[m].double(); qa = q_a[m].double()
        n += int(m.sum())
        s_r += float(r.sum()); s_r2 += float((r * r).sum()); s_e2 += float(((qa - r) ** 2).sum())
        s_q += float(qa.sum()); s_q2 += float((qa * qa).sum())
    model.train()
    if n < 2:
        return {"n": n}
    var_r = s_r2 / n - (s_r / n) ** 2
    return {"n": n, "mse": s_e2 / n, "var_r": var_r, "r2": 1 - (s_e2 / n) / max(var_r, 1e-12),
            "q_std": math.sqrt(max(s_q2 / n - (s_q / n) ** 2, 0.0))}


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu_mem_fraction and device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, 0)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    train_games = args.max_games - args.val_games
    tr_ds = OracleQDataset(args.data, max_games=train_games, offset=0, shuffle_buffer=args.shuffle_buffer)
    va_ds = OracleQDataset(args.data, max_games=args.val_games, offset=train_games)
    mk = lambda ds, bs: DataLoader(ds, batch_size=bs, collate_fn=OracleQDataset.collate_fn,
                                   num_workers=args.num_workers, pin_memory=(device == "cuda"))
    tr_ld, va_ld = mk(tr_ds, args.batch_size), mk(va_ds, args.batch_size * 2)

    model = build_oracle_q(args.bc_model, device, use_oracle=not args.no_oracle)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95))
    est_steps = max(train_games * 510 // args.batch_size, 1) * args.epochs   # 打牌 ≈ 510 / 半荘
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=est_steps, eta_min=args.lr * 0.1)
    step0 = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        step0 = ck.get("step", 0)
        print(f"再開: step {step0}（データは先頭から読み直すので、同じ局面をもう一度見る）")
    tag = "オラクルなし（対照）" if args.no_oracle else "オラクルあり"
    print(f"{tag}  train {train_games:,} 半荘 / val {args.val_games:,} 半荘  推定 {est_steps:,} ステップ  "
          f"パラメータ {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    def save(name, step):
        save_oracle_q(args.save_dir / name, model, args.bc_model,
                      {"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step,
                       "args": {k: str(v) for k, v in vars(args).items()}})

    step = step0
    best_r2 = -1e9
    for ep in range(args.epochs):
        t0 = time.time(); acc = {"lq": 0.0, "lbc": 0.0, "n": 0}
        pbar = tqdm(tr_ld, desc=f"Epoch {ep+1}", total=est_steps // args.epochs)
        for b in pbar:
            loss, lq, lbc, _, _, _ = losses(model, b, device, args.bc_coef)
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            acc["lq"] += float(lq); acc["lbc"] += float(lbc); acc["n"] += 1
            if step % args.log_every == 0:
                pbar.set_postfix({"Lq": f"{acc['lq']/acc['n']:.4f}", "Lbc": f"{acc['lbc']/acc['n']:.3f}",
                                  "lr": f"{sched.get_last_lr()[0]:.1e}"})
                acc = {"lq": 0.0, "lbc": 0.0, "n": 0}
            if args.save_every and step % args.save_every == 0:
                save("oracle_q_mid.pt", step)
            if args.eval_every and step % args.eval_every == 0:
                v = evaluate(model, va_ld, device)
                tqdm.write(f"  step {step:,}  val R²={v.get('r2', float('nan')):.4f}  "
                           f"MSE={v.get('mse', float('nan')):.5f}  Var(R)={v.get('var_r', float('nan')):.5f}  "
                           f"std(Q)={v.get('q_std', float('nan')):.4f}  n={v['n']:,}")
                if v.get("r2", -1e9) > best_r2:
                    best_r2 = v["r2"]; save("oracle_q_best.pt", step)
        v = evaluate(model, va_ld, device)
        print(f"Epoch {ep+1}  val R²={v.get('r2', float('nan')):.4f}  MSE={v.get('mse', float('nan')):.5f}  "
              f"Var(R)={v.get('var_r', float('nan')):.5f}  std(Q)={v.get('q_std', float('nan')):.4f}  "
              f"{time.time()-t0:.0f}s")
        save(f"oracle_q_epoch{ep+1}.pt", step)
        if v.get("r2", -1e9) > best_r2:
            best_r2 = v["r2"]; save("oracle_q_best.pt", step)
    print(f"完了。best val R² = {best_r2:.4f}  → {args.save_dir}/oracle_q_best.pt")


if __name__ == "__main__":
    main()
