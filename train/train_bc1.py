"""
train_bc1.py  ―  フェーズ1: Behavioral Cloning（統合モデル）
================================================================
鳳凰卓の牌譜から、打牌(リーチ選択込み)・鳴き・槓の判断を1つの候補スコアリングモデルで模倣学習する。

使い方:
  python3 train_bc1.py --data ../data/2024.zip --max-games 50000 --epochs 3 --aux-coef 0.3
  python3 train_bc1.py --data ../data/2024.zip --max-games 300 --epochs 1      # 小規模テスト

[今回の改修]
  - --max-games は「半荘数」そのもの。旧版は len(dataset)=半荘数×500 を半荘数として使っており、
    N を指定すると N×500 半荘を読んでいた。旧 --max-games 100 相当は --max-games 50000。
  - 既定の event_filter を "unified" に。self_play は1つのモデルで打牌・リーチ・鳴き・槓を
    決めるので、BC もその全部を同じ候補スコアリング形式で学習する。
  - 損失は「有効候補だけにかけるラベルスムージング付き CE」。旧 ranked_loss の順位項は
    topk の順位から作った定数で勾配が流れておらず、実質 0.7×CE だった。
    スムージング(既定0.03)は BC 方策の過信(top1=0.995)を和らげ、後段の GRPO の探索余地を残す。
  - 補助タスクの放銃は「実際に切った候補の decoder 出力」に教師を当てる（候補ごとの危険度表現を促す）。
  - シャッフルバッファ(--shuffle-buffer)。val はシャッフルしない。
  - 検証で 打牌 / 鳴き・槓 / リーチ判断 の内訳精度を出す。
  - チェックポイントに config を保存（load_policy が構成を自動復元）。
  - デバッグ専用ユーティリティ(_smoke_test, _locate_inf_layer, debug_batch_quality)は削除。
"""

from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mahjong_model import MahjongTransformer, NUMERIC_DIM, SPARSE_VOCAB_SIZE
from mahjong_tokenizer import (PADDING_ACTION_ID, MAX_ACTION_CANDIDATES,
                               ACT_RIICHI_BASE, ACT_PON_BASE, ACT_PASS_BASE)
from load_dataset import TenhouTorchDataset, TenhouAuxDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BC学習フェーズ")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-games", type=int, default=None, help="使う半荘数（Noneで全件）")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--enc-layers", type=int, default=6)
    p.add_argument("--dec-layers", type=int, default=2)
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints1"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--event-filter", type=str, default="unified")
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--gpu-mem-fraction", type=float, default=None)
    p.add_argument("--save-every", type=int, default=0, help="N ステップごとに中間保存（0で無効）")
    p.add_argument("--label-smoothing", type=float, default=0.03)
    p.add_argument("--shuffle-buffer", type=int, default=8192, help="シャッフルバッファのサンプル数（0で無効）")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数（半荘単位で分割される）")
    p.add_argument("--aux-coef", type=float, default=0.0,
                   help="補助タスク(放銃/他家テンパイ/受け入れ/当たり牌/価値)損失の重み。0.3 程度推奨")
    p.add_argument("--init-from", type=Path, default=None,
                   help="重みだけ読み込んで学習開始（optimizer/scheduler は引き継がない）。"
                        "既存 BC からの微調整に使う")
    p.add_argument("--awr-beta", type=float, default=0.0,
                   help="AWR の温度 β。0 で無効（素のBC）。0.5 程度から試す。"
                        "小さいほど結果の良かった打牌を強く重視する")
    p.add_argument("--awr-wmax", type=float, default=20.0, help="AWR の重みの上限")
    p.add_argument("--no-aux-cache", action="store_true",
                   help="補助ラベルを .aux_cache に溜めない。1エポックだけ回す場合はキャッシュを"
                        "読む機会が無いので、溜めるぶんの RAM（半荘数に比例、5万半荘で約13GB）が"
                        "まるごと無駄になる。--epochs 1 のときは付けること")
    p.add_argument("--awr-warmup", type=int, default=2000,
                   help="最初の N ステップは AWR を切る。価値ヘッドは初期値がランダムなので、"
                        "A = R − V(s) が意味を持つまで素のBCで回す")
    # 旧オプション（受け取るだけ。順位項は勾配が無かったので廃止）
    p.add_argument("--loss-alpha", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--loss-w2", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--loss-w3", type=float, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def policy_loss(logits: torch.Tensor, label: torch.Tensor, candidates: torch.Tensor,
                smoothing: float = 0.0, weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    有効候補だけに分配するラベルスムージング付き CE。

    weights を渡すとサンプルごとに重み付けする（AWR）。重みはバッチ内で平均1に
    正規化してから渡す前提なので、実効学習率は β によって変わらない。
    """
    valid = candidates != PADDING_ACTION_ID
    logp = F.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)
    nll = -logp.gather(1, label.unsqueeze(1)).squeeze(1)
    if smoothing > 0:
        uni = -(logp.masked_fill(~valid, 0.0)).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        per = (1 - smoothing) * nll + smoothing * uni
    else:
        per = nll
    if weights is None:
        return per.mean()
    return (per * weights).mean()


def awr_weights(aux: dict, batch: dict, device: str, beta: float, w_max: float):
    """
    AWR の重み w = clip(exp((R − V)/β), 0, w_max)。V には勾配を流さない。
    戻り値: (weights(B,), 統計dict)。ラベルが無いサンプルは重み1（素のBC扱い）。
    """
    R = batch["aux_ret"].to(device).float()
    ok = batch["aux_ret_valid"].to(device).float()
    V = aux["value"].detach().float()
    adv = (R - V) * ok
    w = torch.exp((adv / max(beta, 1e-6)).clamp(-20.0, 20.0)).clamp(max=w_max)
    w = torch.where(ok > 0, w, torch.ones_like(w))
    w = w / w.mean().clamp(min=1e-6)            # バッチ内で平均1に正規化
    return w, {"awr_w_max": float(w.max()), "awr_w_std": float(w.std()),
               "awr_adv_std": float(adv[ok > 0].std()) if float(ok.sum()) > 1 else 0.0}


def aux_loss(aux: dict, batch: dict, label: torch.Tensor, device: str,
             w_houjuu=1.0, w_tenpai=1.0, w_ukeire=0.1, w_ron=1.0, w_value=1.0,
             ukeire_norm=24.0, ron_pos_weight=20.0):
    valid = batch["aux_valid"].to(device).float()
    n_valid = valid.sum().clamp(min=1.0)
    hj_logit = aux["houjuu"].gather(1, label.unsqueeze(1)).squeeze(1)      # 実際に切った候補の位置
    l_h = F.binary_cross_entropy_with_logits(hj_logit, batch["aux_houjuu"].to(device).float(), reduction="none")
    l_t = F.binary_cross_entropy_with_logits(aux["tenpai"], batch["aux_tenpai_opp"].to(device).float(),
                                             reduction="none").mean(dim=1)
    l_u = F.smooth_l1_loss(aux["ukeire"], batch["aux_ukeire"].to(device).float() / ukeire_norm, reduction="none")
    l_h, l_t, l_u = [(x * valid).sum() / n_valid for x in (l_h, l_t, l_u)]
    loss = w_houjuu * l_h + w_tenpai * l_t + w_ukeire * l_u
    stats = {"aux_houjuu": float(l_h), "aux_tenpai": float(l_t), "aux_ukeire": float(l_u)}

    # 当たり牌ヘッド: (B,3,34) の多ラベル。正例(実際にロンされる牌)は全体の数%なので
    # pos_weight で底上げする。正例が無いサンプルも「どれも当たらない」という教師になる。
    if "aux_ron" in batch and w_ron > 0:
        rv = batch["aux_ron_valid"].to(device).float().view(-1, 1, 1)
        tgt = batch["aux_ron"].to(device).float()
        l_r = F.binary_cross_entropy_with_logits(
            aux["ron"], tgt, reduction="none",
            pos_weight=torch.tensor(ron_pos_weight, device=device))
        l_r = (l_r * rv).sum() / (rv.sum().clamp(min=1.0) * 3 * 34)
        loss = loss + w_ron * l_r
        stats["aux_ron"] = float(l_r)

    # 価値ヘッド: その局の順位期待値の変化 R_局 を回帰する。AWR のベースライン。
    if "aux_ret" in batch and "value" in aux and w_value > 0:
        rv = batch["aux_ret_valid"].to(device).float()
        l_v = F.smooth_l1_loss(aux["value"], batch["aux_ret"].to(device).float(),
                               reduction="none")
        l_v = (l_v * rv).sum() / rv.sum().clamp(min=1.0)
        loss = loss + w_value * l_v
        stats["aux_value"] = float(l_v)
    return loss, stats


def validate_batch(batch: dict) -> tuple[bool, str]:
    label, cands = batch["label"], batch["candidates"]
    if (label < 0).any() or (label >= MAX_ACTION_CANDIDATES).any():
        return False, "label が候補範囲外"
    if (cands[torch.arange(label.size(0)), label] == PADDING_ACTION_ID).any():
        return False, "label が PAD スロットを指している"
    return True, ""


def _forward(model, batch, device, return_aux=False):
    return model(batch["sparse"].to(device), batch["numeric"].to(device),
                 batch["progression"].to(device), batch["candidates"].to(device),
                 batch["cand_feats"].to(device), return_aux=return_aux)


def train_one_epoch(model, loader, optimizer, scheduler, device, args, epoch,
                    steps_per_epoch=None, resume_step=0, best_val_acc=0.0) -> dict:
    model.train()
    tot_loss = tot_correct = tot_n = skipped = 0
    aux_acc: dict = {}
    t0 = time.time()
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [BC]", leave=False, total=steps_per_epoch)
    for step, batch in enumerate(pbar):
        if steps_per_epoch is not None and step >= steps_per_epoch:
            break
        if step < resume_step:
            continue
        ok, reason = validate_batch(batch)
        if not ok:
            skipped += 1
            if skipped <= 5:
                tqdm.write(f"  [WARN] バッチスキップ (step={step}): {reason}")
            continue
        label = batch["label"].to(device)
        cands = batch["candidates"].to(device)
        use_aux = args.aux_coef > 0 and "aux_valid" in batch
        out = _forward(model, batch, device, return_aux=use_aux)
        logits, aux_out = out if use_aux else (out, None)
        w = None
        awr_on = args.awr_beta > 0 and (epoch > 0 or step >= args.awr_warmup)
        if use_aux and awr_on and "aux_ret" in batch:
            w, awr_st = awr_weights(aux_out, batch, device, args.awr_beta, args.awr_wmax)
            for _k, _v in awr_st.items():
                aux_acc[_k] = aux_acc.get(_k, 0.0) + _v
        loss = policy_loss(logits, label, cands, args.label_smoothing, weights=w)
        if use_aux:
            l_aux, aux_st = aux_loss(aux_out, batch, label, device)
            if torch.isfinite(l_aux):
                loss = loss + args.aux_coef * l_aux
                for _k, _v in aux_st.items():
                    aux_acc[_k] = aux_acc.get(_k, 0.0) + _v
                aux_acc["_n"] = aux_acc.get("_n", 0.0) + 1
        if not torch.isfinite(loss):
            skipped += 1; optimizer.zero_grad(); continue
        optimizer.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(gn):
            skipped += 1; optimizer.zero_grad(); continue
        optimizer.step(); scheduler.step()

        B = label.size(0)
        tot_loss += loss.item() * B; tot_n += B
        tot_correct += (logits.argmax(dim=-1) == label).sum().item()
        if args.save_every and (step + 1) % args.save_every == 0:
            save_checkpoint(args.save_dir / "model_bc_mid.pt", model, optimizer, scheduler, epoch,
                            best_val_acc, args, mid_step=step + 1)
        if (step + 1) % args.log_every == 0:
            pbar.set_postfix({"loss": f"{tot_loss/max(tot_n,1):.4f}", "acc": f"{tot_correct/max(tot_n,1):.4f}",
                              "lr": f"{scheduler.get_last_lr()[0]:.2e}", "skip": skipped})
    n_aux = aux_acc.pop("_n", 0.0)
    aux_mean = {k: v / n_aux for k, v in aux_acc.items()} if n_aux else {}
    return {"aux": aux_mean,
            "loss": tot_loss / max(tot_n, 1), "accuracy": tot_correct / max(tot_n, 1),
            "samples": tot_n, "elapsed": time.time() - t0, "skipped_batches": skipped}


@torch.no_grad()
def evaluate(model, loader, device, args) -> dict:
    model.eval()
    agg = {k: [0, 0] for k in ("all", "dahai", "call", "riichi_turn", "riichi_yesno")}
    top3 = 0; tot_loss = 0.0
    for batch in tqdm(loader, desc="  Eval", leave=False):
        ok, _ = validate_batch(batch)
        if not ok:
            continue
        label = batch["label"].to(device); cands = batch["candidates"].to(device)
        logits = _forward(model, batch, device)
        masked = logits.masked_fill(cands == PADDING_ACTION_ID, float("-inf"))
        pred = masked.argmax(dim=-1)
        hit = pred == label
        B = label.size(0)
        tot_loss += policy_loss(logits, label, cands).item() * B
        top3 += (masked.topk(min(3, masked.size(-1)), dim=-1).indices == label.unsqueeze(1)).any(1).sum().item()
        has_pass = (cands == ACT_PASS_BASE).any(1)
        has_riichi = ((cands >= ACT_RIICHI_BASE) & (cands < ACT_PON_BASE)).any(1)
        ar = torch.arange(B, device=device)
        is_r = lambda idx: (cands[ar, idx] >= ACT_RIICHI_BASE) & (cands[ar, idx] < ACT_PON_BASE)
        for key, m, h in (("all", torch.ones_like(hit), hit), ("dahai", ~has_pass, hit), ("call", has_pass, hit),
                          ("riichi_turn", has_riichi, hit), ("riichi_yesno", has_riichi, is_r(pred) == is_r(label))):
            agg[key][0] += int((h & m).sum()); agg[key][1] += int(m.sum())
    n = max(agg["all"][1], 1)
    out = {"loss": tot_loss / n, "top1_acc": agg["all"][0] / n, "top3_acc": top3 / n, "samples": agg["all"][1]}
    for k in ("dahai", "call", "riichi_turn", "riichi_yesno"):
        out[f"acc_{k}"] = agg[k][0] / max(agg[k][1], 1)
    return out


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_acc, args, mid_step=None):
    ck = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
          "scheduler": scheduler.state_dict(), "best_val_acc": best_val_acc,
          "config": {"d_model": args.d_model, "nhead": max(1, args.d_model // 32),
                     "enc_layers": args.enc_layers, "dec_layers": args.dec_layers,
                     "numeric_dim": NUMERIC_DIM, "sparse_vocab": SPARSE_VOCAB_SIZE,
                     "event_filter": args.event_filter}}
    if mid_step is not None:
        ck.update({"mid_epoch": True, "step": mid_step})
    torch.save(ck, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
    if ck.get("mid_epoch"):
        return ck.get("epoch", 0), ck.get("best_val_acc", 0.0), ck.get("step", 0)
    return ck.get("epoch", 0) + 1, ck.get("best_val_acc", 0.0), 0


def main():
    args = parse_args()
    if args.loss_alpha is not None or args.loss_w2 is not None or args.loss_w3 is not None:
        print("[INFO] --loss-alpha/--loss-w2/--loss-w3 は廃止しました（順位項には勾配が流れていませんでした）。")
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                             else "cpu")
    print(f"device: {device}")
    if args.gpu_mem_fraction is not None and device == "cuda":
        torch.cuda.set_per_process_memory_fraction(max(0.05, min(args.gpu_mem_fraction, 1.0)), 0)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    DS = TenhouAuxDataset if args.aux_coef > 0 else TenhouTorchDataset
    total_games = DS(args.data, max_games=args.max_games, event_filter=args.event_filter).n_games()
    val_games = max(1, int(total_games * args.val_ratio))
    train_games = max(total_games - val_games, 1)
    print(f"半荘数: {total_games:,}  (train {train_games:,} / val {val_games:,})  filter={args.event_filter}"
          + (f"  aux_coef={args.aux_coef}" if args.aux_coef > 0 else ""))

    # TenhouTorchDataset は use_cache を取らないので、aux 版のときだけ渡す
    cache_kw = {} if DS is TenhouTorchDataset else {"use_cache": not args.no_aux_cache}
    train_ds = DS(args.data, max_games=train_games, offset=0, event_filter=args.event_filter,
                  shuffle_buffer=args.shuffle_buffer, **cache_kw)
    val_ds = DS(args.data, max_games=val_games, offset=train_games, event_filter=args.event_filter,
                **cache_kw)
    if args.no_aux_cache and DS is TenhouAuxDataset:
        print("補助ラベルのキャッシュ: 無効（毎エポック再計算。RAM を使わない）")
    mk = lambda ds, bs: DataLoader(ds, batch_size=bs, collate_fn=TenhouTorchDataset.collate_fn,
                                   num_workers=args.num_workers, pin_memory=(device == "cuda"))
    train_loader, val_loader = mk(train_ds, args.batch_size), mk(val_ds, args.batch_size * 2)

    model = MahjongTransformer(d_model=args.d_model, nhead=max(1, args.d_model // 32),
                               enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                               dim_feedforward=args.d_model * 4, dropout=0.1).to(device)
    print(f"パラメータ数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M  numeric_dim={NUMERIC_DIM}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95))
    steps_per_epoch = max(len(train_ds) // args.batch_size, 1)     # 概算（実データが先に尽きればそこで終わる）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps_per_epoch * args.epochs, 1), eta_min=args.lr * 0.1)

    start_epoch, best_val_acc, resume_step = 0, 0.0, 0
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"重みを初期値として読み込み: {args.init_from}")
        if missing:
            print(f"  新規ヘッド（ランダム初期化）: {[k.split('.')[0] for k in missing][:6]}")
        if unexpected:
            print(f"  使われなかった重み: {[k.split('.')[0] for k in unexpected][:6]}")
    if args.resume:
        start_epoch, best_val_acc, resume_step = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        print(f"再開: epoch {start_epoch+1}" + (f" step {resume_step}" if resume_step else ""))

    for epoch in range(start_epoch, start_epoch + args.epochs):
        try:
            tr = train_one_epoch(model, train_loader, optimizer, scheduler, device, args, epoch,
                                 steps_per_epoch=None, resume_step=resume_step if epoch == start_epoch else 0,
                                 best_val_acc=best_val_acc)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[OOM] --batch-size を下げるか --gpu-mem-fraction を調整してください。"
                      + (f" 再開: --resume {args.save_dir}/model_bc_mid.pt" if args.save_every else ""))
            raise
        vl = evaluate(model, val_loader, device, args)
        print(f"Epoch {epoch+1:3d} | train loss={tr['loss']:.4f} acc={tr['accuracy']:.4f} | "
              f"val loss={vl['loss']:.4f} top1={vl['top1_acc']:.4f} top3={vl['top3_acc']:.4f} | "
              f"打牌={vl['acc_dahai']:.4f} 鳴き/槓={vl['acc_call']:.4f} "
              f"リーチ手番={vl['acc_riichi_turn']:.4f} リーチ/ダマ一致={vl['acc_riichi_yesno']:.4f} | "
              + ("aux[" + " ".join(f"{k.replace('aux_','')}={v:.4f}" for k, v in tr.get("aux", {}).items()) + "] | "
                 if tr.get("aux") else "")
              + f"{tr['elapsed']:.0f}s skip={tr['skipped_batches']}")
        save_checkpoint(args.save_dir / f"model_bc_epoch{epoch+1}.pt", model, optimizer, scheduler,
                        epoch, best_val_acc, args)
        if vl["top1_acc"] > best_val_acc:
            best_val_acc = vl["top1_acc"]
            save_checkpoint(args.save_dir / "model_bc_best.pt", model, optimizer, scheduler,
                            epoch, best_val_acc, args)
            print(f"  → best 更新: top1={best_val_acc:.4f}")
    print(f"\n学習完了。次: python3 train_grpo.py --bc-model {args.save_dir}/model_bc_best.pt")


if __name__ == "__main__":
    main()
