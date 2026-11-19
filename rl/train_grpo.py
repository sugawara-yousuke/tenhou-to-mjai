"""
train_grpo.py  ―  フェーズ2: GRPO（統合版）
================================================================
旧 train_grpo.py / train_grpo_plan3(_plot).py / train_grpo_genbutsu(_iqn).py を1本にまとめた。

  python3 train_grpo.py --bc-model checkpoints1/model_bc_best.pt \
      --epochs 10 --steps-per-epoch 50 --groups-per-step 16 --G 6 --n-playout 3

  # まず動作確認（数分）
  python3 train_grpo.py --bc-model checkpoints1/model_bc_best.pt \
      --epochs 1 --steps-per-epoch 3 --groups-per-step 4 --n-playout 2

設計の要点（詳細は rollout.py / grpo.py / reward.py の冒頭）:
  - 分岐はスナップショットから再開。N 本の山の引き直しを全枝で共有（共通乱数）。
  - 1回の更新に groups-per-step 個の局面を使う。
  - 対戦相手は既定で BC(ref) 固定。環境が動かないので学習信号が安定し、eval_vs_bc と同じ物差しになる。
    --opponents self で従来どおりの自己対戦。
  - rl_player の席は毎半荘ランダム（旧版は常に起家で、北家スタートやラス親を経験しなかった）。
  - dropout は恒久的に 0（disable_dropout）。ロールアウトと更新で確率が一致する。
  - critic は使わない。同一局面の G 枝を比べる GRPO では「グループ平均」がその局面の最良の
    ベースラインで、V(s) を足しても全枝に同じ定数が乗るだけ（critic が外すと全枝が一斉に
    正/負になり、むしろ害）。共有 encoder に value 回帰の勾配を流すのも方策側のノイズになる。
  - ログの見方: pref_gain と top_match が上がっていけば学習が進んでいる。R(mean_reward) は
    山の運で上下するだけなので見ない。tie% は「どの枝も同じ結果で捨てた分岐点」の割合。
"""

from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
import argparse
import copy
import csv
import random
import time
from pathlib import Path

import torch

from mahjong_model import load_policy, disable_dropout
from self_play import MahjongSimulator
from rollout import RolloutConfig, collect_groups
from grpo import GRPOConfig, GRPOTrainer


def parse_args():
    p = argparse.ArgumentParser("GRPO 学習")
    p.add_argument("--bc-model", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--groups-per-step", type=int, default=16)
    p.add_argument("--G", type=int, default=6)
    p.add_argument("--n-playout", type=int, default=3)
    p.add_argument("--horizon", type=int, default=0,
                   help="分岐から N 巡で打ち切り、kyoku_value の期待局収支で評価（0で従来どおり局終了まで）")
    p.add_argument("--resample", choices=["wall", "none"], default="wall")
    p.add_argument("--continuation", choices=["greedy", "sample"], default="greedy")
    p.add_argument("--reward", choices=["placement", "near"], default="placement")
    p.add_argument("--opponents", choices=["ref", "self"], default="ref")
    p.add_argument("--branch-weighting", choices=["entropy", "uniform"], default="entropy")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta-kl", type=float, default=0.05)
    p.add_argument("--kl-target", type=float, default=0.05)
    p.add_argument("--mini-epochs", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=0, help="N ステップごとに ref と対戦評価（0で無効）")
    p.add_argument("--eval-games", type=int, default=60)
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints1/grpo"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def save_model(path, model, args, extra=None):
    ck = {"model": model.state_dict(), "config": {"source": str(args.bc_model)}}
    ck.update(extra or {})
    torch.save(ck, path)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); torch.manual_seed(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    model = disable_dropout(load_policy(args.bc_model, device))
    ref_model = copy.deepcopy(model).eval()
    for p_ in ref_model.parameters():
        p_.requires_grad_(False)
    for p_ in model.parameters():
        p_.requires_grad_(True)
    # 補助ヘッドは RL では使わない
    params = [p_ for n, p_ in model.named_parameters() if not n.startswith("aux_")]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    trainer = GRPOTrainer(model, ref_model, optimizer,
                          GRPOConfig(beta_kl=args.beta_kl, kl_target=args.kl_target,
                                     mini_epochs=args.mini_epochs, temperature=args.temperature), device)
    rcfg = RolloutConfig(G=args.G, n_playout=args.n_playout, resample=args.resample,
                         temperature=args.temperature, continuation=args.continuation,
                         reward=args.reward, branch_weighting=args.branch_weighting,
                         horizon=args.horizon)
    sim = MahjongSimulator(model, device, watch=False)
    rng = random.Random(args.seed)
    print(f"device={device}  opponents={args.opponents}  reward={args.reward}  "
          f"G={args.G} N={args.n_playout}({args.resample})  horizon={args.horizon or 'full'}  "
          f"groups/step={args.groups_per_step}  lr={args.lr}")

    hist_path = args.save_dir / "grpo_history.csv"
    cols = ["global_step", "epoch", "step", "pref_gain", "top_match", "policy_loss", "kl_loss", "beta_kl",
            "entropy", "grad_norm", "frac_nonzero_adv", "mean_reward", "n_groups", "tie_rate", "captures", "sec"]
    history = []
    gstep = 0
    for epoch in range(args.epochs):
        ma = {"pref_gain": [], "top_match": [], "kl_loss": []}
        for step in range(args.steps_per_epoch):
            t0 = time.time()
            # rl_player 以外の席は毎回張り直す（席は capture ごとに変わるので全席に ref を入れておけばよい）
            sim.player_models = {s: ref_model for s in range(4)} if args.opponents == "ref" else None
            groups, rs = collect_groups(sim, args.groups_per_step, rcfg, rng)
            if len(groups) < 2:
                print(f"  step {step}: グループが集まらずスキップ ({rs})"); gstep += 1; continue
            st = trainer.step(groups)
            st.update({"global_step": gstep, "epoch": epoch + 1, "step": step,
                       "tie_rate": rs["tie"] / max(rs["tried"], 1), "captures": rs["captures"],
                       "sec": time.time() - t0})
            history.append(st); gstep += 1
            for k in ma:
                ma[k].append(st[k])
            m = lambda k: sum(ma[k][-20:]) / len(ma[k][-20:])
            print(f"  ep{epoch+1} step{step:3d} | pref_gain {st['pref_gain']:+.4f} (MA {m('pref_gain'):+.4f})  "
                  f"top_match {st['top_match']:.2f} (MA {m('top_match'):.2f})  kl {st['kl_loss']:.4f} "
                  f"β {st['beta_kl']:.3f}  pol {st['policy_loss']:+.4f}  |g| {st['grad_norm']:.2f}  "
                  f"tie {st['tie_rate']*100:.0f}%  {st['sec']:.0f}s", flush=True)

            if args.eval_every and gstep % args.eval_every == 0:
                from eval_vs_bc import run_match
                r = run_match(model, ref_model, device, args.eval_games, seed=12345)
                hj = lambda s: s["houju"] / max(s["kyoku"], 1) * 100
                print(f"  [eval vs ref] 平均順位差 {r['rank_diff']:+.3f} ± {r['se']:.3f}  "
                      f"放銃率 A {hj(r['A']):.2f}% / ref {hj(r['B']):.2f}%", flush=True)

        with open(hist_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(history)
        save_model(args.save_dir / f"model_grpo_epoch{epoch+1}.pt", model, args, {"epoch": epoch})
        try:
            from reward_plot import save_grpo_history
            save_grpo_history(history, args.save_dir)
        except Exception as e:
            print(f"  (グラフ保存スキップ: {e})")
        if ma["pref_gain"]:
            print(f"→ Epoch {epoch+1}: pref_gain {sum(ma['pref_gain'])/len(ma['pref_gain']):+.4f}  "
                  f"top_match {sum(ma['top_match'])/len(ma['top_match']):.3f}  "
                  f"kl {sum(ma['kl_loss'])/len(ma['kl_loss']):.4f}")
    save_model(args.save_dir / "model_grpo_final.pt", model, args)
    print(f"学習完了: {args.save_dir}/model_grpo_final.pt\n"
          f"次: python3 eval_vs_bc.py --model-a {args.save_dir}/model_grpo_final.pt --model-b {args.bc_model}")


if __name__ == "__main__":
    main()
