"""
grpo.py  ―  Group Relative Policy Optimization の更新ロジック
================================================================
同じ局面から G 個の初手を強制して打ち切り、グループ内の相対報酬でアドバンテージを作る。

[今回の改修]
  - 1回の更新に複数グループ（=複数局面）を使う。旧版は1局面(G行)だけで optimizer.step して
    いた。Adam は勾配の大きさを正規化するので、1局面ぶんのノイズでも毎回 lr ぶん動いてしまい、
    KL に引き戻されるだけのランダムウォークになっていた。
  - 1グループ＝1局面なので forward は局面あたり1回（旧版は同じ局面を G 行複製していた）。
  - 更新もロールアウトも eval モード固定（dropout 無効）。旧版は更新後に train モードのまま
    ロールアウトに入り、dropout が効いた状態で対局していた。
  - 低確率手の減衰は「アドバンテージが負のとき」だけ。旧版は正のときも減衰させており、
    BC が 0.99 で確信している局面では、良かった代替手の勾配までほぼ 0 に潰していた。
  - old_log_prob は収集時に記録した値を使う（mini_epochs>1 でも PPO のクリップが正しく効く）。
  - 学習が進んでいるかの指標を追加:
      pref_gain = Σ_a π(a|評価した枝) r(a) − mean_a r(a)   … 方策が良い枝を好んでいれば正
      top_match = 方策の最有力枝 == 報酬最大の枝 だった割合
    mean_reward は山の運で上下するだけで、学習の進み具合は分からない。
"""

from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mahjong_model import samples_to_batch, PADDING_ACTION_ID


@dataclass
class GRPOConfig:
    clip_eps:      float = 0.2
    beta_kl:       float = 0.05
    grad_clip:     float = 1.0
    mini_epochs:   int   = 1
    kl_target:     float = 0.05      # 0 以下で適応制御なし
    kl_adapt_rate: float = 1.2       # 1.5 だと β が1stepごとに大きく振れる
    beta_kl_min:   float = 0.01
    beta_kl_max:   float = 0.5       # 5.0 だと更新の大半が「BC に戻す」方向に使われる
    adv_clip:      float = 4.0       # 2.0 だと満貫級の和了/放銃が頭打ちになり、押しが過小評価される
    prob_floor:    float = 0.05      # A<0 かつ π(a)<floor の枝は (π/floor) 倍に減衰
    temperature:   float = 1.0       # ロールアウトと同じ温度
    entropy_coef:  float = 0.0
    tie_eps:       float = 1e-6


def compute_advantages(rewards: list[float], tie_eps: float = 1e-6) -> torch.Tensor:
    """
    グループ内相対アドバンテージ（同点ロバスト版）。基準は中央値、同点の枝は 0。
    麻雀では「大半の枝が同じ結果で、少数だけ違う」グループが多い。平均基準だと同点の枝すべてに
    符号つきの値が付き、たまたま同点だった手を一斉に動かしてしまう。
    """
    r = torch.tensor(rewards, dtype=torch.float32)
    if float(r.max() - r.min()) < tie_eps:
        return torch.zeros_like(r)
    centered = r - r.median()
    # 分母に下限を置く。std で割るだけだと「5 枝同値・1 枝だけ僅差」のグループでも
    # ±2（クリップ上限）まで膨らみ、満貫放銃の差と同じ重みで学習してしまう。
    adv = centered / max(float(centered.std()), 0.1)
    return adv.masked_fill(centered.abs() < tie_eps, 0.0)


class GRPOTrainer:
    def __init__(self, model, ref_model, optimizer, config: GRPOConfig = None, device: str = "cpu"):
        self.model, self.ref_model, self.optimizer = model, ref_model, optimizer
        self.config = config or GRPOConfig()
        self.device = device
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

    def _forward(self, model, batch):
        return model(batch["sparse"], batch["numeric"], batch["progression"],
                     batch["candidates"], batch["cand_feats"])

    def step(self, groups: list) -> dict:
        cfg = self.config
        dev = self.device
        batch = samples_to_batch([g.sample for g in groups], dev)
        pad = batch["candidates"] == PADDING_ACTION_ID

        rows, cols, advs, olds, rews = [], [], [], [], []
        for b, g in enumerate(groups):
            a = compute_advantages(g.rewards, cfg.tie_eps).clamp(-cfg.adv_clip, cfg.adv_clip)
            rows += [b] * len(g.action_idxs); cols += list(g.action_idxs)
            advs.append(a); olds += list(g.old_log_probs); rews += list(g.rewards)
        rows = torch.tensor(rows, device=dev); cols = torch.tensor(cols, device=dev)
        adv = torch.cat(advs).to(dev)
        old_lp = torch.tensor(olds, dtype=torch.float32, device=dev)
        rew = torch.tensor(rews, dtype=torch.float32, device=dev)

        self.model.eval()                      # dropout 無効のまま更新する（eval でも勾配は流れる）
        with torch.no_grad():
            ref_logp = F.log_softmax(self._forward(self.ref_model, batch).masked_fill(pad, float("-inf")), -1)
            p_ref = ref_logp.exp()

        stats_acc = []
        for it in range(max(1, cfg.mini_epochs)):
            logits = self._forward(self.model, batch).masked_fill(pad, -1e9)
            logp_T = F.log_softmax(logits / max(cfg.temperature, 1e-6), -1)
            logp_1 = F.log_softmax(logits, -1)
            new_lp = logp_T[rows, cols]
            ratio = (new_lp - old_lp).clamp(-10, 10).exp()
            with torch.no_grad():
                down = torch.where(adv < 0, (new_lp.exp() / cfg.prob_floor).clamp(max=1.0),
                                   torch.ones_like(adv))
            a_eff = adv * down
            pol = -torch.min(ratio * a_eff, ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * a_eff).mean()
            kl = (p_ref * (ref_logp.masked_fill(pad, 0.0) - logp_1).masked_fill(pad, 0.0)).sum(-1).mean()
            ent = -(logp_1.exp() * logp_1.masked_fill(pad, 0.0)).sum(-1).mean()
            loss = pol + cfg.beta_kl * kl - cfg.entropy_coef * ent

            self.optimizer.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            if torch.isfinite(gnorm):
                self.optimizer.step()

            with torch.no_grad():
                if it == 0:
                    gains, match = [], []
                    for b, g in enumerate(groups):
                        m = rows == b
                        pi = F.softmax(new_lp[m], 0); r = rew[m]
                        gains.append(float((pi * r).sum() - r.mean()))
                        match.append(float(int(pi.argmax()) == int(r.argmax())))
                    pref_gain, top_match = sum(gains) / len(gains), sum(match) / len(match)
                stats_acc.append({"loss": loss.item(), "policy_loss": pol.item(), "kl_loss": kl.item(),
                                  "entropy": ent.item(), "grad_norm": float(gnorm),
                                  "clip_frac": ((ratio - 1).abs() > cfg.clip_eps).float().mean().item()})

        out = {k: sum(s[k] for s in stats_acc) / len(stats_acc) for k in stats_acc[0]}
        out.update({"pref_gain": pref_gain, "top_match": top_match,
                    "mean_reward": float(rew.mean()), "n_groups": len(groups),
                    "frac_nonzero_adv": float((adv != 0).float().mean())})
        if cfg.kl_target and cfg.kl_target > 0:
            if out["kl_loss"] > 2.0 * cfg.kl_target:
                cfg.beta_kl = min(cfg.beta_kl * cfg.kl_adapt_rate, cfg.beta_kl_max)
            elif out["kl_loss"] < 0.5 * cfg.kl_target:
                cfg.beta_kl = max(cfg.beta_kl / cfg.kl_adapt_rate, cfg.beta_kl_min)
        out["beta_kl"] = cfg.beta_kl
        return out
