"""reward_plot.py ― GRPO 学習履歴のグラフ化。matplotlib が無ければ何もしない。"""
from __future__ import annotations


def _ma(xs, w):
    out = []
    for i in range(len(xs)):
        seg = xs[max(0, i - w + 1):i + 1]
        out.append(sum(seg) / len(seg))
    return out


def save_grpo_history(history, save_dir, ma_window=20):
    from pathlib import Path
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not history:
        return
    gs = [h["global_step"] for h in history]
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for k, c in (("pref_gain", "#1f4e79"),):
        ax[0].plot(gs, [h[k] for h in history], color="#b0c4de", lw=0.8)
        ax[0].plot(gs, _ma([h[k] for h in history], ma_window), color=c, lw=2, label=f"{k} (MA{ma_window})")
    ax[0].axhline(0, color="gray", ls="--", lw=0.8); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("pref_gain: policy-weighted reward minus group mean (should rise above 0)")
    ax[1].plot(gs, _ma([h["top_match"] for h in history], ma_window), color="#27ae60", lw=1.6, label="top_match")
    ax[1].plot(gs, _ma([h["tie_rate"] for h in history], ma_window), color="#999999", lw=1.2, label="tie_rate")
    ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(gs, _ma([h["kl_loss"] for h in history], ma_window), color="#c0392b", lw=1.4, label="KL(ref||pi)")
    ax[2].plot(gs, _ma([h["entropy"] for h in history], ma_window), color="#8e44ad", lw=1.4, label="entropy")
    ax[2].set_xlabel("global step"); ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "grpo_history.png", dpi=120)
    plt.close(fig)
