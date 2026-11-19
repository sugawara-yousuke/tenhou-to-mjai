"""
test_grpo_pipeline.py ― GRPO パイプラインの単体・結合テスト（小モデル・CPU で数分）
    python3 test_grpo_pipeline.py [--model ck/model_bc_best.pt]
--model を渡すとその BC モデルで、無ければランダム初期化の小モデルで回す。
"""
from __future__ import annotations
import _pathsetup  # noqa: F401
import copy, random, sys
import torch

from mahjong_model import MahjongTransformer, load_policy, disable_dropout, samples_to_batch
from self_play import MahjongSimulator, RLContext, resample_wall
from rollout import RolloutConfig, RolloutGroup, collect_groups, collect_group
from grpo import GRPOConfig, GRPOTrainer, compute_advantages


def _model():
    if "--model" in sys.argv:
        return disable_dropout(load_policy(sys.argv[sys.argv.index("--model") + 1], "cpu"))
    torch.manual_seed(0)
    return disable_dropout(MahjongTransformer(d_model=64, nhead=4, enc_layers=2, dec_layers=1,
                                              dim_feedforward=128)).eval()


def _records(sim, seed=0, seat=0):
    rl = RLContext(rl_player=seat, temperature=1.0)
    sim.capture_game(rl, seed=seed)
    return [r for r in rl.records if r.snapshot is not None]


def test_capture_and_snapshots():
    sim = MahjongSimulator(_model(), "cpu")
    recs = _records(sim, seed=1, seat=2)
    assert len(recs) > 20 and all(r.player == 2 for r in recs)
    r = recs[5]
    assert abs(sum(torch.tensor(r.log_probs).exp().tolist()) - 1) < 1e-4
    print(f"[OK] capture: {len(recs)} 判断点 / スナップショット付き")


def test_train_mode_does_not_change_rollout():
    """モデルが train() のままでも、ロールアウトの確率と結果が変わらないこと。"""
    m = _model(); sim = MahjongSimulator(m, "cpu")
    r = _records(sim, seed=2)[10]
    m.eval();  a, _ = sim.playout_from(r.snapshot, RLContext(rl_player=0, forced_first_idx=0))
    m.train(); b, _ = sim.playout_from(r.snapshot, RLContext(rl_player=0, forced_first_idx=0))
    assert a["score_delta"] == b["score_delta"]
    print("[OK] train()/eval() どちらでもロールアウト結果が同一")


def test_common_random_numbers():
    """同じ引き直しの山を使えば、同じ初手の結果は一致する（枝間の差は初手だけに由来）。"""
    sim = MahjongSimulator(_model(), "cpu")
    r = _records(sim, seed=3)[8]
    W = resample_wall(r.snapshot, random.Random(7))
    x, _ = sim.playout_from(W, RLContext(rl_player=0, forced_first_idx=1))
    y, _ = sim.playout_from(W, RLContext(rl_player=0, forced_first_idx=1))
    assert x["score_delta"] == y["score_delta"]
    print("[OK] 共通乱数: 同じ山・同じ初手 → 同じ結果")


def test_advantage_tie_handling():
    a = compute_advantages([0, 0, -0.12, 0, 0, 0, 0, 0])
    assert all(abs(float(a[i])) < 1e-6 for i in range(8) if i != 2) and float(a[2]) < 0
    assert all(abs(float(x)) < 1e-6 for x in compute_advantages([0.3] * 6))
    a3 = compute_advantages([1.0, -1.0, 0.5, -0.5])
    assert float(a3[0]) > 0 > float(a3[1])
    print("[OK] 同点の枝はアドバンテージ 0")


def _synthetic_groups(sim, n=8):
    recs = [r for r in _records(sim, seed=4) if r.n_valid >= 4][:n]
    groups = []
    for r in recs:
        idxs = list(range(4))
        groups.append(RolloutGroup(sample=r.sample, action_idxs=idxs,
                                   old_log_probs=[r.log_probs[i] for i in idxs],
                                   rewards=[1.0, -1.0, 0.0, 0.0], reward_samples=[], kyoku_id=0,
                                   rl_player=0, under_threat=False, entropy=r.entropy))
    return groups


def _logp(m, g, i):
    b = samples_to_batch([g.sample], "cpu")
    with torch.no_grad():
        lg = m(b["sparse"], b["numeric"], b["progression"], b["candidates"], b["cand_feats"])[0]
    return torch.log_softmax(lg, -1)[i].item()


def test_learning_direction_and_metrics():
    """
    (a) 1グループだけで更新 → 高報酬の枝は上がり、低報酬の枝は下がる（厳密に確認）。
    (b) 8グループ同時 → pref_gain が上がる（集計で確認）。
        合成報酬は「候補 index 0 が最良・1 が最悪」と決め打ちしている。連続した手番では同じ牌が
        あるグループでは最良、別のグループでは最悪の位置に来ることがあり、その牌については
        「全グループで上がる/下がる」は原理的に両立しない。だから (b) では個別ではなく集計を見る。
    """
    # (a)
    m = _model(); ref = copy.deepcopy(m)
    g = _synthetic_groups(MahjongSimulator(m, "cpu"), n=1)
    tr = GRPOTrainer(m, ref, torch.optim.AdamW(m.parameters(), lr=3e-4),
                     GRPOConfig(beta_kl=0.0, kl_target=0.0), "cpu")
    b_hi, b_lo = _logp(m, g[0], 0), _logp(m, g[0], 1)
    st = tr.step(g)                     # 勾配が流れているかは初回で見る（同じグループを繰り返すと
    for _ in range(14):                 # PPO のクリップに達して勾配 0 になるのが正しい挙動）
        tr.step(g)
    a_hi, a_lo = _logp(m, g[0], 0), _logp(m, g[0], 1)
    assert a_hi > b_hi, f"高報酬の枝の確率が上がっていない {b_hi:.3f}->{a_hi:.3f}"
    assert a_lo < b_lo, f"低報酬の枝の確率が下がっていない {b_lo:.3f}->{a_lo:.3f}"
    assert st["grad_norm"] > 0, "初回更新で勾配が流れていない"

    # (b)
    m = _model(); ref = copy.deepcopy(m)
    groups = _synthetic_groups(MahjongSimulator(m, "cpu"))
    tr = GRPOTrainer(m, ref, torch.optim.AdamW(m.parameters(), lr=3e-4),
                     GRPOConfig(beta_kl=0.0, kl_target=0.0), "cpu")
    best = {x.sample["possible_actions"][0] for x in groups}
    worst = {x.sample["possible_actions"][1] for x in groups}
    first = tr.step(groups)
    for _ in range(14):
        last = tr.step(groups)
    assert last["pref_gain"] > first["pref_gain"], (first["pref_gain"], last["pref_gain"])
    print(f"[OK] 学習方向: 単独グループ 高 {b_hi:.2f}→{a_hi:.2f} / 低 {b_lo:.2f}→{a_lo:.2f},  "
          f"8グループ pref_gain {first['pref_gain']:+.3f} → {last['pref_gain']:+.3f} "
          f"(最良/最悪で役が入れ替わる牌 {len(best & worst)} 種)")


def test_kl_pulls_back():
    res = {}
    for beta in (0.0, 5.0):
        m = _model(); ref = copy.deepcopy(m)
        groups = _synthetic_groups(MahjongSimulator(m, "cpu"))
        tr = GRPOTrainer(m, ref, torch.optim.AdamW(m.parameters(), lr=3e-4),
                         GRPOConfig(beta_kl=beta, kl_target=0.0), "cpu")
        for _ in range(15):
            st = tr.step(groups)
        res[beta] = st["kl_loss"]
    assert res[5.0] < res[0.0], res
    print(f"[OK] KL 正則化が効く: β=0 → KL {res[0.0]:.4f},  β=5 → KL {res[5.0]:.4f}")


def test_collect_and_step():
    m = _model(); ref = copy.deepcopy(m)
    sim = MahjongSimulator(m, "cpu")
    sim.player_models = {s: ref for s in range(4)}
    cfg = RolloutConfig(G=4, n_playout=2, groups_per_capture=6)
    groups, rs = collect_groups(sim, 3, cfg, random.Random(0), max_captures=6)
    print(f"     収集: {len(groups)} グループ / {rs}")
    if len(groups) >= 2:
        for g in groups:
            assert max(g.rewards) - min(g.rewards) > 0 and len(g.reward_samples[0]) == 2
        st = GRPOTrainer(m, ref, torch.optim.AdamW(m.parameters(), lr=1e-5), GRPOConfig(), "cpu").step(groups)
        assert st["loss"] == st["loss"]
        print(f"[OK] 収集→更新 1 ステップ  (kl={st['kl_loss']:.5f}, |g|={st['grad_norm']:.3f})")
    else:
        print("[--] ランダム初期化モデルでは和了がほぼ起きず差のつくグループが集まらない。"
              "--model に BC モデルを渡して再実行してください。")


if __name__ == "__main__":
    tests = [test_capture_and_snapshots, test_train_mode_does_not_change_rollout,
             test_common_random_numbers, test_advantage_tie_handling,
             test_learning_direction_and_metrics, test_kl_pulls_back, test_collect_and_step]
    ok = 0
    for t in tests:
        try:
            t(); ok += 1
        except Exception as e:
            import traceback
            print(f"[FAIL] {t.__name__}: {e}"); traceback.print_exc()
    print(f"\n{ok}/{len(tests)} passed")
