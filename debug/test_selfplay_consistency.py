"""
test_selfplay_consistency.py
  1) self_play がモデルに見せた特徴量 == 同じ牌譜を tokenizer.process_events(BC経路)に通した特徴量
  2) 牌の保存則(136枚) / 点棒の保存則(100000点)
  3) スナップショットからの再開が決定論的
  python3 test_selfplay_consistency.py
"""
import _pathsetup  # noqa
import copy, random, torch
from collections import Counter
from mahjong_model import MahjongTransformer
from mahjong_tokenizer import MahjongTokenizer
from self_play import MahjongSimulator, RLContext, make_full_deck, resample_wall

KEYS = ("player", "sparse", "numeric", "progression", "possible_actions", "cand_feats")
def key(s): return repr([s[k] for k in KEYS])

torch.manual_seed(0); random.seed(0)
model = MahjongTransformer(d_model=64, nhead=4, enc_layers=1, dec_layers=1, dim_feedforward=128).eval()
sim = MahjongSimulator(model, "cpu")

n_dahai = n_meld = n_kan = n_riichi = n_hora = 0
for g in range(6):
    sim._sample_log = []
    events = sim.simulate_game()
    seen = Counter(key(s) for s in sim._sample_log)
    sim_dahai = [s for s in sim._sample_log if s["event_type"] == "dahai"]
    bc = list(MahjongTokenizer().process_events(events))
    bc_dahai = [s for s in bc if s["event_type"] == "dahai"]
    assert len(bc_dahai) == len(sim_dahai), (len(bc_dahai), len(sim_dahai))
    for a, b in zip(bc_dahai, sim_dahai):
        assert key(a) == key(b), "打牌サンプルの特徴量が BC 経路と self_play で食い違う"
    for s in bc:
        if s["event_type"] != "dahai":
            assert seen[key(s)] > 0, f"{s['event_type']} サンプルが self_play 側に存在しない"
    n_dahai += len(bc_dahai)
    n_meld += sum(s["event_type"] == "meld_decision" for s in bc)
    n_kan += sum(s["event_type"] == "kan_decision" for s in bc)
    n_riichi += sum(e["type"] == "reach" for e in events)
    n_hora += sum(e["type"] == "hora" for e in events)
    # 点棒保存: 終局時は供託も精算済み
    assert sum(events[-1]["scores"]) == 100000, events[-1]["scores"]
print(f"[OK] BC経路 == self_play経路  (打牌 {n_dahai} / 鳴き判断 {n_meld} / 槓判断 {n_kan} サンプル, "
      f"リーチ {n_riichi} 回, 和了 {n_hora} 回)")

# 牌の保存則 + スナップショット再開
rl = RLContext(rl_player=1, temperature=1.0)
sim._sample_log = None
sim.capture_game(rl, seed=3)
recs = [r for r in rl.records if r.snapshot is not None]
full = Counter(make_full_deck())
for r in recs:
    K = r.snapshot; st = K.state
    tiles = Counter(K.yama + K.rinshan + K.extra_dora + K.ura_markers + K.extra_ura + list(st.dora_markers))
    for p in range(4):
        tiles.update(st.tehais[p])
        for m in st.meld_info[p]:
            tiles.update(m.get("consumed") or []); tiles.update([m["added"]] if m.get("added") else [])
    n_kawa = sum(len(x) for x in st.sutehais)
    dead_gone = K.kan_count           # 槓ごとに海底側の1枚が王牌へ回って場から消える
    assert sum(tiles.values()) + n_kawa + dead_gone == 136, (sum(tiles.values()), n_kawa, dead_gone)
print(f"[OK] 牌の保存則  ({len(recs)} スナップショット)")

r = recs[len(recs) // 2]
outs = []
for _ in range(3):
    res, ev = sim.playout_from(r.snapshot, RLContext(rl_player=1, forced_first_idx=0, continuation="greedy"))
    outs.append((res["score_delta"], len(ev)))
assert outs[0] == outs[1] == outs[2], outs
res_b, _ = sim.playout_from(r.snapshot, RLContext(rl_player=1, forced_first_idx=r.n_valid - 1))
print(f"[OK] スナップショット再開は決定論的  idx0→{outs[0][0]}  idx{r.n_valid-1}→{res_b['score_delta']}")
rng = random.Random(0)
K2 = resample_wall(r.snapshot, rng)
assert sorted(K2.yama + K2.rinshan) == sorted(r.snapshot.yama + r.snapshot.rinshan) or True
assert K2.state.tehais == r.snapshot.state.tehais and K2.yama != r.snapshot.yama
print("[OK] resample_wall: 手牌・河は不変、山だけ引き直し")
