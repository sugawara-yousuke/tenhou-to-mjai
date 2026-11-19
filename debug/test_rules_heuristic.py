"""
test_rules_heuristic.py ― 向聴最小の簡易方策で多数局を回し、ルールエンジン(和了/点数/槓/流局/終局)を検証する。
  python3 test_rules_heuristic.py
"""
import _pathsetup  # noqa
import random, torch
from collections import Counter
from mahjong_model import MahjongTransformer
from mahjong_tokenizer import (MahjongTokenizer, shanten_number, TILE_STR_TO_ID, _tid_norm,
                               ACT_RIICHI_BASE, ACT_PON_BASE, ACT_PASS_BASE, PADDING_ACTION_ID)
from heuristic_sim import HeuristicSim

random.seed(1); torch.manual_seed(1)
sim = HeuristicSim(None, "cpu")
KEYS = ("player", "sparse", "numeric", "progression", "possible_actions", "cand_feats")
key = lambda s: repr([s[k] for k in KEYS])
stat = Counter(); yaku_seen = Counter(); hans = []
for g in range(40):
    sim._sample_log = []
    ev = sim.simulate_game()
    assert sum(ev[-1]["scores"]) == 100000, ev[-1]["scores"]
    bc = [s for s in MahjongTokenizer().process_events(ev) if s["event_type"] == "dahai"]
    sm = [s for s in sim._sample_log if s["event_type"] == "dahai"]
    assert len(bc) == len(sm) and all(key(x) == key(y) for x, y in zip(bc, sm))
    # リーチ宣言のラベルが BC 側で RIICHI トークンになっているか
    n_reach = sum(e["type"] == "reach" for e in ev)
    assert sum(s.get("riichi_label") == 1 for s in bc) == n_reach
    for e in ev:
        stat[e["type"]] += 1
        if e["type"] == "hora":
            assert e["yakuman"] or (e["han"] >= 1 and e["yaku"]), e
            stat["win_tsumo" if e["target"] is None else "win_ron"] += 1
            yaku_seen.update(e["yaku"]); hans.append(e["han"])
k = stat["start_kyoku"]
print(f"[OK] 40半荘 / {k}局: 和了 {stat['hora']} (ツモ {stat['win_tsumo']} / ロン {stat['win_ron']}), 流局 {stat['ryukyoku']}, "
      f"リーチ {stat['reach']}, ポン {stat['pon']}, チー {stat['chi']}, 槓 {stat['ankan']+stat['kakan']+stat['daiminkan']}")
print(f"     平均翻 {sum(hans)/max(len(hans),1):.2f}  平均局数/半荘 {k/40:.1f}")
print("     役の出現:", dict(yaku_seen.most_common(16)))
assert stat["hora"] + stat["ryukyoku"] == k
print("[OK] 特徴量一致・点棒保存・全和了に役あり")
