"""test_tokenizer.py ― 入力表現の回帰テスト（torch 不要）。 python3 test_tokenizer.py"""
import _pathsetup  # noqa
from mahjong_tokenizer import *

tok = MahjongTokenizer()
h0 = ["1m","1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p"]
h1 = ["1s","2s","3s","4s","5s","6s","7s","8s","9s","1z","2z","3z","4z"]
h2 = ["7z","7z","7z","7z","9m","1p","3p","5p","7p","9p","1s","3s","5s"]
h3 = ["2m","4m","6m","8m","2p","4p","6p","8p","2s","4s","6s","5z","6z"]
EV = [
 {"type":"start_game"},
 {"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"dora_marker":"3p","scores":[25000]*4,"tehais":[h0,h1,h2,h3]},
 {"type":"tsumo","actor":0,"pai":"5p"}, {"type":"dahai","actor":0,"pai":"1m","tsumogiri":False},
 {"type":"tsumo","actor":1,"pai":"5z"}, {"type":"dahai","actor":1,"pai":"4z","tsumogiri":False},
 {"type":"tsumo","actor":2,"pai":"2z"},
 {"type":"ankan","actor":2,"consumed":["7z","7z","7z","7z"]},
 {"type":"dora","dora_marker":"1s"},
 {"type":"tsumo","actor":2,"pai":"8s"}, {"type":"dahai","actor":2,"pai":"9p","tsumogiri":False},
 {"type":"end_kyoku"},
 {"type":"start_kyoku","bakaze":"S","kyoku":4,"honba":0,"kyotaku":0,"oya":3,"dora_marker":"3p","scores":[33000,17000,25000,25000],"tehais":[h0,h1,h2,h3]},
 {"type":"tsumo","actor":3,"pai":"5z"}, {"type":"dahai","actor":3,"pai":"6z","tsumogiri":False},
 {"type":"tsumo","actor":0,"pai":"4p"},
 {"type":"reach","actor":0}, {"type":"dahai","actor":0,"pai":"1m","tsumogiri":False}, {"type":"reach_accepted","actor":0},
 {"type":"tsumo","actor":1,"pai":"5z"}, {"type":"dahai","actor":1,"pai":"4z","tsumogiri":False},
 {"type":"end_kyoku"}, {"type":"end_game","scores":[32000,17000,25000,25000]},
]
S = list(tok.process_events(EV))
D = [s for s in S if s["event_type"] == "dahai"]

def check(name, cond):
    print(("[OK] " if cond else "[NG] ") + name); assert cond, name

# 1) PAD 衝突なし
check("sparse/progression に 0 が出ない", all(0 not in s["sparse"] and 0 not in s["progression"] for s in S))
check("手牌1m×2 が手牌トークンとして2個見える", D[0]["sparse"].count(SPARSE_TEHAI_OFFSET + 0) == 2)
# 2) リーク無し: P1 のサンプルに P0 のツモ(5p)が入らない。P0の1m手出しは rel=3(上家) で見える
p1 = D[1]
check("他家ツモが progression に無い", tsumo_token(TILE_STR_TO_ID["5p"]) not in p1["progression"])
check("自分のツモは入る", tsumo_token(TILE_STR_TO_ID["5z"]) in p1["progression"])
check("P0の打牌が P1 視点で上家(rel=3)", dahai_token(0, 3, False, False) in p1["progression"])
# 3) 槓で手牌が減る / 槓判断サンプル
K = [s for s in S if s["event_type"] == "kan_decision"]
check("暗槓の正例サンプル", len(K) == 1 and K[0]["possible_actions"][K[0]["action_label"]] == ACT_ANKAN_BASE + 33)
p2 = D[2]
check("暗槓後の手牌トークンは11枚", sum(1 for t in p2["sparse"] if SPARSE_TEHAI_OFFSET <= t < SPARSE_DORA_OFFSET) == 11)
check("槓した7zが打牌候補に残らない", (ACT_DAHAI_BASE + 33) not in p2["possible_actions"])
# 4) 点数: 南4局で実点数・相対順・順位特徴
s3 = D[3]   # P3(親) 17000? → scores=[33000,17000,25000,25000] の P3=25000
check("点数が相対席順 (自分,下家,対面,上家)", s3["numeric"][:4] == [1.0, 33000/25000, 17000/25000, 1.0])
check("オーラスで局進行=1.0 / 親フラグ", s3["numeric"][25] == 1.0 and s3["numeric"][24] == 1.0)
# 5) リーチは混在候補の中のラベル
r = D[4]
check("リーチ判断: 候補に DAHAI と RIICHI が混在しラベルが RIICHI+1m",
      r["riichi_label"] == 1 and r["possible_actions"][r["action_label"]] == ACT_RIICHI_BASE + 0
      and any(a < ACT_RIICHI_BASE for a in r["possible_actions"]))
check("リーチ判断時点の自分リーチフラグは 0", r["numeric"][5] == 0.0)
p1b = D[5]
check("リーチ棒が点数に反映 (P1視点の上家=P0 が 32000)", abs(p1b["numeric"][3] - 32000/25000) < 1e-9)
check("リーチ者が P1 視点で上家に立つ", p1b["numeric"][8] == 1.0 and riichi_token(3) in p1b["progression"])
check("numeric / cand_feats の次元", all(len(s["numeric"]) == NUMERIC_DIM and len(s["cand_feats"]) == MAX_ACTION_CANDIDATES
      and len(s["cand_feats"][0]) == CAND_FEAT_DIM for s in S))
# 6) 赤5絡みの鳴き
check("_can_pon([0m,5m],5m)", tok._can_pon(["0m","5m","1z"], "5m"))
check("_can_chi 赤5を含む搭子", tok._can_chi(["0m","6m","1z"], "4m", 0, 1) and tok._can_chi(["3m","4m"], "0m", 0, 1))
# 7) 食い替え
st = KyokuState(); st.tehais[1] = ["4m","7m","9p","9p"]; st.last_call[1] = ("chi", TILE_STR_TO_ID["4m"], (TILE_STR_TO_ID["5m"], TILE_STR_TO_ID["6m"]))
acts = tok.enumerate_dahai_actions(st, 1)
check("チー(4m で 56m)後は 4m と 7m が切れない", acts == [ACT_DAHAI_BASE + TILE_STR_TO_ID["9p"]])
print("\nall tokenizer tests passed")
