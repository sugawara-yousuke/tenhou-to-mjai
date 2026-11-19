"""heuristic_sim.py ― 向聴最小で打つ簡易方策（テスト・動作確認用。torch モデル不要）"""
import _pathsetup  # noqa
import random
from mahjong_tokenizer import (shanten_number, TILE_STR_TO_ID, _tid_norm,
                               ACT_RIICHI_BASE, ACT_PASS_BASE, PADDING_ACTION_ID)
from self_play import MahjongSimulator


class HeuristicSim(MahjongSimulator):
    def _decide_dahai(self, K, a, sample):
        st = K.state
        if self._sample_log is not None: self._sample_log.append(sample)
        ids = [TILE_STR_TO_ID[t] for t in st.tehais[a]]
        best = []
        for i, tok in enumerate(sample["possible_actions"]):
            if tok == PADDING_ACTION_ID: continue
            k = tok % 37
            j = next(x for x, t in enumerate(ids) if _tid_norm(t) == k)
            sh = shanten_number(ids[:j] + ids[j+1:], st.meld_counts[a])
            best.append((sh - (0.5 if tok >= ACT_RIICHI_BASE else 0), random.random(), tok, i))
        _, _, tok, i = min(best)
        return tok, i, {"pred_idx": i, "pred_token": tok, "probs": None, "poss": sample["possible_actions"]}
    def _choose(self, player, sample, kind, context=None):
        if self._sample_log is not None: self._sample_log.append(sample)
        valid = [t for t in sample["possible_actions"] if t != PADDING_ACTION_ID]
        if kind == "kan": return valid[-1]
        return random.choice(valid[1:]) if random.random() < 0.25 else ACT_PASS_BASE


if __name__ == "__main__":
    import argparse, json, zipfile
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--out", default="heuristic_replays.zip")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    sim = HeuristicSim(None, "cpu")
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for g in range(a.games):
            ev = sim.simulate_game()
            for e in ev:
                e.pop("_infer", None)
            zf.writestr(f"game_{g:05d}.mjson", "\n".join(json.dumps(e, ensure_ascii=False) for e in ev))
    print(f"{a.out}: {a.games} games")
