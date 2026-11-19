"""
diag_distribution.py  ―  実戦で放銃率が高い原因を「方策の差」と「局面分布の差」に分ける

  # 1) BC 同士の自己対戦の牌譜を作る（全員の手牌が入った mjai 形式）
  python3 rl/self_play.py --model train/checkpoints1_ronhead_off/model_bc_best.pt --games 300 --out replays/selfplay_B --seed 1

  # 2) 人間の牌譜と自己対戦の牌譜に、同じモデルで同じ集計をかける
  python3 debug/diag_distribution.py --model train/checkpoints1_ronhead_off/model_bc_best.pt --data data/2024.zip replays/selfplay_B --max-games 300 --offset 150000

列ごとに出すもの（--data の順）:
  [局単位]  リーチ率・和了率・放銃率（1人1局あたり）
  [打牌]   被リーチの打牌の割合
  [被リーチの打牌]  行動の主体（人間の牌譜なら人間、自己対戦ならモデル）と、同じ局面でのモデルの argmax について
           ・押し率: 手牌にリーチ者全員の現物があるのに、現物でない牌を切った割合
           ・即放銃率: その牌が誰かの当たり牌だった割合（aux_ron）
           ・打牌後聴牌のときの押し率／聴牌でないときの押し率

読み方:
  人間の牌譜の列で「主体（人間）」と「モデル」を比べると、同じ局面での方策の差がわかる。
  自己対戦の列では主体＝モデルなので両者はほぼ一致する（一致しなければ牌譜の再生がずれている）。
  人間の牌譜と自己対戦で、被リーチの割合や押し率の条件付きの値がどう違うかが、局面分布の差。
  例: 人間の局面ではモデルの押し率が人間と同じなのに、自己対戦では被リーチの局面自体が多い
      → 放銃率の高さは方策ではなく、卓全体のリーチの多さ（分布）から来ている。
"""
from __future__ import annotations
import _pathsetup  # noqa: F401
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from mahjong_model import load_policy, samples_to_batch
from mahjong_tokenizer import (MahjongTokenizer, PADDING_ACTION_ID, _tid_norm,
                               ACT_DAHAI_BASE, ACT_RIICHI_BASE, ACT_PON_BASE)
from load_dataset import TenhouDatasetLoader
from aux_labels import attach_ron_labels

_HONOR = {h: 27 + i for i, h in enumerate(["E", "S", "W", "N", "P", "F", "C"])}
_SUIT = {"m": 0, "p": 9, "s": 18}


def _str_to_k(s):
    if s in _HONOR:
        return _HONOR[s]
    if len(s) >= 2 and s[0].isdigit():
        n = int(s[0]) or 5
        if s[1] in _SUIT and 1 <= n <= 9:
            return _SUIT[s[1]] + n - 1
        if s[1] == "z" and 1 <= n <= 7:
            return 27 + n - 1
    return None


def _tile_of(tok):
    if ACT_DAHAI_BASE <= tok < ACT_RIICHI_BASE:
        k = _tid_norm(tok - ACT_DAHAI_BASE)
    elif ACT_RIICHI_BASE <= tok < ACT_PON_BASE:
        k = _tid_norm(tok - ACT_RIICHI_BASE)
    else:
        return None
    return k if 0 <= k < 34 else None


class Rate:
    __slots__ = ("n", "k")

    def __init__(self):
        self.n = 0; self.k = 0

    def add(self, hit):
        self.n += 1; self.k += int(bool(hit))

    def fmt(self):
        if self.n == 0:
            return "      -      "
        return f"{self.k/self.n*100:6.2f}% ({self.n:>6,})"


def analyze(src: Path, model, tok, device, max_games, offset, batch):
    loader = TenhouDatasetLoader(src)
    loader.tokenizer._emit_aux_fields = True
    off = offset if src.suffix == ".zip" else 0
    R = {k: Rate() for k in ("riichi", "agari", "houju", "under", "agree",
                             "push_a", "push_m", "ron_a", "ron_m",
                             "push_a_t", "push_a_nt", "push_m_t", "push_m_nt")}
    pending = []
    n_games = 0

    def flush():
        if not pending:
            return
        b = samples_to_batch(pending, device)
        with torch.no_grad():
            lg = model(b["sparse"], b["numeric"], b["progression"], b["candidates"], b["cand_feats"])
        am = lg.masked_fill(b["candidates"] == PADDING_ACTION_ID, float("-inf")).argmax(dim=1).cpu()
        for i, s in enumerate(pending):
            poss = s["possible_actions"]; lab = s["action_label"]; mi = int(am[i])
            R["agree"].add(mi == lab)
            player = s.get("player", 0)
            rq = [q for q in range(4) if q != player and s["_riichi_flags"][q]]
            if not rq:
                continue
            safe = set(s["_discards34"][rq[0]])
            for q in rq[1:]:
                safe &= set(s["_discards34"][q])
            hs, nm = s.get("_hand_strs") or [], s.get("_num_melds", 0)
            kinds = [_str_to_k(t) for t in hs]
            has_safe = any(k in safe for k in kinds if k is not None)
            for who, idx in (("a", lab), ("m", mi)):
                k = _tile_of(poss[idx])
                if k is None:
                    continue
                R[f"ron_{who}"].add(any(s["aux_ron"][r][k] for r in range(3)))
                if not has_safe:
                    continue
                push = k not in safe
                R[f"push_{who}"].add(push)
                tile_s = next((t for t in hs if _str_to_k(t) == k), None)
                if tile_s is None:
                    continue
                rest = list(hs); rest.remove(tile_s)
                try:
                    tenpai = tok._shanten_of(rest, nm) == 0
                except Exception:
                    continue
                R[f"push_{who}_{'t' if tenpai else 'nt'}"].add(push)
        pending.clear()

    for _gid, events in loader.iter_games(max_games=max_games, offset=off):
        n_games += 1
        # 局単位: 1人1局あたりのリーチ・和了・放銃
        n_k = sum(1 for e in events if e.get("type") == "start_kyoku")
        reach = sum(1 for e in events if e.get("type") == "reach")
        hora = [e for e in events if e.get("type") == "hora"]
        R["riichi"].n += n_k * 4; R["riichi"].k += reach
        R["agari"].n += n_k * 4; R["agari"].k += len(hora)
        R["houju"].n += n_k * 4
        R["houju"].k += sum(1 for e in hora if e.get("target") is not None and e.get("target") != e.get("actor"))
        samples = list(loader.tokenizer.process_events(events))
        attach_ron_labels(samples)
        for s in samples:
            if s.get("event_type") != "dahai" or not s.get("aux_ron_valid"):
                continue
            R["under"].add(any(s["_riichi_flags"][q] for q in range(4) if q != s.get("player", 0)))
            pending.append(s)
            if len(pending) >= batch:
                flush()
    flush()
    return R, n_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="zip（人間の牌譜）や replays/ ディレクトリ")
    ap.add_argument("--max-games", type=int, default=300)
    ap.add_argument("--offset", type=int, default=150000, help="zip にだけ効く")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy(args.model, device)
    tok = MahjongTokenizer()

    cols = []
    for src in args.data:
        R, ng = analyze(src, model, tok, device, args.max_games, args.offset, args.batch)
        cols.append((src.name, R, ng))
        print(f"  {src}: {ng} 半荘 集計済み")

    rows = [("[局] リーチ率（1人1局）", "riichi"), ("[局] 和了率", "agari"), ("[局] 放銃率", "houju"),
            ("[打牌] 被リーチの割合", "under"), ("[打牌] 主体とモデルの一致率", "agree"),
            ("[被リーチ] 押し率・主体", "push_a"), ("[被リーチ] 押し率・モデル", "push_m"),
            ("  打牌後聴牌・主体", "push_a_t"), ("  打牌後聴牌・モデル", "push_m_t"),
            ("  聴牌でない・主体", "push_a_nt"), ("  聴牌でない・モデル", "push_m_nt"),
            ("[被リーチ] 即放銃率・主体", "ron_a"), ("[被リーチ] 即放銃率・モデル", "ron_m")]
    W = 30 + 22 * len(cols)
    print("=" * W)
    print(f"  {'':<28}" + "".join(f"{c[0][:20]:>22}" for c in cols))
    for label, key in rows:
        print(f"  {label:<28}" + "".join(f"{c[1][key].fmt():>22}" for c in cols))
    print("=" * W)
    print("  押し率 = 手牌にリーチ者全員の現物があるのに、現物でない牌を切った割合（括弧内は母数）")
    print("  主体 = 人間の牌譜なら人間、自己対戦ならその席のモデル")


if __name__ == "__main__":
    main()
