"""
oracle_q_eval.py  ―  オラクル Q で「打牌の質」を測り、それが実力推定として順位より優れているかを検証する

  python3 eval/oracle_q_eval.py --data data/2024.zip --qmodel train/oracle_q/oracle_q_best.pt --ref-model train/checkpoints1_ronhead_off/model_bc_best.pt --offset 150000 --max-games 34000

  （まず --max-games 300 で動作確認してから本番を回すこと）

出すもの:
  [1] 当てはまり: 学習に使っていない半荘で R² = 1 − MSE(Q(s,a_実際), R_局) / Var(R_局)
  [2] 打牌の質: オラクル・アドバンテージ  A(s,a) = Q(s,a) − Σ_b π_ref(b|s) Q(s,b)
        人間の実際の打牌の A と、参照方策（BC）の argmax の A を、全体・平場・被リーチ・不一致局面で比べる。
        「不一致局面で人間の A > モデルの A」なら、人間の選んだ方がオラクル的に良かったということ。
  [3] ★実力推定の再現性: プレイヤーごとに A（人間の打牌）を平均し、半荘を前半／後半に分けて
        ・打牌の質の前後半相関 r_A
        ・平均順位の前後半相関 r_P（同じ人・同じ半荘で計算）
        ・前半の打牌の質で後半の平均順位をどれだけ予測できるか（vs 前半の平均順位で予測）
      を並べる。r_A が r_P（count_players では 0.116）を大きく上回れば、1年分のデータで
      打ち手の強さを選び分けられる。
  player_quality.csv に全員分を書き出す。

注意:
  ・Q は実際に選ばれた候補でしか教師を受けていない。人間がほとんど選ばない候補の Q は外挿なので、
    ベースラインは参照方策の確率で重み付けした平均にしている（確率の低い候補の寄与は小さい）。
  ・Q は「全員が鳳凰卓の人間のように打ち続けたら」という期待値。相手の手牌は見えているが山は見えていない。
"""
from __future__ import annotations
import _pathsetup  # noqa: F401
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from mahjong_model import load_policy, samples_to_batch
from mahjong_tokenizer import PADDING_ACTION_ID
from load_dataset import TenhouDatasetLoader
from aux_labels import attach_ron_labels, attach_return_labels
from oracle_q import load_oracle_q, oracle_feat


def _final_scores(events):
    scores = None
    for ev in events:
        t = ev.get("type")
        if t == "start_kyoku" and ev.get("scores"):
            scores = list(ev["scores"])
        elif t in ("hora", "ryukyoku"):
            if ev.get("scores"):
                scores = list(ev["scores"])
            elif ev.get("deltas") and scores is not None:
                scores = [s + d for s, d in zip(scores, ev["deltas"])]
        elif t == "end_game" and ev.get("scores"):
            scores = list(ev["scores"])
    return scores


def _placements(scores):
    order = sorted(range(4), key=lambda i: (-scores[i], i))
    rank = [0] * 4
    for r, seat in enumerate(order):
        rank[seat] = r + 1
    return rank


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


class Mean:
    __slots__ = ("n", "s", "s2")

    def __init__(self):
        self.n = 0; self.s = 0.0; self.s2 = 0.0

    def add(self, x):
        self.n += 1; self.s += x; self.s2 += x * x

    def mean(self):
        return self.s / self.n if self.n else float("nan")

    def se(self):
        if self.n < 2:
            return float("nan")
        m = self.mean()
        return math.sqrt(max(self.s2 / self.n - m * m, 0.0) / self.n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--qmodel", type=Path, required=True)
    ap.add_argument("--ref-model", type=Path, required=True, help="参照方策（BC）。ベースラインと比較対象")
    ap.add_argument("--offset", type=int, default=150000, help="Q の学習に使っていない範囲の先頭")
    ap.add_argument("--max-games", type=int, default=34000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--min-games-half", type=int, default=40, help="[3] で前後半それぞれに要る半荘数")
    ap.add_argument("--out", type=Path, default=Path("player_quality.csv"))
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    qm = load_oracle_q(args.qmodel, device)
    ref = load_policy(args.ref_model, device)
    loader = TenhouDatasetLoader(args.data)
    loader.tokenizer._emit_aux_fields = True
    n_games_total = min(args.max_games, max(loader.count_games() - args.offset, 0))
    half_at = n_games_total // 2

    # [1]
    r_stat = {"n": 0, "s_r": 0.0, "s_r2": 0.0, "s_e2": 0.0}
    # [2]
    adv = defaultdict(lambda: {"h": Mean(), "m": Mean()})
    # [3] player → half → [Aの和, 打牌数, 順位和, 半荘数]
    pl = defaultdict(lambda: [[0.0, 0, 0, 0], [0.0, 0, 0, 0]])

    pending = []

    def flush():
        if not pending:
            return
        smps = [x[0] for x in pending]
        b = samples_to_batch(smps, device)
        orc = torch.tensor([oracle_feat(s) for s in smps], dtype=torch.float32, device=device)
        with torch.no_grad():
            _, q = qm(b["sparse"], b["numeric"], b["progression"], b["candidates"], b["cand_feats"], orc)
            lg = ref(b["sparse"], b["numeric"], b["progression"], b["candidates"], b["cand_feats"])
        valid = b["candidates"] != PADDING_ACTION_ID
        p = F.softmax(lg.masked_fill(~valid, float("-inf")), dim=-1)
        base = (p * q).sum(dim=1)
        am = p.argmax(dim=1)
        q = q.cpu(); base = base.cpu(); am = am.cpu()
        for i, (s, name, half, rkey) in enumerate(pending):
            lab = s["action_label"]
            a_h = float(q[i, lab] - base[i])
            a_m = float(q[i, int(am[i])] - base[i])
            if s.get("aux_ret_valid"):
                R = float(s["aux_ret"]); e = float(q[i, lab]) - R
                r_stat["n"] += 1; r_stat["s_r"] += R; r_stat["s_r2"] += R * R; r_stat["s_e2"] += e * e
            groups = ["全体", rkey] + (["不一致"] if int(am[i]) != lab else [])
            if int(am[i]) != lab:
                groups.append(f"不一致・{rkey}")
            for g in groups:
                adv[g]["h"].add(a_h); adv[g]["m"].add(a_m)
            if name is not None:
                rec = pl[name][half]
                rec[0] += a_h; rec[1] += 1
        pending.clear()

    for gi, (_gid, events) in enumerate(loader.iter_games(max_games=args.max_games, offset=args.offset)):
        half = 0 if gi < half_at else 1
        sg = next((e for e in events if e.get("type") == "start_game"), None)
        names = sg.get("names") if sg else None
        sc = _final_scores(events)
        if names and sc and len(sc) == 4:
            for seat, r in enumerate(_placements(sc)):
                rec = pl[names[seat]][half]; rec[2] += r; rec[3] += 1
        samples = list(loader.tokenizer.process_events(events))
        attach_ron_labels(samples)
        attach_return_labels(events, samples)
        for s in samples:
            if s.get("event_type") != "dahai" or not s.get("aux_ron_valid"):
                continue
            rkey = "被リーチ" if any(s["numeric"][6:9]) else "平場"
            name = names[s.get("player", 0)] if names else None
            pending.append((s, name, half, rkey))
            if len(pending) >= args.batch:
                flush()
        if (gi + 1) % 1000 == 0:
            print(f"  {gi+1:,} / {n_games_total:,} 半荘")
    flush()

    W = 76
    print("=" * W)
    print("[1] 当てはまり（学習に使っていない半荘）")
    n = r_stat["n"]
    if n > 1:
        var_r = r_stat["s_r2"] / n - (r_stat["s_r"] / n) ** 2
        print(f"  R² = {1 - (r_stat['s_e2']/n)/max(var_r,1e-12):.4f}   MSE {r_stat['s_e2']/n:.5f}   "
              f"Var(R) {var_r:.5f}   打牌 {n:,}")

    print("=" * W)
    print("[2] 打牌の質（オラクル・アドバンテージ。参照方策で重み付けした平均からの差）")
    print(f"  {'':<14}{'打牌数':>10}{'人間の A':>14}{'モデルの A':>14}{'差(人−モ)':>14}")
    for g in ("全体", "平場", "被リーチ", "不一致", "不一致・平場", "不一致・被リーチ"):
        if g not in adv:
            continue
        h, m = adv[g]["h"], adv[g]["m"]
        print(f"  {g:<14}{h.n:>10,}{h.mean():>+11.5f}±{h.se():.5f}"
              f"{m.mean():>+11.5f}±{m.se():.5f}{h.mean()-m.mean():>+12.5f}")
    print("  （単位は局リターン R_局 と同じ。不一致局面で 人−モ が正なら人間の選択の方が良かった）")

    print("=" * W)
    k = args.min_games_half
    sel = [(nm, d) for nm, d in pl.items() if d[0][3] >= k and d[1][3] >= k and d[0][1] and d[1][1]]
    print(f"[3] 実力推定の再現性（前後半それぞれ {k} 半荘以上の {len(sel)} 人。前半 {half_at:,} 半荘 / 後半 "
          f"{n_games_total-half_at:,} 半荘）")
    if len(sel) >= 10:
        a1 = [d[0][0] / d[0][1] for _, d in sel]; a2 = [d[1][0] / d[1][1] for _, d in sel]
        p1 = [d[0][2] / d[0][3] for _, d in sel]; p2 = [d[1][2] / d[1][3] for _, d in sel]
        se = 1 / math.sqrt(max(len(sel) - 3, 1))
        rA, rP = _pearson(a1, a2), _pearson(p1, p2)
        print(f"  打牌の質の前後半相関   r_A = {rA:+.3f}")
        print(f"  平均順位の前後半相関   r_P = {rP:+.3f}     （r の目安の誤差 ±{se:.3f}）")
        print(f"  後半の平均順位の予測:  前半の打牌の質 → {-_pearson(a1, p2):+.3f}   "
              f"前半の平均順位 → {_pearson(p1, p2):+.3f}   （どちらも正なら当たる方向。大きいほど良い）")
        print(f"  前半の打牌の質と前半の平均順位の相関: {-_pearson(a1, p1):+.3f}")
    else:
        print("  人数が足りない。--max-games を増やすか --min-games-half を下げること")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "dec_h1", "A_h1", "games_h1", "ar_h1", "dec_h2", "A_h2", "games_h2", "ar_h2"])
        for nm, d in sorted(pl.items(), key=lambda x: -(x[1][0][3] + x[1][1][3])):
            row = [nm]
            for h in (0, 1):
                s_a, nd, s_r, ng = d[h]
                row += [nd, f"{s_a/nd:.6f}" if nd else "", ng, f"{s_r/ng:.4f}" if ng else ""]
            w.writerow(row)
    print(f"全員分を {args.out} に保存")
    print("=" * W)


if __name__ == "__main__":
    main()
