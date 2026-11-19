"""
count_players.py  ―  牌譜からプレイヤーごとの半荘数と成績を集計し、「運を差し引いた強さ」を推定する

  python3 debug/count_players.py --data data/2024.zip --inspect
  python3 debug/count_players.py --data data/2024.zip --workers 32

出すもの:
  [A] 安定段位 × 半荘数のグリッド（点推定 / −1SE の保守版）。セル = 人数 / 席数
  [B] 前半・後半の再現性（split-half）
      日付の前半（1〜6月）と後半（7〜12月）で同じ人の平均順位がどれだけ相関するか。
      これが「観測された差のうち実力の割合」を直接測る。
  [C] 経験ベイズ縮小
      平均順位の真の分散 τ² をモーメント法で推定し、各人を全体平均 2.5 に向けて縮める。
      縮小後の平均順位で選んだときの人数 / 席数のグリッドと上位一覧。
  全員分を player_stats.csv に保存（縮小後の平均順位つき）。

安定段位は鳳凰卓東南の式 (6*n1 + 3*n2)/n4 − 2（卓全体の平均は厳密に 7.0）。SE ≈ 22.4/√n。
平均順位の1半荘あたり分散は 1.25（順位が一様のとき）なので SE = √(1.25/n)。
目安: 天鳳位全員の合計 平均順位 2.424（安定段位 8.72）、太くないお 2.413（8.88）。
順位は最終持ち点の降順、同点は起家に近い席が上。
"""
from __future__ import annotations
import _pathsetup  # noqa: F401
import argparse
import csv
import json
import math
import re
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from load_dataset import TenhouDatasetLoader

_DATE_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})")
VAR1 = 1.25   # 平均順位の1半荘あたり分散（順位一様のとき）


def _new_stat():
    # n1..n4, 最初の日付, 最後の日付, 前半の半荘数, 前半の順位和, 後半の半荘数, 後半の順位和
    return [0, 0, 0, 0, "", "", 0, 0, 0, 0]


def _final_scores(events):
    """最終持ち点。end_game.scores → 最後の hora/ryukyoku の scores → start_kyoku.scores + deltas の順で取る。"""
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


def _worker(args):
    path, names = args
    loader = TenhouDatasetLoader(path)
    stats = defaultdict(_new_stat)
    n_ok = n_noname = n_noscore = 0
    with zipfile.ZipFile(path) as zf:
        for name in names:
            events = loader._read_mjson(zf.read(name))
            sg = next((e for e in events if e.get("type") == "start_game"), None)
            pnames = sg.get("names") if sg else None
            if not pnames or len(pnames) != 4:
                n_noname += 1
                continue
            sc = _final_scores(events)
            if not sc or len(sc) != 4:
                n_noscore += 1
                continue
            m = _DATE_RE.search(name)
            date = "".join(m.groups()) if m else ""
            for seat, r in enumerate(_placements(sc)):
                st = stats[pnames[seat]]
                st[r - 1] += 1
                if date:
                    if not st[4] or date < st[4]:
                        st[4] = date
                    if date > st[5]:
                        st[5] = date
                    if date[4:6] <= "06":
                        st[6] += 1; st[7] += r
                    else:
                        st[8] += 1; st[9] += r
            n_ok += 1
    return dict(stats), n_ok, n_noname, n_noscore


def _stable_dan(n1, n2, n4):
    return (6 * n1 + 3 * n2) / n4 - 2 if n4 else float("inf")


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--inspect", action="store_true", help="最初の1半荘の start_game / end_game を表示して終わる")
    ap.add_argument("--out", type=Path, default=Path("player_stats.csv"))
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--eb-min-games", type=int, default=100, help="τ² の推定に使う最小半荘数")
    ap.add_argument("--split-min-games", type=int, default=150, help="split-half で前後半それぞれに要る半荘数")
    args = ap.parse_args()

    with zipfile.ZipFile(args.data) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".mjson"))
    if args.max_games:
        names = names[:args.max_games]

    if args.inspect:
        loader = TenhouDatasetLoader(args.data)
        with zipfile.ZipFile(args.data) as zf:
            events = loader._read_mjson(zf.read(names[0]))
        print(f"ファイル名: {names[0]}")
        for t in ("start_game", "end_game"):
            ev = next((e for e in events if e.get("type") == t), None)
            print(f"\n[{t}]")
            print(json.dumps(ev, ensure_ascii=False)[:1500] if ev else "  （無し）")
        last = [e for e in events if e.get("type") in ("hora", "ryukyoku")][-1:]
        print("\n[最後の hora/ryukyoku のキー]", list(last[0].keys()) if last else "（無し）")
        print("\n最終持ち点:", _final_scores(events))
        return

    chunks = [names[i::args.workers] for i in range(args.workers)]
    total = defaultdict(_new_stat)
    n_ok = n_noname = n_noscore = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for stats, ok, nn, ns in ex.map(_worker, [(str(args.data), c) for c in chunks]):
            n_ok += ok; n_noname += nn; n_noscore += ns
            for p, s in stats.items():
                t = total[p]
                for i in (0, 1, 2, 3, 6, 7, 8, 9):
                    t[i] += s[i]
                if s[4] and (not t[4] or s[4] < t[4]):
                    t[4] = s[4]
                if s[5] > t[5]:
                    t[5] = s[5]

    print(f"集計した半荘 {n_ok:,}   名前なし {n_noname:,}   持ち点不明 {n_noscore:,}")
    if n_ok == 0:
        print("名前か持ち点が取れていない。--inspect で形式を確認すること")
        return

    # 行: name, n, n1..n4, 平均順位, 安定段位, 安定段位SE, first, last, h1n, h1ar, h2n, h2ar
    rows = []
    for p, s in total.items():
        n1, n2, n3, n4 = s[0], s[1], s[2], s[3]
        n = n1 + n2 + n3 + n4
        ar = (n1 + 2 * n2 + 3 * n3 + 4 * n4) / n
        h1ar = s[7] / s[6] if s[6] else float("nan")
        h2ar = s[9] / s[8] if s[8] else float("nan")
        rows.append({"name": p, "n": n, "n1": n1, "n2": n2, "n3": n3, "n4": n4, "ar": ar,
                     "sd": _stable_dan(n1, n2, n4), "sd_se": 22.4 / math.sqrt(n),
                     "first": s[4], "last": s[5], "h1n": s[6], "h1ar": h1ar, "h2n": s[8], "h2ar": h2ar})
    rows.sort(key=lambda r: -r["n"])
    print(f"プレイヤー数 {len(rows):,}   席の総数 {sum(r['n'] for r in rows):,}（= 半荘数×4 になるはず）")

    W = 76
    min_games = [100, 300, 500, 1000, 2000]

    # ---------------------------------------------------------------- [A]
    dans = [7.5, 8.0, 8.5, 9.0]
    for label, key in (("安定段位（点推定）", lambda r: r["sd"]),
                       ("安定段位 − 1SE（保守的）", lambda r: r["sd"] - r["sd_se"])):
        print("\n" + "=" * W)
        print(f"[A] {label} ≥ 閾値 かつ 半荘数 ≥ 下限   セル = 人数 / 席数")
        print(f"  {'半荘数≥':<8}" + "".join(f"{'≥' + str(d):>16}" for d in dans))
        for mg in min_games:
            cells = []
            for d in dans:
                sel = [r for r in rows if r["n"] >= mg and key(r) >= d]
                cells.append(f"{len(sel)}人/{sum(r['n'] for r in sel):,}")
            print(f"  {mg:<8}" + "".join(f"{c:>16}" for c in cells))

    # ---------------------------------------------------------------- [C] τ² の推定（[B] でも使う）
    eb = [r for r in rows if r["n"] >= args.eb_min_games]
    mu = 2.5
    obs_var = sum((r["ar"] - mu) ** 2 for r in eb) / len(eb)
    noise_var = sum(VAR1 / r["n"] for r in eb) / len(eb)
    tau2 = max(obs_var - noise_var, 1e-8)
    for r in rows:
        v = VAR1 / r["n"]
        r["shrink"] = tau2 / (tau2 + v)
        r["ar_eb"] = mu + r["shrink"] * (r["ar"] - mu)
        r["ar_eb_se"] = math.sqrt(1.0 / (1.0 / tau2 + 1.0 / v))   # 事後SD

    # ---------------------------------------------------------------- [B]
    print("\n" + "=" * W)
    print("[B] 前半（1〜6月）と後半（7〜12月）の再現性")
    k = args.split_min_games
    sp = [r for r in rows if r["h1n"] >= k and r["h2n"] >= k]
    if len(sp) >= 10:
        xs = [r["h1ar"] for r in sp]; ys = [r["h2ar"] for r in sp]
        rho = _pearson(xs, ys)
        # τ² が正しければ期待される相関
        exp_rho = sum(tau2 / math.sqrt((tau2 + VAR1 / r["h1n"]) * (tau2 + VAR1 / r["h2n"])) for r in sp) / len(sp)
        print(f"  前後半それぞれ {k} 戦以上: {len(sp)} 人")
        print(f"  平均順位の前後半相関 r = {rho:.3f}   （τ² から期待される値 {exp_rho:.3f}）")
        sp_sorted = sorted(sp, key=lambda r: r["h1ar"])
        for frac in (0.05, 0.10, 0.25):
            m = max(1, int(len(sp_sorted) * frac))
            top = sp_sorted[:m]
            a1 = sum(r["h1ar"] for r in top) / m
            a2 = sum(r["h2ar"] for r in top) / m
            print(f"  前半の上位 {int(frac*100):>2}%（{m}人）: 前半 {a1:.3f} → 後半 {a2:.3f}   "
                  f"（2.5 からの差の {((2.5-a2)/(2.5-a1))*100 if a1 < 2.5 else float('nan'):.0f}% が残る）")
    else:
        print(f"  前後半それぞれ {k} 戦以上の人が {len(sp)} 人しかいない（日付がファイル名に無い可能性）")

    print("\n" + "=" * W)
    print("[C] 経験ベイズ縮小（平均順位）")
    print(f"  τ（真の実力の標準偏差、平均順位単位）= {math.sqrt(tau2):.4f}   "
          f"観測の分散 {obs_var:.5f} = 真の分散 {tau2:.5f} + ノイズ {noise_var:.5f}   "
          f"（{args.eb_min_games} 戦以上 {len(eb)} 人）")
    print(f"  観測された差のうち実力の割合（平均）: {tau2/obs_var*100:.0f}%")
    for n in (300, 1000, 3000):
        print(f"    {n:>5} 戦の人の縮小率 {tau2/(tau2+VAR1/n):.2f}")
    print("  目安: 天鳳位合計 2.424 / 太くないお 2.413")
    ths = [2.46, 2.44, 2.42, 2.40]
    print(f"\n  縮小後の平均順位 ≤ 閾値 かつ 半荘数 ≥ 下限   セル = 人数 / 席数")
    print(f"  {'半荘数≥':<8}" + "".join(f"{'≤' + str(t):>16}" for t in ths))
    for mg in min_games:
        cells = []
        for t in ths:
            sel = [r for r in rows if r["n"] >= mg and r["ar_eb"] <= t]
            cells.append(f"{len(sel)}人/{sum(r['n'] for r in sel):,}")
        print(f"  {mg:<8}" + "".join(f"{c:>16}" for c in cells))

    print(f"\n  縮小後の平均順位の上位 {args.top} 人")
    print(f"  {'名前':<20}{'半荘':>6}{'平均順位':>9}{'縮小後':>8}{'±':>6}{'安定段位':>9}  期間")
    for r in sorted(rows, key=lambda r: r["ar_eb"])[:args.top]:
        print(f"  {r['name'][:20]:<20}{r['n']:>6}{r['ar']:>9.3f}{r['ar_eb']:>8.3f}{r['ar_eb_se']:>6.3f}"
              f"{r['sd']:>9.2f}  {r['first']}-{r['last']}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = ["name", "n", "n1", "n2", "n3", "n4", "ar", "ar_eb", "ar_eb_se", "sd", "sd_se",
                "first", "last", "h1n", "h1ar", "h2n", "h2ar"]
        w.writerow(cols)
        for r in rows:
            w.writerow([f"{r[c]:.4f}" if isinstance(r[c], float) else r[c] for c in cols])
    print(f"\n全員分を {args.out} に保存")


if __name__ == "__main__":
    main()
