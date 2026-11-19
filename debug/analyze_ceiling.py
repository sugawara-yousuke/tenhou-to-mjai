"""
analyze_ceiling.py  ―  打牌一致率 77% の中身を分解する

  python3 debug/analyze_ceiling.py --data data/2024.zip \
      --model train/checkpoints_150k/model_bc_best.pt --max-games 200 --offset 150000

何を出すか:

 [1] 校正（calibration）
     予測確率で分類して、実際の的中率と比べる。
     注意: 校正が合っていれば top1 ≈ E[max π] は恒等的に成り立つ。これは
     「モデル自身の確信度の範囲で取りこぼしがない」ことしか意味せず、模倣の天井
     （ベイズ最適な一致率）ではない。より強いモデルなら π が尖って両方とも上がりうる。

 [2] 局面の種類ごとの一致率
     候補数・巡目・被リーチ・打牌後の向聴で分けて、どこで外しているかを見る。

 [3] 不一致時の即放銃（オラクル判定）
     aux_ron（全員の手牌が既知なので確定値）で、人間の牌／モデルの牌が当たり牌だったかを比べる。
     排他的なペア（片方だけ当たり）で McNemar 検定をかける。

 [4] ラベルの整合性チェック（aux_ron vs aux_houjuu）

 [5] 不一致時の牌効率比較 ★
     不一致局面で、人間の牌とモデルの牌それぞれについて打牌後の向聴数と受け入れ枚数を計算し
       モデル劣（向聴で負け／同向聴で受け入れ負け）・同等・人間劣
     に分類する。「同等かつ両方とも当たり牌でない」は牌効率と即放銃の観点では区別できない選択で、
     本質的に決まらない部分の推定値になる。
     注意: 受け入れは自分の手牌だけを引いた理論枚数（河・ドラ表示・副露は引かない）。
           打点（役・ドラ・赤）と押し引きは見ていないので、被リーチ時の「モデル劣」は
           守備的な正着の可能性がある。

 [6] 被リーチ時の押し引き（オラクル判定）★
     被リーチの不一致局面で、人間の選択／モデルの選択それぞれの打牌後の手牌に
     「リーチ者の当たり牌」が何枚残るかを数える。リーチ者の待ちは宣言後に固定されるので、
     aux_ron[リーチ者] がそのまま待ちの集合になる（1手先ではなく、この先降り切れるかの指標）。
     [5] の区分（モデルだけ降り／人間だけ降り／向聴同じ）と、人間の打牌後向聴で分けて出す。
     参考として、リーチしていない聴牌者も含めた枚数（全テンパイ者）も出す。
     前提: numeric[6+i] が相対位置 i+1 のリーチ、aux_ron[i] が同じ相手の当たり牌。
           対応が正しいかを「リーチ者の行が空でない割合」で自己検査する。

--offset は学習に使っていない範囲を指すこと（15万半荘で学習したなら 150000 以上）。
"""
from __future__ import annotations
import _pathsetup  # noqa: F401
import argparse
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from mahjong_model import load_policy, samples_to_batch
from mahjong_tokenizer import (MahjongTokenizer, PADDING_ACTION_ID, _tid_norm,
                               ACT_DAHAI_BASE, ACT_RIICHI_BASE, ACT_PON_BASE)
from load_dataset import TenhouDatasetLoader
from aux_labels import attach_aux_labels, attach_ron_labels


def _tile_of(tok: int):
    """打牌 / リーチ宣言のトークン → 牌種34。それ以外は None。"""
    if ACT_DAHAI_BASE <= tok < ACT_RIICHI_BASE:
        k = _tid_norm(tok - ACT_DAHAI_BASE)
    elif ACT_RIICHI_BASE <= tok < ACT_PON_BASE:
        k = _tid_norm(tok - ACT_RIICHI_BASE)
    else:
        return None
    return k if 0 <= k < 34 else None


def _is_ron_tile(smp: dict, k: int) -> bool:
    """その牌を今切ったら誰かに当たるか（全員の手牌が既知なので確定値）。"""
    if k is None or not smp.get("aux_ron_valid"):
        return False
    return any(smp["aux_ron"][rel][k] for rel in range(3))


# ---------------------------------------------------------------- [5] 牌効率
_HONOR_MJAI = ["E", "S", "W", "N", "P", "F", "C"]
_HONOR_IDX = {h: 27 + i for i, h in enumerate(_HONOR_MJAI)}
_SUIT_BASE = {"m": 0, "p": 9, "s": 18}


def _str_to_k(s: str):
    """牌文字列 → 牌種34。mjai形式（5m, 5mr, E..C）と 1z..7z 形式の両方を受ける。"""
    if s in _HONOR_IDX:
        return _HONOR_IDX[s]
    if len(s) >= 2 and s[0].isdigit():
        n = int(s[0])
        if n == 0:          # 0m = 赤5 の流儀
            n = 5
        suit = s[1]
        if suit in _SUIT_BASE and 1 <= n <= 9:
            return _SUIT_BASE[suit] + n - 1
        if suit == "z" and 1 <= n <= 7:
            return 27 + n - 1
    return None


def _k_to_str(k: int, zstyle: bool) -> str:
    if k >= 27:
        return f"{k - 26}z" if zstyle else _HONOR_MJAI[k - 27]
    return f"{k % 9 + 1}{'mps'[k // 9]}"


def _shanten_ukeire(tok, rest, nm, zstyle):
    """打牌後の手 rest について (向聴, 受け入れ枚数)。受け入れは自分の手牌だけ引いた理論枚数。"""
    base = tok._shanten_of(rest, nm)
    cnt = [0] * 34
    for s in rest:
        k = _str_to_k(s)
        if k is not None:
            cnt[k] += 1
    uk = 0
    for t in range(34):
        if cnt[t] >= 4:
            continue
        if tok._shanten_of(rest + [_k_to_str(t, zstyle)], nm) < base:
            uk += 4 - cnt[t]
    return base, uk


def _danger(rest, rows):
    """手牌 rest のうち、rows（相手ごとの当たり牌34次元）のどれかに当たる牌の枚数。"""
    if not rows:
        return 0
    n = 0
    for s in rest:
        k = _str_to_k(s)
        if k is not None and any(r[k] for r in rows):
            n += 1
    return n


class Acc:
    __slots__ = ("n", "hit")

    def __init__(self):
        self.n = 0; self.hit = 0

    def add(self, hit):
        self.n += 1; self.hit += int(hit)

    def __str__(self):
        return f"{self.hit/max(self.n,1):.4f} (n={self.n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--max-games", type=int, default=200)
    ap.add_argument("--offset", type=int, default=150000,
                    help="学習に使っていない範囲の先頭。必ず学習に使った半荘数以上にすること")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy(args.model, device); model.eval()
    tok = MahjongTokenizer()

    loader = TenhouDatasetLoader(args.data)
    loader.tokenizer._emit_aux_fields = True

    # [1] 校正
    bins = [Acc() for _ in range(10)]
    bin_p = [0.0] * 10
    sum_maxp, sum_ent, n_all, n_amb = 0.0, 0.0, 0, 0
    top1 = top2 = top3 = 0
    # [2] 分解
    by_ncand, by_turn, by_riichi, by_shanten = (defaultdict(Acc) for _ in range(4))
    # [3] 不一致時のオラクル判定
    dis = {"n": 0, "human_ron": 0, "model_ron": 0, "both": 0, "neither": 0,
           "model_safer": 0, "human_safer": 0}
    agree_ron = {"n": 0, "ron": 0}
    # [4] 整合性
    cons = {"n": 0, "match": 0, "ron_but_no_houjuu": 0, "houjuu_but_no_ron": 0}
    # [5] 牌効率。グループ → 区分 → 件数／確率の和
    eff = defaultdict(lambda: defaultdict(float))
    eff_chk = {"dahai_parse_ok": 0, "dahai_parse_ng": 0, "same_kind": 0,
               "model_not_in_hand": 0, "err": 0, "skip": 0}
    # [6] 被リーチ時の押し引き。グループ → 集計値
    dang = defaultdict(lambda: defaultdict(float))
    dchk = {"flag_n": 0, "flag_nonempty": 0, "noflag_n": 0, "noflag_nonempty": 0}

    pending = []

    def eff_add(groups, key, p_top, p_hum):
        for g in groups:
            e = eff[g]
            e["n"] += 1
            e[key] += 1
            e[key + "_ptop"] += p_top
            e[key + "_phum"] += p_hum

    def flush():
        if not pending:
            return
        nonlocal sum_maxp, sum_ent, n_all, n_amb, top1, top2, top3
        b = samples_to_batch(pending, device)
        with torch.no_grad():
            logits = model(b["sparse"], b["numeric"], b["progression"],
                           b["candidates"], b["cand_feats"])
        cand = b["candidates"]
        logits = logits.masked_fill(cand == PADDING_ACTION_ID, float("-inf"))
        probs = F.softmax(logits, dim=-1).cpu()
        for i, smp in enumerate(pending):
            poss = smp["possible_actions"]
            lab = smp["action_label"]
            p = probs[i]
            valid = [j for j, t in enumerate(poss) if t != PADDING_ACTION_ID]
            order = sorted(valid, key=lambda j: -float(p[j]))
            hit = (order[0] == lab)
            n_all += 1
            top1 += int(hit)
            top2 += int(lab in order[:2])
            top3 += int(lab in order[:3])

            mx = float(p[order[0]])
            sum_maxp += mx
            if mx < 0.5:
                n_amb += 1
            pv = p[valid].clamp(min=1e-12)
            sum_ent += float(-(pv * pv.log()).sum())

            k = min(int(mx * 10), 9)
            bins[k].add(hit); bin_p[k] += mx

            nc = len(valid)
            by_ncand[min(nc, 14)].add(hit)
            turn = int(round(smp["numeric"][15] * 30))
            by_turn[min(turn // 3 * 3, 15)].add(hit)
            riichi_key = "被リーチ" if any(smp["numeric"][6:9]) else "平場"
            by_riichi[riichi_key].add(hit)
            hs, nm, ds = smp.get("_hand_strs"), smp.get("_num_melds", 0), smp.get("_dahai_str")
            if hs and ds and ds in hs:
                rest = list(hs); rest.remove(ds)
                try:
                    by_shanten[min(tok._shanten_of(rest, nm), 5)].add(hit)
                except Exception:
                    pass

            # [3][4] オラクル判定
            k_h = _tile_of(poss[lab])
            k_m = _tile_of(poss[order[0]])
            r_h = r_m = False
            if smp.get("aux_ron_valid") and k_h is not None:
                r_h = _is_ron_tile(smp, k_h)
                cons["n"] += 1
                hj = bool(smp.get("aux_houjuu", 0))
                cons["match"] += int(r_h == hj)
                cons["ron_but_no_houjuu"] += int(r_h and not hj)
                cons["houjuu_but_no_ron"] += int(hj and not r_h)
                if k_m is not None:
                    r_m = _is_ron_tile(smp, k_m)
                    if hit:
                        agree_ron["n"] += 1; agree_ron["ron"] += int(r_h)
                    else:
                        dis["n"] += 1
                        dis["human_ron"] += int(r_h); dis["model_ron"] += int(r_m)
                        dis["both"] += int(r_h and r_m)
                        dis["neither"] += int(not r_h and not r_m)
                        dis["model_safer"] += int(r_h and not r_m)
                        dis["human_safer"] += int(r_m and not r_h)

            # [5] 不一致時の牌効率
            if hit:
                continue
            if k_h is None or k_m is None or not hs or not ds or ds not in hs:
                eff_chk["skip"] += 1
                continue
            if _str_to_k(ds) == k_h:
                eff_chk["dahai_parse_ok"] += 1
            else:
                eff_chk["dahai_parse_ng"] += 1
                continue
            if k_h == k_m:
                # 同じ牌種（赤／黒、リーチ宣言の有無など）の違い
                eff_chk["same_kind"] += 1
                continue
            ms = next((s for s in hs if _str_to_k(s) == k_m), None)
            if ms is None:
                eff_chk["model_not_in_hand"] += 1
                continue
            zstyle = any(s.endswith("z") for s in hs)
            rest_h = list(hs); rest_h.remove(ds)
            rest_m = list(hs); rest_m.remove(ms)
            try:
                sh, uh = _shanten_ukeire(tok, rest_h, nm, zstyle)
                sm, um = _shanten_ukeire(tok, rest_m, nm, zstyle)
            except Exception:
                eff_chk["err"] += 1
                continue
            if sm > sh:
                key = "m_shan"
            elif sm < sh:
                key = "h_shan"
            elif um < uh:
                key = "m_uk"
            elif um > uh:
                key = "h_uk"
            else:
                key = "eq_safe" if not (r_h or r_m) else "eq_ron"
            if key in ("m_uk", "h_uk") and abs(um - uh) <= 2:
                key += "_near"
            groups = ["全体", riichi_key, f"向聴{min(sh, 3)}"]
            eff_add(groups, key, float(p[order[0]]), float(p[lab]))

            # [6] 被リーチ時、打牌後の手牌に残る当たり牌の枚数
            if riichi_key != "被リーチ" or not smp.get("aux_ron_valid"):
                continue
            ron = smp["aux_ron"]
            flags = [bool(x) for x in smp["numeric"][6:9]]
            for rel in range(3):
                nonempty = any(ron[rel])
                if flags[rel]:
                    dchk["flag_n"] += 1; dchk["flag_nonempty"] += int(nonempty)
                else:
                    dchk["noflag_n"] += 1; dchk["noflag_nonempty"] += int(nonempty)
            rr = [ron[rel] for rel in range(3) if flags[rel]]
            ra = [ron[rel] for rel in range(3)]
            dh_r, dm_r = _danger(rest_h, rr), _danger(rest_m, rr)
            dh_a, dm_a = _danger(rest_h, ra), _danger(rest_m, ra)
            pf = {"m_shan": "モデルだけ降り", "h_shan": "人間だけ降り"}.get(key, "向聴同じ")
            for g in ("全体", pf, f"向聴{min(sh, 3)}"):
                e = dang[g]
                e["n"] += 1
                e["h_r"] += dh_r; e["m_r"] += dm_r; e["h_a"] += dh_a; e["m_a"] += dm_a
                e["m_more"] += int(dm_r > dh_r); e["h_more"] += int(dh_r > dm_r)
                e["eq"] += int(dm_r == dh_r)
                e["h_zero"] += int(dh_r == 0); e["m_zero"] += int(dm_r == 0)
                e["h_ron_now"] += int(r_h); e["m_ron_now"] += int(r_m)
        pending.clear()

    for _gid, events in loader.iter_games(max_games=args.max_games, offset=args.offset):
        samples = list(loader.tokenizer.process_events(events))
        attach_aux_labels(loader.tokenizer, events, samples)
        attach_ron_labels(samples)
        for s in samples:
            if s.get("event_type") != "dahai":
                continue
            pending.append(s)
            if len(pending) >= args.batch:
                flush()
    flush()

    W = 72
    t1 = top1 / max(n_all, 1)
    print("=" * W)
    print(f"打牌サンプル {n_all:,}   top1 {t1:.4f}  "
          f"top2 {top2/max(n_all,1):.4f}  top3 {top3/max(n_all,1):.4f}")
    print("=" * W)
    print("[1] 校正")
    print(f"  {'予測確率':<12}{'件数':>9}{'平均予測':>10}{'実際の的中':>12}{'差':>9}")
    for i, a in enumerate(bins):
        if a.n == 0:
            continue
        mp = bin_p[i] / a.n; ac = a.hit / a.n
        print(f"  {i/10:.1f}-{(i+1)/10:.1f}{'':<4}{a.n:>9,}{mp:>10.3f}{ac:>12.3f}{ac-mp:>+9.3f}")
    ceil = sum_maxp / max(n_all, 1)
    print(f"\n  E[max π] = {ceil:.4f}   実際の top1 = {t1:.4f}   差 {ceil-t1:+.4f}")
    print("  ※ 校正が合っていれば両者が一致するのは恒等的。模倣の天井ではない")
    print(f"  平均エントロピー {sum_ent/max(n_all,1):.3f} nats   "
          f"max π < 0.5 の局面 {n_amb/max(n_all,1)*100:.1f}%")

    print("=" * W)
    print("[2] 局面の種類ごとの一致率")
    print("  候補数:   " + "  ".join(f"{k}={v.hit/max(v.n,1):.3f}" for k, v in sorted(by_ncand.items()) if v.n >= 50))
    print("  巡目:     " + "  ".join(f"{k}-{k+2}={v.hit/max(v.n,1):.3f}" for k, v in sorted(by_turn.items()) if v.n >= 50))
    print("  場況:     " + "  ".join(f"{k}={v}" for k, v in by_riichi.items()))
    print("  打牌後向聴: " + "  ".join(f"{k}={v.hit/max(v.n,1):.3f}" for k, v in sorted(by_shanten.items()) if v.n >= 50))

    print("=" * W)
    print("[3] 不一致時の即放銃（オラクル判定）")
    d = dis["n"]
    if d:
        print(f"  不一致した局面: {d:,}")
        print(f"    人間の牌が当たり牌:       {dis['human_ron']:>6} ({dis['human_ron']/d*100:.2f}%)")
        print(f"    モデルの牌が当たり牌:     {dis['model_ron']:>6} ({dis['model_ron']/d*100:.2f}%)")
        print(f"    モデルだけ安全だった:     {dis['model_safer']:>6}")
        print(f"    人間だけ安全だった:       {dis['human_safer']:>6}")
        a, c = dis["model_safer"], dis["human_safer"]
        z = (a - c) / math.sqrt(a + c) if a + c else 0.0
        print(f"\n  McNemar z = {z:+.2f}  （正ならモデルのほうが安全）")
        if abs(z) < 1.96:
            print("     → 有意差なし。即放銃の観点ではモデルと人間は互角")
        elif z > 0:
            print("     → モデルのほうが有意に安全")
        else:
            print("     → 人間のほうが有意に安全")
        print(f"  即放銃が絡む不一致は全打牌の {(a+c)/max(n_all,1)*100:.2f}%")
    if agree_ron["n"]:
        print(f"  一致した局面での放銃率: {agree_ron['ron']/agree_ron['n']*100:.2f}% "
              f"(n={agree_ron['n']:,})")

    print("=" * W)
    print("[4] ラベル整合性（aux_ron vs 実際の放銃）")
    if cons["n"]:
        print(f"  照合 {cons['n']:,} 件   一致 {cons['match']/cons['n']*100:.2f}%")
        print(f"    当たり牌判定だが放銃していない: {cons['ron_but_no_houjuu']:,} "
              f"（見逃し・頭ハネ・流局などで起こりうる）")
        print(f"    放銃したが当たり牌判定でない:   {cons['houjuu_but_no_ron']:,} "
              f"← これが多いならラベルにバグ")

    print("=" * W)
    print("[5] 不一致時の牌効率（打牌後の向聴・受け入れ）★")
    ck = eff_chk
    print(f"  前処理: 牌文字列の解釈 OK {ck['dahai_parse_ok']:,} / NG {ck['dahai_parse_ng']:,}"
          f"   同じ牌種 {ck['same_kind']:,}   モデル牌が手牌に無い {ck['model_not_in_hand']:,}"
          f"   例外 {ck['err']:,}   対象外 {ck['skip']:,}")
    if ck["dahai_parse_ng"] > ck["dahai_parse_ok"] * 0.01:
        print("  ！ 牌文字列の解釈 NG が多い。_hand_strs の形式を確認すること（以下の数字は信用しない）")
    cols = [("m_shan", "モ向聴負"), ("m_uk", "モ受入負"), ("m_uk_near", "モ受入≤2"),
            ("eq_safe", "同等安全"), ("eq_ron", "同等当り"),
            ("h_uk_near", "人受入≤2"), ("h_uk", "人受入負"), ("h_shan", "人向聴負")]
    print(f"\n  {'':<10}{'件数':>7}" + "".join(f"{lbl:>10}" for _, lbl in cols))
    order_g = ["全体", "平場", "被リーチ", "向聴0", "向聴1", "向聴2", "向聴3"]
    for g in order_g:
        e = eff.get(g)
        if not e or e["n"] == 0:
            continue
        n = e["n"]
        print(f"  {g:<10}{int(n):>7,}" + "".join(f"{e[k]/n*100:>9.1f}%" for k, _ in cols))
    print("  （向聴は人間の打牌後。向聴3 は 3 以上。受入≤2 は同向聴で受け入れ差が 2 枚以内）")

    e = eff.get("全体")
    if e and e["n"]:
        print("\n  区分ごとのモデルの確信度（平均 p(モデル top1) / 平均 p(人間の牌)）")
        for k, lbl in cols:
            if e[k]:
                print(f"    {lbl:<10} {e[k+'_ptop']/e[k]:.3f} / {e[k+'_phum']/e[k]:.3f}   (n={int(e[k]):,})")
        eq = e["eq_safe"]
        near = e["m_uk_near"] + e["h_uk_near"]
        print(f"\n  同等かつ安全を正解扱いにした一致率: {t1:.4f} → {(top1+eq)/max(n_all,1):.4f}")
        print(f"  さらに受け入れ差 2 枚以内も含めると: {(top1+eq+near)/max(n_all,1):.4f}")
        print(f"  同じ牌種の不一致（赤／黒・リーチ有無）: 全打牌の {ck['same_kind']/max(n_all,1)*100:.2f}%")

    print("=" * W)
    print("[6] 被リーチ時の押し引き（打牌後の手牌に残るリーチ者の当たり牌）★")
    fn, nn = dchk["flag_n"], dchk["noflag_n"]
    if fn:
        print(f"  対応の自己検査: リーチ者の行が空でない {dchk['flag_nonempty']/fn*100:.1f}% (n={fn:,})"
              f"   非リーチ者の行が空でない {dchk['noflag_nonempty']/max(nn,1)*100:.1f}% (n={nn:,})")
        if dchk["flag_nonempty"] / fn < 0.8:
            print("  ！ リーチ者の行が空のことが多い。numeric[6:9] と aux_ron の相対位置の対応がずれている"
                  "可能性がある（以下の数字は信用しない）")
    print(f"\n  {'':<14}{'件数':>7}{'人間':>8}{'モデル':>8}{'差(モ−人)':>11}"
          f"{'モが多い':>9}{'同じ':>7}{'人が多い':>9}{'人0枚':>8}{'モ0枚':>8}{'全聴牌者 人/モ':>16}")
    for g in ("全体", "人間だけ降り", "モデルだけ降り", "向聴同じ", "向聴0", "向聴1", "向聴2", "向聴3"):
        e = dang.get(g)
        if not e or e["n"] == 0:
            continue
        n = e["n"]
        hr, mr = e["h_r"] / n, e["m_r"] / n
        print(f"  {g:<14}{int(n):>7,}{hr:>8.2f}{mr:>8.2f}{mr-hr:>+11.2f}"
              f"{e['m_more']/n*100:>8.1f}%{e['eq']/n*100:>6.1f}%{e['h_more']/n*100:>8.1f}%"
              f"{e['h_zero']/n*100:>7.1f}%{e['m_zero']/n*100:>7.1f}%"
              f"{e['h_a']/n:>9.2f} / {e['m_a']/n:.2f}")
    e = dang.get("全体")
    if e and e["n"]:
        a, c = e["m_more"], e["h_more"]
        z = (a - c) / math.sqrt(a + c) if a + c else 0.0
        print(f"\n  符号検定 z = {z:+.2f}  （正ならモデルの手に当たり牌が多く残る＝押しすぎ）")
        print(f"  この打牌での即放銃: 人間 {int(e['h_ron_now'])}  モデル {int(e['m_ron_now'])}")
    print("  （枚数は打牌後の手牌のうちリーチ者の待ちに含まれる牌の数。向聴は人間の打牌後）")
    print("=" * W)


if __name__ == "__main__":
    main()
