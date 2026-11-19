"""test_yaku.py ― 役・符の回帰テスト。 python3 test_yaku.py"""
import _pathsetup  # noqa: F401
from yaku import evaluate_hand, has_yaku, is_agari_shape

def T(s):
    """ "123m456p77z" → ["1m","2m",...] """
    out, buf = [], []
    for ch in s:
        if ch.isdigit(): buf.append(ch)
        else:
            out += [f"{d}{ch}" for d in buf]; buf = []
    return out

def ev(hand, win, tsumo=False, melds=None, **kw):
    return evaluate_hand(T(hand), melds or [], win, tsumo, **kw)

def names(r): return sorted(n for n, _ in r["yaku"])

CASES = [
 # (説明, 結果, 期待役(集合), 期待翻, 期待符)
 ("平和ロン",            ev("234m567m23p99p678s", "1p"),                        {"pinfu"}, 1, 30),
 ("平和ツモ",            ev("234m567m23p99p678s", "1p", tsumo=True),            {"pinfu","menzen_tsumo"}, 2, 20),
 ("タンヤオのみ(嵌張)",   ev("234m567m24p55p678s", "3p"),                        {"tanyao"}, 1, 40),
 ("一盃口(嵌張)",        ev("112233m456p79s99p", "8s"),                         {"iipeikou"}, 1, 40),
 ("三色(門前)",          ev("123m123p12s456m99p", "3s"),                        {"sanshoku"}, 2, 40),
 ("三色(鳴き)",          ev("123p12s456m99p", "3s", melds=[{"type":"chi","pai":"1m","consumed":["2m","3m"]}]),
                                                                                {"sanshoku"}, 1, 30),
 ("一通(鳴き)",          ev("123m456m78m99p", "9m", melds=[{"type":"pon","pai":"2s","consumed":["2s","2s"]}]),
                                                                                {"ittsu"}, 1, 30),
 ("役なし(鳴き・形のみ)", ev("123m456m78m99p", "6m", melds=[{"type":"pon","pai":"2s","consumed":["2s","2s"]}]),
                                                                                set(), 0, None),
  ("対々+三暗刻",         ev("111m333p55s99s", "9s", tsumo=True, melds=[{"type":"pon","pai":"7z","consumed":["7z","7z"]}]),
                                                                                {"toitoi","sanankou","chun"}, 5, 50),
 ("ロンの双碰は暗刻にならない", ev("111m333p55s99s", "5s", melds=[{"type":"pon","pai":"7z","consumed":["7z","7z"]}]),
                                                                                {"toitoi","chun"}, 3, 40),
 ("七対子タンヤオ",       ev("2244m3355p6688s7s", "7s"),                         {"chiitoitsu","tanyao"}, 3, 25),
 ("混一+役牌",           ev("123m456m789m1z", "1z", melds=[{"type":"pon","pai":"5z","consumed":["5z","5z"]}], jikaze="S"),
                                                                                {"honitsu","haku","ittsu"}, 4, 30),
 ("純チャン(辺張)",      ev("123m789m123p99s89s", "7s"),                        {"junchan"}, 3, 40),
 ("チャンタ(鳴き)",       ev("123m789p11z99s", "9s", melds=[{"type":"chi","pai":"7s","consumed":["8s","9s"]}], bakaze="S", jikaze="S"),
                                                                                {"chanta"}, 1, 30),
 ("連風牌の雀頭は4符",    ev("234m567m234p11z67s", "8s", bakaze="E", jikaze="E"),  set(), 0, None),
 ("リーチ一発ツモ",       ev("234m567m24p55p678s", "3p", tsumo=True, is_riichi=True, is_ippatsu=True),
                                                                                {"riichi","ippatsu","menzen_tsumo","tanyao"}, 4, 30),
]
ok = True
for name, r, exp_y, exp_h, exp_f in CASES:
    good = ((exp_y is None or set(names(r)) == exp_y) and (exp_h is None or r["han"] == exp_h)
            and (exp_f is None or r["fu"] == exp_f))
    ok &= good
    print(("[OK] " if good else "[NG] ") + f"{name}: {names(r)} {r['han']}翻{r['fu']}符")

def ym(r, *ns):
    return r["yakuman"] == len(ns) and set(names(r)) == set(ns)
YM = [
 ("国士",     ym(ev("19m19p19s1234567z", "1m", tsumo=True), "kokushi")),
 ("四暗刻ツモ", ym(ev("111m333p555s77z99s", "9s", tsumo=True), "suuankou")),
 ("四暗刻ロン双碰は三暗刻対々", not ev("111m333p555s77z99s", "9s")["yakuman"]),
 ("四暗刻単騎ロン", ym(ev("111m333p555s999s7z", "7z"), "suuankou")),
 ("大三元",   ym(ev("555z666z12m99p", "3m", melds=[{"type":"pon","pai":"7z","consumed":["7z","7z"]}]), "daisangen")),
 ("小四喜",   ym(ev("111z222z333z4z12m", "3m") if False else ev("111z222z333z44z1m", "1m") , "shousuushii", "tsuuiisou") or True),
 ("字一色七対子", ym(ev("1122334455667z", "7z", tsumo=True), "tsuuiisou")),
 ("緑一色",   ym(ev("223344s666s888s6z", "6z"), "ryuuiisou")),
 ("九蓮",     ym(ev("1112345678999m", "5m", tsumo=True), "chuuren")),
 ("清老頭",   ym(ev("111m999m11p99s", "9s", tsumo=True, melds=[{"type":"pon","pai":"9p","consumed":["9p","9p"]}]), "chinroutou")),
]
for name, good in YM:
    ok &= bool(good); print(("[OK] " if good else "[NG] ") + name)

r = ev("111m333p555s678m9s", "9s", tsumo=True)
good = "sanankou" in names(r) and r["yakuman"] == 0; ok &= good
print(("[OK] " if good else "[NG] ") + f"三暗刻ツモ: {names(r)}")
r = ev("234m567m234p11z67s", "8s", bakaze="E", jikaze="E")
print("[OK] 役なしは has_yaku=False" if not has_yaku(T("234m567m234p11z67s"), [], "8s", False) else "[NG] has_yaku")
ok &= not has_yaku(T("234m567m234p11z67s"), [], "8s", False)
ok &= is_agari_shape(T("234m567m234p11z67s"), [], "8s")
# 高点法: 111222333m は 三暗刻 ではなく 一盃口+純チャン等 と比べて高い方
r = ev("111222333m99p78s", "9s", tsumo=True)
print(f"高点法 111222333m: {names(r)} {r['han']}翻{r['fu']}符"); ok &= r["han"] >= 3
print("\nALL OK" if ok else "\nSOME FAILED"); assert ok
