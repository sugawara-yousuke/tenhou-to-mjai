# tenhou-to-mjai — Transformer 麻雀AI（天鳳鳳凰卓の牌譜で模倣学習）

**English summary.** A four-player Japanese (riichi) mahjong AI that imitates Tenhou Houou-table players with a single Transformer deciding discards, riichi, calls and kans. It reaches the supervised-learning accuracy reported for Suphx (discard 77.0% vs 76.7%). The repository includes a game simulator following Tenhou rules (exceptions are listed below), a paired (duplicate) evaluation script, a browser replay viewer, and trained weights, so you can run games and evaluations right after cloning. Tenhou game logs are not included.

天鳳鳳凰卓の牌譜から、打牌・リーチ・鳴き・槓の判断を **1つの Transformer** で模倣学習した四人麻雀AIです。
自前の対局シミュレータ、同卓複式の評価スクリプト、ブラウザで見られる牌譜ビューアを含みます。
学習済みの重み（`weights/model_bc_best.pt`）を同梱しているので、clone してすぐに対局と評価を回せます。

- モデル: MahjongTransformer（約 2,000 万パラメータ、d_model 384、エンコーダ 6 層 / デコーダ 2 層）
- 入力: 手牌・河・副露などの集合トークン、打牌の履歴、点数状況や危険度などの数値特徴（46 次元）
- 出力: その局面で取れる候補（打牌・リーチ宣言牌・鳴き・見送り・槓）それぞれのスコア

## 精度（教師あり学習）

学習に使っていない牌譜での一致率です。Suphx 論文（Li et al., 2020）Table 3 の教師あり学習の値と並べています。

| 判断 | このモデル | Suphx（教師あり） |
|---|---|---|
| 打牌 | 77.0% | 76.7% |
| リーチ／ダマ | 85.3% | 85.7% |
| 鳴き・槓 | 92.8%（チー・ポン・槓の統合） | チー 95.0% / ポン 91.9% / 槓 94.0% |

学習データは 2024 年の鳳凰卓 5 万半荘 × 3 エポックです（Suphx は打牌に 1,500 万サンプル）。
データを 15 万半荘に増やしても一致率はほぼ変わらず（+0.1pt）、データ量よりも、模倣そのものの天井かモデルの容量が制約になっていると考えています（未検証）。

## 強さ

同梱の重みと、ルールベースの打ち手（`rule`、後述）を同じ卓で 2 人ずつ打たせた結果です（300 ペア = 600 半荘、同卓複式）。

| 打ち手 | 平均順位 | 和了率 | 放銃率 | リーチ率 |
|---|---|---|---|---|
| このモデル | 1.681 | 34.8% | 8.2% | 23.1% |
| ルールベース | 3.319 | 7.9% | 22.6% | 13.0% |

平均順位差は −1.638 ± 0.028（SE）です。

ルールベースは「切った後の向聴数が最小になる牌を切る」だけの弱い打ち手で、この結果はシミュレータとモデルが正しく動いていることの確認にとどまります。
天鳳での段位や、Mortal など既存の AI との比較はまだ測っていません。

## 必要なもの

- Python 3.8 以上
- PyTorch 2.2 以上（GPU があれば速いが、CPU でも動く）

```bash
git clone https://github.com/sugawara-yousuke/tenhou-to-mjai
cd tenhou-to-mjai
pip3 install -r requirements.txt
```

速度の目安（AI 4 人で東南戦 1 半荘）: RTX A6000 で約 3 秒、CPU で約 15〜20 秒。

## 使い方

すべてリポジトリのルートで実行します。`--device cpu` を付ければ GPU なしで動きます。

### 1. AI 同士で対局する

```bash
python3 rl/self_play.py --model weights/model_bc_best.pt --games 10 --out replays --seed 0
```

`replays/replay_0000.mjson` などに mjai 形式（1 行 1 イベントの JSON）の牌譜が保存されます。
`--watch` を付けると、打牌や鳴きを 1 手ずつ表示します。

### 2. 2 つの打ち手の強さを比べる（同卓複式）

```bash
python3 eval/eval_paired.py --model-a weights/model_bc_best.pt --model-b rule --pairs 100
```

`--model-a` / `--model-b` には `.pt` のパスか、ルールベースの打ち手を表す `rule` を渡します。
同じ山・同じ席で A と B を入れ替えて 2 回打ち、配牌の運を相殺して比べます。
出力は平均順位差（SE と 95% 信頼区間つき）と、放銃率・和了率・リーチ率の差です。

- 動作確認: A と B に同じファイルを渡すと、ペア差がすべて 0 になります。
- 判定のコツ: 平均順位は数百半荘では誤差が大きいので（600 半荘で SE ±0.05 程度）、局単位の放銃率・和了率を主に見てください。

`rule` は「切った後の向聴数が最小になる牌を切る」だけの簡易な打ち手で、動作確認と最低限の基準線用です。

### 3. 牌譜ビューアで見る

```bash
python3 serve/server.py
```

ブラウザで http://127.0.0.1:8000 を開くと、AI 同士の対局を生成して 1 手ずつ再生できます。
各手番でモデルが候補ごとにどれだけの確率を付けたかも表示されます。生成した牌譜は `replays/` に保存されます。

人間が AI と対戦する機能（`/human/*`）は、現在このリポジトリには含まれていません。

## 自分で学習する

天鳳の牌譜は利用規約により再配布できないため、牌譜そのものも、そこから作ったデータセットもこのリポジトリには含めていません。各自で入手してください。

学習スクリプトが読むのは、`YYYY/*.mjson` を並べた zip です（各ファイルは gzip 圧縮した mjai 形式の JSON Lines）。
Hugging Face で配布されている `tenhou-houou-YYYY.tar`（中身は `*.mjai.json.zst`）形式なら、次で変換できます。

```bash
python3 debug/convert_hf_tar.py data/hf/tenhou-houou-2024.tar data/2024.zip
```

学習（同梱の重みは 5 万半荘 × 3 エポック、A6000 で約 2 日）:

```bash
python3 train/train_bc1.py --data data/2024.zip --max-games 50000 --epochs 3 --shuffle-buffer 30000 --save-every 2000 --save-dir train/checkpoints1
```

- `--max-games` は半荘数です。まずは `--max-games 300 --epochs 1` で動作確認するのがおすすめです。
- `--aux-coef 0.3` で補助ヘッド（放銃・テンパイ・受け入れ・当たり牌）を同時に学習します。
- 学習の所要時間の目安: ラベル計算 0.24 秒 / 半荘、学習 1.0 秒 / 半荘。

## フォルダ構成

| フォルダ | 中身 |
|---|---|
| `core/` | トークナイザ（局面 → モデル入力）、モデル本体、データローダ |
| `features/` | 危険度・現物などの特徴量、補助ヘッドのラベル（牌譜の全員の手牌から作る確定値） |
| `train/` | 模倣学習（`train_bc1.py`） |
| `rl/` | 対局シミュレータ（`self_play.py`）、役・点数計算（`yaku.py`）、強化学習の実験コード |
| `eval/` | 同卓複式の評価（`eval_paired.py`）、ルールベースの打ち手、押し引きの集計など |
| `serve/` | 牌譜ビューア（FastAPI + `ui.html`） |
| `debug/` | テストと診断用スクリプト |
| `weights/` | 学習済みの重み |

各スクリプトは冒頭の `import _pathsetup` で、フォルダをまたいだ import を解決しています。

## テスト

```bash
python3 debug/test_yaku.py
python3 debug/test_tokenizer.py
python3 debug/test_rules_heuristic.py
```

役・符・点数の計算、トークナイザの局面更新、シミュレータのルール（点棒の保存、和了に役があるか）を確認します。

## ルールの対応状況

天鳳の段位戦（東南戦・喰いタンあり・赤ありの四人麻雀）に合わせています。

- 対応済み: 飛び終了、南 4 局で 30,000 点未満なら西入（サドンデス）、オーラスの和了止め・聴牌止め、一般的な役・符計算、フリテン（同巡・リーチ後）、赤ドラ、裏ドラ、槓ドラ（即めくり）、リーチ・一発・海底・河底・嶺上、ダブロンは頭ハネ、流局時のテンパイ料
- 未実装: パオ（責任払い）、流し満貫、四槓散了、四家立直、九種九牌、国士無双の暗槓搶槓、リーチ後の暗槓
- 未検証: 点数額を点数表と網羅的に照合すること

## これまでに試したこと

模倣学習のあとに強化学習でさらに強くする実験もしていて、そのコードも `rl/` に残しています。結果は次のとおりです。

- **GRPO（7 回）**: いずれも模倣学習のモデルに有意に勝てませんでした。同じ局面から局の終わりまで打ち切っても、枝どうしの順位の相関が 0.24 しかなく、報酬が「どちらの手が良かったか」を測れていないことが原因でした。
- **AWR（優位度で重み付けした模倣学習）**: 一致率は少し上がりましたが、対戦では元のモデルと互角でした。鳳凰卓の打ち手はすでに強く、重み付けの信号が弱いためです。
- **当たり牌を予測する補助ヘッド**: AUC 0.94 で、手計算の危険度（0.91）を上回りました。

## 既知の問題

- 牌譜ビューアの画面に人間対局のボタンが残っていますが、対応するサーバー側の機能（`human_game.py`）が含まれていないため、押すとエラー（501）になります。
- 同卓複式でも、鳴きで打順がずれると以降のツモが分岐するため、配牌の運を相殺する効果は小さめです（上の評価では SE がほぼ変わりませんでした）。

## 参考

- J. Li et al., "Suphx: Mastering Mahjong with Deep Reinforcement Learning", arXiv:2003.13590, 2020.
- [mjai](https://github.com/gimite/mjai) — 麻雀AIどうしの対局プロトコル。牌譜の形式に使っています。
- [Mortal](https://github.com/Equim-chan/Mortal) — OSS の麻雀AI。設計の比較対象として参照しました。

## ライセンス

MIT License（`LICENSE`）。学習済みの重みも同じ条件で使えます。
天鳳の牌譜そのもの、およびそこから作ったデータセットは含まれていません。
