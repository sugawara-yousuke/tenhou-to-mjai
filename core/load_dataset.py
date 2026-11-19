"""
load_dataset.py
===============
NikkeTryHard/tenhou-to-mjai データセットローダー。

event_filter:
  None / "unified"   統合学習（推奨）: 打牌(リーチ選択込み) ＋ 鳴き判断 ＋ 槓判断
  "dahai"            自分の手番の打牌判断のみ（リーチ宣言も同じサンプルのラベルとして含む）
  "meld_decision"    鳴き判断のみ
  "kan_decision"     暗槓/加槓判断のみ
  "riichi_decision"  リーチ可能だった手番のみ（riichi_label 付き）
  "dahai_and_meld"   打牌 ＋ 鳴き判断

[今回の改修]
  - 旧 "chi"/"pon" イベントサンプル（候補が [PASS, 鳴き] で正解が常に 1 の自明サンプル）を廃止。
    旧 "dahai" フィルタはこれを「鳴き後打牌」と誤認して混ぜており、精度の底上げと
    鳴き過剰バイアスの原因になっていた。鳴き後の打牌は普通の "dahai" サンプルに入っている。
  - cand_feats（候補ごとの特徴）をテンソル化。
  - シャッフルバッファを実装（旧版はヘッダに書いてあるだけで未実装。1バッチがほぼ同じ半荘の
    連続局面で埋まり、勾配が強く相関していた）。
  - DataLoader の worker 分割に対応（num_workers>0 でもデータが重複しない）。
"""

from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
import gzip
import json
import random
import zipfile
from pathlib import Path
from typing import Iterator

from mahjong_tokenizer import MahjongTokenizer

_FILTERS = {
    None:              lambda s: True,
    "unified":         lambda s: True,
    "dahai":           lambda s: s["event_type"] == "dahai",
    "meld_decision":   lambda s: s["event_type"] == "meld_decision",
    "kan_decision":    lambda s: s["event_type"] == "kan_decision",
    "riichi_decision": lambda s: "riichi_label" in s,
    "dahai_and_meld":  lambda s: s["event_type"] in ("dahai", "meld_decision"),
}


class TenhouDatasetLoader:
    AUX_CACHE_VERSION = 5   # v5: 当たり牌ラベル aux_ron を追加

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.tokenizer = MahjongTokenizer()
        self._game_count_cache: int | None = None

    def _read_mjson(self, raw_bytes: bytes) -> list[dict]:
        text = (gzip.decompress(raw_bytes) if raw_bytes[:2] == b'\x1f\x8b' else raw_bytes).decode("utf-8")
        events = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

    def iter_games(self, max_games: int | None = None, offset: int = 0,
                   shard: tuple[int, int] = (0, 1)) -> Iterator[tuple[str, list[dict]]]:
        """shard=(i, n): i 番目の worker は n 個おきに読む（DataLoader の重複防止）。"""
        wi, wn = shard
        if self.path.suffix == ".zip":
            with zipfile.ZipFile(self.path) as zf:
                names = sorted(n for n in zf.namelist() if n.endswith(".mjson"))[offset:]
                if max_games is not None:
                    names = names[:max_games]
                for name in names[wi::wn]:
                    yield name, self._read_mjson(zf.read(name))
        elif self.path.is_dir():
            files = sorted(self.path.glob("**/*.mjson"))[offset:]
            if max_games is not None:
                files = files[:max_games]
            for fpath in files[wi::wn]:
                yield str(fpath), self._read_mjson(fpath.read_bytes())
        elif offset == 0 and wi == 0:
            yield str(self.path), self._read_mjson(self.path.read_bytes())

    def iter_samples(self, max_games=None, offset=0, event_filter: str | None = None,
                     shard=(0, 1)) -> Iterator[dict]:
        if event_filter not in _FILTERS:
            raise ValueError(f"unknown event_filter: {event_filter!r}  (choices: {list(_FILTERS)})")
        keep = _FILTERS[event_filter]
        for game_id, events in self.iter_games(max_games=max_games, offset=offset, shard=shard):
            for sample in self.tokenizer.process_events(events):
                if keep(sample):
                    sample["game_id"] = game_id
                    yield sample

    def _aux_cache_path(self, max_games, offset, event_filter, shard):
        import hashlib
        try:
            st = self.path.stat()
            sig = f"{self.path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
        except OSError:
            sig = str(self.path.resolve())
        key = f"{sig}|mg={max_games}|off={offset}|filt={event_filter}|sh={shard}|v={self.AUX_CACHE_VERSION}"
        cache_dir = Path(".aux_cache"); cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"aux_{hashlib.md5(key.encode()).hexdigest()[:16]}.pkl"

    def iter_samples_with_aux(self, max_games=None, offset=0, event_filter: str | None = "dahai",
                              use_cache: bool = True, shard=(0, 1)) -> Iterator[dict]:
        """補助ラベル(放銃/他家テンパイ/受け入れ/当たり牌)付き。重い計算は .aux_cache にキャッシュ。"""
        self.tokenizer._emit_aux_fields = True
        import pickle
        from aux_labels import attach_aux_labels, attach_ron_labels, attach_return_labels
        keep = _FILTERS[event_filter]
        cache_file = self._aux_cache_path(max_games, offset, event_filter, shard) if use_cache else None
        cached = None
        if cache_file is not None and cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
            except Exception:
                cached = None
        collected = [] if (use_cache and cached is None) else None
        idx = 0
        for game_id, events in self.iter_games(max_games=max_games, offset=offset, shard=shard):
            game_samples = list(self.tokenizer.process_events(events))
            if cached is None:
                attach_aux_labels(self.tokenizer, events, game_samples)
                attach_ron_labels(game_samples)
            # 局リターン（AWR 用）は点数の四則演算だけで軽いのでキャッシュに入れず毎回計算する。
            # → AUX_CACHE_VERSION を上げずに済む（既存キャッシュがそのまま使える）
            attach_return_labels(events, game_samples)
            for sample in game_samples:
                if not keep(sample):
                    continue
                sample["game_id"] = game_id
                if cached is not None:
                    rec = cached[idx]; idx += 1
                    (sample["aux_houjuu"], sample["aux_tenpai_opp"],
                     sample["aux_ukeire"], sample["aux_valid"], ron_sparse, ron_valid) = rec
                    # aux_ron は「1が立つ位置だけ」で保存してある（密に持つと数GBになる）
                    ron = [[0] * 34 for _ in range(3)]
                    for r, k in ron_sparse:
                        ron[r][k] = 1
                    sample["aux_ron"] = ron
                    sample["aux_ron_valid"] = ron_valid
                elif collected is not None:
                    ron = sample.get("aux_ron") or [[0] * 34 for _ in range(3)]
                    ron_sparse = tuple((r, k) for r in range(3) for k in range(34) if ron[r][k])
                    collected.append((sample.get("aux_houjuu", 0), sample.get("aux_tenpai_opp", [0, 0, 0]),
                                      sample.get("aux_ukeire", 0.0), sample.get("aux_valid", 0),
                                      ron_sparse, sample.get("aux_ron_valid", 0)))
                yield sample
        if collected is not None and cache_file is not None:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(collected, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass

    def count_games(self) -> int:
        if self._game_count_cache is None:
            if self.path.suffix == ".zip":
                with zipfile.ZipFile(self.path) as zf:
                    self._game_count_cache = sum(1 for n in zf.namelist() if n.endswith(".mjson"))
            elif self.path.is_dir():
                self._game_count_cache = sum(1 for _ in self.path.glob("**/*.mjson"))
            else:
                self._game_count_cache = 1
        return self._game_count_cache


# ---------------------------------------------------------------------------
# PyTorch Dataset ラッパー
# ---------------------------------------------------------------------------
try:
    import torch
    from torch.utils.data import IterableDataset, get_worker_info
    from torch.nn.utils.rnn import pad_sequence

    def _to_item(sample: dict, with_aux: bool) -> dict:
        item = {
            "sparse":      torch.tensor(sample["sparse"],           dtype=torch.long),
            "numeric":     torch.tensor(sample["numeric"],          dtype=torch.float32),
            "progression": torch.tensor(sample["progression"] or [0], dtype=torch.long),
            "candidates":  torch.tensor(sample["possible_actions"], dtype=torch.long),
            "cand_feats":  torch.tensor(sample["cand_feats"],       dtype=torch.float32),
            "label":       torch.tensor(sample["action_label"],     dtype=torch.long),
        }
        if "meld_label" in sample:
            item["meld_label"]     = torch.tensor(sample["meld_label"],     dtype=torch.long)
            item["called_tile_id"] = torch.tensor(sample["called_tile_id"], dtype=torch.long)
        if "riichi_label" in sample:
            item["riichi_label"] = torch.tensor(sample["riichi_label"], dtype=torch.long)
        if with_aux:
            item["aux_ret"]       = torch.tensor(sample.get("aux_ret", 0.0),               dtype=torch.float32)
            item["aux_ret_valid"] = torch.tensor(sample.get("aux_ret_valid", 0),           dtype=torch.long)
            item["aux_ron"]       = torch.tensor(sample.get("aux_ron") or [[0] * 34 for _ in range(3)],
                                                 dtype=torch.float32)
            item["aux_ron_valid"] = torch.tensor(sample.get("aux_ron_valid", 0),          dtype=torch.long)
            item["aux_houjuu"]     = torch.tensor(sample.get("aux_houjuu", 0),            dtype=torch.long)
            item["aux_tenpai_opp"] = torch.tensor(sample.get("aux_tenpai_opp", [0, 0, 0]), dtype=torch.float32)
            item["aux_ukeire"]     = torch.tensor(sample.get("aux_ukeire", 0.0),          dtype=torch.float32)
            item["aux_valid"]      = torch.tensor(sample.get("aux_valid", 0),             dtype=torch.long)
        return item

    class TenhouTorchDataset(IterableDataset):
        """
        max_games は「半荘数」。shuffle_buffer>0 でバッファシャッフル
        （バッファが埋まったらランダムに1件ずつ吐く。半荘をまたいで局面が混ざる）。
        """
        # 2024.zip の実測（20半荘）: 打牌 510 / 鳴き判断 132 / 槓判断 7
        _AVG_SAMPLES = {None: 650, "unified": 650, "dahai": 510, "meld_decision": 132,
                        "kan_decision": 7, "riichi_decision": 25, "dahai_and_meld": 642}
        WITH_AUX = False

        def __init__(self, path, max_games=None, offset=0, event_filter=None,
                     shuffle_buffer: int = 0, seed: int = 0):
            self.loader, self.max_games, self.offset = TenhouDatasetLoader(path), max_games, offset
            self.event_filter, self.shuffle_buffer, self.seed = event_filter, shuffle_buffer, seed
            self._epoch = 0

        def _source(self, shard):
            return self.loader.iter_samples(max_games=self.max_games, offset=self.offset,
                                            event_filter=self.event_filter, shard=shard)

        def __iter__(self):
            wi = get_worker_info()
            shard = (wi.id, wi.num_workers) if wi is not None else (0, 1)
            self._epoch += 1
            rng = random.Random(self.seed * 1000003 + self._epoch * 101 + shard[0])
            src = self._source(shard)
            if self.shuffle_buffer <= 1:
                for s in src:
                    yield _to_item(s, self.WITH_AUX)
                return
            buf = []
            for s in src:
                if len(buf) < self.shuffle_buffer:
                    buf.append(s); continue
                j = rng.randrange(len(buf))
                buf[j], s = s, buf[j]
                yield _to_item(s, self.WITH_AUX)
            rng.shuffle(buf)
            for s in buf:
                yield _to_item(s, self.WITH_AUX)

        def __len__(self) -> int:
            """概算サンプル数（半荘数 × 平均サンプル数）。半荘数が欲しいときは n_games()。"""
            return self.n_games() * self._AVG_SAMPLES.get(self.event_filter, 650)

        def n_games(self) -> int:
            total = max(self.loader.count_games() - self.offset, 0)
            return min(self.max_games, total) if self.max_games else total

        @staticmethod
        def collate_fn(batch: list[dict]) -> dict:
            out = {
                "sparse":      pad_sequence([b["sparse"]      for b in batch], batch_first=True),
                "numeric":     torch.stack([b["numeric"]      for b in batch]),
                "progression": pad_sequence([b["progression"] for b in batch], batch_first=True),
                "candidates":  torch.stack([b["candidates"]   for b in batch]),
                "cand_feats":  torch.stack([b["cand_feats"]   for b in batch]),
                "label":       torch.stack([b["label"]        for b in batch]),
            }
            for k in ("aux_houjuu", "aux_tenpai_opp", "aux_ukeire", "aux_valid",
                      "aux_ron", "aux_ron_valid", "aux_ret", "aux_ret_valid"):
                if k in batch[0]:
                    out[k] = torch.stack([b[k] for b in batch])
            return out

        collate_fn_aux = collate_fn

        @staticmethod
        def collate_fn_meld(batch):
            base = TenhouTorchDataset.collate_fn(batch)
            if all("meld_label" in b for b in batch):
                base["meld_label"]     = torch.stack([b["meld_label"]     for b in batch])
                base["called_tile_id"] = torch.stack([b["called_tile_id"] for b in batch])
            return base

        @staticmethod
        def collate_fn_riichi(batch):
            base = TenhouTorchDataset.collate_fn(batch)
            if all("riichi_label" in b for b in batch):
                base["riichi_label"] = torch.stack([b["riichi_label"] for b in batch])
            return base

    class TenhouAuxDataset(TenhouTorchDataset):
        """補助タスク（放銃/他家テンパイ/受け入れ）付き。補助ラベルは打牌サンプルにだけ付く。"""
        WITH_AUX = True

        def __init__(self, path, max_games=None, offset=0, event_filter="unified",
                     shuffle_buffer: int = 0, seed: int = 0, use_cache: bool = True):
            super().__init__(path, max_games, offset, event_filter, shuffle_buffer, seed)
            self.use_cache = use_cache

        def _source(self, shard):
            return self.loader.iter_samples_with_aux(max_games=self.max_games, offset=self.offset,
                                                     event_filter=self.event_filter, shard=shard,
                                                     use_cache=self.use_cache)

except ImportError:
    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--max-games", type=int, default=5)
    parser.add_argument("--filter", type=str, default=None)
    args = parser.parse_args()
    loader = TenhouDatasetLoader(args.path)
    print(f"総試合数: {loader.count_games()}")
    counts: dict[str, int] = {}; rl = {0: 0, 1: 0}; n = 0
    for s in loader.iter_samples(max_games=args.max_games, event_filter=args.filter):
        n += 1
        counts[s["event_type"]] = counts.get(s["event_type"], 0) + 1
        if "riichi_label" in s:
            rl[s["riichi_label"]] += 1
    print(f"取得サンプル数: {n}  (filter={args.filter})")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v:6d}")
    print(f"riichi_label 0(ダマ):{rl[0]}  1(立直):{rl[1]}")
