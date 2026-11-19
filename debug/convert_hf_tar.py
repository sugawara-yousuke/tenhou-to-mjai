"""
convert_hf_tar.py  ―  Hugging Face 版 (tenhou-houou-YYYY.tar, 中身は *.mjai.json.zst) を
                      手元の 2024.zip と同じ形式 (YYYY/*.mjson, 各ファイル gzip 圧縮) の zip に変換する

  pip3 install zstandard
  python3 debug/convert_hf_tar.py data/hf/tenhou-houou-2021.tar data/2021.zip

・tar は展開せずにストリームで読むので、追加のディスクは出力 zip の分だけ（元の tar とほぼ同じ大きさ）。
・中身が JSON Lines でも JSON 配列でも受ける。出力は 1 行 1 イベントの JSON Lines。
・最初の数半荘で start_game に names が入っているかを表示する（選別に使えるかの確認）。
・出力 zip は ZIP_STORED（中身がすでに gzip なので二重圧縮しない）。load_dataset.py がそのまま読める。
"""
from __future__ import annotations
import argparse
import gzip
import io
import json
import sys
import tarfile
import time
import zipfile
from pathlib import Path


def _to_jsonl(raw: bytes) -> bytes:
    text = raw.decode("utf-8").strip()
    if text.startswith("["):
        events = json.loads(text)
        return ("\n".join(json.dumps(e, ensure_ascii=False, separators=(",", ":")) for e in events) + "\n").encode("utf-8")
    return (text + "\n").encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="tenhou-houou-YYYY.tar")
    ap.add_argument("dst", type=Path, help="出力 zip（例: data/2021.zip）")
    ap.add_argument("--gzip-level", type=int, default=6)
    ap.add_argument("--check", type=int, default=5, help="names を確認する半荘数")
    args = ap.parse_args()

    try:
        import zstandard
    except ImportError:
        sys.exit("zstandard が無い。pip3 install zstandard を先に実行すること")
    if args.dst.exists():
        sys.exit(f"{args.dst} は既にある。上書きしないので消すか別名にすること")

    dctx = zstandard.ZstdDecompressor()
    n = n_err = 0
    checked = 0
    t0 = time.time()
    tmp = args.dst.with_suffix(".zip.part")
    with tarfile.open(args.src, "r|*") as tf, zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".zst"):
                continue
            base = Path(m.name).name
            stem = base[:-len(".mjai.json.zst")] if base.endswith(".mjai.json.zst") else base[:-len(".zst")]
            year = stem[:4] if stem[:4].isdigit() else "unknown"
            try:
                raw = dctx.stream_reader(io.BytesIO(tf.extractfile(m).read())).read()
                jl = _to_jsonl(raw)
            except Exception as e:
                n_err += 1
                if n_err <= 5:
                    print(f"  読めない: {m.name} ({e})")
                continue
            if checked < args.check:
                first = json.loads(jl.split(b"\n", 1)[0])
                print(f"  {stem}: 先頭 type={first.get('type')}  names={first.get('names')}")
                checked += 1
            zf.writestr(f"{year}/{stem}.mjson", gzip.compress(jl, compresslevel=args.gzip_level))
            n += 1
            if n % 20000 == 0:
                print(f"  {n:,} 半荘  {time.time()-t0:.0f}s")
    tmp.rename(args.dst)
    print(f"完了: {n:,} 半荘 → {args.dst}   読めなかった {n_err}   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
