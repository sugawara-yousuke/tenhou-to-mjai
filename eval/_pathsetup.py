"""_pathsetup: サブフォルダ分割後も、フラット時代の
`from mahjong_model import ...` のような同一階層前提の相対importを
そのまま動かすためのパス補助。

使い方: 各スクリプト冒頭（プロジェクト内モジュールのimportより前）に
    import _pathsetup  # noqa: F401
を1行入れる。mjai_project 配下の全コードフォルダを sys.path に追加する。
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)  # mjai_project/
for _sub in ("core", "features", "train", "rl", "eval", "serve", "debug"):
    _p = os.path.join(_root, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
