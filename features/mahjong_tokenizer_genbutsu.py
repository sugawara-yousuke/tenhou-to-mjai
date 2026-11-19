"""
mahjong_tokenizer_genbutsu.py ― 互換用の別名。
現物特徴は mahjong_tokenizer.MahjongTokenizer.build_numeric に統合した（numeric[42:46]）。
19→23 の張り替え(adapt_numeric_dim.py)は不要。
"""
from mahjong_tokenizer import MahjongTokenizer as MahjongTokenizerGenbutsu, NUMERIC_DIM as NUMERIC_DIM_G  # noqa: F401
