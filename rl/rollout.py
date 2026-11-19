"""
rollout.py  ―  GRPO 用のグループ収集（スナップショット分岐 ＋ 共通乱数の N 山プレイアウト）
================================================================

手順:
  1. capture_game: rl_player がサンプリングで1半荘打つ。打牌判断ごとに KyokuCtx のスナップショットを記録。
  2. 分岐点を選ぶ。既定は「方策エントロピー＋被リーチ加点」に比例した抽選。
     方策が 0.99 で確信している局面ばかり選ぶと、どの枝も同じ結果になって信号が出ない。
  3. 分岐点で候補 G 個（argmax は必ず含め、残りは方策確率で抽選）を強制。
  4. N 本の「山の引き直し」を先に作り、全ての枝で同じ N 本を使う（共通乱数）。
     旧 plan3 は枝ごとに別の山を引いていたので、枝の差より山の運の差の方がずっと大きかった。
     同じ山で比べれば、差が出るのは「その打牌が効いたとき」だけになる。
     2手目以降は greedy で打つ（既定）。乱数は山だけなので、枝間の差＝初手の差。
  5. 報酬は reward.placement_reward（順位期待値の変化）。全枝同値のグループは捨てる。

旧 rollout_plan3(_iqn).py / plan3_playout.py はこのファイルに置き換え。
  - 「相手の手牌まで引き直す」(hidden) モードは廃止。リーチ者がテンパイでなくなり放銃が
    起きないので、危険牌を押す行動が過大評価される。
  - 手牌品質(向聴/受け入れ)のシェーピングも廃止。ベタオリは受け入れを減らす行動なので、
    受け入れ最大を常に加点すると押し引きの学習と正面から衝突する。
"""

from __future__ import annotations
import _pathsetup  # noqa: F401  (cross-folder import)
import math
import random
from dataclasses import dataclass, field

from self_play import MahjongSimulator, RLContext, WIND_STR, resample_wall, next_game_state
from reward import placement_reward, kyoku_left_before, near_reward_from_events
from mahjong_tokenizer import PADDING_ACTION_ID, ACT_RIICHI_BASE, ACT_PON_BASE


@dataclass
class RolloutGroup:
    sample: dict                 # 分岐点の局面（tokenizer.make_sample の dict）
    action_idxs: list            # 評価した候補 index（G 個）
    old_log_probs: list          # 収集時方策での log π(a)（温度適用後）
    rewards: list                # 各枝の N 山平均報酬
    reward_samples: list         # G × N の生サンプル
    kyoku_id: int
    rl_player: int
    under_threat: bool
    entropy: float


@dataclass
class RolloutConfig:
    G: int = 6
    n_playout: int = 3
    horizon: int = 0                 # >0 で「分岐から horizon 巡で打ち切り、局面価値で評価」
    resample: str = "wall"           # wall: 見えていない山を引き直す / none: 真の山のまま(N=1 相当)
    temperature: float = 1.0
    continuation: str = "greedy"
    reward: str = "placement"        # placement / near
    w_score: float = 0.1
    branch_weighting: str = "entropy"   # entropy / uniform
    threat_bonus: float = 0.5
    groups_per_capture: int = 8
    p_floor: float = 0.02
    tie_eps: float = 1e-6
    rotate_seat: bool = True


def _reward_of(cfg: RolloutConfig, snap, res: dict, events: list, player: int) -> float:
    if cfg.reward == "near":
        return near_reward_from_events(player, res["score_delta"], events)
    st = snap.state
    bk = WIND_STR.index(st.bakaze)
    left_before = kyoku_left_before(bk, st.kyoku)
    if res.get("truncated"):
        # 打ち切り: この局は「期待値どおりに決着した」とみなす。連荘は数えない近似。
        left_after = max(left_before - 1.0, 0.0)
    else:
        finished, bk2, kyoku2, _h, _o = next_game_state(bk, st.kyoku, st.honba, st.oya, res["scores"], res)
        left_after = 0.0 if finished else kyoku_left_before(bk2, kyoku2)
    # 局前の基準は「局開始時の点数」。分岐点までのリーチ棒などは全枝共通なので差には効かない。
    return placement_reward(snap.start_scores, res["scores"], player,
                            left_before, left_after, w_score=cfg.w_score)


def _safest_action(sample: dict, valid: list):
    """
    リーチ者に対する危険度が最小の候補 index。
      cand_feats[i][0:3] : 対 下家/対面/上家 の危険度（打牌・リーチ宣言候補のみ値あり）
      numeric[6:9]       : 下家/対面/上家 がリーチ中か
    リーチ者の危険度だけを見る（非リーチ者への微小な危険度で現物が負けないように）。
    同点ならリーチ宣言より通常打牌を優先する。
    """
    feats = sample.get("cand_feats")
    riichi_rel = [r for r in range(3) if sample["numeric"][6 + r] > 0.5]
    if not feats or not riichi_rel:
        return None
    poss = sample["possible_actions"]
    cands = [i for i in valid if poss[i] < ACT_PON_BASE]   # 打牌 or リーチ宣言
    if not cands:
        return None
    return min(cands, key=lambda i: (max(feats[i][r] for r in riichi_rel),
                                     poss[i] >= ACT_RIICHI_BASE))


def _pick_branch_actions(rec, G: int, p_floor: float, rng: random.Random) -> list:
    valid = [i for i, t in enumerate(rec.sample["possible_actions"]) if t != PADDING_ACTION_ID]
    if len(valid) <= G:
        return valid
    probs = {i: math.exp(rec.log_probs[i]) + p_floor for i in valid}
    top = max(valid, key=lambda i: rec.log_probs[i])
    chosen = [top]
    # 被リーチ時は「リーチ者に対して一番安全な牌」を必ず比較対象に入れる。
    # 方策が 0.99 で押している局面では、確率抽選だけだと降りの手がほぼ選ばれず、
    # 押し vs 降りの比較がグループに入らない。
    if rec.under_threat:
        safest = _safest_action(rec.sample, valid)
        if safest is not None and safest not in chosen:
            chosen.append(safest)
    pool = [i for i in valid if i not in chosen]
    while len(chosen) < G and pool:
        tot = sum(probs[i] for i in pool)
        x, acc = rng.random() * tot, 0.0
        for i in pool:
            acc += probs[i]
            if acc >= x:
                chosen.append(i); pool.remove(i); break
    return chosen


def collect_group(sim: MahjongSimulator, rec, cfg: RolloutConfig, rng: random.Random):
    """1つの分岐点(StepRecord)から1グループ。全枝同値なら None。"""
    snap = rec.snapshot
    if snap is None:
        return None
    idxs = _pick_branch_actions(rec, cfg.G, cfg.p_floor, rng)
    if len(idxs) < 2:
        return None
    n = max(1, cfg.n_playout) if cfg.resample == "wall" else 1
    worlds = [resample_wall(snap, rng) if cfg.resample == "wall" else snap for _ in range(n)]
    samples = []
    for a in idxs:
        row = []
        for W in worlds:
            rl = RLContext(rl_player=rec.player, temperature=cfg.temperature,
                           forced_first_idx=a, continuation=cfg.continuation, take_snapshots=False)
            res, events = sim.playout_from(W, rl, max_turns=(cfg.horizon if cfg.horizon > 0 else None))
            row.append(_reward_of(cfg, snap, res, events, rec.player))
        samples.append(row)
    rewards = [sum(r) / len(r) for r in samples]
    if max(rewards) - min(rewards) < cfg.tie_eps:
        return None
    return RolloutGroup(sample=rec.sample, action_idxs=idxs,
                        old_log_probs=[rec.log_probs[a] for a in idxs],
                        rewards=rewards, reward_samples=samples, kyoku_id=rec.kyoku_id,
                        rl_player=rec.player, under_threat=rec.under_threat, entropy=rec.entropy)


def collect_groups(sim: MahjongSimulator, num_groups: int, cfg: RolloutConfig,
                   rng: random.Random | None = None, max_captures: int | None = None):
    """
    num_groups 個のグループを集める。戻り値 (groups, stats)。
    stats: captures=打った半荘数, tried=試した分岐点数, tie=全枝同値で捨てた数
    """
    rng = rng or random.Random()
    groups, stats = [], {"captures": 0, "tried": 0, "tie": 0}
    max_captures = max_captures or max(4, num_groups)
    while len(groups) < num_groups and stats["captures"] < max_captures:
        seat = rng.randrange(4) if cfg.rotate_seat else 0
        rl = RLContext(rl_player=seat, temperature=cfg.temperature)
        sim.capture_game(rl)
        stats["captures"] += 1
        recs = [r for r in rl.records if r.snapshot is not None and r.n_valid >= 2]
        if not recs:
            continue
        if cfg.branch_weighting == "entropy":
            w = [r.entropy + 0.05 + (cfg.threat_bonus if r.under_threat else 0.0) for r in recs]
        else:
            w = [1.0] * len(recs)
        order = []
        pool = list(range(len(recs)))
        while pool and len(order) < cfg.groups_per_capture * 3:
            tot = sum(w[i] for i in pool)
            x, acc = rng.random() * tot, 0.0
            for i in pool:
                acc += w[i]
                if acc >= x:
                    order.append(i); pool.remove(i); break
        got = 0
        for i in order:
            if len(groups) >= num_groups or got >= cfg.groups_per_capture:
                break
            stats["tried"] += 1
            g = collect_group(sim, recs[i], cfg, rng)
            if g is None:
                stats["tie"] += 1
                continue
            groups.append(g); got += 1
    return groups, stats
