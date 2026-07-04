"""総当たり結果から Bradley-Terry (ロジスティック MLE) でレーティングを推定する。

入力は対戦ペアごとの勝敗集計。完全総当たりを想定しており、MM (minorization-
maximization) 反復で強さ p_i を求め、Elo スケール (400*log10) に変換する。
平均を 0 に正規化して相対比較できるようにする。引き分けは双方 0.5 勝で配分する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PairResult:
    """player a と b の対戦集計。a_win / b_win / draw は試合数。"""

    a: str
    b: str
    a_win: int
    b_win: int
    draw: int


def bradley_terry_ratings(
    players: list[str],
    pairs: list[PairResult],
    *,
    max_iter: int = 10_000,
    tol: float = 1e-9,
) -> dict[str, float]:
    """各 player の Elo 風レーティング (平均 0) を返す。

    引き分けは双方に 0.5 勝として加算する。グラフが連結でない場合の発散を避けるため、
    各 player に微小な仮想対戦 (両者 0.5 勝) を加えて正則化する。
    """
    idx = {p: i for i, p in enumerate(players)}
    n = len(players)
    if n == 0:
        return {}
    wins = [0.0] * n  # 各 player の総勝ち数 (引分は 0.5)
    games = [[0.0] * n for _ in range(n)]  # 対戦総数 (対称)
    for pr in pairs:
        i, j = idx[pr.a], idx[pr.b]
        wins[i] += pr.a_win + 0.5 * pr.draw
        wins[j] += pr.b_win + 0.5 * pr.draw
        g = pr.a_win + pr.b_win + pr.draw
        games[i][j] += g
        games[j][i] += g
    # 正則化: 全ペアに仮想 1 試合 (0.5-0.5) を足し、孤立や全勝/全敗での発散を防ぐ。
    eps = 1.0
    for i in range(n):
        for j in range(n):
            if i != j:
                games[i][j] += eps
                wins[i] += 0.5 * eps

    p = [1.0] * n
    for _ in range(max_iter):
        new_p = [0.0] * n
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                denom += games[i][j] / (p[i] + p[j])
            new_p[i] = wins[i] / denom if denom > 0 else p[i]
        # 幾何平均 1 に正規化。
        gm = math.exp(sum(math.log(max(x, 1e-300)) for x in new_p) / n)
        new_p = [x / gm for x in new_p]
        if max(abs(a - b) for a, b in zip(new_p, p)) < tol:
            p = new_p
            break
        p = new_p

    ratings = {players[i]: 400.0 * math.log10(p[i]) for i in range(n)}
    mean = sum(ratings.values()) / n
    return {k: v - mean for k, v in ratings.items()}


if __name__ == "__main__":
    # 自己検証: 強さ順が明確な 3 者。a >> b >> c なら rating(a) > rating(b) > rating(c)。
    players = ["a", "b", "c"]
    pairs = [
        PairResult("a", "b", 700, 300, 0),
        PairResult("a", "c", 900, 100, 0),
        PairResult("b", "c", 650, 350, 0),
    ]
    r = bradley_terry_ratings(players, pairs)
    print(r)
    assert r["a"] > r["b"] > r["c"], r
    assert abs(sum(r.values())) < 1e-6, r
    print("bradley_terry self-test OK")
