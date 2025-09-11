"""
win_to_place_plackett_luce_v2.py

WIN -> PLACE probability conversion under a Plackett–Luce (Harville-like) race model.
Exact formulas for K in {1, 2, 3}.
"""

from typing import Sequence
import numpy as np


def odds_to_win_probs(odds: Sequence[float], beta: float = 1.0) -> np.ndarray:
    odds = np.asarray(odds, dtype=float)
    if np.any(odds <= 1.0):
        raise ValueError("All odds must be > 1.0 (decimal odds).")
    u = 1.0 / odds
    if beta != 1.0:
        u = np.power(u, beta)
    total = np.sum(u)
    if total <= 0:
        raise ValueError("Invalid odds leading to zero total probability.")
    return u / total


def _place_prob_top2(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Input probabilities must sum to 1.")
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("Probabilities must be in (0,1).")
    t = p / (1.0 - p)
    T = np.sum(t)
    q = p * (1.0 + (T - t))
    return q


def _place_prob_top3(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Input probabilities must sum to 1.")
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("Probabilities must be in (0,1).")

    q = _place_prob_top2(p).astype(float)

    N = p.shape[0]
    for i in range(N):
        add_third = 0.0
        for j in range(N):
            if j == i:
                continue
            for k in range(j + 1, N):
                if k == i:
                    continue
                pj, pk = p[j], p[k]
                denom12 = (1.0 - pj - pk)
                if denom12 <= 0:
                    continue
                term_orders = pj * pk * (1.0 / (1.0 - pj) + 1.0 / (1.0 - pk))
                add_third += term_orders * (p[i] / denom12)
        q[i] += add_third

    return q


def place_probabilities(p: Sequence[float], K: int) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if K not in (1, 2, 3):
        raise ValueError("Only K in {1,2,3} is supported by this exact implementation.")
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Input probabilities must sum to 1.")
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("Probabilities must be in (0,1).")

    if K == 1:
        return p.copy()
    elif K == 2:
        return _place_prob_top2(p)
    else:  # K == 3
        return _place_prob_top3(p)


def fair_place_odds(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if np.any((q <= 0) | (q >= 1)):
        raise ValueError("q must be in (0,1).")
    return 1.0 / q


def ev_back_place(offered_odds: Sequence[float], q: Sequence[float], commission: float = 0.05) -> np.ndarray:
    O = np.asarray(offered_odds, dtype=float)
    q = np.asarray(q, dtype=float)
    c = float(commission)
    if np.any(O <= 1.0):
        raise ValueError("Offered odds must be > 1.0 (decimal odds).")
    if np.any((q < 0) | (q > 1)):
        raise ValueError("q must be in [0,1].")
    return q * (O - 1.0) * (1.0 - c) - (1.0 - q)


def min_odds_for_positive_ev(q: Sequence[float], commission: float = 0.05) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    c = float(commission)
    if np.any((q <= 0) | (q >= 1)):
        raise ValueError("q must be in (0,1).")
    return (1.0 - c * q) / ((1.0 - c) * q)
