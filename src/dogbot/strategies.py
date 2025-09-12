from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

# Registre de stratégies par slots (aligne avec executor.py)
# 4 familles × 10 slots : BACK_WIN / BACK_PLACE / LAY_WIN / LAY_PLACE

from .staking import StakingEngine, StakingResult, Side  # types côté staking

# --------- Contexte runner passé aux conditions ---------

@dataclass
class RunnerCtx:
    market_id: str
    market_type: str    # "WIN" | "PLACE"
    selection_id: int
    course_id: str
    ltp: float          # LTP instantané
    milestone: Optional[int] = None       # ex: 300,150,80,45,2
    secs_to_off: Optional[float] = None   # T- en secondes (approx)

ConditionFn = Callable[[RunnerCtx], bool]

@dataclass
class StrategySlot:
    family: str                 # ex "BACK_WIN"
    slot: int                   # 1..10
    side: Side                  # Side.BACK | Side.LAY
    condition: ConditionFn
    tag: str                    # pour le logging / CSV trades
    # -------- options nouvelles --------
    bet_per_market: bool = False                     # True = une seule fois par marché pour ce slot
    allowed_milestones: Optional[Set[int]] = None    # ex {150,45,2} si tu veux restreindre

# --------- Placeholders (à remplacer par tes vraies règles) ---------

def _false(_: RunnerCtx) -> bool:
    return False

# EXEMPLE de condition : BACK_WIN_1, joue si 1.8 <= LTP <= 4.5
def cond_back_win_1(ctx: RunnerCtx) -> bool:
    return 1.8 <= ctx.ltp <= 4.5

# --------- Construction du registre ---------

def build_registry() -> List[StrategySlot]:
    reg: List[StrategySlot] = []

    # BACK WIN 1..10
    for i in range(1, 11):
        reg.append(StrategySlot(
            family="BACK_WIN",
            slot=i,
            side=Side.BACK,
            condition=cond_back_win_1 if i == 1 else _false,
            tag=f"BW_{i}",
            # EXEMPLE: limiter BW_1 aux jalons 150/45/2 et un pari max par marché
            bet_per_market=True if i == 1 else False,
            allowed_milestones={150, 45, 2} if i == 1 else None,
        ))

    # BACK PLACE 1..10
    for i in range(1, 11):
        reg.append(StrategySlot(
            family="BACK_PLACE",
            slot=i,
            side=Side.BACK,
            condition=_false,
            tag=f"BP_{i}",
            # bet_per_market / allowed_milestones désactivés par défaut
        ))

    # LAY WIN 1..10
    for i in range(1, 11):
        reg.append(StrategySlot(
            family="LAY_WIN",
            slot=i,
            side=Side.LAY,
            condition=_false,
            tag=f"LW_{i}",
        ))

    # LAY PLACE 1..10
    for i in range(1, 11):
        reg.append(StrategySlot(
            family="LAY_PLACE",
            slot=i,
            side=Side.LAY,
            condition=_false,
            tag=f"LP_{i}",
        ))

    return reg

# --------- Déclencheur d'un slot ---------

def try_fire_slot(engine: StakingEngine, slot: StrategySlot, ctx: RunnerCtx) -> Optional[StakingResult]:
    # Cohérence marché/famille
    if "WIN" in slot.family and ctx.market_type != "WIN":
        return None
    if "PLACE" in slot.family and ctx.market_type != "PLACE":
        return None

    # Filtre jalons si demandé
    if slot.allowed_milestones is not None:
        if ctx.milestone not in slot.allowed_milestones:
            return None

    # Condition de slot
    if not slot.condition(ctx):
        return None

    # Calcul de mise via StakingEngine (CAPITAL/LTP/EDGE)
    return engine.compute(
        side=slot.side,
        price_ltp=ctx.ltp,
        family=slot.family,
        slot=slot.slot,
        market_id=ctx.market_id,
        selection_id=ctx.selection_id,
        course_id=ctx.course_id,
        strategy_tag=slot.tag,
    )
