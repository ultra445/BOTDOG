from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set
from enum import Enum

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

# --------- Mode d'exécution par slot (prix LTP vs BSP) ---------

class ExecMode(str, Enum):
    LIMIT_LTP = "LIMIT_LTP"   # Ordre LIMIT au prix de marché (LTP arrondi tick)
    SP_MOC    = "SP_MOC"      # Betfair SP sans limite (MARKET_ON_CLOSE)
    SP_LOC    = "SP_LOC"      # Betfair SP avec limite (LIMIT_ON_CLOSE)

@dataclass
class StrategySlot:
    family: str                 # ex "BACK_WIN"
    slot: int                   # 1..10
    side: Side                  # Side.BACK | Side.LAY
    condition: ConditionFn
    tag: str                    # pour le logging / CSV trades
    # -------- options d'éxécution / sécurité --------
    bet_per_market: bool = False                     # True = une seule fois par marché pour ce slot
    allowed_milestones: Optional[Set[int]] = None    # ex {150,45,2} si tu veux restreindre
    exec_mode: ExecMode = ExecMode.LIMIT_LTP         # LTP par défaut (rien ne change si tu ne touches pas)
    sp_limit: Optional[float] = None                 # pour SP_LOC : BACK=min SP ; LAY=max SP

# --------- Placeholders (à remplacer par tes vraies règles) ---------

def _false(_: RunnerCtx) -> bool:
    return False

# EXEMPLE de condition : BACK_WIN_1, joue si 1.8 <= LTP <= 4.5
def cond_back_win_1(ctx: RunnerCtx) -> bool:
    return 1.8 <= ctx.ltp <= 4.5

# (exemples supplémentaires que tu peux activer si tu veux des slots BSP)
def cond_bw_bsp(ctx: RunnerCtx) -> bool:
    # exemple: autoriser un slot BSP si LTP raisonnable
    return 2.0 <= ctx.ltp <= 6.0

def cond_bw_bsp_min26(ctx: RunnerCtx) -> bool:
    # exemple: jouer BSP avec limite min 2.6 (pour BACK)
    return ctx.ltp >= 2.2

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
            bet_per_market=True if i == 1 else False,
            allowed_milestones={150, 45, 2} if i == 1 else None,
            exec_mode=ExecMode.LIMIT_LTP,   # par défaut: prix de marché (LTP)
            # sp_limit=None,                # (utilisé seulement si exec_mode=SP_LOC)
        ))

    # BACK PLACE 1..10
    for i in range(1, 11):
        reg.append(StrategySlot(
            family="BACK_PLACE",
            slot=i,
            side=Side.BACK,
            condition=_false,
            tag=f"BP_{i}",
            exec_mode=ExecMode.LIMIT_LTP,
        ))

    # LAY WIN 1..10
    for i in range(1, 11):
        reg.append(StrategySlot(
            family="LAY_WIN",
            slot=i,
            side=Side.LAY,
            condition=_false,
            tag=f"LW_{i}",
            exec_mode=ExecMode.LIMIT_LTP,
        ))

    # LAY PLACE 1..10
    for i in range(1, 11):
        reg.append(StrategySlot(
            family="LAY_PLACE",
            slot=i,
            side=Side.LAY,
            condition=_false,
            tag=f"LP_{i}",
            exec_mode=ExecMode.LIMIT_LTP,
        ))

    # ------------------------
    # EXEMPLES (désactivés par défaut) ─ à activer si tu veux tester le BSP :
    # ------------------------
    # # BACK WIN, jouer AU BSP sans limite (MARKET_ON_CLOSE)
    # reg.append(StrategySlot(
    #     family="BACK_WIN",
    #     slot=2,
    #     side=Side.BACK,
    #     condition=cond_bw_bsp,     # ta règle
    #     tag="BW_2_BSP",
    #     bet_per_market=True,
    #     exec_mode=ExecMode.SP_MOC, # BSP sans limite
    # ))
    #
    # # BACK WIN, jouer AU BSP AVEC limite min 2.6 (LIMIT_ON_CLOSE)
    # reg.append(StrategySlot(
    #     family="BACK_WIN",
    #     slot=3,
    #     side=Side.BACK,
    #     condition=cond_bw_bsp_min26,
    #     tag="BW_3_BSP_MIN",
    #     bet_per_market=True,
    #     exec_mode=ExecMode.SP_LOC, # BSP avec limite
    #     sp_limit=2.6,              # BACK=min SP ; LAY=max SP
    # ))

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
