from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Dict, Any
import os
from pathlib import Path
from datetime import datetime, timezone
import csv

from .staking import Side, StakingResult  # type: ignore

# ================= Execution modes =================

class ExecMode(str, Enum):
    LIMIT_LTP = "LIMIT_LTP"   # limit at top-of-book (aggressive/passive resolved in strategy)
    SP_MOC    = "SP_MOC"      # Betfair SP: Market on Close
    SP_LOC    = "SP_LOC"      # Betfair SP: Limit on Close
    HYB       = "HYB"         # Hybrid: decide at runtime from policy file

class LimitStyle(str, Enum):
    AGGRESSIVE = "AGGRESSIVE"  # cross the spread to get filled now
    PASSIVE    = "PASSIVE"     # post at current best on our side


# ================= Context passed from executor =================

@dataclass
class RunnerCtx:
    market_id: str
    market_type: str  # "WIN" | "PLACE"
    selection_id: int
    course_id: str
    ltp: float
    milestone: Optional[int] = None
    secs_to_off: Optional[float] = None

    # Enriched fields for conditions/decisions
    trap: Optional[int] = None
    fav_rank_ltp: Optional[int] = None
    fav_rank_back: Optional[int] = None
    gor: Optional[float] = None          # gap odds ratio with next @ T-2s (WIN only; PLACE via cache)
    mom45: Optional[float] = None        # momentum (WIN) from 45s -> 2s on BASE_WIN
    mom45_place: Optional[float] = None  # momentum (PLACE) from 45s -> 2s on LTP
    # micro-momentum WIN
    d5: Optional[float] = None           # (LTP_now / LTP_5s_ago) - 1
    d30: Optional[float] = None          # (LTP_now / LTP_30s_ago) - 1
    vol60: Optional[float] = None        # std dev of returns over ~60s
    # Prix/context
    base_win: Optional[float] = None     # our price hierarchy value (for bounds)
    bb: Optional[float] = None           # best back price at tick
    bl: Optional[float] = None           # best lay price at tick

# Result of a fired slot (sizing already computed)
@dataclass
class FireResult:
    price: float
    size: float
    liability: Optional[float]
    reason: str
    exec_mode: ExecMode
    sp_limit: Optional[float] = None


# ================= Slot declaration =================

ConditionFn = Callable[[RunnerCtx], bool]

@dataclass
class Slot:
    family: str           # e.g. "LAY_WIN", "BACK_WIN", "BACK_PLACE", "LAY_PLACE"
    slot: int             # 1..10
    side: Side
    condition: ConditionFn
    exec_mode: ExecMode = ExecMode.LIMIT_LTP
    limit_style: LimitStyle = LimitStyle.AGGRESSIVE
    price_for_bounds: str = "BASE"       # "BASE" or "LTP"
    bet_per_market: bool = True          # fire at most once per market
    sp_limit: Optional[float] = None     # for SP_LOC if fixed
    tag: Optional[str] = None            # computed automatically if None

    # Staking params
    edge_env: Optional[str] = None       # env var for EDGE (e.g. EDGE_LAY_WIN_1)
    max_runner_stake_env: Optional[str] = None  # cap per runner env (e.g. MAX_RUNNER_STAKE_LAY_WIN_1)

    def __post_init__(self):
        if self.tag is None:
            self.tag = f"{self.family}_{self.slot}"


# ================= Helpers =================

def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.getenv(name, str(default)))
        return v
    except Exception:
        return default

def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name, "")
    if raw == "":
        return default
    return str(raw).strip().lower() in ("1","true","yes","y","on")

def _pick_bounds_price(ctx: RunnerCtx, pref: str) -> Optional[float]:
    if pref.upper() == "BASE":
        return ctx.base_win if ctx.base_win and ctx.base_win > 1.0 else None
    return ctx.ltp if ctx.ltp and ctx.ltp > 1.0 else None

def _choose_limit_price(side: Side, style: LimitStyle, ctx: RunnerCtx) -> Optional[float]:
    # For LIMIT_LTP, decide the actual limit price to send
    bb, bl = ctx.bb, ctx.bl
    if bb is None and bl is None:
        return None
    if style == LimitStyle.AGGRESSIVE:
        # cross the spread to get matched
        if side == Side.BACK:
            return bl or bb  # cross to best lay if available
        else:
            return bb or bl  # cross to best back if available
    else:
        # passive: post at our side's best
        if side == Side.BACK:
            return bb or bl
        else:
            return bl or bb

def _hyb_decide(ctx: RunnerCtx, slot: Slot) -> Dict[str, Any]:
    """Returns dict like {"mode": ExecMode, "limit_style": LimitStyle|None, "sp_limit": float|None, "limit_price": str|None}"""
    try:
        from .hybrid_policy import choose_action
        return choose_action(ctx, slot)
    except Exception:
        # Si la policy HYB plante, on bascule par défaut en BSP (sécurité)
        return {"mode": ExecMode.SP_MOC, "limit_style": slot.limit_style, "sp_limit": None}

def _compute_stake_safe(staking_engine, side: Side, price: float, edge: float, max_runner_cap: Optional[float]) -> StakingResult:
    # Call into staking engine if possible; otherwise do a safe fallback
    if hasattr(staking_engine, "quote"):
        return staking_engine.quote(side, price, edge, max_runner_cap=max_runner_cap)
    if hasattr(staking_engine, "compute"):
        return staking_engine.compute(side, price, edge, max_runner_cap=max_runner_cap)  # type: ignore
    if hasattr(staking_engine, "size_for"):
        r = staking_engine.size_for(side, price, edge, max_runner_cap=max_runner_cap)  # type: ignore
        if isinstance(r, StakingResult):
            return r

    # Fallback: basic proportional model (capital * edge) with sane floors/caps handled in staking engine absent
    capital = _env_float("CAPITAL", 1000.0)
    base = max(0.0, capital * max(0.0, edge))
    if side == Side.BACK:
        stake = base / max(1.01, price)
        return StakingResult(ok=True, price=price, size=round(stake, 2), liability=None, reason="fallback")
    else:
        liability = base
        stake = liability / max(0.01, price - 1.0)
        return StakingResult(ok=True, price=price, size=round(stake, 2), liability=round(liability, 2), reason="fallback")


# ================= Diagnostics (CSV per-day) =================

_DIAG_HEADERS = [
    "ts","tag","market_id","selection_id","market_type","milestone","secs_to_off",
    "trap","fav_rank_ltp","gor","base_price","bb","bl","mom45","d5","d30","vol60",
    "cond_pass","exec_mode","limit_price_choice","order_price","edge","max_runner_cap",
    "stake","liability","reason","note"
]

def _diag_enabled(slot_tag: str, ctx: RunnerCtx) -> bool:
    # Active par défaut pour LAY_WIN_1, à T-2s uniquement (évite le spam)
    env = os.getenv("DIAG_SLOTS", "LAY_WIN_1")
    tags = [t.strip() for t in env.split(",") if t.strip()]
    if slot_tag not in tags and "ALL" not in [t.upper() for t in tags]:
        return False
    return (ctx.milestone == 2)

def _diag_path(slot_tag: str) -> Path:
    d = Path("./data"); d.mkdir(parents=True, exist_ok=True)
    fname = f"diag_{slot_tag}_{datetime.now(timezone.utc):%Y%m%d}.csv"
    return d / fname

def _diag_write(slot_tag: str, row: Dict[str, Any]) -> None:
    path = _diag_path(slot_tag)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_DIAG_HEADERS)
        if new:
            w.writeheader()
        # cast simple types for csv
        row2 = {k: (v if (isinstance(v, (int,float,str)) or v is None) else str(v)) for k,v in row.items()}
        w.writerow(row2)


# ================= Public API =================

def try_fire_slot(staking_engine, slot: Slot, ctx: RunnerCtx) -> Optional[FireResult]:
    # Filter by family/market
    family = slot.family.upper()
    if "WIN" in family and ctx.market_type.upper() != "WIN":
        return None
    if "PLACE" in family and ctx.market_type.upper() != "PLACE":
        return None

    # Pre-diagnostics context
    do_diag = _diag_enabled(slot.tag or f"{slot.family}_{slot.slot}", ctx)

    # Evaluate condition with protection
    cond_pass = False
    cond_note = ""
    try:
        cond_pass = bool(slot.condition(ctx))
    except Exception as e:
        cond_pass = False
        cond_note = f"cond_error={e!r}"

    # If condition fails and we want diag, log why at T-2s
    if do_diag and not cond_pass:
        _diag_write(slot.tag, {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "tag": slot.tag,
            "market_id": ctx.market_id,
            "selection_id": ctx.selection_id,
            "market_type": ctx.market_type,
            "milestone": ctx.milestone,
            "secs_to_off": ctx.secs_to_off,
            "trap": ctx.trap,
            "fav_rank_ltp": ctx.fav_rank_ltp,
            "gor": ctx.gor,
            "base_price": _pick_bounds_price(ctx, slot.price_for_bounds),
            "bb": ctx.bb, "bl": ctx.bl,
            "mom45": ctx.mom45, "d5": ctx.d5, "d30": ctx.d30, "vol60": ctx.vol60,
            "cond_pass": False,
            "exec_mode": None,
            "limit_price_choice": None,
            "order_price": None,
            "edge": None,
            "max_runner_cap": None,
            "stake": None, "liability": None, "reason": None,
            "note": cond_note or "condition_false"
        })
        return None

    if not cond_pass:
        return None

    # Decide effective exec mode
    mode = slot.exec_mode
    limit_style = slot.limit_style
    sp_limit = slot.sp_limit
    decision: Dict[str, Any] = {}

    if slot.exec_mode == ExecMode.HYB:
        decision = _hyb_decide(ctx, slot)
        mode = decision.get("mode", mode)
        limit_style = decision.get("limit_style", limit_style)
        sp_limit = decision.get("sp_limit", sp_limit)

    # Price to place (for LIMIT) and price for stake bounds
    bounds_price = _pick_bounds_price(ctx, slot.price_for_bounds)
    if bounds_price is None:
        if do_diag:
            _diag_write(slot.tag, {
                "ts": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                "tag": slot.tag, "market_id": ctx.market_id, "selection_id": ctx.selection_id,
                "market_type": ctx.market_type, "milestone": ctx.milestone, "secs_to_off": ctx.secs_to_off,
                "trap": ctx.trap, "fav_rank_ltp": ctx.fav_rank_ltp, "gor": ctx.gor,
                "base_price": None, "bb": ctx.bb, "bl": ctx.bl,
                "mom45": ctx.mom45, "d5": ctx.d5, "d30": ctx.d30, "vol60": ctx.vol60,
                "cond_pass": True,
                "exec_mode": mode, "limit_price_choice": decision.get("limit_price"),
                "order_price": None, "edge": None, "max_runner_cap": None,
                "stake": None, "liability": None, "reason": "no_bounds_price", "note": ""
            })
        return None

    # Choose the actual order price
    if mode == ExecMode.LIMIT_LTP:
        # Priority: explicit limit price from hybrid policy ("CROSS"|"MID"|"OWN")
        order_price: Optional[float] = None
        lp_key = str(decision.get("limit_price", "")).upper() if decision else ""
        if lp_key == "CROSS":
            # BACK at best LAY; LAY at best BACK
            order_price = (ctx.bl if slot.side == Side.BACK else ctx.bb)
        elif lp_key == "OWN":
            # BACK at best BACK; LAY at best LAY (post our side)
            order_price = (ctx.bb if slot.side == Side.BACK else ctx.bl)
        elif lp_key == "MID":
            if ctx.bb is not None and ctx.bl is not None:
                order_price = (ctx.bb + ctx.bl) / 2.0

        # Fallback to style-based logic
        if order_price is None:
            order_price = _choose_limit_price(slot.side, limit_style, ctx)

        # --- HYB fallback: if still None, optionally switch to SP_MOC (BSP) ---
        if order_price is None:
            if _env_bool("HYB_FALLBACK_TO_SP_MOC", True):
                # Use a sane price reference just for stake sizing
                price_for_sizing = _choose_limit_price(slot.side, limit_style, ctx) or bounds_price
                if price_for_sizing is None:
                    if do_diag:
                        _diag_write(slot.tag, {
                            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                            "tag": slot.tag, "market_id": ctx.market_id, "selection_id": ctx.selection_id,
                            "market_type": ctx.market_type, "milestone": ctx.milestone, "secs_to_off": ctx.secs_to_off,
                            "trap": ctx.trap, "fav_rank_ltp": ctx.fav_rank_ltp, "gor": ctx.gor,
                            "base_price": bounds_price, "bb": ctx.bb, "bl": ctx.bl,
                            "mom45": ctx.mom45, "d5": ctx.d5, "d30": ctx.d30, "vol60": ctx.vol60,
                            "cond_pass": True,
                            "exec_mode": mode, "limit_price_choice": lp_key or limit_style.value,
                            "order_price": None, "edge": None, "max_runner_cap": None,
                            "stake": None, "liability": None, "reason": "no_order_price", "note": "fallback_sp_moc_but_no_price_ref"
                        })
                    return None
                # compute sizing for SP_MOC
                edge_env = slot.edge_env or f"EDGE_{slot.family}_{slot.slot}"
                edge = _env_float(edge_env, 0.02)
                max_cap_env = slot.max_runner_stake_env or f"MAX_RUNNER_STAKE_{slot.family}_{slot.slot}"
                max_runner_cap = os.getenv(max_cap_env)
                max_runner_cap = float(max_runner_cap) if max_runner_cap not in (None, "") else None

                sr: StakingResult = _compute_stake_safe(staking_engine, slot.side, float(price_for_sizing), edge, max_runner_cap)
                if not getattr(sr, "ok", False):
                    if do_diag:
                        _diag_write(slot.tag, {
                            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                            "tag": slot.tag, "market_id": ctx.market_id, "selection_id": ctx.selection_id,
                            "market_type": ctx.market_type, "milestone": ctx.milestone, "secs_to_off": ctx.secs_to_off,
                            "trap": ctx.trap, "fav_rank_ltp": ctx.fav_rank_ltp, "gor": ctx.gor,
                            "base_price": bounds_price, "bb": ctx.bb, "bl": ctx.bl,
                            "mom45": ctx.mom45, "d5": ctx.d5, "d30": ctx.d30, "vol60": ctx.vol60,
                            "cond_pass": True,
                            "exec_mode": "SP_MOC", "limit_price_choice": lp_key or limit_style.value,
                            "order_price": None, "edge": edge, "max_runner_cap": max_runner_cap,
                            "stake": None, "liability": None, "reason": getattr(sr, "reason", "stake_not_ok"), "note": "fallback_sp_moc"
                        })
                    return None

                if do_diag:
                    _diag_write(slot.tag, {
                        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                        "tag": slot.tag, "market_id": ctx.market_id, "selection_id": ctx.selection_id,
                        "market_type": ctx.market_type, "milestone": ctx.milestone, "secs_to_off": ctx.secs_to_off,
                        "trap": ctx.trap, "fav_rank_ltp": ctx.fav_rank_ltp, "gor": ctx.gor,
                        "base_price": bounds_price, "bb": ctx.bb, "bl": ctx.bl,
                        "mom45": ctx.mom45, "d5": ctx.d5, "d30": ctx.d30, "vol60": ctx.vol60,
                        "cond_pass": True,
                        "exec_mode": "SP_MOC", "limit_price_choice": lp_key or limit_style.value,
                        "order_price": None, "edge": edge, "max_runner_cap": max_runner_cap,
                        "stake": getattr(sr, "size", None), "liability": getattr(sr, "liability", None),
                        "reason": getattr(sr, "reason", "ok"), "note": "fallback_sp_moc"
                    })

                return FireResult(price=float(getattr(sr, "price", price_for_sizing)),
                                  size=float(getattr(sr, "size", 0.0)),
                                  liability=getattr(sr, "liability", None),
                                  reason="fallback_sp_moc",
                                  exec_mode=ExecMode.SP_MOC,
                                  sp_limit=None)
            # Fallback disabled: log & stop
            if do_diag:
                _diag_write(slot.tag, {
                    "ts": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                    "tag": slot.tag, "market_id": ctx.market_id, "selection_id": ctx.selection_id,
                    "market_type": ctx.market_type, "milestone": ctx.milestone, "secs_to_off": ctx.secs_to_off,
                    "trap": ctx.trap, "fav_rank_ltp": ctx.fav_rank_ltp, "gor": ctx.gor,
                    "base_price": bounds_price, "bb": ctx.bb, "bl": ctx.bl,
                    "mom45": ctx.mom45, "d5": ctx.d5, "d30": ctx.d30, "vol60": ctx.vol60,
                    "cond_pass": True,
                    "exec_mode": mode, "limit_price_choice": lp_key or limit_style.value,
                    "order_price": None, "edge": None, "max_runner_cap": None,
                    "stake": None, "liability": None, "reason": "no_order_price", "note": ""
                })
            return None
    else:
        # For SP_* we still need a price reference for stake sizing
        order_price = _choose_limit_price(slot.side, limit_style, ctx) or bounds_price

    # Edge & caps (normal path)
    edge_env = slot.edge_env or f"EDGE_{slot.family}_{slot.slot}"
    edge = _env_float(edge_env, 0.02)
    max_cap_env = slot.max_runner_stake_env or f"MAX_RUNNER_STAKE_{slot.family}_{slot.slot}"
    max_runner_cap = os.getenv(max_cap_env)
    max_runner_cap = float(max_runner_cap) if max_runner_cap not in (None, "") else None

    # Compute stake
    sr: StakingResult = _compute_stake_safe(staking_engine, slot.side, float(order_price), edge, max_runner_cap)

    if not getattr(sr, "ok", False):
        if do_diag:
            _diag_write(slot.tag, {
                "ts": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                "tag": slot.tag, "market_id": ctx.market_id, "selection_id": ctx.selection_id,
                "market_type": ctx.market_type, "milestone": ctx.milestone, "secs_to_off": ctx.secs_to_off,
                "trap": ctx.trap, "fav_rank_ltp": ctx.fav_rank_ltp, "gor": ctx.gor,
                "base_price": bounds_price, "bb": ctx.bb, "bl": ctx.bl,
                "mom45": ctx.mom45, "d5": ctx.d5, "d30": ctx.d30, "vol60": ctx.vol60,
                "cond_pass": True,
                "exec_mode": mode, "limit_price_choice": decision.get("limit_price"),
                "order_price": order_price, "edge": edge, "max_runner_cap": max_runner_cap,
                "stake": None, "liability": None, "reason": getattr(sr, "reason", "stake_not_ok"), "note": ""
            })
        return None

    # Success path
    if do_diag:
        _diag_write(slot.tag, {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "tag": slot.tag, "market_id": ctx.market_id, "selection_id": ctx.selection_id,
            "market_type": ctx.market_type, "milestone": ctx.milestone, "secs_to_off": ctx.secs_to_off,
            "trap": ctx.trap, "fav_rank_ltp": ctx.fav_rank_ltp, "gor": ctx.gor,
            "base_price": bounds_price, "bb": ctx.bb, "bl": ctx.bl,
            "mom45": ctx.mom45, "d5": ctx.d5, "d30": ctx.d30, "vol60": ctx.vol60,
            "cond_pass": True,
            "exec_mode": mode, "limit_price_choice": decision.get("limit_price"),
            "order_price": getattr(sr, "price", order_price), "edge": edge, "max_runner_cap": max_runner_cap,
            "stake": getattr(sr, "size", None), "liability": getattr(sr, "liability", None),
            "reason": getattr(sr, "reason", "ok"), "note": "ok"
        })

    reason = f"cond_ok edge={edge} bounds_price={round(bounds_price,2)}"
    return FireResult(price=float(getattr(sr, "price", order_price)),
                      size=float(getattr(sr, "size", 0.0)),
                      liability=getattr(sr, "liability", None),
                      reason=reason,
                      exec_mode=mode,
                      sp_limit=sp_limit)


# ================= Registry (declare your systems here) =================

def build_registry() -> List[Slot]:
    slots: List[Slot] = []

    # --- System 1: LAY WIN — Trap=8, price(BASE) in [1.5, 50], fav-rank/LTP & GOR conditions @ T-2s ---
    def cond_lay_win_1(ctx: RunnerCtx) -> bool:
        if ctx.market_type.upper() != "WIN":
            return False
        if ctx.milestone != 2:
            return False
        if ctx.trap != 8:
            return False
        bounds_price = _pick_bounds_price(ctx, "BASE")
        if bounds_price is None or not (1.5 <= bounds_price <= 50.0):
            return False
        r = ctx.fav_rank_ltp
        if r is None:
            return False
        if r not in (1,5,8):
            return True
        if r == 5 and (ctx.gor is not None) and (ctx.gor <= 1.7):
            return True
        if r == 1 and (ctx.gor is not None) and (ctx.gor >= 1.1):
            return True
        return False

    slots.append(Slot(
        family="LAY_WIN",
        slot=1,
        side=Side.LAY,
        condition=cond_lay_win_1,
        exec_mode=ExecMode.HYB,
        limit_style=LimitStyle.AGGRESSIVE,
        price_for_bounds="BASE",
        bet_per_market=True,
        edge_env="EDGE_LAY_WIN_1",
        max_runner_stake_env="MAX_RUNNER_STAKE_LAY_WIN_1",
    ))

    # --- System 2: LAY PLACE — Trap=8, LTP_PLACE in [3, 40], (rank != 5) or (rank == 5 and GOR < 1.75) @ T-2s ---
    def cond_lay_place_1(ctx: RunnerCtx) -> bool:
        if ctx.market_type.upper() != "PLACE":
            return False
        if ctx.milestone != 2:
            return False
        if ctx.trap != 8:
            return False
        # Bornes sur LTP PLACE (demande utilisateur)
        bounds_price = _pick_bounds_price(ctx, "LTP")
        if bounds_price is None or not (3.0 <= bounds_price <= 40.0):
            return False
        # Fav rank par LTP (réutilisé depuis WIN via cache côté executor)
        r = ctx.fav_rank_ltp
        if r is None:
            return False
        if r != 5:
            return True
        # r == 5 -> contrainte GOR
        return (ctx.gor is not None) and (ctx.gor < 1.75)

    slots.append(Slot(
        family="LAY_PLACE",
        slot=1,
        side=Side.LAY,
        condition=cond_lay_place_1,
        exec_mode=ExecMode.HYB,              # HYB + fallback SP_MOC si LIMIT impossible
        limit_style=LimitStyle.AGGRESSIVE,
        price_for_bounds="LTP",              # bornes sur LTP_PLACE
        bet_per_market=True,
        edge_env="EDGE_LAY_PLACE_1",
        max_runner_stake_env="MAX_RUNNER_STAKE_LAY_PLACE_1",
    ))

    # You can append more slots here…

    return slots
