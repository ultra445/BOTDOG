from __future__ import annotations
import os
from typing import List, Dict, Any, Optional, Type
from betfairlightweight import filters
from .strategies import StrategyBase

def _safe_lpt(r):
    v = getattr(r, "last_price_traded", None)
    return v if (v is not None and v > 0) else 999.0

def _env_bool(name: str, default: bool=False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1","true","yes","on")

def _env_float_opt(name: str) -> Optional[float]:
    v = os.getenv(name)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None

# ---------- BACK WIN ----------
class BackWinStrategy1(StrategyBase):
    name = "BACK_WIN_1"; required_market_type = "WIN"; side = "BACK"
    def __init__(self, unit_stake: float): self.unit_stake = unit_stake
    def decide(self, market_book, features, market_index_entry) -> List[Dict[str, Any]]:
        # Démo : back le favori si LTP < 3.0
        runners = [r for r in (market_book.runners or []) if getattr(r, "status", "ACTIVE") == "ACTIVE"]
        if not runners:
            return []
        fav = min(runners, key=_safe_lpt)
        lpt = _safe_lpt(fav)
        if lpt >= 3.0:
            return []
        price = max(1.01, round(lpt - 0.1, 2))
        return [filters.place_instruction(
            order_type="LIMIT",
            selection_id=fav.selection_id,
            side="BACK",
            limit_order=filters.limit_order(price=price, size=self.unit_stake, persistence_type="LAPSE"),
        )]

class BackWinStrategy2(StrategyBase):
    name = "BACK_WIN_2"; required_market_type = "WIN"; side = "BACK"
    def __init__(self, unit_stake: float): self.unit_stake = unit_stake
    def decide(self, market_book, features, market_index_entry) -> List[Dict[str, Any]]: return []

class BackWinStrategy3(BackWinStrategy2): name = "BACK_WIN_3"
class BackWinStrategy4(BackWinStrategy2): name = "BACK_WIN_4"
class BackWinStrategy5(BackWinStrategy2): name = "BACK_WIN_5"
class BackWinStrategy6(BackWinStrategy2): name = "BACK_WIN_6"
class BackWinStrategy7(BackWinStrategy2): name = "BACK_WIN_7"
class BackWinStrategy8(BackWinStrategy2): name = "BACK_WIN_8"
class BackWinStrategy9(BackWinStrategy2): name = "BACK_WIN_9"
class BackWinStrategy10(BackWinStrategy2): name = "BACK_WIN_10"

# ---------- LAY WIN ----------
class LayWinStrategy1(StrategyBase):
    name = "LAY_WIN_1"; required_market_type = "WIN"; side = "LAY"
    def __init__(self, unit_stake: float): self.unit_stake = unit_stake
    def decide(self, market_book, features, market_index_entry) -> List[Dict[str, Any]]: return []
class LayWinStrategy2(LayWinStrategy1): name = "LAY_WIN_2"
class LayWinStrategy3(LayWinStrategy1): name = "LAY_WIN_3"
class LayWinStrategy4(LayWinStrategy1): name = "LAY_WIN_4"
class LayWinStrategy5(LayWinStrategy1): name = "LAY_WIN_5"
class LayWinStrategy6(LayWinStrategy1): name = "LAY_WIN_6"
class LayWinStrategy7(LayWinStrategy1): name = "LAY_WIN_7"
class LayWinStrategy8(LayWinStrategy1): name = "LAY_WIN_8"
class LayWinStrategy9(LayWinStrategy1): name = "LAY_WIN_9"
class LayWinStrategy10(LayWinStrategy1): name = "LAY_WIN_10"

# ---------- BACK PLACE ----------
class BackPlaceStrategy1(StrategyBase):
    name = "BACK_PLACE_1"; required_market_type = "PLACE"; side = "BACK"
    def __init__(self, unit_stake: float): self.unit_stake = unit_stake
    def decide(self, market_book, features, market_index_entry) -> List[Dict[str, Any]]: return []
class BackPlaceStrategy2(BackPlaceStrategy1): name = "BACK_PLACE_2"
class BackPlaceStrategy3(BackPlaceStrategy1): name = "BACK_PLACE_3"
class BackPlaceStrategy4(BackPlaceStrategy1): name = "BACK_PLACE_4"
class BackPlaceStrategy5(BackPlaceStrategy1): name = "BACK_PLACE_5"
class BackPlaceStrategy6(BackPlaceStrategy1): name = "BACK_PLACE_6"
class BackPlaceStrategy7(BackPlaceStrategy1): name = "BACK_PLACE_7"
class BackPlaceStrategy8(BackPlaceStrategy1): name = "BACK_PLACE_8"
class BackPlaceStrategy9(BackPlaceStrategy1): name = "BACK_PLACE_9"
class BackPlaceStrategy10(BackPlaceStrategy1): name = "BACK_PLACE_10"

# ---------- LAY PLACE ----------
class LayPlaceStrategy1(StrategyBase):
    name = "LAY_PLACE_1"; required_market_type = "PLACE"; side = "LAY"
    def __init__(self, unit_stake: float): self.unit_stake = unit_stake
    def decide(self, market_book, features, market_index_entry) -> List[Dict[str, Any]]: return []
class LayPlaceStrategy2(LayPlaceStrategy1): name = "LAY_PLACE_2"
class LayPlaceStrategy3(LayPlaceStrategy1): name = "LAY_PLACE_3"
class LayPlaceStrategy4(LayPlaceStrategy1): name = "LAY_PLACE_4"
class LayPlaceStrategy5(LayPlaceStrategy1): name = "LAY_PLACE_5"
class LayPlaceStrategy6(LayPlaceStrategy1): name = "LAY_PLACE_6"
class LayPlaceStrategy7(LayPlaceStrategy1): name = "LAY_PLACE_7"
class LayPlaceStrategy8(LayPlaceStrategy1): name = "LAY_PLACE_8"
class LayPlaceStrategy9(LayPlaceStrategy1): name = "LAY_PLACE_9"
class LayPlaceStrategy10(LayPlaceStrategy1): name = "LAY_PLACE_10"

# ---------- Loader (inchangé) ----------
_REGISTRY = {
    "BACK_WIN": {i: globals()[f"BackWinStrategy{i}"] for i in range(1, 11)},
    "LAY_WIN":  {i: globals()[f"LayWinStrategy{i}"]  for i in range(1, 11)},
    "BACK_PLACE": {i: globals()[f"BackPlaceStrategy{i}"] for i in range(1, 11)},
    "LAY_PLACE":  {i: globals()[f"LayPlaceStrategy{i}"]  for i in range(1, 11)},
}

def _default_stake(max_market_stake: float) -> float:
    return min(max_market_stake, 2.0)

def _env_float_opt(name: str) -> Optional[float]:
    v = os.getenv(name)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None

def _enabled_cat(cat: str) -> bool:
    return _env_bool(f"ENABLE_{cat}", True)

def _enabled_slot(cat: str, idx: int) -> bool:
    return _env_bool(f"ENABLE_{cat}_{idx}", False)

def _resolve_stake_per_slot(cat: str, idx: int, max_market_stake: float) -> float:
    v = _env_float_opt(f"STAKE_{cat}_{idx}")
    return v if v is not None else _default_stake(max_market_stake)

def load_strategies_from_env(max_market_stake: float, per_cat: int = 10):
    strategies: List[StrategyBase] = []
    for cat in ("BACK_WIN", "LAY_WIN", "BACK_PLACE", "LAY_PLACE"):
        if not _enabled_cat(cat):
            continue
        for i in range(1, per_cat+1):
            if not _enabled_slot(cat, i):
                continue
            cls = _REGISTRY[cat].get(i)
            if not cls:
                continue
            stake = _resolve_stake_per_slot(cat, i, max_market_stake)
            strategies.append(cls(unit_stake=stake))
    return strategies
