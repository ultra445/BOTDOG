from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import os

from betfairlightweight import filters
from .formulas import compute_stake


# ---------- Base ----------

@dataclass
class StrategyBase:
    name: str
    side: str            # "BACK" ou "LAY"
    market_type: str     # "WIN" ou "PLACE"
    unit_stake: float = 2.0

    def decide(
        self,
        market_book,
        market_index_entry: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retourne une place_instruction (dict) OU None."""
        return None


# ---------- Démo concrète (BACK WIN slot 1) ----------

class BackWinDemo(StrategyBase):
    """BACK WIN: back le favori si LTP < 3.0 (démo)."""

    def decide(self, market_book, market_index_entry=None):
        runners = getattr(market_book, "runners", []) or []
        if not runners:
            return None

        def lpt(r):
            v = getattr(r, "last_price_traded", None)
            return v if v is not None else 999.0

        fav = min(runners, key=lpt)
        last = lpt(fav)
        if last >= 3.0:
            return None

        # Mise dynamique via .env (STAKE_FORMULA_BACK_WIN) avec repli sur STAKE_BACK_WIN puis unit_stake
        stake = compute_stake(side="BACK", market_type="WIN", odds=last, default_unit=self.unit_stake)
        if stake <= 0:
            return None

        # Prix limite simple (à ajuster selon ta logique de prise de prix)
        price = max(1.01, round(last - 0.10, 2))

        return filters.place_instruction(
            order_type="LIMIT",
            selection_id=fav.selection_id,
            side="BACK",
            limit_order=filters.limit_order(
                price=price,
                size=stake,
                persistence_type="LAPSE",
            ),
        )


# ---------- Stratégie vide (slots à remplir plus tard) ----------

class NoOp(StrategyBase):
    """Placeholder : ne place rien pour l'instant."""
    def decide(self, market_book, market_index_entry=None):
        return None


# ---------- Manager ----------

class StrategyManager:
    def __init__(self, strategies: List[StrategyBase]):
        self.strategies = strategies

    def decide_all(
        self,
        market_book,
        market_index_entry: Optional[Dict[str, Any]],
    ) -> List[Tuple[StrategyBase, Dict[str, Any]]]:
        """Renvoie [(strategy, instruction_dict), ...] limité au type de marché courant."""
        mtype = None
        if market_index_entry:
            mtype = (market_index_entry.get("market_type") or "").upper() or None
        if not mtype:
            mtype = "WIN"

        out: List[Tuple[StrategyBase, Dict[str, Any]]] = []
        for s in self.strategies:
            if s.market_type.upper() != mtype:
                continue
            instr = s.decide(market_book, market_index_entry)
            if instr:
                out.append((s, instr))
        return out


# ---------- Helpers lecture .env ----------

def _flag(key: str, default: bool = False) -> bool:
    val = os.getenv(key, "true" if default else "false").strip().lower()
    return val in ("1", "true", "yes", "on")

def _stake(key: str, fallback: float) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        return fallback
    try:
        return float(raw)
    except Exception:
        return fallback


# ---------- Construction depuis .env ----------

def _build_slots(enable_cat: bool, cat_prefix: str, market_type: str, side: str, unit: float) -> List[StrategyBase]:
    """
    Crée jusqu’à 10 slots par catégorie.
    - Slot 1 de BACK_WIN utilise la démo concrète (BackWinDemo).
    - Tous les autres slots sont des NoOp, prêts à être codés plus tard.
    - Le flag ENABLE_{cat_prefix}_{i} pilote chaque slot.
    """
    strategies: List[StrategyBase] = []
    if not enable_cat:
        return strategies

    for i in range(1, 11):
        if _flag(f"ENABLE_{cat_prefix}_{i}", i == 1):
            name = f"{cat_prefix}_{i}"
            if cat_prefix == "BACK_WIN" and i == 1:
                strategies.append(BackWinDemo(name=name, side=side, market_type=market_type, unit_stake=unit))
            else:
                strategies.append(NoOp(name=name, side=side, market_type=market_type, unit_stake=unit))
    return strategies


def build_strategies_from_env(default_unit: float = 2.0) -> StrategyManager:
    # Master switches (catégories)
    enable_back_win   = _flag("ENABLE_BACK_WIN",   True)
    enable_lay_win    = _flag("ENABLE_LAY_WIN",    True)
    enable_back_place = _flag("ENABLE_BACK_PLACE", True)
    enable_lay_place  = _flag("ENABLE_LAY_PLACE",  True)

    # Stakes fixes par catégorie (seulement comme *repli* si pas de STAKE_FORMULA_*)
    stake_back_win   = _stake("STAKE_BACK_WIN",   default_unit)
    stake_lay_win    = _stake("STAKE_LAY_WIN",    default_unit)
    stake_back_place = _stake("STAKE_BACK_PLACE", default_unit)
    stake_lay_place  = _stake("STAKE_LAY_PLACE",  default_unit)

    strategies: List[StrategyBase] = []
    strategies += _build_slots(enable_back_win,   "BACK_WIN",   "WIN",   "BACK", stake_back_win)
    strategies += _build_slots(enable_lay_win,    "LAY_WIN",    "WIN",   "LAY",  stake_lay_win)
    strategies += _build_slots(enable_back_place, "BACK_PLACE", "PLACE", "BACK", stake_back_place)
    strategies += _build_slots(enable_lay_place,  "LAY_PLACE",  "PLACE", "LAY",  stake_lay_place)

    return StrategyManager(strategies)
