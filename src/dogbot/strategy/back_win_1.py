from __future__ import annotations
from datetime import datetime
from typing import List, Any
from ..types import Instruction, MarketIndexEntry
from .base import Strategy

class BackWin1(Strategy):
    name = "BACK_WIN_1"

    def decide_all(self, market_book: Any, market_index_entry: MarketIndexEntry, now_utc: datetime) -> List[Instruction]:
        # Garde-fous
        if market_index_entry is None or market_index_entry.event_open_utc is None:
            return []

        # Exemple: n'agit que >= 120s avant l'off
        t_to_off = (market_index_entry.event_open_utc - now_utc).total_seconds()
        if t_to_off < 120:
            return []

        # Favori = runner avec last_price_traded minimal
        try:
            runners = getattr(market_book, "runners", None) or []
            priced = [r for r in runners if getattr(r, "last_price_traded", None)]
            priced.sort(key=lambda r: r.last_price_traded)
        except Exception:
            return []
        if not priced:
            return []

        fav = priced[0]
        lpt = getattr(fav, "last_price_traded", None) or 0.0
        price = max(1.30, min(lpt, 12.0))  # bornes proposées

        # Dry-run: size=0.0
        return [Instruction(selection_id=getattr(fav, "selection_id", 0),
                            side="BACK", price=price, size=0.0)]
