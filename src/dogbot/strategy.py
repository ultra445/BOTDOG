from __future__ import annotations
from typing import List, Dict, Any
from betfairlightweight import filters

class Strategy:
    name: str = "demo_back_fav_under_5"

    def __init__(self, unit_stake: float = 2.0):
        self.unit_stake = unit_stake

    def decide(self, market_book) -> List[Dict[str, Any]]:
        # Pas de coureurs ?
        if not getattr(market_book, "runners", None):
            return []




        # On garde seulement les coureurs actifs
        runners = [r for r in market_book.runners if getattr(r, "status", "ACTIVE") == "ACTIVE"]
        if not runners:
            return []

        # Favori par LTP (last_price_traded)
        fav = min(runners, key=lambda r: (r.last_price_traded or 999.0))
        lpt = fav.last_price_traded or 999.0
        if lpt >= 3.0:
            return []

        # Prix limite légèrement sous le LTP
        price = max(1.01, round(lpt - 0.1, 2))

        instr = filters.place_instruction(
            order_type="LIMIT",
            selection_id=fav.selection_id,
            side="BACK",
            limit_order=filters.limit_order(
                price=price,
                size=self.unit_stake,
                persistence_type="LAPSE",
            ),
        )
        return [instr]
