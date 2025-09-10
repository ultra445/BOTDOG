# back_win_1.py — stratégie WIN minimale, robuste (ne lit plus mie.event_open_utc)
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Any

from ..types import Instruction

@dataclass
class BackWin1:
    name: str = "BACK_WIN_1"

    # fenêtre d’activation (secondes avant le départ)
    entry_min_t: int = 120      # >= 2 min
    entry_max_t: int = 7200     # <= 2h
    price_min: float = 1.30
    price_max: float = 12.0

    def _t_to_off(self, market_book: Any, mie: Any) -> Optional[float]:
        # 1) si l’index a market_start_time
        start = getattr(mie, "event_open_utc", None) or getattr(mie, "market_start_time", None)
        # 2) sinon MarketDefinition.market_time
        if start is None:
            md = getattr(market_book, "market_definition", None)
            start = getattr(md, "market_time", None)
        if start is None:
            return None
        if getattr(start, "tzinfo", None) is None:
            start = start.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (start - now).total_seconds()

    def _market_type(self, market_book: Any, mie: Any) -> str:
        md = getattr(market_book, "market_definition", None)
        mt = (getattr(md, "market_type", None) or "").upper()
        if mt:
            return mt
        # fallback via nom du marché si nécessaire
        name = getattr(mie, "market_name", None) or getattr(mie, "marketName", None) or ""
        s = name.lower()
        if "place" in s: return "PLACE"
        if "win" in s or "winner" in s: return "WIN"
        return ""

    def decide_all(self, market_book: Any, mie: Any, now_utc: datetime) -> List[Instruction]:
        out: List[Instruction] = []

        # seulement WIN, pas in-play
        md = getattr(market_book, "market_definition", None)
        if getattr(market_book, "inplay", False):
            return out
        if self._market_type(market_book, mie) != "WIN":
            return out

        tto = self._t_to_off(market_book, mie)
        if tto is None or not (self.entry_min_t <= tto <= self.entry_max_t):
            return out

        runners = getattr(market_book, "runners", None) or []
        if not runners:
            return out

        # choisit le favori par meilleur back (ou LTP)
        best_sid = None
        best_price = None
        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None: 
                continue
            ltp = getattr(r, "last_price_traded", None)
            try:
                if ltp is not None:
                    price = float(ltp)
                else:
                    ex = getattr(r, "ex", None)
                    ladder = getattr(ex, "available_to_back", None)
                    if ladder:
                        price = float(getattr(ladder[0], "price", None))
                    else:
                        continue
            except Exception:
                continue
            if price <= 0:
                continue
            if best_price is None or price < best_price:
                best_price = price
                best_sid = int(sid)

        if best_sid is None or best_price is None:
            return out

        if not (self.price_min <= best_price <= self.price_max):
            return out

        # Mise = 0 en dry-run (Executor imprime asdict())
        instr = Instruction(
            selection_id=best_sid,
            side="BACK",
            price=float(best_price),
            size=0.0,
            order_type="LIMIT",
            persistence="LAPSE",
            strategy_tag=self.name,
        )
        out.append(instr)
        return out
