from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
from datetime import datetime

Side = Literal["BACK", "LAY"]
PersistenceType = Literal["LAPSE", "PERSIST", "MARKET_ON_CLOSE"]

@dataclass
class Instruction:
    selection_id: int
    side: Side
    price: float
    size: float  # stake en unité monnaie
    order_type: Literal["LIMIT"] = "LIMIT"
    persistence: PersistenceType = "LAPSE"
    strategy_tag: str = "BACK_WIN_1"

@dataclass
class MarketIndexEntry:
    market_id: str
    market_type: str  # "WIN" | "PLACE" | ...
    event_id: str
    event_open_utc: datetime  # heure catalogue (UTC)
    venue: Optional[str]
    country_code: Optional[str]
    event_local_date: Optional[str]  # YYYY-MM-DD (optionnel)
    race_number: Optional[int]
    course_id: Optional[str]
