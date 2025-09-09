# src/dogbot/types.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict

@dataclass
class RunnerMeta:
    selection_id: int
    runner_name: Optional[str] = None
    sort_priority: Optional[int] = None
    trap: Optional[str] = None  # greyhound box number if provided
    draw: Optional[str] = None  # alias (some feeds use DRAW)

@dataclass
class MarketIndexEntry:
    market_id: str
    market_type: str  # "WIN" or "PLACE"
    event_id: Optional[str]
    event_name: Optional[str]
    event_open_utc: Optional[datetime]
    venue: Optional[str]
    country_code: Optional[str]
    event_local_date: Optional[str]
    race_number: Optional[str]
    course_id: Optional[str]
    # linking WIN <-> PLACE
    win_market_id: Optional[str] = None
    place_market_id: Optional[str] = None
    n_places: Optional[int] = None  # for PLACE markets if available
    # runner metadata for this market
    runners_meta: Dict[int, RunnerMeta] = field(default_factory=dict)

@dataclass
class Instruction:
    selection_id: int
    side: str              # "BACK" / "LAY"
    price: float
    size: float
    order_type: str = "LIMIT"
    persistence: str = "LAPSE"
    strategy_tag: Optional[str] = None

    def asdict(self):
        d = asdict(self)
        # Betfair payloads often want camelCase; we keep snake_case for logs/CSV.
        return d
