from __future__ import annotations
from datetime import datetime
from typing import Dict, Iterable, Tuple, Optional

try:
    import pytz  # facultatif; si absent, on fait un fallback simple
except Exception:
    pytz = None

from .types import MarketIndexEntry

_COUNTRY_TZ = {
    "GB": "Europe/London",
    "IE": "Europe/Dublin",
    "AU": "Australia/Sydney",   # à affiner par état si besoin
    "NZ": "Pacific/Auckland",
}

def _local_date(utc_dt: datetime, country_code: Optional[str]) -> str:
    if pytz is None or country_code is None:
        return utc_dt.date().isoformat()
    tzname = _COUNTRY_TZ.get(country_code, "Europe/London")
    tz = pytz.timezone(tzname)
    return utc_dt.astimezone(tz).date().isoformat()

def make_course_id(venue: Optional[str], event_open_utc: datetime,
                   country_code: Optional[str], race_number: Optional[int]) -> str:
    v = (venue or "").strip().upper().replace(" ", "_")
    d = _local_date(event_open_utc, country_code)
    r = f"R{(race_number or 0):02d}"
    return f"{v}:{d}:{r}"

class MarketIndex:
    def __init__(self, entries: Iterable[MarketIndexEntry]):
        self.by_market: Dict[str, MarketIndexEntry] = {}
        self.by_course: Dict[Tuple[str, str], str] = {}  # (course_id, market_type) -> market_id
        for e in entries:
            self.by_market[e.market_id] = e
            if e.course_id:
                self.by_course[(e.course_id, e.market_type)] = e.market_id

    def get(self, market_id: str) -> Optional[MarketIndexEntry]:
        return self.by_market.get(market_id)

    def pair_place(self, course_id: str) -> tuple[Optional[str], Optional[str]]:
        return (
            self.by_course.get((course_id, "WIN")),
            self.by_course.get((course_id, "PLACE")),
        )
