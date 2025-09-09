# src/dogbot/indexer.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Iterable, Dict, Optional, List
from .types import MarketIndexEntry, RunnerMeta

def _norm_dt_utc(dt) -> Optional[datetime]:
    if dt is None:
        return None
    if isinstance(dt, str):
        try:
            # handle "...Z"
            return datetime.fromisoformat(dt.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return None
    if getattr(dt, "tzinfo", None) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def make_course_id(venue: Optional[str], event_open_utc: Optional[datetime], country_code: Optional[str], race_number: Optional[str|int]) -> Optional[str]:
    if not venue or not event_open_utc:
        return None
    d = event_open_utc.astimezone(timezone.utc).date().isoformat()
    rn = f"{int(race_number):02d}" if race_number not in (None, "",) else "R00"
    cc = (country_code or "").upper()
    v = venue.strip().upper().replace(" ", "_")
    return f"{v}:{d}:{rn}:{cc}"

class MarketIndex:
    def __init__(self, entries: Iterable[MarketIndexEntry] | None = None):
        self._by_market: Dict[str, MarketIndexEntry] = {}
        if entries:
            for e in entries:
                self.add(e)

    def add(self, e: MarketIndexEntry) -> None:
        self._by_market[e.market_id] = e

    def get(self, market_id: str) -> Optional[MarketIndexEntry]:
        return self._by_market.get(market_id)

    def values(self) -> Iterable[MarketIndexEntry]:
        return self._by_market.values()

    @classmethod
    def from_catalogue(cls, catalogue_markets) -> "MarketIndex":
        """
        Construit l'index à partir de list_market_catalogue(...).
        Récupère meta runners (nom, sortPriority, trap/draw) quand disponible.
        Tente le linkage WIN<->PLACE par course_id.
        """
        entries: List[MarketIndexEntry] = []
        for m in catalogue_markets:
            event = getattr(m, "event", None)
            venue = getattr(event, "venue", None)
            event_name = getattr(event, "name", None)
            cc = getattr(event, "country_code", None) or getattr(event, "countryCode", None)
            open_utc = getattr(m, "market_start_time", None) or getattr(m, "marketStartTime", None)
            open_utc = _norm_dt_utc(open_utc)

            market_type = getattr(m, "market_type", None) or getattr(m, "marketType", None) or "WIN"
            market_id = getattr(m, "market_id", None)
            event_id = getattr(event, "id", None)

            # try race number from market name
            race_number = None
            mn = getattr(m, "market_name", None) or getattr(m, "marketName", None)
            if isinstance(mn, str):
                # e.g. "14:36 WIN" rarely carries race number; keep None unless parsed from metadata
                pass

            # runner metadata
            runners_meta: Dict[int, RunnerMeta] = {}
            runners = getattr(m, "runners", None) or getattr(m, "runners", None)
            for r in (runners or []):
                sel_id = getattr(r, "selection_id", None) or getattr(r, "selectionId", None)
                if sel_id is None:
                    continue
                name = getattr(r, "runner_name", None) or getattr(r, "runnerName", None)
                sp = getattr(r, "sort_priority", None) or getattr(r, "sortPriority", None)
                meta = getattr(r, "metadata", None) or {}
                trap = None
                draw = None
                if isinstance(meta, dict):
                    trap = meta.get("TRAP") or meta.get("Trap") or meta.get("trap") or meta.get("DRAW")
                    draw = meta.get("DRAW") or meta.get("Draw")
                runners_meta[int(sel_id)] = RunnerMeta(
                    selection_id=int(sel_id),
                    runner_name=name,
                    sort_priority=sp if sp is None else int(sp),
                    trap=str(trap) if trap is not None else None,
                    draw=str(draw) if draw is not None else None,
                )

            course_id = make_course_id(venue, open_utc, cc, race_number)
            entries.append(
                MarketIndexEntry(
                    market_id=market_id,
                    market_type=market_type,
                    event_id=event_id,
                    event_name=event_name,
                    event_open_utc=open_utc,
                    venue=venue,
                    country_code=cc,
                    event_local_date=None,
                    race_number=race_number,
                    course_id=course_id,
                    runners_meta=runners_meta,
                )
            )

        idx = cls(entries)

        # Link WIN<->PLACE by course_id
        by_course: Dict[str, Dict[str, str]] = {}
        for e in idx.values():
            if not e.course_id:
                continue
            d = by_course.setdefault(e.course_id, {})
            d[e.market_type] = e.market_id

        for e in idx.values():
            if not e.course_id:
                continue
            d = by_course.get(e.course_id, {})
            e.win_market_id = d.get("WIN")
            e.place_market_id = d.get("PLACE")
            # N_PLACES : inconnu au catalogue, sera rempli plus tard depuis MarketBook (number_of_winners) si PLACE

        return idx
