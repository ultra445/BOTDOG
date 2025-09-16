# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional

class MarketState:
    """Séries de prix par selection_id.
       WIN: LTP prioritaire; PLACE: BEST BACK prioritaire (fallback LTP).
    """
    def __init__(self, market_id: str, start_time: Any, market_type: str, meta_index: Dict[str, Any] | None = None):
        self.market_id = market_id
        self.market_type = (market_type or "").upper()
        self.start_time = self._norm_ts(start_time)
        self.meta_index = meta_index or {}
        self.ltp_ts: Dict[int, deque] = defaultdict(lambda: deque(maxlen=5000))
        self.status: Dict[int, str] = {}
        self.trap_map: Dict[int, int] = {}
        self.planned_count: Optional[int] = None

    @staticmethod
    def _norm_ts(ts: Any) -> Optional[dt.datetime]:
        if ts is None:
            return None
        if isinstance(ts, dt.datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=dt.timezone.utc)
        try:
            s = str(ts)
            if s.endswith("Z"):
                return dt.datetime.fromisoformat(s.replace("Z","+00:00"))
            d = dt.datetime.fromisoformat(s)
            return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
        except Exception:
            return None

    def update_from_book(self, book) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        mtype = self.market_type or ""
        for r in getattr(book, "runners", []) or []:
            sel_id = getattr(r, "selection_id", None)
            if sel_id is None:
                continue
            self.status[sel_id] = getattr(r, "status", "ACTIVE") or "ACTIVE"

            ref = None
            ex = getattr(r, "ex", None)
            if mtype == "PLACE":
                if ex and getattr(ex, "available_to_back", None):
                    try:
                        ref = float(ex.available_to_back[0].price)
                    except Exception:
                        ref = None
                if ref is None:
                    ltp = getattr(r, "last_price_traded", None)
                    try:
                        ref = float(ltp) if ltp is not None else None
                    except Exception:
                        ref = None
            else:
                ltp = getattr(r, "last_price_traded", None)
                try:
                    ref = float(ltp) if ltp is not None else None
                except Exception:
                    ref = None
                if ref is None and ex and getattr(ex, "available_to_back", None):
                    try:
                        ref = float(ex.available_to_back[0].price)
                    except Exception:
                        ref = None

            if ref is not None and ref > 1.0:
                self.ltp_ts[sel_id].append((now, float(ref)))

    def _ltp_at(self, sel_id: int, target_ts: dt.datetime) -> Optional[float]:
        series = self.ltp_ts.get(sel_id)
        if not series: return None
        for ts, val in reversed(series):
            if ts <= target_ts:
                return val
        return None

    def _ltp_now(self, sel_id: int) -> Optional[float]:
        series = self.ltp_ts.get(sel_id)
        return series[-1][1] if series else None

class FeatureStore:
    def __init__(self):
        self.states: Dict[str, MarketState] = {}

    def prime_market(self, market_id: str, start_time: Any, market_type: str, index_entry: Dict[str, Any] | None = None):
        if market_id not in self.states:
            self.states[market_id] = MarketState(market_id, start_time, market_type, index_entry or {})

    def ingest_book(self, book) -> None:
        mid = getattr(book, "market_id", None)
        if mid is None: return
        st = self.states.get(mid)
        if st is None:
            # fallback
            mtype = ""
            stime = None
            try:
                md = getattr(book, "market_definition", None)
                if md:
                    mtype = getattr(md, "market_type", "") or ""
                    stime = getattr(md, "market_time", None)
            except Exception:
                pass
            st = MarketState(mid, stime, mtype, {})
            self.states[mid] = st
        st.update_from_book(book)
