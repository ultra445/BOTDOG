from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from typing import List, Any, Optional
import csv
import os

from .types import Instruction, MarketIndexEntry
from .indexer import MarketIndex

class SnapshotWriter:
    MIN_COLUMNS = [
        "ts_utc","market_id","market_type","inplay","runners","t_to_off_s",
        "last_price_fav","vol_traded","venue","event_date_local","race_number","course_id"
    ]

    def __init__(self, data_dir: str = "./data"):
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, datetime.now(timezone.utc).strftime("%Y%m%d") + "_snapshots.csv")
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.MIN_COLUMNS)

    def write_row(self, row: dict):
        safe = {k: row.get(k) for k in self.MIN_COLUMNS}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([safe.get(k) for k in self.MIN_COLUMNS])

def _market_type(book: Any) -> Optional[str]:
    md = getattr(book, "market_definition", None)
    if md is None:
        return None
    return getattr(md, "market_type", getattr(md, "marketType", None))

def _inplay(book: Any) -> Optional[bool]:
    return getattr(book, "inplay", getattr(book, "in_play", None))

def _total_matched(book: Any) -> Optional[float]:
    return getattr(book, "total_matched", getattr(book, "totalMatched", None))

def _fav_last_price(book: Any) -> Optional[float]:
    try:
        runners = getattr(book, "runners", None) or []
        vals = [getattr(r, "last_price_traded", None) for r in runners]
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None
    except Exception:
        return None

class Executor:
    def __init__(self, client: Any, strategy: Any, market_index: MarketIndex,
                 dry_run: bool = True, data_dir: str = "./data"):
        self.client = client
        self.strategy = strategy
        self.market_index = market_index
        self.dry_run = dry_run
        self.snaps = SnapshotWriter(data_dir=data_dir)

    def _t_to_off(self, mie: Optional[MarketIndexEntry], now_utc: datetime) -> Optional[float]:
        try:
            return (mie.event_open_utc - now_utc).total_seconds() if (mie and mie.event_open_utc) else None
        except Exception:
            return None

    def process_book(self, market_book: Any) -> List[Instruction]:
        now_utc = datetime.now(timezone.utc)
        mie = self.market_index.get(getattr(market_book, "market_id", ""))

        # --- SNAPSHOT AVANT stratégie ---
        try:
            row = {
                "ts_utc": now_utc.isoformat().replace("+00:00", "Z"),
                "market_id": getattr(market_book, "market_id", None),
                "market_type": _market_type(market_book),
                "inplay": _inplay(market_book),
                "runners": len(getattr(market_book, "runners", None) or []),
                "t_to_off_s": self._t_to_off(mie, now_utc),
                "last_price_fav": _fav_last_price(market_book),
                "vol_traded": _total_matched(market_book),
                "venue": getattr(mie, "venue", None) if mie else None,
                "event_date_local": getattr(mie, "event_local_date", None) if mie else None,
                "race_number": getattr(mie, "race_number", None) if mie else None,
                "course_id": getattr(mie, "course_id", None) if mie else None,
            }
            self.snaps.write_row(row)
        except Exception as e:
            print(f"[SNAPSHOT_ERR] {e}")

        # --- DÉCISION ---
        try:
            instructions = self.strategy.decide_all(market_book, mie, now_utc)
        except Exception as e:
            print(f"[STRATEGY_ERR] {e}")
            instructions = []

        # --- EXÉCUTION ---
        if self.dry_run:
            for ins in instructions:
                try:
                    print(f"[DRY] would place {asdict(ins)} on {getattr(market_book, 'market_id', '?')}")
                except Exception:
                    print("[DRY] would place (unprintable instruction)")
        else:
            self._send_orders(getattr(market_book, "market_id", ""), instructions)

        return instructions

    def _send_orders(self, market_id: str, instructions: List[Instruction]):
        # TODO: Traduire Instruction -> placeOrders (betfairlightweight)
        pass
