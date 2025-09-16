# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict

# --- project imports ---
from .types import Instruction
from .indexer import MarketIndex
from .risk import ExposureManager
from .execution.orders import OrderExecutor
from .staking import Side  # Side.BACK / Side.LAY

# ExecMode fallback (if strategies doesn't expose it)
try:
    from .strategies import ExecMode  # type: ignore
except Exception:
    from enum import Enum
    class ExecMode(str, Enum):  # pragma: no cover
        LIMIT_LTP = "LIMIT_LTP"
        SP_MOC = "SP_MOC"
        SP_LOC = "SP_LOC"
        HYB = "HYB"

# ---------------- utils ----------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _best_on_side(ex_obj: Any, side: str) -> Tuple[Optional[float], Optional[float]]:
    if not ex_obj:
        return (None, None)
    ladder = getattr(ex_obj, "available_to_back", None) if side == "BACK" else getattr(ex_obj, "available_to_lay", None)
    if not ladder:
        return (None, None)
    top = ladder[0]
    try:
        return float(getattr(top, "price", None)), float(getattr(top, "size", None))
    except Exception:
        p = top.get("price") if isinstance(top, dict) else None
        s = top.get("size") if isinstance(top, dict) else None
        return (float(p) if p is not None else None, float(s) if s is not None else None)

def _runner_ltp_or_best_back(r: Any) -> Optional[float]:
    lpt = getattr(r, "last_price_traded", None)
    try:
        if lpt is not None:
            return float(lpt)
    except Exception:
        pass
    ex = getattr(r, "ex", None)
    pb, _ = _best_on_side(ex, "BACK")
    return pb

def _market_type(book: Any) -> str:
    md = getattr(book, "market_definition", None)
    return (getattr(md, "market_type", None) or "").upper() or "WIN"

def _ranks(runners: Iterable[Any]) -> Tuple[Dict[int, int], Dict[int, int]]:
    arr_ltp, arr_bb = [], []
    for r in runners or []:
        sid = getattr(r, "selection_id", None)
        if sid is None:
            continue
        lpt = _runner_ltp_or_best_back(r)
        ex = getattr(r, "ex", None)
        pb, _ = _best_on_side(ex, "BACK")
        if lpt is not None:
            arr_ltp.append((int(sid), float(lpt)))
        if pb is not None:
            arr_bb.append((int(sid), float(pb)))
    arr_ltp.sort(key=lambda t: t[1])
    arr_bb.sort(key=lambda t: t[1])
    return {sid: i + 1 for i, (sid, _) in enumerate(arr_ltp)}, {sid: i + 1 for i, (sid, _) in enumerate(arr_bb)}

# --- Intention Aggregator (LIMIT + SP_MOC + SP_LOC) --------------------------
@dataclass
class _Intent:
    market_id: str
    selection_id: int
    side: str  # "BACK"/"LAY"
    exec_mode: ExecMode
    price: Optional[float] = None        # LIMIT_LTP price
    size: Optional[float] = None         # BACK stake (LIMIT/SP)
    liability: Optional[float] = None    # LAY liability (SP)
    sp_limit: Optional[float] = None     # SP_LOC limit price
    tag: Optional[str] = None
    reason: Optional[str] = None

class IntentAggregator:
    """
    Agrège par (market_id, selection_id, side) et flush en un seul ordre par type:
      - LIMIT_LTP : somme des tailles ; prix unique (BACK = min ; LAY = max)
      - SP_MOC    : BACK = somme des stakes ; LAY = somme des liabilities
      - SP_LOC    : groupé par sp_limit
    Caps post-agrégation : AGGR_MAX_RUNNER_STAKE_BACK, AGGR_MAX_RUNNER_LIABILITY_LAY
    """
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._by_market: Dict[str, List[_Intent]] = defaultdict(list)
        self.cap_back = self._env_float_opt("AGGR_MAX_RUNNER_STAKE_BACK")
        self.cap_lay_liab = self._env_float_opt("AGGR_MAX_RUNNER_LIABILITY_LAY")
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        self.intents_csv = self.data_dir / f"trades_intents_{today}.csv"
        self.orders_csv  = self.data_dir / f"orders_{today}.csv"
        self._ensure_headers()

    @staticmethod
    def _env_float_opt(name: str) -> Optional[float]:
        s = (os.environ.get(name) or "").strip()
        try:
            return float(s) if s else None
        except Exception:
            return None

    def _ensure_headers(self) -> None:
        if not self.intents_csv.exists():
            with self.intents_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts","market_id","selection_id","side","exec_mode","price","size","liability","sp_limit","tag","reason"])
        if not self.orders_csv.exists():
            with self.orders_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["ts","market_id","selection_id","side","kind","price","size","sp_limit","src_count"])
                w.writeheader()

    def push(self, it: _Intent) -> None:
        self._by_market[it.market_id].append(it)
        # log intent row
        with self.intents_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                _now_iso(),
                it.market_id, it.selection_id, it.side, str(it.exec_mode),
                "" if it.price is None else round(float(it.price),2),
                "" if it.size is None else round(float(it.size),2),
                "" if it.liability is None else round(float(it.liability),2),
                "" if it.sp_limit is None else round(float(it.sp_limit),2),
                it.tag or "", it.reason or "",
            ])

    def flush_market(self, market_id: str, *, live: bool, persistence: str,
                     order_exec: Optional[OrderExecutor], exposure: Optional[ExposureManager]) -> None:
        items = self._by_market.pop(market_id, [])
        if not items:
            return
        grouped: Dict[Tuple, List[_Intent]] = defaultdict(list)
        for it in items:
            if it.exec_mode == ExecMode.LIMIT_LTP:
                key = (it.market_id, it.selection_id, it.side, "LIMIT")
            elif it.exec_mode == ExecMode.SP_MOC:
                key = (it.market_id, it.selection_id, it.side, "SP_MOC")
            elif it.exec_mode == ExecMode.SP_LOC:
                key = (it.market_id, it.selection_id, it.side, "SP_LOC", round(float(it.sp_limit or 0.0),2))
            else:
                continue
            grouped[key].append(it)

        # aggregate
        orders: List[Dict[str, Any]] = []
        for key, bucket in grouped.items():
            mkt, sel, side = key[0], int(key[1]), str(key[2])
            kind = key[3]
            if kind == "LIMIT":
                total = sum(float(x.size or 0.0) for x in bucket)
                if total <= 0: continue
                prices = [float(x.price) for x in bucket if x.price is not None]
                if not prices: continue
                price = min(prices) if side == "BACK" else max(prices)
                orders.append(dict(kind="LIMIT", market_id=mkt, selection_id=sel, side=side,
                                   price=round(price,2), size=round(total,2), sp_limit=None, src_count=len(bucket)))
            elif kind == "SP_MOC":
                if side == "BACK":
                    total = sum(float(x.size or 0.0) for x in bucket)
                    if total <= 0: continue
                    orders.append(dict(kind="SP_MOC", market_id=mkt, selection_id=sel, side=side,
                                       price=None, size=round(total,2), sp_limit=None, src_count=len(bucket)))
                else:
                    total = sum(float(x.liability or 0.0) for x in bucket)
                    if total <= 0: continue
                    orders.append(dict(kind="SP_MOC", market_id=mkt, selection_id=sel, side=side,
                                       price=None, size=round(total,2), sp_limit=None, src_count=len(bucket)))
            elif kind == "SP_LOC":
                sp_limit = float(key[4]) if len(key) > 4 else None
                total = sum(float((x.size if side=="BACK" else x.liability) or 0.0) for x in bucket)
                if total <= 0: continue
                orders.append(dict(kind="SP_LOC", market_id=mkt, selection_id=sel, side=side,
                                   price=None, size=round(total,2), sp_limit=sp_limit, src_count=len(bucket)))

        # caps (after aggregation)
        capped: List[Dict[str, Any]] = []
        for od in orders:
            size_val = float(od["size"])
            if od["side"] == "BACK":
                cap = self.cap_back
                effective = size_val
            else:
                cap = self.cap_lay_liab
                effective = size_val
                if od["kind"] == "LIMIT":
                    p = float(od.get("price") or 0.0)
                    if p > 1.0:
                        effective = size_val * (p - 1.0)
            if cap is None or cap <= 0:
                capped.append(od); continue
            if effective > cap:
                scale = cap / effective if effective > 0 else 0.0
                new_size = round(size_val * scale, 2)
                if new_size <= 0: continue
                od = dict(od); od["size"] = new_size
            capped.append(od)

        # write aggregated orders log
        with self.orders_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["ts","market_id","selection_id","side","kind","price","size","sp_limit","src_count"])
            if f.tell() == 0:
                w.writeheader()
            for od in capped:
                w.writerow({
                    "ts": _now_iso(),
                    "market_id": od["market_id"], "selection_id": od["selection_id"], "side": od["side"],
                    "kind": od["kind"], "price": ("" if od.get("price") is None else round(float(od["price"]),2)),
                    "size": round(float(od["size"]),2),
                    "sp_limit": ("" if od.get("sp_limit") is None else round(float(od["sp_limit"]),2)),
                    "src_count": int(od.get("src_count") or 1),
                })

        # live placement
        if live and order_exec is not None:
            for od in capped:
                # exposure check
                ok = True
                planned_stake, planned_liab = float(od["size"]), None
                if exposure is not None:
                    if od["side"] == "LAY":
                        if od["kind"] == "LIMIT":
                            p = float(od.get("price") or 0.0)
                            planned_liab = planned_stake * max(0.0, p - 1.0)
                        else:
                            planned_liab = planned_stake
                    ok, _ = exposure.can_place(
                        Side.BACK if od["side"]=="BACK" else Side.LAY,
                        market_id=str(od["market_id"]),
                        selection_id=int(od["selection_id"]),
                        planned_stake=float(planned_stake),
                        planned_liability=float(planned_liab) if planned_liab is not None else None,
                    )
                if not ok:
                    continue

                if od["kind"] == "LIMIT":
                    order_exec.place_limit(
                        market_id=str(od["market_id"]),
                        selection_id=int(od["selection_id"]),
                        side=str(od["side"]),
                        price=float(od["price"]),
                        size=float(od["size"]),
                        strategy="AGGR",
                        persistence=os.getenv("PERSISTENCE", "LAPSE"),
                    )
                elif od["kind"] == "SP_MOC":
                    order_exec.place_sp_market_on_close(
                        market_id=str(od["market_id"]),
                        selection_id=int(od["selection_id"]),
                        side=str(od["side"]),
                        size_or_liability=float(od["size"]),
                        strategy="AGGR",
                    )
                else:  # SP_LOC
                    order_exec.place_sp_limit_on_close(
                        market_id=str(od["market_id"]),
                        selection_id=int(od["selection_id"]),
                        side=str(od["side"]),
                        size_or_liability=float(od["size"]),
                        sp_limit_price=float(od.get("sp_limit") or 0.0),
                        strategy="AGGR",
                    )

# -------- executor --------
def _map_order_type_to_exec_mode(order_type: str) -> ExecMode:
    ot = (order_type or "LIMIT").upper()
    if ot in ("LIMIT","LIMIT_LTP","LTL","LIMIT_LTP_BACK","LIMIT_LTP_LAY"):
        return ExecMode.LIMIT_LTP
    if ot in ("SP","SP_MOC","MOC"):
        return ExecMode.SP_MOC
    if ot in ("SP_LOC","LOC"):
        return ExecMode.SP_LOC
    return ExecMode.LIMIT_LTP

class Executor:
    """
    - Push des intentions (slots/strategies) puis **flush agrégé par marché** (LIMIT/SP).
    - LAY • LIMIT (WIN & PLACE) : **prix = best back** (forcé côté exécution).
    - BACK PLACE : CROSS/MID/OWN respectés (pas de forçage).
    """
    def __init__(
        self,
        *,
        client: Any,
        strategy: Optional[Any] = None,
        market_index: Optional[MarketIndex] = None,
        dry_run: bool = True,
        data_dir: str = "./data",
        snapshot_enabled: bool = True,
        snapshot_path: Optional[str] = None,
        snapshot_period: float = 5.0,
        poll_interval: float = 2.0,
    ):
        self.client = client
        self.strategy = strategy
        self.market_index = market_index
        self.dry_run = bool(dry_run)
        self.data_dir = Path(data_dir or "./data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_enabled = bool(snapshot_enabled)
        self.snapshot_path = Path(snapshot_path or self.data_dir)
        self.snapshot_path.mkdir(parents=True, exist_ok=True)
        self.snapshot_period = float(snapshot_period)
        self._last_snapshot_ts: Dict[str, float] = {}

        self.poll_interval = float(poll_interval)

        self.order_exec = OrderExecutor(self.client, data_dir=str(self.data_dir)) if not self.dry_run else None
        self.exposure = ExposureManager() if not self.dry_run else None

        self.aggr = IntentAggregator(data_dir=str(self.data_dir))

    def run(self, market_ids: List[str]) -> None:
        if not market_ids:
            return
        while True:
            try:
                books = self.client.get_market_books(market_ids)  # dict {mid: MarketBook}
            except Exception:
                from betfairlightweight.filters import price_projection
                proj = price_projection(price_data=["EX_BEST_OFFERS", "SP_AVAILABLE"], virtualise=True, rollover_stakes=False)
                books_list = self.client.client.betting.list_market_book(market_ids=market_ids, price_projection=proj)  # type: ignore[attr-defined]
                books = {b.market_id: b for b in books_list}

            for mid, book in list(books.items()):
                try:
                    self.process_book(book)
                except Exception as e:
                    print(f"[EXECUTOR_ERR] {mid}: {e}")

            time.sleep(max(0.0, self.poll_interval))

    def process_book(self, book: Any) -> None:
        mid = getattr(book, "market_id", None)
        if not mid:
            return
        mtype = _market_type(book)

        # stratégie -> intentions
        instrs: List[Instruction] = []
        if self.strategy and hasattr(self.strategy, "decide_all"):
            try:
                now_utc = datetime.now(timezone.utc)
                mie = self.market_index.get(mid) if isinstance(self.market_index, MarketIndex) else None
                instrs = self.strategy.decide_all(book, mie, now_utc) or []
            except Exception:
                instrs = []

        for ins in instrs:
            try:
                exec_mode = _map_order_type_to_exec_mode(getattr(ins, "order_type", "LIMIT"))
                price = None if getattr(ins, "price", None) is None else float(ins.price)
                size  = None if getattr(ins, "size", None)  is None else float(ins.size)

                # Enforce LAY LIMIT price = best BACK for both WIN and PLACE
                if str(getattr(ins, "side", "")).upper() == "LAY" and exec_mode == ExecMode.LIMIT_LTP:
                    sid_target = int(getattr(ins, "selection_id"))
                    best_back_price = None
                    for r in getattr(book, "runners", None) or []:
                        if int(getattr(r, "selection_id", -1) or -1) == sid_target:
                            ex = getattr(r, "ex", None)
                            if ex and getattr(ex, "available_to_back", None):
                                try:
                                    best_back_price = float(ex.available_to_back[0].price)
                                except Exception:
                                    best_back_price = None
                            break
                    if best_back_price is not None:
                        price = best_back_price

                self.aggr.push(_Intent(
                    market_id=str(mid),
                    selection_id=int(ins.selection_id),
                    side=str(ins.side),
                    exec_mode=exec_mode,
                    price=price,
                    size=size,
                    tag=getattr(ins, "strategy_tag", None) or getattr(self.strategy, "name", "STRAT"),
                    reason="from_instruction",
                ))
            except Exception:
                continue

        # Flush agrégé pour ce marché
        self.aggr.flush_market(
            str(mid),
            live=(not self.dry_run),
            persistence=os.getenv("PERSISTENCE", "LAPSE"),
            order_exec=(self.order_exec if not self.dry_run else None),
            exposure=(self.exposure if not self.dry_run else None),
        )
