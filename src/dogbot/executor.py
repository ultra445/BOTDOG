# src/dogbot/executor.py
from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, DefaultDict
from collections import defaultdict

# --- types & infra projet ---
from .types import Instruction, MarketIndexEntry, RunnerMeta
from .indexer import MarketIndex
from .risk import ExposureManager
from .execution.orders import OrderExecutor
from .staking import Side  # Side.BACK / Side.LAY

# (facultatif si tu branches les slots avancés)
try:
    from .strategies import RunnerCtx, ExecMode  # type: ignore
except Exception:
    # Fallback minimal si le module des slots n'est pas utilisé
    from enum import Enum
    @dataclass
    class RunnerCtx:  # pragma: no cover - only for fallback typing
        market_id: str
        market_type: str
        selection_id: int
        course_id: str
        ltp: float
        milestone: Optional[int] = None
        secs_to_off: Optional[float] = None
        trap: Optional[int] = None
        fav_rank_ltp: Optional[int] = None
        fav_rank_back: Optional[int] = None
        gor: Optional[float] = None
        mom45: Optional[float] = None
        mom45_place: Optional[float] = None
        d5: Optional[float] = None
        d30: Optional[float] = None
        vol60: Optional[float] = None
        base_win: Optional[float] = None
        bb: Optional[float] = None
        bl: Optional[float] = None

    class ExecMode(str, Enum):  # pragma: no cover
        LIMIT_LTP = "LIMIT_LTP"
        SP_MOC = "SP_MOC"
        SP_LOC = "SP_LOC"
        HYB = "HYB"

# --------- utils ---------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _env_bool(name: str, default: bool) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if v in ("1", "true", "yes", "on"): return True
    if v in ("0", "false", "no", "off"): return False
    return bool(default)

def _env_float_opt(name: str) -> Optional[float]:
    s = (os.environ.get(name) or "").strip()
    if s == "": return None
    try:
        return float(s)
    except Exception:
        return None

def _best_on_side(ex_obj: Any, side: str) -> Tuple[Optional[float], Optional[float]]:
    """Retourne (best_price, best_size) côté BACK ou LAY."""
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

def _secs_to_off_from_index(book: Any, mindex: Optional[MarketIndex]) -> Optional[float]:
    if not mindex:
        return None
    mid = getattr(book, "market_id", None)
    if not mid:
        return None
    mie = mindex.get(mid)
    if not mie or not isinstance(mie, MarketIndexEntry):
        return None
    start = getattr(mie, "event_open_utc", None)
    if not start:
        return None
    if getattr(start, "tzinfo", None) is None:
        start = start.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (start - now).total_seconds()

def _ranks(runners: Iterable[Any]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Classements par LTP et par meilleur BACK (1 = favori)."""
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

# ============== Agrégateur d’intentions ==============

@dataclass
class _Intent:
    market_id: str
    selection_id: int
    side: str  # "BACK"/"LAY"
    exec_mode: ExecMode
    price: Optional[float] = None        # LIMIT_LTP price
    size: Optional[float] = None         # BACK stake (LIMIT/SP) or size-like
    liability: Optional[float] = None    # LAY liability (SP_MOC/LOC), sinon None
    sp_limit: Optional[float] = None     # pour SP_LOC
    tag: Optional[str] = None            # slot/strategy tag
    reason: Optional[str] = None         # trace pour debug

class IntentAggregator:
    """
    Agrège par (market_id, selection_id, side).

    - LIMIT_LTP : somme des tailles ; prix unique (BACK=min, LAY=max)
    - SP_MOC    : BACK somme des stakes ; LAY somme des liabilities
    - SP_LOC    : groupé par sp_limit (clé distincte)
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        self.intents_csv = self.data_dir / f"trades_intents_{today}.csv"
        self.trades_csv = self.data_dir / f"trades_{today}.csv"
        self._ensure_headers()

        # caps (appliqués APRES agrégation)
        self.cap_back = _env_float_opt("AGGR_MAX_RUNNER_STAKE_BACK")
        self.cap_lay_liab = _env_float_opt("AGGR_MAX_RUNNER_LIABILITY_LAY")

        # panier courant (par marché)
        self._by_market: DefaultDict[str, List[_Intent]] = defaultdict(list)

    def push(self, it: _Intent) -> None:
        self._by_market[it.market_id].append(it)
        self._log_intent(it)

    def flush_market(
        self,
        market_id: str,
        *,
        live: bool,
        persistence: str,
        order_exec: Optional[OrderExecutor],
        exposure: Optional[ExposureManager],
    ) -> None:
        intents = self._by_market.pop(market_id, [])
        if not intents:
            return

        # 1) regrouper
        # clés :
        #   LIMIT_LTP  -> (market_id, selection_id, side, "LIMIT")
        #   SP_MOC     -> (market_id, selection_id, side, "SP_MOC")
        #   SP_LOC     -> (market_id, selection_id, side, "SP_LOC", sp_limit)
        grouped: DefaultDict[Tuple, List[_Intent]] = defaultdict(list)
        for it in intents:
            if it.exec_mode == ExecMode.LIMIT_LTP:
                key = (it.market_id, it.selection_id, it.side, "LIMIT")
            elif it.exec_mode == ExecMode.SP_MOC:
                key = (it.market_id, it.selection_id, it.side, "SP_MOC")
            elif it.exec_mode == ExecMode.SP_LOC:
                key = (it.market_id, it.selection_id, it.side, "SP_LOC", round(float(it.sp_limit or 0.0), 2))
            else:
                # inconnu -> ignorer proprement
                continue
            grouped[key].append(it)

        # 2) agrégation
        agg_orders: List[Dict[str, Any]] = []
        for key, items in grouped.items():
            mkt, sel, side = key[0], int(key[1]), str(key[2])

            if key[3] == "LIMIT":
                # somme des tailles, prix unique
                total_size = sum(float(x.size or 0.0) for x in items if x.size)
                # prix agressif par côté
                prices = [float(x.price) for x in items if x.price is not None]
                if not prices or total_size <= 0.0:
                    continue
                price = min(prices) if side == "BACK" else max(prices)
                agg_orders.append(dict(
                    kind="LIMIT",
                    market_id=mkt,
                    selection_id=sel,
                    side=side,
                    price=round(price, 2),
                    size=round(total_size, 2),
                    sp_limit=None,
                    src_count=len(items),
                ))

            elif key[3] == "SP_MOC":
                if side == "BACK":
                    total_stake = sum(float(x.size or 0.0) for x in items if x.size)
                    if total_stake <= 0.0:
                        continue
                    agg_orders.append(dict(
                        kind="SP_MOC",
                        market_id=mkt,
                        selection_id=sel,
                        side=side,
                        price=None,
                        size=round(total_stake, 2),  # stake
                        sp_limit=None,
                        src_count=len(items),
                    ))
                else:  # LAY -> liabilities
                    total_liab = sum(float(x.liability or 0.0) for x in items if x.liability)
                    if total_liab <= 0.0:
                        continue
                    agg_orders.append(dict(
                        kind="SP_MOC",
                        market_id=mkt,
                        selection_id=sel,
                        side=side,
                        price=None,
                        size=round(total_liab, 2),  # liability
                        sp_limit=None,
                        src_count=len(items),
                    ))

            elif key[3] == "SP_LOC":
                # groupé par sp_limit (déjà dans la clé)
                sp_limit = float(key[4]) if len(key) > 4 else None
                if side == "BACK":
                    total_stake = sum(float(x.size or 0.0) for x in items if x.size)
                    if total_stake <= 0.0:
                        continue
                else:
                    total_stake = sum(float(x.liability or 0.0) for x in items if x.liability)
                    if total_stake <= 0.0:
                        continue
                agg_orders.append(dict(
                    kind="SP_LOC",
                    market_id=mkt,
                    selection_id=sel,
                    side=side,
                    price=None,
                    size=round(total_stake, 2),  # stake BACK / liability LAY
                    sp_limit=sp_limit,
                    src_count=len(items),
                ))

        # 3) appliquer CAP per-runner (scaling proportionnel)
        capped = []
        for od in agg_orders:
            key = (od["market_id"], int(od["selection_id"]), od["side"])
            size_val = float(od["size"])
            if od["side"] == "BACK":
                cap = self.cap_back
            else:
                # LAY : cap exprimé en liability pour SP et en stake pour LIMIT.
                # Ici on s'aligne sur la demande : cap **liability** pour LAY, donc
                # - LIMIT : on approx via liability ~ stake*(price-1) si on a un prix
                # - SP(MOC/LOC): size = liability déjà
                cap = self.cap_lay_liab

            if cap is None or cap <= 0:
                capped.append(od)
                continue

            # convertir LIMIT-LAY stake -> liability si possible
            effective = size_val
            if od["side"] == "LAY" and od["kind"] == "LIMIT":
                p = float(od.get("price") or 0.0)
                if p > 1.0:
                    effective = size_val * (p - 1.0)

            if effective > cap:
                scale = cap / effective if effective > 0 else 0.0
                new_size = round(size_val * scale, 2)
                if new_size <= 0:
                    continue
                od = dict(od)
                od["size"] = new_size
            capped.append(od)

        # 4) Live vs Dryrun
        for od in capped:
            # log agrégé (dans tous les cas)
            self._log_trade(od)

            if not live or order_exec is None:
                continue

            # garde-fous d’exposition
            ok = True
            reason = "ok"
            if exposure is not None:
                if od["side"] == "BACK":
                    planned_stake = float(od["size"])
                    planned_liab = None
                else:
                    if od["kind"] == "LIMIT":
                        price = float(od.get("price") or 0.0)
                        planned_stake = float(od["size"])
                        planned_liab = planned_stake * max(0.0, price - 1.0)
                    else:  # SP MOC / LOC : size = liability
                        planned_stake = float(od["size"])  # utilisé pour per-market
                        planned_liab = float(od["size"])
                ok, reason = exposure.can_place(
                    Side.BACK if od["side"] == "BACK" else Side.LAY,
                    market_id=str(od["market_id"]),
                    selection_id=int(od["selection_id"]),
                    planned_stake=float(planned_stake),
                    planned_liability=float(planned_liab) if planned_liab is not None else None,
                )

            if not ok:
                # on n’envoie pas l’ordre
                continue

            # place order
            if od["kind"] == "LIMIT":
                res = order_exec.place_limit(
                    market_id=str(od["market_id"]),
                    selection_id=int(od["selection_id"]),
                    side=str(od["side"]),
                    price=float(od["price"]),
                    size=float(od["size"]),
                    strategy="AGGR",
                    persistence=persistence,
                )
                if exposure and res.ok:
                    # réserve l’expo
                    liab = None
                    if od["side"] == "LAY":
                        liab = float(od["size"]) * max(0.0, float(od["price"]) - 1.0)
                    exposure.on_placed(
                        Side.BACK if od["side"] == "BACK" else Side.LAY,
                        market_id=str(od["market_id"]),
                        selection_id=int(od["selection_id"]),
                        stake=float(od["size"]),
                        liability=liab,
                    )

            elif od["kind"] == "SP_MOC":
                res = order_exec.place_sp_market_on_close(
                    market_id=str(od["market_id"]),
                    selection_id=int(od["selection_id"]),
                    side=str(od["side"]),
                    size_or_liability=float(od["size"]),
                    strategy="AGGR",
                )
                if exposure and res.ok:
                    exposure.on_placed(
                        Side.BACK if od["side"] == "BACK" else Side.LAY,
                        market_id=str(od["market_id"]),
                        selection_id=int(od["selection_id"]),
                        stake=float(od["size"]) if od["side"] == "BACK" else 0.0,
                        liability=float(od["size"]) if od["side"] == "LAY" else None,
                    )

            elif od["kind"] == "SP_LOC":
                res = order_exec.place_sp_limit_on_close(
                    market_id=str(od["market_id"]),
                    selection_id=int(od["selection_id"]),
                    side=str(od["side"]),
                    size_or_liability=float(od["size"]),
                    sp_limit_price=float(od.get("sp_limit") or 0.0),
                    strategy="AGGR",
                )
                if exposure and res.ok:
                    exposure.on_placed(
                        Side.BACK if od["side"] == "BACK" else Side.LAY,
                        market_id=str(od["market_id"]),
                        selection_id=int(od["selection_id"]),
                        stake=float(od["size"]) if od["side"] == "BACK" else 0.0,
                        liability=float(od["size"]) if od["side"] == "LAY" else None,
                    )

    # ----- internes -----
    def _ensure_headers(self) -> None:
        if not self.intents_csv.exists():
            with self.intents_csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["ts","market_id","selection_id","side","exec_mode","price","size","liability","sp_limit","tag","reason"]
                )
        if not self.trades_csv.exists():
            with self.trades_csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["ts","market_id","selection_id","side","kind","price","size","sp_limit","src_count"]
                )

    def _log_intent(self, it: _Intent) -> None:
        with self.intents_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                _now_iso(), it.market_id, it.selection_id, it.side, str(it.exec_mode),
                "" if it.price is None else round(float(it.price), 2),
                "" if it.size is None else round(float(it.size), 2),
                "" if it.liability is None else round(float(it.liability), 2),
                "" if it.sp_limit is None else round(float(it.sp_limit), 2),
                it.tag or "", it.reason or "",
            ])

    def _log_trade(self, od: Dict[str, Any]) -> None:
        with self.trades_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                _now_iso(), od["market_id"], od["selection_id"], od["side"],
                od["kind"],
                "" if od.get("price") is None else round(float(od["price"]), 2),
                round(float(od["size"]), 2),
                "" if od.get("sp_limit") is None else round(float(od["sp_limit"]), 2),
                int(od.get("src_count") or 1),
            ])


# ============== Executor principal ==============

class Executor:
    """
    - Alimente un agrégateur d’intentions pendant le traitement d’un MarketBook,
      puis flush à la fin (LIMIT/SP_MOC/SP_LOC).
    - Compatible dry-run et live.
    - Snapshots runner CSV avec colonnes GAPMIN, GAPMAX, GOR (None si indisponible).
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

        # live infra
        self.persistence = (os.environ.get("PERSISTENCE") or "LAPSE").strip().upper()
        self.order_exec = OrderExecutor(self.client, data_dir=str(self.data_dir)) if not self.dry_run else None
        self.exposure = ExposureManager() if not self.dry_run else None

        # agrégateur
        self.aggr = IntentAggregator(data_dir=str(self.data_dir))

    # -------- boucle polling (utilisée par run.py) --------
    def run(self, market_ids: List[str]) -> None:
        if not market_ids:
            return
        while True:
            try:
                books = self.client.get_market_books(market_ids)  # dict {mid: MarketBook}
            except Exception:
                # fallback à l'appel direct list_market_book si l'enveloppe ne l’expose pas
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

    # -------- traitement d’un seul MarketBook (utilisé aussi par bot_collect.py) --------
    def process_book(self, book: Any) -> None:
        mid = getattr(book, "market_id", None)
        if not mid:
            return
        mtype = _market_type(book)
        mie = self.market_index.get(mid) if isinstance(self.market_index, MarketIndex) else None

        # ---- runners + features mini ----
        runners = getattr(book, "runners", None) or []
        ranks_ltp, ranks_bb = _ranks(runners)

        # (si tu veux plugger un calcul de GOR @T-2s depuis un cache/time-series, place-le ici)
        gor_by_sid: Dict[int, Optional[float]] = {}

        # ---- stratégie -> intentions ----
        # 1) Strat "classique" qui renvoie des Instruction(s)
        instrs: List[Instruction] = []
        if self.strategy and hasattr(self.strategy, "decide_all"):
            try:
                now_utc = datetime.now(timezone.utc)
                instrs = self.strategy.decide_all(book, mie, now_utc) or []
            except Exception:
                instrs = []

        # Convertir ces instructions en intentions LIMIT_LTP (par défaut)
        for ins in instrs:
            try:
                self.aggr.push(_Intent(
                    market_id=str(mid),
                    selection_id=int(ins.selection_id),
                    side=str(ins.side),
                    exec_mode=ExecMode.LIMIT_LTP,
                    price=float(ins.price),
                    size=float(ins.size),
                    tag=ins.strategy_tag or getattr(self.strategy, "name", "STRAT"),
                    reason="from_instruction",
                ))
            except Exception:
                continue

        # 2) (Optionnel) Slots avancés → try_fire_slot(...) :
        #    Si tu utilises la grille de slots/RunnerCtx, appelle-la ici et push dans l’agrégateur
        # try:
        #     registry = build_registry_from_env()  # si tu as un builder
        #     for r in runners:
        #         sid = int(getattr(r, "selection_id", -1))
        #         if sid < 0: continue
        #         ex = getattr(r, "ex", None)
        #         bb, _ = _best_on_side(ex, "BACK")
        #         bl, _ = _best_on_side(ex, "LAY")
        #         ctx = RunnerCtx(
        #             market_id=str(mid),
        #             market_type=mtype,
        #             selection_id=sid,
        #             course_id=getattr(mie, "course_id", None) or "",
        #             ltp=_runner_ltp_or_best_back(r) or 0.0,
        #             secs_to_off=_secs_to_off_from_index(book, self.market_index),
        #             fav_rank_ltp=ranks_ltp.get(sid),
        #             fav_rank_back=ranks_bb.get(sid),
        #             gor=gor_by_sid.get(sid),
        #             bb=bb, bl=bl,
        #         )
        #         fired = try_fire_slot(registry, ctx)  # -> liste de FireResult
        #         for fr in fired:
        #             self.aggr.push(_Intent(
        #                 market_id=str(mid),
        #                 selection_id=sid,
        #                 side=fr.side.value if hasattr(fr.side, "value") else str(fr.side),
        #                 exec_mode=fr.exec_mode,
        #                 price=float(fr.price) if fr.price is not None else None,
        #                 size=float(fr.size) if fr.size is not None else None,
        #                 liability=float(fr.liability) if fr.liability is not None else None,
        #                 sp_limit=float(fr.sp_limit) if fr.sp_limit is not None else None,
        #                 tag=getattr(fr, "tag", None),
        #                 reason=getattr(fr, "reason", None),
        #             ))
        # except Exception:
        #     pass

        # ---- snapshot runners CSV (optionnel) ----
        if self.snapshot_enabled:
            self._maybe_snapshot(book, mie, ranks_ltp, ranks_bb, gor_by_sid)

        # ---- flush agrégé pour CE marché ----
        self.aggr.flush_market(
            str(mid),
            live=(not self.dry_run),
            persistence=self.persistence,
            order_exec=self.order_exec,
            exposure=self.exposure,
        )

    # -------- snapshots runners ----
    def _maybe_snapshot(
        self,
        book: Any,
        mie: Optional[MarketIndexEntry],
        ranks_ltp: Dict[int, int],
        ranks_bb: Dict[int, int],
        gor_by_sid: Dict[int, Optional[float]],
    ) -> None:
        mid = getattr(book, "market_id", None)
        if not mid:
            return
        now_ts = time.time()
        last = self._last_snapshot_ts.get(str(mid), 0.0)
        if now_ts - last < self.snapshot_period:
            return
        self._last_snapshot_ts[str(mid)] = now_ts

        out = self.snapshot_path / f"runners_{str(mid).replace('.', '_')}.csv"
        newfile = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow([
                    "ts","market_id","selection_id","runner_name","trap",
                    "ltp","best_back","best_lay",
                    "rank_ltp","rank_back",
                    "gapmin","gapmax","gor",  # colonnes demandées
                ])
            for r in getattr(book, "runners", None) or []:
                sid = int(getattr(r, "selection_id", -1) or -1)
                if sid < 0: continue
                name = getattr(r, "runner_name", None) or ""
                meta: Optional[RunnerMeta] = (mie.runners_meta.get(sid) if (mie and mie.runners_meta) else None)  # type: ignore
                trap = getattr(meta, "trap", None) if meta else None
                ex = getattr(r, "ex", None)
                pb, _ = _best_on_side(ex, "BACK")
                pl, _ = _best_on_side(ex, "LAY")
                # GAPs bruts: sans historique T-2s on met None
                gapmin = None
                gapmax = None
                gor = gor_by_sid.get(sid)
                w.writerow([
                    _now_iso(), str(mid), sid, name, trap or "",
                    _runner_ltp_or_best_back(r) or "", pb or "", pl or "",
                    ranks_ltp.get(sid) or "", ranks_bb.get(sid) or "",
                    gapmin if gapmin is not None else "", gapmax if gapmax is not None else "", gor if gor is not None else "",
                ])
