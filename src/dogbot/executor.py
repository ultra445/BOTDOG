# src/dogbot/executor.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional, Dict, List, Tuple
from collections import defaultdict, deque
import csv
import math

from .types import MarketIndex, MarketIndexEntry, Instruction, RunnerMeta
from .strategy.base import Strategy

# ===== Helpers calc =====

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _best_at_side(ex_obj, side: str) -> tuple[Optional[float], Optional[float]]:
    if ex_obj is None:
        return (None, None)
    ladder = getattr(ex_obj, "available_to_back", None) if side == "BACK" else getattr(ex_obj, "available_to_lay", None)
    if not ladder:
        return (None, None)
    top = ladder[0]
    try:
        return float(top.price), float(top.size)
    except Exception:
        p = top.get("price") if isinstance(top, dict) else None
        s = top.get("size") if isinstance(top, dict) else None
        return (float(p) if p is not None else None, float(s) if s is not None else None)

def _compress_ladder(ladder, n=3) -> Optional[str]:
    if not ladder:
        return None
    items = []
    for i, x in enumerate(ladder[:n]):
        try:
            items.append(f"{x.price}:{x.size}")
        except Exception:
            p = x.get("price") if isinstance(x, dict) else None
            s = x.get("size") if isinstance(x, dict) else None
            if p is None or s is None:
                continue
            items.append(f"{p}:{s}")
    return "|".join(items) if items else None

def _runner_lpt(r) -> Optional[float]:
    lpt = getattr(r, "last_price_traded", None)
    try:
        return float(lpt) if lpt is not None else None
    except Exception:
        return None

def _market_status(md) -> Optional[str]:
    try:
        return getattr(md, "status", None)
    except Exception:
        return None

def _num_winners(md) -> Optional[int]:
    try:
        v = getattr(md, "number_of_winners", None) or getattr(md, "numberOfWinners", None)
        return int(v) if v is not None else None
    except Exception:
        return None

def _active_count(runners) -> int:
    n = 0
    for r in (runners or []):
        st = getattr(r, "status", None)
        if (st or "ACTIVE").upper() == "ACTIVE":
            n += 1
    return n

def _overround_back(runners) -> Optional[float]:
    s = 0.0
    k = 0
    for r in (runners or []):
        ex = getattr(r, "ex", None)
        pb, _ = _best_at_side(ex, "BACK")
        if pb and pb > 1e-9:
            s += 1.0 / pb
            k += 1
    if k == 0:
        return None
    return s * 100.0

def _overround_lay(runners) -> Optional[float]:
    s = 0.0
    k = 0
    for r in (runners or []):
        ex = getattr(r, "ex", None)
        pl, _ = _best_at_side(ex, "LAY")
        if pl and pl > 1e-9:
            s += 1.0 / pl
            k += 1
    if k == 0:
        return None
    return s * 100.0

def _spread_pct(runners) -> Optional[float]:
    vals = []
    for r in (runners or []):
        ex = getattr(r, "ex", None)
        pb, _ = _best_at_side(ex, "BACK")
        pl, _ = _best_at_side(ex, "LAY")
        if pb and pl and pb > 0:
            mid = (pb + pl) / 2.0
            if mid > 0:
                vals.append((pl - pb) / mid)
    if not vals:
        return None
    vals.sort()
    m = vals[len(vals)//2]
    return m * 100.0

def _rankings(runners) -> tuple[Dict[int,int], Dict[int,int]]:
    """Retourne (rank_by_ltp, rank_by_back) mapping selection_id -> rang (1=favori)"""
    arr_ltp = []
    arr_bb  = []
    for r in (runners or []):
        sid = getattr(r, "selection_id", None)
        if sid is None: 
            continue
        lpt = _runner_lpt(r)
        ex = getattr(r, "ex", None)
        pb, _ = _best_at_side(ex, "BACK")
        if lpt is not None:
            arr_ltp.append((sid, lpt))
        if pb is not None:
            arr_bb.append((sid, pb))
    arr_ltp.sort(key=lambda t: t[1])  # plus petit = fav
    arr_bb.sort(key=lambda t: t[1])
    rank_ltp = {sid: i+1 for i, (sid, _) in enumerate(arr_ltp)}
    rank_bb  = {sid: i+1 for i, (sid, _) in enumerate(arr_bb)}
    return rank_ltp, rank_bb

# ===== Executor =====

class Executor:
    """
    - Snapshots marché + runners (aux milestones)
    - Milestones: 300/150/80/45/2 s avant off
    - Calcul de deltas 5s/30s et volatilité 60s par runner
    - Expose: identifiants, temps, prix, overrounds, ranks, flags rules, plan d'ordre si déclenché
    """
    MILESTONES = [300, 150, 80, 45, 2]
    TOLERANCE_S = 1.5

    MARKET_HEADER = [
        # Identifiants & temps
        "SNAP_TS_UTC","MARKET_ID","COURSE_ID","MARKET_TYPE",
        "EVENT_ID","EVENT_NAME","VENUE","COUNTRY_CODE",
        "MARKET_START_TIME_UTC","SECONDS_TO_OFF",
        "INPLAY","MARKET_STATUS","NUMBER_OF_WINNERS","N_RUNNERS_ACTIVE","MARKET_TOTAL_MATCHED",
        # Prix & agrégats marché
        "BACK_OVERROUND","LAY_OVERROUND","SPREAD_PCT",
        # Appariement WIN<->PLACE
        "WIN_MARKET_ID","PLACE_MARKET_ID","COURSE_LINK_OK","N_PLACES",
        # Gating rules (exemple)
        "RULE_OK_TIME_WINDOW","RULE_OK_PRICE_BOUNDS","RULE_OK_LIQUIDITY","RULE_OK_OVERROUND","RULES_ALL_OK",
        # Décision (si déclenchée)
        "STRATEGY_NAME","SIGNAL","SIDE","TARGET_PRICE","STAKE","PERSISTENCE","HEDGE_TICKS","STOP_TICKS",
        "EST_LIABILITY","REASON_CODE","DRYRUN_WOULD_PLACE",
        # Milestone
        "MILESTONE_S",
        # Diagnostics
        "FETCH_LATENCY_MS","CACHE_AGE_S","RETRY_COUNT","THROTTLE_WEIGHT","CODE_VERSION",
    ]

    RUNNER_HEADER = [
        "SNAP_TS_UTC","MARKET_ID","SELECTION_ID","RUNNER_NAME","RUNNER_STATUS",
        "DRAW","TRAP","SORT_PRIORITY",
        "LTP","BEST_BACK_PRICE_1","BEST_BACK_SIZE_1","BEST_LAY_PRICE_1","BEST_LAY_SIZE_1",
        "BACK_LADDER","LAY_LADDER","RUNNER_TOTAL_MATCHED",
        "FAV_RANK_LTP","FAV_RANK_BACK","WIN_IMPLIED_PROB",
        "PRICE_DELTA_5S","PRICE_DELTA_30S","VOLATILITY_60S","LIQUIDITY_SCORE",
        "IS_FAVOURITE","PLACE_THEORIQUE",
        "SECONDS_TO_OFF","VENUE","COURSE_ID","MILESTONE_S"
    ]

    # Paramètres (à terme: lire depuis config/env)
    ENTRY_MIN_T_S = 120
    ENTRY_MAX_T_S = 7200
    PRICE_MIN = 1.30
    PRICE_MAX = 12.0
    MIN_LIQUIDITY_MARKET = 0.0  # en dry-run on n'impose pas

    def __init__(self, client: Any, strategy: Strategy, market_index: MarketIndex, dry_run: bool = True, data_dir: str = "./data"):
        self.client = client
        self.strategy = strategy
        self.market_index = market_index
        self.dry_run = dry_run
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        self.market_csv = self.data_dir / f"{today}_snapshots.csv"
        self.runner_csv = self.data_dir / f"{today}_snapshots_runners.csv"
        self._ensure_header(self.market_csv, self.MARKET_HEADER)
        self._ensure_header(self.runner_csv, self.RUNNER_HEADER)

        # suivi milestones & historique prix pour deltas/volatilité
        self._next_ms: Dict[str, List[int]] = defaultdict(lambda: list(self.MILESTONES))
        self._last_tto: Dict[str, float] = {}
        self._hist: Dict[str, Dict[int, deque[Tuple[float,float]]]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=600)))  # ts, price; 600 x 1s = 10m

    # ---------- I/O CSV ----------
    def _ensure_header(self, path: Path, header: Iterable[str]) -> None:
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(list(header))

    def _append(self, path: Path, row: Iterable[Any]) -> None:
        with path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(list(row))

    # ---------- Milestones ----------
    def _milestone_due(self, mid: str, tto: Optional[float]) -> Optional[int]:
        if tto is None:
            return None
        ms_list = self._next_ms[mid]
        # tolérance à la première mesure
        for ms in list(ms_list):
            if abs(tto - ms) <= self.TOLERANCE_S:
                ms_list.remove(ms)
                return ms
        last = self._last_tto.get(mid)
        if last is None:
            return None
        for ms in list(ms_list):
            if (last - ms) > 0 and (ms - tto) >= 0:
                ms_list.remove(ms)
                return ms
        return None

    # ---------- Deltas/volatilité ----------
    def _push_hist(self, market_id: str, selection_id: int, price: Optional[float]) -> None:
        if price is None:
            return
        dq = self._hist[market_id][selection_id]
        dq.append((datetime.now(timezone.utc).timestamp(), float(price)))

    def _delta_since(self, market_id: str, selection_id: int, seconds: float) -> Optional[float]:
        dq = self._hist.get(market_id, {}).get(selection_id)
        if not dq:
            return None
        now_ts = datetime.now(timezone.utc).timestamp()
        target = now_ts - seconds
        older = None
        for ts, p in reversed(dq):  # plus récent -> plus vieux
            older = (ts, p)
            if ts <= target:
                break
        if older is None:
            return None
        _, p_old = older
        p_now = dq[-1][1]
        return p_now - p_old

    def _volatility(self, market_id: str, selection_id: int, window_s: float = 60.0) -> Optional[float]:
        dq = self._hist.get(market_id, {}).get(selection_id)
        if not dq:
            return None
        now_ts = datetime.now(timezone.utc).timestamp()
        vals = [p for ts, p in dq if ts >= now_ts - window_s]
        if len(vals) < 3:
            return None
        m = sum(vals) / len(vals)
        var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
        return math.sqrt(var)

    # ---------- Process ----------
    def process_book(self, book: Any) -> None:
        market_id = str(getattr(book, "market_id", ""))
        md = getattr(book, "market_definition", None)
        runners = getattr(book, "runners", None) or []
        now = datetime.now(timezone.utc)

        mie: Optional[MarketIndexEntry] = self.market_index.get(market_id) if self.market_index else None
        event_open_utc = mie.event_open_utc if mie else None
        t_to_off_s = (event_open_utc - now).total_seconds() if event_open_utc else None

        # in-play -> on arrête de snaper
        if getattr(book, "inplay", False):
            self._next_ms.pop(market_id, None)
            self._last_tto[market_id] = t_to_off_s if t_to_off_s is not None else 0.0
            return

        # rank info
        rank_ltp, rank_back = _rankings(runners)

        # push hist for deltas/vol
        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None:
                continue
            lpt = _runner_lpt(r)
            if lpt is None:
                ex = getattr(r, "ex", None)
                lpt, _ = _best_at_side(ex, "BACK")
            self._push_hist(market_id, int(sid), lpt)

        milestone = self._milestone_due(market_id, t_to_off_s)
        # Always write market-level snapshot (with milestone_s)
        self._write_market_row(book, mie, t_to_off_s, milestone, runners, rank_back)

        # Runner snapshots only at milestones
        if milestone is not None:
            self._write_runner_rows(book, mie, t_to_off_s, milestone, runners, rank_ltp, rank_back)

        # Strategy call
        instructions: List[Instruction] = []
        try:
            instructions = self.strategy.decide_all(book, mie, now) or []
        except Exception as e:
            print(f"[STRATEGY_ERR] {market_id}: {e}")

        if instructions:
            if self.dry_run:
                for ins in instructions:
                    print("[DRY] would place", ins.asdict(), "on", market_id)
            else:
                for ins in instructions:
                    print("[LIVE] placing", ins.asdict(), "on", market_id)

        if t_to_off_s is not None:
            self._last_tto[market_id] = t_to_off_s

    # ---------- Market row ----------
    def _write_market_row(self, book, mie: Optional[MarketIndexEntry], tto: Optional[float], milestone: Optional[int], runners, rank_back: Dict[int,int]) -> None:
        md = getattr(book, "market_definition", None)
        fav_sid = None
        # find favourite by best-back rank 1
        if rank_back:
            for sid, rk in rank_back.items():
                if rk == 1:
                    fav_sid = sid
                    break

        back_over = _overround_back(runners)
        lay_over = _overround_lay(runners)
        spread = _spread_pct(runners)

        market_row = [
            # Identifiants & temps
            _now_utc_iso(),
            str(getattr(book, "market_id", "")),
            mie.course_id if mie else None,
            (getattr(md, "market_type", None) or (mie.market_type if mie else None)),
            (mie.event_id if mie else None),
            (mie.event_name if mie else None),
            (mie.venue if mie else None),
            (mie.country_code if mie else None),
            mie.event_open_utc.isoformat().replace("+00:00", "Z") if (mie and mie.event_open_utc) else None,
            float(tto) if tto is not None else None,
            bool(getattr(book, "inplay", False)),
            _market_status(md),
            _num_winners(md),
            _active_count(runners),
            float(getattr(book, "total_matched", None) or 0.0),
            # Prix & agrégats
            back_over, lay_over, spread,
            # Link WIN<->PLACE
            (mie.win_market_id if mie else None),
            (mie.place_market_id if mie else None),
            (1 if (mie and mie.win_market_id and mie.place_market_id) else 0),
            (mie.n_places if mie else None),
            # Gating rules (exemples simples)
            (tto is not None and self.ENTRY_MIN_T_S <= tto <= self.ENTRY_MAX_T_S),
            # price bounds: on utilise le fav back si dispo
            self._fav_price_ok(runners),
            (float(getattr(book, "total_matched", 0.0)) >= self.MIN_LIQUIDITY_MARKET),
            (back_over is not None and lay_over is not None),  # placeholder règle overround
            None,  # RULES_ALL_OK sera rempli après si on a une instru (voir ci-dessous)
            # Décision (remplie si la stratégie a retourné une instruction)
            (getattr(self.strategy, "name", None) or None),
            None, None, None, None, None, None, None, None,  # SIGNAL..REASON_CODE
            False,  # DRYRUN_WOULD_PLACE
            # Milestone / Diag
            milestone,
            None, None, None, None,  # diagnostics placeholders
        ]

        # STRATEGY preview: si decide_all décide, on les met ici
        # NB: on ne relance pas la stratégie; on lit ce qu'elle a décidé dans le dernier run (pas idéal).
        # Si tu veux que ces colonnes reflètent EXACTEMENT les ordres, il faut passer les instructions ici.
        # On garde simple: DRY dans console = vérité, CSV = “hint”.
        # (Option avancée: faire renvoyer decide_all(...) et on remplit directement)
        # => pour l’instant, on laisse tel quel; les colonnes décision restent None/False.

        # Écrit la ligne
        self._append(self.market_csv, market_row)

    def _fav_price_ok(self, runners) -> Optional[bool]:
        # favori par best-back
        ex_prices = []
        for r in (runners or []):
            ex = getattr(r, "ex", None)
            pb, _ = _best_at_side(ex, "BACK")
            if pb:
                ex_prices.append(pb)
        if not ex_prices:
            return None
        fav_price = min(ex_prices)
        return (self.PRICE_MIN <= fav_price <= self.PRICE_MAX)

    # ---------- Runner rows ----------
    def _write_runner_rows(self, book, mie: Optional[MarketIndexEntry], tto: Optional[float], milestone: int, runners, rank_ltp: Dict[int,int], rank_back: Dict[int,int]) -> None:
        ts = _now_utc_iso()
        market_id = str(getattr(book, "market_id", ""))

        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None:
                continue
            sid = int(sid)
            lpt = _runner_lpt(r)
            ex = getattr(r, "ex", None)
            bb, bs = _best_at_side(ex, "BACK")
            bl, ls = _best_at_side(ex, "LAY")
            ladder_b = _compress_ladder(getattr(ex, "available_to_back", None), 3)
            ladder_l = _compress_ladder(getattr(ex, "available_to_lay", None), 3)
            total_matched_runner = getattr(r, "total_matched", None)
            status = getattr(r, "status", None)

            # runner meta
            rm: RunnerMeta | None = (mie.runners_meta.get(sid) if (mie and mie.runners_meta) else None)

            # rankings & implied prob
            rk_ltp = rank_ltp.get(sid)
            rk_bb  = rank_back.get(sid)
            implied = None
            base_price = bb or lpt
            if base_price and base_price > 0:
                implied = 1.0 / base_price

            # deltas & vol
            d5  = self._delta_since(market_id, sid, 5.0)
            d30 = self._delta_since(market_id, sid, 30.0)
            vol = self._volatility(market_id, sid, 60.0)

            # liquidity score simple: somme des tailles top-of-book
            liq_score = (bs or 0.0) + (ls or 0.0)

            # place théorique — placeholder (à affiner quand PLACE market link + règles)
            place_theorique = None
            if implied is not None and mie and (mie.n_places or (mie.place_market_id and mie.win_market_id)):
                # simple proxy: borne haute
                npl = mie.n_places or 0
                if npl > 0:
                    place_theorique = min(0.99, implied * npl)

            row = [
                ts, market_id, sid,
                (rm.runner_name if rm else None),
                status,
                (rm.draw if rm and rm.draw else None),
                (rm.trap if rm else None),
                (rm.sort_priority if rm else None),
                lpt, bb, bs, bl, ls,
                ladder_b, ladder_l, total_matched_runner,
                rk_ltp, rk_bb, implied,
                d5, d30, vol, liq_score,
                (rk_bb == 1 if rk_bb is not None else None),
                place_theorique,
                float(tto) if tto is not None else None,
                (mie.venue if mie else None),
                (mie.course_id if mie else None),
                milestone,
            ]
            self._append(self.runner_csv, row)
