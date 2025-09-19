# src/dogbot/executor.py
# Snapshots WIN + PLACE, jalons 300/150/80/45/2
# - BSP_WIN / BSP_PLACE (priorité: SP_EST si dispo, sinon NEAR) + SP_AVAILABLE_* (1/0)
# - WINPROB=(BSP_WIN+LTP_WIN)/2 ; PLACEPROB=(BSP_PLACE+LTP_PLACE)/2
# - MID/MOYLTP avec seuil de confiance MOYLTP_TOLERANCE_PCT (%.env)
# - DIFF/MOM sur BASE_WIN = (WINPROB -> MOYLTP_WIN -> LTP_WIN -> BEST_BACK)
# - PLACETHEORIQUE = cote (1/q) via Plackett–Luce (Top-K) à partir des prix WIN
# - Gap features @ T−2s (WIN/LTP) : GOR + bornes de binning GAPMIN/GAPMAX ; duplication vers PLACE
# - d5/d30/vol (WIN) : variations micro (5s, 30s) + volatilité 60s à partir de l'historique LTP

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Dict, List, Tuple, Set
from collections import defaultdict, deque
import csv, math, os, re

from .types import Instruction, RunnerMeta
from .indexer import MarketIndex
from .feeds import create_price_feed_from_env

# --- import Plackett–Luce ---
try:
    from .plackett import odds_to_win_probs, place_probabilities, fair_place_odds
except Exception:
    from .placket import odds_to_win_probs, place_probabilities, fair_place_odds  # type: ignore

# --- Staking/Stratégies ---
from .config import load_config
from .staking import StakingEngine, Side
from .strategies import build_registry, try_fire_slot, RunnerCtx, ExecMode, _region_from_book

# --- LIVE ---
from .execution.orders import OrderExecutor
from .risk import ExposureManager


# ---------- utilitaires ----------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _tz_utc(dt: Optional[datetime]):
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

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
    for x in ladder[:n]:
        try:
            items.append(f"{x.price}:{x.size}")
        except Exception:
            p = x.get("price") if isinstance(x, dict) else None
            s = x.get("size") if isinstance(x, dict) else None
            if p is None or s is None:
                continue
            items.append(f"{p}:{s}")
    return "|".join(items) if items else None

def _runner_lpt_or_back(r) -> Optional[float]:
    lpt = getattr(r, "last_price_traded", None)
    try:
        if lpt is not None:
            return float(lpt)
    except Exception:
        pass
    ex = getattr(r, "ex", None)
    pb, _ = _best_at_side(ex, "BACK")
    return pb

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
    s = 0.0; k = 0
    for r in (runners or []):
        pb = _best_at_side(getattr(r, "ex", None), "BACK")[0]
        if pb and pb > 1e-9:
            s += 1.0 / pb; k += 1
    return (s * 100.0) if k else None

def _overround_lay(runners) -> Optional[float]:
    s = 0.0; k = 0
    for r in (runners or []):
        pl = _best_at_side(getattr(r, "ex", None), "LAY")[0]
        if pl and pl > 1e-9:
            s += 1.0 / pl; k += 1
    return (s * 100.0) if k else None

def _spread_pct(runners) -> Optional[float]:
    vals = []
    for r in (runners or []):
        pb = _best_at_side(getattr(r, "ex", None), "BACK")[0]
        pl = _best_at_side(getattr(r, "ex", None), "LAY")[0]
        if pb and pl and pb > 0:
            mid = (pb + pl) / 2.0
            if mid > 0:
                vals.append((pl - pb) / mid)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals)//2] * 100.0

def _rankings(runners) -> tuple[Dict[int,int], Dict[int,int]]:
    arr_ltp, arr_bb = [], []
    for r in (runners or []):
        sid = getattr(r, "selection_id", None)
        if sid is None: continue
        lpt = _runner_lpt_or_back(r)
        ex = getattr(r, "ex", None)
        pb, _ = _best_at_side(ex, "BACK")
        if lpt is not None: arr_ltp.append((sid, lpt))
        if pb is not None:  arr_bb.append((sid, pb))
    arr_ltp.sort(key=lambda t: t[1])
    arr_bb.sort(key=lambda t: t[1])
    return ({sid:i+1 for i,(sid,_) in enumerate(arr_ltp)},
            {sid:i+1 for i,(sid,_) in enumerate(arr_bb)})

# -------- TRAP parsing --------
_TRAP_PATTS = [
    re.compile(r"^\s*(?:TRAP|T)\s*([1-9]\d?)\b", re.I),
    re.compile(r"^\s*([1-9]\d?)\s*[.\-)\s]"),
    re.compile(r"\(\s*(?:TRAP|T)\s*([1-9]\d?)\s*\)", re.I),
]
def _parse_trap(meta: Optional[RunnerMeta], runner_name: Optional[str]) -> Optional[int]:
    for v in [getattr(meta, "trap", None), getattr(meta, "draw", None)]:
        if v is None: continue
        s = str(v).strip()
        m = re.match(r"^[^\d]*([1-9]\d?)", s)
        if m:
            try: return int(m.group(1))
            except Exception: pass
    if runner_name:
        for patt in _TRAP_PATTS:
            m = patt.search(runner_name)
            if m:
                try: return int(m.group(1))
                except Exception: continue
    return None

class _MetaStub:
    def __init__(self, runner_name=None, draw=None, sort_priority=None, trap=None):
        self.runner_name = runner_name
        self.draw = draw
        self.sort_priority = sort_priority
        self.trap = trap


class Executor:
    MILESTONES = [300, 150, 80, 45, 2]
    TOLERANCE_S = 1.5

    # Bins GOR (gauche fermée / droite ouverte) pour GapMin/GapMax sur LTP @ T−2s
    GOR_BINS = [1.00, 1.10, 1.25, 1.50, 2.00, 3.00, float("inf")]

    MARKET_HEADER = [
        "SNAP_TS_UTC","MARKET_ID","COURSE_ID","MARKET_TYPE",
        "EVENT_ID","EVENT_NAME","VENUE","COUNTRY_CODE",
        "MARKET_START_TIME_UTC","SECONDS_TO_OFF",
        "INPLAY","MARKET_STATUS","NUMBER_OF_WINNERS","N_RUNNERS_ACTIVE","MARKET_TOTAL_MATCHED",
        "BACK_OVERROUND","LAY_OVERROUND","SPREAD_PCT",
        "WIN_MARKET_ID","PLACE_MARKET_ID","COURSE_LINK_OK","N_PLACES",
        "RULE_OK_TIME_WINDOW","RULE_OK_PRICE_BOUNDS","RULE_OK_LIQUIDITY","RULE_OK_OVERROUND","RULES_ALL_OK",
        "STRATEGY_NAME","SIGNAL","SIDE","TARGET_PRICE","STAKE","PERSISTENCE","HEDGE_TICKS","STOP_TICKS",
        "EST_LIABILITY","REASON_CODE","DRYRUN_WOULD_PLACE",
        "MILESTONE_S",
        "FETCH_LATENCY_MS","CACHE_AGE_S","RETRY_COUNT","THROTTLE_WEIGHT","CODE_VERSION",
    ]

    RUNNER_HEADER = [
        
        "SNAP_TS_UTC",
        "COURSE_ID",
        "VENUE",
        "MARKET_START_TIME_UTC",
        "SECONDS_TO_OFF",
        "MILESTONE_S",
        "WIN_MARKET_ID",
        "PLACE_MARKET_ID",
        "MARKET_ID",
        "MARKET_TYPE",
        "SELECTION_ID",
        "RUNNER_NAME",
        "RUNNER_STATUS",
        "DRAW",
        "TRAP",
        "VIRTUAL_TRAP",
        "SORT_PRIORITY",
        "N_RUNNERS_ACTIVE",
        "N_PLACES",
        "LTP_WIN",
        "BEST_BACK_PRICE_1_WIN",
        "BEST_BACK_SIZE_1_WIN",
        "BEST_LAY_PRICE_1_WIN",
        "BEST_LAY_SIZE_1_WIN",
        "MID_WIN",
        "MOYLTP_WIN",
        "BACK_LADDER_WIN",
        "LAY_LADDER_WIN",
        "RUNNER_TOTAL_MATCHED_WIN",
        "FAV_RANK_LTP_WIN",
        "FAV_RANK_BACK_WIN",
        "WIN_IMPLIED_PROB_WIN",
        "BSP_WIN",
        "SP_AVAILABLE_WIN",
        "WINPROB",
        "LTP_300_WIN",
        "LTP_150_WIN",
        "LTP_80_WIN",
        "LTP_45_WIN",
        "LTP_2_WIN",
        "DIFF150_300_WIN",
        "DIFF80_150_WIN",
        "DIFF45_80_WIN",
        "MOM45_WIN",
        "MOM80_WIN",
        "MOM150_WIN",
        "MOM300_WIN",
        "PRICE_DELTA_5S_WIN",
        "PRICE_DELTA_30S_WIN",
        "VOLATILITY_60S_WIN",
        "LIQUIDITY_SCORE_WIN",
        "IS_FAVOURITE_WIN",
        "PLACETHEORIQUE",
        "LTP_PLACE",
        "BEST_BACK_PRICE_1_PLACE",
        "BEST_BACK_SIZE_1_PLACE",
        "BEST_LAY_PRICE_1_PLACE",
        "BEST_LAY_SIZE_1_PLACE",
        "MID_PLACE",
        "MOYLTP_PLACE",
        "BACK_LADDER_PLACE",
        "LAY_LADDER_PLACE",
        "RUNNER_TOTAL_MATCHED_PLACE",
        "BSP_PLACE",
        "SP_AVAILABLE_PLACE",
        "PLACEPROB",
        "PLACETHEORIQUE_PLACE",
        "LTP_300_PLACE",
        "LTP_150_PLACE",
        "LTP_80_PLACE",
        "LTP_45_PLACE",
        "LTP_2_PLACE",
        "GAPMIN",
        "GAPMAX",
        "GOR",
        "WINTRADE",
    
    ]

    ENTRY_MIN_T_S = 120
    ENTRY_MAX_T_S = 7200
    PRICE_MIN = 1.30
    PRICE_MAX = 12.0
    MIN_LIQUIDITY_MARKET = 0.0

    def __init__(self, client: Any, strategy, market_index: MarketIndex, dry_run: bool = True, data_dir: str = "./data"):
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

        self._next_ms: Dict[str, List[int]] = defaultdict(lambda: list(self.MILESTONES))
        self._last_tto: Dict[str, float] = {}

        # Milestones:
        self._base_win_ms: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
        self._ltp_place_ms: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))

        # Historique LTP (pour d5/d30/vol)
        self._hist: Dict[str, Dict[int, deque[Tuple[float,float]]]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=600)))

        # Cache PLACETHEORIQUE -> duplication PLACE
        self._last_place_theo_by_market: Dict[str, Dict[int, float]] = defaultdict(dict)

        # Gap bounds & ratio @ T−2s par WIN market
        self._gap_bounds_by_win: Dict[str, Dict[int, Tuple[Optional[float], Optional[float]]]] = defaultdict(dict)
        self._gor_by_win: Dict[str, Dict[int, Optional[float]]] = defaultdict(dict)

        # BSP / SP feed
        self.price_feed = create_price_feed_from_env()

        # Diag
        self._diag_fetch_latency_ms: Optional[float] = None
        self._diag_retry_count: Optional[float] = None
        self._diag_throttle_weight: Optional[float] = None
        self._code_version: Optional[str] = os.environ.get("CODE_VERSION")

        # Tolérance MOYLTP
        try:
            self._moyltp_tol = float(os.getenv("MOYLTP_TOLERANCE_PCT", "30"))
        except Exception:
            self._moyltp_tol = 30.0

        # Staking/strats
        self.cfg = load_config()
        self.staking_engine = StakingEngine(self.cfg)
        self.strategy_registry = build_registry()
        self.trades_dir = self.data_dir
        self._slot_market_fired: Set[tuple[str,int,str]] = set()

        # LIVE
        self.order_executor = OrderExecutor(
            client=self.client,
            data_dir=str(self.data_dir),
            throttle_max_per_minute=int(os.getenv("THROTTLE_MAX_PER_MIN", "25")),
            default_persistence=os.getenv("PERSISTENCE", "LAPSE"),
            fok_ms=int(os.getenv("FOK_MS", "0")) or None,
        )
        self.exposure = ExposureManager(self.cfg)

    def set_diagnostics(self, fetch_latency_ms: Optional[float] = None, retry_count: Optional[int] = None,
                        throttle_weight: Optional[float] = None, code_version: Optional[str] = None) -> None:
        if fetch_latency_ms is not None:
            self._diag_fetch_latency_ms = float(fetch_latency_ms)
        if retry_count is not None:
            self._diag_retry_count = int(retry_count)
        if throttle_weight is not None:
            self._diag_throttle_weight = float(throttle_weight)
        if code_version is not None:
            self._code_version = str(code_version)

    def _ensure_header(self, path: Path, header: Iterable[str]) -> None:
        header_list = list(header)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    first = f.readline().strip()
                current = first.split(",")
                if current != header_list:
                    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    backup = path.with_name(path.stem + f"_old_{ts}" + path.suffix)
                    path.rename(backup)
                    with path.open("w", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(header_list)
                    return
                else:
                    return
            except Exception:
                pass
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header_list)

    def _append(self, path: Path, row: Iterable[Any]) -> None:
        with path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(list(row))

    def _mindex_vals(self) -> List[Any]:
        for try_fn in (lambda i: list(i.values()),
                       lambda i: [v for _, v in i.items()],
                       lambda i: list(i),):
            try: return try_fn(self.market_index)  # type: ignore
            except Exception: pass
        for attr in ("by_id","_by_id","map","_map","d","_d"):
            d = getattr(self.market_index, attr, None)
            if isinstance(d, dict): return list(d.values())
        return []

    def _resolve_link_ids(self, current_market_id: str, mie: Any, md: Any, market_type_hint: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        win_id = getattr(mie, "win_market_id", None) if mie is not None else None
        place_id = getattr(mie, "place_market_id", None) if mie is not None else None
        if win_id and place_id:
            return win_id, place_id

        cur_mt = (getattr(md, "market_type", None) or market_type_hint or "").upper()
        if cur_mt not in ("WIN","PLACE"):
            if getattr(md, "number_of_winners", None) or getattr(md, "numberOfWinners", None):
                nw = int(getattr(md, "number_of_winners", None) or getattr(md, "numberOfWinners", 1))
                cur_mt = "PLACE" if nw >= 2 else "WIN"

        ev_id = getattr(getattr(mie, "event", None), "id", None) if mie is not None else None
        start = _tz_utc(getattr(mie, "market_start_time", None)) if mie is not None else None
        if start is None:
            start = _tz_utc(getattr(md, "market_time", None))

        if ev_id and start:
            for other in self._mindex_vals():
                omid = getattr(other, "market_id", None)
                if not omid or str(omid) == str(current_market_id):
                    continue
                oe = getattr(other, "event", None)
                if getattr(oe, "id", None) != ev_id:
                    continue
                ostart = _tz_utc(getattr(other, "market_start_time", None))
                if ostart and abs((ostart - start).total_seconds()) > 180:
                    continue
                omt = (getattr(other, "market_type", None) or "").upper()
                if omt not in ("WIN","PLACE"):
                    name2 = getattr(other, "market_name", None) or getattr(other, "marketName", None)
                    omt = "PLACE" if (name2 and "place" in name2.lower()) else "WIN"
                if omt == "WIN" and not win_id: win_id = omid
                if omt == "PLACE" and not place_id: place_id = omid

        if cur_mt == "WIN" and not win_id:
            win_id = current_market_id
        if cur_mt == "PLACE" and not place_id:
            place_id = current_market_id

        return (str(win_id) if win_id else None, str(place_id) if place_id else None)

    # --------- binning GOR -> (gapmin, gapmax), avec bornes [a,b) ---------
    def _gap_bounds_from_gor(self, gor: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        if gor is None or gor != gor:  # NaN
            return (None, None)
        x = float(gor)
        bins = self.GOR_BINS
        for i in range(len(bins)-1):
            a, b = bins[i], bins[i+1]
            if x >= a and (x < b or math.isinf(b)):
                return (float(a), (float(b) if not math.isinf(b) else float("inf")))
        return (float(bins[-2]), float("inf"))

    # ---------- cœur ----------
    def process_book(self, book: Any) -> None:
        market_id = str(getattr(book, "market_id", ""))
        md = getattr(book, "market_definition", None)
        runners = getattr(book, "runners", None) or []

        try:
            mie = self.market_index.get(market_id)  # type: ignore
        except Exception:
            mie = None

        # Infos catalogue
        info = self._extract_catalogue_info(mie, md)
        market_type = info["market_type"]
        start_utc = info["start_utc"]
        now = datetime.now(timezone.utc)
        t_to_off_s = (start_utc - now).total_seconds() if start_utc else None

        if getattr(book, "inplay", False):
            self._next_ms.pop(market_id, None)
            self._last_tto[market_id] = t_to_off_s if t_to_off_s is not None else 0.0
            return

        # Historique LTP pour d5/d30/vol (WIN uniquement)
        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None: continue
            lpt = _runner_lpt_or_back(r)
            if lpt is not None:
                self._hist[market_id][int(sid)].append((datetime.now(timezone.utc).timestamp(), float(lpt)))

        rank_ltp, rank_back = _rankings(runners)
        milestone = self._milestone_due(market_id, t_to_off_s)

        self._write_market_row(book, info, market_type, t_to_off_s, milestone, runners)
        if milestone is not None:
            self._write_runner_rows(book, info, market_type, t_to_off_s, milestone, runners, rank_ltp, rank_back)

        # stratégie legacy
        try:
            instructions = self.strategy.decide_all(book, mie, datetime.now(timezone.utc)) or []
        except Exception as e:
            print(f"[STRATEGY_ERR] {market_id}: {e}")
            instructions = []

        if instructions:
            if self.dry_run:
                for ins in instructions:
                    print("[DRY] would place", ins.asdict(), "on", market_id)
            else:
                for ins in instructions:
                    print("[LIVE] placing", ins.asdict(), "on", market_id)

        if t_to_off_s is not None:
            self._last_tto[market_id] = t_to_off_s

    def _write_market_row(self, book, info: Dict[str,Any], market_type: Optional[str],
                          tto: Optional[float], milestone: Optional[int], runners) -> None:
        md = getattr(book, "market_definition", None)
        back_over = _overround_back(runners)
        lay_over  = _overround_lay(runners)
        spread    = _spread_pct(runners)

        mie_current = getattr(self.market_index, "get", lambda _:_)(getattr(book,"market_id",None))
        win_id, place_id = self._resolve_link_ids(str(getattr(book,"market_id","")), mie_current, md, market_type)

        row = [
            _now_utc_iso(),
            str(getattr(book, "market_id", "")),
            None,
            market_type,
            info["event_id"],
            info["event_name"],
            info["venue"],
            info["country_code"],
            (info["start_utc"].isoformat().replace("+00:00","Z") if info["start_utc"] else None),
            float(tto) if tto is not None else None,
            bool(getattr(book, "inplay", False)),
            _market_status(md),
            _num_winners(md),
            _active_count(runners),
            float(getattr(book, "total_matched", None) or 0.0),
            back_over, lay_over, spread,
            win_id, place_id,
            (1 if (win_id and place_id) else 0),
            (_num_winners(md) if (_num_winners(md) is not None) else (3 if _active_count(runners) >= 8 else 2)),
            (tto is not None and self.ENTRY_MIN_T_S <= tto <= self.ENTRY_MAX_T_S),
            self._fav_price_ok(runners),
            (float(getattr(book, "total_matched", 0.0)) >= self.MIN_LIQUIDITY_MARKET),
            (back_over is not None and lay_over is not None),
            None,
            (getattr(self.strategy, "name", None) or None),
            None, None, None, None, None, None, None, None,
            False,
            milestone,
            self._diag_fetch_latency_ms,
            None,
            self._diag_retry_count,
            self._diag_throttle_weight,
            self._code_version,
        ]
        self._append(self.market_csv, row)

    def _fav_price_ok(self, runners) -> Optional[bool]:
        ex_prices = []
        for r in (runners or []):
            pb = _best_at_side(getattr(r, "ex", None), "BACK")[0]
            if pb:
                ex_prices.append(pb)
        if not ex_prices:
            return None
        fav_price = min(ex_prices)
        return (self.PRICE_MIN <= fav_price <= self.PRICE_MAX)

    def _trusted_moyltp(self, ltp: Optional[float], mid: Optional[float]) -> Optional[float]:
        if ltp is None or mid is None or ltp <= 0:
            return None
        gap = abs(mid - ltp) / ltp * 100.0
        return ((ltp + mid) / 2.0) if (gap <= self._moyltp_tol) else None

    def _write_runner_rows(self, book, info: Dict[str,Any], market_type: Optional[str], tto: Optional[float], milestone: int,
                           runners, rank_ltp: Dict[int,int], rank_back: Dict[int,int]) -> None:
        market_id = str(getattr(book, "market_id", ""))
        n_active = _active_count(runners)
        n_places = (_num_winners(getattr(book, "market_definition", None))
                    or (3 if n_active >= 8 else 2))

        meta_by_sid: Dict[int, RunnerMeta] = info.get("meta_by_sid", {}) or {}
        vmap = self._compute_virtual_traps(runners, meta_by_sid)

        mie_current = getattr(self.market_index, "get", lambda _:_)(market_id)
        win_id, place_id = self._resolve_link_ids(market_id, mie_current, getattr(book,"market_definition",None), market_type)

        is_win  = (market_type == "WIN")
        is_place = (market_type == "PLACE")

        # Prépare prix + caches
        sid_prices_for_pl: List[Tuple[int,float]] = []  # (sid, BASE_WIN)
        cache: Dict[int, Dict[str, Optional[float]]] = {}

        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None: continue
            sid = int(sid)

            lpt = _runner_lpt_or_back(r)
            ex = getattr(r, "ex", None)
            bb, _ = _best_at_side(ex, "BACK")
            bl, _ = _best_at_side(ex, "LAY")

            mid = ((bb + bl) / 2.0) if (bb is not None and bl is not None) else None
            moyltp = self._trusted_moyltp(lpt, mid)

            # SP / BSP via feed (REST: SP_AVAILABLE; STREAM: SP_EST)
            tmp = {}
            self.price_feed.enrich_runner_row(tmp, "WIN" if is_win else "PLACE", r)
            if is_win:
                sp_est = tmp.get("SP_EST_WIN")
                near   = tmp.get("NEAR_SP_WIN")
                sp_av  = 1 if tmp.get("SP_AVAILABLE_WIN") else 0
                bsp    = sp_est if (sp_est is not None) else near
                bsp_win = float(bsp) if bsp is not None else None
                sp_av_win = sp_av
                bsp_place = None
                sp_av_place = 0
            else:
                sp_est = tmp.get("SP_EST_PLACE")
                near   = tmp.get("NEAR_SP_PLACE")
                sp_av  = 1 if tmp.get("SP_AVAILABLE_PLACE") else 0
                bsp    = sp_est if (sp_est is not None) else near
                bsp_place = float(bsp) if bsp is not None else None
                sp_av_place = sp_av
                bsp_win = None
                sp_av_win = 0

            # Cache par type
            cache[sid] = {
                "LTP": lpt, "BB": bb, "BL": bl,
                "MID": mid,
                "MOYLTP": moyltp,
                "BSP_WIN": bsp_win,
                "SP_AV_WIN": float(sp_av_win),
                "BSP_PLACE": bsp_place,
                "SP_AV_PLACE": float(sp_av_place),
            }

            # BASE_WIN pour Plackett-Luce (uniquement WIN)
            if is_win:
                winprob = ((bsp_win + lpt)/2.0) if (bsp_win is not None and lpt is not None) else None
                base_win_val = winprob or moyltp or lpt or bb
                cache[sid]["WINPROB"] = winprob
                cache[sid]["BASE_WIN"] = base_win_val
                if base_win_val and base_win_val > 1.0:
                    sid_prices_for_pl.append((sid, float(base_win_val)))

        # PLACETHEORIQUE via Plackett–Luce (cote = 1/q)
        place_theo_by_sid: Dict[int, Optional[float]] = {}
        if is_win and len(sid_prices_for_pl) >= max(2, min(5, n_active)):
            sid_prices_for_pl.sort(key=lambda t: t[0])
            sids = [sid for sid,_ in sid_prices_for_pl]
            odds = [price for _,price in sid_prices_for_pl]
            try:
                p = odds_to_win_probs(odds, beta=1.0)
                q = place_probabilities(p, K=n_places)
                fair = fair_place_odds(q)
                fair_list = list(fair) if not hasattr(fair, "tolist") else fair.tolist()
                for sid, odd_place in zip(sids, fair_list):
                    place_theo_by_sid[sid] = float(odd_place)
            except Exception:
                place_theo_by_sid = {sid: None for sid,_ in sid_prices_for_pl}
            cache_key = (win_id or market_id)
            if cache_key:
                self._last_place_theo_by_market[str(cache_key)] = {
                    sid: v for sid, v in place_theo_by_sid.items() if v is not None
                }

        # ------ Gap @ T−2s (WIN/LTP): calcule GOR + GAPMIN/GAPMAX, et duplique vers PLACE ------
        gapmap: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
        gormap: Dict[int, Optional[float]] = {}
        if is_win and milestone == 2:
            pairs: List[Tuple[int, float]] = []
            for r in runners:
                if (getattr(r, "status", None) or "ACTIVE").upper() != "ACTIVE":
                    continue
                sid = int(getattr(r, "selection_id", 0) or 0)
                if sid == 0:
                    continue
                ltp_val = cache.get(sid, {}).get("LTP")
                if ltp_val is not None and ltp_val > 1.0:
                    pairs.append((sid, float(ltp_val)))
            pairs.sort(key=lambda t: t[1])  # tri croissant
            n = len(pairs)
            for i, (sid, price) in enumerate(pairs):
                if i + 1 < n and price > 0:
                    gor = pairs[i+1][1] / price
                    gapmin, gapmax = self._gap_bounds_from_gor(gor)
                else:
                    gor = None
                    gapmin, gapmax = (None, None)
                gormap[sid] = (float(gor) if gor is not None else None)
                gapmap[sid] = (gapmin, gapmax)
            self._gor_by_win[str(market_id)] = gormap
            self._gap_bounds_by_win[str(market_id)] = gapmap

        # Écriture des lignes runners
        rank_ltp, rank_back = _rankings(runners)
        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None: continue
            sid = int(sid)
            status = getattr(r, "status", None)
            rm: RunnerMeta | None = meta_by_sid.get(sid)
            runner_name = (rm.runner_name if rm else None)
            trap = _parse_trap(rm, runner_name)
            vtrap = vmap.get(sid, trap)

            lpt = cache[sid].get("LTP")
            ex = getattr(r, "ex", None)
            bb, bs = _best_at_side(ex, "BACK")
            bl, ls = _best_at_side(ex, "LAY")
            mid = cache[sid].get("MID")
            moyltp = cache[sid].get("MOYLTP")

            ladder_b = _compress_ladder(getattr(ex, "available_to_back", None), 3)
            ladder_l = _compress_ladder(getattr(ex, "available_to_lay", None), 3)
            total_matched_runner = getattr(r, "total_matched", None)

            rk_ltp = rank_ltp.get(sid)
            rk_bb  = rank_back.get(sid)
            implied = (1.0 / (bb or lpt)) if (bb or lpt) else None

            bsp_win = cache[sid].get("BSP_WIN") if is_win else None
            sp_av_win = cache[sid].get("SP_AV_WIN") if is_win else None
            bsp_place = cache[sid].get("BSP_PLACE") if is_place else None
            sp_av_place = cache[sid].get("SP_AV_PLACE") if is_place else None

            winprob = cache[sid].get("WINPROB") if is_win else None
            base_win_val = cache[sid].get("BASE_WIN") if is_win else None

            # Milestones:
            if is_win and base_win_val is not None:
                self._base_win_ms[market_id][sid][milestone] = float(base_win_val)
            if is_place and lpt is not None:
                self._ltp_place_ms[market_id][sid][milestone] = float(lpt)

            def _get(msdict, ms):
                return msdict.get(market_id, {}).get(sid, {}).get(ms)
            def ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
                if a is None or b is None or b == 0: return None
                return (a / b) - 1.0

            base_300 = _get(self._base_win_ms, 300) if is_win else None
            base_150 = _get(self._base_win_ms, 150) if is_win else None
            base_80  = _get(self._base_win_ms, 80)  if is_win else None
            base_45  = _get(self._base_win_ms, 45)  if is_win else None
            base_2   = _get(self._base_win_ms, 2)   if is_win else None

            diff150_300 = ratio(base_150, base_300) if is_win else None
            diff80_150  = ratio(base_80,  base_150) if is_win else None
            diff45_80   = ratio(base_45,  base_80)  if is_win else None
            mom45  = ratio(base_2,  base_45) if is_win else None
            mom80  = ratio(base_2,  base_80)  if is_win else None
            mom150 = ratio(base_2,  base_150) if is_win else None
            mom300 = ratio(base_2,  base_300) if is_win else None

            # PLACETHEORIQUE (cote) :
            place_theo_win = place_theo_by_sid.get(sid) if is_win else None
            place_theo_place = None
            if is_place and win_id:
                cached = self._last_place_theo_by_market.get(str(win_id))
                if cached is not None:
                    place_theo_place = cached.get(sid)

            # Milestones PLACE LTP
            def _get_place(ms):
                return self._ltp_place_ms.get(market_id, {}).get(sid, {}).get(ms)
            ltp_300_p = _get_place(300) if is_place else None
            ltp_150_p = _get_place(150) if is_place else None
            ltp_80_p  = _get_place(80)  if is_place else None
            ltp_45_p  = _get_place(45)  if is_place else None
            ltp_2_p   = _get_place(2)   if is_place else None

            # --- d5, d30, vol (WIN seulement) ---
            d5 = d30 = vol = None
            if is_win:
                hist = self._hist.get(market_id, {}).get(sid)
                if hist and len(hist) >= 2:
                    now_ts, now_p = hist[-1]
                    # prix il y a 5s / 30s
                    p5 = p30 = None
                    for ts, p in reversed(hist):
                        dt = now_ts - ts
                        if p5 is None and dt >= 5.0:
                            p5 = p
                        if p30 is None and dt >= 30.0:
                            p30 = p
                            # on peut sortir si on a déjà p30
                            if p5 is not None:
                                break
                    if p5 and p5 > 0:
                        d5 = (now_p / p5) - 1.0
                    if p30 and p30 > 0:
                        d30 = (now_p / p30) - 1.0
                    # vol 60s : écart-type des retours sur ~60s
                    rets = []
                    prev_ts, prev_p = None, None
                    for ts, p in reversed(hist):
                        if now_ts - ts > 60.0:
                            break
                        if prev_p is not None and prev_p > 0 and p > 0:
                            rets.append((p / prev_p) - 1.0)
                        prev_ts, prev_p = ts, p
                    if len(rets) >= 2:
                        m = sum(rets) / len(rets)
                        var = sum((x - m) ** 2 for x in rets) / (len(rets) - 1)
                        vol = var ** 0.5

            # --- GapMin/GapMax/GOR au jalon 2s ; duplication sur PLACE ---
            gapmin = gapmax = gor_val = None
            if milestone == 2:
                if is_win:
                    gapmin, gapmax = self._gap_bounds_by_win.get(str(market_id), {}).get(sid, (None, None))
                    gor_val = self._gor_by_win.get(str(market_id), {}).get(sid)
                else:
                    if win_id:
                        gapmin, gapmax = self._gap_bounds_by_win.get(str(win_id), {}).get(sid, (None, None))
                        gor_val = self._gor_by_win.get(str(win_id), {}).get(sid)

            row = [
                # base
                _now_utc_iso(),
                None,
                info["venue"],
                (info["start_utc"].isoformat().replace("+00:00","Z") if info["start_utc"] else None),
                float(tto) if tto is not None else None,
                milestone,
                win_id, place_id,
                market_id, (market_type or ""),
                sid, runner_name, status,
                (getattr(rm, "draw", None) if rm else None),
                trap, vtrap,
                (getattr(rm, "sort_priority", None) if rm else None),
                n_active, n_places,
                # WIN block
                (lpt if is_win else None),
                (bb if is_win else None),
                (bs if is_win else None),
                (bl if is_win else None),
                (ls if is_win else None),
                (mid if is_win else None),
                (moyltp if is_win else None),
                (ladder_b if is_win else None),
                (ladder_l if is_win else None),
                (total_matched_runner if is_win else None),
                (rank_ltp.get(sid) if is_win else None),
                (rank_back.get(sid) if is_win else None),
                ((1.0 / (bb or lpt)) if (is_win and (bb or lpt)) else None),
                (bsp_win if is_win else None),
                (sp_av_win if is_win else None),
                (winprob if is_win else None),
                (_get(self._base_win_ms, 300) if is_win else None),
                (_get(self._base_win_ms, 150) if is_win else None),
                (_get(self._base_win_ms, 80)  if is_win else None),
                (_get(self._base_win_ms, 45)  if is_win else None),
                (_get(self._base_win_ms, 2)   if is_win else None),
                (diff150_300 if is_win else None),
                (diff80_150  if is_win else None),
                (diff45_80   if is_win else None),
                (mom45  if is_win else None),
                (mom80  if is_win else None),
                (mom150 if is_win else None),
                (mom300 if is_win else None),
                (d5 if is_win else None),
                (d30 if is_win else None),
                (vol if is_win else None),
                ((bs or 0.0) + (ls or 0.0) if is_win else None),
                (((rank_back.get(sid) or 0) == 1) if is_win else None),
                (place_theo_win if is_win else None),
                # PLACE block
                (lpt if is_place else None),
                (bb if is_place else None),
                (bs if is_place else None),
                (bl if is_place else None),
                (ls if is_place else None),
                (mid if is_place else None),
                (moyltp if is_place else None),
                (ladder_b if is_place else None),
                (ladder_l if is_place else None),
                (total_matched_runner if is_place else None),
                (bsp_place if is_place else None),
                (sp_av_place if is_place else None),
                (((bsp_place + lpt)/2.0) if (is_place and bsp_place is not None and lpt is not None) else None),
                (place_theo_place if is_place else None),
                (ltp_300_p if is_place else None),
                (ltp_150_p if is_place else None),
                (ltp_80_p  if is_place else None),
                (ltp_45_p if is_place else None),
                (ltp_2_p  if is_place else None),
                # NEW:
                gapmin, gapmax, gor_val,
            ]
            # ----- WINTRADE (intermediate): MOYLTP_WIN if present else LTP_WIN -----
            _wintrade = None
            if is_win:
                _wintrade = (moyltp if (moyltp is not None) else lpt)
            # add as final CSV column (WIN rows only)
            row.append(_wintrade if is_win else None)

            self._append(self.runner_csv, row)

            # --- staking + LIVE ---
            try:
                start_utc = info.get("start_utc")
                venue = (info.get("venue") or "NA")
                if start_utc is not None:
                    course_id = f"{venue}-{start_utc:%Y%m%d%H%M}"
                else:
                    course_id = str(info.get("event_id") or market_id)

                ltp_val = cache.get(sid, {}).get("LTP")
                if ltp_val is not None and ltp_val >= 1.01:
                    # mom45_place pour PLACE (LTP 45s -> 2s)
                    mom45p = ((ltp_2_p / ltp_45_p) - 1.0) if (is_place and ltp_2_p and ltp_45_p and ltp_45_p != 0) else None

                    # Construit un RunnerCtx enrichi (pour stratégies)
                    ctx = RunnerCtx(
                        market_id=market_id,
                        market_type=(market_type or ""),
                        selection_id=int(sid),
                        course_id=str(course_id),
                        ltp=float(ltp_val),
                        milestone=milestone,
                        secs_to_off=float(tto) if tto is not None else None,
                        # enriched fields for systems/HYB:
                        trap=trap,
                        fav_rank_ltp=(rank_ltp.get(sid) if rank_ltp else rk_ltp),
                        fav_rank_back=(rank_back.get(sid) if rank_back else rk_bb),
                        gor=(self._gor_by_win.get(str(win_id or market_id), {}).get(sid) if milestone == 2 else None),
                        mom45=(mom45 if is_win else None),
                        mom45_place=(mom45p if is_place else None),
                        base_win=(base_2 if is_win else None) or base_win_val,
                        bb=bb,
                        bl=bl,
                    )
                    try:
                        ctx.region = _region_from_book(book)
                    except Exception:
                        ctx.region = None

                    for slot in self.strategy_registry:
                        key = (slot.family, slot.slot, market_id)
                        if getattr(slot, "bet_per_market", False) and key in self._slot_market_fired:
                            continue

                        res = try_fire_slot(self.staking_engine, slot, ctx)
                        if not res:
                            continue

                        status = "DRYRUN" if self.dry_run else "LIVE"
                        self._log_trade_row({
                            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                            "market_id": market_id,
                            "market_type": (market_type or ""),
                            "selection_id": int(sid),
                            "course_id": str(course_id),
                            "side": slot.side.value,
                            "price_req": res.price,
                            "size_req": res.size,
                            "liability": round(res.liability or 0.0, 2),
                            "strategy": slot.tag,
                            "status": status,
                            "reason": res.reason,
                        })

                        if getattr(slot, "bet_per_market", False):
                            self._slot_market_fired.add(key)

                        if not self.dry_run:
                            can, reason = self.exposure.can_place(
                                Side(slot.side.value),
                                market_id=market_id,
                                selection_id=int(sid),
                                planned_stake=float(res.size),
                                planned_liability=float(res.liability or 0.0),
                            )
                            if not can:
                                self._log_trade_row({
                                    "ts": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                                    "market_id": market_id,
                                    "market_type": (market_type or ""),
                                    "selection_id": int(sid),
                                    "course_id": str(course_id),
                                    "side": slot.side.value,
                                    "price_req": res.price,
                                    "size_req": res.size,
                                    "liability": round(res.liability or 0.0, 2),
                                    "strategy": slot.tag,
                                    "status": "LIVE_BLOCKED",
                                    "reason": reason,
                                })
                            else:
                                # ---- utilise le mode effectif renvoyé par la stratégie (HYB, LIMIT, SP_MOC/LOC)
                                eff_mode = getattr(res, "exec_mode", slot.exec_mode)

                                orr = None
                                if eff_mode == ExecMode.LIMIT_LTP:
                                    idem_key = f"{market_id}:{sid}:{slot.tag}:{res.price}"
                                    orr = self.order_executor.place_limit(
                                        market_id=market_id,
                                        selection_id=int(sid),
                                        side=slot.side.value,
                                        price=float(res.price),
                                        size=float(res.size),
                                        strategy=slot.tag,
                                        persistence=os.getenv("PERSISTENCE", "LAPSE"),
                                        idem_key=idem_key,
                                        retries=int(os.getenv("ORDER_RETRIES", "2")),
                                        backoff_ms=int(os.getenv("ORDER_BACKOFF_MS", "250")),
                                    )
                                elif eff_mode == ExecMode.SP_MOC:
                                    qty = float(res.size) if slot.side == Side.BACK else float(res.liability or 0.0)
                                    idem_key = f"{market_id}:{sid}:{slot.tag}:SP_MOC"
                                    orr = self.order_executor.place_sp_market_on_close(
                                        market_id=market_id,
                                        selection_id=int(sid),
                                        side=slot.side.value,
                                        size_or_liability=qty,
                                        strategy=slot.tag,
                                        idem_key=idem_key,
                                        retries=int(os.getenv("ORDER_RETRIES", "2")),
                                        backoff_ms=int(os.getenv("ORDER_BACKOFF_MS", "250")),
                                    )
                                elif eff_mode == ExecMode.SP_LOC:
                                    sp_lim = getattr(res, "sp_limit", None) or getattr(slot, "sp_limit", None) or float(res.price)
                                    qty = float(res.size) if slot.side == Side.BACK else float(res.liability or 0.0)
                                    idem_key = f"{market_id}:{sid}:{slot.tag}:SP_LOC:{sp_lim}"
                                    orr = self.order_executor.place_sp_limit_on_close(
                                        market_id=market_id,
                                        selection_id=int(sid),
                                        side=slot.side.value,
                                        size_or_liability=qty,
                                        sp_limit_price=float(sp_lim),
                                        strategy=slot.tag,
                                        idem_key=idem_key,
                                        retries=int(os.getenv("ORDER_RETRIES", "2")),
                                        backoff_ms=int(os.getenv("ORDER_BACKOFF_MS", "250")),
                                    )

                                if orr and orr.ok:
                                    self.exposure.on_placed(
                                        Side(slot.side.value),
                                        market_id=market_id,
                                        selection_id=int(sid),
                                        stake=float(res.size),
                                        liability=float(res.liability or 0.0),
                                    )

            except Exception:
                pass

    # ----- helpers -----
    def _log_trade_row(self, row: dict) -> None:
        fname = self.trades_dir / f"trades_{datetime.now(timezone.utc):%Y%m%d}.csv"
        new_file = not fname.exists()
        with fname.open("a", newline="", encoding="utf-8") as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=[
                "ts","market_id","market_type","selection_id","course_id",
                "side","price_req","size_req","liability","strategy","status","reason"
            ])
            if new_file:
                w.writeheader()
            w.writerow(row)

    def _extract_catalogue_info(self, mie: Any, md: Any) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "event_id": None, "event_name": None, "venue": None, "country_code": None,
            "start_utc": None, "market_type": None, "market_name": None, "meta_by_sid": {},
        }
        info["market_name"] = getattr(mie, "market_name", None) or getattr(mie, "marketName", None)
        start = getattr(mie, "event_open_utc", None) or getattr(mie, "market_start_time", None)
        if start is None and md is not None:
            start = getattr(md, "market_time", None)
        info["start_utc"] = _tz_utc(start)
        ev = getattr(mie, "event", None)
        if ev is not None:
            info["event_id"] = getattr(ev, "id", None)
            info["event_name"] = getattr(ev, "name", None)
            info["venue"] = getattr(ev, "venue", None)
            info["country_code"] = getattr(ev, "country_code", None) or getattr(ev, "countryCode", None)
        m = {}
        try:
            for rc in getattr(mie, "runners", []) or []:
                sid = getattr(rc, "selection_id", None) or getattr(rc, "selectionId", None)
                rn  = getattr(rc, "runner_name", None) or getattr(rc, "runnerName", None)
                mdict = getattr(rc, "metadata", None) or {}
                draw = mdict.get("CLOTH_NUMBER") or mdict.get("clothNumber") or mdict.get("TRAP")
                m[int(sid)] = _MetaStub(runner_name=rn, draw=draw, sort_priority=getattr(rc, "sort_priority", None))
        except Exception:
            pass
        info["meta_by_sid"] = m
        # market_type robuste
        mt_hint = getattr(md, "market_type", None)
        if isinstance(mt_hint, str) and mt_hint.upper() in ("WIN","PLACE"):
            info["market_type"] = mt_hint.upper()
        else:
            nw = _num_winners(md)
            if isinstance(nw, int):
                info["market_type"] = "PLACE" if nw >= 2 else "WIN"
            else:
                nm = info["market_name"] or ""
                info["market_type"] = "PLACE" if ("place" in nm.lower()) else "WIN"
        return info

    @staticmethod
    def _compute_virtual_traps(runners, meta_by_sid):
        active = []
        absent_traps = set()
        for r in (runners or []):
            st = (getattr(r, "status", None) or "ACTIVE").upper()
            sid = getattr(r, "selection_id", None)
            rm = meta_by_sid.get(int(sid)) if sid is not None else None
            runner_name = getattr(rm, "runner_name", None) if rm else None
            trap = _parse_trap(rm, runner_name)
            if trap is None or sid is None:
                continue
            if st == "ACTIVE":
                active.append((int(sid), int(trap)))
            else:
                absent_traps.add(int(trap))
        active.sort(key=lambda t: t[1])
        N = len(active)
        vmap = {}
        if N == 0:
            return vmap
        for i, (sid, _trap) in enumerate(active):
            vmap[sid] = i + 1
        if N >= 7 and 8 in absent_traps:
            rightmost_sid = active[-1][0]
            vmap[rightmost_sid] = 8
        return vmap

    def _milestone_due(self, mid: str, tto: Optional[float]) -> Optional[int]:
        if tto is None:
            return None
        ms_list = self._next_ms[mid]
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