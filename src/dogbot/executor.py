# src/dogbot/executor.py
# Snapshots WIN + PLACE, jalons 300/150/80/45/2
# - MID_PLACE, MOYLTP_PLACE
# - DIFF/MOM sur BASE_WIN = (WINPROB -> MOYLTP_WIN -> LTP_WIN -> BEST_BACK)
# - PLACETHEORIQUE = cote (1/q) via Plackett–Luce (Top-K) à partir des prix WIN
# - Duplication : PLACETHEORIQUE_PLACE (même valeur que sur WIN) affichée sur les lignes PLACE
# - Fallback duplication si WIN_MARKET_ID pas encore résolu : cache par (event_id, start_utc minute)
# - NEW: pré-remplissage du cache PLACETHEORIQUE à CHAQUE passage sur un marché WIN (même sans jalon)

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Dict, List, Tuple
from collections import defaultdict, deque
import csv, math, os, re

from .types import Instruction, RunnerMeta
from .indexer import MarketIndex

# --- import Plackett–Luce (supporte plackett OU placket pour éviter la confusion) ---
try:
    # On s'attend à trouver dans src/dogbot/plackett/__init__.py ces fonctions
    from .plackett import odds_to_win_probs, place_probabilities, fair_place_odds
except Exception:
    # fallback: src/dogbot/placket/__init__.py
    from .placket import odds_to_win_probs, place_probabilities, fair_place_odds  # type: ignore


# ---------- utilitaires généraux ----------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _tz_utc(dt: Optional[datetime]) -> Optional[datetime]:
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

def _get_sp_near_far(r) -> tuple[Optional[float], Optional[float]]:
    sp = getattr(r, "sp", None)
    if sp is None:
        return (None, None)
    near = getattr(sp, "near_price", None)
    far  = getattr(sp, "far_price", None)
    if near is None and isinstance(sp, dict):
        near = sp.get("nearPrice")
    if far is None and isinstance(sp, dict):
        far = sp.get("farPrice")
    try:
        near = float(near) if near is not None else None
    except Exception:
        near = None
    try:
        far = float(far) if far is not None else None
    except Exception:
        far = None
    return (near, far)

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

# ----- petit stub pour ranger les métadonnées catalogue -----
class _MetaStub:
    def __init__(self, runner_name=None, draw=None, sort_priority=None, trap=None):
        self.runner_name = runner_name
        self.draw = draw
        self.sort_priority = sort_priority
        self.trap = trap


# ----- Executor -----
class Executor:
    MILESTONES = [300, 150, 80, 45, 2]
    TOLERANCE_S = 1.5

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
        "SNAP_TS_UTC","COURSE_ID","VENUE","MARKET_START_TIME_UTC","SECONDS_TO_OFF","MILESTONE_S",
        "WIN_MARKET_ID","PLACE_MARKET_ID","MARKET_ID","MARKET_TYPE",
        "SELECTION_ID","RUNNER_NAME","RUNNER_STATUS","DRAW","TRAP","VIRTUAL_TRAP","SORT_PRIORITY",
        "N_RUNNERS_ACTIVE","N_PLACES",
        # WIN
        "LTP_WIN","BEST_BACK_PRICE_1_WIN","BEST_BACK_SIZE_1_WIN","BEST_LAY_PRICE_1_WIN","BEST_LAY_SIZE_1_WIN",
        "MID_WIN","MOYLTP_WIN",
        "BACK_LADDER_WIN","LAY_LADDER_WIN","RUNNER_TOTAL_MATCHED_WIN",
        "FAV_RANK_LTP_WIN","FAV_RANK_BACK_WIN","WIN_IMPLIED_PROB_WIN",
        "NEAR_SP_WIN","FAR_SP_WIN","BSPMOY_WIN","WINPROB",
        "LTP_300_WIN","LTP_150_WIN","LTP_80_WIN","LTP_45_WIN","LTP_2_WIN",
        "DIFF150_300_WIN","DIFF80_150_WIN","DIFF45_80_WIN",
        "MOM45_WIN","MOM80_WIN","MOM150_WIN","MOM300_WIN",
        "PRICE_DELTA_5S_WIN","PRICE_DELTA_30S_WIN","VOLATILITY_60S_WIN","LIQUIDITY_SCORE_WIN",
        "IS_FAVOURITE_WIN","PLACETHEORIQUE",
        # PLACE
        "LTP_PLACE","BEST_BACK_PRICE_1_PLACE","BEST_BACK_SIZE_1_PLACE","BEST_LAY_PRICE_1_PLACE","BEST_LAY_SIZE_1_PLACE",
        "MID_PLACE","MOYLTP_PLACE",
        "BACK_LADDER_PLACE","LAY_LADDER_PLACE","RUNNER_TOTAL_MATCHED_PLACE",
        "NEAR_SP_PLACE","FAR_SP_PLACE","BSPMOY_PLACE","PLACEPROB",
        "PLACETHEORIQUE_PLACE",
        "LTP_300_PLACE","LTP_150_PLACE","LTP_80_PLACE","LTP_45_PLACE","LTP_2_PLACE",
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
        # - série "BASE_WIN" pour DIFF/MOM (WIN)
        self._base_win_ms: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
        # - LTP PLACE pour jalons PLACE
        self._ltp_place_ms: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))

        # Historique LTP (pour d5/d30/vol)
        self._hist: Dict[str, Dict[int, deque[Tuple[float,float]]]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=600)))

        # Cache pour dupliquer PLACETHEORIQUE sur PLACE
        # key = WIN marketId (string) -> { selection_id (int) : place_theo_odds (float) }
        self._last_place_theo_by_market: Dict[str, Dict[int, float]] = defaultdict(dict)
        # NEW: cache de secours par (event_id, start_utc à la minute)
        # key = (event_id, "YYYY-MM-DDTHH:MM")
        self._last_place_theo_by_event: Dict[tuple[str, str], Dict[int, float]] = defaultdict(dict)

        self._diag_fetch_latency_ms: Optional[float] = None
        self._diag_retry_count: Optional[float] = None
        self._diag_throttle_weight: Optional[float] = None
        self._code_version: Optional[str] = os.environ.get("CODE_VERSION")

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
        # compat multi-implémentations de MarketIndex
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
        if start is None and md is not None:
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

    # ---------- NEW: pré-calcul du PLACETHEORIQUE dès qu'on voit un book WIN ----------
    def _compute_and_cache_place_theo_from_win(self, info: Dict[str, Any], win_id: Optional[str],
                                               market_id: str, runners, n_active: int, n_places: int) -> None:
        """Calcule PLACETHEORIQUE à partir des prix WIN du book courant
        et met à jour les caches de duplication (par WIN_MARKET_ID + par (event_id, start_utc minute)).
        Appelée à CHAQUE passage sur un marché WIN, indépendamment des jalons.
        """
        sid_prices_for_pl: List[Tuple[int, float]] = []

        for r in (runners or []):
            sid = getattr(r, "selection_id", None)
            if sid is None:
                continue
            sid = int(sid)

            # Prix instantanés sur WIN
            lpt = _runner_lpt_or_back(r)
            ex = getattr(r, "ex", None)
            bb, _ = _best_at_side(ex, "BACK")
            bl, _ = _best_at_side(ex, "LAY")

            mid = ((bb + bl) / 2.0) if (bb is not None and bl is not None) else None
            moy = ((lpt + mid) / 2.0) if (lpt is not None and mid is not None) else None

            near, far = _get_sp_near_far(r)
            bspmoy = (near + far) / 2.0 if (near is not None and far is not None) else (near if near is not None else far)
            winprob = (bspmoy + lpt) / 2.0 if (bspmoy is not None and lpt is not None) else None

            base = winprob or moy or lpt or bb  # priorité: WINPROB -> MOYLTP -> LTP -> BB
            if base and base > 1.0:
                sid_prices_for_pl.append((sid, float(base)))

        if len(sid_prices_for_pl) < 2:
            return  # pas assez d’infos pour PL Plackett–Luce

        # Tri stable par sid
        sid_prices_for_pl.sort(key=lambda t: t[0])
        sids = [sid for sid, _ in sid_prices_for_pl]
        odds = [price for _, price in sid_prices_for_pl]

        try:
            p = odds_to_win_probs(odds, beta=1.0)       # probas gagnantes
            q = place_probabilities(p, K=n_places)      # probas de place (Top-K)
            fair = fair_place_odds(q)                   # cotes équitables (1/q)
            fair_list = list(fair) if not hasattr(fair, "tolist") else fair.tolist()
            by_sid = {sid: float(od) for sid, od in zip(sids, fair_list)}
        except Exception as e:
            print(f"[PL_ERR_PRE] {market_id}: {e}")
            return

        # Cache par WIN market id
        cache_key = (win_id or market_id)
        if cache_key and by_sid:
            self._last_place_theo_by_market[str(cache_key)] = by_sid

        # Cache fallback par (event_id, start_utc minute)
        ev_id = str(info.get("event_id") or "")
        st = info.get("start_utc")
        if ev_id and st and by_sid:
            ev_key = (ev_id, st.strftime("%Y-%m-%dT%H:%M"))
            self._last_place_theo_by_event[ev_key] = by_sid

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

        # Historique LTP pour d5/d30/vol (WIN uniquement, indicatif)
        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None: continue
            lpt = _runner_lpt_or_back(r)
            if lpt is not None:
                self._hist[market_id][int(sid)].append((datetime.now(timezone.utc).timestamp(), float(lpt)))

        # NEW: pré-remplir le cache PLACETHEORIQUE dès qu'on voit un WIN (même sans jalon)
        if market_type == "WIN":
            n_active = _active_count(runners)
            n_places = (_num_winners(md) or (3 if n_active >= 8 else 2))
            win_id, _ = self._resolve_link_ids(market_id, mie, md, market_type)
            self._compute_and_cache_place_theo_from_win(info, win_id, market_id, runners, n_active, n_places)

        rank_ltp, rank_back = _rankings(runners)
        milestone = self._milestone_due(market_id, t_to_off_s)

        self._write_market_row(book, info, market_type, t_to_off_s, milestone, runners)
        if milestone is not None:
            self._write_runner_rows(book, info, market_type, t_to_off_s, milestone, runners, rank_ltp, rank_back)

        # stratégie (dry-run)
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

        # --- Prépare prix par runner + caches ---
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

            # WIN mid/moyltp
            mid_win = ((bb + bl) / 2.0) if (bb is not None and bl is not None) else None
            moyltp_win = ((lpt + mid_win) / 2.0) if (lpt is not None and mid_win is not None) else None

            # PLACE mid/moyltp
            mid_place = ((bb + bl) / 2.0) if (bb is not None and bl is not None) else None
            moyltp_place = ((lpt + mid_place) / 2.0) if (lpt is not None and mid_place is not None) else None

            near_sp, far_sp = _get_sp_near_far(r)
            bspmoy = (near_sp + far_sp)/2.0 if (near_sp is not None and far_sp is not None) else (near_sp if near_sp is not None else far_sp)
            winprob = (bspmoy + lpt)/2.0 if (bspmoy is not None and lpt is not None) else None

            base_win = winprob or moyltp_win or lpt or bb  # priorité: WINPROB -> MOYLTP -> LTP -> BB

            if is_win and base_win and base_win > 1.0:
                sid_prices_for_pl.append((sid, float(base_win)))

            cache[sid] = {
                "LTP": lpt, "BB": bb, "BL": bl,
                "MID_WIN": mid_win, "MOYLTP_WIN": moyltp_win,
                "MID_PLACE": mid_place, "MOYLTP_PLACE": moyltp_place,
                "NEAR": near_sp, "FAR": far_sp, "BSPMOY": bspmoy,
                "WINPROB": winprob, "BASE_WIN": base_win,
            }

        # --- PLACETHEORIQUE via Plackett–Luce (cote = 1/q) pour le marché WIN (au jalon) ---
        place_theo_by_sid: Dict[int, Optional[float]] = {}
        if is_win and len(sid_prices_for_pl) >= max(2, min(5, n_active)):
            sid_prices_for_pl.sort(key=lambda t: t[0])  # tri par sid
            sids = [sid for sid,_ in sid_prices_for_pl]
            odds = [price for _,price in sid_prices_for_pl]
            try:
                p = odds_to_win_probs(odds, beta=1.0)      # -> probas gagnantes normalisées
                q = place_probabilities(p, K=n_places)     # -> probas de place Top-K (K=2 ou 3)
                fair = fair_place_odds(q)                  # -> cotes équitables (1/q)
                fair_list = list(fair) if not hasattr(fair, "tolist") else fair.tolist()
                for sid, odd_place in zip(sids, fair_list):
                    place_theo_by_sid[sid] = float(odd_place)
            except Exception as e:
                print(f"[PL_ERR] {market_id}: {e}")
                place_theo_by_sid = {sid: None for sid,_ in sid_prices_for_pl}

            # Cache par WIN market id (pour dupliquer côté PLACE)
            cache_key = (win_id or market_id)
            if cache_key:
                self._last_place_theo_by_market[str(cache_key)] = {
                    sid: v for sid, v in place_theo_by_sid.items() if v is not None
                }
            # Cache par (event_id, start_utc minute) pour fallback
            ev_id = str(info.get("event_id") or "")
            st = info.get("start_utc")
            if ev_id and st:
                ev_key = (ev_id, st.strftime("%Y-%m-%dT%H:%M"))
                self._last_place_theo_by_event[ev_key] = {
                    sid: v for sid, v in place_theo_by_sid.items() if v is not None
                }

        # --- Écriture des lignes runners ---
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
            mid_win = cache[sid].get("MID_WIN")
            moyltp_win = cache[sid].get("MOYLTP_WIN")
            mid_place = cache[sid].get("MID_PLACE")
            moyltp_place = cache[sid].get("MOYLTP_PLACE")

            ladder_b = _compress_ladder(getattr(ex, "available_to_back", None), 3)
            ladder_l = _compress_ladder(getattr(ex, "available_to_lay", None), 3)
            total_matched_runner = getattr(r, "total_matched", None)

            rk_ltp = rank_ltp.get(sid)
            rk_bb  = rank_back.get(sid)
            implied = (1.0 / (bb or lpt)) if (bb or lpt) else None

            near_sp = cache[sid].get("NEAR")
            far_sp  = cache[sid].get("FAR")
            bspmoy  = cache[sid].get("BSPMOY")
            winprob = cache[sid].get("WINPROB")
            base_win = cache[sid].get("BASE_WIN")

            # Milestones:
            if is_win and base_win is not None:
                self._base_win_ms[market_id][sid][milestone] = float(base_win)
            if is_place and lpt is not None:
                self._ltp_place_ms[market_id][sid][milestone] = float(lpt)

            # DIFF/MOM (WIN) sur BASE_WIN
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

            # Deltas/vol sur LTP pour info (WIN)
            d5 = d30 = None
            vol = None
            if is_win:
                dq = self._hist.get(market_id, {}).get(sid)
                if dq:
                    now_ts = datetime.now(timezone.utc).timestamp()
                    for window, varname in ((5.0, "d5"), (30.0, "d30")):
                        target = now_ts - window
                        older = None
                        for ts, p in reversed(dq):
                            older = (ts, p)
                            if ts <= target:
                                break
                        if older is not None:
                            p_now = dq[-1][1]; p_old = older[1]
                            if varname == "d5": d5 = p_now - p_old
                            else: d30 = p_now - p_old
                    vals = [p for ts, p in dq if ts >= now_ts - 60.0]
                    if len(vals) >= 3:
                        m = sum(vals)/len(vals)
                        var = sum((x-m)**2 for x in vals)/(len(vals)-1)
                        vol = math.sqrt(var)

            # PLACETHEORIQUE (cote) : calcul si WIN, sinon duplication via cache(s)
            place_theo_win = None
            place_theo_place = None

            if is_win:
                # si calculé au-dessus (jalon)
                place_theo_win = place_theo_by_sid.get(sid)

            if is_place:
                # 1) via WIN_MARKET_ID
                if win_id:
                    cached = self._last_place_theo_by_market.get(str(win_id))
                    if cached:
                        place_theo_place = cached.get(sid)
                # 2) fallback via (event_id, start_utc minute)
                if place_theo_place is None:
                    ev_id = str(info.get("event_id") or "")
                    st = info.get("start_utc")
                    if ev_id and st:
                        ev_key = (ev_id, st.strftime("%Y-%m-%dT%H:%M"))
                        cached2 = self._last_place_theo_by_event.get(ev_key)
                        if cached2:
                            place_theo_place = cached2.get(sid)

            # Milestones PLACE (LTP)
            def _get_place(ms):
                return self._ltp_place_ms.get(market_id, {}).get(sid, {}).get(ms)
            ltp_300_p = _get_place(300) if is_place else None
            ltp_150_p = _get_place(150) if is_place else None
            ltp_80_p  = _get_place(80)  if is_place else None
            ltp_45_p  = _get_place(45)  if is_place else None
            ltp_2_p   = _get_place(2)   if is_place else None

            base = [
                _now_utc_iso(),
                None,  # COURSE_ID (optionnel)
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
            ]

            win_block = [
                (lpt if is_win else None),
                (bb if is_win else None),
                (bs if is_win else None),
                (bl if is_win else None),
                (ls if is_win else None),
                (mid_win if is_win else None),
                (moyltp_win if is_win else None),
                (ladder_b if is_win else None),
                (ladder_l if is_win else None),
                (total_matched_runner if is_win else None),
                (rk_ltp if is_win else None),
                (rk_bb  if is_win else None),
                (implied if is_win else None),
                (near_sp if is_win else None),
                (far_sp  if is_win else None),
                (bspmoy  if is_win else None),
                (winprob if is_win else None),
                (base_300 if is_win else None),
                (base_150 if is_win else None),
                (base_80  if is_win else None),
                (base_45  if is_win else None),
                (base_2   if is_win else None),
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
                (((rk_bb or 0) == 1) if is_win else None),
                (place_theo_win if is_win else None),
            ]

            place_block = [
                (lpt if is_place else None),
                (bb if is_place else None),
                (bs if is_place else None),
                (bl if is_place else None),
                (ls if is_place else None),
                (mid_place if is_place else None),
                (moyltp_place if is_place else None),
                (ladder_b if is_place else None),
                (ladder_l if is_place else None),
                (total_matched_runner if is_place else None),
                (near_sp if is_place else None),
                (far_sp  if is_place else None),
                (bspmoy  if is_place else None),
                (((bspmoy + lpt)/2.0) if (is_place and bspmoy is not None and lpt is not None) else None),  # PLACEPROB proxy
                (place_theo_place if is_place else None),  # <-- DUPLICATION de PLACETHEORIQUE (WIN)
                (ltp_300_p if is_place else None),
                (ltp_150_p if is_place else None),
                (ltp_80_p  if is_place else None),
                (ltp_45_p  if is_place else None),
                (ltp_2_p   if is_place else None),
            ]

            row = base + win_block + place_block
            self._append(self.runner_csv, row)

    # ----- helpers -----
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
