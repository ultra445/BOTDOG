# executor.py — compatible MarketCatalogue brut (pas d'event_open_utc),
# écrit snapshots marché + runners (WIN/PLACE séparés en colonnes),
# BSP estimé, jalons 300/150/80/45/2, Plackett–Luce sur WIN.

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Dict, List, Tuple
from collections import defaultdict, deque
import csv, math, os, re

from .types import Instruction, RunnerMeta
from .indexer import MarketIndex

# ---------- petits utilitaires ----------
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

# ----- Plackett–Luce Top-K -----
def _elem_sym_poly(weights: List[float], K: int) -> List[float]:
    e = [0.0] * (K + 1); e[0] = 1.0
    for w in weights:
        for m in range(K, 0, -1):
            e[m] += w * e[m-1]
    return e

def _place_probs_plackett_luce(weights: List[float], K: int) -> List[float]:
    n = len(weights); K = min(K, n)
    if K <= 0: return [0.0]*n
    e_all = _elem_sym_poly(weights, K)
    res = [0.0]*n
    for i in range(n):
        wi = weights[i]
        if wi <= 0:
            res[i] = 0.0; continue
        w_excl = [weights[j] for j in range(n) if j != i]
        e_ex = _elem_sym_poly(w_excl, K-1 if K>0 else 0)
        s = 0.0
        for m in range(1, K+1):
            denom = e_all[m]; num = wi * (e_ex[m-1] if (m-1) < len(e_ex) else 0.0)
            if denom > 0: s += num / denom
        res[i] = s
    return res

# ----- accès polyvalent à l’index -----
def _mindex_values(idx: MarketIndex) -> List[Any]:
    for try_fn in (
        lambda i: list(i.values()),            # dict-like
        lambda i: [v for _, v in i.items()],   # items()
        lambda i: list(i),                     # itérable
    ):
        try: return try_fn(idx)  # type: ignore
        except Exception: pass
    for attr in ("by_id","_by_id","map","_map","d","_d"):
        d = getattr(idx, attr, None)
        if isinstance(d, dict): return list(d.values())
    return []

# ----- extraction souple depuis MarketCatalogue / MarketBook -----
class _MetaStub:
    def __init__(self, runner_name=None, draw=None, sort_priority=None, trap=None):
        self.runner_name = runner_name
        self.draw = draw
        self.sort_priority = sort_priority
        self.trap = trap

def _extract_catalogue_info(mie: Any, md: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "event_id": None, "event_name": None, "venue": None, "country_code": None,
        "start_utc": None, "market_type": None, "market_name": None, "meta_by_sid": {},
    }
    mt = (getattr(md, "market_type", None) or "").upper()
    info["market_type"] = mt if mt else None

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

    info["market_name"] = getattr(mie, "market_name", None) or getattr(mie, "marketName", None)

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
    return info

def _guess_type_from_name(name: Optional[str]) -> Optional[str]:
    if not name: return None
    s = name.lower()
    if "place" in s: return "PLACE"
    if "win" in s or "winner" in s: return "WIN"
    return None

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
        "BACK_LADDER_WIN","LAY_LADDER_WIN","RUNNER_TOTAL_MATCHED_WIN",
        "FAV_RANK_LTP_WIN","FAV_RANK_BACK_WIN","WIN_IMPLIED_PROB_WIN",
        "NEAR_SP_WIN","FAR_SP_WIN","BSPMOY_WIN","WINPROB",
        "LTP_300_WIN","LTP_150_WIN","LTP_80_WIN","LTP_45_WIN","LTP_2_WIN",
        "DIFF150_300_WIN","DIFF80_150_WIN","DIFF45_80_WIN",
        "MOM45_WIN","MOM80_WIN","MOM150_WIN","MOM300_WIN",
        "PRICE_DELTA_5S_WIN","PRICE_DELTA_30S_WIN","VOLATILITY_60S_WIN","LIQUIDITY_SCORE_WIN",
        "IS_FAVOURITE_WIN","PLACE_THEORIQUE",
        # PLACE
        "LTP_PLACE","BEST_BACK_PRICE_1_PLACE","BEST_BACK_SIZE_1_PLACE","BEST_LAY_PRICE_1_PLACE","BEST_LAY_SIZE_1_PLACE",
        "BACK_LADDER_PLACE","LAY_LADDER_PLACE","RUNNER_TOTAL_MATCHED_PLACE",
        "NEAR_SP_PLACE","FAR_SP_PLACE","BSPMOY_PLACE","PLACEPROB",
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
        self._hist: Dict[str, Dict[int, deque[Tuple[float,float]]]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=600)))
        self._ltp_ms: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))

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
        return _mindex_values(self.market_index)

    def _resolve_link_ids(self, mie: Any, md: Any) -> tuple[Optional[str], Optional[str]]:
        win_id = getattr(mie, "win_market_id", None)
        place_id = getattr(mie, "place_market_id", None)
        if win_id and place_id:
            return win_id, place_id

        info = _extract_catalogue_info(mie, md)
        mt = info["market_type"] or _guess_type_from_name(info["market_name"])
        ev_id = info["event_id"]
        start = info["start_utc"]
        cur_mid = getattr(mie, "market_id", None)

        if not ev_id or not start:
            # on retourne au moins le market courant selon son type
            cur_mt = (mt or "").upper()
            cur_id = getattr(mie, "market_id", None)
            if cur_mt == "WIN":   return cur_id, place_id
            if cur_mt == "PLACE": return win_id, cur_id
            return win_id, place_id

        for other in self._mindex_vals():
            if getattr(other, "market_id", None) == cur_mid:
                continue
            oi = _extract_catalogue_info(other, None)
            if oi["event_id"] != ev_id:
                continue
            if oi["start_utc"] and abs((oi["start_utc"] - start).total_seconds()) > 180:
                continue
            omt = oi["market_type"] or _guess_type_from_name(oi["market_name"])
            omid = getattr(other, "market_id", None)
            if not omt or not omid:
                continue
            omt = omt.upper()
            if omt == "WIN" and not win_id:
                win_id = omid
            if omt == "PLACE" and not place_id:
                place_id = omid

        cur_mt = (mt or "").upper()
        cur_id = getattr(mie, "market_id", None)
        if cur_mt == "WIN" and not win_id:
            win_id = cur_id
        if cur_mt == "PLACE" and not place_id:
            place_id = cur_id

        return win_id, place_id

    # ---------- milestones / histo ----------
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

    def _push_hist(self, market_id: str, selection_id: int, price: Optional[float]) -> None:
        if price is None:
            return
        self._hist[market_id][selection_id].append((datetime.now(timezone.utc).timestamp(), float(price)))

    def _delta_since(self, market_id: str, selection_id: int, seconds: float) -> Optional[float]:
        dq = self._hist.get(market_id, {}).get(selection_id)
        if not dq:
            return None
        now_ts = datetime.now(timezone.utc).timestamp()
        target = now_ts - seconds
        older = None
        for ts, p in reversed(dq):
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

    @staticmethod
    def _compute_virtual_traps(runners, meta_by_sid: Dict[int, RunnerMeta]) -> Dict[int, Optional[int]]:
        active = []
        absent_traps = set()
        for r in (runners or []):
            st = (getattr(r, "status", None) or "ACTIVE").upper()
            sid = getattr(r, "selection_id", None)
            rm = meta_by_sid.get(int(sid)) if (sid is not None) else None
            trap = _parse_trap(rm, getattr(rm, "runner_name", None) if rm else None)
            if st == "ACTIVE" and trap is not None:
                active.append((int(sid), trap))
            elif st != "ACTIVE" and trap is not None:
                absent_traps.add(int(trap))
        active.sort(key=lambda t: t[1])
        N = len(active)
        vmap: Dict[int, Optional[int]] = {}
        if N == 0:
            return vmap
        for i, (sid, _trap) in enumerate(active):
            vmap[sid] = i + 1
        if N >= 7 and 8 in absent_traps:
            rightmost_sid = active[-1][0]
            vmap[rightmost_sid] = 8
        return vmap

    # ---------- main ----------
    def process_book(self, book: Any) -> None:
        market_id = str(getattr(book, "market_id", ""))
        md = getattr(book, "market_definition", None)
        runners = getattr(book, "runners", None) or []

        # entrée d’index (peut être MarketCatalogue)
        try:
            mie = self.market_index.get(market_id)  # type: ignore
        except Exception:
            mie = None

        info = _extract_catalogue_info(mie, md)
        start_utc = info["start_utc"]
        now = datetime.now(timezone.utc)
        t_to_off_s = (start_utc - now).total_seconds() if start_utc else None
        market_type = (info["market_type"] or _guess_type_from_name(info["market_name"]) or "").upper()

        if getattr(book, "inplay", False):
            self._next_ms.pop(market_id, None)
            self._last_tto[market_id] = t_to_off_s if t_to_off_s is not None else 0.0
            return

        rank_ltp, rank_back = _rankings(runners)
        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None: 
                continue
            self._push_hist(market_id, int(sid), _runner_lpt_or_back(r))

        milestone = self._milestone_due(market_id, t_to_off_s)
        self._write_market_row(book, info, t_to_off_s, milestone, runners)

        if milestone is not None:
            self._write_runner_rows(book, info, t_to_off_s, milestone, runners, rank_ltp, rank_back, market_type)

        # stratégie (robuste: la stratégie ne doit plus lire mie.event_open_utc)
        instructions: List[Instruction] = []
        try:
            now_dt = datetime.now(timezone.utc)
            instructions = self.strategy.decide_all(book, mie, now_dt) or []
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

    def _write_market_row(self, book, info: Dict[str,Any], tto: Optional[float], milestone: Optional[int], runners) -> None:
        md = getattr(book, "market_definition", None)
        back_over = _overround_back(runners)
        lay_over  = _overround_lay(runners)
        spread    = _spread_pct(runners)

        # on essaye de relier WIN/PLACE (même event + start proche)
        mie_current = getattr(self.market_index, "get", lambda _:_)(getattr(book,"market_id",None))
        win_id, place_id = self._resolve_link_ids(mie_current, md)

        row = [
            _now_utc_iso(),
            str(getattr(book, "market_id", "")),
            None,  # COURSE_ID (optionnel, non reconstruit ici)
            (info["market_type"] or _guess_type_from_name(info["market_name"])),
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

    def _write_runner_rows(self, book, info: Dict[str,Any], tto: Optional[float], milestone: int,
                           runners, rank_ltp: Dict[int,int], rank_back: Dict[int,int], market_type: str) -> None:
        ts = _now_utc_iso()
        market_id = str(getattr(book, "market_id", ""))
        n_active = _active_count(runners)
        n_places = (_num_winners(getattr(book, "market_definition", None)) 
                    or (3 if n_active >= 8 else 2))

        meta_by_sid: Dict[int, RunnerMeta] = info.get("meta_by_sid", {}) or {}
        vmap = self._compute_virtual_traps(runners, meta_by_sid)

        mie_current = getattr(self.market_index, "get", lambda _:_)(market_id)
        win_id, place_id = self._resolve_link_ids(mie_current, getattr(book,"market_definition",None))

        # Prépare Plackett–Luce (WIN)
        sid_prices_for_pl: Dict[int, float] = {}
        temp_cache: Dict[int, Dict[str, Optional[float]]] = {}
        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None: continue
            sid = int(sid)
            lpt = _runner_lpt_or_back(r)
            ex = getattr(r, "ex", None)
            bb, _ = _best_at_side(ex, "BACK")
            near_sp, far_sp = _get_sp_near_far(r)
            bspmoy = (near_sp + far_sp)/2.0 if (near_sp is not None and far_sp is not None) else (near_sp if near_sp is not None else far_sp)
            winprob_price = (bspmoy + lpt)/2.0 if (bspmoy is not None and lpt is not None) else None
            chosen = winprob_price or bspmoy or lpt or bb
            if chosen and chosen > 0:
                sid_prices_for_pl[sid] = float(chosen)
            temp_cache[sid] = {"LTP": lpt, "BB": bb, "NEAR": near_sp, "FAR": far_sp, "BSPMOY": bspmoy, "WINPROB": winprob_price}

        do_pl = (market_type == "WIN") and (len(sid_prices_for_pl) >= max(2, min(5, n_active)))
        ordered_sids: List[int] = []
        weights: List[float] = []
        if do_pl:
            for r in runners:
                sid = int(getattr(r, "selection_id", 0) or 0)
            # construisons dans le même ordre :
            for r in runners:
                sid = int(getattr(r, "selection_id", 0) or 0)
                price = sid_prices_for_pl.get(sid)
                if price and price > 0:
                    ordered_sids.append(sid); weights.append(1.0 / float(price))
                else:
                    ordered_sids.append(sid); weights.append(0.0)
            q_topk: Dict[int, Optional[float]] = {}
            if sum(weights) > 0 and n_places > 0:
                probs = _place_probs_plackett_luce(weights, K=n_places)
                for sid, qi in zip(ordered_sids, probs):
                    q_topk[sid] = float(qi)
            else:
                q_topk = {sid: None for sid in ordered_sids}
        else:
            q_topk = {}

        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None: continue
            sid = int(sid)
            status = getattr(r, "status", None)
            rm: RunnerMeta | None = meta_by_sid.get(sid)
            runner_name = (rm.runner_name if rm else None)
            trap = _parse_trap(rm, runner_name)
            vtrap = vmap.get(sid, trap)

            lpt = temp_cache.get(sid, {}).get("LTP")
            ex = getattr(r, "ex", None)
            bb, bs = _best_at_side(ex, "BACK")
            bl, ls = _best_at_side(ex, "LAY")
            ladder_b = _compress_ladder(getattr(ex, "available_to_back", None), 3)
            ladder_l = _compress_ladder(getattr(ex, "available_to_lay", None), 3)
            total_matched_runner = getattr(r, "total_matched", None)

            rk_ltp = rank_ltp.get(sid)
            rk_bb  = rank_back.get(sid)
            implied = (1.0 / (bb or lpt)) if (bb or lpt) else None

            near_sp = temp_cache.get(sid, {}).get("NEAR")
            far_sp  = temp_cache.get(sid, {}).get("FAR")
            bspmoy  = temp_cache.get(sid, {}).get("BSPMOY")
            winprob = temp_cache.get(sid, {}).get("WINPROB")

            if lpt is not None:
                self._ltp_ms[market_id][sid][milestone] = float(lpt)
            ms_vals = self._ltp_ms[market_id][sid]
            ltp_300 = ms_vals.get(300); ltp_150 = ms_vals.get(150)
            ltp_80  = ms_vals.get(80);  ltp_45  = ms_vals.get(45); ltp_2 = ms_vals.get(2)

            def ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
                if a is None or b is None or b == 0: return None
                return (a / b) - 1.0

            diff150_300 = ratio(ltp_150, ltp_300)
            diff80_150  = ratio(ltp_80,  ltp_150)
            diff45_80   = ratio(ltp_45,  ltp_80)
            mom45  = ratio(ltp_2,  ltp_45)
            mom80  = ratio(ltp_2,  ltp_80)
            mom150 = ratio(ltp_2,  ltp_150)
            mom300 = ratio(ltp_2,  ltp_300)

            d5  = self._delta_since(market_id, sid, 5.0)
            d30 = self._delta_since(market_id, sid, 30.0)
            vol = self._volatility(market_id, sid, 60.0)
            liq_score = (bs or 0.0) + (ls or 0.0)

            base = [
                _now_utc_iso(),
                None,  # COURSE_ID (non reconstruit ici)
                info["venue"],
                (info["start_utc"].isoformat().replace("+00:00","Z") if info["start_utc"] else None),
                float(tto) if tto is not None else None,
                milestone,
                *self._resolve_link_ids(mie_current, getattr(book,"market_definition",None)),
                market_id, (info["market_type"] or _guess_type_from_name(info["market_name"]) or "").upper(),
                sid, runner_name, status,
                (getattr(rm, "draw", None) if rm else None),
                trap, vtrap,
                (getattr(rm, "sort_priority", None) if rm else None),
                n_active, n_places,
            ]

            win_block = [
                lpt if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None,
                (bb if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (bs if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (bl if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (ls if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (ladder_b if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (ladder_l if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (total_matched_runner if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (rk_ltp if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (rk_bb  if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (implied if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (near_sp if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (far_sp  if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (bspmoy  if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (winprob if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (ltp_300 if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (ltp_150 if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (ltp_80  if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (ltp_45  if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (ltp_2   if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (diff150_300 if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (diff80_150  if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (diff45_80   if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (mom45  if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (mom80  if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (mom150 if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (mom300 if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (d5 if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (d30 if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (vol if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (liq_score if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (((rk_bb or 0) == 1) if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" else None),
                (q_topk.get(sid) if ((info["market_type"] or _guess_type_from_name(info["market_name"])) == "WIN" and do_pl) else None),
            ]

            place_block = [
                lpt if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None,
                (bb if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None),
                (bs if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None),
                (bl if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None),
                (ls if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None),
                (ladder_b if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None),
                (ladder_l if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None),
                (total_matched_runner if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None),
                (near_sp if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None),
                (far_sp  if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None),
                (bspmoy  if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None),
                (((bspmoy + lpt)/2.0) if ((info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" and bspmoy is not None and lpt is not None) else None),
                (ltp_300 if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None),
                (ltp_150 if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None),
                (ltp_80  if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None),
                (ltp_45  if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None),
                (ltp_2   if (info["market_type"] or _guess_type_from_name(info["market_name"])) == "PLACE" else None),
            ]

            row = base + win_block + place_block
            self._append(self.runner_csv, row)
