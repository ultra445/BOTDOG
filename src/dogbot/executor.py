# src/dogbot/executor.py — snapshots enrichis (TRAP, VIRTUAL_TRAP, LTP jalons & variations)
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Dict, List, Tuple
from collections import defaultdict, deque
import csv, math, os, re

from .types import MarketIndexEntry, Instruction, RunnerMeta
from .indexer import MarketIndex

# ---------- petits helpers ----------
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
    """LTP si dispo, sinon meilleure cote BACK."""
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
    s = 0.0
    k = 0
    for r in (runners or []):
        pb = _best_at_side(getattr(r, "ex", None), "BACK")[0]
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
        pl = _best_at_side(getattr(r, "ex", None), "LAY")[0]
        if pl and pl > 1e-9:
            s += 1.0 / pl
            k += 1
    if k == 0:
        return None
    return s * 100.0

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
    m = vals[len(vals)//2]
    return m * 100.0

def _rankings(runners) -> tuple[Dict[int,int], Dict[int,int]]:
    arr_ltp, arr_bb = [], []
    for r in (runners or []):
        sid = getattr(r, "selection_id", None)
        if sid is None:
            continue
        lpt = _runner_lpt_or_back(r)
        ex = getattr(r, "ex", None)
        pb, _ = _best_at_side(ex, "BACK")
        if lpt is not None:
            arr_ltp.append((sid, lpt))
        if pb is not None:
            arr_bb.append((sid, pb))
    arr_ltp.sort(key=lambda t: t[1])
    arr_bb.sort(key=lambda t: t[1])
    return ({sid:i+1 for i,(sid,_) in enumerate(arr_ltp)},
            {sid:i+1 for i,(sid,_) in enumerate(arr_bb)})

# ---------- extraction TRAP ----------
_TRAP_PATTS = [
    re.compile(r"^\s*(?:TRAP|T)\s*([1-9]\d?)\b", re.I),
    re.compile(r"^\s*([1-9]\d?)\s*[.\-)\s]"),
    re.compile(r"\(\s*(?:TRAP|T)\s*([1-9]\d?)\s*\)", re.I),
]

def _parse_trap(meta: Optional[RunnerMeta], runner_name: Optional[str]) -> Optional[int]:
    # priorité aux métadonnées
    for v in [getattr(meta, "trap", None), getattr(meta, "draw", None)]:
        if v is None:
            continue
        s = str(v).strip()
        m = re.match(r"^[^\d]*([1-9]\d?)", s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    # fallback: extraire depuis runner_name
    if runner_name:
        for patt in _TRAP_PATTS:
            m = patt.search(runner_name)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    continue
    return None

# ============ Executor ============
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
        "SNAP_TS_UTC","MARKET_ID","SELECTION_ID","RUNNER_NAME","RUNNER_STATUS",
        "DRAW","TRAP","VIRTUAL_TRAP","SORT_PRIORITY","N_RUNNERS_ACTIVE",
        "LTP","BEST_BACK_PRICE_1","BEST_BACK_SIZE_1","BEST_LAY_PRICE_1","BEST_LAY_SIZE_1",
        "BACK_LADDER","LAY_LADDER","RUNNER_TOTAL_MATCHED",
        "FAV_RANK_LTP","FAV_RANK_BACK","WIN_IMPLIED_PROB",
        "LTP_300","LTP_150","LTP_80","LTP_45","LTP_2",
        "DIFF150_300","DIFF80_150","DIFF45_80",
        "MOM45","MOM80","MOM150","MOM300",
        "PRICE_DELTA_5S","PRICE_DELTA_30S","VOLATILITY_60S","LIQUIDITY_SCORE",
        "IS_FAVOURITE","PLACE_THEORIQUE",
        "SECONDS_TO_OFF","VENUE","COURSE_ID","MILESTONE_S"
    ]

    # Gating par défaut (peut bouger ensuite)
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

        # suivi milestones & historique prix
        self._next_ms: Dict[str, List[int]] = defaultdict(lambda: list(self.MILESTONES))
        self._last_tto: Dict[str, float] = {}
        self._hist: Dict[str, Dict[int, deque[Tuple[float,float]]]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=600)))
        # LTP capturés exactement aux milestones
        self._ltp_ms: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
        # diagnostics (remplis par bot_collect)
        self._diag_fetch_latency_ms: Optional[float] = None
        self._diag_retry_count: Optional[int] = None
        self._diag_throttle_weight: Optional[float] = None
        self._code_version: Optional[str] = os.environ.get("CODE_VERSION")

    # --------- diagnostics setter (utilisé par bot_collect) ---------
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

    # --------- I/O CSV ---------
    def _ensure_header(self, path: Path, header: Iterable[str]) -> None:
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(list(header))

    def _append(self, path: Path, row: Iterable[Any]) -> None:
        with path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(list(row))

    # --------- milestones ---------
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

    # --------- histo / deltas / vol ---------
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

    # --------- virtual traps mapping ---------
    @staticmethod
    def _compute_virtual_traps(runners, meta_by_sid: Dict[int, RunnerMeta]) -> Dict[int, Optional[int]]:
        """
        Règles:
          - Ordonner les partants ACTIFS par TRAP croissant.
          - Assigner VIRTUAL_TRAP = 1..N (gauche->droite).
          - Cas spécial: si N>=7 et le TRAP 8 est non-partant, alors le plus à droite prend VIRTUAL_TRAP=8.
            (les autres gardent 1..6; donc pas de '7' dans ce cas).
          - Sinon, VIRTUAL_TRAP = TRAP (quand aligné, ce qui sera la majorité des cas).
        """
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

        # Trier par trap
        active.sort(key=lambda t: t[1])
        N = len(active)
        vmap: Dict[int, Optional[int]] = {}
        if N == 0:
            return vmap

        # assignation par défaut 1..N
        for i, (sid, _trap) in enumerate(active):
            vmap[sid] = i + 1

        # cas spécial "8 non-partant" si N>=7
        if N >= 7 and 8 in absent_traps:
            # le plus à droite (dernier de la liste triée) reçoit VIRTUAL_TRAP=8
            rightmost_sid = active[-1][0]
            vmap[rightmost_sid] = 8

        return vmap

    # --------- main ---------
    def process_book(self, book: Any) -> None:
        market_id = str(getattr(book, "market_id", ""))
        md = getattr(book, "market_definition", None)
        runners = getattr(book, "runners", None) or []
        now = datetime.now(timezone.utc)

        mie: Optional[MarketIndexEntry] = self.market_index.get(market_id) if self.market_index else None
        event_open_utc = mie.event_open_utc if mie else None
        t_to_off_s = (event_open_utc - now).total_seconds() if event_open_utc else None

        if getattr(book, "inplay", False):
            self._next_ms.pop(market_id, None)
            self._last_tto[market_id] = t_to_off_s if t_to_off_s is not None else 0.0
            return

        # ranks
        rank_ltp, rank_back = _rankings(runners)

        # hist pour deltas/vol
        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None:
                continue
            self._push_hist(market_id, int(sid), _runner_lpt_or_back(r))

        milestone = self._milestone_due(market_id, t_to_off_s)
        # snapshot marché (toujours)
        self._write_market_row(book, mie, t_to_off_s, milestone, runners)

        # runners : seulement aux jalons
        if milestone is not None:
            self._write_runner_rows(book, mie, t_to_off_s, milestone, runners, rank_ltp, rank_back)

        # stratégie
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

    # --------- écriture CSV (market) ---------
    def _write_market_row(self, book, mie: Optional[MarketIndexEntry], tto: Optional[float], milestone: Optional[int], runners) -> None:
        md = getattr(book, "market_definition", None)
        back_over = _overround_back(runners)
        lay_over  = _overround_lay(runners)
        spread    = _spread_pct(runners)

        row = [
            _now_utc_iso(),
            str(getattr(book, "market_id", "")),
            mie.course_id if mie else None,
            (getattr(md, "market_type", None) or (mie.market_type if mie else None)),
            (mie.event_id if mie else None),
            (mie.event_name if mie else None),
            (mie.venue if mie else None),
            (mie.country_code if mie else None),
            mie.event_open_utc.isoformat().replace("+00:00","Z") if (mie and mie.event_open_utc) else None,
            float(tto) if tto is not None else None,
            bool(getattr(book, "inplay", False)),
            _market_status(md),
            _num_winners(md),
            _active_count(runners),
            float(getattr(book, "total_matched", None) or 0.0),
            back_over, lay_over, spread,
            (mie.win_market_id if mie else None),
            (mie.place_market_id if mie else None),
            (1 if (mie and mie.win_market_id and mie.place_market_id) else 0),
            (mie.n_places if mie else None),
            (tto is not None and self.ENTRY_MIN_T_S <= tto <= self.ENTRY_MAX_T_S),
            self._fav_price_ok(runners),
            (float(getattr(book, "total_matched", 0.0)) >= self.MIN_LIQUIDITY_MARKET),
            (back_over is not None and lay_over is not None),
            None,  # RULES_ALL_OK (on le propagera quand la stratégie renverra un plan détaillé)
            (getattr(self.strategy, "name", None) or None),
            None, None, None, None, None, None, None, None,
            False,  # DRYRUN_WOULD_PLACE
            milestone,
            # Diagnostics
            self._diag_fetch_latency_ms,
            None,  # CACHE_AGE_S
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

    # --------- écriture CSV (runners aux jalons) ---------
    def _write_runner_rows(self, book, mie: Optional[MarketIndexEntry], tto: Optional[float], milestone: int,
                           runners, rank_ltp: Dict[int,int], rank_back: Dict[int,int]) -> None:
        ts = _now_utc_iso()
        market_id = str(getattr(book, "market_id", ""))
        n_active = _active_count(runners)
        meta_by_sid = mie.runners_meta if (mie and mie.runners_meta) else {}
        vmap = self._compute_virtual_traps(runners, meta_by_sid)

        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None:
                continue
            sid = int(sid)
            status = getattr(r, "status", None)

            # métadonnées
            rm: RunnerMeta | None = meta_by_sid.get(sid)
            runner_name = (rm.runner_name if rm else None)
            # TRAP réel (extraction robuste)
            trap = _parse_trap(rm, runner_name)
            vtrap = vmap.get(sid, trap)

            # prix instantanés
            lpt = _runner_lpt_or_back(r)
            ex = getattr(r, "ex", None)
            bb, bs = _best_at_side(ex, "BACK")
            bl, ls = _best_at_side(ex, "LAY")
            ladder_b = _compress_ladder(getattr(ex, "available_to_back", None), 3)
            ladder_l = _compress_ladder(getattr(ex, "available_to_lay", None), 3)
            total_matched_runner = getattr(r, "total_matched", None)

            # ranks & implied
            rk_ltp = rank_ltp.get(sid)
            rk_bb  = rank_back.get(sid)
            implied = (1.0 / (bb or lpt)) if (bb or lpt) else None

            # enregistrer LTP au jalon courant
            if lpt is not None:
                self._ltp_ms[market_id][sid][milestone] = float(lpt)
            ms_vals = self._ltp_ms[market_id][sid]
            ltp_300 = ms_vals.get(300)
            ltp_150 = ms_vals.get(150)
            ltp_80  = ms_vals.get(80)
            ltp_45  = ms_vals.get(45)
            ltp_2   = ms_vals.get(2)

            def ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
                if a is None or b is None or b == 0:
                    return None
                return (a / b) - 1.0

            diff150_300 = ratio(ltp_150, ltp_300)
            diff80_150  = ratio(ltp_80,  ltp_150)
            diff45_80   = ratio(ltp_45,  ltp_80)
            mom45  = ratio(ltp_2,  ltp_45)
            mom80  = ratio(ltp_2,  ltp_80)
            mom150 = ratio(ltp_2,  ltp_150)
            mom300 = ratio(ltp_2,  ltp_300)  # logique: 2 vs 300

            # deltas rapides & vol
            d5  = self._delta_since(market_id, sid, 5.0)
            d30 = self._delta_since(market_id, sid, 30.0)
            vol = self._volatility(market_id, sid, 60.0)

            # “liquidity score” simple
            liq_score = (bs or 0.0) + (ls or 0.0)

            row = [
                ts, market_id, sid,
                runner_name,
                status,
                (rm.draw if rm and rm.draw else None),
                trap,
                vtrap,
                (rm.sort_priority if rm else None),
                n_active,
                lpt, bb, bs, bl, ls,
                ladder_b, ladder_l, total_matched_runner,
                rk_ltp, rk_bb, implied,
                ltp_300, ltp_150, ltp_80, ltp_45, ltp_2,
                diff150_300, diff80_150, diff45_80,
                mom45, mom80, mom150, mom300,
                d5, d30, vol, liq_score,
                (rk_bb == 1 if rk_bb is not None else None),
                None,  # PLACE_THEORIQUE (on fera après)
                float(tto) if tto is not None else None,
                (mie.venue if mie else None),
                (mie.course_id if mie else None),
                milestone,
            ]
            self._append(self.runner_csv, row)
