# bot_collect_diag.py — diag jalons runners
# - logs détaillés t_to_off / jalons
# - SNAPSHOT_TOLERANCE_SECS (env) pour élargir la fenêtre
# - FORCE_SNAPSHOT=1 (env) pour écrire les runners même sans jalon (ms=999)
# - même boucle que bot_collect, avec fallback DSC-0018

from __future__ import annotations
import os, sys, time
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

from dotenv import load_dotenv
from betfairlightweight import APIClient
from betfairlightweight.filters import market_filter, price_projection, ex_best_offers_overrides
from betfairlightweight.exceptions import APIError

# Modules projet
from src.dogbot.indexer import MarketIndex  # type: ignore
from src.dogbot.strategy.back_win_1 import BackWin1
from src.dogbot.executor import Executor as BaseExecutor

# --------- helpers env ----------
def _env_int(name: str, default: int) -> int:
    try:
        v = (os.environ.get(name) or "").strip()
        return int(v) if v else int(default)
    except Exception:
        return int(default)

def _env_float(name: str, default: float) -> float:
    try:
        v = (os.environ.get(name) or "").strip()
        return float(v) if v else float(default)
    except Exception:
        return float(default)

def _env_bool(name: str, default: bool) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if v in ("1","true","yes","y","on"): return True
    if v in ("0","false","no","n","off"): return False
    return bool(default)

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def _chunked(seq: List[str], n: int) -> List[List[str]]:
    return [seq[i:i+n] for i in range(0, len(seq), n)]

# --------- price projections ----------
PP_NOSP = price_projection(
    price_data=["EX_BEST_OFFERS"],
    ex_best_offers_overrides=ex_best_offers_overrides(best_prices_depth=3),
    virtualise=True,
    rollover_stakes=False,
)
PP_FULL = price_projection(
    price_data=["EX_BEST_OFFERS","SP_AVAILABLE"],
    ex_best_offers_overrides=ex_best_offers_overrides(best_prices_depth=3),
    virtualise=True,
    rollover_stakes=False,
)
PP_MIN = price_projection(
    price_data=["EX_LTP"],
    virtualise=True,
    rollover_stakes=False,
)

# --------- catalogue ----------
def fetch_catalogue(client: APIClient, lookahead_minutes: int, max_markets: int):
    now = datetime.now(timezone.utc)
    to  = now + timedelta(minutes=lookahead_minutes)
    mf = market_filter(
        event_type_ids=["4339"],               # greyhounds
        market_type_codes=["WIN","PLACE"],     # WIN + PLACE
        market_start_time={"from": now.isoformat().replace("+00:00","Z"),
                           "to":   to.isoformat().replace("+00:00","Z")},
    )
    cats = client.betting.list_market_catalogue(
        filter=mf,
        market_projection=["EVENT","MARKET_START_TIME","RUNNER_DESCRIPTION","RUNNER_METADATA"],
        sort="FIRST_TO_START",
        max_results=max_markets
    )
    return cats

def build_market_index(cats) -> MarketIndex:
    try:
        return MarketIndex(cats)  # type: ignore
    except TypeError:
        mi = MarketIndex([])      # type: ignore
        try:
            mi.ingest(cats)
        except Exception:
            pass
        return mi

# ---------- Executor de diag ----------
class ExecutorDiag(BaseExecutor):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        ms = (os.environ.get("SNAPSHOT_TIMES") or "").strip()
        tol = os.environ.get("SNAPSHOT_TOLERANCE_SECS", "").strip()
        if ms:
            try:
                lst = [int(x) for x in ms.split(",") if x.strip()]
                if lst:
                    self.MILESTONES = lst
                    self._next_ms.clear()
            except Exception:
                pass
        if tol:
            try:
                self.TOLERANCE_S = float(tol)
            except Exception:
                pass
        self._force_snapshot = _env_bool("FORCE_SNAPSHOT", False)

    def _milestone_due(self, mid: str, tto: Optional[float]) -> Optional[int]:
        nxt = list(self._next_ms.get(mid, self.MILESTONES))
        print(f"[MS?] mid={mid} tto={None if tto is None else round(tto,1)} next={nxt} tol={self.TOLERANCE_S}")
        ms = super()._milestone_due(mid, tto)
        if ms is not None:
            print(f"[MS!] mid={mid} fired={ms}")
            return ms
        if self._force_snapshot:
            print(f"[MS+] mid={mid} FORCE_SNAPSHOT -> 999")
            return 999
        return None

    # même signature que la méthode d’origine → milestone loggé correctement
    def _write_runner_rows(self, book, info, market_type, tto, milestone, runners, rank_ltp, rank_back):
        print(f"[SNAP] writing runners milestone={milestone}")
        return super()._write_runner_rows(book, info, market_type, tto, milestone, runners, rank_ltp, rank_back)

# ---------- main ----------
def main():
    load_dotenv()
    if not os.environ.get("PYTHONUNBUFFERED"):
        try:
            sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        except Exception:
            pass

    user  = os.environ.get("BF_USER")
    pwd   = os.environ.get("BF_PASS")
    appk  = os.environ.get("BF_APP_KEY")
    certs = os.environ.get("BF_CERTS_PATH")
    if not (user and pwd and appk and certs):
        print("Manque des variables d'env (BF_USER/BF_PASS/BF_APP_KEY/BF_CERTS_PATH)")
        return

    LOOKAHEAD_MINUTES = _env_int("LOOKAHEAD_MINUTES", 60)
    SCAN_EVERY_S      = _env_int("SCAN_EVERY_S", 60)
    POLL_INTERVAL_S   = _env_float("POLL_INTERVAL_S", 2.0)
    BATCH_SIZE        = _env_int("BATCH_SIZE", 5)
    MAX_MARKETS       = _env_int("MAX_MARKETS", 20)
    INCLUDE_SP        = _env_bool("INCLUDE_SP_IN_BOOK", True)

    print(f"[BOOT] {_now_utc_iso()} lookahead={LOOKAHEAD_MINUTES}m scan_every={SCAN_EVERY_S}s poll={POLL_INTERVAL_S}s batch={BATCH_SIZE} max_markets={MAX_MARKETS} include_sp={INCLUDE_SP}")

    client = APIClient(username=user, password=pwd, app_key=appk, certs=certs)
    print("[LOGIN] connecting...")
    client.login()
    print("[LOGIN] OK")

    print("[CATALOGUE] fetching...")
    cats = fetch_catalogue(client, LOOKAHEAD_MINUTES, MAX_MARKETS)
    print(f"[CATALOGUE] found {len(cats)} markets")
    mindex = build_market_index(cats)

    strat = BackWin1()
    execu = ExecutorDiag(client=client, strategy=strat, market_index=mindex, dry_run=True, data_dir="./data")

    last_scan_ts = time.time()
    last_keepalive = time.time()

    try:
        while True:
            now_ts = time.time()

            if now_ts - last_keepalive > 540:
                try:
                    t0 = time.time()
                    client.keep_alive()
                    print(f"[KEEP-ALIVE] OK ({int((time.time()-t0)*1000)} ms)")
                except Exception as e:
                    print("[KEEP-ALIVE] error:", e)
                last_keepalive = now_ts

            if now_ts - last_scan_ts >= SCAN_EVERY_S:
                try:
                    print("[CATALOGUE] refresh...")
                    t0 = time.time()
                    cats = fetch_catalogue(client, LOOKAHEAD_MINUTES, MAX_MARKETS)
                    print(f"[CATALOGUE] refresh done: {len(cats)} markets ({int((time.time()-t0)*1000)} ms)")
                    mindex = build_market_index(cats)
                    execu.market_index = mindex
                except Exception as e:
                    print("[CATALOGUE] refresh error:", e)
                last_scan_ts = now_ts

            # marketIds à poller
            market_ids: List[str] = []
            try:
                for v in mindex.values():  # type: ignore[attr-defined]
                    mid = getattr(v, "market_id", None) or getattr(v, "marketId", None)
                    if mid:
                        market_ids.append(str(mid))
            except Exception:
                for v in mindex:
                    mid = getattr(v, "market_id", None) or getattr(v, "marketId", None)
                    if mid:
                        market_ids.append(str(mid))
            if not market_ids:
                time.sleep(POLL_INTERVAL_S)
                continue

            chunks = _chunked(market_ids, max(1, BATCH_SIZE))
            print(f"[LOOP] polling books for {len(market_ids)} markets in {len(chunks)} chunks (batch={BATCH_SIZE})")

            for ci, chunk in enumerate(chunks, start=1):
                proj = PP_FULL if INCLUDE_SP else PP_NOSP
                print(f"[BOOKS] chunk {ci}/{len(chunks)} start: {len(chunk)} ids -> {chunk}")
                t0 = time.time()
                try:
                    books = client.betting.list_market_book(market_ids=chunk, price_projection=proj)
                    elapsed = int((time.time()-t0)*1000)
                    print(f"[BOOKS] chunk {ci}/{len(chunks)} got {len(books)} books in {elapsed} ms")
                except APIError as e:
                    if "DSC-0018" in str(e):
                        print(f"[BOOKS] chunk {ci}/{len(chunks)} payload too big -> fallback EX_LTP")
                        t1 = time.time()
                        books = client.betting.list_market_book(market_ids=chunk, price_projection=PP_MIN)
                        elapsed = int((time.time()-t1)*1000)
                        print(f"[BOOKS] chunk {ci}/{len(chunks)} (fallback) got {len(books)} books in {elapsed} ms")
                    else:
                        print(f"[BOOKS] chunk {ci}/{len(chunks)} APIError: {e}")
                        continue

                for b in (books or []):
                    try:
                        execu.process_book(b)
                    except Exception as e:
                        print(f"[EXECUTOR_ERR] {getattr(b,'market_id','?')}: {e}")

                time.sleep(POLL_INTERVAL_S)

    finally:
        try:
            client.logout()
            print("[LOGOUT] OK")
        except Exception:
            pass

if __name__ == "__main__":
    main()
