# sp_probe_plus.py — probe SP near/far avec indicateur bsp_market & t_to_off

from __future__ import annotations
import os, time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from dotenv import load_dotenv
from betfairlightweight import APIClient
from betfairlightweight.filters import market_filter, price_projection, ex_best_offers_overrides
from betfairlightweight.exceptions import APIError

def _nowz() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def _tto(start_utc: Optional[datetime]) -> Optional[float]:
    if not start_utc: return None
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    return (start_utc - datetime.now(timezone.utc)).total_seconds()

def main():
    load_dotenv()
    user  = os.environ.get("BF_USER")
    pwd   = os.environ.get("BF_PASS")
    appk  = os.environ.get("BF_APP_KEY")
    certs = os.environ.get("BF_CERTS_PATH")
    if not (user and pwd and appk and certs):
        print("Manque des variables d'env (BF_USER/BF_PASS/BF_APP_KEY/BF_CERTS_PATH)"); return

    lookahead_minutes = int(os.environ.get("LOOKAHEAD_MINUTES", "60") or "60")

    client = APIClient(username=user, password=pwd, app_key=appk, certs=certs)
    print(f"[BOOT] {_nowz()}  lookahead={lookahead_minutes}m")
    print("[LOGIN] connecting...")
    client.login()
    print("[LOGIN] OK")

    # ------- catalogue : on prend les 10 plus proches -------
    now = datetime.now(timezone.utc); to = now + timedelta(minutes=lookahead_minutes)
    mf = market_filter(
        event_type_ids=["4339"],  # greyhounds
        market_type_codes=["WIN","PLACE"],
        market_start_time={"from": now.isoformat().replace("+00:00","Z"),
                           "to":   to.isoformat().replace("+00:00","Z")},
    )
    cats = client.betting.list_market_catalogue(
        filter=mf,
        market_projection=["EVENT","MARKET_START_TIME","MARKET_DESCRIPTION","RUNNER_DESCRIPTION","RUNNER_METADATA"],
        sort="FIRST_TO_START",
        max_results=10
    )
    cats = sorted(cats, key=lambda c: c.market_start_time)
    mids: List[str] = [c.market_id for c in cats][:6]
    print(f"[CATALOGUE] got {len(cats)}; probe on {len(mids)} markets -> {mids}")

    # ------- books avec SP_AVAILABLE -------
    proj = price_projection(
        price_data=["EX_BEST_OFFERS","SP_AVAILABLE"],
        ex_best_offers_overrides=ex_best_offers_overrides(best_prices_depth=3),
        virtualise=True, rollover_stakes=False,
    )
    try:
        print("[BOOKS] listMarketBook (SP_AVAILABLE) ...")
        t0 = time.time()
        books = client.betting.list_market_book(market_ids=mids, price_projection=proj)
        print(f"[BOOKS] got {len(books)} books in {int((time.time()-t0)*1000)} ms")
    except APIError as e:
        print("[BOOKS] APIError:", e)
        books = []

    # ------- rendu -------
    # on fait une map marketId -> catalogue (pour nom / start / bsp flag)
    by_id = {c.market_id: c for c in cats}
    for b in books:
        mid = getattr(b, "market_id", "?")
        md  = getattr(b, "market_definition", None)
        cat = by_id.get(mid)
        name = getattr(cat, "market_name", None) or getattr(md, "name", None)
        start = getattr(cat, "market_start_time", None)
        tto_s = _tto(start)
        # type
        mtype = getattr(md, "market_type", None)
        if not isinstance(mtype, str):
            sname = (name or "").lower()
            mtype = "PLACE" if "place" in sname else ("WIN" if ("win" in sname or "winner" in sname) else None)
        # SP activé ?
        bsp_market = getattr(md, "bsp_market", None)
        # compteur runners avec near/far
        total = 0; have_sp = 0; sample = []
        for r in getattr(b, "runners", []) or []:
            total += 1
            sp = getattr(r, "sp", None)
            near = getattr(sp, "near_price", None) if sp is not None else None
            far  = getattr(sp, "far_price", None)  if sp is not None else None
            if near is not None or far is not None:
                have_sp += 1
                if len(sample) < 2:
                    sample.append((getattr(r,"selection_id",None), getattr(r,"last_price_traded",None), near, far))
        print(f"[SPCHK] market={mid} type={mtype} bsp_market={bsp_market} t_to_off={None if tto_s is None else round(tto_s)}s  runners={total}  near/far={have_sp}/{total}")
        if sample:
            for sid, ltp, near, far in sample:
                print(f"        sample sid={sid} LTP={ltp} NEAR={near} FAR={far}")
        else:
            print("        (no runner with near/far)")

    try:
        client.logout(); print("[LOGOUT] OK")
    except Exception:
        pass

if __name__ == "__main__":
    main()
