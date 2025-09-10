# sp_probe.py — probe SP near/far (BSP estimés)
# Objectif: vérifier si listMarketBook avec SP_AVAILABLE retourne bien sp.near_price & sp.far_price

from __future__ import annotations
import os, sys, time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from dotenv import load_dotenv
from betfairlightweight import APIClient
from betfairlightweight.filters import market_filter, price_projection, ex_best_offers_overrides
from betfairlightweight.exceptions import APIError

def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def main():
    load_dotenv()
    user  = os.environ.get("BF_USER")
    pwd   = os.environ.get("BF_PASS")
    appk  = os.environ.get("BF_APP_KEY")
    certs = os.environ.get("BF_CERTS_PATH")
    if not (user and pwd and appk and certs):
        print("Manque des variables d'env (BF_USER/BF_PASS/BF_APP_KEY/BF_CERTS_PATH)")
        return

    lookahead_minutes = int(os.environ.get("LOOKAHEAD_MINUTES", "60") or "60")

    client = APIClient(username=user, password=pwd, app_key=appk, certs=certs)
    print(f"[BOOT] {_now()}  lookahead={lookahead_minutes}m")
    print("[LOGIN] connecting...")
    client.login()
    print("[LOGIN] OK")

    now = datetime.now(timezone.utc)
    to  = now + timedelta(minutes=lookahead_minutes)
    mf = market_filter(
        event_type_ids=["4339"],  # greyhound racing
        market_type_codes=["WIN","PLACE"],
        market_start_time={"from": now.isoformat().replace("+00:00","Z"),
                           "to":   to.isoformat().replace("+00:00","Z")},
    )
    print("[CATALOGUE] fetching...")
    cats = client.betting.list_market_catalogue(
        filter=mf,
        market_projection=["EVENT","MARKET_START_TIME","RUNNER_DESCRIPTION","RUNNER_METADATA"],
        sort="FIRST_TO_START",
        max_results=12,   # on prend un petit échantillon
    )
    mids: List[str] = [c.market_id for c in cats][:5]
    print(f"[CATALOGUE] got {len(cats)}; probe on {len(mids)} markets -> {mids}")

    proj = price_projection(
        price_data=["EX_BEST_OFFERS","SP_AVAILABLE"],          # <<<<<< SP_AVAILABLE obligatoire pour near/far
        ex_best_offers_overrides=ex_best_offers_overrides(best_prices_depth=3),
        virtualise=True,
        rollover_stakes=False,
    )

    try:
        print("[BOOKS] calling listMarketBook with SP_AVAILABLE ...")
        t0 = time.time()
        books = client.betting.list_market_book(market_ids=mids, price_projection=proj)
        print(f"[BOOKS] got {len(books)} books in {int((time.time()-t0)*1000)} ms")
    except APIError as e:
        print("[BOOKS] APIError:", e)
        print("Si code 'DSC-0018', l'API refuse la charge -> réduire l'échantillon ou retirer SP_AVAILABLE.")
        books = []

    for b in books:
        mid = getattr(b, "market_id", "?")
        md  = getattr(b, "market_definition", None)
        mtype = getattr(md, "market_type", None)
        if not isinstance(mtype, str):
            # heuristique basique par nom si le type n'est pas renseigné
            name = getattr(md, "name", None) or getattr(b, "market_name", None)
            s = (name or "").lower()
            mtype = "PLACE" if "place" in s else ("WIN" if ("win" in s or "winner" in s) else None)

        total = 0
        have_sp = 0
        sample = []
        for r in getattr(b, "runners", []) or []:
            total += 1
            sp = getattr(r, "sp", None)
            near = getattr(sp, "near_price", None) if sp is not None else None
            far  = getattr(sp, "far_price", None)  if sp is not None else None
            if near is not None or far is not None:
                have_sp += 1
                if len(sample) < 2:
                    sid = getattr(r, "selection_id", None)
                    ltp = getattr(r, "last_price_traded", None)
                    sample.append((sid, ltp, near, far))

        print(f"[SPCHK] market={mid} type={mtype}  runners={total}  near/far present={have_sp}/{total}")
        if sample:
            for (sid, ltp, near, far) in sample:
                print(f"        sample sid={sid}  LTP={ltp}  NEAR={near}  FAR={far}")
        else:
            print("        (no runner with near/far returned)")

    try:
        client.logout()
        print("[LOGOUT] OK")
    except Exception:
        pass

if __name__ == "__main__":
    main()
