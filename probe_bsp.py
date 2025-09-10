# probe_bsp.py — vérifie si les marchés ont BSP activé et si near/far arrivent
# Prérequis env : BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY, BETFAIR_CERTS_PATH

from datetime import datetime, timedelta, timezone
import os
from betfairlightweight import APIClient
from betfairlightweight.filters import market_filter, price_projection

def main():
    lookahead_min = int(os.getenv("LOOKAHEAD_MINUTES", "60"))
    now = datetime.now(timezone.utc)
    to = now + timedelta(minutes=lookahead_min)

    client = APIClient(
        username=os.getenv("BETFAIR_USERNAME"),
        password=os.getenv("BETFAIR_PASSWORD"),
        app_key=os.getenv("BETFAIR_APP_KEY"),
        certs=os.getenv("BETFAIR_CERTS_PATH"),
    )
    print("[LOGIN] connecting...")
    client.login()
    print("[LOGIN] OK")

    # 1) Catalogue Greyhounds WIN+PLACE
    mf = market_filter(
        event_type_ids=["4339"],  # Greyhound Racing
        market_countries=[c.strip() for c in os.getenv("DOG_COUNTRIES","GB,IE,AU,NZ").split(",")],
        market_type_codes=[t.strip() for t in os.getenv("DOG_MARKET_TYPES","WIN,PLACE").split(",")],
        market_start_time={"from": now.isoformat().replace("+00:00","Z"), "to": to.isoformat().replace("+00:00","Z")},
    )
    cats = client.betting.list_market_catalogue(
        filter=mf, max_results=12, market_projection=["EVENT","MARKET_START_TIME","RUNNER_DESCRIPTION"]
    )
    mids = [c.market_id for c in cats][:6]
    print(f"[CATALOGUE] got {len(cats)}; probe on {len(mids)} markets -> {mids}")

    # 2) Books avec SP_AVAILABLE
    pp = price_projection(price_data=["EX_BEST_OFFERS","SP_AVAILABLE"], virtualise=True, rollover_stakes=False)
    print("[BOOKS] listMarketBook (SP_AVAILABLE) ...")
    books = client.betting.list_market_book(market_ids=mids, price_projection=pp)
    print(f"[BOOKS] got {len(books)} books")

    # 3) Diagnostic par marché
    for b in books:
        md = getattr(b, "market_definition", None)
        mt = (getattr(md, "market_type", None) or "").upper() if md else None
        bspm = getattr(md, "bsp_market", None) if md else None  # <-- BSP activé ?
        st = getattr(md, "market_time", None)
        if st and st.tzinfo is None:
            st = st.replace(tzinfo=timezone.utc)
        tto = int((st - datetime.now(timezone.utc)).total_seconds()) if st else None

        near_ok = far_ok = 0
        for r in getattr(b, "runners", []) or []:
            sp = getattr(r, "sp", None)
            near = getattr(sp, "near_price", None) if sp else None
            far  = getattr(sp, "far_price", None)  if sp else None
            if near is not None: near_ok += 1
            if far  is not None: far_ok  += 1

        print(f"[SPCHK] market={b.market_id} type={mt} bsp_market={bspm} t_to_off={tto}s  runners={len(getattr(b,'runners',[]) or [])}  near/far={near_ok}/{far_ok}")
        if near_ok == 0 and far_ok == 0:
            print("        (no runner with near/far)")

    client.logout()
    print("[LOGOUT] OK")

if __name__ == "__main__":
    main()
