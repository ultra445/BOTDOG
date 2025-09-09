# quick_dryrun_poll.py — DRY-RUN réel (Betfair REST), priceProjection minimal + fallbacks
from __future__ import annotations
import os, sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List

from betfairlightweight import APIClient, filters

# Chemin vers ./src pour "from dogbot ..."
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Charge .env explicitement (facultatif si $env déjà définies)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from dogbot.indexer import MarketIndex, make_course_id
from dogbot.types import MarketIndexEntry
from dogbot.strategy.back_win_1 import BackWin1
from dogbot.executor import Executor


def need(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    raise SystemExit("Manque des variables d'env: " + " / ".join(names))


def iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_index(catalogue_markets) -> MarketIndex:
    entries: List[MarketIndexEntry] = []
    for m in catalogue_markets:
        event = getattr(m, "event", None)
        venue = getattr(event, "venue", None)
        cc = getattr(event, "country_code", None) or getattr(event, "countryCode", None)
        open_utc = getattr(m, "market_start_time", None) or getattr(m, "marketStartTime", None)
        if isinstance(open_utc, str):
            open_utc = datetime.fromisoformat(open_utc.replace("Z", "+00:00"))
        if open_utc is not None:
            if open_utc.tzinfo is None:
                open_utc = open_utc.replace(tzinfo=timezone.utc)
            else:
                open_utc = open_utc.astimezone(timezone.utc)
        race_no = None
        course_id = make_course_id(venue, open_utc, cc, race_no) if open_utc else None
        entries.append(
            MarketIndexEntry(
                market_id=m.market_id,
                market_type=getattr(m, "market_type", None) or getattr(m, "marketType", None) or "WIN",
                event_id=getattr(event, "id", None),
                event_open_utc=open_utc,
                venue=venue,
                country_code=cc,
                event_local_date=None,
                race_number=race_no,
                course_id=course_id,
            )
        )
    return MarketIndex(entries)


def main() -> None:
    username  = need("BF_USER", "BETFAIR_USERNAME", "BETFAIR_USER")
    password  = need("BF_PASS", "BETFAIR_PASSWORD")
    app_key   = need("BF_APP_KEY", "BETFAIR_APP_KEY")
    certs_dir = need("BF_CERTS_PATH", "BETFAIR_CERTS_PATH")  # ex: C:\betfair-certs

    p = Path(certs_dir)
    if not p.is_dir():
        raise SystemExit(f"Le chemin certs n'est pas un dossier: {p}")

    # Betfair attend ici un DOSSIER (ta version bflw n'aime pas le tuple)
    client = APIClient(username=username, password=password, app_key=app_key, certs=certs_dir)
    print("[LOGIN] connecting...")
    client.login()
    print("[LOGIN] OK")

    now = datetime.now(timezone.utc)
    lookahead = int(os.environ.get("LOOKAHEAD_MINUTES", "60"))
    t_from = iso_z(now)
    t_to   = iso_z(now + timedelta(minutes=lookahead))

    market_filter = filters.market_filter(
        text_query="Greyhound",
        market_type_codes=["WIN"],
        market_start_time={"from": t_from, "to": t_to},
        in_play_only=False,
    )

    print("[CATALOGUE] fetching...")
    catalogue = client.betting.list_market_catalogue(
        filter=market_filter,
        market_projection=["EVENT", "MARKET_START_TIME", "RUNNER_METADATA"],
        max_results=25,
        sort="FIRST_TO_START",
    )
    print(f"[CATALOGUE] found {len(catalogue)} markets")
    if not catalogue:
        print("[CATALOGUE] 0 marché — élargis LOOKAHEAD_MINUTES ou enlève text_query")
        client.logout()
        return

    index = build_index(catalogue)
    strategy = BackWin1()
    executor = Executor(client=None, strategy=strategy, market_index=index, dry_run=True, data_dir="./data")

    market_ids = [m.market_id for m in catalogue[:5]]

    # --- Tentative 1 : priceProjection minimal (évite DSC-0018) ---
    price_projection = {"priceData": ["EX_LTP"]}
    print("[BOOKS] fetching (minimal priceProjection EX_LTP)...")
    try:
        books = client.betting.list_market_book(market_ids=market_ids, price_projection=price_projection)
    except Exception as e:
        print(f"[BOOKS] first attempt failed: {e}")
        # --- Fallback 1 : EX_BEST_OFFERS avec bestPricesDepth ---
        price_projection = {"priceData": ["EX_BEST_OFFERS"], "exBestOffersOverrides": {"bestPricesDepth": 3}}
        print("[BOOKS] retry with EX_BEST_OFFERS (bestPricesDepth=3)...")
        try:
            books = client.betting.list_market_book(market_ids=market_ids, price_projection=price_projection)
        except Exception as e2:
            print(f"[BOOKS] batch retry failed: {e2}")
            # --- Fallback 2 : minimal par marché pour isoler un éventuel ID invalide ---
            books = []
            for mid in market_ids:
                try:
                    b = client.betting.list_market_book(market_ids=[mid], price_projection={"priceData": ["EX_LTP"]})
                    books.extend(b)
                except Exception as e3:
                    print(f"[BOOKS] skip {mid}: {e3}")

    print(f"[BOOKS] got {len(books)} books")
    for b in books:
        try:
            executor.process_book(b)
        except Exception as e:
            print(f"[PROCESS_ERR] {getattr(b, 'market_id', '?')}: {e}")

    client.logout()
    print("[DONE] Regarde ./data/<date>_snapshots.csv et les logs [DRY]")


if __name__ == "__main__":
    main()
