# quick_dryrun_poll.py — DRY-RUN réel (Betfair REST)
from __future__ import annotations
import os, sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List

# Ajoute ./src au chemin d'import pour "from dogbot ..."
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Charger .env explicitement depuis la racine du projet
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from betfairlightweight import APIClient, filters

from dogbot.indexer import MarketIndex, make_course_id
from dogbot.types import MarketIndexEntry
from dogbot.strategy.back_win_1 import BackWin1
from dogbot.executor import Executor


def _pick(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def _need(*names: str) -> str:
    v = _pick(*names)
    if not v:
        raise SystemExit("Manque des variables d'env: " + " / ".join(names))
    return v


def _resolve_certs(certs_path: str) -> str | tuple[str, str]:
    p = Path(certs_path)
    if p.is_dir():
        crt = p / "client-2048.crt"
        key = p / "client-2048.key"
        if crt.exists() and key.exists():
            return (str(crt), str(key))
        crt_files = list(p.glob("*.crt"))
        key_files = list(p.glob("*.key"))
        if crt_files and key_files:
            return (str(crt_files[0]), str(key_files[0]))
        return str(p)
    else:
        return str(p)


def _iso_z(dt: datetime) -> str:
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
        race_no = None  # TODO: parser si dispo
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
    username = _need("BF_USER", "BETFAIR_USERNAME", "BETFAIR_USER")
    password = _need("BF_PASS", "BETFAIR_PASSWORD")
    app_key  = _need("BF_APP_KEY", "BETFAIR_APP_KEY")
    certs_in = _need("BF_CERTS_PATH", "BETFAIR_CERTS_PATH")

    certs = _resolve_certs(certs_in)

    client = APIClient(username=username, password=password, app_key=app_key, certs=certs)
    print("[LOGIN] connecting...")
    client.login()
    print("[LOGIN] OK")

    now = datetime.now(timezone.utc)
    lookahead = int(os.environ.get("LOOKAHEAD_MINUTES", "60"))
    t_from = _iso_z(now)
    t_to = _iso_z(now + timedelta(minutes=lookahead))

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
    price_projection = filters.price_projection(price_data=["EX_LTP"])  # suffisant pour fav LPT

    print("[BOOKS] fetching...")
    books = client.betting.list_market_book(market_ids=market_ids, price_projection=price_projection)
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
