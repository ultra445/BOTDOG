# quick_dryrun_poll.py — DRY-RUN réel (Betfair REST), EX_BEST_OFFERS direct
from __future__ import annotations
import os, sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

from betfairlightweight import APIClient, filters

# === chemins & .env ===
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from dotenv import load_dotenv  # facultatif
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from dogbot.indexer import MarketIndex
from dogbot.strategy.back_win_1 import BackWin1
from dogbot.executor import Executor


def need(*names: str) -> str:
    """Retourne la première variable d'env trouvée parmi names, sinon exit."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    raise SystemExit("Manque des variables d'env: " + " / ".join(names))


def iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    # --- credentials & certs (MODE DOSSIER attendu par bflw) ---
    username  = need("BF_USER", "BETFAIR_USERNAME", "BETFAIR_USER")
    password  = need("BF_PASS", "BETFAIR_PASSWORD")
    app_key   = need("BF_APP_KEY", "BETFAIR_APP_KEY")
    certs_dir = need("BF_CERTS_PATH", "BETFAIR_CERTS_PATH")  # ex: C:\betfair-certs

    p = Path(certs_dir)
    if not p.is_dir():
        raise SystemExit(f"Le chemin certs n'est pas un dossier: {p}")

    client = APIClient(username=username, password=password, app_key=app_key, certs=certs_dir)
    print("[LOGIN] connecting...")
    client.login()
    print("[LOGIN] OK")

    # --- fenêtre de scan ---
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
        print("[CATALOGUE] 0 marché — augmente LOOKAHEAD_MINUTES ou enlève text_query")
        client.logout()
        return

    # --- index enrichi (runners meta + course_id + liens WIN↔PLACE) ---
    index = MarketIndex.from_catalogue(catalogue)

    # --- stratégie & exécuteur (snapshots + milestones) ---
    strategy = BackWin1()
    executor = Executor(client=None, strategy=strategy, market_index=index, dry_run=True, data_dir="./data")

    # --- lecture des books ---
    market_ids = [m.market_id for m in catalogue[:5]]
    price_projection = {"priceData": ["EX_BEST_OFFERS"], "exBestOffersOverrides": {"bestPricesDepth": 3}}

    print("[BOOKS] fetching...")
    books = client.betting.list_market_book(market_ids=market_ids, price_projection=price_projection)
    print(f"[BOOKS] got {len(books)} books")

    for b in books:
        try:
            executor.process_book(b)
        except Exception as e:
            print(f"[PROCESS_ERR] {getattr(b, 'market_id', '?')}: {e}")

    client.logout()
    print("[DONE] Regarde ./data/<date>_snapshots.csv et ./data/<date>_snapshots_runners.csv")


if __name__ == "__main__":
    main()
