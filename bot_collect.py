# bot_collect.py — baseline stable
# - Charge .env simplement (même logique que quick_login_test)
# - Support WIN + PLACE, polling REST en paquets
# - Option INCLUDE_SP_IN_BOOK (False par défaut) pour tenter BSP projected
# - Fallback propre (sans SP_AVAILABLE) si l'appel échoue (ex. DSC-0018)
# - Shim BF_* -> BETFAIR_* pour compatibilité avec ton .env actuel

from __future__ import annotations
import os
import time
from datetime import datetime, timedelta, timezone
from typing import List

from dotenv import load_dotenv

# === Chargement .env "simple" ===
# (cherche .env dans le répertoire courant où tu lances Python)
load_dotenv()

# --- Shim compatibilité : mappe BF_* -> BETFAIR_* si besoin ---
def _normalize_env_aliases():
    alias = {
        "BF_USER": "BETFAIR_USERNAME",
        "BF_PASS": "BETFAIR_PASSWORD",
        "BF_APP_KEY": "BETFAIR_APP_KEY",
        "BF_CERTS_PATH": "BETFAIR_CERTS_PATH",
    }
    for src, dst in alias.items():
        if not os.getenv(dst) and os.getenv(src):
            os.environ[dst] = os.getenv(src)
_normalize_env_aliases()
# --------------------------------------------------------------

# ====== Imports projet ======
try:
    from src.dogbot.executor import Executor
except Exception:
    from dogbot.executor import Executor  # type: ignore

try:
    from src.dogbot.indexer import MarketIndex  # type: ignore
except Exception:
    try:
        from dogbot.indexer import MarketIndex  # type: ignore
    except Exception:
        class MarketIndex(dict):
            def __init__(self, *a, **k): super().__init__()

try:
    from src.dogbot.strategy import StrategyManager
except Exception:
    try:
        from dogbot.strategy import StrategyManager  # type: ignore
    except Exception:
        class StrategyManager:
            @staticmethod
            def from_env():
                class _Noop:
                    name = "NOOP"
                    def decide_all(self, *a, **k): return []
                return _Noop()

# ====== Betfair client + filters ======
from betfairlightweight import APIClient
from betfairlightweight.filters import market_filter, price_projection
from betfairlightweight.exceptions import APIError


def _env_list(name: str, default_csv: str) -> List[str]:
    raw = os.getenv(name, default_csv)
    return [x.strip() for x in raw.split(",") if x.strip()]

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def main():
    # ====== ENV / paramètres ======
    LOOKAHEAD_MINUTES = int(os.getenv("LOOKAHEAD_MINUTES", "60"))
    POLL_INTERVAL_S   = float(os.getenv("POLL_INTERVAL_S", "2"))
    SCAN_EVERY_S      = float(os.getenv("SCAN_EVERY_S", "240"))
    MAX_MARKETS       = int(os.getenv("MAX_MARKETS", "20"))
    BATCH_SIZE        = int(os.getenv("BATCH_SIZE", "5"))

    DOG_COUNTRIES     = _env_list("DOG_COUNTRIES", "GB,IE,AU,NZ")
    DOG_MARKET_TYPES  = _env_list("DOG_MARKET_TYPES", "WIN,PLACE")

    DRY_RUN           = str(os.getenv("DRY_RUN", "true")).lower() in ("1", "true", "yes", "on")
    DATA_DIR          = os.getenv("DATA_DIR", "./data")

    INCLUDE_SP        = str(os.getenv("INCLUDE_SP_IN_BOOK", "false")).lower() in ("1","true","yes","on")

    print(f"[BOOT] {_now_utc().isoformat().replace('+00:00','Z')}  "
          f"lookahead={LOOKAHEAD_MINUTES}m scan_every={int(SCAN_EVERY_S)}s "
          f"poll={POLL_INTERVAL_S:.1f}s batch={BATCH_SIZE} max_markets={MAX_MARKETS} include_sp={INCLUDE_SP}")

    # ====== Client Betfair ======
    username = os.getenv("BETFAIR_USERNAME")
    password = os.getenv("BETFAIR_PASSWORD")
    app_key  = os.getenv("BETFAIR_APP_KEY")
    certs    = os.getenv("BETFAIR_CERTS_PATH")

    if not all([username, password, app_key, certs]):
        print("[ENV] manque au moins une variable BETFAIR_* (USERNAME/PASSWORD/APP_KEY/CERTS_PATH)")
        return

    client = APIClient(username=username, password=password, app_key=app_key, certs=certs)

    print("[LOGIN] connecting...")
    client.login()
    print("[LOGIN] OK")

    # ====== Stratégie + Executor ======
    strategy = StrategyManager.from_env()
    mindex   = MarketIndex([])  # index vide OK
    executor = Executor(client=client, strategy=strategy, market_index=mindex, dry_run=DRY_RUN, data_dir=DATA_DIR)

    last_scan_ts = 0.0
    market_ids: List[str] = []

    try:
        while True:
            now_ts = time.time()
            now    = _now_utc()

            # ---- Re-scan périodique du catalogue
            if (now_ts - last_scan_ts) >= SCAN_EVERY_S or not market_ids:
                last_scan_ts = now_ts
                fr = now
                to = now + timedelta(minutes=LOOKAHEAD_MINUTES)

                mf = market_filter(
                    event_type_ids=["4339"],  # Greyhound Racing
                    market_countries=DOG_COUNTRIES,
                    market_type_codes=DOG_MARKET_TYPES,
                    market_start_time={
                        "from": fr.isoformat().replace("+00:00", "Z"),
                        "to":   to.isoformat().replace("+00:00", "Z"),
                    },
                )
                print("[CATALOGUE] fetching...")
                cats = client.betting.list_market_catalogue(
                    filter=mf,
                    max_results=MAX_MARKETS,
                    market_projection=["EVENT", "MARKET_START_TIME", "RUNNER_DESCRIPTION", "MARKET_DESCRIPTION"]
                )
                market_ids = [c.market_id for c in cats][:MAX_MARKETS]
                print(f"[CATALOGUE] found {len(market_ids)} markets")

            # ---- Poll des books en paquets
            if not market_ids:
                time.sleep(POLL_INTERVAL_S)
                continue

            # Projections (avec ou sans SP_AVAILABLE selon env)
            if INCLUDE_SP:
                pp_primary = price_projection(
                    price_data=["EX_BEST_OFFERS", "SP_AVAILABLE"],
                    virtualise=True,
                    rollover_stakes=False,
                    ex_best_offers_overrides={"bestPricesDepth": 3},
                )
            else:
                pp_primary = price_projection(
                    price_data=["EX_BEST_OFFERS"],
                    virtualise=True,
                    rollover_stakes=False,
                    ex_best_offers_overrides={"bestPricesDepth": 3},
                )

            pp_fallback = price_projection(
                price_data=["EX_BEST_OFFERS"],
                virtualise=True,
                rollover_stakes=False,
                ex_best_offers_overrides={"bestPricesDepth": 3},
            )

            # Découpe en chunks
            chunks = [market_ids[i:i+BATCH_SIZE] for i in range(0, len(market_ids), BATCH_SIZE)]
            print(f"[LOOP] polling books for {len(market_ids)} markets in {len(chunks)} chunks (batch={BATCH_SIZE})")

            for idx, chunk_ids in enumerate(chunks, 1):
                print(f"[BOOKS] chunk {idx}/{len(chunks)} start: {len(chunk_ids)} ids -> {chunk_ids}")
                t0 = time.time()
                try:
                    books = client.betting.list_market_book(market_ids=chunk_ids, price_projection=pp_primary)
                except APIError as e:
                    print("[BOOKS] primary failed, fallback EX_BEST_OFFERS only:", e)
                    books = client.betting.list_market_book(market_ids=chunk_ids, price_projection=pp_fallback)
                except Exception as e:
                    print("[BOOKS] primary unexpected error, fallback EX_BEST_OFFERS only:", e)
                    books = client.betting.list_market_book(market_ids=chunk_ids, price_projection=pp_fallback)

                elapsed_ms = int((time.time() - t0) * 1000)
                print(f"[BOOKS] chunk {idx}/{len(chunks)} got {len(books)} books in {elapsed_ms} ms")

                # Passe chaque book à l'executor
                for b in books:
                    try:
                        executor.set_diagnostics(fetch_latency_ms=elapsed_ms)
                        executor.process_book(b)
                    except Exception as e:
                        mid = getattr(b, "market_id", None)
                        print(f"[EXECUTOR_ERR] {mid}: {e}")

                # Petit lissage entre paquets
                time.sleep(max(0.0, POLL_INTERVAL_S / 2.0))

            # Rythme de boucle global
            time.sleep(POLL_INTERVAL_S)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            client.logout()
            print("[LOGOUT] OK")
        except Exception:
            pass


if __name__ == "__main__":
    main()
