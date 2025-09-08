# src/dogbot/run.py
from __future__ import annotations

import os
from datetime import datetime, timezone

from loguru import logger

# Chargement facultatif d'un .env (si présent)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from .betfair_client import BetfairClient
from .executor import Executor

# Essaye d'importer la stratégie si elle existe
try:
    from .strategies import StrategyManager  # type: ignore
    DEFAULT_STRATEGY = StrategyManager()  # instance !
except Exception:
    DEFAULT_STRATEGY = None


def _require_env(*keys: str) -> None:
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            "Variables d'environnement manquantes: " + " / ".join(missing)
        )


def main() -> None:
    logger.info("Connexion Betfair…")

    # --- ENV requis
    _require_env("BETFAIR_USERNAME", "BETFAIR_PASSWORD", "BETFAIR_APP_KEY")

    username = os.getenv("BETFAIR_USERNAME")
    password = os.getenv("BETFAIR_PASSWORD")
    app_key = os.getenv("BETFAIR_APP_KEY")

    # Dossier des certs depuis ENV (sinon, laisse None pour fallback dans BetfairClient)
    certs_path = os.getenv("BETFAIR_CERTS_PATH") or os.getenv("BF_CERTS_PATH")

    client = BetfairClient(
        username=username, password=password, app_key=app_key, certs_path=certs_path
    )

    client.login()
    logger.info("Connexion OK.")

    # --- Paramètres de scan
    countries = ["GB", "IE", "AU", "NZ"]
    market_types = ["WIN"]
    lookahead_minutes = int(os.getenv("LOOKAHEAD_MINUTES", "60"))

    logger.info(
        "Scan catalogue… (countries={}, market_types={}, lookahead={}min)",
        countries,
        market_types,
        lookahead_minutes,
    )

    market_ids = client.scan_catalogue(
        countries=countries,
        market_types=market_types,
        lookahead_minutes=lookahead_minutes,
    )
    logger.info("Marchés trouvés: {}", len(market_ids))

    if not market_ids:
        logger.warning(
            "Aucun marché à surveiller. Ajuste LOOKAHEAD_MINUTES (ex: 60/120) ou l'heure de trading."
        )
        return

    # --- Executor
    dry_run = os.getenv("DRY_RUN", "1") not in ("0", "false", "False")
    poll_interval = os.getenv("POLL_INTERVAL", "2.0")
    snapshot_seconds = os.getenv("SNAPSHOT_SECONDS")  # None -> pas de snapshots

    exe = Executor(
        client=client,
        strategy=DEFAULT_STRATEGY,      # << instance (ou None)
        dry_run=dry_run,
        poll_interval=poll_interval,
        snapshot_seconds=snapshot_seconds,
    )

    logger.info("Strategies enabled: {}", "BACK_WIN_1" if DEFAULT_STRATEGY else "NONE")
    logger.info("Démarrage de la boucle de polling…")
    exe.run(market_ids)


if __name__ == "__main__":
    main()
