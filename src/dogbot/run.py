import os
import sys
import logging
from typing import Optional

# Logging simple (tu peux brancher loguru/structlog si tu veux)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

# Charger .env si disponible
try:
    from dotenv import load_dotenv
    load_dotenv()  # charge .env local
except Exception:
    pass

from .betfair_client import BetfairClient
from .executor import Executor

# Stratégie : on essaie d'importer la tienne ; sinon on crée un fallback no-op
def _build_strategy() -> object:
    try:
        # Adapte ce chemin si ta StrategyManager est ailleurs
        from .strategy import StrategyManager  # type: ignore
        return StrategyManager(enabled=["BACK_WIN_1"])
    except Exception as e:
        logger.warning("Using fallback strategy (no-op) because import failed: %s", e)

        class FallbackStrategy:
            def decide_all(self, market_book, market_index_entry, now_utc):
                # Pas de signal → pas d'ordres
                return []

        return FallbackStrategy()


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name, default)
    if val is None:
        return None
    val = val.strip()
    return val if val != "" else None


def main() -> None:
    logger.info("Connexion Betfair…")

    username = _env_str("BETFAIR_USERNAME")
    password = _env_str("BETFAIR_PASSWORD")
    app_key = _env_str("BETFAIR_APP_KEY")
    certs_path = _env_str("BETFAIR_CERTS_PATH") or r"C:\betfair-certs"

    missing = [n for n, v in [
        ("BETFAIR_USERNAME", username),
        ("BETFAIR_PASSWORD", password),
        ("BETFAIR_APP_KEY", app_key),
    ] if not v]
    if missing:
        raise RuntimeError(
            f"Variables d'environnement manquantes: { ' / '.join(missing) }"
        )

    client = BetfairClient(
        username=username,
        password=password,
        app_key=app_key,
        certs_dir=certs_path,
    )
    client.login()
    logger.info("Connexion OK.")

    # --- Scan catalogue ---
    countries = ["GB", "IE", "AU", "NZ"]
    market_types = ["WIN"]
    lookahead_minutes = int(_env_str("LOOKAHEAD_MINUTES", "60") or "60")
    logger.info(
        "Scan catalogue… (countries=%s, market_types=%s, lookahead=%dmin)",
        countries, market_types, lookahead_minutes
    )
    market_ids = client.scan_catalogue(
        countries=countries,
        market_types=market_types,
        lookahead_minutes=lookahead_minutes,
        max_results=1000,
    )
    logger.info("Marchés trouvés: %d", len(market_ids))
    if not market_ids:
        logger.warning("Aucun marché à surveiller. Ajuste LOOKAHEAD_MINUTES (ex: 60) ou l'heure de trading.")
        return

    # --- Stratégie ---
    strategy = _build_strategy()
    logger.info("Strategies enabled: BACK_WIN_1")

    # --- Executor ---
    snapshot_enabled = _env_str("SNAPSHOT_ENABLED", "1") not in ("0", "false", "False", "no", "No")
    snapshot_path = _env_str("SNAPSHOT_PATH", "./data") or "./data"
    snapshot_period = float(_env_str("SNAPSHOT_PERIOD_SECONDS", "5.0") or "5.0")
    dry_run = _env_str("DRY_RUN", "1") not in ("0", "false", "False", "no", "No")

    logger.info("Démarrage de la boucle de polling…")
    exe = Executor(
        client=client,
        strategy=strategy,
        snapshot_enabled=snapshot_enabled,
        snapshot_path=snapshot_path,
        snapshot_period=snapshot_period,
        dry_run=dry_run,
        poll_interval=2.0,
    )
    exe.run(market_ids)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Arrêt demandé (Ctrl+C).")
        sys.exit(0)
