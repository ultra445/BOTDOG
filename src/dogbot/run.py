# src/dogbot/run.py
from __future__ import annotations

import os
import sys
from dotenv import load_dotenv
from loguru import logger

from .betfair_client import BetfairClient
from .executor import Executor
from .risk import RiskLimits, RiskManager
from .strategies import build_strategies_from_env


def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int_list(env_name: str, default_csv: str) -> list[int]:
    raw = os.getenv(env_name, default_csv)
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            pass
    return out


def main() -> None:
    load_dotenv()

    # --- Logging minimal/clair
    logger.remove()
    logger.add(lambda m: print(m, end=""), level=os.getenv("LOG_LEVEL", "INFO"))

    # --- Creds / chemins certs
    username = os.getenv("BETFAIR_USERNAME") or os.getenv("BF_USERNAME") or os.getenv("BF_USER")
    password = os.getenv("BETFAIR_PASSWORD") or os.getenv("BF_PASSWORD") or os.getenv("BF_PASS")
    app_key  = os.getenv("BETFAIR_APP_KEY") or os.getenv("BF_APP_KEY")
    certs_path = os.getenv("BETFAIR_CERTS_PATH") or os.getenv("BF_CERTS_PATH")

    if not (username and password and app_key and certs_path):
        logger.error("Variables d'env incomplètes: il faut BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY, BETFAIR_CERTS_PATH.")
        sys.exit(1)

    # --- Connexion
    logger.info("Connexion Betfair…")
    client = BetfairClient(
        username=username,
        password=password,
        app_key=app_key,
        certs_path=certs_path,
    )
    client.login()
    logger.info("Connexion OK.")

    # --- Paramètres de scan
    countries_csv = os.getenv("DOG_COUNTRIES", "GB,IE,AU,NZ")
    market_types_csv = os.getenv("DOG_MARKET_TYPES", "WIN,PLACE")
    lookahead = int(os.getenv("LOOKAHEAD_MINUTES", "10"))

    countries = [c.strip() for c in countries_csv.split(",") if c.strip()]
    market_types = [m.strip().upper() for m in market_types_csv.split(",") if m.strip()]

    logger.info(f"Scan catalogue… (countries={countries}, market_types={market_types}, lookahead={lookahead}min)")
    market_ids = client.scan_catalogue(
        countries=countries,
        market_types=market_types,
        lookahead_minutes=lookahead,
    )
    logger.info(f"Marchés trouvés: {len(market_ids)}")

    if not market_ids:
        logger.warning("Aucun marché à surveiller. Ajuste LOOKAHEAD_MINUTES (ex: 60) ou l'heure de trading.")
        return

    # --- Risk
    limits = RiskLimits(
        max_daily_exposure=float(os.getenv("MAX_DAILY_EXPOSURE", "50")),
        max_market_stake=float(os.getenv("MAX_MARKET_STAKE", "2")),
        block_in_play=_get_bool("BLOCK_IN_PLAY", True),
        trading_start_hhmm=os.getenv("TRADING_START_HHMM", "00:00"),
        trading_end_hhmm=os.getenv("TRADING_END_HHMM", "23:59"),
    )
    risk = RiskManager(limits)

    # --- Strategies
    strategy_manager = build_strategies_from_env(
        default_unit=min(limits.max_market_stake, 2.0)
    )
    logger.info("Strategies enabled: " + ", ".join(s.name for s in strategy_manager.strategies))

    # --- Snapshots
    snapshot_enabled = _get_bool("SNAPSHOT_ENABLED", False) or _get_bool("SNAPSHOT_ENABLE", False)
    snapshot_dir = os.getenv("SNAPSHOT_DIR", os.path.join(os.getcwd(), "snapshots"))
    snapshot_seconds = _parse_int_list("SNAPSHOT_SECONDS", os.getenv("SNAPSHOT_TIMES", "300,150,80,45,2"))
    snapshot_tolerance = int(os.getenv("SNAPSHOT_TOLERANCE", "1"))
    snapshot_only_if_decision = _get_bool("SNAPSHOT_ONLY_IF_DECISION", False)

    # --- Executor
    exe = Executor(
        client=client,
        risk=risk,
        strategy_manager=strategy_manager,
        snapshot_enabled=snapshot_enabled,
        snapshot_dir=snapshot_dir,
        snapshot_seconds=snapshot_seconds,
        snapshot_tolerance=snapshot_tolerance,
        snapshot_only_if_decision=snapshot_only_if_decision,
    )

    logger.info("Démarrage de la boucle de polling…")
    try:
        exe.run(market_ids)
    except KeyboardInterrupt:
        logger.info("Arrêt demandé (CTRL+C).")
    except Exception as e:
        logger.exception(f"Erreur fatale dans la boucle de polling: {e}")


if __name__ == "__main__":
    main()
