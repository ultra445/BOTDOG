# src/dogbot/run.py
from __future__ import annotations
import os
from loguru import logger

# --- charger .env le plus tôt possible ---
import os, pathlib
try:
    from dotenv import load_dotenv, find_dotenv
    _env_path = find_dotenv(usecwd=True)  # cherche .env en remontant depuis le CWD
    if _env_path:
        load_dotenv(_env_path, override=True)  # charge et ECRASE l'env si besoin
        print(f"[.env] chargé depuis: {_env_path}")
    else:
        print("[.env] introuvable depuis le dossier courant")
except Exception as _e:
    print(f"[.env] chargement ignoré ({_e})")
# -----------------------------------------


from .betfair_client import BetfairClient
from .executor import Executor

# Ajoute ce bloc très tôt dans run.py
try:
    from dotenv import load_dotenv, find_dotenv
    _env_path = find_dotenv(usecwd=True)  # cherche .env depuis le cwd en remontant
    if _env_path:
        load_dotenv(_env_path)  # n’écrase pas les variables déjà présentes
        print(f"[.env] chargé depuis: {_env_path}")
    else:
        print("[.env] fichier introuvable (aucun .env trouvé en remontant depuis le cwd)")
except Exception as _e:
    print(f"[.env] chargement ignoré ({_e})")



def _env_list(key: str, default: list[str]) -> list[str]:
    raw = os.getenv(key)
    if not raw:
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


def _as_bool(s: str | None, default: bool = False) -> bool:
    if s is None:
        return default
    s = s.strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _as_float(s: str | None, default: float | None = None) -> float | None:
    if s is None or s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default


def main() -> None:
    logger.info("Connexion Betfair…")

    # ---- Identifiants Betfair via ENV ----
    username = os.getenv("BETFAIR_USERNAME")
    password = os.getenv("BETFAIR_PASSWORD")
    app_key  = os.getenv("BETFAIR_APP_KEY")

    # Chemins des certificats (par défaut ton dossier Windows)
    certs_dir = os.getenv("BETFAIR_CERTS_DIR", r"C:\betfair-certs")
    cert_crt  = os.getenv("BETFAIR_CERT_CRT", "client-2048.crt")
    cert_key  = os.getenv("BETFAIR_CERT_KEY", "client-2048.key")

    if not (username and password and app_key):
        raise RuntimeError(
            "Variables d'environnement manquantes: "
            "BETFAIR_USERNAME / BETFAIR_PASSWORD / BETFAIR_APP_KEY"
        )

    client = BetfairClient(
        username=username,
        password=password,
        app_key=app_key,
        certs_dir=certs_dir,
        cert_crt=cert_crt,
        cert_key=cert_key,
    )

    client.login()
    logger.info("Connexion OK.")

    # ---- Paramètres de scan ----
    countries = _env_list("COUNTRIES", ["GB", "IE", "AU", "NZ"])
    market_types = _env_list("MARKET_TYPES", ["WIN"])
    lookahead_minutes = int(os.getenv("LOOKAHEAD_MINUTES", "60"))

    logger.info(
        "Scan catalogue… (countries={}, market_types={}, lookahead={}min)",
        countries, market_types, lookahead_minutes
    )

    market_ids = client.scan_catalogue(
        countries=countries,
        market_types=market_types,
        lookahead_minutes=lookahead_minutes,
    )
    logger.info("Marchés trouvés: {}", len(market_ids))
    if not market_ids:
        logger.warning("Aucun marché à surveiller. Ajuste LOOKAHEAD_MINUTES (ex: 60) ou l'heure de trading.")
        return

    logger.info("Strategies enabled: {}", os.getenv("STRATEGIES", "BACK_WIN_1"))

    # ---- Executor ----
    dry_run = _as_bool(os.getenv("DRY_RUN", "true"), True)
    poll_interval = _as_float(os.getenv("POLL_INTERVAL", "2.0"), 2.0)
    snapshot_seconds = os.getenv("SNAPSHOT_SECONDS")  # Executor sait convertir

    exe = Executor(
        client=client,
        strategy=None,               # NoOpStrategy interne si None
        dry_run=dry_run,
        poll_interval=poll_interval,
        snapshot_seconds=snapshot_seconds,
    )

    logger.info("Démarrage de la boucle de polling…")
    exe.run(market_ids)


if __name__ == "__main__":
    main()
