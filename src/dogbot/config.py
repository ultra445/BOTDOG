from __future__ import annotations
from pydantic import BaseModel
from typing import List, Optional
import os


class Settings(BaseModel):
    # Identifiants / certifs
    bf_user: str
    bf_pass: str
    bf_app_key: str
    bf_certs_path: str

    # Portée marchés
    dog_countries: List[str]
    dog_market_types: List[str]
    lookahead_minutes: int  # utilisé au 1er scan (si continuous_scan=False, c'est aussi l'horizon de scan)

    # Risque
    max_daily_exposure: float
    max_market_stake: float
    block_in_play: bool
    trading_start_hhmm: str
    trading_end_hhmm: str

    # Logs & modes
    log_level: str
    dry_run: bool
    use_streaming: bool

    # Orchestration / boucle
    continuous_scan: bool             # si true, scan en continu 24/24
    scan_window_minutes: int          # taille de la fenêtre glissante (ex: 180)
    market_refresh_minutes: int       # fréquence de rafraîchissement de la liste des marchés
    poll_interval_sec: float          # cadence de polling des books (si pas de streaming)

    # Anti-doublons / timing
    preoff_minutes: int               # ne pas agir avant X minutes pré-off (filtre stratégie/executor)
    signal_cooldown_sec: int          # anti-spam : pause entre deux signaux sur même marché/runner
    one_bet_per_market: bool          # hard cap "1 pari max par marché" (optionnel)

    # Durcissement client / API
    max_markets_per_chunk: int        # chunk size pour list_market_book
    include_sp_in_book: bool          # inclure SP_AVAILABLE / SP_TRADED dans les books
    api_retries: int                  # nb de tentatives en cas d'erreur API
    api_retry_backoff_ms: int         # backoff (ms) entre retries
    rate_limit_sleep_ms: int          # pause (ms) entre appels pour limiter la charge

    # Catégories ON/OFF (masters)
    enable_back_win: bool
    enable_lay_win: bool
    enable_back_place: bool
    enable_lay_place: bool

    # Stakes fixes (optionnels)
    stake_back_win: Optional[float] = None
    stake_lay_win: Optional[float] = None
    stake_back_place: Optional[float] = None
    stake_lay_place: Optional[float] = None

    # Formules de mise (optionnelles, si tu veux piloter depuis .env)
    stake_formula_back_win: Optional[str] = None
    stake_formula_lay_win: Optional[str] = None
    stake_formula_back_place: Optional[str] = None
    stake_formula_lay_place: Optional[str] = None

    # ======= Snapshots (nouveau) =======
    # Active l’enregistrement CSV des variables à des instants clés pré-off
    snapshot_enabled: bool            # SNAPSHOT_ENABLED
    snapshot_dir: str                 # SNAPSHOT_DIR
    snapshot_fields: List[str]        # SNAPSHOT_FIELDS (liste de noms de colonnes)
    snapshot_when_seconds: List[int]  # SNAPSHOT_WHEN_SECONDS (ex: 300,150,80,45,2)
    snapshot_runner_limit: Optional[int] = None  # SNAPSHOT_RUNNER_LIMIT (optionnel)

    @classmethod
    def load(cls) -> "Settings":
        def env(name, default=None):
            return os.getenv(name, default)

        def env_list(name, default):
            raw = os.getenv(name)
            if not raw:
                return default
            return [s.strip() for s in raw.split(",") if s.strip()]

        def env_int_list(name, default):
            raw = os.getenv(name)
            if not raw:
                return default
            out: List[int] = []
            for s in raw.split(","):
                s = s.strip()
                if not s:
                    continue
                try:
                    out.append(int(s))
                except Exception:
                    continue
            return out if out else default

        def env_bool(name, default="false"):
            return env(name, default).lower() in ("1", "true", "yes", "on")

        def env_float_opt(name):
            v = os.getenv(name)
            if v is None or v == "":
                return None
            try:
                return float(v)
            except Exception:
                return None

        return cls(
            # Identifiants
            bf_user=env("BF_USER"),
            bf_pass=env("BF_PASS"),
            bf_app_key=env("BF_APP_KEY"),
            bf_certs_path=env("BF_CERTS_PATH"),

            # Portée
            dog_countries=env_list("DOG_COUNTRIES", ["GB", "IE"]),
            dog_market_types=env_list("DOG_MARKET_TYPES", ["WIN"]),
            lookahead_minutes=int(env("LOOKAHEAD_MINUTES", "120")),

            # Risque
            max_daily_exposure=float(env("MAX_DAILY_EXPOSURE", "100")),
            max_market_stake=float(env("MAX_MARKET_STAKE", "5")),
            block_in_play=env_bool("BLOCK_IN_PLAY", "true"),
            trading_start_hhmm=env("TRADING_START_HHMM", "00:00"),
            trading_end_hhmm=env("TRADING_END_HHMM", "23:59"),

            # Logs & modes
            log_level=env("LOG_LEVEL", "INFO"),
            dry_run=env_bool("DRY_RUN", "true"),
            use_streaming=env_bool("USE_STREAMING", "false"),

            # Orchestration
            continuous_scan=env_bool("CONTINUOUS_SCAN", "true"),
            scan_window_minutes=int(env("SCAN_WINDOW_MINUTES", "180")),
            market_refresh_minutes=int(env("MARKET_REFRESH_MINUTES", "5")),
            poll_interval_sec=float(env("POLL_INTERVAL_SEC", "0.8")),

            # Anti-doublons / timing
            preoff_minutes=int(env("PREOFF_MINUTES", "3")),
            signal_cooldown_sec=int(env("SIGNAL_COOLDOWN_SEC", "120")),
            one_bet_per_market=env_bool("ONE_BET_PER_MARKET", "false"),

            # Durcissement client / API
            max_markets_per_chunk=int(env("MAX_MARKETS_PER_CHUNK", "8")),
            include_sp_in_book=env_bool("INCLUDE_SP_IN_BOOK", "true"),
            api_retries=int(env("API_RETRIES", "3")),
            api_retry_backoff_ms=int(env("API_RETRY_BACKOFF_MS", "300")),
            rate_limit_sleep_ms=int(env("RATE_LIMIT_SLEEP_MS", "200")),

            # Catégories
            enable_back_win=env_bool("ENABLE_BACK_WIN", "true"),
            enable_lay_win=env_bool("ENABLE_LAY_WIN", "true"),
            enable_back_place=env_bool("ENABLE_BACK_PLACE", "true"),
            enable_lay_place=env_bool("ENABLE_LAY_PLACE", "true"),

            # Stakes fixes
            stake_back_win=env_float_opt("STAKE_BACK_WIN"),
            stake_lay_win=env_float_opt("STAKE_LAY_WIN"),
            stake_back_place=env_float_opt("STAKE_BACK_PLACE"),
            stake_lay_place=env_float_opt("STAKE_LAY_PLACE"),

            # Formules (optionnel)
            stake_formula_back_win=env("STAKE_FORMULA_BACK_WIN"),
            stake_formula_lay_win=env("STAKE_FORMULA_LAY_WIN"),
            stake_formula_back_place=env("STAKE_FORMULA_BACK_PLACE"),
            stake_formula_lay_place=env("STAKE_FORMULA_LAY_PLACE"),

            # Snapshots (nouveau)
            snapshot_enabled=env_bool("SNAPSHOT_ENABLED", "false"),
            snapshot_dir=env("SNAPSHOT_DIR", "snapshots"),
            snapshot_fields=env_list(
                "SNAPSHOT_FIELDS",
                # par défaut un set simple et lisible
                ["ts", "market_id", "selection_id", "runner_name", "secs_to_off",
                 "ltp", "best_back_price", "best_lay_price", "traded_volume"]
            ),
            snapshot_when_seconds=env_int_list("SNAPSHOT_WHEN_SECONDS", [300, 150, 80, 45, 2]),
            snapshot_runner_limit=int(env("SNAPSHOT_RUNNER_LIMIT", "0")) or None,
        )
