# src/dogbot/run.py
from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

from loguru import logger

from .config import Settings
from .betfair_client import BetfairClient
from .risk import RiskLimits, RiskManager
from .strategies import build_strategies_from_env
from .executor import Executor


def _parse_snapshot_seconds() -> list[int]:
    # Accepte SNAPSHOT_SECONDS ou SNAPSHOT_TIMES, format "300,150,80,45,2"
    raw = os.getenv("SNAPSHOT_SECONDS") or os.getenv("SNAPSHOT_TIMES") or "300,150,80,45,2"
    vals: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except ValueError:
            pass
    return vals or [300, 150, 80, 45, 2]


def main() -> None:
    load_dotenv()

    cfg = Settings.load()

    # Logging compact
    logger.remove()
    logger.add(lambda m: print(m, end=""), level=cfg.log_level)

    # --- Betfair client
    client = BetfairClient(
        user=cfg.bf_user,
        password=cfg.bf_pass,
        app_key=cfg.bf_app_key,
        certs_path=cfg.bf_certs_path,
    )
    client.login()

    # --- Scan du catalogue → liste de market_ids et index internes
    market_ids = client.scan_catalogue(
        countries=cfg.dog_countries,
        market_types=cfg.dog_market_types,
        lookahead_minutes=cfg.lookahead_minutes,
    )

    # --- Risk manager
    limits = RiskLimits(
        max_daily_exposure=cfg.max_daily_exposure,
        max_market_stake=cfg.max_market_stake,
        block_in_play=cfg.block_in_play,
        trading_start_hhmm=cfg.trading_start_hhmm,
        trading_end_hhmm=cfg.trading_end_hhmm,
    )
    risk = RiskManager(limits)

    # --- Strategies
    strategy_manager = build_strategies_from_env(default_unit=min(cfg.max_market_stake, 2.0))
    logger.info("Strategies enabled: " + ", ".join(s.name for s in strategy_manager.strategies))

    # --- Snapshots (depuis .env, avec tolérance sur les noms)
    snapshot_enabled = (os.getenv("SNAPSHOT_ENABLED", "false").lower() == "true"
                        or os.getenv("SNAPSHOT_ENABLE", "false").lower() == "true")
    snapshot_dir = os.getenv("SNAPSHOT_DIR", os.path.join(os.getcwd(), "snapshots"))
    snapshot_seconds = _parse_snapshot_seconds()
    snapshot_tolerance = int(os.getenv("SNAPSHOT_TOLERANCE", "1"))
    snapshot_only_if_decision = os.getenv("SNAPSHOT_ONLY_IF_DECISION", "false").lower() == "true"

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

    # --- Run
    exe.run(market_ids)


if __name__ == "__main__":
    main()

