from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogbot.gruss.gruss_dryrun_engine import (
    GrussDryRunRunner,
    create_configured_gruss_feed,
    describe_state,
    race_key,
    read_gruss_dryrun_state,
    validate_gruss_dryrun_provider_config,
)
from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH


DATA_DIR = ROOT / "data"


def main() -> int:
    print("Gruss engine dry-run once")
    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")
    print("Aucun ordre ne sera envoye.")

    try:
        config = validate_gruss_dryrun_provider_config()
    except Exception as exc:
        print(f"SKIP: configuration provider invalide: {exc}")
        return 1
    print(f"providers: data={config.data_provider} order={config.order_provider}")

    feed = create_configured_gruss_feed(config)
    try:
        state = read_gruss_dryrun_state(feed)
    except Exception as exc:
        print(f"SKIP: lecture Gruss impossible: {exc}")
        return 1

    print(f"course detectee: {describe_state(state)}")
    if state.skip_reason:
        print(f"skip: {state.skip_reason}")
        print("no signal")
        return 0

    key = race_key(state.win_snapshot, state.place_snapshot)
    runner = GrussDryRunRunner(DATA_DIR)
    if runner.processed_store.has_seen(key):
        print(f"skip: course deja traitee ({key})")
        print("no signal")
        return 0

    trade_count_before = runner.trade_row_count()
    try:
        runner.evaluate(state.win_snapshot, state.place_snapshot)
    except Exception as exc:
        print(f"SKIP: evaluation dry-run impossible: {exc}")
        print("no signal")
        return 1

    runner.processed_store.mark_seen(
        key,
        state.win_snapshot.metadata.market_id,
        state.place_snapshot.metadata.market_id,
    )
    trade_count_after = runner.trade_row_count()
    generated = max(0, trade_count_after - trade_count_before)
    if generated:
        print(f"signaux generes: {generated}")
        print(f"trades.csv: {_trade_path()}")
    else:
        print("no signal")
    return 0


def _trade_path() -> Path:
    return DATA_DIR / f"trades_{datetime.now(timezone.utc):%Y%m%d}.csv"


def _trade_row_count() -> int:
    path = _trade_path()
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


if __name__ == "__main__":
    raise SystemExit(main())
