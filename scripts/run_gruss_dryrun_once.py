from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogbot.gruss.gruss_dryrun_engine import (
    GrussDryRunRunner,
    create_configured_gruss_feed,
    describe_state,
    print_strategy_registry_diagnostics,
    race_key,
    read_gruss_dryrun_state,
    validate_gruss_dryrun_provider_config,
)
from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH


DATA_DIR = ROOT / "data"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print("Gruss engine dry-run once")
    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")
    print("Aucun ordre ne sera envoye.")

    try:
        config = validate_gruss_dryrun_provider_config()
    except Exception as exc:
        print(f"SKIP: configuration provider invalide: {exc}")
        return 1
    print(f"providers: data={config.data_provider} order={config.order_provider}")
    print_strategy_registry_diagnostics()
    print("MOM45 buffer inactif: lecture unique, missing_mom45 attendu sauf cache executor deja alimente.")

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
        runner.evaluate(
            state.win_snapshot,
            state.place_snapshot,
            debug_strategies=args.debug_strategies,
        )
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
        print(f"trades.csv: {runner.trade_path()}")
        order_results = runner.log_gruss_order_intents(
            runner.trade_rows_since(trade_count_before),
            state.win_snapshot,
            state.place_snapshot,
        )
        if order_results:
            print(f"ordres Gruss dry-run journalises: {len(order_results)}")
            print(f"orders_gruss_dryrun.csv: {order_results[0].output_path}")
    else:
        print("no signal")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Gruss dry-run engine evaluation.")
    parser.add_argument("--debug-strategies", action="store_true", help="Print detailed PLACE strategy evaluations.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
