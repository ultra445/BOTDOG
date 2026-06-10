from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogbot.config import DATA_PROVIDER_GRUSS_EXCEL, ORDER_PROVIDER_GRUSS_EXCEL_REAL, load_provider_config
from dogbot.gruss.gruss_dryrun_engine import (
    GrussDryRunRunner,
    ProcessedRaceStore,
    build_order_intents_from_trade_rows,
    countdown_wait_reason as active_countdown_wait_reason,
    describe_state,
    describe_current_strategy_milestone,
    gruss_region_for_snapshots,
    current_strategy_milestone,
    print_strategy_registry_diagnostics,
    race_key,
    read_gruss_dryrun_state,
    strategy_milestone_key,
)
from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH
from dogbot.gruss.gruss_feed import GrussFeed
from dogbot.gruss.gruss_momentum import GrussMomentumBuffer
from dogbot.gruss.gruss_real_orders import GrussExcelOrderProvider, GrussRealOrderContext


DATA_DIR = ROOT / "data"
TRIGGER_SECONDS = 2


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.interval <= 0:
        print("ERREUR: --interval doit etre superieur a 0")
        return 1
    if args.max_ticks is not None and args.max_ticks <= 0:
        print("ERREUR: --max-ticks doit etre superieur a 0")
        return 1

    try:
        validate_write_no_trigger_environment()
    except RuntimeError as exc:
        print(f"ERREUR SECURITE: {exc}")
        return 1

    print("Gruss watcher - WRITE NO TRIGGER")
    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")
    print("Odds/Stake peuvent etre ecrits. La cellule Trigger ne sera jamais ecrite.")
    print("Aucun ordre reel ne peut etre envoye par ce script.")
    print_strategy_registry_diagnostics()

    feed = GrussFeed(DEFAULT_WORKBOOK_PATH)
    runner = GrussDryRunRunner(DATA_DIR)
    runner.processed_store = ProcessedRaceStore(DATA_DIR / "gruss_write_no_trigger_processed.csv")
    provider = GrussExcelOrderProvider(DATA_DIR, bridge=feed.bridge, write_no_trigger_guard=True)
    momentum_buffer = GrussMomentumBuffer()

    tick = 0
    while args.max_ticks is None or tick < args.max_ticks:
        tick += 1
        try:
            state = read_gruss_dryrun_state(feed)
        except Exception as exc:
            print(f"tick={tick} skip: lecture Gruss impossible: {exc}")
            _sleep(args, tick)
            continue

        print(f"tick={tick} {describe_state(state)}")
        if not state.validation_warnings:
            captured, anchor = momentum_buffer.add_snapshot_pair(state.win_snapshot, state.place_snapshot)
            if captured and anchor is not None:
                print(f"tick={tick} MOM45 anchor captured countdown={anchor.countdown_seconds}s")

        if state.skip_reason:
            print(f"tick={tick} skip: {state.skip_reason}")
            _sleep(args, tick)
            continue

        wait_reason = countdown_wait_reason(
            state.win_snapshot.metadata.countdown_seconds,
            state.place_snapshot.metadata.countdown_seconds,
        )
        if wait_reason:
            print(f"tick={tick} {wait_reason}")
            _sleep(args, tick)
            continue
        print(
            f"tick={tick} "
            f"{describe_current_strategy_milestone(state.win_snapshot.metadata.countdown_seconds, state.place_snapshot.metadata.countdown_seconds)}"
        )

        key = race_key(state.win_snapshot, state.place_snapshot)
        milestone = current_strategy_milestone(
            state.win_snapshot.metadata.countdown_seconds,
            state.place_snapshot.metadata.countdown_seconds,
        )
        processed_key = strategy_milestone_key(key, milestone)
        if runner.processed_store.has_seen(processed_key):
            print(f"tick={tick} skip: milestone deja traite ({processed_key})")
            _sleep(args, tick)
            continue

        trade_count_before = runner.trade_row_count()
        try:
            runner.evaluate(
                state.win_snapshot,
                state.place_snapshot,
                debug_strategies=args.debug_strategies,
                momentum_buffer=momentum_buffer,
            )
            trade_rows = runner.trade_rows_since(trade_count_before)
            intents = build_order_intents_from_trade_rows(
                trade_rows,
                state.win_snapshot,
                state.place_snapshot,
            )
            context = build_write_no_trigger_context(state, key)
            results, already_processed = process_write_no_trigger_batch(
                provider=provider,
                intents=intents,
                context=context,
                processed_store=runner.processed_store,
                key=processed_key,
                win_market_id=state.win_snapshot.metadata.market_id,
                place_market_id=state.place_snapshot.metadata.market_id,
            )
            if already_processed:
                print(f"tick={tick} skip: course deja traitee ({key})")
                _sleep(args, tick)
                continue
            print(
                f"tick={tick} course={context.course or key} "
                f"countdown={context.countdown_seconds}s "
                f"signaux_generes={len(trade_rows)} write_attempts={len(results)}"
            )
            if not results:
                print(f"tick={tick} no signal")
            for index, result in enumerate(results, start=1):
                cells = ",".join(result.excel_cells_written) or "-"
                print(
                    f"tick={tick} attempt={index} status={result.status} reason={result.reason} "
                    f"sheet={result.excel_sheet or '-'} row={result.excel_row or '-'} "
                    f"cells_written={cells} intended_trigger={result.intended_trigger or '-'} "
                    f"trigger_written={result.trigger_written}"
                )
        except Exception as exc:
            print(f"tick={tick} skip: evaluation/write-no-trigger impossible: {exc}")

        _sleep(args, tick)

    return 0


def process_write_no_trigger_batch(
    *,
    provider,
    intents,
    context: GrussRealOrderContext,
    processed_store,
    key: str,
    win_market_id: str | None,
    place_market_id: str | None,
):
    """Process every intent before marking the race as seen."""
    if processed_store.has_seen(key):
        return [], True

    results = []
    for intent in intents:
        write_intent = replace(
            intent,
            provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
            dry_run=False,
        )
        results.append(provider.place_order(write_intent, context))

    processed_store.mark_seen(key, win_market_id, place_market_id)
    return results, False


def validate_write_no_trigger_environment(env: Mapping[str, str] | None = None) -> None:
    values = env if env is not None else os.environ
    if not _is_true(values.get("DOGBOT_GRUSS_WRITE_NO_TRIGGER")):
        raise RuntimeError("DOGBOT_GRUSS_WRITE_NO_TRIGGER=true est obligatoire")
    if _is_true(values.get("DOGBOT_GRUSS_REAL_PREVIEW")):
        raise RuntimeError("DOGBOT_GRUSS_REAL_PREVIEW=false est obligatoire en write-no-trigger")

    providers = _provider_values(values)
    if env is None:
        config = load_provider_config()
        providers = (config.data_provider, config.order_provider)
    if providers[0] != DATA_PROVIDER_GRUSS_EXCEL:
        raise RuntimeError("DOGBOT_DATA_PROVIDER=gruss_excel est obligatoire")
    if providers[1] != ORDER_PROVIDER_GRUSS_EXCEL_REAL:
        raise RuntimeError("DOGBOT_ORDER_PROVIDER=gruss_excel_real est obligatoire")


def countdown_wait_reason(
    win_countdown_seconds: int | None,
    place_countdown_seconds: int | None,
) -> str | None:
    return active_countdown_wait_reason(win_countdown_seconds, place_countdown_seconds)


def build_write_no_trigger_context(state, key: str) -> GrussRealOrderContext:
    seconds = state.win_snapshot.metadata.countdown_seconds
    if seconds is None:
        seconds = state.place_snapshot.metadata.countdown_seconds
    return GrussRealOrderContext(
        validation_ok=not state.validation_warnings,
        tradable=state.tradable,
        region=gruss_region_for_snapshots(state.win_snapshot, state.place_snapshot),
        countdown_seconds=seconds,
        course=str(state.win_snapshot.metadata.event_path or state.place_snapshot.metadata.event_path or key),
        market_already_processed=False,
        win_market_id=state.win_snapshot.metadata.market_id,
        place_market_id=state.place_snapshot.metadata.market_id,
    )


def _provider_values(env: Mapping[str, str]) -> tuple[str, str]:
    return (
        str(env.get("DOGBOT_DATA_PROVIDER", "")).strip().lower(),
        str(env.get("DOGBOT_ORDER_PROVIDER", "")).strip().lower(),
    )


def _is_true(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on", "y"}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Gruss preparation cells without ever writing Trigger.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between Excel reads.")
    parser.add_argument("--max-ticks", type=int, default=None, help="Stop after N reads.")
    parser.add_argument("--debug-strategies", action="store_true", help="Print detailed PLACE strategy evaluations.")
    return parser.parse_args(argv)


def _sleep(args: argparse.Namespace, tick: int) -> None:
    if args.max_ticks is not None and tick >= args.max_ticks:
        return
    time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
