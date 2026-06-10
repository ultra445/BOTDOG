from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogbot.gruss.gruss_dryrun_engine import (
    GrussDryRunRunner,
    active_strategy_milestones,
    create_configured_gruss_feed,
    countdown_wait_reason,
    current_strategy_milestone,
    describe_current_strategy_milestone,
    describe_state,
    print_strategy_registry_diagnostics,
    race_key,
    read_gruss_dryrun_state,
    strategy_milestone_key,
    validate_gruss_dryrun_provider_config,
)
from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH
from dogbot.gruss.gruss_momentum import GrussMomentumBuffer


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.interval <= 0:
        print("ERREUR: --interval doit etre superieur a 0")
        return 1
    if args.max_ticks is not None and args.max_ticks <= 0:
        print("ERREUR: --max-ticks doit etre superieur a 0")
        return 1

    print("Gruss engine dry-run watcher")
    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")
    print("Aucun ordre ne sera envoye.")
    print(f"active_milestones={list(active_strategy_milestones())}")

    try:
        config = validate_gruss_dryrun_provider_config()
    except Exception as exc:
        print(f"SKIP: configuration provider invalide: {exc}")
        return 1
    print(f"providers: data={config.data_provider} order={config.order_provider}")
    print_strategy_registry_diagnostics()
    momentum_buffer = GrussMomentumBuffer()
    print(
        "MOM45 buffer actif: retention="
        f"{momentum_buffer.retention_seconds}s "
        f"anchor=T-{momentum_buffer.anchor_max_seconds}..T-{momentum_buffer.anchor_min_seconds}s"
    )

    feed = create_configured_gruss_feed(config)
    runner = GrussDryRunRunner(ROOT / "data")
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
                print(
                    "tick="
                    f"{tick} MOM45 anchor captured "
                    f"countdown={anchor.countdown_seconds}s "
                    f"traps={len(anchor.win_base_by_trap)}"
                )
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

        _print_mom45_status(tick, momentum_buffer, state.win_snapshot, state.place_snapshot)
        trade_count_before = runner.trade_row_count()
        try:
            runner.evaluate(
                state.win_snapshot,
                state.place_snapshot,
                debug_strategies=args.debug_strategies,
                momentum_buffer=momentum_buffer,
            )
            runner.processed_store.mark_seen(
                processed_key,
                state.win_snapshot.metadata.market_id,
                state.place_snapshot.metadata.market_id,
            )
            generated = max(0, runner.trade_row_count() - trade_count_before)
            if generated:
                print(f"tick={tick} signal: {generated} lignes DRYRUN ecrites dans {runner.trade_path()}")
                order_results = runner.log_gruss_order_intents(
                    runner.trade_rows_since(trade_count_before),
                    state.win_snapshot,
                    state.place_snapshot,
                )
                if order_results:
                    print(f"tick={tick} ordres Gruss dry-run journalises: {len(order_results)}")
            else:
                print(f"tick={tick} no signal")
            print(f"tick={tick} evaluation dry-run terminee: {processed_key}")
        except Exception as exc:
            print(f"tick={tick} skip: evaluation dry-run impossible: {exc}")

        _sleep(args, tick)

    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gruss Excel through the bot engine in dry-run mode.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between Excel reads.")
    parser.add_argument("--max-ticks", type=int, default=None, help="Stop after N reads.")
    parser.add_argument(
        "--trigger-seconds",
        type=int,
        default=2,
        help="Deprecated compatibility option; PRE/POST active milestones drive evaluation.",
    )
    parser.add_argument("--debug-strategies", action="store_true", help="Print detailed PLACE strategy evaluations.")
    return parser.parse_args(argv)


def _sleep(args: argparse.Namespace, tick: int) -> None:
    if args.max_ticks is not None and tick >= args.max_ticks:
        return
    time.sleep(args.interval)


def _print_mom45_status(tick: int, momentum_buffer: GrussMomentumBuffer, win_snapshot, place_snapshot) -> None:
    values = momentum_buffer.momentum_by_trap(win_snapshot, place_snapshot)
    status = momentum_buffer.course_status(win_snapshot, place_snapshot)
    available = [value for value in values.values() if value.has_mom45]
    if available:
        sample = available[0]
        print(
            "tick="
            f"{tick} MOM45 available "
            f"traps={len(available)}/{len(values)} "
            f"source_countdown={sample.source_countdown_seconds}s "
            f"first_seen_countdown={status.first_seen_countdown} "
            f"t45_anchor_found={status.t45_anchor_found} "
            f"mom45_reason={status.mom45_reason or ''}"
        )
        return
    print(
        f"tick={tick} missing_mom45 "
        f"mom45_reason={status.mom45_reason or 'no_t45_anchor'} "
        f"first_seen_countdown={status.first_seen_countdown} "
        f"t45_anchor_found={status.t45_anchor_found}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
