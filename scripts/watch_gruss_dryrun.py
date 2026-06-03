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
    create_configured_gruss_feed,
    describe_state,
    race_key,
    read_gruss_dryrun_state,
    validate_gruss_dryrun_provider_config,
)
from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH


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

    try:
        config = validate_gruss_dryrun_provider_config()
    except Exception as exc:
        print(f"SKIP: configuration provider invalide: {exc}")
        return 1
    print(f"providers: data={config.data_provider} order={config.order_provider}")

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
        if state.skip_reason:
            print(f"tick={tick} skip: {state.skip_reason}")
            _sleep(args, tick)
            continue

        seconds = state.win_snapshot.metadata.countdown_seconds
        if seconds is None:
            seconds = state.place_snapshot.metadata.countdown_seconds
        if seconds is None:
            print(f"tick={tick} skip: countdown_seconds_unavailable")
            _sleep(args, tick)
            continue
        if seconds > args.trigger_seconds:
            print(f"tick={tick} wait: countdown_seconds={seconds} > trigger={args.trigger_seconds}")
            _sleep(args, tick)
            continue

        key = race_key(state.win_snapshot, state.place_snapshot)
        if runner.processed_store.has_seen(key):
            print(f"tick={tick} skip: marche deja traite ({key})")
            _sleep(args, tick)
            continue

        trade_count_before = runner.trade_row_count()
        try:
            runner.evaluate(state.win_snapshot, state.place_snapshot)
            runner.processed_store.mark_seen(
                key,
                state.win_snapshot.metadata.market_id,
                state.place_snapshot.metadata.market_id,
            )
            generated = max(0, runner.trade_row_count() - trade_count_before)
            if generated:
                print(f"tick={tick} signal: {generated} lignes DRYRUN ecrites dans {runner.trade_path()}")
            else:
                print(f"tick={tick} no signal")
            print(f"tick={tick} evaluation dry-run terminee: {key}")
        except Exception as exc:
            print(f"tick={tick} skip: evaluation dry-run impossible: {exc}")

        _sleep(args, tick)

    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gruss Excel through the bot engine in dry-run mode.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between Excel reads.")
    parser.add_argument("--max-ticks", type=int, default=None, help="Stop after N reads.")
    parser.add_argument("--trigger-seconds", type=int, default=2, help="Evaluate once when countdown <= this value.")
    return parser.parse_args(argv)


def _sleep(args: argparse.Namespace, tick: int) -> None:
    if args.max_ticks is not None and tick >= args.max_ticks:
        return
    time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
