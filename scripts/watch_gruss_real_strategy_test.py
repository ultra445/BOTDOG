from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogbot.config import DATA_PROVIDER_GRUSS_EXCEL, ORDER_PROVIDER_GRUSS_EXCEL_REAL, load_provider_config
from dogbot.gruss.gruss_dryrun_engine import (
    GrussDryRunRunner,
    ProcessedRaceStore,
    build_order_intents_from_trade_rows,
    describe_current_strategy_milestone,
    describe_state,
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
from watch_gruss_real_test import countdown_wait_reason, ensure_open_visible_workbook, gruss_region_for_snapshots


DATA_DIR = ROOT / "data"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.interval <= 0:
        print("ERREUR: --interval doit etre superieur a 0")
        return 1
    if args.max_ticks is not None and args.max_ticks <= 0:
        print("ERREUR: --max-ticks doit etre superieur a 0")
        return 1

    try:
        max_orders, max_stake, force_stake = validate_real_strategy_test_environment()
    except RuntimeError as exc:
        print(f"ERREUR SECURITE: {exc}")
        return 1

    print("REAL STRATEGY TEST MODE")
    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")
    print(f"max_orders={max_orders}")
    print(f"max_stake={max_stake:g}")
    print(f"force_stake={force_stake:g}")
    print("Les vraies strategies du registry seront evaluees.")
    print("Un trigger Excel reel peut etre ecrit. Maximum strict: 1 ordre par course.")
    print_strategy_registry_diagnostics()

    feed = GrussFeed(DEFAULT_WORKBOOK_PATH)
    try:
        ensure_open_visible_workbook(feed.bridge)
    except RuntimeError as exc:
        print(f"ERREUR SECURITE: {exc}")
        return 1

    runner = GrussDryRunRunner(DATA_DIR)
    runner.processed_store = ProcessedRaceStore(DATA_DIR / "gruss_real_strategy_test_processed.csv")
    provider = GrussExcelOrderProvider(DATA_DIR, bridge=feed.bridge)
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

        try:
            trade_count_before = runner.trade_row_count()
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
            context = build_real_strategy_test_context(state, key)
            results, already_processed = process_real_strategy_test_batch(
                provider=provider,
                intents=intents,
                context=context,
                processed_store=runner.processed_store,
                key=processed_key,
                win_market_id=state.win_snapshot.metadata.market_id,
                place_market_id=state.place_snapshot.metadata.market_id,
                force_stake=force_stake,
            )
            if already_processed:
                print(f"tick={tick} skip: course deja traitee ({key})")
                _sleep(args, tick)
                continue

            print(
                f"tick={tick} course={context.course or key} "
                f"countdown={context.countdown_seconds}s "
                f"signaux_generes={len(trade_rows)} real_attempts={len(results)}"
            )
            if not results:
                print(f"tick={tick} no signal")
            for index, result in enumerate(results, start=1):
                print_real_attempt(tick, index, result)
        except Exception as exc:
            print(f"tick={tick} skip: evaluation/real-strategy-test impossible: {exc}")

        _sleep(args, tick)

    return 0


def process_real_strategy_test_batch(
    *,
    provider,
    intents,
    context: GrussRealOrderContext,
    processed_store,
    key: str,
    win_market_id: str | None,
    place_market_id: str | None,
    force_stake: float,
):
    """Process every real strategy signal so rejected max-order attempts are logged."""
    phase_keys = [_processed_phase_key(key, _execution_phase_for_intent(intent)) for intent in intents]
    if phase_keys and all(processed_store.has_seen(phase_key) for phase_key in phase_keys):
        return [], True

    results = []
    for intent in intents:
        phase_key = _processed_phase_key(key, _execution_phase_for_intent(intent))
        if processed_store.has_seen(phase_key):
            continue
        stake_original = intent.stake
        real_intent = replace(
            intent,
            provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
            dry_run=False,
            stake=force_stake,
            stake_original=stake_original,
            stake_forced=True,
        )
        results.append(provider.place_order(real_intent, context))

    for phase_key in sorted(set(phase_keys)):
        processed_store.mark_seen(phase_key, win_market_id, place_market_id)
    return results, False


def _execution_phase_for_intent(intent) -> str:
    phase = str(getattr(intent, "execution_phase", "") or "POST").strip().upper()
    return phase if phase in {"PRE", "POST"} else "POST"


def _processed_phase_key(key: str, execution_phase: str) -> str:
    return f"{key}|{execution_phase}"


def validate_real_strategy_test_environment(
    env: Mapping[str, str] | None = None,
) -> tuple[int, float, float]:
    values = env if env is not None else os.environ
    data_provider = str(values.get("DOGBOT_DATA_PROVIDER", "")).strip().lower()
    order_provider = str(values.get("DOGBOT_ORDER_PROVIDER", "")).strip().lower()
    if env is None:
        config = load_provider_config()
        data_provider = config.data_provider
        order_provider = config.order_provider

    if data_provider != DATA_PROVIDER_GRUSS_EXCEL:
        raise RuntimeError("DOGBOT_DATA_PROVIDER=gruss_excel est obligatoire")
    if order_provider != ORDER_PROVIDER_GRUSS_EXCEL_REAL:
        raise RuntimeError("DOGBOT_ORDER_PROVIDER=gruss_excel_real est obligatoire")
    if not _is_explicit_true(values.get("DOGBOT_GRUSS_ENABLE_REAL_ORDERS")):
        raise RuntimeError("DOGBOT_GRUSS_ENABLE_REAL_ORDERS=true est obligatoire")
    if not _is_explicit_true(values.get("DOGBOT_GRUSS_REAL_TEST_MODE")):
        raise RuntimeError("DOGBOT_GRUSS_REAL_TEST_MODE=true est obligatoire")
    if not _is_explicit_true(values.get("DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED")):
        raise RuntimeError("DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED=true est obligatoire")

    forbidden = {
        "DOGBOT_GRUSS_FORCE_TEST_BACK_PLACE_LIMIT": values.get("DOGBOT_GRUSS_FORCE_TEST_BACK_PLACE_LIMIT"),
        "DOGBOT_GRUSS_FORCE_TEST_BSP_PLACE": values.get("DOGBOT_GRUSS_FORCE_TEST_BSP_PLACE"),
        "DOGBOT_GRUSS_REAL_PREVIEW": values.get("DOGBOT_GRUSS_REAL_PREVIEW"),
        "DOGBOT_GRUSS_WRITE_NO_TRIGGER": values.get("DOGBOT_GRUSS_WRITE_NO_TRIGGER"),
    }
    for name, value in forbidden.items():
        if _is_true(value):
            raise RuntimeError(f"{name}=true est interdit avec ce script")

    max_orders = _required_int(values, "DOGBOT_GRUSS_REAL_MAX_ORDERS")
    if max_orders != 1:
        raise RuntimeError("DOGBOT_GRUSS_REAL_MAX_ORDERS doit etre exactement 1")
    max_stake = _required_float(values, "DOGBOT_GRUSS_REAL_MAX_STAKE")
    if not math.isfinite(max_stake) or max_stake != 2:
        raise RuntimeError("DOGBOT_GRUSS_REAL_MAX_STAKE doit etre exactement 2")
    force_stake = _required_float(values, "DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE")
    if not math.isfinite(force_stake) or force_stake != 2:
        raise RuntimeError("DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE doit etre exactement 2")
    if force_stake > max_stake:
        raise RuntimeError(
            "DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE doit etre <= DOGBOT_GRUSS_REAL_MAX_STAKE"
        )
    return max_orders, max_stake, force_stake


def build_real_strategy_test_context(state, key: str) -> GrussRealOrderContext:
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


def print_real_attempt(tick: int, index: int, result) -> None:
    cells = ",".join(result.excel_cells_written) or "-"
    outcome = (
        "ordre_envoye"
        if result.status == "GRUSS_REAL_WRITTEN" and result.post_write_verified is True
        else "ordre_refuse"
    )
    wrote = {str(address).upper(): value for address, value in result.write_plan}
    verification = (
        "OK"
        if result.post_write_verified is True
        else "KO" if result.post_write_verified is False else "N/A"
    )
    print(
        f"tick={tick} attempt={index} {outcome} "
        f"status={result.status} reason={result.reason} "
        f"stake_original={result.stake_original} stake_used={result.stake_used} "
        f"stake_forced={result.stake_forced} "
        f"sheet={result.excel_sheet or '-'} row={result.excel_row or '-'} "
        f"cells={cells} trigger={result.intended_trigger or '-'} "
        f"trigger_cell_address={result.trigger_cell_address or '-'} "
        f"trigger_written={result.trigger_written} "
        f"trigger_value_written={result.trigger_value_written or '-'} "
        f"trigger_clear_delay_ms={result.trigger_clear_delay_ms} "
        f"trigger_cleared={result.trigger_cleared}"
    )
    print(
        f"tick={tick} attempt={index} "
        f"wrote R/S/Q: "
        f"{result.post_write_odds_cell_address or '-'}="
        f"{wrote.get(result.post_write_odds_cell_address, '-')!r} | "
        f"{result.post_write_stake_cell_address or '-'}="
        f"{wrote.get(result.post_write_stake_cell_address, '-')!r} | "
        f"{result.post_write_trigger_cell_address or '-'}="
        f"{wrote.get(result.post_write_trigger_cell_address, '-')!r}"
    )
    print(
        f"tick={tick} attempt={index} "
        f"readback R/S/Q: "
        f"{result.post_write_odds_cell_address or '-'}={result.post_write_odds_value!r} | "
        f"{result.post_write_stake_cell_address or '-'}={result.post_write_stake_value!r} | "
        f"{result.post_write_trigger_cell_address or '-'}={result.post_write_trigger_value!r} | "
        f"verification {verification}"
    )


def _required_int(env: Mapping[str, str], name: str) -> int:
    raw = str(env.get(name, "")).strip()
    if not raw:
        raise RuntimeError(f"{name} est obligatoire")
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} doit etre un entier") from exc


def _required_float(env: Mapping[str, str], name: str) -> float:
    raw = str(env.get(name, "")).strip()
    if not raw:
        raise RuntimeError(f"{name} est obligatoire")
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} doit etre un nombre") from exc


def _is_true(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on", "y"}


def _is_explicit_true(value: str | None) -> bool:
    return str(value or "").strip().casefold() == "true"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the first ultra-limited real Gruss strategy test."
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between Excel reads.")
    parser.add_argument("--max-ticks", type=int, default=None, help="Stop after N reads.")
    parser.add_argument("--debug-strategies", action="store_true", help="Print detailed strategy evaluations.")
    return parser.parse_args(argv)


def _sleep(args: argparse.Namespace, tick: int) -> None:
    if args.max_ticks is not None and tick >= args.max_ticks:
        return
    time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
