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
from dogbot.gruss.gruss_orders import make_order_intent
from dogbot.gruss.gruss_real_orders import (
    GrussExcelOrderProvider,
    GrussRealOrderContext,
    GrussTriggerLayout,
)


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
        max_orders, max_stake, force_stake, force_test_bsp_place = validate_real_test_environment()
    except RuntimeError as exc:
        print(f"ERREUR SECURITE: {exc}")
        return 1
    force_test_back_place_limit = _is_explicit_true(
        os.getenv("DOGBOT_GRUSS_FORCE_TEST_BACK_PLACE_LIMIT")
    )

    print("REAL TEST MODE")
    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")
    print(f"max_orders={max_orders}")
    print(f"max_stake={max_stake:g}")
    print(f"force_stake={force_stake:g}" if force_stake is not None else "force_stake=disabled")
    print(f"force_test_bsp_place={force_test_bsp_place}")
    print(f"force_test_back_place_limit={force_test_back_place_limit}")
    print("Un trigger Excel reel peut etre ecrit. Maximum strict: 1 ordre par course.")
    if force_test_bsp_place:
        print("FORCED BSP PLACE TEST: les strategies normales ne seront pas evaluees.")
    elif force_test_back_place_limit:
        print("FORCED BACK PLACE LIMIT TEST: les strategies normales ne seront pas evaluees.")
    else:
        print_strategy_registry_diagnostics()

    feed = GrussFeed(DEFAULT_WORKBOOK_PATH)
    try:
        ensure_open_visible_workbook(feed.bridge)
    except RuntimeError as exc:
        print(f"ERREUR SECURITE: {exc}")
        return 1

    runner = GrussDryRunRunner(DATA_DIR)
    runner.processed_store = ProcessedRaceStore(DATA_DIR / "gruss_real_test_processed.csv")
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
            trade_rows = []

            def normal_intents_factory():
                trade_count_before = runner.trade_row_count()
                runner.evaluate(
                    state.win_snapshot,
                    state.place_snapshot,
                    debug_strategies=args.debug_strategies,
                    momentum_buffer=momentum_buffer,
                )
                trade_rows.extend(runner.trade_rows_since(trade_count_before))
                return build_order_intents_from_trade_rows(
                    trade_rows,
                    state.win_snapshot,
                    state.place_snapshot,
                )

            intents = build_real_test_intents(
                force_test_bsp_place=force_test_bsp_place,
                force_test_back_place_limit=force_test_back_place_limit,
                place_snapshot=state.place_snapshot,
                normal_intents_factory=normal_intents_factory,
                layout=provider.layout,
            )
            if force_test_bsp_place and intents:
                forced = intents[0]
                print(
                    f"tick={tick} strategy_id={forced.strategy_id} "
                    f"selected_reason={forced.selected_reason} "
                    f"selected_runner={forced.selected_runner} "
                    f"selected_trap={forced.selected_trap} "
                    f"selected_place_odds={forced.selected_place_odds}"
                )
            if force_test_back_place_limit and intents:
                forced = intents[0]
                print(
                    f"tick={tick} strategy_id={forced.strategy_id} "
                    f"selected_reason={forced.selected_reason} "
                    f"selected_runner={forced.selected_runner} "
                    f"selected_trap={forced.selected_trap} "
                    f"selected_place_back_odds={forced.selected_place_back_odds} "
                    f"selected_place_lay_odds={forced.selected_place_lay_odds} "
                    f"price_used={forced.price_used}"
                )
            context = build_real_test_context(state, key)
            results, already_processed = process_real_test_batch(
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
                    f"trigger_cell_current_value={result.trigger_cell_current_value!r} "
                    f"trigger_cell_expected_empty={result.trigger_cell_expected_empty} "
                    f"trigger_mapping_name={result.trigger_mapping_name or '-'} "
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
                if result.hold_trigger_for_visual_test and result.trigger_written:
                    print(
                        f"tick={tick} attempt={index} holding trigger for visual test "
                        f"delay_ms={result.trigger_clear_delay_ms}"
                    )
        except Exception as exc:
            print(f"tick={tick} skip: evaluation/real-test impossible: {exc}")

        _sleep(args, tick)

    return 0


def process_real_test_batch(
    *,
    provider,
    intents,
    context: GrussRealOrderContext,
    processed_store,
    key: str,
    win_market_id: str | None,
    place_market_id: str | None,
    force_stake: float | None = None,
):
    """Process and log the complete batch before marking the race as seen."""
    if processed_store.has_seen(key):
        return [], True

    results = []
    for intent in intents:
        stake_original = intent.stake
        stake_used = force_stake if force_stake is not None else stake_original
        real_intent = replace(
            intent,
            provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
            dry_run=False,
            stake=stake_used,
            stake_original=stake_original,
            stake_forced=force_stake is not None,
        )
        results.append(provider.place_order(real_intent, context))

    processed_store.mark_seen(key, win_market_id, place_market_id)
    return results, False


def build_real_test_intents(
    *,
    force_test_bsp_place: bool,
    place_snapshot,
    normal_intents_factory,
    layout: GrussTriggerLayout,
    force_test_back_place_limit: bool = False,
):
    if force_test_bsp_place and force_test_back_place_limit:
        raise RuntimeError("forced_test_modes_are_mutually_exclusive")
    if force_test_bsp_place:
        return [build_force_test_bsp_place_intent(place_snapshot, layout)]
    if force_test_back_place_limit:
        return [build_force_test_back_place_limit_intent(place_snapshot)]
    if not force_test_bsp_place:
        return normal_intents_factory()


def build_force_test_bsp_place_intent(place_snapshot, layout: GrussTriggerLayout):
    if place_snapshot is None or str(getattr(place_snapshot, "sheet_name", "")).upper() != "PLACE":
        raise RuntimeError("forced_bsp_place_market_absent")
    if not str(layout.back_sp_moc_trigger or "").strip():
        raise RuntimeError("back_sp_mapping_unavailable")

    candidates = [
        runner
        for runner in getattr(place_snapshot, "runners", [])
        if _positive_finite_odds(getattr(runner, "best_back", None))
    ]
    if not candidates:
        raise RuntimeError("forced_bsp_place_no_available_runner_odds")
    selected = min(candidates, key=lambda runner: float(runner.best_back))
    metadata = place_snapshot.metadata
    selected_odds = float(selected.best_back)
    return make_order_intent(
        provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
        market_type="PLACE",
        market_id=metadata.market_id or "",
        parent_id=metadata.parent_id,
        runner_name=selected.runner_name,
        trap=selected.trap,
        side="BACK",
        order_type="SP_MOC",
        price=selected_odds,
        stake=1.0,
        strategy_id="GRUSS_FORCE_TEST_BSP_PLACE",
        course_id=str(metadata.parent_id or metadata.event_path or metadata.market_id or ""),
        dry_run=False,
        stake_original=1.0,
        stake_forced=True,
        force_test_bsp_place=True,
        selected_reason="lowest_place_odds",
        selected_runner=selected.runner_name,
        selected_trap=selected.trap,
        selected_place_odds=selected_odds,
    )


def build_force_test_back_place_limit_intent(place_snapshot):
    if place_snapshot is None or str(getattr(place_snapshot, "sheet_name", "")).upper() != "PLACE":
        raise RuntimeError("forced_back_place_limit_market_absent")

    candidates = [
        runner
        for runner in getattr(place_snapshot, "runners", [])
        if _positive_finite_odds(getattr(runner, "best_back", None))
    ]
    if not candidates:
        raise RuntimeError("forced_back_place_limit_no_available_runner_odds")
    selected = min(candidates, key=lambda runner: float(runner.best_back))
    selected_lay = getattr(selected, "best_lay", None)
    if not _positive_finite_odds(selected_lay):
        raise RuntimeError("missing_place_best_lay")

    metadata = place_snapshot.metadata
    selected_back = float(selected.best_back)
    price_used = float(selected_lay)
    return make_order_intent(
        provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
        market_type="PLACE",
        market_id=metadata.market_id or "",
        parent_id=metadata.parent_id,
        runner_name=selected.runner_name,
        trap=selected.trap,
        side="BACK",
        order_type="LIMIT",
        price=price_used,
        stake=1.0,
        strategy_id="GRUSS_FORCE_TEST_BACK_PLACE_LIMIT",
        course_id=str(metadata.parent_id or metadata.event_path or metadata.market_id or ""),
        dry_run=False,
        stake_original=1.0,
        stake_forced=True,
        force_test_back_place_limit=True,
        selected_reason="lowest_place_odds_best_lay_price",
        selected_runner=selected.runner_name,
        selected_trap=selected.trap,
        selected_place_back_odds=selected_back,
        selected_place_lay_odds=price_used,
        price_used=price_used,
    )


def validate_real_test_environment(
    env: Mapping[str, str] | None = None,
) -> tuple[int, float, float | None, bool]:
    values = env if env is not None else os.environ
    providers = _provider_values(values)
    if env is None:
        config = load_provider_config()
        providers = (config.data_provider, config.order_provider)

    if providers[0] != DATA_PROVIDER_GRUSS_EXCEL:
        raise RuntimeError("DOGBOT_DATA_PROVIDER=gruss_excel est obligatoire")
    if providers[1] != ORDER_PROVIDER_GRUSS_EXCEL_REAL:
        raise RuntimeError("DOGBOT_ORDER_PROVIDER=gruss_excel_real est obligatoire")
    if not _is_explicit_true(values.get("DOGBOT_GRUSS_ENABLE_REAL_ORDERS")):
        raise RuntimeError("DOGBOT_GRUSS_ENABLE_REAL_ORDERS=true est obligatoire")
    real_test_mode = _is_explicit_true(values.get("DOGBOT_GRUSS_REAL_TEST_MODE"))
    if _has_value(values.get("DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE")) and not real_test_mode:
        raise RuntimeError("DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE exige DOGBOT_GRUSS_REAL_TEST_MODE=true")
    if not real_test_mode:
        raise RuntimeError("DOGBOT_GRUSS_REAL_TEST_MODE=true est obligatoire")
    if _is_true(values.get("DOGBOT_GRUSS_REAL_PREVIEW")):
        raise RuntimeError("DOGBOT_GRUSS_REAL_PREVIEW=true est incompatible avec ce script")
    if _is_true(values.get("DOGBOT_GRUSS_WRITE_NO_TRIGGER")):
        raise RuntimeError("DOGBOT_GRUSS_WRITE_NO_TRIGGER=true est incompatible avec ce script")
    if not _is_explicit_true(values.get("DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED")):
        raise RuntimeError("DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED=true est obligatoire")

    max_orders = _required_int(values, "DOGBOT_GRUSS_REAL_MAX_ORDERS")
    if max_orders != 1:
        raise RuntimeError("DOGBOT_GRUSS_REAL_MAX_ORDERS doit etre exactement 1")
    max_stake = _required_float(values, "DOGBOT_GRUSS_REAL_MAX_STAKE")
    if not math.isfinite(max_stake) or max_stake <= 0 or max_stake > 2:
        raise RuntimeError("DOGBOT_GRUSS_REAL_MAX_STAKE doit etre > 0 et <= 2")
    force_stake = _optional_positive_float(values, "DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE")
    if force_stake is not None and force_stake > max_stake:
        raise RuntimeError(
            "DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE doit etre <= DOGBOT_GRUSS_REAL_MAX_STAKE"
        )
    force_test_bsp_place = _is_explicit_true(values.get("DOGBOT_GRUSS_FORCE_TEST_BSP_PLACE"))
    force_test_back_place_limit = _is_explicit_true(
        values.get("DOGBOT_GRUSS_FORCE_TEST_BACK_PLACE_LIMIT")
    )
    if force_test_bsp_place and force_test_back_place_limit:
        raise RuntimeError(
            "DOGBOT_GRUSS_FORCE_TEST_BSP_PLACE et "
            "DOGBOT_GRUSS_FORCE_TEST_BACK_PLACE_LIMIT sont incompatibles"
        )
    if force_test_bsp_place and force_stake is None:
        raise RuntimeError("DOGBOT_GRUSS_FORCE_TEST_BSP_PLACE exige DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE")
    if force_test_back_place_limit and force_stake is None:
        raise RuntimeError(
            "DOGBOT_GRUSS_FORCE_TEST_BACK_PLACE_LIMIT exige DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE"
        )
    return max_orders, max_stake, force_stake, force_test_bsp_place


def ensure_open_visible_workbook(bridge) -> None:
    try:
        bridge.connect_open_workbook()
        visible = bridge.is_workbook_visible()
    except Exception as exc:
        raise RuntimeError(f"workbook Excel Gruss ouvert et visible requis: {exc}") from exc
    if not visible:
        raise RuntimeError("workbook Excel Gruss ouvert mais non visible")


def countdown_wait_reason(
    win_countdown_seconds: int | None,
    place_countdown_seconds: int | None,
) -> str | None:
    return active_countdown_wait_reason(win_countdown_seconds, place_countdown_seconds)


def build_real_test_context(state, key: str) -> GrussRealOrderContext:
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


def _optional_positive_float(env: Mapping[str, str], name: str) -> float | None:
    raw = str(env.get(name, "")).strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} doit etre un nombre positif") from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{name} doit etre un nombre positif")
    return value


def _has_value(value: str | None) -> bool:
    return bool(str(value or "").strip())


def _positive_finite_odds(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 1.01


def _is_true(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on", "y"}


def _is_explicit_true(value: str | None) -> bool:
    return str(value or "").strip().casefold() == "true"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first ultra-limited real Gruss order test.")
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
