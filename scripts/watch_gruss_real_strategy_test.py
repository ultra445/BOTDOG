from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
    active_strategy_milestones,
    build_order_intents_from_trade_rows,
    describe_state,
    current_strategy_milestone,
    print_strategy_registry_diagnostics,
    race_key,
    read_gruss_dryrun_state,
    strategy_milestone_key,
)
from dogbot.executor import _execution_phase_for_milestone
from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH
from dogbot.gruss.gruss_feed import GrussFeed
from dogbot.gruss.gruss_momentum import GrussMomentumBuffer
from dogbot.gruss.gruss_real_orders import GrussExcelOrderProvider, GrussRealOrderContext
from dogbot.pre_ladder import BETFAIR_PRICE_BANDS, round_to_betfair_tick
from watch_gruss_real_test import countdown_wait_reason, ensure_open_visible_workbook, gruss_region_for_snapshots


DATA_DIR = ROOT / "data"
REAL_PRE_LADDER_ENV_DEFAULTS = {
    "DOGBOT_PRE_LADDER_ENABLED": "true",
    "DOGBOT_PRE_LADDER_PREVIEW": "false",
    "DOGBOT_PRE_LADDER_STEPS": "45,32,20,14",
    "DOGBOT_PRE_LADDER_REAL_REQUIRE_BET_REF_FOR_REPLACE": "true",
    "DOGBOT_PRE_LADDER_REAL_STOP_IF_NO_BET_REF": "true",
    "DOGBOT_PRE_LADDER_REAL_NO_STACKING": "true",
    "DOGBOT_PRE_POST_INDEPENDENT": "true",
    "DOGBOT_PRE_CANCEL_BEFORE_POST": "false",
    "DOGBOT_PRE_CANCEL_ONLY_IF_POST_PENDING": "false",
    "DOGBOT_POST_SKIP_IF_PRE_MATCHED": "false",
    "DOGBOT_PRE_CANCEL_SECONDS_BEFORE_OFF": "1",
    "DOGBOT_POST_ALLOW_AFTER_SCHEDULED_OFF_SECONDS": "5",
    "DOGBOT_GRUSS_REPLACE_MIN_COUNTDOWN_SECONDS": "10",
}
REAL_PRE_LADDER_CONFIGURABLE_DEFAULTS = {
    "DOGBOT_PRE_LADDER_REAL_MAX_LADDERS": "50",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.interval <= 0:
        print("ERREUR: --interval doit etre superieur a 0")
        return 1
    if args.max_ticks is not None and args.max_ticks <= 0:
        print("ERREUR: --max-ticks doit etre superieur a 0")
        return 1

    load_strategy_test_env_file()
    configured_pre_ladder_env = configure_real_pre_ladder_for_strategy_test()
    try:
        max_orders, max_stake, force_stake = validate_real_strategy_test_environment()
    except RuntimeError as exc:
        print(f"ERREUR SECURITE: {exc}")
        return 1

    print("REAL STRATEGY TEST MODE")
    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")
    print(f"max_orders={max_orders}")
    print(f"max_stake={max_stake:g}")
    print(f"force_stake={force_stake:g}" if force_stake is not None else "force_stake=variable")
    print(
        "real_env: "
        f"dry_run={os.getenv('DRY_RUN', '') or '<unset>'} "
        f"order_provider={os.getenv('DOGBOT_ORDER_PROVIDER', '') or '<unset>'} "
        f"data_provider={os.getenv('DOGBOT_DATA_PROVIDER', '') or '<unset>'} "
        f"gruss_real_orders_enabled={os.getenv('DOGBOT_GRUSS_ENABLE_REAL_ORDERS', '') or '<unset>'} "
        f"real_variable_stakes={os.getenv('DOGBOT_GRUSS_REAL_VARIABLE_STAKES', '') or '<unset>'} "
        f"real_max_orders={os.getenv('DOGBOT_GRUSS_REAL_MAX_ORDERS', '') or '<unset>'} "
        f"real_max_stake={os.getenv('DOGBOT_GRUSS_REAL_MAX_STAKE', '') or '<unset>'} "
        f"pre_ladder_real_max_ladders={os.getenv('DOGBOT_PRE_LADDER_REAL_MAX_LADDERS', '') or '<unset>'}"
    )
    print("Les vraies strategies du registry seront evaluees.")
    print(f"Des triggers Excel reels peuvent etre ecrits. Maximum session: {max_orders} ecritures.")
    print(
        "PRE ladder reel arme: "
        + ", ".join(f"{name}={value}" for name, value in configured_pre_ladder_env.items())
    )
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
    if hasattr(provider, "cleanup_stale_command_cells"):
        cleanup = provider.cleanup_stale_command_cells(reason="startup")
        print(
            "startup_command_cells_cleanup "
            f"attempted={cleanup.get('attempted')} "
            f"done={cleanup.get('done')} "
            f"addresses={cleanup.get('addresses') or '-'} "
            f"reason={cleanup.get('reason')}"
        )
        if cleanup.get("failed"):
            unsafe_reason = cleanup.get("unsafe_stop_reason") or "unsafe_stale_gruss_triggers_cleanup_failed"
            print(
                f"ERREUR SECURITE: {unsafe_reason} "
                f"reason={cleanup.get('reason')}"
            )
            return 1
    print(
        "gruss_clear_command_cells_delay_ms_effective="
        f"{provider.command_cells_clear_delay_ms}"
    )
    momentum_buffer = GrussMomentumBuffer()
    milestone_tracker = _MilestoneTracker()
    stale_cleanup_keys: set[str] = set()

    tick = 0
    try:
        while args.max_ticks is None or tick < args.max_ticks:
            tick += 1
            if hasattr(provider, "drain_due_command_cell_clears"):
                provider.drain_due_command_cell_clears()
            if hasattr(provider, "cleanup_stale_command_cells"):
                cleanup = provider.cleanup_stale_command_cells(reason=f"periodic:tick={tick}")
                if cleanup.get("failed"):
                    unsafe_reason = cleanup.get("unsafe_stop_reason") or "unsafe_stale_gruss_triggers_cleanup_failed"
                    print(
                        f"ERREUR SECURITE: {unsafe_reason} "
                        f"tick={tick} reason={cleanup.get('reason')}"
                    )
                    return 1
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

            key = _real_strategy_course_market_key(state.win_snapshot, state.place_snapshot)
            if key not in stale_cleanup_keys and hasattr(provider, "cleanup_stale_command_cells"):
                cleanup = provider.cleanup_stale_command_cells(reason=f"course_change:{key}")
                stale_cleanup_keys.add(key)
                print(
                    f"tick={tick} market_change_command_cells_cleanup "
                    f"done={cleanup.get('done')} "
                    f"addresses={cleanup.get('addresses') or '-'} "
                    f"reason={cleanup.get('reason')}"
                )
                if cleanup.get("failed"):
                    unsafe_reason = cleanup.get("unsafe_stop_reason") or "unsafe_stale_gruss_triggers_cleanup_failed"
                    print(
                        f"ERREUR SECURITE: {unsafe_reason} "
                        f"tick={tick} reason={cleanup.get('reason')}"
                    )
                    return 1
            milestone = milestone_tracker.due_milestone(
                key,
                state.win_snapshot.metadata.countdown_seconds,
                state.place_snapshot.metadata.countdown_seconds,
            )
            if milestone is None:
                wait_reason = countdown_wait_reason(
                    state.win_snapshot.metadata.countdown_seconds,
                    state.place_snapshot.metadata.countdown_seconds,
                )
                if wait_reason is None:
                    wait_reason = "wait: no_strategy_milestone_due"
                print(f"tick={tick} {wait_reason}")
                _sleep(args, tick)
                continue
            print(f"tick={tick} {_describe_due_milestone(milestone)}")

            processed_key = _real_strategy_milestone_key(key, milestone)
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
                    force_milestone=milestone,
                )
                trade_rows = runner.trade_rows_since(trade_count_before)
                trade_rows_for_intents, promoted_pre_ladder_rows = prepare_trade_rows_for_real_provider(
                    trade_rows
                )
                if promoted_pre_ladder_rows:
                    print(
                        f"tick={tick} pre_ladder_real_promoted_from_preview="
                        f"{promoted_pre_ladder_rows}"
                    )
                intents = build_order_intents_from_trade_rows(
                    trade_rows_for_intents,
                    state.win_snapshot,
                    state.place_snapshot,
                )
                context = build_real_strategy_test_context(state, key, milestone_seen=milestone)
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

                update_trade_rows_with_real_results(
                    runner.trade_path(),
                    trade_count_before,
                    intents,
                    results,
                    context,
                )
                print(
                    f"tick={tick} course={context.course or key} "
                    f"countdown={context.countdown_seconds}s "
                    f"signaux_generes={len(trade_rows)} real_attempts={len(results)}"
                )
                if not results:
                    print(f"tick={tick} no signal")
                elif getattr(results[0], "total_runners_in_gruss_sheet", 0):
                    print(
                        f"tick={tick} runner_mapping "
                        f"total_runners_in_gruss_sheet={results[0].total_runners_in_gruss_sheet} "
                        f"mapped_runners_count={results[0].mapped_runners_count} "
                        f"unmapped_runners_count={results[0].unmapped_runners_count} "
                        f"mapped_selection_ids={results[0].mapped_selection_ids or '-'} "
                        f"unmapped_selection_ids={results[0].unmapped_selection_ids or '-'} "
                        f"mapped_excel_rows={results[0].mapped_excel_rows or '-'}"
                    )
                for index, result in enumerate(results, start=1):
                    print_real_attempt(tick, index, result)
            except Exception as exc:
                print(f"tick={tick} skip: evaluation/real-strategy-test impossible: {exc}")

            _sleep(args, tick)
    except KeyboardInterrupt:
        print("shutdown requested: KeyboardInterrupt")
    finally:
        if hasattr(provider, "drain_due_command_cell_clears"):
            provider.drain_due_command_cell_clears(force=True)
        if hasattr(provider, "cleanup_stale_command_cells"):
            cleanup = provider.cleanup_stale_command_cells(reason="shutdown")
            print(
                "shutdown_command_cells_cleanup "
                f"done={cleanup.get('done')} "
                f"addresses={cleanup.get('addresses') or '-'} "
                f"reason={cleanup.get('reason')}"
            )
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
    force_stake: float | None,
):
    """Process every real strategy signal so rejected max-order attempts are logged."""
    intents = list(intents)
    selected_intents, rejected_intents = resolve_back_lay_same_phase_conflicts(intents)
    all_intents = selected_intents + [intent for intent, _reason in rejected_intents]
    mapping_summary = {}
    if all_intents and hasattr(provider, "runner_mapping_summary"):
        try:
            mapping_summary = provider.runner_mapping_summary(all_intents)
        except Exception as exc:
            mapping_summary = {
                "mapping_reason": f"mapping_summary_failed:{exc}",
            }
    phase_keys = [
        _processed_intent_key(
            key,
            intent,
            context=context,
            win_market_id=win_market_id,
            place_market_id=place_market_id,
        )
        for intent in all_intents
    ]

    results = []
    if hasattr(provider, "set_batch_log_context"):
        provider.set_batch_log_context(**mapping_summary)
    for intent, reason in rejected_intents:
        phase_key = _processed_intent_key(
            key,
            intent,
            context=context,
            win_market_id=win_market_id,
            place_market_id=place_market_id,
        )
        if processed_store.has_seen(phase_key):
            if str(getattr(intent, "execution_phase", "") or "POST").strip().upper() == "POST":
                if hasattr(provider, "set_batch_log_context"):
                    provider.set_batch_log_context(
                        **mapping_summary,
                        post_provider_called=False,
                        post_processed_key=phase_key,
                        post_processed_key_scope="course_market",
                        processed_key_seen=True,
                        processed_key_seen_matching_existing_key=phase_key,
                    )
                results.append(provider.reject_order(intent, context, "post_provider_not_called_processed_key_seen"))
            continue
        if (
            reason == "conflicting_back_lay_no_bet"
            and str(getattr(intent, "execution_phase", "") or "POST").strip().upper() == "POST"
            and hasattr(provider, "set_batch_log_context")
        ):
            provider.set_batch_log_context(
                **mapping_summary,
                post_provider_called=False,
                post_write_attempted=False,
                post_write_status="REJECTED_REAL",
                post_write_reason=reason,
            )
        results.append(provider.reject_order(intent, context, reason))

    selected_to_process = []
    for intent in selected_intents:
        phase_key = _processed_intent_key(
            key,
            intent,
            context=context,
            win_market_id=win_market_id,
            place_market_id=place_market_id,
        )
        if processed_store.has_seen(phase_key):
            if str(getattr(intent, "execution_phase", "") or "POST").strip().upper() == "POST":
                if hasattr(provider, "set_batch_log_context"):
                    provider.set_batch_log_context(
                        **mapping_summary,
                        post_provider_called=False,
                        post_processed_key=phase_key,
                        post_processed_key_scope="course_market",
                        processed_key_seen=True,
                        processed_key_seen_matching_existing_key=phase_key,
                    )
                results.append(provider.reject_order(intent, context, "post_provider_not_called_processed_key_seen"))
            continue
        selected_to_process.append(intent)
    pre_post_independent = _pre_post_independent()
    pre_cancel_enabled = _pre_cancel_before_post_enabled()
    pre_cancel_skip_reason = ""
    if pre_post_independent:
        pre_cancel_skip_reason = "pre_post_independent"
    elif not pre_cancel_enabled:
        pre_cancel_skip_reason = "pre_cancel_before_post_disabled"
    if hasattr(provider, "cancel_pre_ladders_before_post") and not pre_cancel_skip_reason:
        cancel_results = provider.cancel_pre_ladders_before_post(
            context,
            post_intents=[
                intent
                for intent in selected_to_process
                if str(getattr(intent, "execution_phase", "") or "POST").strip().upper() == "POST"
            ],
        )
        results.extend(cancel_results)
    if _is_true(os.getenv("DOGBOT_POST_SKIP_IF_PRE_MATCHED")) and _has_post_intents(selected_to_process):
        if hasattr(provider, "has_matched_active_pre_ladder") and provider.has_matched_active_pre_ladder(context):
            for intent in selected_to_process:
                results.append(provider.reject_order(intent, context, "post_skipped_pre_matched"))
            selected_to_process = []
    batch_write_start_timestamp = _utc_now()
    batch_write_start = time.perf_counter()
    batch_processed_keys: list[str] = []
    selected_results = []
    direct_lim_candidates = [intent for intent in selected_to_process if _is_direct_lim_candidate(intent)]
    direct_lim_candidate_keys = {
        _direct_lim_candidate_key(intent): index
        for index, intent in enumerate(direct_lim_candidates, start=1)
    }
    pre_batch_candidates = [intent for intent in selected_to_process if _is_pre_initial_batch_candidate(intent)]
    pre_batch_candidate_keys = {
        _direct_lim_candidate_key(intent): index
        for index, intent in enumerate(pre_batch_candidates, start=1)
    }
    pre_batch_context = context
    pre_batch_milestone = context.milestone_seen if context.milestone_seen is not None else context.countdown_seconds
    pre_batch_authorized = bool(
        pre_batch_candidates
        and pre_batch_milestone is not None
        and _execution_phase_for_milestone(int(pre_batch_milestone)) == "PRE"
    )
    if pre_batch_authorized:
        pre_batch_context = replace(
            context,
            pre_batch_milestone_authorized=True,
            pre_batch_milestone_seconds=int(pre_batch_milestone),
            pre_batch_started_countdown_seconds=context.countdown_seconds,
            pre_batch_write_grace_seconds=_pre_initial_batch_write_grace_seconds(),
        )
    for order_index, intent in enumerate(selected_to_process, start=1):
        phase_key = _processed_intent_key(key, intent)
        stake_original = intent.stake
        direct_lim_candidate = _is_direct_lim_candidate(intent)
        direct_lim_candidate_index = direct_lim_candidate_keys.get(_direct_lim_candidate_key(intent), "")
        pre_batch_candidate_index = pre_batch_candidate_keys.get(_direct_lim_candidate_key(intent), "")
        if force_stake is None:
            real_intent = replace(
                intent,
                provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                dry_run=False,
                stake_original=stake_original,
                stake_forced=False,
            )
        else:
            real_intent = replace(
                intent,
                provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
                dry_run=False,
                stake=force_stake,
                stake_original=stake_original,
                stake_forced=True,
            )
        if hasattr(provider, "set_batch_log_context"):
            is_post_intent = str(getattr(real_intent, "execution_phase", "") or "POST").strip().upper() == "POST"
            phase_key_seen = processed_store.has_seen(phase_key)
            provider.set_batch_log_context(
                **mapping_summary,
                batch_size=len(selected_to_process),
                batch_write_start_timestamp=batch_write_start_timestamp,
                order_index_in_batch=order_index,
                direct_lim_candidates_count=len(direct_lim_candidates),
                direct_lim_candidate_index=direct_lim_candidate_index,
                direct_lim_provider_called=str(bool(direct_lim_candidate)),
                direct_lim_provider_skip_reason="" if direct_lim_candidate else "not_direct_lim_candidate",
                pre_batch_candidate_index=pre_batch_candidate_index,
                pre_batch_candidates_count=len(pre_batch_candidates),
                post_provider_called=is_post_intent,
                post_processed_key=phase_key if is_post_intent else "",
                post_processed_key_scope="course_market" if is_post_intent else "",
                parent_id=_parent_id_from_key(key),
                course_id=_course_id_from_context_or_intent(context, real_intent),
                win_market_id=win_market_id or "",
                place_market_id=place_market_id or "",
                processed_key_seen=phase_key_seen,
                processed_key_seen_matching_existing_key=phase_key if phase_key_seen else "",
                pre_post_independent=pre_post_independent,
                pre_existing_order_allowed=is_post_intent and pre_post_independent,
                pre_cancel_required_before_post=not pre_post_independent and pre_cancel_enabled,
                pre_cancel_skip_reason=pre_cancel_skip_reason if is_post_intent else "",
                stake_limit_scope="per_order",
                market_query=context.course or key,
            )
        result = provider.place_order(real_intent, pre_batch_context)
        results.append(result)
        selected_results.append(result)
        batch_processed_keys.append(result.processed_key)
        if _is_pre_initial_batch_candidate(real_intent):
            has_later_pre_initial = any(
                _is_pre_initial_batch_candidate(candidate)
                for candidate in selected_to_process[order_index:]
            )
            sleep_ms = _pre_batch_write_sleep_ms() if has_later_pre_initial else 0
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

    if selected_to_process and hasattr(provider, "update_batch_write_log"):
        direct_lim_results = [result for result in selected_results if _result_is_direct_lim_candidate(result)]
        direct_lim_written_count = sum(1 for result in direct_lim_results if bool(getattr(result, "direct_lim_order_written", False)))
        direct_lim_rejected_count = sum(1 for result in direct_lim_results if str(getattr(result, "status", "") or "").upper() != "GRUSS_PRE_LADDER_WRITTEN")
        provider.update_batch_write_log(
            batch_processed_keys,
            batch_write_end_timestamp=_utc_now(),
            batch_write_duration_ms=int(round((time.perf_counter() - batch_write_start) * 1000)),
            extra_fields={
                "direct_lim_batch_processed_count": len(direct_lim_results),
                "direct_lim_written_count": direct_lim_written_count,
                "direct_lim_rejected_count": direct_lim_rejected_count,
            },
        )
    if selected_results and hasattr(provider, "collect_pre_ladder_bet_refs"):
        provider.collect_pre_ladder_bet_refs(selected_results)
    if hasattr(provider, "clear_batch_log_context"):
        provider.clear_batch_log_context()

    processed_phase_keys = {
        _processed_intent_key(
            key,
            intent,
            context=context,
            win_market_id=win_market_id,
            place_market_id=place_market_id,
        )
        for intent, result in zip(selected_to_process, selected_results)
        if _real_result_should_mark_processed(result)
    }
    for phase_key in sorted(processed_phase_keys):
        processed_store.mark_seen(phase_key, win_market_id, place_market_id)
    return results, False


def update_trade_rows_with_real_results(
    trade_path: Path,
    row_count_before: int,
    intents: list[Any],
    results: list[Any],
    context: GrussRealOrderContext,
) -> int:
    """Reflect real Gruss provider outcomes back into the trades CSV rows."""
    if not trade_path.exists() or not results:
        return 0
    with trade_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows or "status" not in fieldnames:
        return 0

    result_by_key = {
        str(getattr(result, "processed_key", "") or ""): result
        for result in results
        if str(getattr(result, "processed_key", "") or "")
    }
    if not result_by_key:
        return 0

    intent_keys = {
        _real_provider_processed_key(intent, context)
        for intent in intents
    }
    target_keys = result_by_key.keys() & intent_keys
    if not target_keys:
        return 0

    changed = 0
    seen: set[str] = set()
    for row in rows[max(0, int(row_count_before)):]:
        row_key = _trade_row_real_provider_key(row, context)
        if row_key not in target_keys or row_key in seen:
            continue
        result = result_by_key[row_key]
        row["status"] = _trade_status_for_real_result(result)
        if "reason" in fieldnames:
            row["reason"] = str(getattr(result, "reason", "") or "")
        seen.add(row_key)
        changed += 1

    if not changed:
        return 0
    with trade_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return changed


def _trade_status_for_real_result(result: Any) -> str:
    status = str(getattr(result, "status", "") or "")
    if (
        status == "GRUSS_PRE_LADDER_WRITTEN"
        and bool(getattr(result, "direct_lim_order_written", False))
    ):
        return "DIRECT_LIM_ORDER_WRITTEN"
    return status


def _real_result_should_mark_processed(result: Any) -> bool:
    status = str(getattr(result, "status", "") or "").upper()
    return status in {"GRUSS_REAL_WRITTEN", "GRUSS_PRE_LADDER_WRITTEN"}


def _is_direct_lim_candidate(intent: Any) -> bool:
    return (
        bool(getattr(intent, "pre_ladder", False))
        and _execution_phase_for_intent(intent) == "PRE"
        and str(getattr(intent, "market_type", "") or "").upper() == "PLACE"
        and bool(
            getattr(intent, "direct_lim_order_planned", False)
            or getattr(intent, "direct_lim_order_written", False)
        )
        and bool(getattr(intent, "no_replace_steps_for_direct_lim", False))
        and _ladder_step_index(getattr(intent, "ladder_step", "")) == 0
    )


def _is_pre_initial_batch_candidate(intent: Any) -> bool:
    return (
        bool(getattr(intent, "pre_ladder", False))
        and _execution_phase_for_intent(intent) == "PRE"
        and _ladder_step_index(getattr(intent, "ladder_step", "")) == 0
    )


def _result_is_direct_lim_candidate(result: Any) -> bool:
    return (
        bool(getattr(result, "pre_ladder", False))
        and str(getattr(result, "execution_phase", "") or "").upper() == "PRE"
        and bool(
            getattr(result, "direct_lim_order_planned", False)
            or getattr(result, "direct_lim_order_written", False)
        )
        and bool(getattr(result, "no_replace_steps_for_direct_lim", False))
        and _ladder_step_index(getattr(result, "ladder_step", "")) == 0
    )


def _direct_lim_candidate_key(intent: Any) -> str:
    return "|".join(
        str(part)
        for part in (
            getattr(intent, "market_id", "") or "",
            getattr(intent, "selection_id", None) or getattr(intent, "trap", "") or "",
            getattr(intent, "side", "") or "",
            getattr(intent, "strategy_id", "") or "",
            getattr(intent, "ladder_id", "") or getattr(intent, "ladder_tracking_key", "") or "",
            getattr(intent, "ladder_step", "") or "",
        )
    )


def _pre_initial_batch_write_grace_seconds() -> int:
    raw = str(os.getenv("DOGBOT_PRE_INITIAL_BATCH_WRITE_GRACE_SECONDS", "10") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return 10
    if value < 0:
        return 10
    return min(value, 60)


def _pre_batch_write_sleep_ms() -> int:
    raw = str(os.getenv("DOGBOT_GRUSS_PRE_BATCH_WRITE_SLEEP_MS", "250") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return 250
    if value < 0:
        return 250
    return min(value, 5000)


def _trade_row_real_provider_key(row: Mapping[str, str], context: GrussRealOrderContext) -> str:
    race_id = str(
        context.course
        or row.get("course_id")
        or row.get("parent_id")
        or ""
    ).strip()
    selection_id = row.get("selection_id") or row.get("trap") or ""
    parts = [
        race_id,
        row.get("market_id") or "",
        selection_id,
        row.get("side") or "",
        row.get("market_type") or "",
        row.get("execution_phase") or "POST",
    ]
    if _is_true(row.get("pre_ladder")):
        parts.extend(
            [
                f"milestone={context.milestone_seen if context.milestone_seen is not None else context.countdown_seconds}",
                f"ladder_step={row.get('ladder_step') or ''}",
                f"ladder_id={row.get('ladder_id') or row.get('ladder_tracking_key') or ''}",
            ]
        )
    return "|".join(str(part) for part in parts)


def _real_provider_processed_key(intent: Any, context: GrussRealOrderContext) -> str:
    race_id = str(
        context.course
        or getattr(intent, "course_id", None)
        or getattr(intent, "parent_id", None)
        or ""
    ).strip()
    selection_id = getattr(intent, "selection_id", None)
    if selection_id is None:
        selection_id = getattr(intent, "trap", "")
    parts = [
        race_id,
        getattr(intent, "market_id", "") or "",
        selection_id,
        getattr(intent, "side", "") or "",
        getattr(intent, "market_type", "") or "",
        _execution_phase_for_intent(intent),
    ]
    if bool(getattr(intent, "pre_ladder", False)):
        parts.extend(
            [
                f"milestone={context.milestone_seen if context.milestone_seen is not None else context.countdown_seconds}",
                f"ladder_step={getattr(intent, 'ladder_step', '') or ''}",
                f"ladder_id={getattr(intent, 'ladder_id', None) or getattr(intent, 'ladder_tracking_key', None) or ''}",
            ]
        )
    return "|".join(str(part) for part in parts)


def prepare_trade_rows_for_real_provider(
    trade_rows: list[dict[str, str]],
    env: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, str]], int]:
    """Promote eligible PRE ladder preview rows for the real strategy watcher only.

    The executor keeps PRE ladder preview as the safe default. In this real test
    script, once the ladder is explicitly armed, eligible PLACE/PRE preview rows
    must be passed to the real Gruss provider so BACK/BACKR/CANCEL can be tested.
    """
    values = env if env is not None else os.environ
    if not _is_true(values.get("DOGBOT_PRE_LADDER_ENABLED")):
        return list(trade_rows), 0
    if _is_true(values.get("DOGBOT_PRE_LADDER_PREVIEW")):
        return list(trade_rows), 0

    prepared: list[dict[str, str]] = []
    promoted = 0
    for row in trade_rows:
        copied = dict(row)
        if _is_promotable_pre_ladder_preview_row(copied):
            copied["status"] = "PRE_LADDER_REAL_READY"
            copied["real_strategy_original_status"] = "PRE_LADDER_PREVIEW"
            copied["real_strategy_promotion_reason"] = "pre_ladder_real_enabled"
            promoted += 1
        prepared.append(copied)
    return prepared, promoted


def configure_real_pre_ladder_for_strategy_test(
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Arm the validated Gruss PRE ladder path for this real strategy test script."""
    target = env if env is not None else os.environ
    applied: dict[str, str] = {}
    for name, value in REAL_PRE_LADDER_ENV_DEFAULTS.items():
        current = str(target.get(name, "")).strip()
        if current != value:
            target[name] = value
        applied[name] = str(target.get(name, value))
    for name, value in REAL_PRE_LADDER_CONFIGURABLE_DEFAULTS.items():
        current = str(target.get(name, "")).strip()
        if not current:
            target[name] = value
        applied[name] = str(target.get(name, value))
    return applied


def load_strategy_test_env_file(
    env: dict[str, str] | None = None,
    env_path: Path | None = None,
) -> dict[str, str]:
    """Load local .env values before applying script defaults.

    Exported environment variables keep priority; the file only fills missing
    keys so a live shell override remains explicit and visible.
    """
    target = env if env is not None else os.environ
    path = env_path or (ROOT / ".env")
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name:
            continue
        if str(target.get(name, "")).strip():
            continue
        cleaned = _clean_env_file_value(value.strip())
        target[name] = cleaned
        loaded[name] = cleaned
    return loaded


def _clean_env_file_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _is_promotable_pre_ladder_preview_row(row: Mapping[str, str]) -> bool:
    if str(row.get("status", "")).strip().upper() != "PRE_LADDER_PREVIEW":
        return False
    if str(row.get("execution_phase", "")).strip().upper() != "PRE":
        return False
    if str(row.get("market_type", "")).strip().upper() != "PLACE":
        return False
    if not str(row.get("ladder_id") or "").strip():
        return False
    if not str(row.get("ladder_step") or "").strip():
        return False
    step_index = _ladder_step_index(row.get("ladder_step"))
    if _is_true(row.get("no_replace_steps_for_direct_lim")) and step_index > 0:
        return False
    frozen_prices = _split_ladder_prices(row.get("ladder_prices_frozen") or row.get("ladder_prices"))
    if len(frozen_prices) == 1 and step_index > 0:
        return False
    if len(frozen_prices) > 1 and len(set(frozen_prices)) == 1:
        return False
    if _positive_float(row.get("current_ladder_price")) is None:
        return False
    if _positive_float(row.get("current_step_stake")) is None:
        return False
    return True


def _ladder_step_index(value: object) -> int:
    text = str(value or "").strip()
    if "/" in text:
        text = text.split("/", 1)[0]
    try:
        number = int(text)
    except (TypeError, ValueError):
        return -1
    return max(0, number - 1)


def _split_ladder_prices(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def resolve_back_lay_same_phase_conflicts(intents: list[Any]) -> tuple[list[Any], list[tuple[Any, str]]]:
    grouped: dict[tuple[Any, ...], list[Any]] = {}
    for intent in intents:
        key = (
            getattr(intent, "course_id", None) or getattr(intent, "parent_id", None),
            getattr(intent, "market_id", None),
            getattr(intent, "selection_id", None) or getattr(intent, "trap", None),
            str(getattr(intent, "market_type", "")).upper(),
            _execution_phase_for_intent(intent),
        )
        grouped.setdefault(key, []).append(intent)

    decisions: dict[int, tuple[Any, str | None]] = {}
    for group in grouped.values():
        backs = [intent for intent in group if str(getattr(intent, "side", "")).upper() == "BACK"]
        lays = [intent for intent in group if str(getattr(intent, "side", "")).upper() == "LAY"]
        if not backs or not lays:
            continue
        decision = _reject_back_lay_group(backs, lays)
        for original, resolved, reason in decision:
            decisions[id(original)] = (resolved, reason)

    selected: list[Any] = []
    rejected: list[tuple[Any, str]] = []
    for intent in intents:
        decision = decisions.get(id(intent))
        if decision is None:
            selected.append(intent)
            continue
        resolved_intent, reason = decision
        if reason is None:
            selected.append(resolved_intent)
        else:
            rejected.append((resolved_intent, reason))
    return selected, rejected


def resolve_pre_ladder_back_lay_conflicts(intents: list[Any]) -> tuple[list[Any], list[tuple[Any, str]]]:
    return resolve_back_lay_same_phase_conflicts(intents)


def _reject_back_lay_group(backs: list[Any], lays: list[Any]) -> list[tuple[Any, Any, str]]:
    back = backs[0]
    lay = lays[0]
    back_price = _positive_float(getattr(back, "price", None))
    lay_price = _positive_float(getattr(lay, "price", None))
    reference = _conflict_market_reference(back, lay)
    group_key = _conflict_group_key(back)
    execution_phase = _execution_phase_for_intent(back)
    back_edge = _numeric_or_none(getattr(back, "strategy_edge", None))
    lay_edge = _numeric_or_none(getattr(lay, "strategy_edge", None))
    back_score = _numeric_or_none(getattr(back, "strategy_score", None))
    lay_score = _numeric_or_none(getattr(lay, "strategy_score", None))
    reason = "conflicting_back_lay_no_bet"
    base = {
        "conflict_detected": True,
        "conflict_type": "back_lay_same_runner_market_phase",
        "back_price": back_price,
        "lay_price": lay_price,
        "market_reference_price": reference,
        "conflict_group_key": group_key,
        "conflict_candidates_count": len(backs) + len(lays),
        "selected_side": "NONE",
        "rejected_side": "BOTH",
        "winning_side": "",
        "losing_side": "",
        "winning_strategy_id": "",
        "losing_strategy_id": "",
        "winning_edge": back_edge,
        "losing_edge": lay_edge,
        "winning_score": back_score,
        "losing_score": lay_score,
        "winning_lim_price": back_price,
        "losing_lim_price": lay_price,
        "back_systems": "|".join(_intent_strategy_id(intent) for intent in backs),
        "lay_systems": "|".join(_intent_strategy_id(intent) for intent in lays),
        "conflict_resolution_reason": reason,
    }
    if execution_phase == "PRE":
        return _resolve_pre_back_lay_group(
            backs,
            lays,
            base,
            back_price=back_price,
            lay_price=lay_price,
        )
    back_distance = (
        abs(back_price - reference) / reference
        if reference is not None and back_price is not None
        else None
    )
    lay_distance = (
        abs(lay_price - reference) / reference
        if reference is not None and lay_price is not None
        else None
    )
    fields = {**base, "back_distance": back_distance, "lay_distance": lay_distance}
    return [(intent, replace(intent, **fields), reason) for intent in [*backs, *lays]]


def _resolve_pre_back_lay_group(
    backs: list[Any],
    lays: list[Any],
    base: dict[str, Any],
    *,
    back_price: float | None,
    lay_price: float | None,
) -> list[tuple[Any, Any, str | None]]:
    back_distance = _nearest_pre_conflict_distance(backs, reference_attr="best_same_side_lay_offer")
    lay_distance = _nearest_pre_conflict_distance(lays, reference_attr="best_same_side_back_offer")
    if back_distance is None or lay_distance is None:
        chosen_side = "NONE"
        rejected_side = "BOTH"
        pre_reason = "pre_conflict_missing_reference_no_bet"
    elif math.isclose(back_distance, lay_distance, rel_tol=1e-9, abs_tol=1e-9):
        chosen_side = "NONE"
        rejected_side = "BOTH"
        pre_reason = "pre_conflict_equal_distance_no_bet"
    elif back_distance < lay_distance:
        chosen_side = "BACK"
        rejected_side = "LAY"
        pre_reason = "pre_conflict_back_nearer"
    else:
        chosen_side = "LAY"
        rejected_side = "BACK"
        pre_reason = "pre_conflict_lay_nearer"
    back_reference = _positive_float(getattr(backs[0], "best_same_side_lay_offer", None)) if backs else None
    lay_reference = _positive_float(getattr(lays[0], "best_same_side_back_offer", None)) if lays else None

    fields = {
        **base,
        "conflict_resolution_reason": "per_runner_nearest_price",
        "back_distance": back_distance,
        "lay_distance": lay_distance,
        "selected_side": chosen_side,
        "rejected_side": rejected_side,
        "pre_back_lay_conflict": True,
        "pre_conflict_resolution": "per_runner_nearest_price",
        "pre_conflict_chosen_side": chosen_side,
        "pre_conflict_rejected_side": rejected_side,
        "pre_conflict_reason": pre_reason,
        "pre_conflict_group_key": base.get("conflict_group_key", ""),
        "pre_conflict_course_id": getattr(backs[0], "course_id", None) or getattr(backs[0], "parent_id", ""),
        "pre_conflict_market_id": getattr(backs[0], "market_id", ""),
        "pre_conflict_market_type": str(getattr(backs[0], "market_type", "")).upper(),
        "pre_conflict_selection_id": getattr(backs[0], "selection_id", None) or getattr(backs[0], "trap", ""),
        "pre_conflict_runner_name": getattr(backs[0], "runner_name", "") or getattr(lays[0], "runner_name", ""),
        "pre_back_target_price": back_price,
        "pre_lay_target_price": lay_price,
        "pre_current_best_lay": back_reference,
        "pre_current_best_back": lay_reference,
        "pre_back_distance_ticks": back_distance,
        "pre_lay_distance_ticks": lay_distance,
    }
    if chosen_side == "NONE":
        return [(intent, replace(intent, **fields), pre_reason) for intent in [*backs, *lays]]

    resolved: list[tuple[Any, Any, str | None]] = []
    for intent in backs:
        if chosen_side == "BACK":
            resolved.append((intent, replace(intent, **fields), None))
        else:
            resolved.append((intent, replace(intent, **fields), "conflicting_back_lay_lost_priority"))
    for intent in lays:
        if chosen_side == "LAY":
            resolved.append((intent, replace(intent, **fields), None))
        else:
            resolved.append((intent, replace(intent, **fields), "conflicting_back_lay_lost_priority"))
    return resolved


def _nearest_pre_conflict_distance(candidates: list[Any], *, reference_attr: str) -> float | None:
    distances: list[float] = []
    for candidate in candidates:
        distance = _betfair_tick_distance(getattr(candidate, "price", None), getattr(candidate, reference_attr, None))
        if distance is not None:
            distances.append(distance)
    return min(distances) if distances else None


def _intent_strategy_id(intent: Any) -> str:
    systems = str(getattr(intent, "triggered_systems", "") or "").strip()
    if systems:
        return systems
    return str(getattr(intent, "strategy_id", "") or "").strip()


def _conflict_annotation_fields(
    base: dict[str, Any],
    *,
    winner: Any | None,
    loser: Any | None,
    reason: str,
) -> dict[str, Any]:
    fields = dict(base)
    if winner is not None:
        fields.update(
            {
                "winning_side": str(getattr(winner, "side", "")).upper(),
                "winning_strategy_id": getattr(winner, "strategy_id", "") or "",
                "winning_edge": _numeric_or_none(getattr(winner, "strategy_edge", None)),
                "winning_score": _numeric_or_none(getattr(winner, "strategy_score", None)),
                "winning_lim_price": _positive_float(getattr(winner, "price", None)),
            }
        )
    if loser is not None:
        fields.update(
            {
                "losing_side": str(getattr(loser, "side", "")).upper(),
                "losing_strategy_id": getattr(loser, "strategy_id", "") or "",
                "losing_edge": _numeric_or_none(getattr(loser, "strategy_edge", None)),
                "losing_score": _numeric_or_none(getattr(loser, "strategy_score", None)),
                "losing_lim_price": _positive_float(getattr(loser, "price", None)),
            }
        )
    fields["conflict_resolution_reason"] = reason
    return fields


def _conflict_group_key(intent: Any) -> str:
    return "|".join(
        str(part or "")
        for part in (
            getattr(intent, "course_id", None) or getattr(intent, "parent_id", ""),
            getattr(intent, "market_id", ""),
            str(getattr(intent, "market_type", "")).upper(),
            getattr(intent, "selection_id", None) or getattr(intent, "trap", ""),
            _execution_phase_for_intent(intent),
        )
    )


def _numeric_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _relative_distance(price: Any, reference: Any) -> float | None:
    try:
        price_value = float(price)
        reference_value = float(reference)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price_value) or not math.isfinite(reference_value) or reference_value <= 0:
        return None
    return abs(price_value - reference_value) / reference_value


def _betfair_tick_distance(price: Any, reference: Any) -> float | None:
    price_value = _positive_float(price)
    reference_value = _positive_float(reference)
    if price_value is None or reference_value is None:
        return None
    ticks = _betfair_ticks_for_distance()
    try:
        price_tick = round_to_betfair_tick(price_value)
        reference_tick = round_to_betfair_tick(reference_value)
        return float(abs(ticks.index(price_tick) - ticks.index(reference_tick)))
    except (TypeError, ValueError):
        return None


def _betfair_ticks_for_distance() -> tuple[float, ...]:
    ticks: list[float] = []
    for start, end, step in BETFAIR_PRICE_BANDS:
        value = start
        while value < end - 1e-9:
            rounded = round(value, 2)
            if not ticks or not math.isclose(ticks[-1], rounded, rel_tol=1e-9, abs_tol=1e-9):
                ticks.append(rounded)
            value += step
    ticks.append(1000.0)
    return tuple(ticks)


def _conflict_market_reference(back: Any, lay: Any) -> float | None:
    back_offer = _positive_float(getattr(back, "best_same_side_back_offer", None))
    lay_offer = _positive_float(getattr(lay, "best_same_side_lay_offer", None))
    midpoint = (back_offer + lay_offer) / 2.0 if back_offer is not None and lay_offer is not None else None
    return _first_positive(
        getattr(back, "market_reference_price_at_signal", None),
        getattr(lay, "market_reference_price_at_signal", None),
        midpoint,
        back_offer,
        lay_offer,
    )


def _is_pre_ladder_place_intent(intent: Any) -> bool:
    return (
        bool(getattr(intent, "pre_ladder", False))
        and str(getattr(intent, "market_type", "")).upper() == "PLACE"
        and _execution_phase_for_intent(intent) == "PRE"
    )


def _has_post_intents(intents: list[Any]) -> bool:
    return any(_execution_phase_for_intent(intent) == "POST" for intent in intents)


def _first_positive(*values: Any) -> float | None:
    for value in values:
        number = _positive_float(value)
        if number is not None:
            return number
    return None


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 1.0:
        return None
    return number


class _MilestoneTracker:
    def __init__(self) -> None:
        self._last_countdown_by_key: dict[str, int] = {}

    def due_milestone(
        self,
        key: str,
        win_countdown_seconds: int | None,
        place_countdown_seconds: int | None,
    ) -> int | None:
        seconds = _countdown_value(win_countdown_seconds, place_countdown_seconds)
        if seconds is None:
            return None
        milestones = active_strategy_milestones()
        exact = current_strategy_milestone(seconds, seconds)
        last = self._last_countdown_by_key.get(key)
        if exact is not None:
            self._last_countdown_by_key[key] = seconds
            return exact
        if last is None:
            self._last_countdown_by_key[key] = seconds
            return None
        crossed = [milestone for milestone in milestones if last > milestone >= seconds]
        if crossed:
            milestone = max(crossed)
            self._last_countdown_by_key[key] = milestone
            return milestone
        self._last_countdown_by_key[key] = seconds
        return None


def _countdown_value(
    win_countdown_seconds: int | None,
    place_countdown_seconds: int | None,
) -> int | None:
    seconds = win_countdown_seconds
    if seconds is None:
        seconds = place_countdown_seconds
    if seconds is None:
        return None
    try:
        return int(seconds)
    except (TypeError, ValueError):
        return None


def _describe_due_milestone(milestone: int) -> str:
    phase = _execution_phase_for_milestone(milestone) or "UNKNOWN"
    parts = [f"milestone={milestone}", f"execution_phase={phase}"]
    if phase == "PRE":
        steps = active_strategy_milestones()
        pre_steps = [step for step in steps if _execution_phase_for_milestone(step) == "PRE"]
        if milestone in pre_steps:
            parts.append(f"pre_ladder_step={pre_steps.index(milestone) + 1}/{len(pre_steps)}")
        parts.append("evaluating PRE systems")
    elif phase == "POST":
        parts.append("evaluating POST systems")
    return " ".join(parts)


def _execution_phase_for_intent(intent) -> str:
    phase = str(getattr(intent, "execution_phase", "") or "POST").strip().upper()
    return phase if phase in {"PRE", "POST"} else "POST"


def _processed_intent_key(
    key: str,
    intent,
    *,
    context: GrussRealOrderContext | None = None,
    win_market_id: str | None = None,
    place_market_id: str | None = None,
) -> str:
    phase = _execution_phase_for_intent(intent)
    scoped_key = _course_market_scope_key(
        key,
        context=context,
        intent=intent,
        win_market_id=win_market_id,
        place_market_id=place_market_id,
    )
    if getattr(intent, "pre_ladder", False):
        ladder_step = str(getattr(intent, "ladder_step", "") or "").strip() or "unknown_step"
        ladder_id = str(getattr(intent, "ladder_id", "") or "").strip() or "unknown_ladder"
        market_id = str(getattr(intent, "market_id", "") or "").strip() or "unknown_market"
        selection_id = str(
            getattr(intent, "selection_id", None)
            or getattr(intent, "trap", "")
            or "unknown_selection"
        ).strip()
        side = str(getattr(intent, "side", "") or "").strip().upper() or "unknown_side"
        strategy = str(getattr(intent, "strategy_id", "") or "").strip() or "unknown_strategy"
        return (
            f"{scoped_key}|{phase}|market={market_id}|selection={selection_id}|side={side}|"
            f"strategy={strategy}|{ladder_step}|{ladder_id}"
        )
    return _processed_phase_key(scoped_key, phase)


def _processed_phase_key(key: str, execution_phase: str) -> str:
    return f"{key}|{execution_phase}"


def _real_strategy_course_market_key(win_snapshot, place_snapshot) -> str:
    parent = str(win_snapshot.metadata.parent_id or place_snapshot.metadata.parent_id or "").strip()
    course = str(
        getattr(win_snapshot.metadata, "course_id", "")
        or getattr(place_snapshot.metadata, "course_id", "")
        or win_snapshot.metadata.event_path
        or place_snapshot.metadata.event_path
        or ""
    ).strip()
    win_market = str(win_snapshot.metadata.market_id or "").strip()
    place_market = str(place_snapshot.metadata.market_id or "").strip()
    parts = []
    if parent:
        parts.append(f"parent:{parent}")
    if course:
        parts.append(f"course:{course}")
    parts.extend([f"win:{win_market or 'unknown'}", f"place:{place_market or 'unknown'}"])
    return "|".join(parts)


def _real_strategy_milestone_key(key: str, milestone: int) -> str:
    phase = _execution_phase_for_milestone(milestone) or "UNKNOWN"
    return f"{key}|milestone:{milestone}|phase:{phase}"


def _course_market_scope_key(
    key: str,
    *,
    context: GrussRealOrderContext | None = None,
    intent: Any | None = None,
    win_market_id: str | None = None,
    place_market_id: str | None = None,
) -> str:
    parent = _parent_id_from_key(key)
    course = _course_id_from_context_or_intent(context, intent)
    win_market = str(win_market_id or getattr(context, "win_market_id", "") or "").strip()
    place_market = str(place_market_id or getattr(context, "place_market_id", "") or "").strip()
    milestone = _milestone_from_key(key)
    if intent is not None:
        market_type = str(getattr(intent, "market_type", "") or "").strip().upper()
        market_id = str(getattr(intent, "market_id", "") or "").strip()
        if market_type == "WIN" and not win_market:
            win_market = market_id
        elif market_type == "PLACE" and not place_market:
            place_market = market_id
    if not (parent or course or win_market or place_market):
        return key
    return "|".join(
        part
        for part in (
            f"parent:{parent}" if parent else "",
            f"course:{course}" if course else "",
            f"win:{win_market or 'unknown'}",
            f"place:{place_market or 'unknown'}",
            f"milestone:{milestone}" if milestone else "",
        )
        if part
    )


def _parent_id_from_key(key: str) -> str:
    for part in str(key or "").split("|"):
        if part.startswith("parent:"):
            return part.split(":", 1)[1]
    return ""


def _milestone_from_key(key: str) -> str:
    for part in str(key or "").split("|"):
        if part.startswith("milestone:"):
            return part.split(":", 1)[1]
        if part.startswith("milestone="):
            return part.split("=", 1)[1]
    return ""


def _course_id_from_context_or_intent(
    context: GrussRealOrderContext | None,
    intent: Any | None,
) -> str:
    for value in (
        getattr(intent, "course_id", None),
        getattr(context, "course", None),
        getattr(intent, "parent_id", None),
    ):
        text = str(value or "").strip()
        if text and not text.startswith(("parent:", "markets:")):
            return text
    return ""


def _pre_post_independent() -> bool:
    return _is_true(os.getenv("DOGBOT_PRE_POST_INDEPENDENT", "false"))


def _pre_cancel_before_post_enabled() -> bool:
    return _is_true(os.getenv("DOGBOT_PRE_CANCEL_BEFORE_POST", "false"))


def validate_real_strategy_test_environment(
    env: Mapping[str, str] | None = None,
) -> tuple[int, float, float | None]:
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
    if _is_true(values.get("DOGBOT_PRE_LADDER_ENABLED")) and _is_true(
        values.get("DOGBOT_PRE_LADDER_PREVIEW")
    ):
        raise RuntimeError("DOGBOT_PRE_LADDER_PREVIEW=false est obligatoire pour le test ladder reel")
    if _is_true(values.get("DOGBOT_PRE_LADDER_ENABLED")) and _is_true(
        values.get("DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST")
    ):
        raise RuntimeError("DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST=true est interdit avec le ladder reel")
    if _is_true(values.get("DOGBOT_PRE_LADDER_ENABLED")):
        max_ladders = _required_int(values, "DOGBOT_PRE_LADDER_REAL_MAX_LADDERS")
        if max_ladders <= 0:
            raise RuntimeError("DOGBOT_PRE_LADDER_REAL_MAX_LADDERS doit etre un entier positif")

    max_orders = _required_int(values, "DOGBOT_GRUSS_REAL_MAX_ORDERS")
    if max_orders <= 0:
        raise RuntimeError("DOGBOT_GRUSS_REAL_MAX_ORDERS doit etre un entier positif")
    variable_stakes = _is_true(values.get("DOGBOT_GRUSS_REAL_VARIABLE_STAKES"))
    max_stake = _required_float(values, "DOGBOT_GRUSS_REAL_MAX_STAKE")
    if not math.isfinite(max_stake) or max_stake <= 0:
        raise RuntimeError("DOGBOT_GRUSS_REAL_MAX_STAKE doit etre positif")
    if variable_stakes:
        if max_stake > 5:
            raise RuntimeError("DOGBOT_GRUSS_REAL_MAX_STAKE doit etre <= 5 en mode variable")
        force_stake = None
    else:
        if max_stake != 2:
            raise RuntimeError("DOGBOT_GRUSS_REAL_MAX_STAKE doit etre exactement 2 en mode force_stake")
        force_stake = _required_float(values, "DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE")
        if not math.isfinite(force_stake) or force_stake != 2:
            raise RuntimeError("DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE doit etre exactement 2 en mode force_stake")
        if force_stake > max_stake:
            raise RuntimeError(
                "DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE doit etre <= DOGBOT_GRUSS_REAL_MAX_STAKE"
            )
    return max_orders, max_stake, force_stake


def build_real_strategy_test_context(
    state,
    key: str,
    *,
    milestone_seen: int | None = None,
) -> GrussRealOrderContext:
    seconds = state.win_snapshot.metadata.countdown_seconds
    if seconds is None:
        seconds = state.place_snapshot.metadata.countdown_seconds
    effective_countdown = milestone_seen if milestone_seen is not None else seconds
    return GrussRealOrderContext(
        validation_ok=not state.validation_warnings,
        tradable=state.tradable,
        region=gruss_region_for_snapshots(state.win_snapshot, state.place_snapshot),
        countdown_seconds=effective_countdown,
        course=str(state.win_snapshot.metadata.event_path or state.place_snapshot.metadata.event_path or key),
        market_already_processed=False,
        win_market_id=state.win_snapshot.metadata.market_id,
        place_market_id=state.place_snapshot.metadata.market_id,
        milestone_seen=milestone_seen,
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
        f"ladder_step={getattr(result, 'ladder_step', '-') or '-'} "
        f"sheet={result.excel_sheet or '-'} row={result.excel_row or '-'} "
        f"mapping_found={getattr(result, 'mapping_found', False)} "
        f"mapping_reason={getattr(result, 'mapping_reason', '') or '-'} "
        f"command_cells={getattr(result, 'command_cells', '') or '-'} "
        f"cells={cells} trigger={result.intended_trigger or '-'} "
        f"trigger_cell_address={result.trigger_cell_address or '-'} "
        f"trigger_written={result.trigger_written} "
        f"trigger_value_written={result.trigger_value_written or '-'} "
        f"bet_ref_before={getattr(result, 'bet_ref_before', '') or '-'} "
        f"bet_ref_after={getattr(result, 'bet_ref_after', '') or '-'} "
        f"update_allowed={getattr(result, 'update_allowed', False)} "
        f"update_skipped_reason={getattr(result, 'update_skipped_reason', '') or '-'} "
        f"matched_stake={getattr(result, 'matched_stake', None)} "
        f"active_pre_ladder_id={getattr(result, 'active_pre_ladder_id', '') or '-'} "
        f"continuing_active_pre_ladder={getattr(result, 'continuing_active_pre_ladder', False)} "
        f"milestone_seen={getattr(result, 'milestone_seen', '') or '-'} "
        f"next_ladder_step_due={getattr(result, 'next_ladder_step_due', '') or '-'} "
        f"skipped_step_reason={getattr(result, 'skipped_step_reason', '') or '-'} "
        f"processed_key={getattr(result, 'processed_key', '') or '-'} "
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
