from __future__ import annotations

import csv
import math
import os
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dogbot.config import ORDER_PROVIDER_GRUSS_EXCEL_REAL
from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH, GrussExcelBridge
from dogbot.gruss.gruss_mapper import extract_trap, normalize_runner_name, parse_countdown_seconds
from dogbot.gruss.gruss_orders import OrderIntent, validate_order_intent
from dogbot.pre_ladder import round_final_lim_to_ladder_tick, round_to_betfair_tick


GRUSS_REAL_ATTEMPTS_HEADER = [
    "timestamp",
    "dry_run_or_real",
    "enabled",
    "provider",
    "course",
    "market_id",
    "market_type",
    "runner",
    "trap",
    "selection_id",
    "side",
    "order_type",
    "execution_phase",
    "processed_key",
    "post_processed_key",
    "post_processed_key_scope",
    "parent_id",
    "course_id",
    "win_market_id",
    "place_market_id",
    "processed_key_seen",
    "processed_key_seen_matching_existing_key",
    "pre_post_independent",
    "pre_existing_order_allowed",
    "pre_cancel_required_before_post",
    "stake_limit_scope",
    "triggered_systems",
    "triggered_prices",
    "pre_ladder",
    "ladder_id",
    "ladder_step",
    "active_pre_ladder_id",
    "continuing_active_pre_ladder",
    "active_pre_ladder_count",
    "max_active_pre_ladders",
    "configured_ladder_steps",
    "ladder_plan_frozen",
    "ladder_plan_created_step",
    "ladder_prices_frozen",
    "current_ladder_price_from_frozen_plan",
    "computed_limit_price_raw",
    "computed_limit_price_effective",
    "min_price_floor_applied",
    "pre_value_target_price",
    "ladder_planned_price",
    "sent_price_before_value_clamp",
    "sent_price_after_value_clamp",
    "value_clamp_applied",
    "value_limit_breached",
    "value_limit_skip_reason",
    "tick_rounding_direction",
    "best_same_side_offer_at_creation",
    "best_back_displayed",
    "best_lay_displayed",
    "start_price_source",
    "final_lim_price",
    "ladder_direction",
    "ladder_disabled_lim_not_in_ladder_direction",
    "direct_lim_order_planned",
    "direct_lim_order_written",
    "direct_lim_candidates_count",
    "direct_lim_candidate_index",
    "direct_lim_provider_called",
    "direct_lim_provider_skip_reason",
    "direct_lim_batch_processed_count",
    "direct_lim_written_count",
    "direct_lim_rejected_count",
    "no_replace_steps_for_direct_lim",
    "current_milestone",
    "computed_step_index",
    "expected_ladder_step",
    "milestone_seen",
    "next_ladder_step_due",
    "skipped_step_reason",
    "active_ladder_completed",
    "active_ladder_release_reason",
    "countdown_authorization_reason",
    "signal_timestamp",
    "write_timestamp",
    "write_delay_since_signal_seconds",
    "countdown_at_signal",
    "countdown_at_write",
    "pre_batch_milestone_authorized",
    "pre_batch_milestone_seconds",
    "pre_batch_started_countdown_seconds",
    "pre_batch_write_grace_seconds",
    "pre_batch_candidate_index",
    "pre_batch_candidates_count",
    "pre_batch_late_write_allowed",
    "pre_batch_late_write_seconds_after_start",
    "no_stacking_check_passed",
    "active_ladder_count",
    "max_ladders_limit",
    "market_reference_price_at_signal",
    "current_market_price_at_write",
    "stale_distance",
    "stale_price_limit",
    "stale_check_ignored_for_pre",
    "conflict_detected",
    "conflict_type",
    "back_price",
    "lay_price",
    "market_reference_price",
    "back_distance",
    "lay_distance",
    "selected_side",
    "rejected_side",
    "conflict_group_key",
    "conflict_candidates_count",
    "winning_side",
    "losing_side",
    "winning_strategy_id",
    "losing_strategy_id",
    "winning_edge",
    "losing_edge",
    "winning_score",
    "losing_score",
    "winning_lim_price",
    "losing_lim_price",
    "back_systems",
    "lay_systems",
    "conflict_resolution_reason",
    "pre_back_lay_conflict",
    "pre_conflict_resolution",
    "pre_conflict_chosen_side",
    "pre_conflict_rejected_side",
    "pre_conflict_reason",
    "pre_conflict_group_key",
    "pre_conflict_course_id",
    "pre_conflict_market_id",
    "pre_conflict_market_type",
    "pre_conflict_selection_id",
    "pre_conflict_runner_name",
    "pre_back_target_price",
    "pre_lay_target_price",
    "pre_current_best_lay",
    "pre_current_best_back",
    "pre_back_distance_ticks",
    "pre_lay_distance_ticks",
    "intended_trigger",
    "trigger",
    "stake",
    "stake_original",
    "stake_used",
    "stake_forced",
    "stake_min_floor_applied",
    "stake_before_min_floor",
    "stake_after_min_floor",
    "stake_final",
    "stake_capped",
    "stake_cap_value",
    "staking_formula",
    "staking_alpha",
    "staking_back_alpha",
    "staking_lay_alpha",
    "stake_raw_before_caps",
    "stake_after_caps",
    "lay_liability_after_sizing",
    "lay_liability_cap",
    "lay_liability_cap_hit",
    "force_test_bsp_place",
    "force_test_back_place_limit",
    "selected_reason",
    "selected_runner",
    "selected_trap",
    "selected_place_odds",
    "selected_place_back_odds",
    "selected_place_lay_odds",
    "price_used",
    "price",
    "price_raw_before_tick",
    "price_tick_rounded",
    "price_tick_rounding_side",
    "price_is_valid_betfair_tick",
    "strategy_id",
    "status",
    "reason",
    "mapping_found",
    "mapping_reason",
    "command_cells",
    "total_runners_in_gruss_sheet",
    "raw_gruss_runner_rows",
    "raw_selection_ids_seen",
    "raw_runner_names_seen",
    "mapped_runners_count",
    "unmapped_runners_count",
    "mapped_selection_ids",
    "unmapped_selection_ids",
    "ignored_runner_rows",
    "ignored_runner_reason",
    "mapped_excel_rows",
    "excel_sheet",
    "excel_row",
    "excel_cells_written",
    "cells_written",
    "excel_write_attempt",
    "excel_write_retry_count",
    "excel_write_retry_backoff_ms",
    "excel_write_final_status",
    "excel_unavailable_recovered",
    "excel_operation_name",
    "excel_com_attempt",
    "excel_com_retry_count",
    "excel_com_retry_backoff_ms",
    "excel_com_retryable_error",
    "mapping_attempt_count",
    "cleanup_retry_count",
    "cleanup_final_status",
    "trigger_cell_address",
    "trigger_cell_current_value",
    "trigger_cell_expected_empty",
    "trigger_mapping_name",
    "trigger_written",
    "trigger_value_written",
    "action",
    "bet_ref_before",
    "bet_ref_after",
    "bet_ref_poll_attempts",
    "bet_ref_poll_duration_ms",
    "pre_write_attempt_id",
    "pre_bet_ref_required",
    "pre_bet_ref_confirmed",
    "pre_bet_ref_found",
    "pre_bet_ref_missing",
    "pre_bet_ref_poll_attempts",
    "pre_bet_ref_poll_duration_ms",
    "pre_bet_ref_missing_retryable",
    "pre_bet_ref_late_detected",
    "pre_bet_ref_late_value",
    "pre_retry_count",
    "pre_retry_allowed",
    "pre_retry_reason",
    "pre_retry_block_reason",
    "pre_unconfirmed_reason",
    "bet_ref_lookup_sources",
    "bet_ref_lookup_source_used",
    "bet_ref_lookup_source",
    "bet_ref_lookup_matched_runner",
    "row_t_value",
    "selections_rows_scanned",
    "selections_match_found",
    "selections_match_reason",
    "selections_runner",
    "selections_side",
    "selections_stake",
    "selections_bet_ref",
    "selections_req_odds",
    "selections_market_name",
    "selections_debug_recent_rows",
    "selections_top_candidates",
    "bet_ref_row_t_dump",
    "bet_ref_diagnostic_hold_after_batch",
    "selections_market_query",
    "selections_current_market_rows",
    "selections_current_runner_rows",
    "runner_qz_dump",
    "selections_sheet_headers",
    "selections_full_recent_rows",
    "workbook_sheet_names",
    "diagnostic_keep_triggers",
    "active_ladder_bet_ref_stored",
    "active_ladder_created",
    "pending_ladder_created",
    "matched_evidence_found",
    "selection_row_evidence_found",
    "no_stacking_blocked_retry",
    "replace_allowed",
    "replace_trigger",
    "bet_ref_suffix_n_handled",
    "bet_ref_status_value",
    "replace_bet_ref_wait_attempted",
    "replace_bet_ref_wait_ms",
    "replace_bet_ref_poll_ms",
    "replace_bet_ref_wait_result",
    "bet_ref_before_wait",
    "bet_ref_after_wait",
    "active_ladder_bet_ref_updated",
    "replace_skipped_bet_ref_still_pending",
    "pre_ladder_initial_order_failed",
    "pre_ladder_disabled_after_initial_failure",
    "no_replace_steps_for_failed_initial",
    "requested_price",
    "requested_stake",
    "ladder_step_index",
    "ladder_step_count",
    "matched_after_step",
    "matched_after_step_avg_odds",
    "matched_after_step_stake",
    "avg_matched_odds_cell_address",
    "avg_matched_odds_cell_value",
    "matched_stake_cell_address",
    "matched_stake_cell_value",
    "profit_loss_cell_address",
    "profit_loss_cell_value",
    "batch_size",
    "batch_write_start_timestamp",
    "batch_write_end_timestamp",
    "batch_write_duration_ms",
    "order_index_in_batch",
    "bet_ref_collection_phase_start",
    "bet_ref_collection_phase_end",
    "bet_ref_collection_duration_ms",
    "bet_ref_found_count",
    "bet_ref_missing_count",
    "runner_row",
    "runner_order_in_sheet",
    "update_allowed",
    "update_skipped_reason",
    "matched_stake",
    "pre_cancel_attempted",
    "pre_cancel_written",
    "pre_cancel_skip_reason",
    "pre_cancel_only_if_post_pending",
    "post_pending_for_runner",
    "post_after_pre_cancel_attempted",
    "bet_ref_at_cancel",
    "matched_stake_at_cancel",
    "countdown_seconds_at_cancel",
    "trigger_clear_attempted",
    "trigger_cleared",
    "trigger_clear_reason",
    "trigger_cell_value_before_clear",
    "trigger_clear_delay_ms",
    "command_cells_clear_attempted",
    "command_cells_cleared",
    "command_cells_clear_reason",
    "command_cells_clear_addresses",
    "command_cells_clear_delay_ms",
    "command_cells_clear_scheduled",
    "command_cells_clear_due_time",
    "command_cells_clear_non_blocking",
    "command_cells_clear_executed",
    "command_cells_clear_lag_ms",
    "startup_command_cells_cleanup_attempted",
    "startup_command_cells_cleanup_done",
    "stale_command_cells_cleanup_attempted",
    "stale_command_cells_cleanup_addresses",
    "stale_command_cells_cleanup_reason",
    "stale_scan_attempt_count",
    "stale_scan_retry_count",
    "stale_scan_recovered",
    "stale_triggers_confirmed",
    "stale_cleanup_retry_count",
    "stale_cleanup_recovered",
    "stale_cleanup_final_status",
    "unsafe_stop_reason",
    "shutdown_command_cells_cleanup_done",
    "market_change_command_cells_cleanup_done",
    "post_write_odds_cell_address",
    "post_write_odds_value",
    "post_write_stake_cell_address",
    "post_write_stake_value",
    "post_write_trigger_cell_address",
    "post_write_trigger_value",
    "post_write_verified",
    "post_provider_called",
    "post_batch_id",
    "post_batch_market_id",
    "post_batch_market_name",
    "post_batch_candidate_count",
    "post_batch_written_count",
    "post_batch_write_duration_ms",
    "post_batch_confirmation_started",
    "post_batch_confirmation_duration_ms",
    "post_batch_runner_index",
    "post_batch_total_runners",
    "post_send_seconds_before_off",
    "post_allow_after_scheduled_off_seconds",
    "post_trigger_window_hit",
    "post_write_attempted",
    "post_write_status",
    "post_write_reason",
    "post_bet_ref_required",
    "post_bet_ref_wait_attempted",
    "post_bet_ref_wait_ms",
    "post_bet_ref_poll_ms",
    "post_existing_bet_ref_before",
    "post_existing_pre_bet_ref",
    "post_existing_matched_before",
    "post_existing_pre_matched_stake",
    "post_existing_avg_odds_before",
    "post_existing_pre_avg_odds",
    "post_independent_mode_enabled",
    "post_row_prepared_for_new_order",
    "post_pre_bet_ref_cleared_for_write",
    "post_pre_bet_ref_preserved_in_state",
    "post_new_bet_ref_expected",
    "post_new_bet_ref_found",
    "post_new_bet_ref",
    "post_added_stake_confirmed",
    "post_added_stake_amount",
    "post_total_matched_before",
    "post_total_matched_after",
    "post_total_matched_delta",
    "post_expected_market_id",
    "post_expected_market_type",
    "post_expected_runner",
    "post_expected_selection_id",
    "post_expected_side",
    "post_expected_stake",
    "post_expected_price",
    "post_write_timestamp",
    "post_order_write_timestamp",
    "post_bet_ref_after",
    "post_bet_ref_changed",
    "post_bet_ref_confirmed_new",
    "post_bet_ref_poll_attempts",
    "post_bet_ref_poll_duration_ms",
    "post_order_confirmed",
    "post_order_confirmation_source",
    "post_confirmation_source",
    "post_selections_lookup_attempted",
    "post_selections_match_found",
    "post_selections_match_reason",
    "post_selections_reject_reason",
    "post_clear_after_bet_ref",
    "post_cells_clear_delay_ms",
    "post_cells_cleared_after_confirmation",
    "post_cells_cleared_after_unconfirmed",
    "post_clear_reason",
    "post_write_unconfirmed_reason",
    "post_unconfirmed_reason",
    "post_reject_reason",
    "countdown_seconds_at_post_write",
    "market_status_at_post_write",
    "hold_trigger_for_visual_test",
]


@dataclass(frozen=True)
class GrussRealOrderContext:
    validation_ok: bool
    tradable: bool
    region: str
    countdown_seconds: int | None
    course: str | None
    market_already_processed: bool = False
    win_market_id: str | None = None
    place_market_id: str | None = None
    milestone_seen: int | None = None
    pre_batch_milestone_authorized: bool = False
    pre_batch_milestone_seconds: int | None = None
    pre_batch_started_countdown_seconds: int | None = None
    pre_batch_write_grace_seconds: int | None = None


@dataclass(frozen=True)
class GrussTriggerLayout:
    """Standard Gruss triggered-betting columns for a sheet starting at A1."""

    trigger_column: str = "Q"
    odds_column: str = "R"
    stake_column: str = "S"
    bet_ref_column: str = "T"
    avg_matched_odds_column: str = "V"
    matched_stake_column: str = "W"
    profit_loss_column: str = "X"
    back_limit_trigger: str = "BACK"
    lay_limit_trigger: str = "LAY"
    back_replace_trigger: str = "BACKR"
    lay_replace_trigger: str = "LAYR"
    cancel_trigger: str = "CANCEL"
    back_sp_moc_trigger: str = "BACKSP"
    lay_sp_moc_trigger: str = "LAYSP"

    @classmethod
    def from_env(cls) -> GrussTriggerLayout:
        return cls(
            trigger_column=_column_env("DOGBOT_GRUSS_TRIGGER_COLUMN", "Q"),
            odds_column=_column_env("DOGBOT_GRUSS_ODDS_COLUMN", "R"),
            stake_column=_column_env("DOGBOT_GRUSS_STAKE_COLUMN", "S"),
            bet_ref_column=_column_env("DOGBOT_GRUSS_BET_REF_COLUMN", "T"),
            avg_matched_odds_column=_column_env("DOGBOT_GRUSS_AVG_MATCHED_ODDS_COLUMN", "V"),
            matched_stake_column=_column_env("DOGBOT_GRUSS_MATCHED_STAKE_COLUMN", "W"),
            profit_loss_column=_column_env("DOGBOT_GRUSS_PROFIT_LOSS_COLUMN", "X"),
            back_limit_trigger=os.getenv("DOGBOT_GRUSS_BACK_LIMIT_TRIGGER", "BACK").strip().upper(),
            lay_limit_trigger=os.getenv("DOGBOT_GRUSS_LAY_LIMIT_TRIGGER", "LAY").strip().upper(),
            back_replace_trigger=os.getenv("DOGBOT_GRUSS_BACK_REPLACE_TRIGGER", "BACKR").strip().upper(),
            lay_replace_trigger=os.getenv("DOGBOT_GRUSS_LAY_REPLACE_TRIGGER", "LAYR").strip().upper(),
            cancel_trigger=os.getenv("DOGBOT_GRUSS_CANCEL_TRIGGER", "CANCEL").strip().upper(),
            back_sp_moc_trigger=os.getenv("DOGBOT_GRUSS_BACK_SP_MOC_TRIGGER", "BACKSP").strip().upper(),
            lay_sp_moc_trigger=os.getenv("DOGBOT_GRUSS_LAY_SP_MOC_TRIGGER", "LAYSP").strip().upper(),
        )

    def trigger_mapping_name(self, side: str, order_type: str) -> str:
        key = (str(side).upper(), str(order_type).upper())
        mappings = {
            ("BACK", "LIMIT"): self.back_limit_trigger,
            ("LAY", "LIMIT"): self.lay_limit_trigger,
            ("BACK", "SP_MOC"): self.back_sp_moc_trigger,
            ("LAY", "SP_MOC"): self.lay_sp_moc_trigger,
        }
        return mappings[key]

    def trigger_address(self, row: int) -> str:
        return f"{self.trigger_column}{row}".upper()

    def odds_address(self, row: int) -> str:
        return f"{self.odds_column}{row}".upper()

    def stake_address(self, row: int) -> str:
        return f"{self.stake_column}{row}".upper()

    def bet_ref_address(self, row: int) -> str:
        return f"{self.bet_ref_column}{row}".upper()

    def avg_matched_odds_address(self, row: int) -> str:
        return f"{self.avg_matched_odds_column}{row}".upper()

    def matched_stake_address(self, row: int) -> str:
        return f"{self.matched_stake_column}{row}".upper()

    def profit_loss_address(self, row: int) -> str:
        return f"{self.profit_loss_column}{row}".upper()

    def replace_trigger_name(self, side: str) -> str:
        upper_side = str(side or "").strip().upper()
        if upper_side == "BACK":
            return self.back_replace_trigger
        if upper_side == "LAY":
            return self.lay_replace_trigger
        raise KeyError(upper_side)


@dataclass(frozen=True)
class GrussRealOrderResult:
    status: str
    reason: str
    output_path: Path
    excel_sheet: str = ""
    excel_row: int | None = None
    excel_cells_written: tuple[str, ...] = ()
    write_plan: tuple[tuple[str, Any], ...] = ()
    trigger_written: bool = False
    intended_trigger: str = ""
    stake_original: float | None = None
    stake_used: float | None = None
    stake_forced: bool = False
    stake_min_floor_applied: bool = False
    stake_before_min_floor: float | None = None
    stake_after_min_floor: float | None = None
    stake_final: float | None = None
    stake_capped: bool = False
    stake_cap_value: float | None = None
    execution_phase: str = "POST"
    market_type: str = ""
    market_id: str = ""
    runner_name: str = ""
    trap: int | None = None
    selection_id: str | int | None = None
    side: str = ""
    order_type: str = ""
    strategy_id: str = ""
    processed_key: str = ""
    post_processed_key: str = ""
    post_processed_key_scope: str = ""
    parent_id: str = ""
    course_id: str = ""
    win_market_id: str = ""
    place_market_id: str = ""
    processed_key_seen: bool = False
    processed_key_seen_matching_existing_key: str = ""
    pre_post_independent: bool = False
    pre_existing_order_allowed: bool = False
    pre_cancel_required_before_post: bool = False
    stake_limit_scope: str = ""
    pre_ladder: bool = False
    ladder_id: str = ""
    ladder_step: str = ""
    active_pre_ladder_id: str = ""
    continuing_active_pre_ladder: bool = False
    active_pre_ladder_count: int = 0
    max_active_pre_ladders: int = 1
    configured_ladder_steps: str = ""
    ladder_plan_frozen: bool = False
    ladder_plan_created_step: str | int | None = None
    ladder_prices_frozen: str = ""
    current_ladder_price_from_frozen_plan: bool = False
    computed_limit_price_raw: float | None = None
    computed_limit_price_effective: float | None = None
    min_price_floor_applied: bool = False
    pre_value_target_price: float | None = None
    ladder_planned_price: float | None = None
    sent_price_before_value_clamp: float | None = None
    sent_price_after_value_clamp: float | None = None
    value_clamp_applied: bool = False
    value_limit_breached: bool = False
    value_limit_skip_reason: str = ""
    tick_rounding_direction: str = ""
    best_same_side_offer_at_creation: float | None = None
    best_back_displayed: float | None = None
    best_lay_displayed: float | None = None
    start_price_source: str = ""
    final_lim_price: float | None = None
    ladder_direction: str = ""
    ladder_disabled_lim_not_in_ladder_direction: bool = False
    direct_lim_order_planned: bool = False
    direct_lim_order_written: bool = False
    no_replace_steps_for_direct_lim: bool = False
    current_milestone: int | None = None
    computed_step_index: int | None = None
    expected_ladder_step: str = ""
    milestone_seen: int | None = None
    next_ladder_step_due: str = ""
    skipped_step_reason: str = ""
    active_ladder_completed: bool = False
    active_ladder_release_reason: str = ""
    signal_timestamp: str = ""
    write_timestamp: str = ""
    write_delay_since_signal_seconds: float | None = None
    countdown_at_signal: int | None = None
    countdown_at_write: int | None = None
    pre_batch_milestone_authorized: bool = False
    pre_batch_milestone_seconds: int | None = None
    pre_batch_started_countdown_seconds: int | None = None
    pre_batch_write_grace_seconds: int | None = None
    pre_batch_candidate_index: int | None = None
    pre_batch_candidates_count: int | None = None
    pre_batch_late_write_allowed: bool = False
    pre_batch_late_write_seconds_after_start: int | None = None
    no_stacking_check_passed: bool = False
    market_reference_price_at_signal: float | None = None
    current_market_price_at_write: float | None = None
    stale_distance: float | None = None
    stale_price_limit: float | None = None
    stale_check_ignored_for_pre: bool = False
    conflict_detected: bool = False
    conflict_type: str = ""
    back_price: float | None = None
    lay_price: float | None = None
    market_reference_price: float | None = None
    back_distance: float | None = None
    lay_distance: float | None = None
    selected_side: str = ""
    rejected_side: str = ""
    conflict_group_key: str = ""
    conflict_candidates_count: int = 0
    winning_side: str = ""
    losing_side: str = ""
    winning_strategy_id: str = ""
    losing_strategy_id: str = ""
    winning_edge: float | None = None
    losing_edge: float | None = None
    winning_score: float | None = None
    losing_score: float | None = None
    winning_lim_price: float | None = None
    losing_lim_price: float | None = None
    back_systems: str = ""
    lay_systems: str = ""
    conflict_resolution_reason: str = ""
    pre_back_lay_conflict: bool = False
    pre_conflict_resolution: str = ""
    pre_conflict_chosen_side: str = ""
    pre_conflict_rejected_side: str = ""
    pre_conflict_reason: str = ""
    pre_conflict_group_key: str = ""
    pre_conflict_course_id: str = ""
    pre_conflict_market_id: str = ""
    pre_conflict_market_type: str = ""
    pre_conflict_selection_id: str = ""
    pre_conflict_runner_name: str = ""
    pre_back_target_price: float | None = None
    pre_lay_target_price: float | None = None
    pre_current_best_lay: float | None = None
    pre_current_best_back: float | None = None
    pre_back_distance_ticks: float | None = None
    pre_lay_distance_ticks: float | None = None
    trigger_cell_address: str = ""
    trigger_cell_current_value: Any = None
    trigger_cell_expected_empty: bool | None = None
    trigger_mapping_name: str = ""
    trigger_value_written: str = ""
    action: str = ""
    bet_ref_before: str = ""
    bet_ref_after: str = ""
    bet_ref_poll_attempts: int = 0
    bet_ref_poll_duration_ms: int = 0
    pre_write_attempt_id: str = ""
    pre_bet_ref_required: bool = False
    pre_bet_ref_confirmed: bool = False
    pre_bet_ref_found: bool = False
    pre_bet_ref_missing: bool = False
    pre_bet_ref_poll_attempts: int = 0
    pre_bet_ref_poll_duration_ms: int = 0
    pre_bet_ref_missing_retryable: bool = False
    pre_bet_ref_late_detected: bool = False
    pre_bet_ref_late_value: str = ""
    pre_retry_count: int = 0
    pre_retry_allowed: bool = False
    pre_retry_reason: str = ""
    pre_retry_block_reason: str = ""
    pre_unconfirmed_reason: str = ""
    bet_ref_lookup_sources: str = ""
    bet_ref_lookup_source_used: str = ""
    bet_ref_lookup_source: str = ""
    bet_ref_lookup_matched_runner: str = ""
    row_t_value: str = ""
    selections_rows_scanned: int = 0
    selections_match_found: bool = False
    selections_match_reason: str = ""
    selections_runner: str = ""
    selections_side: str = ""
    selections_stake: float | None = None
    selections_bet_ref: str = ""
    selections_req_odds: float | None = None
    selections_market_name: str = ""
    selections_debug_recent_rows: str = ""
    selections_top_candidates: str = ""
    bet_ref_row_t_dump: str = ""
    bet_ref_diagnostic_hold_after_batch: bool = False
    selections_market_query: str = ""
    selections_current_market_rows: str = ""
    selections_current_runner_rows: str = ""
    runner_qz_dump: str = ""
    selections_sheet_headers: str = ""
    selections_full_recent_rows: str = ""
    workbook_sheet_names: str = ""
    diagnostic_keep_triggers: bool = False
    active_ladder_bet_ref_stored: bool = False
    active_ladder_created: bool = False
    pending_ladder_created: bool = False
    matched_evidence_found: bool = False
    selection_row_evidence_found: bool = False
    no_stacking_blocked_retry: bool = False
    replace_allowed: bool = False
    replace_trigger: str = ""
    bet_ref_suffix_n_handled: bool = False
    bet_ref_status_value: str = ""
    replace_bet_ref_wait_attempted: bool = False
    replace_bet_ref_wait_ms: int = 0
    replace_bet_ref_poll_ms: int = 0
    replace_bet_ref_wait_result: str = ""
    bet_ref_before_wait: str = ""
    bet_ref_after_wait: str = ""
    active_ladder_bet_ref_updated: bool = False
    replace_skipped_bet_ref_still_pending: bool = False
    pre_ladder_initial_order_failed: bool = False
    pre_ladder_disabled_after_initial_failure: bool = False
    no_replace_steps_for_failed_initial: bool = False
    requested_price: float | None = None
    requested_stake: float | None = None
    ladder_step_index: int | None = None
    ladder_step_count: int | None = None
    matched_after_step: bool = False
    matched_after_step_avg_odds: float | None = None
    matched_after_step_stake: float | None = None
    avg_matched_odds_cell_address: str = ""
    avg_matched_odds_cell_value: Any = None
    matched_stake_cell_address: str = ""
    matched_stake_cell_value: Any = None
    profit_loss_cell_address: str = ""
    profit_loss_cell_value: Any = None
    batch_size: int = 0
    batch_write_start_timestamp: str = ""
    batch_write_end_timestamp: str = ""
    batch_write_duration_ms: int = 0
    order_index_in_batch: int = 0
    bet_ref_collection_phase_start: str = ""
    bet_ref_collection_phase_end: str = ""
    bet_ref_collection_duration_ms: int = 0
    bet_ref_found_count: int = 0
    bet_ref_missing_count: int = 0
    runner_row: int | None = None
    runner_order_in_sheet: int | None = None
    mapping_found: bool = False
    mapping_reason: str = ""
    command_cells: str = ""
    total_runners_in_gruss_sheet: int = 0
    raw_gruss_runner_rows: str = ""
    raw_selection_ids_seen: str = ""
    raw_runner_names_seen: str = ""
    mapped_runners_count: int = 0
    unmapped_runners_count: int = 0
    mapped_selection_ids: str = ""
    unmapped_selection_ids: str = ""
    ignored_runner_rows: str = ""
    ignored_runner_reason: str = ""
    mapped_excel_rows: str = ""
    excel_write_attempt: int = 0
    excel_write_retry_count: int = 0
    excel_write_retry_backoff_ms: str = ""
    excel_write_final_status: str = ""
    excel_unavailable_recovered: bool = False
    excel_operation_name: str = ""
    excel_com_attempt: int = 0
    excel_com_retry_count: int = 0
    excel_com_retry_backoff_ms: str = ""
    excel_com_retryable_error: bool = False
    mapping_attempt_count: int = 0
    cleanup_retry_count: int = 0
    cleanup_final_status: str = ""
    update_allowed: bool = False
    update_skipped_reason: str = ""
    matched_stake: float | None = None
    pre_cancel_attempted: bool = False
    pre_cancel_written: bool = False
    pre_cancel_skip_reason: str = ""
    pre_cancel_only_if_post_pending: bool = False
    post_pending_for_runner: bool = False
    post_after_pre_cancel_attempted: bool = False
    bet_ref_at_cancel: str = ""
    matched_stake_at_cancel: float | None = None
    countdown_seconds_at_cancel: int | None = None
    trigger_clear_attempted: bool = False
    trigger_cleared: bool = False
    trigger_clear_reason: str = ""
    trigger_cell_value_before_clear: Any = None
    trigger_clear_delay_ms: int = 0
    command_cells_clear_attempted: bool = False
    command_cells_cleared: bool = False
    command_cells_clear_reason: str = ""
    command_cells_clear_addresses: str = ""
    command_cells_clear_delay_ms: int = 0
    command_cells_clear_scheduled: bool = False
    command_cells_clear_due_time: str = ""
    command_cells_clear_non_blocking: bool = False
    command_cells_clear_executed: bool = False
    command_cells_clear_lag_ms: int | None = None
    startup_command_cells_cleanup_attempted: bool = False
    startup_command_cells_cleanup_done: bool = False
    stale_command_cells_cleanup_attempted: bool = False
    stale_command_cells_cleanup_addresses: str = ""
    stale_command_cells_cleanup_reason: str = ""
    post_write_odds_cell_address: str = ""
    post_write_odds_value: Any = None
    post_write_stake_cell_address: str = ""
    post_write_stake_value: Any = None
    post_write_trigger_cell_address: str = ""
    post_write_trigger_value: Any = None
    post_write_verified: bool | None = None
    post_provider_called: bool | None = None
    post_batch_id: str = ""
    post_batch_market_id: str = ""
    post_batch_market_name: str = ""
    post_batch_candidate_count: int | None = None
    post_batch_written_count: int | None = None
    post_batch_write_duration_ms: int | None = None
    post_batch_confirmation_started: bool = False
    post_batch_confirmation_duration_ms: int | None = None
    post_batch_runner_index: int | None = None
    post_batch_total_runners: int | None = None
    post_send_seconds_before_off: int | None = None
    post_allow_after_scheduled_off_seconds: int | None = None
    post_trigger_window_hit: bool | None = None
    post_write_attempted: bool | None = None
    post_write_status: str = ""
    post_write_reason: str = ""
    post_bet_ref_required: bool = False
    post_bet_ref_wait_attempted: bool = False
    post_bet_ref_wait_ms: int = 0
    post_bet_ref_poll_ms: int = 0
    post_existing_bet_ref_before: str = ""
    post_existing_pre_bet_ref: str = ""
    post_existing_matched_before: float | None = None
    post_existing_pre_matched_stake: float | None = None
    post_existing_avg_odds_before: float | None = None
    post_existing_pre_avg_odds: float | None = None
    post_independent_mode_enabled: bool = False
    post_row_prepared_for_new_order: bool = False
    post_pre_bet_ref_cleared_for_write: bool = False
    post_pre_bet_ref_preserved_in_state: bool = False
    post_new_bet_ref_expected: bool = False
    post_new_bet_ref_found: bool = False
    post_new_bet_ref: str = ""
    post_added_stake_confirmed: bool = False
    post_added_stake_amount: float | None = None
    post_total_matched_before: float | None = None
    post_total_matched_after: float | None = None
    post_total_matched_delta: float | None = None
    post_expected_market_id: str = ""
    post_expected_market_type: str = ""
    post_expected_runner: str = ""
    post_expected_selection_id: str = ""
    post_expected_side: str = ""
    post_expected_stake: float | None = None
    post_expected_price: float | None = None
    post_write_timestamp: str = ""
    post_order_write_timestamp: str = ""
    post_bet_ref_after: str = ""
    post_bet_ref_changed: bool = False
    post_bet_ref_confirmed_new: bool = False
    post_bet_ref_poll_attempts: int = 0
    post_bet_ref_poll_duration_ms: int = 0
    post_order_confirmed: bool = False
    post_order_confirmation_source: str = ""
    post_confirmation_source: str = ""
    post_selections_lookup_attempted: bool = False
    post_selections_match_found: bool = False
    post_selections_match_reason: str = ""
    post_selections_reject_reason: str = ""
    post_clear_after_bet_ref: bool = False
    post_cells_clear_delay_ms: int = 0
    post_cells_cleared_after_confirmation: bool = False
    post_cells_cleared_after_unconfirmed: bool = False
    post_clear_reason: str = ""
    post_write_unconfirmed_reason: str = ""
    post_unconfirmed_reason: str = ""
    post_reject_reason: str = ""
    countdown_seconds_at_post_write: int | None = None
    market_status_at_post_write: Any = None
    hold_trigger_for_visual_test: bool = False
    price_raw_before_tick: float | None = None
    price_tick_rounded: float | None = None
    price_tick_rounding_side: str = ""
    price_is_valid_betfair_tick: bool | None = None


@dataclass(frozen=True)
class _TriggerClearOutcome:
    attempted: bool = False
    cleared: bool = False
    reason: str = ""
    command_reason: str = ""
    value_before_clear: Any = None
    delay_ms: int = 0
    addresses: tuple[str, ...] = ()
    scheduled: bool = False
    due_time: str = ""
    non_blocking: bool = False
    executed: bool = False
    lag_ms: int | None = None


@dataclass
class _PendingCommandCellsClear:
    processed_key: str
    sheet_name: str
    trigger_address: str
    trigger_value_written: str
    addresses: tuple[str, ...]
    expected_values: tuple[tuple[str, Any], ...]
    due_monotonic: float
    due_time: str
    delay_ms: int


@dataclass(frozen=True)
class _BetRefPollResult:
    bet_ref: str = ""
    attempts: int = 0
    duration_ms: int = 0
    lookup_source: str = "excel_row_poll"


@dataclass(frozen=True)
class _PostSelectionsLookupResult:
    match: _SelectionsBetRefCandidate | None = None
    attempted: bool = False
    rows_scanned: int = 0
    match_reason: str = ""
    reject_reason: str = ""
    lookup_source: str = ""


@dataclass(frozen=True)
class _ExcelWriteResult:
    written: tuple[str, ...] = ()
    attempt: int = 0
    retry_count: int = 0
    retry_backoff_ms: str = ""
    final_status: str = ""
    recovered: bool = False
    exception: Exception | None = None


@dataclass(frozen=True)
class _ExcelComResult:
    operation_name: str = ""
    value: Any = None
    attempt: int = 0
    retry_count: int = 0
    retry_backoff_ms: str = ""
    retryable_error: bool = False
    final_status: str = ""
    recovered: bool = False
    exception: Exception | None = None


@dataclass(frozen=True)
class _ReplaceBetRefWaitResult:
    bet_ref: str = ""
    before_wait: str = ""
    after_wait: str = ""
    attempts: int = 0
    duration_ms: int = 0
    wait_ms: int = 0
    poll_ms: int = 0
    result: str = "timeout"


@dataclass(frozen=True)
class _SelectionsBetRefCandidate:
    row_number: int
    runner: str
    trap: int | None
    selection_id: str
    side: str
    stake: float | None
    req_odds: float | None
    average_odds: float | None
    result: str
    market_name: str
    market_id: str
    market_type: str
    matched_odds: float | None
    matched_stake: float | None
    bet_ref: str
    timestamp: str = ""


@dataclass(frozen=True)
class _PostWriteVerification:
    odds_cell_address: str
    odds_value: Any
    stake_cell_address: str
    stake_value: Any
    trigger_cell_address: str
    trigger_value: Any
    verified: bool


@dataclass
class _ActivePreLadderState:
    course_key: str
    market_type: str
    market_id: str
    selection_id: str
    runner_name: str
    trap: int | None
    side: str
    row: int
    bet_ref: str = ""
    pending_confirmation: bool = False


class GrussExcelOrderProvider:
    """Ultra-guarded Gruss Excel order provider.

    The provider refuses all attempts unless real orders are explicitly armed.
    Preview mode is enabled by default and never calls the bridge write method.
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        *,
        bridge: GrussExcelBridge | None = None,
        layout: GrussTriggerLayout | None = None,
        processed_markets: set[str] | None = None,
        real_order_counts: dict[str, int] | None = None,
        preview_only_guard: bool = False,
        write_no_trigger_guard: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_path = self.data_dir / "gruss_real_order_attempts.csv"
        self.bridge = bridge or GrussExcelBridge(DEFAULT_WORKBOOK_PATH)
        self.preview_only_guard = bool(preview_only_guard)
        self.write_no_trigger_guard = bool(write_no_trigger_guard)
        self.order_provider = os.getenv("DOGBOT_ORDER_PROVIDER", "").strip().lower()
        self.enabled = _env_bool("DOGBOT_GRUSS_ENABLE_REAL_ORDERS", False)
        self.preview = _env_bool("DOGBOT_GRUSS_REAL_PREVIEW", False)
        self.layout_confirmed = _env_bool("DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED", False)
        self.layout = layout or GrussTriggerLayout.from_env()
        self.processed_markets = processed_markets if processed_markets is not None else set()
        self.real_order_counts = real_order_counts if real_order_counts is not None else {}
        self.active_pre_ladder_id: str | None = None
        self.active_pre_ladder_course: str | None = None
        self.active_pre_ladders: dict[str, _ActivePreLadderState] = {}
        self.pre_bet_ref_missing_retry_counts: dict[str, int] = {}
        self._pending_active_ladder_release_id: str | None = None
        self._pending_active_ladder_release_reason: str = ""
        self._batch_log_context: dict[str, Any] = {}
        self.write_no_trigger = _env_bool("DOGBOT_GRUSS_WRITE_NO_TRIGGER", False)
        self.real_test_mode = _env_bool("DOGBOT_GRUSS_REAL_TEST_MODE", False)
        self.real_max_orders = _real_max_orders(self.real_test_mode)
        self.real_max_orders_by_phase = _real_max_orders_by_phase(self.real_test_mode, self.real_max_orders)
        self.real_max_stake = _real_max_stake(self.real_test_mode)
        self.trigger_clear_delay_ms = _trigger_clear_delay_ms()
        self.command_cells_clear_enabled = _env_bool("DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_AFTER_WRITE", True)
        self.command_cells_clear_delay_ms = _command_cells_clear_delay_ms()
        self.command_cells_clear_non_blocking = _env_bool(
            "DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_NON_BLOCKING",
            True,
        )
        self.command_cells_clear_columns = _command_cells_clear_columns_from_env()
        self._pending_command_cell_clears: list[_PendingCommandCellsClear] = []
        self._last_excel_com_result = _ExcelComResult()
        self._last_mapping_com_result = _ExcelComResult(operation_name="mapping")
        self._last_cleanup_com_result = _ExcelComResult(operation_name="cleanup")
        self.hold_trigger_for_visual_test = _env_bool(
            "DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST",
            False,
        )

    def set_batch_log_context(self, **fields: Any) -> None:
        self._batch_log_context = {
            **dict(fields),
            "bet_ref_lookup_sources": ",".join(_bet_ref_lookup_sources_from_env()),
        }

    def clear_batch_log_context(self) -> None:
        self._batch_log_context = {}

    def cleanup_stale_command_cells(
        self,
        *,
        reason: str = "startup",
        sheets: Iterable[str] = ("WIN", "PLACE"),
    ) -> dict[str, Any]:
        self._refresh_safety_flags()
        cleaned: list[str] = []
        skipped_pending: list[str] = []
        stale_triggers = {"BACK", "LAY", "BACKR", "LAYR", "BACKSP", "LAYSP", "CANCEL"}
        reason_text = str(reason or "").strip().lower()
        force_attempt_log = reason_text in {"startup", "shutdown"} or reason_text.startswith(
            ("course_change", "market_change", "periodic")
        )
        pending_trigger_keys = {
            (pending.sheet_name.upper(), pending.trigger_address.upper())
            for pending in self._pending_command_cell_clears
        }
        sheet_names = tuple(
            sheet_name
            for sheet_name in (str(raw or "").strip().upper() for raw in sheets)
            if sheet_name in {"WIN", "PLACE"}
        )

        def scan_stale_triggers() -> tuple[list[tuple[str, str, tuple[str, ...]]], list[str]]:
            self.bridge.connect_open_workbook()
            stale: list[tuple[str, str, tuple[str, ...]]] = []
            skipped: list[str] = []
            for sheet_name in sheet_names:
                row_values = self._read_runner_row_values(sheet_name)
                for row, _runner in row_values:
                    trigger_address = self.layout.trigger_address(row)
                    if (sheet_name, trigger_address.upper()) in pending_trigger_keys:
                        skipped.append(f"{sheet_name}!{trigger_address}")
                        continue
                    trigger_value = _clean_text(self.bridge.read_cell(sheet_name, trigger_address)).upper()
                    if trigger_value not in stale_triggers:
                        continue
                    stale.append(
                        (
                            sheet_name,
                            trigger_address,
                            self._command_cell_addresses_for_trigger(trigger_address),
                        )
                    )
            return stale, skipped

        scan_result = self._excel_com_retry("cleanup_stale_scan_command_cells", scan_stale_triggers)
        self._last_cleanup_com_result = scan_result
        stale_items: list[tuple[str, str, tuple[str, ...]]] = []
        scan_attempt_count = scan_result.attempt
        scan_retry_count = scan_result.retry_count
        scan_recovered = scan_result.recovered
        blind_clear_result: _ExcelComResult | None = None
        if scan_result.exception is not None:
            blind_clear_result = self._blind_clear_command_cells(sheet_names)
            rescan_result = self._excel_com_retry(
                "cleanup_stale_rescan_after_blind_clear",
                scan_stale_triggers,
            )
            self._last_cleanup_com_result = rescan_result
            scan_attempt_count += rescan_result.attempt
            scan_retry_count += rescan_result.retry_count
            scan_recovered = bool(scan_recovered or rescan_result.recovered)
            if rescan_result.exception is not None:
                exc = rescan_result.exception
                cleanup_reason = f"unsafe_stale_scan_unavailable_after_retries:{exc}"
                self._append_command_cells_cleanup_attempt(
                    reason=reason,
                    attempted=True,
                    done=False,
                    addresses=(),
                    cleanup_reason=cleanup_reason,
                    stale_scan_attempt_count=scan_attempt_count,
                    stale_scan_retry_count=scan_retry_count,
                    stale_scan_recovered=scan_recovered,
                    stale_triggers_confirmed=False,
                    stale_cleanup_retry_count=(
                        blind_clear_result.retry_count if blind_clear_result is not None else 0
                    ),
                    stale_cleanup_recovered=(
                        blind_clear_result.recovered if blind_clear_result is not None else False
                    ),
                    stale_cleanup_final_status=(
                        blind_clear_result.final_status if blind_clear_result is not None else ""
                    ),
                    unsafe_stop_reason="unsafe_stale_scan_unavailable_after_retries",
                )
                return {
                    "attempted": True,
                    "done": False,
                    "addresses": "",
                    "reason": cleanup_reason,
                    "failed": True,
                    "unsafe_stop_reason": "unsafe_stale_scan_unavailable_after_retries",
                    "stale_triggers_confirmed": False,
                }
            stale_items, skipped_pending = rescan_result.value
        else:
            stale_items, skipped_pending = scan_result.value

        stale_triggers_confirmed = bool(stale_items)
        cleanup_retry_count = blind_clear_result.retry_count if blind_clear_result is not None else 0
        cleanup_recovered = blind_clear_result.recovered if blind_clear_result is not None else False
        cleanup_final_status = blind_clear_result.final_status if blind_clear_result is not None else ""
        for sheet_name, trigger_address, addresses in stale_items:
            clear_result = self._excel_com_retry(
                f"cleanup_stale_clear_command_cells:{sheet_name}!{';'.join(addresses)}",
                lambda sheet_name=sheet_name, addresses=addresses: self.bridge.clear_trigger_cells(
                    sheet_name,
                    addresses,
                    trigger_column=self.layout.trigger_column,
                    command_columns=self.command_cells_clear_columns,
                    allow_clear=True,
                ),
            )
            self._last_cleanup_com_result = clear_result
            cleanup_retry_count += clear_result.retry_count
            cleanup_recovered = bool(cleanup_recovered or clear_result.recovered)
            cleanup_final_status = clear_result.final_status
            if clear_result.exception is not None:
                exc = clear_result.exception
                attempted_addresses = [*cleaned, *(f"{sheet_name}!{address}" for address in addresses)]
                cleanup_reason = (
                    "unsafe_confirmed_stale_gruss_triggers_cleanup_failed:"
                    f"{sheet_name}!{trigger_address}:{exc}"
                )
                self._append_command_cells_cleanup_attempt(
                    reason=reason,
                    attempted=True,
                    done=False,
                    addresses=tuple(attempted_addresses),
                    cleanup_reason=cleanup_reason,
                    stale_scan_attempt_count=scan_attempt_count,
                    stale_scan_retry_count=scan_retry_count,
                    stale_scan_recovered=scan_recovered,
                    stale_triggers_confirmed=True,
                    stale_cleanup_retry_count=cleanup_retry_count,
                    stale_cleanup_recovered=cleanup_recovered,
                    stale_cleanup_final_status=cleanup_final_status,
                    unsafe_stop_reason="unsafe_confirmed_stale_gruss_triggers_cleanup_failed",
                )
                return {
                    "attempted": True,
                    "done": False,
                    "addresses": ";".join(attempted_addresses),
                    "reason": cleanup_reason,
                    "failed": True,
                    "unsafe_stop_reason": "unsafe_confirmed_stale_gruss_triggers_cleanup_failed",
                    "stale_triggers_confirmed": True,
                }
            cleared = _normalize_command_cell_addresses(clear_result.value)
            if not cleared:
                cleared = addresses
            cleaned.extend(f"{sheet_name}!{address}" for address in cleared)
        cleanup_reason = "stale_command_cells_cleaned" if cleaned else "no_stale_command_cells"
        if blind_clear_result is not None:
            cleanup_reason += f";blind_clear={blind_clear_result.final_status or 'unknown'}"
        if skipped_pending:
            cleanup_reason += f";skipped_pending={','.join(skipped_pending)}"
        done = True
        attempted = bool(stale_items or force_attempt_log or blind_clear_result is not None)
        self._append_command_cells_cleanup_attempt(
            reason=reason,
            attempted=attempted,
            done=done,
            addresses=tuple(cleaned),
            cleanup_reason=cleanup_reason,
            stale_scan_attempt_count=scan_attempt_count,
            stale_scan_retry_count=scan_retry_count,
            stale_scan_recovered=scan_recovered,
            stale_triggers_confirmed=stale_triggers_confirmed,
            stale_cleanup_retry_count=cleanup_retry_count,
            stale_cleanup_recovered=cleanup_recovered,
            stale_cleanup_final_status=cleanup_final_status,
            unsafe_stop_reason="",
        )
        return {
            "attempted": attempted,
            "done": done,
            "addresses": ";".join(cleaned),
            "reason": cleanup_reason,
            "failed": False,
            "unsafe_stop_reason": "",
            "stale_triggers_confirmed": stale_triggers_confirmed,
            "stale_scan_recovered": scan_recovered,
        }

    def _blind_clear_command_cells(self, sheet_names: Iterable[str]) -> _ExcelComResult:
        def clear_all() -> tuple[str, ...]:
            cleared: list[str] = []
            for sheet_name in sheet_names:
                addresses: list[str] = []
                for row in range(5, 85):
                    addresses.extend(self._command_cell_addresses_for_trigger(self.layout.trigger_address(row)))
                cleared_addresses = self.bridge.clear_trigger_cells(
                    sheet_name,
                    addresses,
                    trigger_column=self.layout.trigger_column,
                    command_columns=self.command_cells_clear_columns,
                    allow_clear=True,
                )
                cleared.extend(
                    f"{sheet_name}!{address}" for address in _normalize_command_cell_addresses(cleared_addresses)
                )
            return tuple(cleared)

        result = self._excel_com_retry("cleanup_stale_blind_clear_command_cells", clear_all)
        self._last_cleanup_com_result = result
        return result

    def runner_mapping_summary(self, intents: Iterable[OrderIntent]) -> dict[str, Any]:
        intents_by_sheet: dict[str, list[OrderIntent]] = {}
        for intent in intents:
            sheet_name = str(intent.market_type or "").strip().upper()
            if sheet_name in {"WIN", "PLACE"}:
                intents_by_sheet.setdefault(sheet_name, []).append(intent)
        summaries: list[dict[str, Any]] = []
        for sheet_name, sheet_intents in sorted(intents_by_sheet.items()):
            summaries.append(self._runner_mapping_summary_for_sheet(sheet_name, sheet_intents))
        total_runners = sum(int(summary.get("total_runners_in_gruss_sheet") or 0) for summary in summaries)
        mapped_ids = _join_unique(
            part
            for summary in summaries
            for part in str(summary.get("mapped_selection_ids") or "").split("|")
            if part
        )
        unmapped_ids = _join_unique(
            part
            for summary in summaries
            for part in str(summary.get("unmapped_selection_ids") or "").split("|")
            if part
        )
        mapped_rows = _join_unique(
            part
            for summary in summaries
            for part in str(summary.get("mapped_excel_rows") or "").split("|")
            if part
        )
        raw_rows = _join_unique(
            part
            for summary in summaries
            for part in str(summary.get("raw_gruss_runner_rows") or "").split("|")
            if part
        )
        raw_selection_ids = _join_unique(
            part
            for summary in summaries
            for part in str(summary.get("raw_selection_ids_seen") or "").split("|")
            if part
        )
        raw_runner_names = _join_unique(
            part
            for summary in summaries
            for part in str(summary.get("raw_runner_names_seen") or "").split("|")
            if part
        )
        ignored_rows = _join_unique(
            part
            for summary in summaries
            for part in str(summary.get("ignored_runner_rows") or "").split("|")
            if part
        )
        ignored_reasons = _join_unique(
            part
            for summary in summaries
            for part in str(summary.get("ignored_runner_reason") or "").split("|")
            if part
        )
        return {
            "total_runners_in_gruss_sheet": total_runners,
            "raw_gruss_runner_rows": raw_rows,
            "raw_selection_ids_seen": raw_selection_ids,
            "raw_runner_names_seen": raw_runner_names,
            "mapped_runners_count": len([part for part in mapped_ids.split("|") if part]),
            "unmapped_runners_count": len([part for part in unmapped_ids.split("|") if part]),
            "mapped_selection_ids": mapped_ids,
            "unmapped_selection_ids": unmapped_ids,
            "ignored_runner_rows": ignored_rows,
            "ignored_runner_reason": ignored_reasons,
            "mapped_excel_rows": mapped_rows,
        }

    def _runner_mapping_summary_for_sheet(
        self,
        sheet_name: str,
        intents: Iterable[OrderIntent],
    ) -> dict[str, Any]:
        try:
            row_values = self._read_runner_row_values(sheet_name)
        except Exception:
            row_values = []
        mapped_ids: list[str] = []
        unmapped_ids: list[str] = []
        mapped_rows: list[str] = []
        raw_rows: list[str] = []
        raw_selection_ids: list[str] = []
        raw_runner_names: list[str] = []
        for intent in intents:
            row = _find_runner_row_in_values(row_values, intent)
            selection_id = _selection_id_for_log(intent)
            if row is None:
                unmapped_ids.append(selection_id)
            else:
                mapped_ids.append(selection_id)
                mapped_rows.append(str(row))
        mapped_row_set = set(mapped_rows)
        for row_number, value in row_values:
            trap = extract_trap(value)
            raw_rows.append(f"{sheet_name}!A{row_number}={_clean_text(value)}")
            raw_selection_ids.append(str(trap) if trap is not None else f"row{row_number}")
            raw_runner_names.append(_strip_runner_trap_for_log(value))
        ignored_rows = [
            f"{sheet_name}!A{row_number}"
            for row_number, _value in row_values
            if str(row_number) not in mapped_row_set
        ]
        return {
            "total_runners_in_gruss_sheet": len(row_values),
            "raw_gruss_runner_rows": _join_unique(raw_rows),
            "raw_selection_ids_seen": _join_unique(raw_selection_ids),
            "raw_runner_names_seen": _join_unique(raw_runner_names),
            "mapped_runners_count": len(mapped_ids),
            "unmapped_runners_count": len(unmapped_ids),
            "mapped_selection_ids": _join_unique(mapped_ids),
            "unmapped_selection_ids": _join_unique(unmapped_ids),
            "ignored_runner_rows": _join_unique(ignored_rows),
            "ignored_runner_reason": "no_signal_or_no_matching_intent" if ignored_rows else "",
            "mapped_excel_rows": _join_unique(mapped_rows),
        }

    def cancel_pre_ladders_before_post(
        self,
        context: GrussRealOrderContext,
        post_intents: Iterable[OrderIntent] | None = None,
    ) -> list[GrussRealOrderResult]:
        self._refresh_safety_flags()
        self._sync_legacy_active_ladder_state()
        if not _pre_cancel_before_post_enabled():
            return []
        milestone = _current_milestone(context)
        if milestone != _pre_cancel_seconds_before_off():
            return []
        if not self.active_pre_ladders:
            return [
                self._finish(
                    _empty_cancel_intent(context),
                    context,
                    "GRUSS_PRE_CANCEL_BEFORE_POST_SKIPPED",
                    "pre_cancel_skipped_no_active_pre_ladders",
                    update_skipped_reason="pre_cancel_skipped_no_active_pre_ladders",
                )
            ]
        results: list[GrussRealOrderResult] = []
        post_intents_provided = post_intents is not None
        post_keys = _post_pending_ladder_keys(post_intents or ())
        only_if_post_pending = bool(post_intents_provided and _pre_cancel_only_if_post_pending())
        try:
            self.bridge.connect_open_workbook()
        except Exception as exc:
            for ladder_id, state in list(self.active_pre_ladders.items()):
                results.append(
                    self._finish(
                        _cancel_intent_from_state(ladder_id, state, context),
                        context,
                        "GRUSS_PRE_CANCEL_BEFORE_POST_SKIPPED",
                        f"pre_cancel_skipped_excel_unavailable: {exc}",
                    )
                )
            return results
        for ladder_id, state in list(self.active_pre_ladders.items()):
            if only_if_post_pending and _active_ladder_post_key(state) not in post_keys:
                results.append(
                    self._finish(
                        _cancel_intent_from_state(ladder_id, state, context),
                        context,
                        "GRUSS_PRE_CANCEL_BEFORE_POST_SKIPPED",
                        "no_post_pending",
                        excel_sheet=state.market_type or "PLACE",
                        excel_row=state.row if state.row > 0 else None,
                        trigger_cell_address=self.layout.trigger_address(state.row) if state.row > 0 else "",
                        trigger_mapping_name=self.layout.cancel_trigger,
                        pre_cancel_attempted=False,
                        pre_cancel_written=False,
                        pre_cancel_skip_reason="no_post_pending",
                        pre_cancel_only_if_post_pending=True,
                        post_pending_for_runner=False,
                        post_after_pre_cancel_attempted=False,
                    )
                )
                continue
            results.append(
                self._cancel_one_pre_ladder_before_post(
                    ladder_id,
                    state,
                    context,
                    pre_cancel_only_if_post_pending=only_if_post_pending,
                    post_pending_for_runner=_active_ladder_post_key(state) in post_keys,
                )
            )
        self._refresh_legacy_active_ladder_snapshot()
        return results

    def has_matched_active_pre_ladder(self, context: GrussRealOrderContext) -> bool:
        self._sync_legacy_active_ladder_state()
        if not self.active_pre_ladders:
            return False
        for state in self.active_pre_ladders.values():
            if context.course and state.course_key not in {"", context.course}:
                continue
            try:
                matched_stake = _matched_stake_value(
                    self.bridge.read_cell(state.market_type or "PLACE", self.layout.matched_stake_address(state.row))
                )
            except Exception:
                matched_stake = None
            if matched_stake is not None and matched_stake > 0:
                return True
        return False

    def _cancel_one_pre_ladder_before_post(
        self,
        ladder_id: str,
        state: _ActivePreLadderState,
        context: GrussRealOrderContext,
        *,
        pre_cancel_only_if_post_pending: bool = False,
        post_pending_for_runner: bool = True,
    ) -> GrussRealOrderResult:
        intent = _cancel_intent_from_state(ladder_id, state, context)
        sheet_name = state.market_type or intent.market_type or "PLACE"
        row = state.row
        trigger_address = self.layout.trigger_address(row)
        bet_ref_address = self.layout.bet_ref_address(row)
        matched_stake_address = self.layout.matched_stake_address(row)
        cancel_trigger = self.layout.cancel_trigger

        try:
            market_status = self._read_cell_with_retry(sheet_name, "F2", f"read_market_status:{sheet_name}!F2")
        except Exception:
            market_status = ""
        if _is_untradable_market_status(market_status):
            return self._finish(
                intent,
                context,
                "GRUSS_PRE_CANCEL_BEFORE_POST_SKIPPED",
                "pre_cancel_skipped_market_suspended",
                excel_sheet=sheet_name,
                excel_row=row,
                write_plan=(),
                trigger_cell_address=trigger_address,
                trigger_mapping_name=cancel_trigger,
            )

        try:
            bet_ref_before = normalise_gruss_bet_ref(
                self._read_cell_with_retry(sheet_name, bet_ref_address, f"read_bet_ref:{sheet_name}!{bet_ref_address}")
            )
        except Exception:
            bet_ref_before = ""
        if not is_valid_bet_ref(bet_ref_before):
            return self._finish(
                intent,
                context,
                "GRUSS_PRE_CANCEL_BEFORE_POST_SKIPPED",
                "pre_cancel_before_post_skipped_invalid_bet_ref",
                excel_sheet=sheet_name,
                excel_row=row,
                write_plan=(),
                trigger_cell_address=trigger_address,
                trigger_mapping_name=cancel_trigger,
                bet_ref_before=bet_ref_before,
                bet_ref_status_value=bet_ref_before,
            )

        try:
            matched_stake_cell_value = self._read_cell_with_retry(
                sheet_name,
                matched_stake_address,
                f"read_matched_stake:{sheet_name}!{matched_stake_address}",
            )
        except Exception:
            matched_stake_cell_value = None
        matched_stake_value = _matched_stake_value(matched_stake_cell_value)
        if matched_stake_value is None:
            return self._finish(
                intent,
                context,
                "GRUSS_PRE_CANCEL_BEFORE_POST_SKIPPED",
                "pre_cancel_before_post_skipped_matched_stake_unavailable",
                excel_sheet=sheet_name,
                excel_row=row,
                write_plan=(),
                trigger_cell_address=trigger_address,
                trigger_mapping_name=cancel_trigger,
                bet_ref_before=bet_ref_before,
                matched_stake_cell_address=matched_stake_address,
                matched_stake_cell_value=matched_stake_cell_value,
            )
        if matched_stake_value > 0:
            return self._finish(
                intent,
                context,
                "GRUSS_PRE_CANCEL_BEFORE_POST_SKIPPED",
                "pre_cancel_skipped_matched_stake_gt_zero",
                excel_sheet=sheet_name,
                excel_row=row,
                write_plan=(),
                trigger_cell_address=trigger_address,
                trigger_mapping_name=cancel_trigger,
                bet_ref_before=bet_ref_before,
                matched_stake_cell_address=matched_stake_address,
                matched_stake_cell_value=matched_stake_cell_value,
            )

        max_orders = self._max_orders_for_intent(intent)
        max_key = _max_orders_key(intent, context)
        if max_orders is not None and self.real_order_counts.get(max_key, 0) >= max_orders:
            return self._finish(
                intent,
                context,
                "GRUSS_PRE_CANCEL_BEFORE_POST_SKIPPED",
                "pre_cancel_before_post_skipped_max_orders_reached",
                excel_sheet=sheet_name,
                excel_row=row,
                write_plan=(),
                trigger_cell_address=trigger_address,
                trigger_mapping_name=cancel_trigger,
                bet_ref_before=bet_ref_before,
                matched_stake_cell_address=matched_stake_address,
                matched_stake_cell_value=matched_stake_cell_value,
            )

        plan: list[tuple[str, Any]] = []
        stripped_bet_ref = strip_gruss_ref_suffix(bet_ref_before)
        bet_ref_suffix_n_handled = stripped_bet_ref != bet_ref_before
        if bet_ref_suffix_n_handled:
            plan.append((bet_ref_address, stripped_bet_ref))
        plan.append((trigger_address, cancel_trigger))

        try:
            trigger_value = self._read_cell_with_retry(
                sheet_name,
                trigger_address,
                f"read_trigger_cell:{sheet_name}!{trigger_address}",
            )
        except Exception as exc:
            return self._finish(
                intent,
                context,
                "GRUSS_PRE_CANCEL_BEFORE_POST_SKIPPED",
                f"pre_cancel_before_post_skipped_trigger_read_failed: {exc}",
                excel_sheet=sheet_name,
                excel_row=row,
                write_plan=tuple(plan),
                trigger_cell_address=trigger_address,
                trigger_mapping_name=cancel_trigger,
                bet_ref_before=bet_ref_before,
            )
        if trigger_value not in (None, ""):
            return self._finish(
                intent,
                context,
                "GRUSS_PRE_CANCEL_BEFORE_POST_SKIPPED",
                "pre_cancel_before_post_skipped_trigger_cell_not_empty",
                excel_sheet=sheet_name,
                excel_row=row,
                write_plan=tuple(plan),
                trigger_cell_address=trigger_address,
                trigger_cell_current_value=trigger_value,
                trigger_cell_expected_empty=False,
                trigger_mapping_name=cancel_trigger,
                bet_ref_before=bet_ref_before,
            )

        write_result = self._write_cells_with_retry(sheet_name, tuple(plan))
        if write_result.exception is not None:
            return self._finish(
                intent,
                context,
                "GRUSS_PRE_CANCEL_BEFORE_POST_SKIPPED",
                f"pre_cancel_before_post_skipped_{write_result.final_status}: {write_result.exception}",
                excel_sheet=sheet_name,
                excel_row=row,
                write_plan=tuple(plan),
                trigger_cell_address=trigger_address,
                trigger_mapping_name=cancel_trigger,
                bet_ref_before=bet_ref_before,
                excel_write_attempt=write_result.attempt,
                excel_write_retry_count=write_result.retry_count,
                excel_write_retry_backoff_ms=write_result.retry_backoff_ms,
                excel_write_final_status=write_result.final_status,
                excel_unavailable_recovered=write_result.recovered,
            )
        written = write_result.written

        try:
            post_write_trigger_value = self.bridge.read_cell(sheet_name, trigger_address)
        except Exception:
            post_write_trigger_value = None
        trigger_written = _values_match(cancel_trigger, post_write_trigger_value)
        clear_outcome = _TriggerClearOutcome()
        if trigger_written:
            clear_outcome = self._clear_written_trigger(
                sheet_name,
                trigger_address,
                cancel_trigger,
                hold_for_visual_test=False,
                delay_override_ms=_pre_ladder_trigger_clear_delay_override_ms(),
                processed_key=_processed_key(intent, context),
                expected_command_cell_values=tuple(plan),
            )
            self.real_order_counts[_processed_key(intent, context)] = (
                self.real_order_counts.get(_processed_key(intent, context), 0) + 1
            )
            self.real_order_counts[max_key] = self.real_order_counts.get(max_key, 0) + 1
            self.active_pre_ladders.pop(ladder_id, None)
        return self._finish(
            intent,
            context,
            "GRUSS_PRE_CANCEL_BEFORE_POST_WRITTEN" if trigger_written else "GRUSS_PRE_CANCEL_BEFORE_POST_SKIPPED",
            "pre_cancel_before_post_written" if trigger_written else "pre_cancel_before_post_write_not_verified",
            excel_sheet=sheet_name,
            excel_row=row,
            excel_cells_written=written,
            write_plan=tuple(plan),
            trigger_written=trigger_written,
            trigger_cell_address=trigger_address,
            trigger_cell_current_value=trigger_value,
            trigger_cell_expected_empty=True,
            trigger_mapping_name=cancel_trigger,
            trigger_value_written=cancel_trigger if trigger_written else "",
            bet_ref_before=stripped_bet_ref if bet_ref_suffix_n_handled else bet_ref_before,
            bet_ref_suffix_n_handled=bet_ref_suffix_n_handled,
            matched_stake_cell_address=matched_stake_address,
            matched_stake_cell_value=matched_stake_cell_value,
            trigger_clear_attempted=clear_outcome.attempted,
            trigger_cleared=clear_outcome.cleared,
            trigger_clear_reason=clear_outcome.reason,
            trigger_cell_value_before_clear=clear_outcome.value_before_clear,
            trigger_clear_delay_ms=clear_outcome.delay_ms,
            command_cells_clear_attempted=clear_outcome.attempted,
            command_cells_cleared=clear_outcome.cleared,
            command_cells_clear_reason=clear_outcome.command_reason,
            command_cells_clear_addresses=";".join(clear_outcome.addresses),
            command_cells_clear_delay_ms=clear_outcome.delay_ms,
            command_cells_clear_scheduled=clear_outcome.scheduled,
            command_cells_clear_due_time=clear_outcome.due_time,
            command_cells_clear_non_blocking=clear_outcome.non_blocking,
            command_cells_clear_executed=clear_outcome.executed,
            command_cells_clear_lag_ms=clear_outcome.lag_ms,
            excel_write_attempt=write_result.attempt,
            excel_write_retry_count=write_result.retry_count,
            excel_write_retry_backoff_ms=write_result.retry_backoff_ms,
            excel_write_final_status=write_result.final_status,
            excel_unavailable_recovered=write_result.recovered,
            no_stacking_check_passed=True,
        )

    def update_batch_write_log(
        self,
        processed_keys: Iterable[str],
        *,
        batch_write_end_timestamp: str,
        batch_write_duration_ms: int,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        keys = {str(key) for key in processed_keys if key}
        if not keys:
            return
        common_fields = {
            "batch_write_end_timestamp": batch_write_end_timestamp,
            "batch_write_duration_ms": batch_write_duration_ms,
        }
        if extra_fields:
            common_fields.update(extra_fields)
        self._update_attempt_log_rows(
            keys,
            common_fields=common_fields,
        )

    def confirm_post_order_batch(
        self,
        results: Iterable[GrussRealOrderResult],
        context: GrussRealOrderContext,
    ) -> None:
        pending = [
            result
            for result in results
            if str(getattr(result, "status", "") or "") == "POST_WRITE_PENDING_CONFIRMATION"
            and bool(getattr(result, "trigger_written", False))
            and bool(getattr(result, "post_write_verified", False))
        ]
        if not pending:
            return
        wait_ms = _post_bet_ref_wait_ms()
        poll_ms = _post_bet_ref_poll_ms()
        started = time.perf_counter()
        deadline = started + (wait_ms / 1000)
        unresolved = {result.processed_key: result for result in pending if result.processed_key}
        attempts = {result.processed_key: 0 for result in pending if result.processed_key}
        last_bet_refs = {result.processed_key: "" for result in pending if result.processed_key}
        confirmation_source = {result.processed_key: "" for result in pending if result.processed_key}
        confirmed: set[str] = set()

        while unresolved:
            for key, result in list(unresolved.items()):
                attempts[key] = attempts.get(key, 0) + 1
                bet_ref_address = self.layout.bet_ref_address(int(result.excel_row or 0))
                try:
                    bet_ref = normalise_gruss_bet_ref(
                        self._read_cell_with_retry(
                            result.excel_sheet,
                            bet_ref_address,
                            f"post_batch_read_bet_ref:{result.excel_sheet}!{bet_ref_address}",
                        )
                    )
                except Exception:
                    bet_ref = ""
                last_bet_refs[key] = bet_ref
                existing = normalise_gruss_bet_ref(result.post_existing_bet_ref_before)
                if is_valid_bet_ref(bet_ref) and (
                    not is_valid_bet_ref(existing)
                    or strip_gruss_ref_suffix(bet_ref) != strip_gruss_ref_suffix(existing)
                ):
                    confirmed.add(key)
                    confirmation_source[key] = f"post_batch_excel_row_poll:{bet_ref_address}"
                    unresolved.pop(key, None)
            if not unresolved or wait_ms <= 0 or time.perf_counter() >= deadline:
                break
            remaining_ms = max(0.0, (deadline - time.perf_counter()) * 1000)
            time.sleep(min(float(poll_ms), remaining_ms) / 1000)

        for key, result in list(unresolved.items()):
            lookup = self._lookup_post_selection_confirmation(
                result.excel_sheet,
                _intent_from_result(result),
                context,
                post_write_timestamp=result.post_write_timestamp or result.post_order_write_timestamp,
                existing_bet_ref_before=result.post_existing_bet_ref_before,
                expected_market_id=result.post_expected_market_id,
                expected_market_type=result.post_expected_market_type,
                expected_runner=result.post_expected_runner,
                expected_selection_id=result.post_expected_selection_id,
                expected_side=result.post_expected_side,
                expected_stake=result.post_expected_stake,
                expected_price=result.post_expected_price,
            )
            _set_result_attr(result, "post_selections_lookup_attempted", lookup.attempted)
            _set_result_attr(result, "post_selections_match_found", lookup.match is not None)
            _set_result_attr(result, "post_selections_match_reason", lookup.match_reason)
            _set_result_attr(result, "post_selections_reject_reason", lookup.reject_reason)
            if lookup.match is not None:
                confirmed.add(key)
                last_bet_refs[key] = lookup.match.bet_ref
                confirmation_source[key] = lookup.lookup_source
                unresolved.pop(key, None)

        confirmation_duration_ms = int(round((time.perf_counter() - started) * 1000))
        per_key: dict[str, dict[str, Any]] = {}
        confirmed_count = 0
        for result in pending:
            key = result.processed_key
            bet_ref = last_bet_refs.get(key, "")
            is_confirmed = key in confirmed
            existing = normalise_gruss_bet_ref(result.post_existing_bet_ref_before)
            bet_ref_changed = bool(
                is_valid_bet_ref(bet_ref)
                and (
                    not is_valid_bet_ref(existing)
                    or strip_gruss_ref_suffix(bet_ref) != strip_gruss_ref_suffix(existing)
                )
            )
            matched_before = (
                result.post_total_matched_before
                if result.post_total_matched_before is not None
                else result.post_existing_matched_before
            )
            matched_after = _matched_stake_value(
                _read_cell_quiet(
                    self.bridge,
                    result.excel_sheet,
                    self.layout.matched_stake_address(int(result.excel_row or 0)),
                )
            )
            matched_delta = _matched_stake_delta(matched_before, matched_after)
            added_stake_confirmed = bool(matched_delta is not None and matched_delta > 0)
            unconfirmed_reason = ""
            order_source = confirmation_source.get(key, "")
            if added_stake_confirmed:
                if not is_confirmed:
                    is_confirmed = True
                    order_source = "post_matched_stake_delta"
                _set_result_attr(result, "post_added_stake_confirmed", True)
                _set_result_attr(result, "post_added_stake_amount", matched_delta)
            confirmed_count += 1 if is_confirmed else 0
            if not is_confirmed:
                if is_valid_bet_ref(bet_ref) and is_valid_bet_ref(existing):
                    unconfirmed_reason = "POST_BET_REF_NOT_NEW_AND_NO_STAKE_DELTA"
                    order_source = "post_existing_bet_ref_unchanged"
                else:
                    unconfirmed_reason = "POST_WRITE_ATTEMPTED_BUT_NO_NEW_ORDER_EVIDENCE"
            status = "GRUSS_REAL_WRITTEN" if is_confirmed else (
                "POST_WRITE_UNCONFIRMED_EXISTING_PRE_BETREF"
                if unconfirmed_reason == "POST_BET_REF_NOT_NEW_AND_NO_STAKE_DELTA"
                else "POST_WRITE_UNCONFIRMED"
            )
            reason = "excel_trigger_written" if is_confirmed else unconfirmed_reason
            clear_outcome = _TriggerClearOutcome()
            cells_cleared_after_confirmation = False
            cells_cleared_after_unconfirmed = False
            post_clear_reason = ""
            if _post_clear_after_bet_ref() and result.trigger_cell_address:
                clear_outcome = self._clear_written_trigger(
                    result.excel_sheet,
                    result.trigger_cell_address,
                    result.trigger_mapping_name,
                    hold_for_visual_test=self.hold_trigger_for_visual_test,
                    processed_key=key,
                    expected_command_cell_values=result.write_plan,
                    delay_override_ms=_post_command_cells_clear_delay_ms(),
                )
                cleared = bool(clear_outcome.attempted and (clear_outcome.cleared or clear_outcome.scheduled))
                if is_confirmed:
                    cells_cleared_after_confirmation = cleared
                    post_clear_reason = "confirmed_post_cleanup" if cleared else clear_outcome.command_reason or clear_outcome.reason
                else:
                    cells_cleared_after_unconfirmed = cleared
                    post_clear_reason = "unconfirmed_post_cleanup" if cleared else clear_outcome.command_reason or clear_outcome.reason

            _set_result_attr(result, "status", status)
            _set_result_attr(result, "reason", reason)
            _set_result_attr(result, "post_write_status", status)
            _set_result_attr(result, "post_write_reason", reason)
            _set_result_attr(result, "post_bet_ref_wait_attempted", True)
            _set_result_attr(result, "post_bet_ref_after", bet_ref)
            _set_result_attr(result, "post_bet_ref_changed", bet_ref_changed)
            _set_result_attr(result, "post_bet_ref_confirmed_new", bool(is_confirmed and bet_ref_changed))
            _set_result_attr(result, "post_new_bet_ref_found", bool(bet_ref_changed and is_valid_bet_ref(bet_ref)))
            _set_result_attr(
                result,
                "post_new_bet_ref",
                bet_ref if bet_ref_changed and is_valid_bet_ref(bet_ref) else "",
            )
            _set_result_attr(result, "post_total_matched_before", matched_before)
            _set_result_attr(result, "post_total_matched_after", matched_after)
            _set_result_attr(result, "post_total_matched_delta", matched_delta)
            if not added_stake_confirmed:
                _set_result_attr(result, "post_added_stake_confirmed", False)
                _set_result_attr(result, "post_added_stake_amount", None)
            _set_result_attr(result, "post_bet_ref_poll_attempts", attempts.get(key, 0))
            _set_result_attr(result, "post_bet_ref_poll_duration_ms", confirmation_duration_ms)
            _set_result_attr(result, "post_order_confirmed", is_confirmed)
            _set_result_attr(result, "post_order_confirmation_source", order_source)
            _set_result_attr(result, "post_confirmation_source", order_source)
            _set_result_attr(result, "post_write_unconfirmed_reason", unconfirmed_reason)
            _set_result_attr(result, "post_unconfirmed_reason", unconfirmed_reason)
            _set_result_attr(result, "post_reject_reason", "" if is_confirmed else unconfirmed_reason)
            _set_result_attr(result, "post_cells_cleared_after_confirmation", cells_cleared_after_confirmation)
            _set_result_attr(result, "post_cells_cleared_after_unconfirmed", cells_cleared_after_unconfirmed)
            _set_result_attr(result, "post_clear_reason", post_clear_reason)
            _set_result_attr(result, "post_batch_confirmation_started", True)
            _set_result_attr(result, "post_batch_confirmation_duration_ms", confirmation_duration_ms)
            if is_confirmed:
                self.real_order_counts[key] = self.real_order_counts.get(key, 0) + 1

            per_key[key] = {
                "status": status,
                "reason": reason,
                "post_write_status": status,
                "post_write_reason": reason,
                "post_bet_ref_wait_attempted": True,
                "post_bet_ref_after": bet_ref,
                "post_bet_ref_changed": bet_ref_changed,
                "post_bet_ref_confirmed_new": bool(is_confirmed and bet_ref_changed),
                "post_new_bet_ref_found": bool(bet_ref_changed and is_valid_bet_ref(bet_ref)),
                "post_new_bet_ref": bet_ref if bet_ref_changed and is_valid_bet_ref(bet_ref) else "",
                "post_added_stake_confirmed": added_stake_confirmed,
                "post_added_stake_amount": matched_delta if added_stake_confirmed else "",
                "post_total_matched_before": matched_before,
                "post_total_matched_after": matched_after,
                "post_total_matched_delta": matched_delta,
                "post_bet_ref_poll_attempts": attempts.get(key, 0),
                "post_bet_ref_poll_duration_ms": confirmation_duration_ms,
                "post_order_confirmed": is_confirmed,
                "post_order_confirmation_source": order_source,
                "post_confirmation_source": order_source,
                "post_selections_lookup_attempted": result.post_selections_lookup_attempted,
                "post_selections_match_found": result.post_selections_match_found,
                "post_selections_match_reason": result.post_selections_match_reason,
                "post_selections_reject_reason": result.post_selections_reject_reason,
                "post_write_unconfirmed_reason": unconfirmed_reason,
                "post_unconfirmed_reason": unconfirmed_reason,
                "post_reject_reason": "" if is_confirmed else unconfirmed_reason,
                "post_cells_cleared_after_confirmation": cells_cleared_after_confirmation,
                "post_cells_cleared_after_unconfirmed": cells_cleared_after_unconfirmed,
                "post_clear_reason": post_clear_reason,
                "post_batch_written_count": len(pending),
                "post_batch_confirmation_started": True,
                "post_batch_confirmation_duration_ms": confirmation_duration_ms,
            }
        self._update_attempt_log_rows(
            per_key.keys(),
            per_key_fields=per_key,
        )

    def collect_pre_ladder_bet_refs(self, results: Iterable[GrussRealOrderResult]) -> None:
        candidates: list[tuple[GrussRealOrderResult, _ActivePreLadderState]] = []
        for result in results:
            if (
                result.status != "GRUSS_PRE_LADDER_WRITTEN"
                or not result.trigger_written
                or (
                    _ladder_step_index(result.ladder_step) != 0
                    and not bool(getattr(result, "pre_retry_allowed", False))
                )
            ):
                continue
            state = self.active_pre_ladders.get(result.ladder_id)
            if state is None or state.bet_ref:
                continue
            candidates.append((result, state))
        if not candidates:
            return

        diagnostic_hold_after_batch = _env_bool("DOGBOT_GRUSS_BET_REF_DIAGNOSTIC_HOLD_AFTER_BATCH", False)
        if diagnostic_hold_after_batch:
            attempts_limit = _bounded_int_env("DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS", 40, 1, 60)
            interval_ms = _bounded_int_env("DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS", 250, 0, 500)
        else:
            attempts_limit = _bounded_int_env("DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS", 10, 1, 20)
            interval_ms = _bounded_int_env("DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS", 250, 0, 300)
        lookup_sources = _bet_ref_lookup_sources_from_env()
        lookup_sources_text = ",".join(lookup_sources)
        market_query = _clean_text(self._batch_log_context.get("market_query"))
        diagnostic_keep_triggers = _diagnostic_keep_triggers()
        collection_start = _utc_now()
        started = time.perf_counter()
        attempts_by_key = {result.processed_key: 0 for result, _state in candidates}
        row_t_value_by_key = {result.processed_key: "" for result, _state in candidates}
        row_t_dump_by_key = {result.processed_key: "" for result, _state in candidates}
        runner_qz_dump_by_key = {result.processed_key: "" for result, _state in candidates}
        selections_headers_by_key = {result.processed_key: "" for result, _state in candidates}
        selections_full_recent_by_key = {result.processed_key: "" for result, _state in candidates}
        workbook_sheet_names = _workbook_sheet_names(self.bridge)
        source_by_key = {result.processed_key: "bet_ref_lookup_timeout" for result, _state in candidates}
        source_used_by_key = {result.processed_key: "" for result, _state in candidates}
        selections_rows_scanned_by_key = {result.processed_key: 0 for result, _state in candidates}
        selections_match_found_by_key = {result.processed_key: False for result, _state in candidates}
        selections_match_reason_by_key = {result.processed_key: "" for result, _state in candidates}
        selection_evidence_found_by_key = {result.processed_key: False for result, _state in candidates}
        selection_evidence_reason_by_key = {result.processed_key: "" for result, _state in candidates}
        selections_runner_by_key = {result.processed_key: "" for result, _state in candidates}
        selections_side_by_key = {result.processed_key: "" for result, _state in candidates}
        selections_stake_by_key: dict[str, float | None] = {
            result.processed_key: None for result, _state in candidates
        }
        selections_bet_ref_by_key = {result.processed_key: "" for result, _state in candidates}
        selections_req_odds_by_key: dict[str, float | None] = {
            result.processed_key: None for result, _state in candidates
        }
        selections_market_name_by_key = {result.processed_key: "" for result, _state in candidates}
        selections_debug_rows_by_key = {result.processed_key: "" for result, _state in candidates}
        selections_top_candidates_by_key = {result.processed_key: "" for result, _state in candidates}
        selections_market_rows_by_key = {result.processed_key: "" for result, _state in candidates}
        selections_runner_rows_by_key = {result.processed_key: "" for result, _state in candidates}
        used_bet_refs: set[str] = set()
        missing = list(candidates)
        for attempt in range(1, attempts_limit + 1):
            if "ROW_T" in lookup_sources:
                for result, state in missing:
                    attempts_by_key[result.processed_key] = attempt
                    address = self.layout.bet_ref_address(state.row)
                    sheet_name = state.market_type or result.excel_sheet or "PLACE"
                    try:
                        bet_ref = normalise_gruss_bet_ref(self.bridge.read_cell(sheet_name, address))
                    except Exception:
                        bet_ref = ""
                    row_t_value_by_key[result.processed_key] = bet_ref
                    if not bet_ref:
                        source_by_key[result.processed_key] = f"excel_row_batch_poll_timeout:{address}"
                        continue
                    if not is_valid_bet_ref(bet_ref):
                        source_by_key[result.processed_key] = f"excel_row_batch_poll_invalid:{address}:{bet_ref}"
                        continue
                    if bet_ref in used_bet_refs:
                        source_by_key[result.processed_key] = f"excel_row_batch_poll_duplicate:{address}"
                        continue
                    used_bet_refs.add(bet_ref)
                    state.bet_ref = bet_ref
                    _set_result_attr(result, "bet_ref_after", bet_ref)
                    _set_result_attr(result, "pre_bet_ref_required", True)
                    _set_result_attr(result, "pre_bet_ref_confirmed", True)
                    _set_result_attr(result, "pre_bet_ref_missing", False)
                    _set_result_attr(result, "bet_ref_lookup_source", f"excel_row_batch_poll:{address}")
                    _set_result_attr(result, "bet_ref_lookup_source_used", "ROW_T")
                    _set_result_attr(result, "bet_ref_lookup_matched_runner", "True")
                    _set_result_attr(result, "active_ladder_bet_ref_stored", True)
                    source_by_key[result.processed_key] = f"excel_row_batch_poll:{address}"
                    source_used_by_key[result.processed_key] = "ROW_T"

            missing = [(result, state) for result, state in missing if not state.bet_ref]
            if missing and "SELECTIONS_SHEET" in lookup_sources:
                by_sheet: dict[str, list[tuple[GrussRealOrderResult, _ActivePreLadderState]]] = {}
                for result, state in missing:
                    sheet_name = f"{(state.market_type or result.excel_sheet or 'PLACE').upper()}_Selections"
                    by_sheet.setdefault(sheet_name, []).append((result, state))
                for sheet_name, sheet_missing in by_sheet.items():
                    selection_candidates, rows_scanned, selection_headers = self._read_selection_bet_ref_candidates(
                        sheet_name
                    )
                    full_recent_rows = _format_recent_selection_rows(selection_candidates, limit=20)
                    for result, state in sheet_missing:
                        attempts_by_key[result.processed_key] = attempt
                        selections_rows_scanned_by_key[result.processed_key] += rows_scanned
                        selections_headers_by_key[result.processed_key] = selection_headers
                        selections_full_recent_by_key[result.processed_key] = full_recent_rows
                        match, reason, top_candidates = _match_selection_bet_ref_candidate(
                            state,
                            result,
                            selection_candidates,
                            used_bet_refs=used_bet_refs,
                        )
                        evidence_match, evidence_reason = _match_selection_activity_evidence_candidate(
                            state,
                            result,
                            selection_candidates,
                        )
                        if evidence_match is not None:
                            selection_evidence_found_by_key[result.processed_key] = True
                            selection_evidence_reason_by_key[result.processed_key] = evidence_reason
                        selections_match_reason_by_key[result.processed_key] = reason
                        selections_debug_rows_by_key[result.processed_key] = _format_recent_selection_rows(
                            selection_candidates
                        )
                        selections_top_candidates_by_key[result.processed_key] = top_candidates
                        selections_market_rows_by_key[result.processed_key] = _format_selection_rows_for_market(
                            selection_candidates,
                            market_query,
                        )
                        selections_runner_rows_by_key[result.processed_key] = _format_selection_rows_for_runner(
                            selection_candidates,
                            state,
                        )
                        if match is None:
                            source_by_key[result.processed_key] = (
                                f"selections_sheet_poll_timeout:{sheet_name}"
                            )
                            if evidence_reason:
                                selections_match_reason_by_key[result.processed_key] = evidence_reason
                            continue
                        used_bet_refs.add(match.bet_ref)
                        state.bet_ref = match.bet_ref
                        selections_match_found_by_key[result.processed_key] = True
                        selection_evidence_found_by_key[result.processed_key] = True
                        selection_evidence_reason_by_key[result.processed_key] = reason
                        selections_runner_by_key[result.processed_key] = match.runner
                        selections_side_by_key[result.processed_key] = match.side
                        selections_stake_by_key[result.processed_key] = match.stake
                        selections_bet_ref_by_key[result.processed_key] = match.bet_ref
                        selections_req_odds_by_key[result.processed_key] = match.req_odds
                        selections_market_name_by_key[result.processed_key] = match.market_name
                        _set_result_attr(result, "bet_ref_after", match.bet_ref)
                        _set_result_attr(result, "pre_bet_ref_required", True)
                        _set_result_attr(result, "pre_bet_ref_confirmed", True)
                        _set_result_attr(result, "pre_bet_ref_missing", False)
                        _set_result_attr(
                            result,
                            "bet_ref_lookup_source",
                            f"selections_sheet_poll:{sheet_name}!row{match.row_number}",
                        )
                        _set_result_attr(result, "bet_ref_lookup_source_used", "SELECTIONS_SHEET")
                        _set_result_attr(result, "bet_ref_lookup_matched_runner", "True")
                        _set_result_attr(result, "active_ladder_bet_ref_stored", True)
                        source_by_key[result.processed_key] = (
                            f"selections_sheet_poll:{sheet_name}!row{match.row_number}"
                        )
                        source_used_by_key[result.processed_key] = "SELECTIONS_SHEET"

            missing = [(result, state) for result, state in missing if not state.bet_ref]
            if not missing:
                break
            if attempt < attempts_limit and interval_ms > 0:
                time.sleep(interval_ms / 1000)
        duration_ms = int(round((time.perf_counter() - started) * 1000))
        collection_end = _utc_now()
        found_count = len(candidates) - len(missing)
        missing_count = len(missing)
        row_t_dump = self._dump_bet_ref_cells(candidates)
        runner_qz_dump = self._dump_runner_qz_cells(candidates)
        for result, _state in candidates:
            row_t_dump_by_key[result.processed_key] = row_t_dump
            runner_qz_dump_by_key[result.processed_key] = runner_qz_dump

        per_key: dict[str, dict[str, Any]] = {}
        for result, state in candidates:
            stored = bool(state.bet_ref)
            matched_evidence_found = _result_matched_evidence_found(result)
            selection_row_evidence_found = bool(selection_evidence_found_by_key.get(result.processed_key, False))
            evidence_found = matched_evidence_found or selection_row_evidence_found
            retry_key = _pre_retry_key_for_result(result)
            retry_count = int(self.pre_bet_ref_missing_retry_counts.get(retry_key, 0))
            retry_allowed = False
            retry_reason = str(getattr(result, "pre_retry_reason", "") or "")
            retry_block_reason = ""
            missing_retryable = False
            action = str(getattr(result, "action", "") or result.status)
            active_ladder_created = bool(stored)
            pending_ladder_created = False
            no_stacking_blocked_retry = False
            if stored:
                self.pre_bet_ref_missing_retry_counts.pop(retry_key, None)
                state.pending_confirmation = False
                retry_count = 0
            elif evidence_found:
                state.pending_confirmation = True
                pending_ladder_created = True
                no_stacking_blocked_retry = True
                retry_block_reason = (
                    "matched_evidence_found" if matched_evidence_found else "selection_row_evidence_found"
                )
                action = "PRE_LADDER_BET_REF_MISSING"
            else:
                max_retries = _pre_bet_ref_missing_max_retries()
                if retry_count < max_retries:
                    retry_allowed = True
                    missing_retryable = True
                    retry_reason = "bet_ref_missing_retry_next_pre_step"
                else:
                    retry_block_reason = "pre_bet_ref_missing_retry_limit_reached"
                action = "PRE_LADDER_BET_REF_MISSING"
            _set_result_attr(result, "bet_ref_poll_attempts", attempts_by_key.get(result.processed_key, 0))
            _set_result_attr(result, "bet_ref_poll_duration_ms", duration_ms)
            _set_result_attr(result, "pre_bet_ref_required", True)
            _set_result_attr(result, "pre_bet_ref_confirmed", stored)
            _set_result_attr(result, "pre_bet_ref_found", stored)
            _set_result_attr(result, "pre_bet_ref_missing", not stored)
            _set_result_attr(result, "pre_bet_ref_poll_attempts", attempts_by_key.get(result.processed_key, 0))
            _set_result_attr(result, "pre_bet_ref_poll_duration_ms", duration_ms)
            _set_result_attr(result, "pre_bet_ref_missing_retryable", missing_retryable)
            _set_result_attr(result, "pre_retry_count", retry_count)
            _set_result_attr(result, "pre_retry_allowed", retry_allowed)
            _set_result_attr(result, "pre_retry_reason", retry_reason)
            _set_result_attr(result, "pre_retry_block_reason", retry_block_reason)
            _set_result_attr(result, "bet_ref_lookup_sources", lookup_sources_text)
            _set_result_attr(result, "bet_ref_lookup_source_used", source_used_by_key.get(result.processed_key, ""))
            _set_result_attr(result, "bet_ref_collection_phase_start", collection_start)
            _set_result_attr(result, "bet_ref_collection_phase_end", collection_end)
            _set_result_attr(result, "bet_ref_collection_duration_ms", duration_ms)
            _set_result_attr(result, "bet_ref_found_count", found_count)
            _set_result_attr(result, "bet_ref_missing_count", missing_count)
            _set_result_attr(result, "row_t_value", row_t_value_by_key.get(result.processed_key, ""))
            _set_result_attr(
                result,
                "selections_rows_scanned",
                selections_rows_scanned_by_key.get(result.processed_key, 0),
            )
            _set_result_attr(
                result,
                "selections_match_found",
                selections_match_found_by_key.get(result.processed_key, False),
            )
            _set_result_attr(
                result,
                "selections_match_reason",
                selections_match_reason_by_key.get(result.processed_key, ""),
            )
            _set_result_attr(result, "selection_row_evidence_found", selection_row_evidence_found)
            _set_result_attr(result, "selections_runner", selections_runner_by_key.get(result.processed_key, ""))
            _set_result_attr(result, "selections_side", selections_side_by_key.get(result.processed_key, ""))
            _set_result_attr(result, "selections_stake", selections_stake_by_key.get(result.processed_key))
            _set_result_attr(result, "selections_bet_ref", selections_bet_ref_by_key.get(result.processed_key, ""))
            _set_result_attr(result, "selections_req_odds", selections_req_odds_by_key.get(result.processed_key))
            _set_result_attr(
                result,
                "selections_market_name",
                selections_market_name_by_key.get(result.processed_key, ""),
            )
            _set_result_attr(
                result,
                "selections_debug_recent_rows",
                selections_debug_rows_by_key.get(result.processed_key, ""),
            )
            _set_result_attr(
                result,
                "selections_top_candidates",
                selections_top_candidates_by_key.get(result.processed_key, ""),
            )
            _set_result_attr(result, "bet_ref_row_t_dump", row_t_dump_by_key.get(result.processed_key, ""))
            _set_result_attr(result, "bet_ref_diagnostic_hold_after_batch", diagnostic_hold_after_batch)
            _set_result_attr(result, "selections_market_query", market_query)
            _set_result_attr(
                result,
                "selections_current_market_rows",
                selections_market_rows_by_key.get(result.processed_key, ""),
            )
            _set_result_attr(
                result,
                "selections_current_runner_rows",
                selections_runner_rows_by_key.get(result.processed_key, ""),
            )
            _set_result_attr(result, "runner_qz_dump", runner_qz_dump_by_key.get(result.processed_key, ""))
            _set_result_attr(
                result,
                "selections_sheet_headers",
                selections_headers_by_key.get(result.processed_key, ""),
            )
            _set_result_attr(
                result,
                "selections_full_recent_rows",
                selections_full_recent_by_key.get(result.processed_key, ""),
            )
            _set_result_attr(result, "workbook_sheet_names", workbook_sheet_names)
            _set_result_attr(result, "diagnostic_keep_triggers", diagnostic_keep_triggers)
            _set_result_attr(result, "active_ladder_bet_ref_stored", bool(stored))
            _set_result_attr(result, "active_ladder_created", active_ladder_created)
            _set_result_attr(result, "pending_ladder_created", pending_ladder_created)
            _set_result_attr(result, "matched_evidence_found", matched_evidence_found)
            _set_result_attr(result, "no_stacking_blocked_retry", no_stacking_blocked_retry)
            _set_result_attr(result, "action", action)
            if not stored and not evidence_found:
                self.active_pre_ladders.pop(result.ladder_id, None)
                if self.active_pre_ladder_id == result.ladder_id:
                    self._refresh_legacy_active_ladder_snapshot()
                status = "PRE_LADDER_BET_REF_MISSING_RETRYABLE" if retry_allowed else "PRE_LADDER_BET_REF_MISSING"
                reason = (
                    "pre_ladder_bet_ref_missing_retryable_after_poll"
                    if retry_allowed
                    else retry_block_reason or "pre_ladder_bet_ref_missing_after_poll"
                )
                _set_result_attr(result, "status", status)
                _set_result_attr(result, "reason", reason)
                _set_result_attr(result, "pre_unconfirmed_reason", "bet_ref_missing_after_poll")
                _set_result_attr(
                    result,
                    "bet_ref_lookup_source",
                    source_by_key.get(result.processed_key, ""),
                )
                _set_result_attr(result, "bet_ref_lookup_matched_runner", "False")
                _set_result_attr(result, "active_ladder_bet_ref_stored", False)
            elif not stored and evidence_found:
                _set_result_attr(result, "status", "PRE_LADDER_BET_REF_LATE_OR_MATCH_EVIDENCE")
                _set_result_attr(result, "reason", retry_block_reason)
                _set_result_attr(result, "pre_unconfirmed_reason", "bet_ref_missing_with_order_evidence")
                _set_result_attr(
                    result,
                    "bet_ref_lookup_source",
                    source_by_key.get(result.processed_key, ""),
                )
                _set_result_attr(result, "bet_ref_lookup_matched_runner", "False")
            per_key[result.processed_key] = {
                "bet_ref_after": state.bet_ref,
                "action": action,
                "bet_ref_poll_attempts": attempts_by_key.get(result.processed_key, 0),
                "bet_ref_poll_duration_ms": duration_ms,
                "bet_ref_lookup_sources": lookup_sources_text,
                "bet_ref_lookup_source_used": source_used_by_key.get(result.processed_key, ""),
                "bet_ref_lookup_source": result.bet_ref_lookup_source
                or source_by_key.get(result.processed_key, ""),
                "bet_ref_lookup_matched_runner": "True" if stored else "False",
                "row_t_value": row_t_value_by_key.get(result.processed_key, ""),
                "selections_rows_scanned": selections_rows_scanned_by_key.get(result.processed_key, 0),
                "selections_match_found": str(
                    bool(selections_match_found_by_key.get(result.processed_key, False))
                ),
                "selections_match_reason": selections_match_reason_by_key.get(result.processed_key, ""),
                "selection_row_evidence_found": str(bool(selection_row_evidence_found)),
                "selections_runner": selections_runner_by_key.get(result.processed_key, ""),
                "selections_side": selections_side_by_key.get(result.processed_key, ""),
                "selections_stake": ""
                if selections_stake_by_key.get(result.processed_key) is None
                else selections_stake_by_key.get(result.processed_key),
                "selections_bet_ref": selections_bet_ref_by_key.get(result.processed_key, ""),
                "selections_req_odds": ""
                if selections_req_odds_by_key.get(result.processed_key) is None
                else selections_req_odds_by_key.get(result.processed_key),
                "selections_market_name": selections_market_name_by_key.get(result.processed_key, ""),
                "selections_debug_recent_rows": selections_debug_rows_by_key.get(result.processed_key, ""),
                "selections_top_candidates": selections_top_candidates_by_key.get(result.processed_key, ""),
                "bet_ref_row_t_dump": row_t_dump_by_key.get(result.processed_key, ""),
                "bet_ref_diagnostic_hold_after_batch": str(bool(diagnostic_hold_after_batch)),
                "selections_market_query": market_query,
                "selections_current_market_rows": selections_market_rows_by_key.get(result.processed_key, ""),
                "selections_current_runner_rows": selections_runner_rows_by_key.get(result.processed_key, ""),
                "runner_qz_dump": runner_qz_dump_by_key.get(result.processed_key, ""),
                "selections_sheet_headers": selections_headers_by_key.get(result.processed_key, ""),
                "selections_full_recent_rows": selections_full_recent_by_key.get(result.processed_key, ""),
                "workbook_sheet_names": workbook_sheet_names,
                "diagnostic_keep_triggers": str(bool(diagnostic_keep_triggers)),
                "active_ladder_bet_ref_stored": str(bool(stored)),
                "active_ladder_created": str(bool(active_ladder_created)),
                "pending_ladder_created": str(bool(pending_ladder_created)),
                "matched_evidence_found": str(bool(matched_evidence_found)),
                "no_stacking_blocked_retry": str(bool(no_stacking_blocked_retry)),
                "pre_bet_ref_required": "True",
                "pre_bet_ref_confirmed": str(bool(stored)),
                "pre_bet_ref_found": str(bool(stored)),
                "pre_bet_ref_missing": str(not bool(stored)),
                "pre_bet_ref_poll_attempts": attempts_by_key.get(result.processed_key, 0),
                "pre_bet_ref_poll_duration_ms": duration_ms,
                "pre_bet_ref_missing_retryable": str(bool(missing_retryable)),
                "pre_retry_count": retry_count,
                "pre_retry_allowed": str(bool(retry_allowed)),
                "pre_retry_reason": retry_reason,
                "pre_retry_block_reason": retry_block_reason,
                "pre_unconfirmed_reason": "" if stored else getattr(result, "pre_unconfirmed_reason", "") or "bet_ref_missing_after_poll",
            }
            if not stored:
                status = (
                    "PRE_LADDER_BET_REF_LATE_OR_MATCH_EVIDENCE"
                    if evidence_found
                    else ("PRE_LADDER_BET_REF_MISSING_RETRYABLE" if retry_allowed else "PRE_LADDER_BET_REF_MISSING")
                )
                reason = (
                    retry_block_reason
                    if evidence_found or not retry_allowed
                    else "pre_ladder_bet_ref_missing_retryable_after_poll"
                )
                per_key[result.processed_key].update(
                    {
                        "status": status,
                        "reason": reason,
                    }
                )

        self._update_attempt_log_rows(
            per_key.keys(),
            common_fields={
                "bet_ref_collection_phase_start": collection_start,
                "bet_ref_collection_phase_end": collection_end,
                "bet_ref_collection_duration_ms": duration_ms,
                "bet_ref_found_count": found_count,
                "bet_ref_missing_count": missing_count,
            },
            per_key_fields=per_key,
        )

    def _read_selection_bet_ref_candidates(
        self,
        sheet_name: str,
    ) -> tuple[list[_SelectionsBetRefCandidate], int, str]:
        try:
            if not self.bridge.has_sheet(sheet_name):
                return [], 0, ""
            values = self.bridge.read_sheet(sheet_name, rows=_selections_scan_rows(), columns=80)
        except Exception:
            return [], 0, ""
        header_index, columns = _detect_selection_columns(values)
        header_row = values[header_index] if header_index < len(values) else []
        header_text = _format_header_row(header_row)
        rows_scanned = 0
        candidates: list[_SelectionsBetRefCandidate] = []
        for row_number, row in enumerate(values[header_index + 1 :], start=header_index + 2):
            if not any(_clean_text(cell) for cell in row):
                continue
            rows_scanned += 1
            bet_ref = normalise_gruss_bet_ref(_value_at(row, columns.get("bet_ref")))
            runner = _clean_text(_value_at(row, columns.get("runner")))
            side = _normalise_selection_side(_value_at(row, columns.get("side")))
            stake = _positive_float_or_none(
                _first_non_empty(
                    _value_at(row, columns.get("req_stake")),
                    _value_at(row, columns.get("stake")),
                    _value_at(row, columns.get("amount")),
                )
            )
            req_odds = _positive_float_or_none(
                _first_non_empty(
                    _value_at(row, columns.get("req_odds")),
                    _value_at(row, columns.get("odds")),
                )
            )
            candidates.append(
                _SelectionsBetRefCandidate(
                    row_number=row_number,
                    runner=runner,
                    trap=extract_trap(runner),
                    selection_id=_normalise_identifier(_value_at(row, columns.get("selection_id"))),
                    side=side,
                    stake=stake,
                    req_odds=req_odds,
                    average_odds=_positive_float_or_none(_value_at(row, columns.get("average_odds"))),
                    result=_clean_text(_value_at(row, columns.get("result"))),
                    market_name=_clean_text(_value_at(row, columns.get("market_name"))),
                    market_id=_normalise_identifier(_value_at(row, columns.get("market_id"))),
                    market_type=_clean_text(_value_at(row, columns.get("market_type"))).upper(),
                    matched_odds=_positive_float_or_none(_value_at(row, columns.get("matched_odds"))),
                    matched_stake=_matched_stake_value(_value_at(row, columns.get("matched_stake"))),
                    bet_ref=bet_ref,
                    timestamp=_clean_text(_value_at(row, columns.get("timestamp"))),
                )
            )
        return candidates, rows_scanned, header_text

    def _dump_bet_ref_cells(
        self,
        candidates: Iterable[tuple[GrussRealOrderResult, _ActivePreLadderState]],
    ) -> str:
        parts: list[str] = []
        seen: set[tuple[str, str]] = set()
        for result, state in sorted(candidates, key=lambda item: (item[1].market_type, item[1].row)):
            if state.row <= 0:
                continue
            sheet_name = state.market_type or result.excel_sheet or "PLACE"
            address = self.layout.bet_ref_address(state.row)
            key = (sheet_name, address)
            if key in seen:
                continue
            seen.add(key)
            try:
                value = _clean_text(self.bridge.read_cell(sheet_name, address))
            except Exception as exc:
                value = f"read_failed:{exc}"
            parts.append(f"{sheet_name}!{address}={value}")
        return " ; ".join(parts)

    def _dump_runner_qz_cells(
        self,
        candidates: Iterable[tuple[GrussRealOrderResult, _ActivePreLadderState]],
    ) -> str:
        parts: list[str] = []
        seen: set[tuple[str, int]] = set()
        for result, state in sorted(candidates, key=lambda item: (item[1].market_type, item[1].row)):
            if state.row <= 0:
                continue
            sheet_name = state.market_type or result.excel_sheet or "PLACE"
            key = (sheet_name, state.row)
            if key in seen:
                continue
            seen.add(key)
            cells: list[str] = []
            for column in "QRSTUVWXYZ":
                address = f"{column}{state.row}"
                try:
                    value = _clean_text(self.bridge.read_cell(sheet_name, address))
                except Exception as exc:
                    value = f"read_failed:{exc}"
                cells.append(f"{address}={value}")
            parts.append(f"{sheet_name}!row{state.row}:" + ",".join(cells))
        return " ; ".join(parts)

    def reject_order(
        self,
        intent: OrderIntent,
        context: GrussRealOrderContext,
        reason: str,
    ) -> GrussRealOrderResult:
        self._refresh_safety_flags()
        return self._finish(intent, context, "REJECTED_REAL", reason)

    def place_order(
        self,
        intent: OrderIntent,
        context: GrussRealOrderContext,
    ) -> GrussRealOrderResult:
        self._refresh_safety_flags()
        self._release_active_ladder_if_post_context(intent, context)
        errors = self._preflight_errors(intent, context)
        if errors:
            return self._finish(intent, context, "REJECTED_REAL", "; ".join(errors))

        sheet_name = intent.market_type.upper()
        try:
            self.bridge.connect_open_workbook()
            if not self.bridge.is_workbook_visible():
                return self._finish(intent, context, "REJECTED_REAL", "excel_workbook_not_visible")
            for required_sheet in ("WIN", "PLACE"):
                if not self.bridge.has_sheet(required_sheet):
                    return self._finish(
                        intent,
                        context,
                        "REJECTED_REAL",
                        f"missing_excel_sheet={required_sheet}",
                        excel_sheet=sheet_name,
                    )
            market_error = self._current_market_error(intent, context)
            if market_error:
                return self._finish(
                    intent,
                    context,
                    "REJECTED_REAL",
                    market_error,
                    excel_sheet=sheet_name,
                )
            runner_row = self._find_runner_row(sheet_name, intent)
        except Exception as exc:
            mapping_result = self._last_mapping_com_result
            reason = (
                f"excel_mapping_unavailable_after_retries: {exc}"
                if mapping_result.exception is not None or "excel_mapping_unavailable_after_retries" in str(exc)
                else f"excel_unavailable: {exc}"
            )
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                reason,
                excel_sheet=sheet_name,
                excel_operation_name=mapping_result.operation_name,
                excel_com_attempt=mapping_result.attempt,
                excel_com_retry_count=mapping_result.retry_count,
                excel_com_retry_backoff_ms=mapping_result.retry_backoff_ms,
                excel_com_retryable_error=mapping_result.retryable_error,
                mapping_attempt_count=mapping_result.attempt,
            )

        if runner_row is None:
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                "runner_row_not_found",
                excel_sheet=sheet_name,
            )

        try:
            stake_value = float(intent.stake)
        except (TypeError, ValueError):
            stake_value = math.nan
        if not math.isfinite(stake_value) or stake_value <= 0.0:
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                "invalid_stake",
                excel_sheet=sheet_name,
                excel_row=runner_row,
            )

        plan = self._build_write_plan(intent, runner_row)
        if self.write_no_trigger_guard:
            try:
                preparation_plan = self._without_trigger(plan, runner_row)
            except Exception as exc:
                return self._finish(
                    intent,
                    context,
                    "REJECTED_REAL",
                    f"unsafe_preparation_layout: {exc}",
                    excel_sheet=sheet_name,
                    excel_row=runner_row,
                )
            trigger_address = self.layout.trigger_address(runner_row)
            trigger_mapping_name = self._trigger_for(intent)
            try:
                trigger_value = self._read_cell_with_retry(
                    sheet_name,
                    trigger_address,
                    f"read_trigger_cell:{sheet_name}!{trigger_address}",
                )
            except Exception as exc:
                return self._finish(
                    intent,
                    context,
                    "REJECTED_REAL",
                    f"trigger_cell_read_failed: {exc}",
                    excel_sheet=sheet_name,
                    excel_row=runner_row,
                    write_plan=preparation_plan,
                    trigger_cell_address=trigger_address,
                    trigger_mapping_name=trigger_mapping_name,
                )
            if trigger_value not in (None, ""):
                return self._finish(
                    intent,
                    context,
                    "REJECTED_REAL",
                    "trigger_cell_not_empty",
                    excel_sheet=sheet_name,
                    excel_row=runner_row,
                    write_plan=preparation_plan,
                    trigger_cell_address=trigger_address,
                    trigger_cell_current_value=trigger_value,
                    trigger_cell_expected_empty=False,
                    trigger_mapping_name=trigger_mapping_name,
                )
            try:
                written = tuple(
                    self.bridge.write_cells_without_trigger(
                        sheet_name,
                        preparation_plan,
                        trigger_address=trigger_address,
                        allow_write=True,
                    )
                )
            except Exception as exc:
                return self._finish(
                    intent,
                    context,
                    "REJECTED_REAL",
                    f"excel_write_failed: {exc}",
                    excel_sheet=sheet_name,
                    excel_row=runner_row,
                    write_plan=preparation_plan,
                    trigger_cell_address=trigger_address,
                    trigger_cell_current_value=trigger_value,
                    trigger_cell_expected_empty=True,
                    trigger_mapping_name=trigger_mapping_name,
                )
            return self._finish(
                intent,
                context,
                "GRUSS_WRITE_NO_TRIGGER",
                "no_trigger_written",
                excel_sheet=sheet_name,
                excel_row=runner_row,
                excel_cells_written=written,
                write_plan=preparation_plan,
                trigger_written=False,
                trigger_cell_address=trigger_address,
                trigger_cell_current_value=trigger_value,
                trigger_cell_expected_empty=True,
                trigger_mapping_name=trigger_mapping_name,
            )

        if self.preview:
            return self._finish(
                intent,
                context,
                "GRUSS_REAL_PREVIEW",
                "preview_only_no_excel_write",
                excel_sheet=sheet_name,
                excel_row=runner_row,
                write_plan=plan,
            )

        intent, stake_capped, stake_cap_value = self._cap_real_stake(intent)
        plan = self._build_write_plan(intent, runner_row)

        if _is_pre_ladder_intent(intent):
            return self._place_pre_ladder_order(
                intent,
                context,
                sheet_name,
                runner_row,
                stake_capped=stake_capped,
                stake_cap_value=stake_cap_value,
            )

        if not self.layout_confirmed:
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                "trigger_layout_not_confirmed",
                excel_sheet=sheet_name,
                excel_row=runner_row,
                write_plan=plan,
                stake_capped=stake_capped,
                stake_cap_value=stake_cap_value,
            )

        trigger_address = self.layout.trigger_address(runner_row)
        trigger_mapping_name = self._trigger_for(intent)
        try:
            trigger_value = self._read_cell_with_retry(
                sheet_name,
                trigger_address,
                f"read_trigger_cell:{sheet_name}!{trigger_address}",
            )
        except Exception as exc:
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                f"trigger_cell_read_failed: {exc}",
                excel_sheet=sheet_name,
                excel_row=runner_row,
                write_plan=plan,
                trigger_cell_address=trigger_address,
                trigger_mapping_name=trigger_mapping_name,
                stake_capped=stake_capped,
                stake_cap_value=stake_cap_value,
            )
        if trigger_value not in (None, ""):
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                "trigger_cell_not_empty",
                excel_sheet=sheet_name,
                excel_row=runner_row,
                write_plan=plan,
                trigger_cell_address=trigger_address,
                trigger_cell_current_value=trigger_value,
                trigger_cell_expected_empty=False,
                trigger_mapping_name=trigger_mapping_name,
                stake_capped=stake_capped,
                stake_cap_value=stake_cap_value,
            )

        is_post = _execution_phase(intent) == "POST"
        post_batch_defer_confirmation = bool(
            is_post and self._batch_log_context.get("post_batch_defer_confirmation")
        )
        bet_ref_address = self.layout.bet_ref_address(runner_row)
        post_existing_bet_ref_before = ""
        post_existing_matched_before = None
        post_existing_avg_odds_before = None
        post_expected_market_id = ""
        post_expected_market_type = ""
        post_expected_runner = ""
        post_expected_selection_id = ""
        post_expected_side = ""
        post_expected_stake = None
        post_expected_price = None
        post_write_timestamp = ""
        post_independent_mode_enabled = bool(
            is_post and _post_independent_mode_enabled(self._batch_log_context)
        )
        post_row_prepared_for_new_order = False
        post_pre_bet_ref_cleared_for_write = False
        post_pre_bet_ref_preserved_in_state = False
        post_new_bet_ref_expected = False
        post_new_bet_ref_found = False
        post_new_bet_ref = ""
        post_added_stake_confirmed = False
        post_added_stake_amount = None
        post_total_matched_before = None
        post_total_matched_after = None
        post_total_matched_delta = None
        if is_post:
            try:
                post_existing_bet_ref_before = normalise_gruss_bet_ref(
                    self._read_cell_with_retry(
                        sheet_name,
                        bet_ref_address,
                        f"post_read_existing_bet_ref_before_write:{sheet_name}!{bet_ref_address}",
                    )
                )
            except Exception:
                post_existing_bet_ref_before = ""
            post_existing_matched_before = _matched_stake_value(
                _read_cell_quiet(
                    self.bridge,
                    sheet_name,
                    self.layout.matched_stake_address(runner_row),
                )
            )
            post_existing_avg_odds_before = _positive_float_or_none(
                _read_cell_quiet(
                    self.bridge,
                    sheet_name,
                    self.layout.avg_matched_odds_address(runner_row),
                )
            )
            post_total_matched_before = post_existing_matched_before
            post_expected_market_id = _normalise_identifier(intent.market_id)
            post_expected_market_type = str(intent.market_type or sheet_name or "").upper()
            post_expected_runner = str(intent.runner_name or "")
            post_expected_selection_id = _normalise_identifier(
                intent.selection_id if intent.selection_id is not None else intent.trap
            )
            post_expected_side = str(intent.side or "").upper()
            post_expected_stake = _positive_float_or_none(intent.stake)
            post_expected_price = _price_to_write_for_gruss(intent, trigger_mapping_name)
            post_write_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            post_pre_bet_ref_preserved_in_state = is_valid_bet_ref(post_existing_bet_ref_before)
            post_pre_bet_ref_cleared_for_write = bool(
                post_independent_mode_enabled and is_valid_bet_ref(post_existing_bet_ref_before)
            )
            post_row_prepared_for_new_order = bool(post_independent_mode_enabled)
            post_new_bet_ref_expected = bool(_post_bet_ref_required())
            if post_pre_bet_ref_cleared_for_write:
                plan = self._build_write_plan(intent, runner_row, bet_ref_override="")

        write_result = self._write_cells_with_retry(sheet_name, plan)
        if write_result.exception is not None:
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                f"{write_result.final_status}: {write_result.exception}",
                excel_sheet=sheet_name,
                excel_row=runner_row,
                write_plan=plan,
                trigger_cell_address=trigger_address,
                trigger_cell_current_value=trigger_value,
                trigger_cell_expected_empty=True,
                trigger_mapping_name=trigger_mapping_name,
                stake_capped=stake_capped,
                stake_cap_value=stake_cap_value,
                excel_write_attempt=write_result.attempt,
                excel_write_retry_count=write_result.retry_count,
                excel_write_retry_backoff_ms=write_result.retry_backoff_ms,
                excel_write_final_status=write_result.final_status,
                excel_unavailable_recovered=write_result.recovered,
            )
        written = write_result.written

        verification = self._verify_real_write(sheet_name, runner_row, plan)
        trigger_written = _values_match(trigger_mapping_name, verification.trigger_value)
        post_bet_ref_required = bool(is_post and _post_bet_ref_required())
        post_clear_after_bet_ref = bool(is_post and _post_clear_after_bet_ref())
        post_bet_ref_wait_attempted = False
        post_bet_ref_wait_ms = _post_bet_ref_wait_ms() if post_bet_ref_required else 0
        post_bet_ref_poll_ms = _post_bet_ref_poll_ms() if post_bet_ref_required else 0
        post_bet_ref_after = ""
        post_bet_ref_changed = False
        post_bet_ref_confirmed_new = False
        post_bet_ref_poll_attempts = 0
        post_bet_ref_poll_duration_ms = 0
        post_order_confirmed = bool(is_post and not post_bet_ref_required and trigger_written)
        post_order_confirmation_source = ""
        post_selections_lookup_attempted = False
        post_selections_match_found = False
        post_selections_match_reason = ""
        post_selections_reject_reason = ""
        post_cells_clear_delay_ms = (
            _post_command_cells_clear_delay_ms()
            if is_post and post_clear_after_bet_ref
            else self.command_cells_clear_delay_ms
        )
        post_cells_cleared_after_confirmation = False
        post_cells_cleared_after_unconfirmed = False
        post_clear_reason = ""
        post_write_unconfirmed_reason = ""
        clear_outcome = _TriggerClearOutcome()
        should_defer_post_clear = bool(
            trigger_written
            and is_post
            and post_clear_after_bet_ref
            and post_bet_ref_required
            and not post_batch_defer_confirmation
        )
        if trigger_written and not should_defer_post_clear and not post_batch_defer_confirmation:
            clear_outcome = self._clear_written_trigger(
                sheet_name,
                trigger_address,
                trigger_mapping_name,
                hold_for_visual_test=self.hold_trigger_for_visual_test,
                processed_key=_processed_key(intent, context),
                expected_command_cell_values=plan,
                delay_override_ms=post_cells_clear_delay_ms if is_post and post_clear_after_bet_ref else None,
            )
        if (
            trigger_written
            and verification.verified
            and post_bet_ref_required
            and not post_batch_defer_confirmation
        ):
            post_bet_ref_wait_attempted = True
            poll_result = self._poll_post_bet_ref(
                sheet_name,
                bet_ref_address,
                existing_bet_ref_before=post_existing_bet_ref_before,
            )
            post_bet_ref_after = poll_result.bet_ref
            post_bet_ref_poll_attempts = poll_result.attempts
            post_bet_ref_poll_duration_ms = poll_result.duration_ms
            post_bet_ref_changed = bool(
                is_valid_bet_ref(poll_result.bet_ref)
                and (
                    not is_valid_bet_ref(post_existing_bet_ref_before)
                    or strip_gruss_ref_suffix(poll_result.bet_ref)
                    != strip_gruss_ref_suffix(post_existing_bet_ref_before)
                )
            )
            post_bet_ref_confirmed_new = bool(is_valid_bet_ref(poll_result.bet_ref) and post_bet_ref_changed)
            post_order_confirmed = post_bet_ref_confirmed_new
            if post_order_confirmed:
                post_order_confirmation_source = poll_result.lookup_source
            elif is_valid_bet_ref(poll_result.bet_ref) and is_valid_bet_ref(post_existing_bet_ref_before):
                post_write_unconfirmed_reason = "POST_BET_REF_NOT_NEW"
                post_order_confirmation_source = "post_existing_bet_ref_unchanged"
            else:
                post_write_unconfirmed_reason = "POST_BET_REF_NOT_READY"
            if not post_order_confirmed:
                selections_lookup = self._lookup_post_selection_confirmation(
                    sheet_name,
                    intent,
                    context,
                    post_write_timestamp=post_write_timestamp,
                    existing_bet_ref_before=post_existing_bet_ref_before,
                    expected_market_id=post_expected_market_id,
                    expected_market_type=post_expected_market_type,
                    expected_runner=post_expected_runner,
                    expected_selection_id=post_expected_selection_id,
                    expected_side=post_expected_side,
                    expected_stake=post_expected_stake,
                    expected_price=post_expected_price,
                )
                post_selections_lookup_attempted = selections_lookup.attempted
                post_selections_match_found = selections_lookup.match is not None
                post_selections_match_reason = selections_lookup.match_reason
                post_selections_reject_reason = selections_lookup.reject_reason
                if selections_lookup.match is not None:
                    post_bet_ref_after = selections_lookup.match.bet_ref
                    post_bet_ref_changed = True
                    post_bet_ref_confirmed_new = True
                    post_order_confirmed = True
                    post_order_confirmation_source = selections_lookup.lookup_source
                    post_write_unconfirmed_reason = ""
            post_total_matched_after = _matched_stake_value(
                _read_cell_quiet(
                    self.bridge,
                    sheet_name,
                    self.layout.matched_stake_address(runner_row),
                )
            )
            post_total_matched_delta = _matched_stake_delta(
                post_total_matched_before,
                post_total_matched_after,
            )
            post_added_stake_confirmed = bool(
                post_total_matched_delta is not None and post_total_matched_delta > 0
            )
            if post_added_stake_confirmed:
                post_added_stake_amount = post_total_matched_delta
                if not post_order_confirmed:
                    post_order_confirmed = True
                    post_order_confirmation_source = "post_matched_stake_delta"
                    post_write_unconfirmed_reason = ""
            post_new_bet_ref_found = bool(post_bet_ref_confirmed_new)
            post_new_bet_ref = post_bet_ref_after if post_new_bet_ref_found else ""
            if not post_order_confirmed:
                if post_write_unconfirmed_reason == "POST_BET_REF_NOT_NEW":
                    post_write_unconfirmed_reason = "POST_BET_REF_NOT_NEW_AND_NO_STAKE_DELTA"
                else:
                    post_write_unconfirmed_reason = "POST_WRITE_ATTEMPTED_BUT_NO_NEW_ORDER_EVIDENCE"
        if trigger_written and should_defer_post_clear:
            clear_outcome = self._clear_written_trigger(
                sheet_name,
                trigger_address,
                trigger_mapping_name,
                hold_for_visual_test=self.hold_trigger_for_visual_test,
                processed_key=_processed_key(intent, context),
                expected_command_cell_values=plan,
                delay_override_ms=post_cells_clear_delay_ms,
            )
            post_cleared_after_poll = bool(
                clear_outcome.attempted and (clear_outcome.cleared or clear_outcome.scheduled)
            )
            if post_order_confirmed:
                post_cells_cleared_after_confirmation = post_cleared_after_poll
                post_clear_reason = (
                    "confirmed_post_cleanup"
                    if post_cleared_after_poll
                    else clear_outcome.command_reason or clear_outcome.reason
                )
            else:
                post_cells_cleared_after_unconfirmed = post_cleared_after_poll
                post_clear_reason = (
                    "unconfirmed_post_cleanup"
                    if post_cleared_after_poll
                    else clear_outcome.command_reason or clear_outcome.reason
                )
        finish_kwargs = {
            "excel_sheet": sheet_name,
            "excel_row": runner_row,
            "excel_cells_written": written,
            "write_plan": plan,
            "trigger_written": trigger_written,
            "trigger_cell_address": trigger_address,
            "trigger_cell_current_value": trigger_value,
            "trigger_cell_expected_empty": True,
            "trigger_mapping_name": trigger_mapping_name,
            "trigger_value_written": trigger_mapping_name if trigger_written else "",
            "trigger_clear_attempted": clear_outcome.attempted,
            "trigger_cleared": clear_outcome.cleared,
            "trigger_clear_reason": clear_outcome.reason,
            "trigger_cell_value_before_clear": clear_outcome.value_before_clear,
            "trigger_clear_delay_ms": clear_outcome.delay_ms,
            "command_cells_clear_attempted": clear_outcome.attempted,
            "command_cells_cleared": clear_outcome.cleared,
            "command_cells_clear_reason": clear_outcome.command_reason,
            "command_cells_clear_addresses": ";".join(clear_outcome.addresses),
            "command_cells_clear_delay_ms": clear_outcome.delay_ms,
            "command_cells_clear_scheduled": clear_outcome.scheduled,
            "command_cells_clear_due_time": clear_outcome.due_time,
            "command_cells_clear_non_blocking": clear_outcome.non_blocking,
            "command_cells_clear_executed": clear_outcome.executed,
            "command_cells_clear_lag_ms": clear_outcome.lag_ms,
            "post_write_verification": verification,
            "hold_trigger_for_visual_test": self.hold_trigger_for_visual_test,
            "stake_capped": stake_capped,
            "stake_cap_value": stake_cap_value,
            "bet_ref_after": post_bet_ref_after,
            "bet_ref_poll_attempts": post_bet_ref_poll_attempts,
            "bet_ref_poll_duration_ms": post_bet_ref_poll_duration_ms,
            "bet_ref_lookup_source": post_order_confirmation_source,
            "post_bet_ref_required": post_bet_ref_required,
            "post_batch_id": str(self._batch_log_context.get("post_batch_id") or ""),
            "post_batch_market_id": str(self._batch_log_context.get("post_batch_market_id") or ""),
            "post_batch_market_name": str(self._batch_log_context.get("post_batch_market_name") or ""),
            "post_batch_candidate_count": _int_from_context(self._batch_log_context.get("post_batch_candidate_count")),
            "post_batch_written_count": _int_from_context(self._batch_log_context.get("post_batch_written_count")),
            "post_batch_write_duration_ms": _int_from_context(self._batch_log_context.get("post_batch_write_duration_ms")),
            "post_batch_confirmation_started": bool(self._batch_log_context.get("post_batch_confirmation_started")),
            "post_batch_confirmation_duration_ms": _int_from_context(
                self._batch_log_context.get("post_batch_confirmation_duration_ms")
            ),
            "post_batch_runner_index": _int_from_context(self._batch_log_context.get("post_batch_runner_index")),
            "post_batch_total_runners": _int_from_context(self._batch_log_context.get("post_batch_total_runners")),
            "post_bet_ref_wait_attempted": post_bet_ref_wait_attempted,
            "post_bet_ref_wait_ms": post_bet_ref_wait_ms,
            "post_bet_ref_poll_ms": post_bet_ref_poll_ms,
            "post_existing_bet_ref_before": post_existing_bet_ref_before,
            "post_existing_pre_bet_ref": post_existing_bet_ref_before,
            "post_existing_matched_before": post_existing_matched_before,
            "post_existing_pre_matched_stake": post_existing_matched_before,
            "post_existing_avg_odds_before": post_existing_avg_odds_before,
            "post_existing_pre_avg_odds": post_existing_avg_odds_before,
            "post_independent_mode_enabled": post_independent_mode_enabled,
            "post_row_prepared_for_new_order": post_row_prepared_for_new_order,
            "post_pre_bet_ref_cleared_for_write": post_pre_bet_ref_cleared_for_write,
            "post_pre_bet_ref_preserved_in_state": post_pre_bet_ref_preserved_in_state,
            "post_new_bet_ref_expected": post_new_bet_ref_expected,
            "post_new_bet_ref_found": post_new_bet_ref_found,
            "post_new_bet_ref": post_new_bet_ref,
            "post_added_stake_confirmed": post_added_stake_confirmed,
            "post_added_stake_amount": post_added_stake_amount,
            "post_total_matched_before": post_total_matched_before,
            "post_total_matched_after": post_total_matched_after,
            "post_total_matched_delta": post_total_matched_delta,
            "post_expected_market_id": post_expected_market_id,
            "post_expected_market_type": post_expected_market_type,
            "post_expected_runner": post_expected_runner,
            "post_expected_selection_id": post_expected_selection_id,
            "post_expected_side": post_expected_side,
            "post_expected_stake": post_expected_stake,
            "post_expected_price": post_expected_price,
            "post_write_timestamp": post_write_timestamp,
            "post_bet_ref_after": post_bet_ref_after,
            "post_bet_ref_changed": post_bet_ref_changed,
            "post_bet_ref_confirmed_new": post_bet_ref_confirmed_new,
            "post_bet_ref_poll_attempts": post_bet_ref_poll_attempts,
            "post_bet_ref_poll_duration_ms": post_bet_ref_poll_duration_ms,
            "post_order_confirmed": post_order_confirmed,
            "post_order_confirmation_source": post_order_confirmation_source,
            "post_confirmation_source": post_order_confirmation_source,
            "post_selections_lookup_attempted": post_selections_lookup_attempted,
            "post_selections_match_found": post_selections_match_found,
            "post_selections_match_reason": post_selections_match_reason,
            "post_selections_reject_reason": post_selections_reject_reason,
            "post_clear_after_bet_ref": post_clear_after_bet_ref,
            "post_cells_clear_delay_ms": post_cells_clear_delay_ms,
            "post_cells_cleared_after_confirmation": post_cells_cleared_after_confirmation,
            "post_cells_cleared_after_unconfirmed": post_cells_cleared_after_unconfirmed,
            "post_clear_reason": post_clear_reason,
            "post_write_unconfirmed_reason": post_write_unconfirmed_reason,
            "post_unconfirmed_reason": post_write_unconfirmed_reason,
            "post_reject_reason": "" if post_order_confirmed else post_write_unconfirmed_reason,
            "excel_write_attempt": write_result.attempt,
            "excel_write_retry_count": write_result.retry_count,
            "excel_write_retry_backoff_ms": write_result.retry_backoff_ms,
            "excel_write_final_status": write_result.final_status,
            "excel_unavailable_recovered": write_result.recovered,
        }
        if not verification.verified:
            return self._finish(
                intent,
                context,
                "GRUSS_WRITE_FAILED",
                "post_write_verification_failed",
                **finish_kwargs,
            )

        if post_batch_defer_confirmation and post_bet_ref_required and trigger_written:
            return self._finish(
                intent,
                context,
                "POST_WRITE_PENDING_CONFIRMATION",
                "post_batch_written_pending_confirmation",
                **finish_kwargs,
            )

        if post_bet_ref_required and not post_order_confirmed:
            unconfirmed_status = (
                "POST_WRITE_UNCONFIRMED_EXISTING_PRE_BETREF"
                if post_write_unconfirmed_reason == "POST_BET_REF_NOT_NEW_AND_NO_STAKE_DELTA"
                else "POST_WRITE_UNCONFIRMED"
            )
            return self._finish(
                intent,
                context,
                unconfirmed_status,
                post_write_unconfirmed_reason or "POST_BET_REF_NOT_READY",
                **finish_kwargs,
            )

        real_key = _processed_key(intent, context)
        max_key = _max_orders_key(intent, context)
        self.real_order_counts[real_key] = self.real_order_counts.get(real_key, 0) + 1
        self.real_order_counts[max_key] = self.real_order_counts.get(max_key, 0) + 1
        return self._finish(
            intent,
            context,
            "GRUSS_REAL_WRITTEN",
            "excel_trigger_written",
            **finish_kwargs,
        )

    def _place_pre_ladder_order(
        self,
        intent: OrderIntent,
        context: GrussRealOrderContext,
        sheet_name: str,
        runner_row: int,
        *,
        stake_capped: bool,
        stake_cap_value: float | None,
    ) -> GrussRealOrderResult:
        plan = self._build_write_plan(intent, runner_row)
        trigger_address = self.layout.trigger_address(runner_row)
        bet_ref_address = self.layout.bet_ref_address(runner_row)
        matched_stake_address = self.layout.matched_stake_address(runner_row)
        ladder_id = str(intent.ladder_id or intent.ladder_tracking_key or "").strip()
        step_index = _ladder_step_index(intent.ladder_step)
        course_key = _pre_ladder_course_key(intent, context)
        bet_ref_before = ""
        bet_ref_after = ""
        bet_ref_status_value = ""
        matched_stake_cell_value: Any = None
        replace_allowed = False
        replace_trigger = ""
        bet_ref_suffix_n_handled = False
        replace_bet_ref_wait_attempted = False
        replace_bet_ref_wait_ms = 0
        replace_bet_ref_poll_ms = 0
        replace_bet_ref_wait_result = ""
        bet_ref_before_wait = ""
        bet_ref_after_wait = ""
        active_ladder_bet_ref_updated = False
        replace_skipped_bet_ref_still_pending = False
        pre_retry_allowed = False
        pre_retry_reason = ""
        pre_retry_count = _pre_retry_count_for_intent(self, intent)
        pre_retry_block_reason = ""
        pre_bet_ref_late_detected = False
        pre_bet_ref_late_value = ""
        active_ladder_created = False
        pending_ladder_created = False
        matched_evidence_found = False
        selection_row_evidence_found = False
        no_stacking_blocked_retry = False

        if not self.layout_confirmed:
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                "trigger_layout_not_confirmed",
                excel_sheet=sheet_name,
                excel_row=runner_row,
                write_plan=plan,
                stake_capped=stake_capped,
                stake_cap_value=stake_cap_value,
                update_skipped_reason="trigger_layout_not_confirmed",
            )
        if not ladder_id:
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                "missing_pre_ladder_id",
                excel_sheet=sheet_name,
                excel_row=runner_row,
                write_plan=plan,
                stake_capped=stake_capped,
                stake_cap_value=stake_cap_value,
                update_skipped_reason="missing_pre_ladder_id",
            )
        previous_course_released = False
        self._sync_legacy_active_ladder_state()
        if (
            self.active_pre_ladders
            and all(state.course_key != course_key for state in self.active_pre_ladders.values())
        ):
            self.active_pre_ladders.clear()
            self.active_pre_ladder_id = None
            self.active_pre_ladder_course = None
            previous_course_released = True

        active_count = len(self.active_pre_ladders)
        max_ladders = _pre_ladder_real_max_ladders()
        if ladder_id not in self.active_pre_ladders and active_count >= max_ladders:
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                "max_active_pre_ladder_reached",
                excel_sheet=sheet_name,
                excel_row=runner_row,
                write_plan=plan,
                stake_capped=stake_capped,
                stake_cap_value=stake_cap_value,
                update_skipped_reason="max_active_pre_ladder_reached",
            )

        lookup_source = "excel_row"
        lookup_matched_runner = ""
        stored_state = self.active_pre_ladders.get(ladder_id)
        if step_index > 0 and stored_state is None:
            try:
                row_bet_ref = normalise_gruss_bet_ref(
                    self._read_cell_with_retry(
                        sheet_name,
                        bet_ref_address,
                        f"read_bet_ref_for_initial_retry:{sheet_name}!{bet_ref_address}",
                    )
                )
            except Exception as exc:
                return self._finish(
                    intent,
                    context,
                    "GRUSS_PRE_LADDER_REPLACE_SKIPPED",
                    f"bet_ref_read_failed: {exc}",
                    excel_sheet=sheet_name,
                    excel_row=runner_row,
                    write_plan=(),
                    trigger_cell_address=trigger_address,
                    trigger_mapping_name=self.layout.replace_trigger_name(intent.side),
                    stake_capped=stake_capped,
                    stake_cap_value=stake_cap_value,
                    bet_ref_before="",
                    bet_ref_after="",
                    update_allowed=False,
                    update_skipped_reason="bet_ref_read_failed",
                    bet_ref_lookup_source="active_ladder_state_missing_bet_ref_read_failed",
                    bet_ref_lookup_matched_runner="False",
                    no_stacking_check_passed=True,
                )
            state_probe = _active_pre_ladder_state_from_intent(intent, course_key, runner_row)
            if is_valid_bet_ref(row_bet_ref):
                stored_state = _active_pre_ladder_state_from_intent(
                    intent,
                    course_key,
                    runner_row,
                    bet_ref=row_bet_ref,
                    pending_confirmation=False,
                )
                self.active_pre_ladders[ladder_id] = stored_state
                self.active_pre_ladder_id = ladder_id
                self.active_pre_ladder_course = course_key
                self.pre_bet_ref_missing_retry_counts.pop(_pre_retry_key_for_intent(intent), None)
                pre_retry_count = 0
                bet_ref_before = row_bet_ref
                lookup_source = "active_ladder_state_missing_row_t_late_attached"
                lookup_matched_runner = "True"
                pre_bet_ref_late_detected = True
                pre_bet_ref_late_value = row_bet_ref
                active_ladder_created = True
            elif row_bet_ref:
                if is_pending_bet_ref_status(row_bet_ref):
                    skip_reason = "bet_ref_not_ready"
                elif is_terminal_bet_status(row_bet_ref):
                    skip_reason = "row_status_not_replaceable"
                else:
                    skip_reason = "invalid_bet_ref_for_replace"
                pre_retry_block_reason = skip_reason
                no_stacking_blocked_retry = True
                return self._finish(
                    intent,
                    context,
                    "GRUSS_PRE_LADDER_REPLACE_SKIPPED",
                    skip_reason,
                    excel_sheet=sheet_name,
                    excel_row=runner_row,
                    write_plan=(),
                    trigger_cell_address=trigger_address,
                    trigger_mapping_name=self.layout.replace_trigger_name(intent.side),
                    stake_capped=stake_capped,
                    stake_cap_value=stake_cap_value,
                    bet_ref_before=row_bet_ref,
                    bet_ref_after="",
                    update_allowed=False,
                    update_skipped_reason=skip_reason,
                    bet_ref_lookup_source="active_ladder_state_missing_row_t_present",
                    bet_ref_lookup_matched_runner="False",
                    no_stacking_check_passed=True,
                    pre_retry_count=pre_retry_count,
                    pre_retry_block_reason=pre_retry_block_reason,
                    no_stacking_blocked_retry=no_stacking_blocked_retry,
                )
            else:
                matched_stake_cell_value = _read_cell_quiet(self.bridge, sheet_name, matched_stake_address)
                avg_matched_odds_cell_value = _read_cell_quiet(
                    self.bridge,
                    sheet_name,
                    self.layout.avg_matched_odds_address(runner_row),
                )
                matched_stake_value = _matched_stake_value(matched_stake_cell_value)
                avg_matched_odds_value = _positive_float_or_none(avg_matched_odds_cell_value)
                matched_evidence_found = bool(
                    (matched_stake_value is not None and matched_stake_value > 0)
                    or avg_matched_odds_value is not None
                )
                if not matched_evidence_found and "SELECTIONS_SHEET" in _bet_ref_lookup_sources_from_env():
                    selections_sheet = f"{(state_probe.market_type or sheet_name or 'PLACE').upper()}_Selections"
                    selection_candidates, _rows_scanned, _selection_headers = self._read_selection_bet_ref_candidates(
                        selections_sheet
                    )
                    used_bet_refs = {
                        state.bet_ref
                        for state in self.active_pre_ladders.values()
                        if is_valid_bet_ref(state.bet_ref)
                    }
                    match, match_reason, _top_candidates = _match_selection_bet_ref_candidate(
                        state_probe,
                        GrussRealOrderResult(
                            status="",
                            reason="",
                            output_path=self.output_path,
                            write_plan=tuple(plan),
                            stake_original=intent.stake_original if intent.stake_original is not None else intent.stake,
                            stake_used=intent.stake,
                        ),
                        selection_candidates,
                        used_bet_refs=used_bet_refs,
                    )
                    if match is not None:
                        row_bet_ref = match.bet_ref
                        stored_state = _active_pre_ladder_state_from_intent(
                            intent,
                            course_key,
                            runner_row,
                            bet_ref=row_bet_ref,
                            pending_confirmation=False,
                        )
                        self.active_pre_ladders[ladder_id] = stored_state
                        self.active_pre_ladder_id = ladder_id
                        self.active_pre_ladder_course = course_key
                        self.pre_bet_ref_missing_retry_counts.pop(_pre_retry_key_for_intent(intent), None)
                        pre_retry_count = 0
                        bet_ref_before = row_bet_ref
                        lookup_source = f"active_ladder_state_missing_selections_late_attached:{selections_sheet}!row{match.row_number}"
                        lookup_matched_runner = "True"
                        pre_bet_ref_late_detected = True
                        pre_bet_ref_late_value = row_bet_ref
                        active_ladder_created = True
                        selection_row_evidence_found = True
                    else:
                        evidence_match, evidence_reason = _match_selection_activity_evidence_candidate(
                            state_probe,
                            GrussRealOrderResult(
                                status="",
                                reason="",
                                output_path=self.output_path,
                                write_plan=tuple(plan),
                                stake_original=(
                                    intent.stake_original if intent.stake_original is not None else intent.stake
                                ),
                                stake_used=intent.stake,
                            ),
                            selection_candidates,
                        )
                        if evidence_match is not None:
                            selection_row_evidence_found = True
                            pre_retry_block_reason = "selection_row_evidence_found"
                            lookup_source = f"active_ladder_state_missing_selections_evidence:{selections_sheet}:{evidence_reason or match_reason}"
                if not bet_ref_before and (matched_evidence_found or selection_row_evidence_found):
                    if matched_evidence_found:
                        pre_retry_block_reason = pre_retry_block_reason or "matched_evidence_found"
                    pending_state = _active_pre_ladder_state_from_intent(
                        intent,
                        course_key,
                        runner_row,
                        pending_confirmation=True,
                    )
                    self.active_pre_ladders[ladder_id] = pending_state
                    self.active_pre_ladder_id = ladder_id
                    self.active_pre_ladder_course = course_key
                    pending_ladder_created = True
                    no_stacking_blocked_retry = True
                    return self._finish(
                        intent,
                        context,
                        "PRE_LADDER_BET_REF_LATE_OR_MATCH_EVIDENCE",
                        pre_retry_block_reason,
                        excel_sheet=sheet_name,
                        excel_row=runner_row,
                        write_plan=(),
                        trigger_cell_address=trigger_address,
                        trigger_mapping_name=self.layout.replace_trigger_name(intent.side),
                        stake_capped=stake_capped,
                        stake_cap_value=stake_cap_value,
                        bet_ref_before="",
                        bet_ref_after="",
                        update_allowed=False,
                        update_skipped_reason=pre_retry_block_reason,
                        action="PRE_LADDER_BET_REF_MISSING",
                        pre_bet_ref_required=True,
                        pre_bet_ref_confirmed=False,
                        pre_bet_ref_missing=True,
                        pre_retry_count=pre_retry_count,
                        pre_retry_allowed=False,
                        pre_retry_block_reason=pre_retry_block_reason,
                        pre_unconfirmed_reason="bet_ref_missing_with_order_evidence",
                        bet_ref_lookup_source=lookup_source or "active_ladder_state_missing_order_evidence",
                        bet_ref_lookup_matched_runner="False",
                        active_ladder_bet_ref_stored=False,
                        pending_ladder_created=pending_ladder_created,
                        matched_evidence_found=matched_evidence_found,
                        selection_row_evidence_found=selection_row_evidence_found,
                        no_stacking_blocked_retry=no_stacking_blocked_retry,
                        matched_stake_cell_address=matched_stake_address,
                        matched_stake_cell_value=matched_stake_cell_value,
                        no_stacking_check_passed=True,
                    )
                if not bet_ref_before:
                    max_retries = _pre_bet_ref_missing_max_retries()
                    if pre_retry_count >= max_retries:
                        pre_retry_block_reason = "pre_bet_ref_missing_retry_limit_reached"
                        return self._finish(
                            intent,
                            context,
                            "GRUSS_PRE_LADDER_REPLACE_SKIPPED",
                            pre_retry_block_reason,
                            excel_sheet=sheet_name,
                            excel_row=runner_row,
                            write_plan=(),
                            trigger_cell_address=trigger_address,
                            trigger_mapping_name=self.layout.trigger_mapping_name(intent.side, intent.order_type),
                            stake_capped=stake_capped,
                            stake_cap_value=stake_cap_value,
                            bet_ref_before="",
                            bet_ref_after="",
                            update_allowed=False,
                            update_skipped_reason=pre_retry_block_reason,
                            pre_retry_count=pre_retry_count,
                            pre_retry_allowed=False,
                            pre_retry_block_reason=pre_retry_block_reason,
                            bet_ref_lookup_source="active_ladder_state_missing_retry_limit_reached",
                            bet_ref_lookup_matched_runner="False",
                            no_stacking_check_passed=True,
                        )
                    pre_retry_count += 1
                    self.pre_bet_ref_missing_retry_counts[_pre_retry_key_for_intent(intent)] = pre_retry_count
                    pre_retry_allowed = True
                    pre_retry_reason = "missing_bet_ref_retry_initial_at_next_pre_step"
                    lookup_source = "active_ladder_state_missing_row_t_empty_retry_initial"
                    lookup_matched_runner = "False"
        effective_step_index = 0 if pre_retry_allowed else step_index
        if effective_step_index > 0 and stored_state is not None:
            if _active_ladder_state_matches(stored_state, intent, runner_row):
                bet_ref_before = normalise_gruss_bet_ref(stored_state.bet_ref)
                if not pre_bet_ref_late_detected:
                    lookup_source = "active_ladder_state" if bet_ref_before else "active_ladder_state_empty"
                    lookup_matched_runner = str(is_valid_bet_ref(bet_ref_before))
                if is_pending_bet_ref_status(bet_ref_before):
                    wait_result = self._wait_for_replace_bet_ref(sheet_name, bet_ref_address)
                    replace_bet_ref_wait_attempted = True
                    replace_bet_ref_wait_ms = wait_result.wait_ms
                    replace_bet_ref_poll_ms = wait_result.poll_ms
                    replace_bet_ref_wait_result = wait_result.result
                    bet_ref_before_wait = wait_result.before_wait or bet_ref_before
                    bet_ref_after_wait = wait_result.after_wait
                    if is_valid_bet_ref(wait_result.bet_ref):
                        bet_ref_before = wait_result.bet_ref
                        stored_state.bet_ref = wait_result.bet_ref
                        active_ladder_bet_ref_updated = True
                        lookup_source = "active_ladder_state_pending_resolved"
                        lookup_matched_runner = "True"
                    else:
                        bet_ref_before = wait_result.after_wait or bet_ref_before
            else:
                return self._finish(
                    intent,
                    context,
                    "GRUSS_PRE_LADDER_REPLACE_SKIPPED",
                    "active_ladder_runner_mismatch_do_not_replace",
                    excel_sheet=sheet_name,
                    excel_row=runner_row,
                    write_plan=(),
                    trigger_cell_address=trigger_address,
                    trigger_mapping_name=self.layout.replace_trigger_name(intent.side),
                    stake_capped=stake_capped,
                    stake_cap_value=stake_cap_value,
                    bet_ref_before="",
                    bet_ref_after="",
                    update_allowed=False,
                    update_skipped_reason="active_ladder_runner_mismatch_do_not_replace",
                    bet_ref_lookup_source="active_ladder_state_mismatch",
                    bet_ref_lookup_matched_runner="False",
                )
        if not bet_ref_before:
            try:
                bet_ref_before = normalise_gruss_bet_ref(
                    self._read_cell_with_retry(sheet_name, bet_ref_address, f"read_bet_ref:{sheet_name}!{bet_ref_address}")
                )
                if (
                    is_valid_bet_ref(bet_ref_before)
                    and stored_state is not None
                    and _active_ladder_state_matches(stored_state, intent, runner_row)
                ):
                    stored_state.bet_ref = bet_ref_before
                lookup_source = "excel_row" if bet_ref_before else lookup_source
                lookup_matched_runner = str(is_valid_bet_ref(bet_ref_before)) if bet_ref_before else lookup_matched_runner
            except Exception as exc:
                return self._finish(
                    intent,
                    context,
                    "REJECTED_REAL",
                    f"bet_ref_read_failed: {exc}",
                    excel_sheet=sheet_name,
                    excel_row=runner_row,
                    write_plan=plan,
                    stake_capped=stake_capped,
                    stake_cap_value=stake_cap_value,
                    update_skipped_reason="bet_ref_read_failed",
                    bet_ref_lookup_source=lookup_source,
                    bet_ref_lookup_matched_runner=lookup_matched_runner,
                )

        if pre_retry_allowed and bet_ref_before:
            if is_valid_bet_ref(bet_ref_before):
                stored_state = _active_pre_ladder_state_from_intent(
                    intent,
                    course_key,
                    runner_row,
                    bet_ref=bet_ref_before,
                    pending_confirmation=False,
                )
                self.active_pre_ladders[ladder_id] = stored_state
                self.active_pre_ladder_id = ladder_id
                self.active_pre_ladder_course = course_key
                self.pre_bet_ref_missing_retry_counts.pop(_pre_retry_key_for_intent(intent), None)
                pre_retry_allowed = False
                pre_retry_reason = ""
                pre_retry_count = 0
                pre_retry_block_reason = "late_bet_ref_attached_before_retry"
                pre_bet_ref_late_detected = True
                pre_bet_ref_late_value = bet_ref_before
                active_ladder_created = True
                lookup_source = "pre_retry_row_t_late_attached_before_write"
                lookup_matched_runner = "True"
                effective_step_index = step_index
            elif is_pending_bet_ref_status(bet_ref_before):
                retry_skip_reason = "bet_ref_not_ready"
                retry_lookup_matched_runner = "False"
            elif is_terminal_bet_status(bet_ref_before):
                retry_skip_reason = "row_status_not_replaceable"
                retry_lookup_matched_runner = "False"
            else:
                retry_skip_reason = "invalid_bet_ref_for_replace"
                retry_lookup_matched_runner = "False"
            if not is_valid_bet_ref(bet_ref_before):
                pre_retry_block_reason = retry_skip_reason
                no_stacking_blocked_retry = True
                return self._finish(
                    intent,
                    context,
                    "GRUSS_PRE_LADDER_REPLACE_SKIPPED",
                    retry_skip_reason,
                    excel_sheet=sheet_name,
                    excel_row=runner_row,
                    write_plan=(),
                    trigger_cell_address=trigger_address,
                    trigger_mapping_name=self.layout.trigger_mapping_name(intent.side, intent.order_type),
                    stake_capped=stake_capped,
                    stake_cap_value=stake_cap_value,
                    bet_ref_before=bet_ref_before,
                    bet_ref_after="",
                    update_allowed=False,
                    update_skipped_reason=retry_skip_reason,
                    bet_ref_lookup_source="pre_retry_row_t_changed_before_write",
                    bet_ref_lookup_matched_runner=retry_lookup_matched_runner,
                    no_stacking_check_passed=True,
                    pre_retry_count=pre_retry_count,
                    pre_retry_allowed=False,
                    pre_retry_reason="retry_blocked_bet_ref_present",
                    pre_retry_block_reason=pre_retry_block_reason,
                    no_stacking_blocked_retry=no_stacking_blocked_retry,
                )

        if effective_step_index > 0:
            replace_trigger = self.layout.replace_trigger_name(intent.side)
            try:
                market_status_for_replace = _clean_text(
                    self._read_cell_with_retry(sheet_name, "F2", f"read_market_status:{sheet_name}!F2")
                )
            except Exception:
                market_status_for_replace = ""
            if _is_untradable_market_status(market_status_for_replace):
                replace_reason = "market_suspended_no_replace"
                return self._finish(
                    intent,
                    context,
                    "GRUSS_PRE_LADDER_REPLACE_SKIPPED",
                    replace_reason,
                    excel_sheet=sheet_name,
                    excel_row=runner_row,
                    write_plan=(),
                    trigger_cell_address=trigger_address,
                    trigger_mapping_name=replace_trigger,
                    stake_capped=stake_capped,
                    stake_cap_value=stake_cap_value,
                    bet_ref_before=bet_ref_before,
                    update_allowed=False,
                    update_skipped_reason=replace_reason,
                    bet_ref_lookup_source=lookup_source,
                    bet_ref_lookup_matched_runner=lookup_matched_runner,
                    no_stacking_check_passed=True,
                )
            countdown_for_replace = self._current_countdown_seconds(sheet_name)
            if countdown_for_replace is None:
                return self._finish(
                    intent,
                    context,
                    "GRUSS_PRE_LADDER_REPLACE_SKIPPED",
                    "countdown_unavailable_no_replace",
                    excel_sheet=sheet_name,
                    excel_row=runner_row,
                    write_plan=(),
                    trigger_cell_address=trigger_address,
                    trigger_mapping_name=replace_trigger,
                    stake_capped=stake_capped,
                    stake_cap_value=stake_cap_value,
                    bet_ref_before=bet_ref_before,
                    update_allowed=False,
                    update_skipped_reason="countdown_unavailable_no_replace",
                    bet_ref_lookup_source=lookup_source,
                    bet_ref_lookup_matched_runner=lookup_matched_runner,
                    no_stacking_check_passed=True,
                )
            replace_min_countdown = _replace_min_countdown_seconds()
            if countdown_for_replace <= replace_min_countdown:
                return self._finish(
                    intent,
                    context,
                    "GRUSS_PRE_LADDER_REPLACE_SKIPPED",
                    "countdown_too_low_no_replace",
                    excel_sheet=sheet_name,
                    excel_row=runner_row,
                    write_plan=(),
                    trigger_cell_address=trigger_address,
                    trigger_mapping_name=replace_trigger,
                    stake_capped=stake_capped,
                    stake_cap_value=stake_cap_value,
                    bet_ref_before=bet_ref_before,
                    update_allowed=False,
                    update_skipped_reason="countdown_too_low_no_replace",
                    bet_ref_lookup_source=lookup_source,
                    bet_ref_lookup_matched_runner=lookup_matched_runner,
                    countdown_at_write=countdown_for_replace,
                    no_stacking_check_passed=True,
                )
            bet_ref_status_value = bet_ref_before
            if bet_ref_before == "PENDINGR":
                skip_reason = "replace_skipped_bet_ref_still_pending"
                replace_skipped_bet_ref_still_pending = True
            elif bet_ref_before in {"", "PENDING"}:
                skip_reason = "bet_ref_not_ready"
            elif is_terminal_bet_status(bet_ref_before):
                skip_reason = "row_status_not_replaceable"
            elif not is_valid_bet_ref(bet_ref_before):
                skip_reason = "invalid_bet_ref_for_replace"
            else:
                try:
                    matched_stake_cell_value = self._read_cell_with_retry(
                        sheet_name,
                        matched_stake_address,
                        f"read_matched_stake:{sheet_name}!{matched_stake_address}",
                    )
                except Exception as exc:
                    return self._finish(
                        intent,
                        context,
                        "GRUSS_PRE_LADDER_REPLACE_SKIPPED",
                        f"matched_stake_read_failed: {exc}",
                        excel_sheet=sheet_name,
                        excel_row=runner_row,
                        write_plan=(),
                        trigger_cell_address=trigger_address,
                        trigger_mapping_name=replace_trigger,
                        stake_capped=stake_capped,
                        stake_cap_value=stake_cap_value,
                        bet_ref_before=bet_ref_before,
                        bet_ref_after=bet_ref_after,
                        update_allowed=False,
                        update_skipped_reason="matched_stake_read_failed",
                        bet_ref_lookup_source=lookup_source,
                        bet_ref_lookup_matched_runner=lookup_matched_runner,
                        matched_stake_cell_address=matched_stake_address,
                        matched_stake_cell_value=matched_stake_cell_value,
                    )
                matched_stake_value = _matched_stake_value(matched_stake_cell_value)
                if matched_stake_value is None:
                    skip_reason = "matched_stake_unavailable_no_replace"
                elif matched_stake_value > 0:
                    skip_reason = "matched_stake_positive_no_replace"
                else:
                    skip_reason = ""
                    replace_allowed = True
            if skip_reason:
                return self._finish(
                    intent,
                    context,
                    "GRUSS_PRE_LADDER_REPLACE_SKIPPED",
                    skip_reason,
                    excel_sheet=sheet_name,
                    excel_row=runner_row,
                    write_plan=(),
                    trigger_cell_address=trigger_address,
                    trigger_mapping_name=replace_trigger,
                    stake_capped=stake_capped,
                    stake_cap_value=stake_cap_value,
                    bet_ref_before=bet_ref_before,
                    bet_ref_after=bet_ref_after,
                    update_allowed=False,
                    update_skipped_reason=skip_reason,
                    bet_ref_lookup_source=lookup_source,
                    bet_ref_lookup_matched_runner=lookup_matched_runner,
                    bet_ref_status_value=bet_ref_status_value,
                    replace_bet_ref_wait_attempted=replace_bet_ref_wait_attempted,
                    replace_bet_ref_wait_ms=replace_bet_ref_wait_ms,
                    replace_bet_ref_poll_ms=replace_bet_ref_poll_ms,
                    replace_bet_ref_wait_result=replace_bet_ref_wait_result,
                    bet_ref_before_wait=bet_ref_before_wait,
                    bet_ref_after_wait=bet_ref_after_wait,
                    active_ladder_bet_ref_updated=active_ladder_bet_ref_updated,
                    replace_skipped_bet_ref_still_pending=replace_skipped_bet_ref_still_pending,
                    matched_stake_cell_address=matched_stake_address,
                    matched_stake_cell_value=matched_stake_cell_value,
                    no_stacking_check_passed=True,
                )
            stripped_bet_ref = strip_gruss_ref_suffix(bet_ref_before)
            if stripped_bet_ref != bet_ref_before:
                bet_ref_suffix_n_handled = True
                bet_ref_before = stripped_bet_ref

        guard_reason, countdown_at_write, current_market_price, stale_distance, stale_limit, stale_ignored = (
            self._pre_ladder_write_guard(intent, context, sheet_name, runner_row)
        )
        if guard_reason:
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                guard_reason,
                excel_sheet=sheet_name,
                excel_row=runner_row,
                write_plan=plan,
                trigger_cell_address=trigger_address,
                stake_capped=stake_capped,
                stake_cap_value=stake_cap_value,
                bet_ref_before=bet_ref_before,
                update_allowed=replace_allowed,
                update_skipped_reason=guard_reason,
                countdown_at_write=countdown_at_write,
                current_market_price_at_write=current_market_price,
                stale_distance=stale_distance,
                stale_price_limit=stale_limit,
                stale_check_ignored_for_pre=stale_ignored,
                no_stacking_check_passed=True,
            )

        trigger_mapping_name = (
            self.layout.replace_trigger_name(intent.side)
            if effective_step_index > 0
            else self.layout.trigger_mapping_name(intent.side, intent.order_type)
        )
        plan = self._build_write_plan(
            intent,
            runner_row,
            trigger_override=trigger_mapping_name,
            bet_ref_override=bet_ref_before if bet_ref_suffix_n_handled else None,
        )

        try:
            trigger_value = self._read_cell_with_retry(
                sheet_name,
                trigger_address,
                f"read_trigger_cell:{sheet_name}!{trigger_address}",
            )
        except Exception as exc:
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                f"trigger_cell_read_failed: {exc}",
                excel_sheet=sheet_name,
                excel_row=runner_row,
                write_plan=plan,
                trigger_cell_address=trigger_address,
                trigger_mapping_name=trigger_mapping_name,
                stake_capped=stake_capped,
                stake_cap_value=stake_cap_value,
                bet_ref_before=bet_ref_before,
                update_allowed=replace_allowed,
                update_skipped_reason="trigger_cell_read_failed",
            )
        if trigger_value not in (None, ""):
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                "trigger_cell_not_empty",
                excel_sheet=sheet_name,
                excel_row=runner_row,
                write_plan=plan,
                trigger_cell_address=trigger_address,
                trigger_cell_current_value=trigger_value,
                trigger_cell_expected_empty=False,
                trigger_mapping_name=trigger_mapping_name,
                stake_capped=stake_capped,
                stake_cap_value=stake_cap_value,
                bet_ref_before=bet_ref_before,
                update_allowed=replace_allowed,
                update_skipped_reason="trigger_cell_not_empty",
            )

        write_result = self._write_cells_with_retry(sheet_name, plan)
        if write_result.exception is not None:
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                f"{write_result.final_status}: {write_result.exception}",
                excel_sheet=sheet_name,
                excel_row=runner_row,
                write_plan=plan,
                trigger_cell_address=trigger_address,
                trigger_cell_current_value=trigger_value,
                trigger_cell_expected_empty=True,
                trigger_mapping_name=trigger_mapping_name,
                stake_capped=stake_capped,
                stake_cap_value=stake_cap_value,
                bet_ref_before=bet_ref_before,
                update_allowed=replace_allowed,
                update_skipped_reason=write_result.final_status,
                excel_write_attempt=write_result.attempt,
                excel_write_retry_count=write_result.retry_count,
                excel_write_retry_backoff_ms=write_result.retry_backoff_ms,
                excel_write_final_status=write_result.final_status,
                excel_unavailable_recovered=write_result.recovered,
            )
        written = write_result.written

        verification = self._verify_real_write(sheet_name, runner_row, plan)
        trigger_written = _values_match(trigger_mapping_name, verification.trigger_value)
        clear_outcome = _TriggerClearOutcome()
        if trigger_written:
            if _diagnostic_keep_triggers():
                clear_outcome = _TriggerClearOutcome(
                    attempted=False,
                    reason="diagnostic_keep_triggers_enabled",
                    value_before_clear=trigger_mapping_name,
                    delay_ms=0,
                )
            else:
                clear_outcome = self._clear_written_trigger(
                    sheet_name,
                    trigger_address,
                    trigger_mapping_name,
                    hold_for_visual_test=False,
                    delay_override_ms=_pre_ladder_trigger_clear_delay_override_ms(),
                    processed_key=_processed_key(intent, context),
                    expected_command_cell_values=plan,
                )
        poll_attempts = 0
        poll_duration_ms = 0
        if effective_step_index == 0:
            lookup_source = "pending_batch_bet_ref_collection"
            lookup_matched_runner = "False"
        else:
            wait_result = self._wait_for_replace_bet_ref(sheet_name, bet_ref_address)
            replace_bet_ref_wait_attempted = True
            replace_bet_ref_wait_ms = wait_result.wait_ms
            replace_bet_ref_poll_ms = wait_result.poll_ms
            replace_bet_ref_wait_result = wait_result.result
            bet_ref_before_wait = wait_result.before_wait
            bet_ref_after_wait = wait_result.after_wait
            bet_ref_after = wait_result.bet_ref or wait_result.after_wait
            if is_valid_bet_ref(bet_ref_after):
                if not pre_bet_ref_late_detected and lookup_source not in {"active_ladder_state", "excel_row"}:
                    lookup_source = "excel_row_after_replace_wait"
                lookup_matched_runner = "True"
                active_ladder_bet_ref_updated = True

        finish_kwargs = {
            "excel_sheet": sheet_name,
            "excel_row": runner_row,
            "excel_cells_written": written,
            "write_plan": plan,
            "trigger_written": trigger_written,
            "trigger_cell_address": trigger_address,
            "trigger_cell_current_value": trigger_value,
            "trigger_cell_expected_empty": True,
            "trigger_mapping_name": trigger_mapping_name,
            "trigger_value_written": trigger_mapping_name if trigger_written else "",
            "trigger_clear_attempted": clear_outcome.attempted,
            "trigger_cleared": clear_outcome.cleared,
            "trigger_clear_reason": clear_outcome.reason,
            "trigger_cell_value_before_clear": clear_outcome.value_before_clear,
            "trigger_clear_delay_ms": clear_outcome.delay_ms,
            "command_cells_clear_attempted": clear_outcome.attempted,
            "command_cells_cleared": clear_outcome.cleared,
            "command_cells_clear_reason": clear_outcome.command_reason,
            "command_cells_clear_addresses": ";".join(clear_outcome.addresses),
            "command_cells_clear_delay_ms": clear_outcome.delay_ms,
            "command_cells_clear_scheduled": clear_outcome.scheduled,
            "command_cells_clear_due_time": clear_outcome.due_time,
            "command_cells_clear_non_blocking": clear_outcome.non_blocking,
            "command_cells_clear_executed": clear_outcome.executed,
            "command_cells_clear_lag_ms": clear_outcome.lag_ms,
            "post_write_verification": verification,
            "hold_trigger_for_visual_test": False,
            "stake_capped": stake_capped,
            "stake_cap_value": stake_cap_value,
            "bet_ref_before": bet_ref_before,
            "bet_ref_after": bet_ref_after,
            "bet_ref_poll_attempts": poll_attempts,
            "bet_ref_poll_duration_ms": poll_duration_ms,
            "bet_ref_lookup_source": lookup_source,
            "bet_ref_lookup_matched_runner": lookup_matched_runner,
            "update_allowed": replace_allowed,
            "update_skipped_reason": "",
            "intended_trigger_override": trigger_mapping_name,
            "replace_allowed": replace_allowed,
            "replace_trigger": replace_trigger,
            "bet_ref_suffix_n_handled": bet_ref_suffix_n_handled,
            "bet_ref_status_value": bet_ref_status_value,
            "replace_bet_ref_wait_attempted": replace_bet_ref_wait_attempted,
            "replace_bet_ref_wait_ms": replace_bet_ref_wait_ms,
            "replace_bet_ref_poll_ms": replace_bet_ref_poll_ms,
            "replace_bet_ref_wait_result": replace_bet_ref_wait_result,
            "bet_ref_before_wait": bet_ref_before_wait,
            "bet_ref_after_wait": bet_ref_after_wait,
            "active_ladder_bet_ref_updated": active_ladder_bet_ref_updated,
            "replace_skipped_bet_ref_still_pending": replace_skipped_bet_ref_still_pending,
            "matched_stake_cell_address": matched_stake_address if step_index > 0 else "",
            "matched_stake_cell_value": matched_stake_cell_value,
            "countdown_at_write": countdown_at_write,
            "current_market_price_at_write": current_market_price,
            "stale_distance": stale_distance,
            "stale_price_limit": stale_limit,
            "stale_check_ignored_for_pre": stale_ignored,
            "no_stacking_check_passed": True,
            "excel_write_attempt": write_result.attempt,
            "excel_write_retry_count": write_result.retry_count,
            "excel_write_retry_backoff_ms": write_result.retry_backoff_ms,
            "excel_write_final_status": write_result.final_status,
            "excel_unavailable_recovered": write_result.recovered,
            "pre_bet_ref_late_detected": pre_bet_ref_late_detected,
            "pre_bet_ref_late_value": pre_bet_ref_late_value,
            "pre_retry_count": pre_retry_count,
            "pre_retry_allowed": pre_retry_allowed,
            "pre_retry_reason": pre_retry_reason,
            "pre_retry_block_reason": pre_retry_block_reason,
            "active_ladder_created": active_ladder_created,
            "pending_ladder_created": pending_ladder_created,
            "matched_evidence_found": matched_evidence_found,
            "selection_row_evidence_found": selection_row_evidence_found,
            "no_stacking_blocked_retry": no_stacking_blocked_retry,
        }
        if not verification.verified:
            return self._finish(
                intent,
                context,
                "GRUSS_WRITE_FAILED",
                "post_write_verification_failed",
                **finish_kwargs,
            )
        real_key = _processed_key(intent, context)
        max_key = _max_orders_key(intent, context)
        self.real_order_counts[real_key] = self.real_order_counts.get(real_key, 0) + 1
        self.real_order_counts[max_key] = self.real_order_counts.get(max_key, 0) + 1
        if trigger_written and ladder_id not in self.active_pre_ladders:
            confirmed_bet_ref_after = is_valid_bet_ref(bet_ref_after)
            self.active_pre_ladders[ladder_id] = _active_pre_ladder_state_from_intent(
                intent,
                course_key,
                runner_row,
                bet_ref=bet_ref_after,
                pending_confirmation=not confirmed_bet_ref_after,
            )
            self.active_pre_ladder_id = ladder_id
            self.active_pre_ladder_course = course_key
            active_ladder_created = confirmed_bet_ref_after
            pending_ladder_created = not confirmed_bet_ref_after
        elif trigger_written and ladder_id in self.active_pre_ladders:
            self.active_pre_ladders[ladder_id].bet_ref = bet_ref_after
            self.active_pre_ladders[ladder_id].pending_confirmation = not is_valid_bet_ref(bet_ref_after)
        finish_kwargs["active_ladder_bet_ref_stored"] = bool(
            ladder_id in self.active_pre_ladders
            and is_valid_bet_ref(self.active_pre_ladders[ladder_id].bet_ref)
        )
        finish_kwargs["active_ladder_created"] = active_ladder_created
        finish_kwargs["pending_ladder_created"] = pending_ladder_created
        active_ladder_completed = trigger_written and _is_final_ladder_step(intent) and not pre_retry_allowed
        active_ladder_release_reason = ""
        if active_ladder_completed:
            active_ladder_release_reason = "final_ladder_step_completed"
        elif previous_course_released:
            active_ladder_release_reason = "previous_course_released_for_new_ladder"
        active_snapshot = ladder_id if ladder_id in self.active_pre_ladders else self.active_pre_ladder_id or ladder_id
        result = self._finish(
            intent,
            context,
            "GRUSS_PRE_LADDER_WRITTEN",
            (
                "pre_ladder_retry_initial_written"
                if pre_retry_allowed
                else ("pre_ladder_replace_written" if step_index > 0 else "pre_ladder_step_written")
            ),
            active_pre_ladder_id_snapshot=active_snapshot,
            active_ladder_completed=active_ladder_completed,
            active_ladder_release_reason=active_ladder_release_reason,
            **finish_kwargs,
        )
        if active_ladder_completed:
            self.active_pre_ladders.pop(ladder_id, None)
            self._refresh_legacy_active_ladder_snapshot()
        return result

    def _preflight_errors(
        self,
        intent: OrderIntent,
        context: GrussRealOrderContext,
    ) -> list[str]:
        errors: list[str] = []
        if self.order_provider != ORDER_PROVIDER_GRUSS_EXCEL_REAL:
            errors.append(f"real_provider_not_selected={self.order_provider or 'unset'}")
        if self.preview_only_guard and self.write_no_trigger_guard:
            errors.append("conflicting_provider_safety_guards")
        if self.write_no_trigger_guard and not self.write_no_trigger:
            errors.append("write_no_trigger_mode_not_enabled")
        if self.write_no_trigger_guard and self.preview:
            errors.append("write_no_trigger_requires_real_preview_false")
        if self.write_no_trigger and not self.write_no_trigger_guard:
            errors.append("write_no_trigger_requires_guarded_provider")
        if self.preview_only_guard and self.enabled:
            errors.append("preview_only_refuses_real_orders_enabled")
        if self.preview_only_guard and not self.preview:
            errors.append("preview_only_requires_preview")
        unarmed_safe_mode = self.preview or (self.write_no_trigger_guard and self.write_no_trigger)
        if not self.enabled and not unarmed_safe_mode:
            errors.append("real_orders_not_enabled: set DOGBOT_GRUSS_ENABLE_REAL_ORDERS=true")
        if not context.validation_ok:
            errors.append("win_place_validation_failed")
        if not context.tradable:
            errors.append("market_not_tradable")
        if str(context.region or "").upper() == "UNKNOWN":
            errors.append("unknown_region")
        if context.countdown_seconds is None:
            errors.append("countdown_seconds_unavailable")
        elif self._is_true_real_mode() and _execution_phase(intent) == "POST":
            if context.countdown_seconds < -_post_allow_after_scheduled_off_seconds():
                errors.append("after_off_do_not_write")
            elif context.countdown_seconds > max(_post_send_seconds_before_off(), 3):
                errors.append("countdown_above_3_seconds")
        elif context.countdown_seconds < 0 and _execution_phase(intent) != "POST":
            errors.append("countdown_elapsed")
        elif _is_pre_ladder_intent(intent):
            if not _is_valid_pre_ladder_milestone(intent, context):
                errors.append("pre_ladder_countdown_not_configured_milestone")
        elif context.countdown_seconds > (2 if self.write_no_trigger_guard else 3):
            errors.append(
                "countdown_above_2_seconds"
                if self.write_no_trigger_guard
                else "countdown_above_3_seconds"
            )
        processed_key = _processed_key(intent, context)
        if (
            context.market_already_processed
            or processed_key in self.processed_markets
            or self.real_order_counts.get(processed_key, 0) > 0
        ):
            errors.append("market_already_processed")
        if (
            intent.stake_forced
            and not self.preview
            and not (self._is_true_real_mode() and self.real_test_mode)
        ):
            errors.append("forced_stake_requires_real_test_mode")
        if (
            self.hold_trigger_for_visual_test
            and not self.preview
            and not (self._is_true_real_mode() and self.real_test_mode)
        ):
            errors.append("hold_trigger_for_visual_test_requires_real_test_mode")
        if intent.force_test_bsp_place and intent.force_test_back_place_limit:
            errors.append("forced_test_modes_are_mutually_exclusive")
        if intent.force_test_bsp_place:
            errors.extend(self._force_test_bsp_place_errors(intent))
        if intent.force_test_back_place_limit:
            errors.extend(self._force_test_back_place_limit_errors(intent))
        if _is_pre_ladder_intent(intent):
            errors.extend(self._pre_ladder_real_test_errors(intent))
        if self._is_true_real_mode():
            max_orders = self._max_orders_for_intent(intent)
            max_key = _max_orders_key(intent, context)
            if max_orders is not None and self.real_order_counts.get(max_key, 0) >= max_orders:
                errors.append("max_orders_reached")

        minimum_stake = float("-inf")
        errors.extend(validate_order_intent(intent, minimum_stake=minimum_stake))
        if str(intent.order_type or "").upper() not in {"LIMIT", "SP_MOC"}:
            errors.append("invalid_order_type")
        if not _positive_finite(intent.stake):
            errors.append("invalid_stake")
        if not _valid_order_price(intent.price):
            errors.append("invalid_price")
        return _dedupe(errors)

    def _pre_ladder_real_test_errors(self, intent: OrderIntent) -> list[str]:
        errors: list[str] = []
        if not _env_bool("DOGBOT_PRE_LADDER_ENABLED", False):
            errors.append("pre_ladder_enabled_required")
        if _env_bool("DOGBOT_PRE_LADDER_PREVIEW", True):
            errors.append("pre_ladder_preview_must_be_false_for_real_test")
        if not (self._is_true_real_mode() and self.real_test_mode):
            errors.append("pre_ladder_real_requires_real_test_mode")
        max_ladders_error = _pre_ladder_real_max_ladders_config_error()
        if max_ladders_error:
            errors.append(max_ladders_error)
        if not _env_bool("DOGBOT_PRE_LADDER_REAL_REQUIRE_BET_REF_FOR_REPLACE", True):
            errors.append("pre_ladder_real_require_bet_ref_for_replace_required")
        if not _env_bool("DOGBOT_PRE_LADDER_REAL_STOP_IF_NO_BET_REF", True):
            errors.append("pre_ladder_real_stop_if_no_bet_ref_required")
        if not _env_bool("DOGBOT_PRE_LADDER_REAL_NO_STACKING", True):
            errors.append("pre_ladder_real_no_stacking_required")
        variable_stakes = _env_bool("DOGBOT_GRUSS_REAL_VARIABLE_STAKES", False)
        if variable_stakes:
            variable_cap_error = _variable_stake_cap_config_error(self.real_max_stake)
            if variable_cap_error:
                errors.append(variable_cap_error)
        else:
            if self.real_max_stake is None or self.real_max_stake != 2.0:
                errors.append("pre_ladder_real_requires_max_stake_eq_2")
            if not intent.stake_forced:
                errors.append("pre_ladder_real_requires_forced_stake_eq_2")
            elif _float_or_infinity(intent.stake) != 2.0:
                errors.append("pre_ladder_real_stake_must_equal_2")
        if str(intent.market_type or "").upper() != "PLACE":
            errors.append("pre_ladder_real_requires_place_market")
        if _execution_phase(intent) != "PRE":
            errors.append("pre_ladder_real_requires_pre_phase")
        if str(intent.order_type or "").upper() != "LIMIT":
            errors.append("pre_ladder_real_requires_limit")
        return errors

    def _force_test_bsp_place_errors(self, intent: OrderIntent) -> list[str]:
        errors: list[str] = []
        if not (self._is_true_real_mode() and self.real_test_mode):
            errors.append("force_test_bsp_place_requires_real_test_mode")
        if self.real_max_orders != 1:
            errors.append("force_test_bsp_place_requires_max_orders_1")
        if self.real_max_stake is None or self.real_max_stake <= 0 or self.real_max_stake > 2.0:
            errors.append("force_test_bsp_place_requires_max_stake_lte_2")
        if not intent.stake_forced:
            errors.append("force_test_bsp_place_requires_forced_stake")
        elif _float_or_infinity(intent.stake) > min(self.real_max_stake or 0.0, 2.0):
            errors.append("force_test_bsp_place_forced_stake_above_max")
        if str(intent.market_type or "").upper() != "PLACE":
            errors.append("force_test_bsp_place_requires_place_market")
        if str(intent.side or "").upper() != "BACK":
            errors.append("force_test_bsp_place_requires_back")
        if str(intent.order_type or "").upper() != "SP_MOC":
            errors.append("force_test_bsp_place_requires_sp_moc")
        if str(intent.strategy_id or "") != "GRUSS_FORCE_TEST_BSP_PLACE":
            errors.append("force_test_bsp_place_invalid_strategy_id")
        if not str(self.layout.back_sp_moc_trigger or "").strip():
            errors.append("back_sp_mapping_unavailable")
        return errors

    def _force_test_back_place_limit_errors(self, intent: OrderIntent) -> list[str]:
        errors: list[str] = []
        if not (self._is_true_real_mode() and self.real_test_mode):
            errors.append("force_test_back_place_limit_requires_real_test_mode")
        if self.real_max_orders != 1:
            errors.append("force_test_back_place_limit_requires_max_orders_1")
        if self.real_max_stake is None or self.real_max_stake <= 0 or self.real_max_stake > 2.0:
            errors.append("force_test_back_place_limit_requires_max_stake_lte_2")
        if not intent.stake_forced:
            errors.append("force_test_back_place_limit_requires_forced_stake")
        elif _float_or_infinity(intent.stake) > min(self.real_max_stake or 0.0, 2.0):
            errors.append("force_test_back_place_limit_forced_stake_above_max")
        if str(intent.market_type or "").upper() != "PLACE":
            errors.append("force_test_back_place_limit_requires_place_market")
        if str(intent.side or "").upper() != "BACK":
            errors.append("force_test_back_place_limit_requires_back")
        if str(intent.order_type or "").upper() != "LIMIT":
            errors.append("force_test_back_place_limit_requires_limit")
        if str(intent.strategy_id or "") != "GRUSS_FORCE_TEST_BACK_PLACE_LIMIT":
            errors.append("force_test_back_place_limit_invalid_strategy_id")
        if not _valid_order_price(intent.selected_place_lay_odds):
            errors.append("missing_place_best_lay")
        if _float_or_infinity(intent.price) != _float_or_infinity(intent.selected_place_lay_odds):
            errors.append("force_test_back_place_limit_price_must_equal_best_lay")
        return errors

    def _find_runner_row(self, sheet_name: str, intent: OrderIntent) -> int | None:
        row_values = self._read_runner_row_values(sheet_name)
        return _find_runner_row_in_values(row_values, intent)

    def _read_runner_row_values(self, sheet_name: str) -> list[tuple[int, Any]]:
        range_address = "A5:A84"
        result = self._excel_com_retry(
            f"mapping_read_runner_rows:{sheet_name}!{range_address}",
            lambda: self.bridge.read_range(sheet_name, range_address),
        )
        self._last_mapping_com_result = result
        if result.exception is not None:
            exc = result.exception
            raise RuntimeError(
                f"excel_mapping_unavailable_after_retries sheet={sheet_name} range={range_address}: {exc}"
            ) from exc
        flatten_result = self._excel_com_retry(
            f"mapping_flatten_runner_rows:{sheet_name}!{range_address}",
            lambda: list(_flatten_single_column(result.value)),
        )
        if flatten_result.retry_count or flatten_result.exception is not None:
            self._last_mapping_com_result = flatten_result
        if flatten_result.exception is not None:
            exc = flatten_result.exception
            raise RuntimeError(
                f"excel_mapping_unavailable_after_retries sheet={sheet_name} range={range_address}: {exc}"
            ) from exc
        candidates = list(flatten_result.value or [])
        return [(5 + offset, value) for offset, value in enumerate(candidates) if value not in (None, "")]

    def _current_market_error(
        self,
        intent: OrderIntent,
        context: GrussRealOrderContext,
    ) -> str | None:
        expected_ids = {
            "WIN": context.win_market_id,
            "PLACE": context.place_market_id,
        }
        expected_ids[intent.market_type.upper()] = intent.market_id
        for sheet_name, expected_id in expected_ids.items():
            if not expected_id:
                continue
            current_id = _normalise_identifier(
                self._read_cell_with_retry(sheet_name, "N3", f"read_market_id:{sheet_name}!N3")
            )
            if not current_id:
                return f"current_market_id_unavailable={sheet_name}"
            if current_id != _normalise_identifier(expected_id):
                return f"current_market_id_mismatch={sheet_name}:{current_id}"
            market_status = self._read_cell_with_retry(sheet_name, "F2", f"read_market_status:{sheet_name}!F2")
            if _is_untradable_market_status(market_status):
                return f"current_market_suspended={sheet_name}"
            current_countdown = self._current_countdown_seconds(sheet_name)
            if (
                _execution_phase(intent) == "POST"
                and _post_countdown_allowed(context.countdown_seconds)
                and (
                    current_countdown is None
                    or current_countdown > _post_send_seconds_before_off()
                )
            ):
                current_countdown = context.countdown_seconds
            if (
                self._is_true_real_mode()
                and current_countdown is not None
                and _execution_phase(intent) == "POST"
                and current_countdown < -_post_allow_after_scheduled_off_seconds()
            ):
                return "after_off_do_not_write"
            if (
                self._is_true_real_mode()
                and current_countdown is not None
                and _execution_phase(intent) != "POST"
                and current_countdown <= 0
            ):
                return "after_off_do_not_write"
        return None

    def _current_countdown_seconds(self, sheet_name: str) -> int | None:
        try:
            return parse_countdown_seconds(
                self._read_cell_with_retry(sheet_name, "D2", f"read_countdown:{sheet_name}!D2")
            )
        except Exception:
            return None

    def _poll_bet_ref_after_initial_write(
        self,
        sheet_name: str,
        bet_ref_address: str,
    ) -> _BetRefPollResult:
        attempts_limit = _bounded_int_env("DOGBOT_PRE_LADDER_BET_REF_POLL_ATTEMPTS", 6, 1, 10)
        interval_ms = _bounded_int_env("DOGBOT_PRE_LADDER_BET_REF_POLL_INTERVAL_MS", 250, 0, 300)
        started = time.perf_counter()
        last_attempt = 0
        for attempt in range(1, attempts_limit + 1):
            last_attempt = attempt
            try:
                bet_ref = normalise_gruss_bet_ref(
                    self._read_cell_with_retry(sheet_name, bet_ref_address, f"read_bet_ref:{sheet_name}!{bet_ref_address}")
                )
            except Exception:
                bet_ref = ""
            if is_valid_bet_ref(bet_ref):
                duration_ms = int(round((time.perf_counter() - started) * 1000))
                return _BetRefPollResult(
                    bet_ref=bet_ref,
                    attempts=attempt,
                    duration_ms=duration_ms,
                    lookup_source=f"excel_row_poll:{bet_ref_address}",
                )
            if attempt < attempts_limit and interval_ms > 0:
                time.sleep(interval_ms / 1000)
        duration_ms = int(round((time.perf_counter() - started) * 1000))
        return _BetRefPollResult(
            bet_ref="",
            attempts=last_attempt,
            duration_ms=duration_ms,
            lookup_source=f"excel_row_poll_timeout:{bet_ref_address}",
        )

    def _poll_post_bet_ref(
        self,
        sheet_name: str,
        bet_ref_address: str,
        *,
        existing_bet_ref_before: str = "",
    ) -> _BetRefPollResult:
        wait_ms = _post_bet_ref_wait_ms()
        poll_ms = _post_bet_ref_poll_ms()
        existing_normalised = normalise_gruss_bet_ref(existing_bet_ref_before)
        started = time.perf_counter()
        deadline = started + (wait_ms / 1000)
        attempts = 0
        last_bet_ref = ""
        last_lookup_source = f"post_excel_row_poll_timeout:{bet_ref_address}"
        while True:
            attempts += 1
            try:
                bet_ref = normalise_gruss_bet_ref(
                    self._read_cell_with_retry(
                        sheet_name,
                        bet_ref_address,
                        f"post_read_bet_ref:{sheet_name}!{bet_ref_address}",
                    )
                )
            except Exception:
                bet_ref = ""
            last_bet_ref = bet_ref
            duration_ms = int(round((time.perf_counter() - started) * 1000))
            if is_valid_bet_ref(bet_ref) and (
                not is_valid_bet_ref(existing_normalised)
                or strip_gruss_ref_suffix(bet_ref) != strip_gruss_ref_suffix(existing_normalised)
            ):
                return _BetRefPollResult(
                    bet_ref=bet_ref,
                    attempts=attempts,
                    duration_ms=duration_ms,
                    lookup_source=f"post_excel_row_poll:{bet_ref_address}",
                )
            if is_valid_bet_ref(bet_ref) and is_valid_bet_ref(existing_normalised):
                last_lookup_source = f"post_excel_row_poll_unchanged:{bet_ref_address}"
            if wait_ms <= 0 or time.perf_counter() >= deadline:
                return _BetRefPollResult(
                    bet_ref=last_bet_ref,
                    attempts=attempts,
                    duration_ms=duration_ms,
                    lookup_source=last_lookup_source,
                )
            remaining_ms = max(0.0, (deadline - time.perf_counter()) * 1000)
            time.sleep(min(float(poll_ms), remaining_ms) / 1000)

    def _lookup_post_selection_confirmation(
        self,
        sheet_name: str,
        intent: OrderIntent,
        context: GrussRealOrderContext,
        *,
        post_write_timestamp: str,
        existing_bet_ref_before: str,
        expected_market_id: str,
        expected_market_type: str,
        expected_runner: str,
        expected_selection_id: str,
        expected_side: str,
        expected_stake: float | None,
        expected_price: float | None,
    ) -> _PostSelectionsLookupResult:
        market_type = str(intent.market_type or sheet_name or "PLACE").upper()
        selections_sheet = f"{market_type}_Selections"
        candidates, rows_scanned, _headers = self._read_selection_bet_ref_candidates(selections_sheet)
        if not candidates:
            return _PostSelectionsLookupResult(
                attempted=True,
                rows_scanned=rows_scanned,
                reject_reason=f"no_selection_candidates:{selections_sheet}",
            )

        matches: list[tuple[_SelectionsBetRefCandidate, str]] = []
        reject_reasons: list[str] = []
        for candidate in candidates:
            reject_reason = _post_selection_reject_reason(
                candidate,
                context,
                post_write_timestamp=post_write_timestamp,
                existing_bet_ref_before=existing_bet_ref_before,
                expected_market_id=expected_market_id,
                expected_market_type=expected_market_type,
                expected_runner=expected_runner,
                expected_trap=intent.trap,
                expected_selection_id=expected_selection_id,
                expected_side=expected_side,
                expected_stake=expected_stake,
                expected_price=expected_price,
            )
            if reject_reason:
                reject_reasons.append(reject_reason)
                continue
            matches.append((candidate, "post_selections_strict_match"))

        if not matches:
            return _PostSelectionsLookupResult(
                attempted=True,
                rows_scanned=rows_scanned,
                reject_reason=_summarise_post_selection_rejections(reject_reasons),
            )
        if len(matches) > 1:
            return _PostSelectionsLookupResult(
                attempted=True,
                rows_scanned=rows_scanned,
                reject_reason=f"ambiguous_post_selection_rows:{len(matches)}",
            )
        match, reason = matches[0]
        return _PostSelectionsLookupResult(
            match=match,
            attempted=True,
            rows_scanned=rows_scanned,
            match_reason=reason,
            lookup_source=f"post_selections_sheet:{selections_sheet}!row{match.row_number}",
        )

    def _wait_for_replace_bet_ref(
        self,
        sheet_name: str,
        bet_ref_address: str,
    ) -> _ReplaceBetRefWaitResult:
        wait_ms = _bounded_int_env("DOGBOT_GRUSS_REPLACE_BET_REF_WAIT_MS", 6000, 0, 10000)
        poll_ms = _bounded_int_env("DOGBOT_GRUSS_REPLACE_BET_REF_POLL_MS", 250, 10, 1000)
        started = time.perf_counter()
        deadline = started + (wait_ms / 1000)
        attempts = 0
        before_wait = ""
        after_wait = ""
        while True:
            attempts += 1
            try:
                current = normalise_gruss_bet_ref(
                    self._read_cell_with_retry(sheet_name, bet_ref_address, f"read_bet_ref:{sheet_name}!{bet_ref_address}")
                )
            except Exception:
                current = ""
            if attempts == 1:
                before_wait = current
            after_wait = current
            duration_ms = int(round((time.perf_counter() - started) * 1000))
            if is_valid_bet_ref(current):
                return _ReplaceBetRefWaitResult(
                    bet_ref=current,
                    before_wait=before_wait,
                    after_wait=current,
                    attempts=attempts,
                    duration_ms=duration_ms,
                    wait_ms=wait_ms,
                    poll_ms=poll_ms,
                    result="resolved",
                )
            if is_terminal_bet_status(current):
                return _ReplaceBetRefWaitResult(
                    bet_ref="",
                    before_wait=before_wait,
                    after_wait=current,
                    attempts=attempts,
                    duration_ms=duration_ms,
                    wait_ms=wait_ms,
                    poll_ms=poll_ms,
                    result="terminal_status",
                )
            if wait_ms <= 0 or time.perf_counter() >= deadline:
                return _ReplaceBetRefWaitResult(
                    bet_ref="",
                    before_wait=before_wait,
                    after_wait=after_wait,
                    attempts=attempts,
                    duration_ms=duration_ms,
                    wait_ms=wait_ms,
                    poll_ms=poll_ms,
                    result="timeout",
                )
            time.sleep(min(poll_ms / 1000, max(0.0, deadline - time.perf_counter())))

    def _current_market_reference_price(self, sheet_name: str, row: int) -> float | None:
        ltp = _positive_float_or_none(_safe_read_cell(self.bridge, sheet_name, f"O{row}"))
        best_back = _positive_float_or_none(_safe_read_cell(self.bridge, sheet_name, f"F{row}"))
        best_lay = _positive_float_or_none(_safe_read_cell(self.bridge, sheet_name, f"H{row}"))
        midpoint = (best_back + best_lay) / 2.0 if best_back is not None and best_lay is not None else None
        return _first_positive(ltp, midpoint, best_back, best_lay)

    def _pre_ladder_write_guard(
        self,
        intent: OrderIntent,
        context: GrussRealOrderContext,
        sheet_name: str,
        row: int,
    ) -> tuple[str | None, int | None, float | None, float | None, float | None, bool]:
        countdown_at_write = self._current_countdown_seconds(sheet_name)
        if countdown_at_write is None:
            countdown_at_write = context.countdown_seconds
        current_market_price = self._current_market_reference_price(sheet_name, row)
        milestone = _current_milestone(context)
        if countdown_at_write is not None:
            if countdown_at_write <= 0:
                return "after_off_do_not_write", countdown_at_write, current_market_price, None, _stale_price_limit(), False
            if milestone in _pre_ladder_steps_from_env() and not _countdown_in_pre_ladder_window(
                milestone,
                countdown_at_write,
            ):
                if _pre_batch_late_write_allowed(intent, context, milestone, countdown_at_write):
                    stale_limit = _stale_price_limit()
                    signal_reference = intent.market_reference_price_at_signal
                    stale_distance = _relative_distance(current_market_price, signal_reference)
                    if stale_distance is not None and stale_distance > stale_limit:
                        if _pre_ignore_stale_price_before_write():
                            return None, countdown_at_write, current_market_price, stale_distance, stale_limit, True
                        return (
                            "stale_market_price_before_write",
                            countdown_at_write,
                            current_market_price,
                            stale_distance,
                            stale_limit,
                            False,
                        )
                    return None, countdown_at_write, current_market_price, stale_distance, stale_limit, False
                return (
                    "pre_ladder_milestone_window_missed",
                    countdown_at_write,
                    current_market_price,
                    None,
                    _stale_price_limit(),
                    False,
                )
        signal_reference = intent.market_reference_price_at_signal
        stale_limit = _stale_price_limit()
        stale_distance = _relative_distance(current_market_price, signal_reference)
        if stale_distance is not None and stale_distance > stale_limit:
            if _pre_ignore_stale_price_before_write():
                return None, countdown_at_write, current_market_price, stale_distance, stale_limit, True
            return (
                "stale_market_price_before_write",
                countdown_at_write,
                current_market_price,
                stale_distance,
                stale_limit,
                False,
            )
        return None, countdown_at_write, current_market_price, stale_distance, stale_limit, False

    def _build_write_plan(
        self,
        intent: OrderIntent,
        row: int,
        *,
        trigger_override: str | None = None,
        bet_ref_override: str | None = None,
    ) -> tuple[tuple[str, Any], ...]:
        trigger = trigger_override or self._trigger_for(intent)
        cells: list[tuple[str, Any]] = []
        if bet_ref_override is not None:
            cells.append((self.layout.bet_ref_address(row), bet_ref_override))
        cells.append((self.layout.odds_address(row), _price_to_write_for_gruss(intent, trigger)))
        cells.append((self.layout.stake_address(row), float(intent.stake)))
        # Trigger is deliberately written last.
        cells.append((self.layout.trigger_address(row), trigger))
        return tuple(cells)

    def _cap_real_stake(self, intent: OrderIntent) -> tuple[OrderIntent, bool, float | None]:
        cap_value = self.real_max_stake
        try:
            stake_to_write = float(intent.stake)
        except (TypeError, ValueError):
            return intent, False, cap_value
        if not math.isfinite(stake_to_write) or stake_to_write <= 0.0:
            return intent, False, cap_value
        stake_original = intent.stake_original if intent.stake_original is not None else stake_to_write
        if 0.0 < stake_to_write < 1.0:
            intent = replace(intent, stake=1.0, stake_original=stake_original)
            stake_to_write = 1.0
        if cap_value is None:
            return intent, False, None
        try:
            cap = float(cap_value)
        except (TypeError, ValueError):
            return intent, False, cap_value
        if not math.isfinite(cap) or cap <= 0:
            return intent, False, cap
        if stake_to_write <= cap:
            return intent, False, cap
        return replace(intent, stake=cap, stake_original=stake_original), True, cap

    def _trigger_for(self, intent: OrderIntent) -> str:
        return self.layout.trigger_mapping_name(intent.side, intent.order_type)

    def _without_trigger(
        self,
        plan: Iterable[tuple[str, Any]],
        row: int,
    ) -> tuple[tuple[str, Any], ...]:
        trigger_address = self.layout.trigger_address(row)
        preparation = tuple(
            (address, value)
            for address, value in plan
            if str(address).upper() != trigger_address
        )
        if any(str(address).upper() == trigger_address for address, _ in preparation):
            raise RuntimeError("trigger_cell_present_in_no_trigger_plan")
        addresses = [str(address).upper() for address, _ in preparation]
        expected = {
            self.layout.odds_address(row),
            self.layout.stake_address(row),
        }
        if len(addresses) != 2 or len(set(addresses)) != 2 or set(addresses) != expected:
            raise RuntimeError("preparation_plan_must_contain_only_distinct_odds_and_stake_cells")
        return preparation

    def _clear_written_trigger(
        self,
        sheet_name: str,
        trigger_address: str,
        trigger_value_written: str,
        *,
        hold_for_visual_test: bool = False,
        delay_override_ms: int | None = None,
        processed_key: str = "",
        expected_command_cell_values: Iterable[tuple[str, Any]] = (),
    ) -> _TriggerClearOutcome:
        if not self.command_cells_clear_enabled:
            return _TriggerClearOutcome(
                attempted=False,
                reason="command_cells_clear_disabled",
                command_reason="command_cells_clear_disabled",
                addresses=(),
            )
        delay_ms = self.command_cells_clear_delay_ms if delay_override_ms is None else max(0, delay_override_ms)
        clear_addresses = self._command_cell_addresses_for_trigger(trigger_address)
        expected_values = self._expected_command_cell_values_for_clear(
            trigger_address,
            trigger_value_written,
            clear_addresses,
            expected_command_cell_values,
        )
        if self.command_cells_clear_non_blocking and delay_ms > 0 and not hold_for_visual_test:
            due_monotonic = time.monotonic() + (delay_ms / 1000)
            due_time = _iso_from_timestamp(time.time() + (delay_ms / 1000))
            self._pending_command_cell_clears.append(
                _PendingCommandCellsClear(
                    processed_key=processed_key,
                    sheet_name=sheet_name,
                    trigger_address=trigger_address,
                    trigger_value_written=trigger_value_written,
                    addresses=clear_addresses,
                    expected_values=expected_values,
                    due_monotonic=due_monotonic,
                    due_time=due_time,
                    delay_ms=delay_ms,
                )
            )
            return _TriggerClearOutcome(
                attempted=True,
                reason="trigger_clear_scheduled",
                command_reason="command_cells_clear_scheduled",
                delay_ms=delay_ms,
                addresses=clear_addresses,
                scheduled=True,
                due_time=due_time,
                non_blocking=True,
            )
        if hold_for_visual_test:
            print(
                "holding trigger for visual test "
                f"{sheet_name}!{trigger_address}={trigger_value_written} "
                f"delay_ms={delay_ms}"
            )
        if delay_ms:
            time.sleep(delay_ms / 1000)
        read_result = self._excel_com_retry(
            f"cleanup_read_trigger:{sheet_name}!{trigger_address}",
            lambda: self.bridge.read_cell(sheet_name, trigger_address),
        )
        self._last_cleanup_com_result = read_result
        if read_result.exception is not None:
            exc = read_result.exception
            return _TriggerClearOutcome(
                attempted=True,
                reason=f"trigger_clear_read_failed: {exc}",
                command_reason=f"command_cells_clear_read_failed: {exc}",
                delay_ms=delay_ms,
                addresses=clear_addresses,
            )
        current_value = read_result.value

        if trigger_value_written not in {"BACK", "LAY", "BACKSP", "LAYSP", "BACKR", "LAYR", "CANCEL"}:
            return _TriggerClearOutcome(
                attempted=True,
                reason="trigger_clear_skipped_unrecognized_trigger",
                command_reason="command_cells_clear_skipped_unrecognized_trigger",
                value_before_clear=current_value,
                delay_ms=delay_ms,
                addresses=clear_addresses,
            )
        if current_value != trigger_value_written and current_value != "CANCEL":
            return _TriggerClearOutcome(
                attempted=True,
                reason="trigger_clear_skipped_value_changed",
                command_reason="command_cells_clear_skipped_trigger_changed",
                value_before_clear=current_value,
                delay_ms=delay_ms,
                addresses=clear_addresses,
            )
        changed_address = self._changed_command_cell_address(sheet_name, expected_values)
        if changed_address:
            return _TriggerClearOutcome(
                attempted=True,
                reason="trigger_clear_skipped_value_changed",
                command_reason="command_cells_clear_skipped_trigger_changed",
                value_before_clear=current_value,
                delay_ms=delay_ms,
                addresses=clear_addresses,
            )

        clear_result = self._excel_com_retry(
            f"cleanup_clear_command_cells:{sheet_name}!{';'.join(clear_addresses)}",
            lambda: self.bridge.clear_trigger_cells(
                sheet_name,
                clear_addresses,
                trigger_column=self.layout.trigger_column,
                command_columns=self.command_cells_clear_columns,
                allow_clear=True,
            ),
        )
        self._last_cleanup_com_result = clear_result
        if clear_result.exception is not None:
            exc = clear_result.exception
            return _TriggerClearOutcome(
                attempted=True,
                reason=f"trigger_clear_failed: {exc}",
                command_reason=f"command_cells_clear_failed: {exc}",
                value_before_clear=current_value,
                delay_ms=delay_ms,
                addresses=clear_addresses,
            )
        verify_result = self._excel_com_retry(
            f"cleanup_verify_command_cells:{sheet_name}!{';'.join(clear_addresses)}",
            lambda: [self.bridge.read_cell(sheet_name, address) for address in clear_addresses],
        )
        self._last_cleanup_com_result = verify_result
        if verify_result.exception is not None:
            exc = verify_result.exception
            return _TriggerClearOutcome(
                attempted=True,
                reason=f"trigger_clear_failed: {exc}",
                command_reason=f"command_cells_clear_failed: {exc}",
                value_before_clear=current_value,
                delay_ms=delay_ms,
                addresses=clear_addresses,
            )
        values_after_clear = list(verify_result.value or [])
        if any(value not in (None, "") for value in values_after_clear):
            return _TriggerClearOutcome(
                attempted=True,
                reason="trigger_clear_verify_failed",
                command_reason="command_cells_clear_verify_failed",
                value_before_clear=current_value,
                delay_ms=delay_ms,
                addresses=clear_addresses,
            )
        return _TriggerClearOutcome(
            attempted=True,
            cleared=True,
            reason="trigger_cleared",
            command_reason="command_cells_cleared",
            value_before_clear=current_value,
            delay_ms=delay_ms,
            addresses=clear_addresses,
            )

    def drain_due_command_cell_clears(self, *, force: bool = False) -> None:
        self._drain_due_command_cell_clears(force=force)

    def _drain_due_command_cell_clears(self, *, force: bool = False) -> None:
        if not self._pending_command_cell_clears:
            return
        now_monotonic = time.monotonic()
        still_pending: list[_PendingCommandCellsClear] = []
        for pending in self._pending_command_cell_clears:
            if not force and pending.due_monotonic > now_monotonic:
                still_pending.append(pending)
                continue
            lag_ms = int(round(max(0.0, now_monotonic - pending.due_monotonic) * 1000))
            updates: dict[str, Any] = {
                "command_cells_clear_executed": False,
                "command_cells_clear_lag_ms": lag_ms,
            }
            read_result = self._excel_com_retry(
                f"cleanup_read_pending_trigger:{pending.sheet_name}!{pending.trigger_address}",
                lambda: self.bridge.read_cell(pending.sheet_name, pending.trigger_address),
            )
            self._last_cleanup_com_result = read_result
            if read_result.exception is not None:
                exc = read_result.exception
                reason = f"command_cells_clear_read_failed: {exc}"
                updates.update(
                    {
                        "trigger_clear_reason": reason.replace(
                            "command_cells_clear_read_failed",
                            "trigger_clear_read_failed",
                            1,
                        ),
                        "command_cells_clear_reason": reason,
                    }
                )
                self._update_attempt_log_rows([pending.processed_key], common_fields=updates)
                continue
            current_value = read_result.value
            if pending.trigger_value_written not in {"BACK", "LAY", "BACKSP", "LAYSP", "BACKR", "LAYR", "CANCEL"}:
                updates.update(
                    {
                        "trigger_clear_reason": "trigger_clear_skipped_unrecognized_trigger",
                        "command_cells_clear_reason": "command_cells_clear_skipped_unrecognized_trigger",
                        "trigger_cell_value_before_clear": current_value,
                    }
                )
                self._update_attempt_log_rows([pending.processed_key], common_fields=updates)
                continue
            if current_value != pending.trigger_value_written and current_value != "CANCEL":
                updates.update(
                    {
                        "trigger_clear_reason": "trigger_clear_skipped_value_changed",
                        "command_cells_clear_reason": "command_cells_clear_skipped_trigger_changed",
                        "trigger_cell_value_before_clear": current_value,
                    }
                )
                self._update_attempt_log_rows([pending.processed_key], common_fields=updates)
                continue
            changed_address = self._changed_command_cell_address(pending.sheet_name, pending.expected_values)
            if changed_address:
                updates.update(
                    {
                        "trigger_clear_reason": "trigger_clear_skipped_value_changed",
                        "command_cells_clear_reason": "command_cells_clear_skipped_trigger_changed",
                        "trigger_cell_value_before_clear": current_value,
                    }
                )
                self._update_attempt_log_rows([pending.processed_key], common_fields=updates)
                continue
            clear_result = self._excel_com_retry(
                f"cleanup_clear_pending_command_cells:{pending.sheet_name}!{';'.join(pending.addresses)}",
                lambda: self.bridge.clear_trigger_cells(
                    pending.sheet_name,
                    pending.addresses,
                    trigger_column=self.layout.trigger_column,
                    command_columns=self.command_cells_clear_columns,
                    allow_clear=True,
                ),
            )
            self._last_cleanup_com_result = clear_result
            if clear_result.exception is None:
                verify_result = self._excel_com_retry(
                    f"cleanup_verify_pending_command_cells:{pending.sheet_name}!{';'.join(pending.addresses)}",
                    lambda: [
                    self.bridge.read_cell(pending.sheet_name, address)
                    for address in pending.addresses
                    ],
                )
                self._last_cleanup_com_result = verify_result
            else:
                verify_result = clear_result
            if verify_result.exception is not None:
                exc = verify_result.exception
                reason = f"command_cells_clear_failed: {exc}"
                updates.update(
                    {
                        "trigger_clear_reason": reason.replace(
                            "command_cells_clear_failed",
                            "trigger_clear_failed",
                            1,
                        ),
                        "command_cells_clear_reason": reason,
                        "trigger_cell_value_before_clear": current_value,
                    }
                )
                self._update_attempt_log_rows([pending.processed_key], common_fields=updates)
                continue
            values_after_clear = list(verify_result.value or [])
            if any(value not in (None, "") for value in values_after_clear):
                updates.update(
                    {
                        "trigger_clear_reason": "trigger_clear_verify_failed",
                        "command_cells_clear_reason": "command_cells_clear_verify_failed",
                        "trigger_cell_value_before_clear": current_value,
                    }
                )
                self._update_attempt_log_rows([pending.processed_key], common_fields=updates)
                continue
            updates.update(
                {
                    "trigger_cleared": True,
                    "trigger_clear_reason": "trigger_cleared",
                    "trigger_cell_value_before_clear": current_value,
                    "command_cells_cleared": True,
                    "command_cells_clear_reason": "command_cells_cleared",
                    "command_cells_clear_executed": True,
                }
            )
            self._update_attempt_log_rows([pending.processed_key], common_fields=updates)
        self._pending_command_cell_clears = still_pending

    def _expected_command_cell_values_for_clear(
        self,
        trigger_address: str,
        trigger_value_written: str,
        clear_addresses: tuple[str, ...],
        expected_command_cell_values: Iterable[tuple[str, Any]],
    ) -> tuple[tuple[str, Any], ...]:
        clear_address_set = {str(address).upper() for address in clear_addresses}
        values: dict[str, Any] = {
            str(address).upper(): value
            for address, value in expected_command_cell_values
            if str(address).upper() in clear_address_set
        }
        values[str(trigger_address).upper()] = trigger_value_written
        return tuple((address, values[address]) for address in sorted(values))

    def _changed_command_cell_address(
        self,
        sheet_name: str,
        expected_values: Iterable[tuple[str, Any]],
    ) -> str:
        for address, expected in expected_values:
            result = self._excel_com_retry(
                f"cleanup_read_expected_command_cell:{sheet_name}!{address}",
                lambda address=address: self.bridge.read_cell(sheet_name, address),
            )
            self._last_cleanup_com_result = result
            if result.exception is not None:
                return str(address)
            actual = result.value
            if not _values_match(expected, actual):
                return str(address)
        return ""

    def _command_cell_addresses_for_trigger(self, trigger_address: str) -> tuple[str, ...]:
        row_text = "".join(ch for ch in str(trigger_address or "") if ch.isdigit())
        if not row_text:
            return (str(trigger_address).upper(),)
        row = int(row_text)
        address_by_column = {
            self.layout.trigger_column: self.layout.trigger_address(row),
            self.layout.odds_column: self.layout.odds_address(row),
            self.layout.stake_column: self.layout.stake_address(row),
        }
        return tuple(
            address_by_column[column]
            for column in self.command_cells_clear_columns
            if column in address_by_column
        )

    def _excel_com_retry(self, operation_name: str, fn: Any) -> _ExcelComResult:
        backoffs = _excel_com_retry_backoff_ms()
        used_backoffs: list[int] = []
        attempt = 0
        while True:
            attempt += 1
            try:
                value = fn()
                result = _ExcelComResult(
                    operation_name=operation_name,
                    value=value,
                    attempt=attempt,
                    retry_count=max(0, attempt - 1),
                    retry_backoff_ms=";".join(str(value) for value in used_backoffs),
                    retryable_error=bool(used_backoffs),
                    final_status="ok",
                    recovered=attempt > 1,
                )
                self._last_excel_com_result = result
                return result
            except Exception as exc:
                retryable = _is_temporary_excel_write_error(exc)
                if not retryable or attempt > len(backoffs):
                    result = _ExcelComResult(
                        operation_name=operation_name,
                        attempt=attempt,
                        retry_count=max(0, attempt - 1),
                        retry_backoff_ms=";".join(str(value) for value in used_backoffs),
                        retryable_error=retryable,
                        final_status=(
                            "excel_unavailable_after_retries"
                            if retryable and used_backoffs
                            else "excel_unavailable"
                        ),
                        recovered=False,
                        exception=exc,
                    )
                    self._last_excel_com_result = result
                    return result
                delay_ms = backoffs[attempt - 1]
                used_backoffs.append(delay_ms)
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)

    def _read_cell_with_retry(self, sheet_name: str, address: str, operation_name: str | None = None) -> Any:
        result = self._excel_com_retry(
            operation_name or f"read_cell:{sheet_name}!{address}",
            lambda: self.bridge.read_cell(sheet_name, address),
        )
        if result.exception is not None:
            raise result.exception
        return result.value

    def _write_cells_with_retry(
        self,
        sheet_name: str,
        plan: Iterable[tuple[str, Any]],
    ) -> _ExcelWriteResult:
        prepared = tuple(plan)
        result = self._excel_com_retry(
            f"write_cells:{sheet_name}",
            lambda: tuple(self.bridge.write_cells(sheet_name, prepared, allow_write=True)),
        )
        if result.exception is not None:
            return _ExcelWriteResult(
                attempt=result.attempt,
                retry_count=result.retry_count,
                retry_backoff_ms=result.retry_backoff_ms,
                final_status=(
                    "excel_unavailable_after_retries"
                    if result.final_status == "excel_unavailable_after_retries"
                    else "excel_write_failed"
                ),
                recovered=False,
                exception=result.exception,
            )
        return _ExcelWriteResult(
            written=tuple(result.value or ()),
            attempt=result.attempt,
            retry_count=result.retry_count,
            retry_backoff_ms=result.retry_backoff_ms,
            final_status="written",
            recovered=result.recovered,
        )

    def _verify_real_write(
        self,
        sheet_name: str,
        row: int,
        plan: Iterable[tuple[str, Any]],
    ) -> _PostWriteVerification:
        expected = {str(address).upper(): value for address, value in plan}
        odds_address = self.layout.odds_address(row)
        stake_address = self.layout.stake_address(row)
        trigger_address = self.layout.trigger_address(row)
        odds_value = stake_value = trigger_value = None
        try:
            odds_value = self._read_cell_with_retry(
                sheet_name,
                odds_address,
                f"verify_read_odds:{sheet_name}!{odds_address}",
            )
            stake_value = self._read_cell_with_retry(
                sheet_name,
                stake_address,
                f"verify_read_stake:{sheet_name}!{stake_address}",
            )
            trigger_value = self._read_cell_with_retry(
                sheet_name,
                trigger_address,
                f"verify_read_trigger:{sheet_name}!{trigger_address}",
            )
        except Exception:
            return _PostWriteVerification(
                odds_address,
                odds_value,
                stake_address,
                stake_value,
                trigger_address,
                trigger_value,
                False,
            )
        verified = all(
            _values_match(expected.get(address), actual)
            for address, actual in (
                (odds_address, odds_value),
                (stake_address, stake_value),
                (trigger_address, trigger_value),
            )
        )
        return _PostWriteVerification(
            odds_address,
            odds_value,
            stake_address,
            stake_value,
            trigger_address,
            trigger_value,
            verified,
        )

    def _finish(
        self,
        intent: OrderIntent,
        context: GrussRealOrderContext,
        status: str,
        reason: str,
        *,
        excel_sheet: str = "",
        excel_row: int | None = None,
        excel_cells_written: Iterable[str] = (),
        write_plan: Iterable[tuple[str, Any]] = (),
        trigger_written: bool = False,
        trigger_cell_address: str = "",
        trigger_cell_current_value: Any = None,
        trigger_cell_expected_empty: bool | None = None,
        trigger_mapping_name: str = "",
        trigger_value_written: str = "",
        action: str = "",
        trigger_clear_attempted: bool = False,
        trigger_cleared: bool = False,
        trigger_clear_reason: str = "",
        trigger_cell_value_before_clear: Any = None,
        trigger_clear_delay_ms: int = 0,
        command_cells_clear_attempted: bool | None = None,
        command_cells_cleared: bool | None = None,
        command_cells_clear_reason: str = "",
        command_cells_clear_addresses: str = "",
        command_cells_clear_delay_ms: int | None = None,
        command_cells_clear_scheduled: bool = False,
        command_cells_clear_due_time: str = "",
        command_cells_clear_non_blocking: bool = False,
        command_cells_clear_executed: bool | None = None,
        command_cells_clear_lag_ms: int | None = None,
        post_write_verification: _PostWriteVerification | None = None,
        hold_trigger_for_visual_test: bool = False,
        stake_capped: bool = False,
        stake_cap_value: float | None = None,
        bet_ref_before: str = "",
        bet_ref_after: str = "",
        bet_ref_poll_attempts: int = 0,
        bet_ref_poll_duration_ms: int = 0,
        pre_bet_ref_required: bool | None = None,
        pre_bet_ref_confirmed: bool | None = None,
        pre_bet_ref_missing: bool | None = None,
        pre_bet_ref_poll_attempts: int | None = None,
        pre_bet_ref_poll_duration_ms: int | None = None,
        pre_bet_ref_missing_retryable: bool = False,
        pre_bet_ref_late_detected: bool = False,
        pre_bet_ref_late_value: str = "",
        pre_retry_count: int | None = None,
        pre_retry_allowed: bool = False,
        pre_retry_reason: str = "",
        pre_retry_block_reason: str = "",
        pre_unconfirmed_reason: str = "",
        bet_ref_lookup_source: str = "",
        bet_ref_lookup_matched_runner: str = "",
        active_ladder_bet_ref_stored: bool = False,
        active_ladder_created: bool = False,
        pending_ladder_created: bool = False,
        matched_evidence_found: bool = False,
        selection_row_evidence_found: bool = False,
        no_stacking_blocked_retry: bool = False,
        update_allowed: bool = False,
        update_skipped_reason: str = "",
        intended_trigger_override: str | None = None,
        active_pre_ladder_id_snapshot: str | None = None,
        active_ladder_completed: bool = False,
        active_ladder_release_reason: str = "",
        replace_allowed: bool = False,
        replace_trigger: str = "",
        bet_ref_suffix_n_handled: bool = False,
        bet_ref_status_value: str = "",
        replace_bet_ref_wait_attempted: bool = False,
        replace_bet_ref_wait_ms: int = 0,
        replace_bet_ref_poll_ms: int = 0,
        replace_bet_ref_wait_result: str = "",
        bet_ref_before_wait: str = "",
        bet_ref_after_wait: str = "",
        active_ladder_bet_ref_updated: bool = False,
        replace_skipped_bet_ref_still_pending: bool = False,
        matched_stake_cell_address: str = "",
        matched_stake_cell_value: Any = None,
        countdown_at_write: int | None = None,
        current_market_price_at_write: float | None = None,
        stale_distance: float | None = None,
        stale_price_limit: float | None = None,
        stale_check_ignored_for_pre: bool = False,
        no_stacking_check_passed: bool = False,
        pre_ladder_initial_order_failed: bool = False,
        pre_ladder_disabled_after_initial_failure: bool = False,
        no_replace_steps_for_failed_initial: bool = False,
        pre_cancel_attempted: bool | None = None,
        pre_cancel_written: bool | None = None,
        pre_cancel_skip_reason: str = "",
        pre_cancel_only_if_post_pending: bool | None = None,
        post_pending_for_runner: bool | None = None,
        post_after_pre_cancel_attempted: bool | None = None,
        post_bet_ref_required: bool | None = None,
        post_batch_id: str = "",
        post_batch_market_id: str = "",
        post_batch_market_name: str = "",
        post_batch_candidate_count: int | None = None,
        post_batch_written_count: int | None = None,
        post_batch_write_duration_ms: int | None = None,
        post_batch_confirmation_started: bool = False,
        post_batch_confirmation_duration_ms: int | None = None,
        post_batch_runner_index: int | None = None,
        post_batch_total_runners: int | None = None,
        post_bet_ref_wait_attempted: bool = False,
        post_bet_ref_wait_ms: int = 0,
        post_bet_ref_poll_ms: int = 0,
        post_existing_bet_ref_before: str = "",
        post_existing_pre_bet_ref: str = "",
        post_existing_matched_before: float | None = None,
        post_existing_pre_matched_stake: float | None = None,
        post_existing_avg_odds_before: float | None = None,
        post_existing_pre_avg_odds: float | None = None,
        post_independent_mode_enabled: bool = False,
        post_row_prepared_for_new_order: bool = False,
        post_pre_bet_ref_cleared_for_write: bool = False,
        post_pre_bet_ref_preserved_in_state: bool = False,
        post_new_bet_ref_expected: bool = False,
        post_new_bet_ref_found: bool = False,
        post_new_bet_ref: str = "",
        post_added_stake_confirmed: bool = False,
        post_added_stake_amount: float | None = None,
        post_total_matched_before: float | None = None,
        post_total_matched_after: float | None = None,
        post_total_matched_delta: float | None = None,
        post_expected_market_id: str = "",
        post_expected_market_type: str = "",
        post_expected_runner: str = "",
        post_expected_selection_id: str = "",
        post_expected_side: str = "",
        post_expected_stake: float | None = None,
        post_expected_price: float | None = None,
        post_write_timestamp: str = "",
        post_order_write_timestamp: str = "",
        post_bet_ref_after: str = "",
        post_bet_ref_changed: bool = False,
        post_bet_ref_confirmed_new: bool = False,
        post_bet_ref_poll_attempts: int = 0,
        post_bet_ref_poll_duration_ms: int = 0,
        post_order_confirmed: bool | None = None,
        post_order_confirmation_source: str = "",
        post_confirmation_source: str = "",
        post_selections_lookup_attempted: bool = False,
        post_selections_match_found: bool = False,
        post_selections_match_reason: str = "",
        post_selections_reject_reason: str = "",
        post_clear_after_bet_ref: bool | None = None,
        post_cells_clear_delay_ms: int = 0,
        post_cells_cleared_after_confirmation: bool | None = None,
        post_cells_cleared_after_unconfirmed: bool = False,
        post_clear_reason: str = "",
        post_write_unconfirmed_reason: str = "",
        post_unconfirmed_reason: str = "",
        post_reject_reason: str = "",
        excel_write_attempt: int = 0,
        excel_write_retry_count: int = 0,
        excel_write_retry_backoff_ms: str = "",
        excel_write_final_status: str = "",
        excel_unavailable_recovered: bool = False,
        excel_operation_name: str = "",
        excel_com_attempt: int = 0,
        excel_com_retry_count: int = 0,
        excel_com_retry_backoff_ms: str = "",
        excel_com_retryable_error: bool = False,
        mapping_attempt_count: int = 0,
        cleanup_retry_count: int = 0,
        cleanup_final_status: str = "",
    ) -> GrussRealOrderResult:
        addresses = tuple(excel_cells_written)
        plan = tuple(write_plan)
        if (
            _is_pre_ladder_intent(intent)
            and _ladder_step_index(intent.ladder_step) == 0
            and status != "GRUSS_PRE_LADDER_WRITTEN"
        ):
            pre_ladder_initial_order_failed = True
            pre_ladder_disabled_after_initial_failure = True
            no_replace_steps_for_failed_initial = True
        if intended_trigger_override is not None:
            intended_trigger = intended_trigger_override
        else:
            try:
                intended_trigger = self._trigger_for(intent)
            except Exception:
                intended_trigger = ""
        pending_release_id = self._pending_active_ladder_release_id
        if pending_release_id and not active_ladder_completed:
            active_ladder_completed = True
            active_ladder_release_reason = self._pending_active_ladder_release_reason
        command_cells_clear_attempted_value = (
            trigger_clear_attempted
            if command_cells_clear_attempted is None
            else bool(command_cells_clear_attempted)
        )
        command_cells_cleared_value = (
            trigger_cleared
            if command_cells_cleared is None
            else bool(command_cells_cleared)
        )
        command_cells_clear_reason_value = command_cells_clear_reason or trigger_clear_reason
        if not command_cells_clear_reason and trigger_clear_reason == "trigger_cleared":
            command_cells_clear_reason_value = "command_cells_cleared"
        elif not command_cells_clear_reason and trigger_clear_reason == "trigger_clear_skipped_value_changed":
            command_cells_clear_reason_value = "command_cells_clear_skipped_trigger_changed"
        elif not command_cells_clear_reason and trigger_clear_reason.startswith("trigger_clear_failed"):
            command_cells_clear_reason_value = trigger_clear_reason.replace(
                "trigger_clear_failed",
                "command_cells_clear_failed",
                1,
            )
        command_cells_clear_delay_ms_value = (
            trigger_clear_delay_ms
            if command_cells_clear_delay_ms is None
            else int(command_cells_clear_delay_ms)
        )
        command_cells_clear_executed_value = (
            bool(command_cells_cleared_value)
            if command_cells_clear_executed is None
            else bool(command_cells_clear_executed)
        )
        if not command_cells_clear_addresses and trigger_clear_attempted and trigger_cell_address:
            command_cells_clear_addresses = ";".join(self._command_cell_addresses_for_trigger(trigger_cell_address))
        active_pre_ladder_id_value = (
            active_pre_ladder_id_snapshot
            if active_pre_ladder_id_snapshot is not None
            else pending_release_id or self.active_pre_ladder_id or ""
        )
        command_cells = (
            ";".join(self._command_cell_addresses_for_trigger(self.layout.trigger_address(excel_row)))
            if excel_row is not None
            else ""
        )
        odds_address = self.layout.odds_address(excel_row) if excel_row is not None else ""
        price_tick_rounded = _odds_value_from_plan(plan, odds_address) if odds_address else None
        should_log_tick_price = _should_tick_round_limit_price(intent, intended_trigger)
        price_raw_before_tick = None
        price_tick_rounding_side = ""
        price_is_valid_betfair_tick: bool | None = None
        if should_log_tick_price:
            price_raw_before_tick = _positive_float_or_none(intent.price)
            if price_tick_rounded is None and price_raw_before_tick is not None:
                price_tick_rounded = _price_to_write_for_gruss(intent, intended_trigger)
            upper_side = str(intent.side or "").upper()
            price_tick_rounding_side = "BACK_CEIL" if upper_side == "BACK" else "LAY_FLOOR"
            price_is_valid_betfair_tick = _is_valid_betfair_tick(price_tick_rounded)
        mapping_found = excel_row is not None
        mapping_reason = "mapping_found" if mapping_found else "mapping_missing_for_runner"
        avg_matched_odds_cell_address = self.layout.avg_matched_odds_address(excel_row) if excel_row is not None else ""
        matched_stake_cell_address = matched_stake_cell_address or (
            self.layout.matched_stake_address(excel_row) if excel_row is not None else ""
        )
        profit_loss_cell_address = self.layout.profit_loss_address(excel_row) if excel_row is not None else ""
        avg_matched_odds_cell_value = _read_cell_quiet(
            self.bridge,
            excel_sheet,
            avg_matched_odds_cell_address,
        )
        if matched_stake_cell_value is None:
            matched_stake_cell_value = _read_cell_quiet(
                self.bridge,
                excel_sheet,
                matched_stake_cell_address,
            )
        profit_loss_cell_value = _read_cell_quiet(
            self.bridge,
            excel_sheet,
            profit_loss_cell_address,
        )
        matched_after_step_stake = _matched_stake_value(matched_stake_cell_value)
        matched_after_step_avg_odds = _positive_float_or_none(avg_matched_odds_cell_value)
        stake_before_min_floor = _finite_float_or_none(
            intent.stake_original if intent.stake_original is not None else intent.stake
        )
        stake_final = _finite_float_or_none(intent.stake)
        stake_min_floor_applied = bool(
            stake_before_min_floor is not None
            and stake_final is not None
            and 0.0 < stake_before_min_floor < 1.0
            and stake_final >= 1.0
        )
        stake_after_min_floor = (
            1.0
            if stake_min_floor_applied
            else stake_before_min_floor
        )
        internal_ladder_step_index = _ladder_step_index(intent.ladder_step) if intent.ladder_step else None
        ladder_step_index = (
            internal_ladder_step_index + 1
            if internal_ladder_step_index is not None and internal_ladder_step_index >= 0
            else None
        )
        ladder_step_count = _ladder_step_count(intent.ladder_step)
        direct_lim_order_planned_value = bool(
            getattr(intent, "direct_lim_order_planned", False)
            or getattr(intent, "direct_lim_order_written", False)
        )
        direct_lim_order_written_value = bool(
            direct_lim_order_planned_value
            and _is_pre_ladder_intent(intent)
            and internal_ladder_step_index == 0
            and status == "GRUSS_PRE_LADDER_WRITTEN"
            and trigger_written
        )
        is_post = _execution_phase(intent) == "POST"
        post_countdown_at_write = countdown_at_write if countdown_at_write is not None else context.countdown_seconds
        post_market_status = ""
        if is_post and excel_sheet:
            post_market_status = _read_cell_quiet(self.bridge, excel_sheet, "F2")
        post_write_attempted = bool(
            is_post
            and not self.preview
            and not self.write_no_trigger_guard
            and (bool(addresses) or status in {"GRUSS_REAL_WRITTEN", "GRUSS_WRITE_FAILED"})
        )
        pre_batch_grace = (
            context.pre_batch_write_grace_seconds
            if context.pre_batch_write_grace_seconds is not None
            else (_pre_initial_batch_write_grace_seconds() if context.pre_batch_milestone_authorized else None)
        )
        pre_batch_late_seconds = _pre_batch_late_write_seconds_after_start(
            context.pre_batch_started_countdown_seconds,
            countdown_at_write,
        )
        pre_batch_late_write_allowed = bool(
            context.pre_batch_milestone_seconds is not None
            and countdown_at_write is not None
            and _is_pre_ladder_intent(intent)
            and internal_ladder_step_index == 0
            and not _countdown_in_pre_ladder_window(context.pre_batch_milestone_seconds, countdown_at_write)
            and _pre_batch_late_write_allowed(
                intent,
                context,
                context.pre_batch_milestone_seconds,
                countdown_at_write,
            )
        )
        is_pre_cancel = (
            str(getattr(intent, "strategy_id", "") or "") == "PRE_CANCEL_BEFORE_POST"
            or str(getattr(intent, "order_type", "") or "").upper() == "CANCEL"
            or str(trigger_mapping_name or intended_trigger or "").upper() == self.layout.cancel_trigger
        )
        pre_cancel_written_value = bool(
            is_pre_cancel and trigger_written and status == "GRUSS_PRE_CANCEL_BEFORE_POST_WRITTEN"
        ) if pre_cancel_written is None else bool(pre_cancel_written)
        pre_cancel_attempted_value = bool(is_pre_cancel) if pre_cancel_attempted is None else bool(pre_cancel_attempted)
        pre_cancel_skip_reason_value = pre_cancel_skip_reason or (
            str(self._batch_log_context.get("pre_cancel_skip_reason", "") or "")
            if is_post
            else ""
        ) or (
            "" if pre_cancel_written_value else (reason if is_pre_cancel else "")
        )
        pre_cancel_only_if_post_pending_value = (
            _pre_cancel_only_if_post_pending()
            if pre_cancel_only_if_post_pending is None and is_pre_cancel
            else bool(pre_cancel_only_if_post_pending)
        )
        post_pending_for_runner_value = (
            bool(post_pending_for_runner)
            if post_pending_for_runner is not None
            else (True if is_pre_cancel and not pre_cancel_only_if_post_pending_value else False)
        )
        post_after_pre_cancel_attempted_value = (
            bool(post_after_pre_cancel_attempted)
            if post_after_pre_cancel_attempted is not None
            else bool(is_pre_cancel and post_pending_for_runner_value)
        )
        bet_ref_at_cancel = bet_ref_before if is_pre_cancel else ""
        matched_stake_at_cancel = (
            _matched_stake_value(matched_stake_cell_value)
            if is_pre_cancel and matched_stake_cell_value is not None
            else None
        )
        countdown_seconds_at_cancel = (
            countdown_at_write if countdown_at_write is not None else context.countdown_seconds
        ) if is_pre_cancel else None
        post_provider_called_value = (
            bool(self._batch_log_context.get("post_provider_called"))
            if is_post and self._batch_log_context.get("post_provider_called") not in (None, "")
            else (True if is_post else None)
        )
        post_bet_ref_required_value = (
            _post_bet_ref_required()
            if post_bet_ref_required is None and is_post
            else bool(post_bet_ref_required)
        )
        post_clear_after_bet_ref_value = (
            _post_clear_after_bet_ref()
            if post_clear_after_bet_ref is None and is_post
            else bool(post_clear_after_bet_ref)
        )
        post_order_confirmed_value = (
            bool(post_order_confirmed)
            if post_order_confirmed is not None
            else bool(is_post and post_bet_ref_required_value and is_valid_bet_ref(post_bet_ref_after or bet_ref_after))
        )
        post_cells_clear_delay_ms_value = int(post_cells_clear_delay_ms or 0)
        post_cells_cleared_after_confirmation_value = (
            bool(post_cells_cleared_after_confirmation)
            if post_cells_cleared_after_confirmation is not None
            else False
        )
        excel_com_result = self._last_excel_com_result
        excel_operation_name_value = excel_operation_name or excel_com_result.operation_name
        excel_com_attempt_value = excel_com_attempt or excel_com_result.attempt
        excel_com_retry_count_value = excel_com_retry_count or excel_com_result.retry_count
        excel_com_retry_backoff_ms_value = excel_com_retry_backoff_ms or excel_com_result.retry_backoff_ms
        excel_com_retryable_error_value = (
            bool(excel_com_retryable_error) or bool(excel_com_result.retryable_error)
        )
        mapping_attempt_count_value = mapping_attempt_count or self._last_mapping_com_result.attempt
        excel_unavailable_recovered_value = bool(
            excel_unavailable_recovered
            or excel_com_result.recovered
            or self._last_mapping_com_result.recovered
            or self._last_cleanup_com_result.recovered
        )
        cleanup_retry_count_value = cleanup_retry_count or self._last_cleanup_com_result.retry_count
        cleanup_final_status_value = cleanup_final_status or self._last_cleanup_com_result.final_status
        self._append_attempt(
            intent,
            context,
            status,
            reason,
            excel_sheet=excel_sheet,
            excel_row=excel_row,
            excel_cells_written=addresses,
            trigger_written=trigger_written,
            intended_trigger=intended_trigger,
            trigger_cell_address=trigger_cell_address,
            trigger_cell_current_value=trigger_cell_current_value,
            trigger_cell_expected_empty=trigger_cell_expected_empty,
            trigger_mapping_name=trigger_mapping_name or intended_trigger,
            trigger_value_written=trigger_value_written,
            action=action or status,
            trigger_clear_attempted=trigger_clear_attempted,
            trigger_cleared=trigger_cleared,
            trigger_clear_reason=trigger_clear_reason,
            trigger_cell_value_before_clear=trigger_cell_value_before_clear,
            trigger_clear_delay_ms=trigger_clear_delay_ms,
            command_cells_clear_attempted=command_cells_clear_attempted_value,
            command_cells_cleared=command_cells_cleared_value,
            command_cells_clear_reason=command_cells_clear_reason_value,
            command_cells_clear_addresses=command_cells_clear_addresses,
            command_cells_clear_delay_ms=command_cells_clear_delay_ms_value,
            command_cells_clear_scheduled=command_cells_clear_scheduled,
            command_cells_clear_due_time=command_cells_clear_due_time,
            command_cells_clear_non_blocking=command_cells_clear_non_blocking,
            command_cells_clear_executed=command_cells_clear_executed_value,
            command_cells_clear_lag_ms=command_cells_clear_lag_ms,
            startup_command_cells_cleanup_attempted=False,
            startup_command_cells_cleanup_done=False,
            stale_command_cells_cleanup_attempted=False,
            stale_command_cells_cleanup_addresses="",
            stale_command_cells_cleanup_reason="",
            post_write_verification=post_write_verification,
            hold_trigger_for_visual_test=hold_trigger_for_visual_test,
            stake_capped=stake_capped,
            stake_cap_value=stake_cap_value,
            bet_ref_before=bet_ref_before,
            bet_ref_after=bet_ref_after,
            bet_ref_poll_attempts=bet_ref_poll_attempts,
            bet_ref_poll_duration_ms=bet_ref_poll_duration_ms,
            pre_write_attempt_id=_pre_write_attempt_id(intent, context, excel_write_attempt or 0),
            pre_bet_ref_required=(
                _is_pre_ladder_intent(intent)
                and (_ladder_step_index(intent.ladder_step) == 0 or bool(pre_retry_allowed))
                if pre_bet_ref_required is None
                else bool(pre_bet_ref_required)
            ),
            pre_bet_ref_confirmed=bool(pre_bet_ref_confirmed),
            pre_bet_ref_missing=bool(pre_bet_ref_missing),
            pre_bet_ref_poll_attempts=(
                bet_ref_poll_attempts if pre_bet_ref_poll_attempts is None else pre_bet_ref_poll_attempts
            ),
            pre_bet_ref_poll_duration_ms=(
                bet_ref_poll_duration_ms
                if pre_bet_ref_poll_duration_ms is None
                else pre_bet_ref_poll_duration_ms
            ),
            pre_bet_ref_missing_retryable=pre_bet_ref_missing_retryable,
            pre_bet_ref_late_detected=pre_bet_ref_late_detected,
            pre_bet_ref_late_value=pre_bet_ref_late_value,
            pre_retry_count=(
                _pre_retry_count_for_intent(self, intent)
                if pre_retry_count is None
                else int(pre_retry_count)
            ),
            pre_retry_allowed=pre_retry_allowed,
            pre_retry_reason=pre_retry_reason,
            pre_retry_block_reason=pre_retry_block_reason,
            pre_unconfirmed_reason=pre_unconfirmed_reason,
            bet_ref_lookup_sources=str(self._batch_log_context.get("bet_ref_lookup_sources") or ""),
            bet_ref_lookup_source_used="",
            bet_ref_lookup_source=bet_ref_lookup_source,
            bet_ref_lookup_matched_runner=bet_ref_lookup_matched_runner,
            row_t_value="",
            selections_rows_scanned=0,
            selections_match_found=False,
            selections_match_reason="",
            selections_runner="",
            selections_side="",
            selections_stake=None,
            selections_bet_ref="",
            selections_req_odds=None,
            selections_market_name="",
            selections_debug_recent_rows="",
            selections_top_candidates="",
            bet_ref_row_t_dump="",
            bet_ref_diagnostic_hold_after_batch=False,
            selections_market_query="",
            selections_current_market_rows="",
            selections_current_runner_rows="",
            runner_qz_dump="",
            selections_sheet_headers="",
            selections_full_recent_rows="",
            workbook_sheet_names="",
            diagnostic_keep_triggers=False,
            active_ladder_bet_ref_stored=active_ladder_bet_ref_stored,
            active_ladder_created=active_ladder_created,
            pending_ladder_created=pending_ladder_created,
            matched_evidence_found=matched_evidence_found,
            selection_row_evidence_found=selection_row_evidence_found,
            no_stacking_blocked_retry=no_stacking_blocked_retry,
            replace_allowed=replace_allowed,
            replace_trigger=replace_trigger,
            bet_ref_suffix_n_handled=bet_ref_suffix_n_handled,
            bet_ref_status_value=bet_ref_status_value,
            replace_bet_ref_wait_attempted=replace_bet_ref_wait_attempted,
            replace_bet_ref_wait_ms=replace_bet_ref_wait_ms,
            replace_bet_ref_poll_ms=replace_bet_ref_poll_ms,
            replace_bet_ref_wait_result=replace_bet_ref_wait_result,
            bet_ref_before_wait=bet_ref_before_wait,
            bet_ref_after_wait=bet_ref_after_wait,
            active_ladder_bet_ref_updated=active_ladder_bet_ref_updated,
            replace_skipped_bet_ref_still_pending=replace_skipped_bet_ref_still_pending,
            matched_stake_cell_address=matched_stake_cell_address,
            matched_stake_cell_value=matched_stake_cell_value,
            update_allowed=update_allowed,
            update_skipped_reason=update_skipped_reason,
            active_pre_ladder_id_snapshot=active_pre_ladder_id_value,
            active_ladder_completed=active_ladder_completed,
            active_ladder_release_reason=active_ladder_release_reason,
            countdown_at_write=countdown_at_write,
            current_market_price_at_write=current_market_price_at_write,
            stale_distance=stale_distance,
            stale_price_limit=stale_price_limit,
            stale_check_ignored_for_pre=stale_check_ignored_for_pre,
            pre_batch_late_write_allowed=pre_batch_late_write_allowed,
            pre_batch_late_write_seconds_after_start=pre_batch_late_seconds,
            pre_cancel_attempted=pre_cancel_attempted_value,
            pre_cancel_written=pre_cancel_written_value,
            pre_cancel_skip_reason=pre_cancel_skip_reason_value,
            pre_cancel_only_if_post_pending=pre_cancel_only_if_post_pending_value,
            post_pending_for_runner=post_pending_for_runner_value,
            post_after_pre_cancel_attempted=post_after_pre_cancel_attempted_value,
            bet_ref_at_cancel=bet_ref_at_cancel,
            matched_stake_at_cancel=matched_stake_at_cancel,
            countdown_seconds_at_cancel=countdown_seconds_at_cancel,
            post_provider_called=post_provider_called_value,
            post_batch_id=post_batch_id if is_post else "",
            post_batch_market_id=post_batch_market_id if is_post else "",
            post_batch_market_name=post_batch_market_name if is_post else "",
            post_batch_candidate_count=post_batch_candidate_count if is_post else None,
            post_batch_written_count=post_batch_written_count if is_post else None,
            post_batch_write_duration_ms=post_batch_write_duration_ms if is_post else None,
            post_batch_confirmation_started=post_batch_confirmation_started if is_post else False,
            post_batch_confirmation_duration_ms=post_batch_confirmation_duration_ms if is_post else None,
            post_batch_runner_index=post_batch_runner_index if is_post else None,
            post_batch_total_runners=post_batch_total_runners if is_post else None,
            post_send_seconds_before_off=_post_send_seconds_before_off() if is_post else None,
            post_allow_after_scheduled_off_seconds=_post_allow_after_scheduled_off_seconds() if is_post else None,
            post_trigger_window_hit=(
                _post_countdown_allowed(post_countdown_at_write)
                if is_post and post_countdown_at_write is not None
                else None
            ),
            post_write_attempted=post_write_attempted if is_post else None,
            post_write_status=status if is_post else "",
            post_write_reason=reason if is_post else "",
            post_bet_ref_required=post_bet_ref_required_value if is_post else False,
            post_bet_ref_wait_attempted=post_bet_ref_wait_attempted if is_post else False,
            post_bet_ref_wait_ms=post_bet_ref_wait_ms if is_post else 0,
            post_bet_ref_poll_ms=post_bet_ref_poll_ms if is_post else 0,
            post_existing_bet_ref_before=post_existing_bet_ref_before if is_post else "",
            post_existing_pre_bet_ref=(post_existing_pre_bet_ref or post_existing_bet_ref_before) if is_post else "",
            post_existing_matched_before=post_existing_matched_before if is_post else None,
            post_existing_pre_matched_stake=(
                post_existing_pre_matched_stake
                if post_existing_pre_matched_stake is not None
                else post_existing_matched_before
            ) if is_post else None,
            post_existing_avg_odds_before=post_existing_avg_odds_before if is_post else None,
            post_existing_pre_avg_odds=(
                post_existing_pre_avg_odds
                if post_existing_pre_avg_odds is not None
                else post_existing_avg_odds_before
            ) if is_post else None,
            post_independent_mode_enabled=post_independent_mode_enabled if is_post else False,
            post_row_prepared_for_new_order=post_row_prepared_for_new_order if is_post else False,
            post_pre_bet_ref_cleared_for_write=post_pre_bet_ref_cleared_for_write if is_post else False,
            post_pre_bet_ref_preserved_in_state=post_pre_bet_ref_preserved_in_state if is_post else False,
            post_new_bet_ref_expected=post_new_bet_ref_expected if is_post else False,
            post_new_bet_ref_found=post_new_bet_ref_found if is_post else False,
            post_new_bet_ref=post_new_bet_ref if is_post else "",
            post_added_stake_confirmed=post_added_stake_confirmed if is_post else False,
            post_added_stake_amount=post_added_stake_amount if is_post else None,
            post_total_matched_before=post_total_matched_before if is_post else None,
            post_total_matched_after=post_total_matched_after if is_post else None,
            post_total_matched_delta=post_total_matched_delta if is_post else None,
            post_expected_market_id=post_expected_market_id if is_post else "",
            post_expected_market_type=post_expected_market_type if is_post else "",
            post_expected_runner=post_expected_runner if is_post else "",
            post_expected_selection_id=post_expected_selection_id if is_post else "",
            post_expected_side=post_expected_side if is_post else "",
            post_expected_stake=post_expected_stake if is_post else None,
            post_expected_price=post_expected_price if is_post else None,
            post_write_timestamp=post_write_timestamp if is_post else "",
            post_order_write_timestamp=(post_order_write_timestamp or post_write_timestamp) if is_post else "",
            post_bet_ref_after=post_bet_ref_after if is_post else "",
            post_bet_ref_changed=post_bet_ref_changed if is_post else False,
            post_bet_ref_confirmed_new=post_bet_ref_confirmed_new if is_post else False,
            post_bet_ref_poll_attempts=post_bet_ref_poll_attempts if is_post else 0,
            post_bet_ref_poll_duration_ms=post_bet_ref_poll_duration_ms if is_post else 0,
            post_order_confirmed=post_order_confirmed_value if is_post else False,
            post_order_confirmation_source=post_order_confirmation_source if is_post else "",
            post_confirmation_source=(post_confirmation_source or post_order_confirmation_source) if is_post else "",
            post_selections_lookup_attempted=post_selections_lookup_attempted if is_post else False,
            post_selections_match_found=post_selections_match_found if is_post else False,
            post_selections_match_reason=post_selections_match_reason if is_post else "",
            post_selections_reject_reason=post_selections_reject_reason if is_post else "",
            post_clear_after_bet_ref=post_clear_after_bet_ref_value if is_post else False,
            post_cells_clear_delay_ms=post_cells_clear_delay_ms_value if is_post else 0,
            post_cells_cleared_after_confirmation=(
                post_cells_cleared_after_confirmation_value if is_post else False
            ),
            post_cells_cleared_after_unconfirmed=post_cells_cleared_after_unconfirmed if is_post else False,
            post_clear_reason=post_clear_reason if is_post else "",
            post_write_unconfirmed_reason=post_write_unconfirmed_reason if is_post else "",
            post_unconfirmed_reason=(post_unconfirmed_reason or post_write_unconfirmed_reason) if is_post else "",
            post_reject_reason=(post_reject_reason or (reason if status != "GRUSS_REAL_WRITTEN" else "")) if is_post else "",
            countdown_seconds_at_post_write=post_countdown_at_write if is_post else None,
            market_status_at_post_write=post_market_status if is_post else "",
            excel_write_attempt=excel_write_attempt,
            excel_write_retry_count=excel_write_retry_count,
            excel_write_retry_backoff_ms=excel_write_retry_backoff_ms,
            excel_write_final_status=excel_write_final_status,
            excel_unavailable_recovered=excel_unavailable_recovered_value,
            excel_operation_name=excel_operation_name_value,
            excel_com_attempt=excel_com_attempt_value,
            excel_com_retry_count=excel_com_retry_count_value,
            excel_com_retry_backoff_ms=excel_com_retry_backoff_ms_value,
            excel_com_retryable_error=excel_com_retryable_error_value,
            mapping_attempt_count=mapping_attempt_count_value,
            cleanup_retry_count=cleanup_retry_count_value,
            cleanup_final_status=cleanup_final_status_value,
            no_stacking_check_passed=no_stacking_check_passed,
            mapping_found=mapping_found,
            mapping_reason=mapping_reason,
            command_cells=command_cells,
            pre_ladder_initial_order_failed=pre_ladder_initial_order_failed,
            pre_ladder_disabled_after_initial_failure=pre_ladder_disabled_after_initial_failure,
            no_replace_steps_for_failed_initial=no_replace_steps_for_failed_initial,
            direct_lim_order_planned=direct_lim_order_planned_value,
            direct_lim_order_written=direct_lim_order_written_value,
            requested_price=intent.price,
            requested_stake=intent.stake,
            price_raw_before_tick=price_raw_before_tick,
            price_tick_rounded=price_tick_rounded,
            price_tick_rounding_side=price_tick_rounding_side,
            price_is_valid_betfair_tick=price_is_valid_betfair_tick,
            ladder_step_index=ladder_step_index,
            ladder_step_count=ladder_step_count,
            matched_after_step=bool(matched_after_step_stake is not None and matched_after_step_stake > 0),
            matched_after_step_avg_odds=matched_after_step_avg_odds,
            matched_after_step_stake=matched_after_step_stake,
            avg_matched_odds_cell_address=avg_matched_odds_cell_address,
            avg_matched_odds_cell_value=avg_matched_odds_cell_value,
            profit_loss_cell_address=profit_loss_cell_address,
            profit_loss_cell_value=profit_loss_cell_value,
        )
        self._pending_active_ladder_release_id = None
        self._pending_active_ladder_release_reason = ""
        return GrussRealOrderResult(
            status=status,
            reason=reason,
            output_path=self.output_path,
            excel_sheet=excel_sheet,
            excel_row=excel_row,
            excel_cells_written=addresses,
            write_plan=plan,
            trigger_written=trigger_written,
            intended_trigger=intended_trigger,
            stake_original=intent.stake_original if intent.stake_original is not None else intent.stake,
            stake_used=intent.stake,
            stake_forced=bool(intent.stake_forced),
            stake_min_floor_applied=stake_min_floor_applied,
            stake_before_min_floor=stake_before_min_floor,
            stake_after_min_floor=stake_after_min_floor,
            stake_final=stake_final,
            stake_capped=stake_capped,
            stake_cap_value=stake_cap_value,
            execution_phase=_execution_phase(intent),
            market_type=str(intent.market_type or ""),
            market_id=str(intent.market_id or ""),
            runner_name=str(intent.runner_name or ""),
            trap=intent.trap,
            selection_id=intent.selection_id if intent.selection_id is not None else intent.trap,
            side=str(intent.side or ""),
            order_type=str(intent.order_type or ""),
            strategy_id=str(intent.strategy_id or ""),
            processed_key=_processed_key(intent, context),
            post_processed_key=str(self._batch_log_context.get("post_processed_key", "")),
            post_processed_key_scope=str(self._batch_log_context.get("post_processed_key_scope", "")),
            parent_id=str(self._batch_log_context.get("parent_id", "")),
            course_id=str(self._batch_log_context.get("course_id", "")),
            win_market_id=str(self._batch_log_context.get("win_market_id", "")),
            place_market_id=str(self._batch_log_context.get("place_market_id", "")),
            processed_key_seen=bool(self._batch_log_context.get("processed_key_seen", False)),
            processed_key_seen_matching_existing_key=str(
                self._batch_log_context.get("processed_key_seen_matching_existing_key", "")
            ),
            pre_post_independent=bool(self._batch_log_context.get("pre_post_independent", False)),
            pre_existing_order_allowed=bool(self._batch_log_context.get("pre_existing_order_allowed", False)),
            pre_cancel_required_before_post=bool(
                self._batch_log_context.get("pre_cancel_required_before_post", False)
            ),
            stake_limit_scope=str(self._batch_log_context.get("stake_limit_scope", "")),
            pre_ladder=bool(getattr(intent, "pre_ladder", False)),
            ladder_id=intent.ladder_id or "",
            ladder_step=intent.ladder_step or "",
            active_pre_ladder_id=active_pre_ladder_id_value,
            continuing_active_pre_ladder=_continuing_active_pre_ladder(intent, active_pre_ladder_id_value),
            active_pre_ladder_count=len(self.active_pre_ladders),
            max_active_pre_ladders=_pre_ladder_real_max_ladders(),
            configured_ladder_steps=",".join(str(step) for step in _pre_ladder_steps_from_env()),
            ladder_plan_frozen=bool(getattr(intent, "ladder_plan_frozen", False)),
            ladder_plan_created_step=getattr(intent, "ladder_plan_created_step", None),
            ladder_prices_frozen=getattr(intent, "ladder_prices_frozen", "") or "",
            current_ladder_price_from_frozen_plan=bool(
                getattr(intent, "current_ladder_price_from_frozen_plan", False)
            ),
            computed_limit_price_raw=getattr(intent, "computed_limit_price_raw", None),
            computed_limit_price_effective=_pre_value_target_effective_from_intent(intent),
            min_price_floor_applied=bool(getattr(intent, "min_price_floor_applied", False)),
            pre_value_target_price=_pre_value_target_effective_from_intent(intent),
            ladder_planned_price=getattr(intent, "ladder_planned_price", None),
            sent_price_before_value_clamp=getattr(intent, "sent_price_before_value_clamp", None),
            sent_price_after_value_clamp=price_tick_rounded,
            value_clamp_applied=bool(getattr(intent, "value_clamp_applied", False))
            or _value_clamp_was_needed(intent),
            value_limit_breached=bool(getattr(intent, "value_limit_breached", False))
            or _value_clamp_was_needed(intent),
            value_limit_skip_reason=getattr(intent, "value_limit_skip_reason", "") or "",
            tick_rounding_direction=getattr(intent, "tick_rounding_direction", "") or price_tick_rounding_side,
            best_same_side_offer_at_creation=getattr(
                intent,
                "best_same_side_offer_at_creation",
                None,
            ),
            best_back_displayed=getattr(intent, "best_back_displayed", None),
            best_lay_displayed=getattr(intent, "best_lay_displayed", None),
            start_price_source=getattr(intent, "start_price_source", "") or "",
            final_lim_price=intent.price,
            ladder_direction=getattr(intent, "ladder_direction", "") or "",
            ladder_disabled_lim_not_in_ladder_direction=bool(
                getattr(intent, "ladder_disabled_lim_not_in_ladder_direction", False)
            ),
            direct_lim_order_planned=direct_lim_order_planned_value,
            direct_lim_order_written=direct_lim_order_written_value,
            no_replace_steps_for_direct_lim=bool(
                getattr(intent, "no_replace_steps_for_direct_lim", False)
            ),
            current_milestone=_current_milestone(context),
            computed_step_index=_ladder_step_index(intent.ladder_step),
            expected_ladder_step=_expected_ladder_step_for_context(context),
            milestone_seen=context.milestone_seen
            if context.milestone_seen is not None
            else context.countdown_seconds,
            next_ladder_step_due=intent.ladder_step or "",
            skipped_step_reason=update_skipped_reason,
            active_ladder_completed=active_ladder_completed,
            active_ladder_release_reason=active_ladder_release_reason,
            signal_timestamp=intent.timestamp,
            write_timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            write_delay_since_signal_seconds=_seconds_since_iso_timestamp(intent.timestamp),
            countdown_at_signal=intent.signal_countdown_seconds,
            countdown_at_write=countdown_at_write,
            no_stacking_check_passed=no_stacking_check_passed,
            market_reference_price_at_signal=intent.market_reference_price_at_signal,
            current_market_price_at_write=current_market_price_at_write,
            stale_distance=stale_distance,
            stale_price_limit=stale_price_limit,
            stale_check_ignored_for_pre=stale_check_ignored_for_pre,
            pre_batch_milestone_authorized=bool(context.pre_batch_milestone_authorized),
            pre_batch_milestone_seconds=context.pre_batch_milestone_seconds,
            pre_batch_started_countdown_seconds=context.pre_batch_started_countdown_seconds,
            pre_batch_write_grace_seconds=pre_batch_grace,
            pre_batch_candidate_index=_int_from_context(
                self._batch_log_context.get("pre_batch_candidate_index")
            ),
            pre_batch_candidates_count=_int_from_context(
                self._batch_log_context.get("pre_batch_candidates_count")
            ),
            pre_batch_late_write_allowed=pre_batch_late_write_allowed,
            pre_batch_late_write_seconds_after_start=pre_batch_late_seconds,
            pre_cancel_attempted=pre_cancel_attempted_value,
            pre_cancel_written=pre_cancel_written_value,
            pre_cancel_skip_reason=pre_cancel_skip_reason_value,
            pre_cancel_only_if_post_pending=pre_cancel_only_if_post_pending_value,
            post_pending_for_runner=post_pending_for_runner_value,
            post_after_pre_cancel_attempted=post_after_pre_cancel_attempted_value,
            bet_ref_at_cancel=bet_ref_at_cancel,
            matched_stake_at_cancel=matched_stake_at_cancel,
            countdown_seconds_at_cancel=countdown_seconds_at_cancel,
            post_provider_called=post_provider_called_value,
            post_batch_id=post_batch_id if is_post else "",
            post_batch_market_id=post_batch_market_id if is_post else "",
            post_batch_market_name=post_batch_market_name if is_post else "",
            post_batch_candidate_count=post_batch_candidate_count if is_post else None,
            post_batch_written_count=post_batch_written_count if is_post else None,
            post_batch_write_duration_ms=post_batch_write_duration_ms if is_post else None,
            post_batch_confirmation_started=post_batch_confirmation_started if is_post else False,
            post_batch_confirmation_duration_ms=post_batch_confirmation_duration_ms if is_post else None,
            post_batch_runner_index=post_batch_runner_index if is_post else None,
            post_batch_total_runners=post_batch_total_runners if is_post else None,
            post_send_seconds_before_off=_post_send_seconds_before_off() if is_post else None,
            post_allow_after_scheduled_off_seconds=_post_allow_after_scheduled_off_seconds() if is_post else None,
            post_trigger_window_hit=(
                _post_countdown_allowed(post_countdown_at_write)
                if is_post and post_countdown_at_write is not None
                else None
            ),
            post_write_attempted=post_write_attempted if is_post else None,
            post_write_status=status if is_post else "",
            post_write_reason=reason if is_post else "",
            post_bet_ref_required=post_bet_ref_required_value if is_post else False,
            post_bet_ref_wait_attempted=post_bet_ref_wait_attempted if is_post else False,
            post_bet_ref_wait_ms=post_bet_ref_wait_ms if is_post else 0,
            post_bet_ref_poll_ms=post_bet_ref_poll_ms if is_post else 0,
            post_existing_bet_ref_before=post_existing_bet_ref_before if is_post else "",
            post_existing_pre_bet_ref=(post_existing_pre_bet_ref or post_existing_bet_ref_before) if is_post else "",
            post_existing_matched_before=post_existing_matched_before if is_post else None,
            post_existing_pre_matched_stake=(
                post_existing_pre_matched_stake
                if post_existing_pre_matched_stake is not None
                else post_existing_matched_before
            ) if is_post else None,
            post_existing_avg_odds_before=post_existing_avg_odds_before if is_post else None,
            post_existing_pre_avg_odds=(
                post_existing_pre_avg_odds
                if post_existing_pre_avg_odds is not None
                else post_existing_avg_odds_before
            ) if is_post else None,
            post_independent_mode_enabled=post_independent_mode_enabled if is_post else False,
            post_row_prepared_for_new_order=post_row_prepared_for_new_order if is_post else False,
            post_pre_bet_ref_cleared_for_write=post_pre_bet_ref_cleared_for_write if is_post else False,
            post_pre_bet_ref_preserved_in_state=post_pre_bet_ref_preserved_in_state if is_post else False,
            post_new_bet_ref_expected=post_new_bet_ref_expected if is_post else False,
            post_new_bet_ref_found=post_new_bet_ref_found if is_post else False,
            post_new_bet_ref=post_new_bet_ref if is_post else "",
            post_added_stake_confirmed=post_added_stake_confirmed if is_post else False,
            post_added_stake_amount=post_added_stake_amount if is_post else None,
            post_total_matched_before=post_total_matched_before if is_post else None,
            post_total_matched_after=post_total_matched_after if is_post else None,
            post_total_matched_delta=post_total_matched_delta if is_post else None,
            post_expected_market_id=post_expected_market_id if is_post else "",
            post_expected_market_type=post_expected_market_type if is_post else "",
            post_expected_runner=post_expected_runner if is_post else "",
            post_expected_selection_id=post_expected_selection_id if is_post else "",
            post_expected_side=post_expected_side if is_post else "",
            post_expected_stake=post_expected_stake if is_post else None,
            post_expected_price=post_expected_price if is_post else None,
            post_write_timestamp=post_write_timestamp if is_post else "",
            post_order_write_timestamp=(post_order_write_timestamp or post_write_timestamp) if is_post else "",
            post_bet_ref_after=post_bet_ref_after if is_post else "",
            post_bet_ref_changed=post_bet_ref_changed if is_post else False,
            post_bet_ref_confirmed_new=post_bet_ref_confirmed_new if is_post else False,
            post_bet_ref_poll_attempts=post_bet_ref_poll_attempts if is_post else 0,
            post_bet_ref_poll_duration_ms=post_bet_ref_poll_duration_ms if is_post else 0,
            post_order_confirmed=post_order_confirmed_value if is_post else False,
            post_order_confirmation_source=post_order_confirmation_source if is_post else "",
            post_confirmation_source=(post_confirmation_source or post_order_confirmation_source) if is_post else "",
            post_selections_lookup_attempted=post_selections_lookup_attempted if is_post else False,
            post_selections_match_found=post_selections_match_found if is_post else False,
            post_selections_match_reason=post_selections_match_reason if is_post else "",
            post_selections_reject_reason=post_selections_reject_reason if is_post else "",
            post_clear_after_bet_ref=post_clear_after_bet_ref_value if is_post else False,
            post_cells_clear_delay_ms=post_cells_clear_delay_ms_value if is_post else 0,
            post_cells_cleared_after_confirmation=(
                post_cells_cleared_after_confirmation_value if is_post else False
            ),
            post_cells_cleared_after_unconfirmed=post_cells_cleared_after_unconfirmed if is_post else False,
            post_clear_reason=post_clear_reason if is_post else "",
            post_write_unconfirmed_reason=post_write_unconfirmed_reason if is_post else "",
            post_unconfirmed_reason=(post_unconfirmed_reason or post_write_unconfirmed_reason) if is_post else "",
            post_reject_reason=(post_reject_reason or (reason if status != "GRUSS_REAL_WRITTEN" else "")) if is_post else "",
            countdown_seconds_at_post_write=post_countdown_at_write if is_post else None,
            market_status_at_post_write=post_market_status if is_post else "",
            mapping_found=mapping_found,
            mapping_reason=mapping_reason,
            command_cells=command_cells,
            total_runners_in_gruss_sheet=_int_from_context(
                self._batch_log_context.get("total_runners_in_gruss_sheet")
            ),
            raw_gruss_runner_rows=str(self._batch_log_context.get("raw_gruss_runner_rows") or ""),
            raw_selection_ids_seen=str(self._batch_log_context.get("raw_selection_ids_seen") or ""),
            raw_runner_names_seen=str(self._batch_log_context.get("raw_runner_names_seen") or ""),
            mapped_runners_count=_int_from_context(self._batch_log_context.get("mapped_runners_count")),
            unmapped_runners_count=_int_from_context(self._batch_log_context.get("unmapped_runners_count")),
            mapped_selection_ids=str(self._batch_log_context.get("mapped_selection_ids") or ""),
            unmapped_selection_ids=str(self._batch_log_context.get("unmapped_selection_ids") or ""),
            ignored_runner_rows=str(self._batch_log_context.get("ignored_runner_rows") or ""),
            ignored_runner_reason=str(self._batch_log_context.get("ignored_runner_reason") or ""),
            mapped_excel_rows=str(self._batch_log_context.get("mapped_excel_rows") or ""),
            excel_write_attempt=excel_write_attempt,
            excel_write_retry_count=excel_write_retry_count,
            excel_write_retry_backoff_ms=excel_write_retry_backoff_ms,
            excel_write_final_status=excel_write_final_status,
            excel_unavailable_recovered=excel_unavailable_recovered_value,
            excel_operation_name=excel_operation_name_value,
            excel_com_attempt=excel_com_attempt_value,
            excel_com_retry_count=excel_com_retry_count_value,
            excel_com_retry_backoff_ms=excel_com_retry_backoff_ms_value,
            excel_com_retryable_error=excel_com_retryable_error_value,
            mapping_attempt_count=mapping_attempt_count_value,
            cleanup_retry_count=cleanup_retry_count_value,
            cleanup_final_status=cleanup_final_status_value,
            conflict_detected=bool(intent.conflict_detected),
            conflict_type=getattr(intent, "conflict_type", "") or "",
            back_price=intent.back_price,
            lay_price=intent.lay_price,
            market_reference_price=intent.market_reference_price,
            back_distance=intent.back_distance,
            lay_distance=intent.lay_distance,
            selected_side=intent.selected_side or "",
            rejected_side=intent.rejected_side or "",
            conflict_group_key=getattr(intent, "conflict_group_key", "") or "",
            conflict_candidates_count=_int_from_context(getattr(intent, "conflict_candidates_count", None)),
            winning_side=getattr(intent, "winning_side", "") or "",
            losing_side=getattr(intent, "losing_side", "") or "",
            winning_strategy_id=getattr(intent, "winning_strategy_id", "") or "",
            losing_strategy_id=getattr(intent, "losing_strategy_id", "") or "",
            winning_edge=getattr(intent, "winning_edge", None),
            losing_edge=getattr(intent, "losing_edge", None),
            winning_score=getattr(intent, "winning_score", None),
            losing_score=getattr(intent, "losing_score", None),
            winning_lim_price=getattr(intent, "winning_lim_price", None),
            losing_lim_price=getattr(intent, "losing_lim_price", None),
            back_systems=getattr(intent, "back_systems", "") or "",
            lay_systems=getattr(intent, "lay_systems", "") or "",
            conflict_resolution_reason=intent.conflict_resolution_reason or "",
            pre_back_lay_conflict=bool(getattr(intent, "pre_back_lay_conflict", False)),
            pre_conflict_resolution=getattr(intent, "pre_conflict_resolution", "") or "",
            pre_conflict_chosen_side=getattr(intent, "pre_conflict_chosen_side", "") or "",
            pre_conflict_rejected_side=getattr(intent, "pre_conflict_rejected_side", "") or "",
            pre_conflict_reason=getattr(intent, "pre_conflict_reason", "") or "",
            pre_conflict_group_key=getattr(intent, "pre_conflict_group_key", "") or "",
            pre_conflict_course_id=getattr(intent, "pre_conflict_course_id", "") or "",
            pre_conflict_market_id=getattr(intent, "pre_conflict_market_id", "") or "",
            pre_conflict_market_type=getattr(intent, "pre_conflict_market_type", "") or "",
            pre_conflict_selection_id=getattr(intent, "pre_conflict_selection_id", "") or "",
            pre_conflict_runner_name=getattr(intent, "pre_conflict_runner_name", "") or "",
            pre_back_target_price=getattr(intent, "pre_back_target_price", None),
            pre_lay_target_price=getattr(intent, "pre_lay_target_price", None),
            pre_current_best_lay=getattr(intent, "pre_current_best_lay", None),
            pre_current_best_back=getattr(intent, "pre_current_best_back", None),
            pre_back_distance_ticks=getattr(intent, "pre_back_distance_ticks", None),
            pre_lay_distance_ticks=getattr(intent, "pre_lay_distance_ticks", None),
            trigger_cell_address=trigger_cell_address,
            trigger_cell_current_value=trigger_cell_current_value,
            trigger_cell_expected_empty=trigger_cell_expected_empty,
            trigger_mapping_name=trigger_mapping_name or intended_trigger,
            trigger_value_written=trigger_value_written,
            action=action or status,
            bet_ref_before=bet_ref_before,
            bet_ref_after=bet_ref_after,
            bet_ref_poll_attempts=bet_ref_poll_attempts,
            bet_ref_poll_duration_ms=bet_ref_poll_duration_ms,
            pre_write_attempt_id=_pre_write_attempt_id(intent, context, excel_write_attempt or 0),
            pre_bet_ref_required=(
                _is_pre_ladder_intent(intent)
                and (_ladder_step_index(intent.ladder_step) == 0 or bool(pre_retry_allowed))
                if pre_bet_ref_required is None
                else bool(pre_bet_ref_required)
            ),
            pre_bet_ref_confirmed=bool(pre_bet_ref_confirmed),
            pre_bet_ref_found=bool(pre_bet_ref_confirmed),
            pre_bet_ref_missing=bool(pre_bet_ref_missing),
            pre_bet_ref_poll_attempts=(
                bet_ref_poll_attempts if pre_bet_ref_poll_attempts is None else pre_bet_ref_poll_attempts
            ),
            pre_bet_ref_poll_duration_ms=(
                bet_ref_poll_duration_ms
                if pre_bet_ref_poll_duration_ms is None
                else pre_bet_ref_poll_duration_ms
            ),
            pre_bet_ref_missing_retryable=pre_bet_ref_missing_retryable,
            pre_bet_ref_late_detected=pre_bet_ref_late_detected,
            pre_bet_ref_late_value=pre_bet_ref_late_value,
            pre_retry_count=(
                _pre_retry_count_for_intent(self, intent)
                if pre_retry_count is None
                else int(pre_retry_count)
            ),
            pre_retry_allowed=pre_retry_allowed,
            pre_retry_reason=pre_retry_reason,
            pre_retry_block_reason=pre_retry_block_reason,
            pre_unconfirmed_reason=pre_unconfirmed_reason,
            bet_ref_lookup_sources=str(self._batch_log_context.get("bet_ref_lookup_sources") or ""),
            bet_ref_lookup_source_used="",
            bet_ref_lookup_source=bet_ref_lookup_source,
            bet_ref_lookup_matched_runner=bet_ref_lookup_matched_runner,
            row_t_value="",
            selections_rows_scanned=0,
            selections_match_found=False,
            selections_match_reason="",
            selections_runner="",
            selections_side="",
            selections_stake=None,
            selections_bet_ref="",
            selections_req_odds=None,
            selections_market_name="",
            selections_debug_recent_rows="",
            selections_top_candidates="",
            bet_ref_row_t_dump="",
            bet_ref_diagnostic_hold_after_batch=False,
            selections_market_query="",
            selections_current_market_rows="",
            selections_current_runner_rows="",
            runner_qz_dump="",
            selections_sheet_headers="",
            selections_full_recent_rows="",
            workbook_sheet_names="",
            diagnostic_keep_triggers=False,
            active_ladder_bet_ref_stored=active_ladder_bet_ref_stored,
            active_ladder_created=active_ladder_created,
            pending_ladder_created=pending_ladder_created,
            matched_evidence_found=matched_evidence_found,
            selection_row_evidence_found=selection_row_evidence_found,
            no_stacking_blocked_retry=no_stacking_blocked_retry,
            replace_allowed=replace_allowed,
            replace_trigger=replace_trigger,
            bet_ref_suffix_n_handled=bet_ref_suffix_n_handled,
            bet_ref_status_value=bet_ref_status_value,
            replace_bet_ref_wait_attempted=replace_bet_ref_wait_attempted,
            replace_bet_ref_wait_ms=replace_bet_ref_wait_ms,
            replace_bet_ref_poll_ms=replace_bet_ref_poll_ms,
            replace_bet_ref_wait_result=replace_bet_ref_wait_result,
            bet_ref_before_wait=bet_ref_before_wait,
            bet_ref_after_wait=bet_ref_after_wait,
            active_ladder_bet_ref_updated=active_ladder_bet_ref_updated,
            replace_skipped_bet_ref_still_pending=replace_skipped_bet_ref_still_pending,
            matched_stake_cell_address=matched_stake_cell_address,
            matched_stake_cell_value=matched_stake_cell_value,
            batch_size=_int_from_context(self._batch_log_context.get("batch_size")),
            batch_write_start_timestamp=str(
                self._batch_log_context.get("batch_write_start_timestamp") or ""
            ),
            batch_write_end_timestamp=str(
                self._batch_log_context.get("batch_write_end_timestamp") or ""
            ),
            batch_write_duration_ms=_int_from_context(
                self._batch_log_context.get("batch_write_duration_ms")
            ),
            order_index_in_batch=_int_from_context(
                self._batch_log_context.get("order_index_in_batch")
            ),
            runner_row=excel_row,
            runner_order_in_sheet=_runner_order_in_sheet(excel_row),
            update_allowed=update_allowed,
            update_skipped_reason=update_skipped_reason,
            matched_stake=intent.matched_stake,
            pre_ladder_initial_order_failed=pre_ladder_initial_order_failed,
            pre_ladder_disabled_after_initial_failure=pre_ladder_disabled_after_initial_failure,
            no_replace_steps_for_failed_initial=no_replace_steps_for_failed_initial,
            requested_price=intent.price,
            requested_stake=intent.stake,
            ladder_step_index=ladder_step_index,
            ladder_step_count=ladder_step_count,
            matched_after_step=bool(matched_after_step_stake is not None and matched_after_step_stake > 0),
            matched_after_step_avg_odds=matched_after_step_avg_odds,
            matched_after_step_stake=matched_after_step_stake,
            avg_matched_odds_cell_address=avg_matched_odds_cell_address,
            avg_matched_odds_cell_value=avg_matched_odds_cell_value,
            profit_loss_cell_address=profit_loss_cell_address,
            profit_loss_cell_value=profit_loss_cell_value,
            trigger_clear_attempted=trigger_clear_attempted,
            trigger_cleared=trigger_cleared,
            trigger_clear_reason=trigger_clear_reason,
            trigger_cell_value_before_clear=trigger_cell_value_before_clear,
            trigger_clear_delay_ms=trigger_clear_delay_ms,
            command_cells_clear_attempted=command_cells_clear_attempted_value,
            command_cells_cleared=command_cells_cleared_value,
            command_cells_clear_reason=command_cells_clear_reason_value,
            command_cells_clear_addresses=command_cells_clear_addresses,
            command_cells_clear_delay_ms=command_cells_clear_delay_ms_value,
            command_cells_clear_scheduled=command_cells_clear_scheduled,
            command_cells_clear_due_time=command_cells_clear_due_time,
            command_cells_clear_non_blocking=command_cells_clear_non_blocking,
            command_cells_clear_executed=command_cells_clear_executed_value,
            command_cells_clear_lag_ms=command_cells_clear_lag_ms,
            post_write_odds_cell_address=(
                post_write_verification.odds_cell_address if post_write_verification else ""
            ),
            post_write_odds_value=post_write_verification.odds_value if post_write_verification else None,
            post_write_stake_cell_address=(
                post_write_verification.stake_cell_address if post_write_verification else ""
            ),
            post_write_stake_value=(
                post_write_verification.stake_value if post_write_verification else None
            ),
            post_write_trigger_cell_address=(
                post_write_verification.trigger_cell_address if post_write_verification else ""
            ),
            post_write_trigger_value=(
                post_write_verification.trigger_value if post_write_verification else None
            ),
            post_write_verified=post_write_verification.verified if post_write_verification else None,
            hold_trigger_for_visual_test=hold_trigger_for_visual_test,
            price_raw_before_tick=price_raw_before_tick,
            price_tick_rounded=price_tick_rounded,
            price_tick_rounding_side=price_tick_rounding_side,
            price_is_valid_betfair_tick=price_is_valid_betfair_tick,
        )

    def _refresh_safety_flags(self) -> None:
        """Re-read every arming flag for each individual order attempt."""
        self.order_provider = os.getenv("DOGBOT_ORDER_PROVIDER", "").strip().lower()
        self.enabled = _env_bool("DOGBOT_GRUSS_ENABLE_REAL_ORDERS", False)
        self.preview = _env_bool("DOGBOT_GRUSS_REAL_PREVIEW", False)
        self.layout_confirmed = _env_bool("DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED", False)
        self.write_no_trigger = _env_bool("DOGBOT_GRUSS_WRITE_NO_TRIGGER", False)
        self.real_test_mode = _env_bool("DOGBOT_GRUSS_REAL_TEST_MODE", False)
        self.real_max_orders = _real_max_orders(self.real_test_mode)
        self.real_max_orders_by_phase = _real_max_orders_by_phase(self.real_test_mode, self.real_max_orders)
        self.real_max_stake = _real_max_stake(self.real_test_mode)
        self.trigger_clear_delay_ms = _trigger_clear_delay_ms()
        self.command_cells_clear_enabled = _env_bool("DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_AFTER_WRITE", True)
        self.command_cells_clear_delay_ms = _command_cells_clear_delay_ms()
        self.command_cells_clear_non_blocking = _env_bool(
            "DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_NON_BLOCKING",
            True,
        )
        self.command_cells_clear_columns = _command_cells_clear_columns_from_env()
        self._drain_due_command_cell_clears()
        self.hold_trigger_for_visual_test = _env_bool(
            "DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST",
            False,
        )

    def _is_true_real_mode(self) -> bool:
        return (
            self.order_provider == ORDER_PROVIDER_GRUSS_EXCEL_REAL
            and self.enabled
            and not self.preview_only_guard
            and not self.write_no_trigger_guard
            and not self.write_no_trigger
            and not self.preview
        )

    def _max_orders_for_intent(self, intent: OrderIntent) -> int | None:
        return self.real_max_orders_by_phase.get(_execution_phase(intent), self.real_max_orders)

    def _release_active_ladder_if_post_context(
        self,
        intent: OrderIntent,
        context: GrussRealOrderContext,
    ) -> None:
        self._sync_legacy_active_ladder_state()
        if _is_pre_ladder_intent(intent) or not self.active_pre_ladders:
            return
        if not _post_countdown_allowed(context.countdown_seconds):
            return
        self._pending_active_ladder_release_id = "|".join(sorted(self.active_pre_ladders))
        self._pending_active_ladder_release_reason = "post_milestone_reached"
        self.active_pre_ladders.clear()
        self.active_pre_ladder_id = None
        self.active_pre_ladder_course = None

    def _sync_legacy_active_ladder_state(self) -> None:
        if self.active_pre_ladder_id and self.active_pre_ladder_id not in self.active_pre_ladders:
            self.active_pre_ladders[self.active_pre_ladder_id] = _ActivePreLadderState(
                course_key=self.active_pre_ladder_course or "",
                market_type="",
                market_id="",
                selection_id="",
                runner_name="",
                trap=None,
                side="",
                row=0,
                bet_ref="",
            )
        self._refresh_legacy_active_ladder_snapshot()

    def _refresh_legacy_active_ladder_snapshot(self) -> None:
        if not self.active_pre_ladders:
            self.active_pre_ladder_id = None
            self.active_pre_ladder_course = None
            return
        ladder_id = next(iter(self.active_pre_ladders))
        self.active_pre_ladder_id = ladder_id
        self.active_pre_ladder_course = self.active_pre_ladders[ladder_id].course_key

    def _append_attempt(
        self,
        intent: OrderIntent,
        context: GrussRealOrderContext,
        status: str,
        reason: str,
        *,
        excel_sheet: str,
        excel_row: int | None,
        excel_cells_written: tuple[str, ...],
        trigger_written: bool,
        intended_trigger: str,
        trigger_cell_address: str,
        trigger_cell_current_value: Any,
        trigger_cell_expected_empty: bool | None,
        trigger_mapping_name: str,
        trigger_value_written: str,
        action: str,
        trigger_clear_attempted: bool,
        trigger_cleared: bool,
        trigger_clear_reason: str,
        trigger_cell_value_before_clear: Any,
        trigger_clear_delay_ms: int,
        command_cells_clear_attempted: bool,
        command_cells_cleared: bool,
        command_cells_clear_reason: str,
        command_cells_clear_addresses: str,
        command_cells_clear_delay_ms: int,
        command_cells_clear_scheduled: bool,
        command_cells_clear_due_time: str,
        command_cells_clear_non_blocking: bool,
        command_cells_clear_executed: bool,
        command_cells_clear_lag_ms: int | None,
        startup_command_cells_cleanup_attempted: bool,
        startup_command_cells_cleanup_done: bool,
        stale_command_cells_cleanup_attempted: bool,
        stale_command_cells_cleanup_addresses: str,
        stale_command_cells_cleanup_reason: str,
        post_write_verification: _PostWriteVerification | None,
        hold_trigger_for_visual_test: bool,
        stake_capped: bool,
        stake_cap_value: float | None,
        bet_ref_before: str,
        bet_ref_after: str,
        bet_ref_poll_attempts: int,
        bet_ref_poll_duration_ms: int,
        pre_bet_ref_required: bool,
        pre_bet_ref_confirmed: bool,
        pre_bet_ref_missing: bool,
        pre_bet_ref_poll_attempts: int,
        pre_bet_ref_poll_duration_ms: int,
        pre_write_attempt_id: str,
        pre_bet_ref_missing_retryable: bool,
        pre_bet_ref_late_detected: bool,
        pre_bet_ref_late_value: str,
        pre_retry_count: int,
        pre_retry_allowed: bool,
        pre_retry_reason: str,
        pre_retry_block_reason: str,
        pre_unconfirmed_reason: str,
        bet_ref_lookup_sources: str,
        bet_ref_lookup_source_used: str,
        bet_ref_lookup_source: str,
        bet_ref_lookup_matched_runner: str,
        row_t_value: str,
        selections_rows_scanned: int,
        selections_match_found: bool,
        selections_match_reason: str,
        selections_runner: str,
        selections_side: str,
        selections_stake: float | None,
        selections_bet_ref: str,
        selections_req_odds: float | None,
        selections_market_name: str,
        selections_debug_recent_rows: str,
        selections_top_candidates: str,
        bet_ref_row_t_dump: str,
        bet_ref_diagnostic_hold_after_batch: bool,
        selections_market_query: str,
        selections_current_market_rows: str,
        selections_current_runner_rows: str,
        runner_qz_dump: str,
        selections_sheet_headers: str,
        selections_full_recent_rows: str,
        workbook_sheet_names: str,
        diagnostic_keep_triggers: bool,
        active_ladder_bet_ref_stored: bool,
        active_ladder_created: bool,
        pending_ladder_created: bool,
        matched_evidence_found: bool,
        selection_row_evidence_found: bool,
        no_stacking_blocked_retry: bool,
        replace_allowed: bool,
        replace_trigger: str,
        bet_ref_suffix_n_handled: bool,
        bet_ref_status_value: str,
        replace_bet_ref_wait_attempted: bool,
        replace_bet_ref_wait_ms: int,
        replace_bet_ref_poll_ms: int,
        replace_bet_ref_wait_result: str,
        bet_ref_before_wait: str,
        bet_ref_after_wait: str,
        active_ladder_bet_ref_updated: bool,
        replace_skipped_bet_ref_still_pending: bool,
        matched_stake_cell_address: str,
        matched_stake_cell_value: Any,
        update_allowed: bool,
        update_skipped_reason: str,
        active_pre_ladder_id_snapshot: str,
        active_ladder_completed: bool,
        active_ladder_release_reason: str,
        countdown_at_write: int | None,
        current_market_price_at_write: float | None,
        stale_distance: float | None,
        stale_price_limit: float | None,
        stale_check_ignored_for_pre: bool,
        pre_batch_late_write_allowed: bool,
        pre_batch_late_write_seconds_after_start: int | None,
        pre_cancel_attempted: bool,
        pre_cancel_written: bool,
        pre_cancel_skip_reason: str,
        pre_cancel_only_if_post_pending: bool,
        post_pending_for_runner: bool,
        post_after_pre_cancel_attempted: bool,
        bet_ref_at_cancel: str,
        matched_stake_at_cancel: float | None,
        countdown_seconds_at_cancel: int | None,
        post_provider_called: bool | None,
        post_batch_id: str,
        post_batch_market_id: str,
        post_batch_market_name: str,
        post_batch_candidate_count: int | None,
        post_batch_written_count: int | None,
        post_batch_write_duration_ms: int | None,
        post_batch_confirmation_started: bool,
        post_batch_confirmation_duration_ms: int | None,
        post_batch_runner_index: int | None,
        post_batch_total_runners: int | None,
        post_send_seconds_before_off: int | None,
        post_allow_after_scheduled_off_seconds: int | None,
        post_trigger_window_hit: bool | None,
        post_write_attempted: bool | None,
        post_write_status: str,
        post_write_reason: str,
        post_bet_ref_required: bool,
        post_bet_ref_wait_attempted: bool,
        post_bet_ref_wait_ms: int,
        post_bet_ref_poll_ms: int,
        post_existing_bet_ref_before: str,
        post_existing_pre_bet_ref: str,
        post_existing_matched_before: float | None,
        post_existing_pre_matched_stake: float | None,
        post_existing_avg_odds_before: float | None,
        post_existing_pre_avg_odds: float | None,
        post_independent_mode_enabled: bool,
        post_row_prepared_for_new_order: bool,
        post_pre_bet_ref_cleared_for_write: bool,
        post_pre_bet_ref_preserved_in_state: bool,
        post_new_bet_ref_expected: bool,
        post_new_bet_ref_found: bool,
        post_new_bet_ref: str,
        post_added_stake_confirmed: bool,
        post_added_stake_amount: float | None,
        post_total_matched_before: float | None,
        post_total_matched_after: float | None,
        post_total_matched_delta: float | None,
        post_expected_market_id: str,
        post_expected_market_type: str,
        post_expected_runner: str,
        post_expected_selection_id: str,
        post_expected_side: str,
        post_expected_stake: float | None,
        post_expected_price: float | None,
        post_write_timestamp: str,
        post_order_write_timestamp: str,
        post_bet_ref_after: str,
        post_bet_ref_changed: bool,
        post_bet_ref_confirmed_new: bool,
        post_bet_ref_poll_attempts: int,
        post_bet_ref_poll_duration_ms: int,
        post_order_confirmed: bool,
        post_order_confirmation_source: str,
        post_confirmation_source: str,
        post_selections_lookup_attempted: bool,
        post_selections_match_found: bool,
        post_selections_match_reason: str,
        post_selections_reject_reason: str,
        post_clear_after_bet_ref: bool,
        post_cells_clear_delay_ms: int,
        post_cells_cleared_after_confirmation: bool,
        post_cells_cleared_after_unconfirmed: bool,
        post_clear_reason: str,
        post_write_unconfirmed_reason: str,
        post_unconfirmed_reason: str,
        post_reject_reason: str,
        countdown_seconds_at_post_write: int | None,
        market_status_at_post_write: Any,
        no_stacking_check_passed: bool,
        mapping_found: bool,
        mapping_reason: str,
        command_cells: str,
        pre_ladder_initial_order_failed: bool,
        pre_ladder_disabled_after_initial_failure: bool,
        no_replace_steps_for_failed_initial: bool,
        direct_lim_order_planned: bool,
        direct_lim_order_written: bool,
        requested_price: float | None,
        requested_stake: float | None,
        price_raw_before_tick: float | None,
        price_tick_rounded: float | None,
        price_tick_rounding_side: str,
        price_is_valid_betfair_tick: bool | None,
        ladder_step_index: int | None,
        ladder_step_count: int | None,
        matched_after_step: bool,
        matched_after_step_avg_odds: float | None,
        matched_after_step_stake: float | None,
        avg_matched_odds_cell_address: str,
        avg_matched_odds_cell_value: Any,
        profit_loss_cell_address: str,
        profit_loss_cell_value: Any,
        excel_write_attempt: int,
        excel_write_retry_count: int,
        excel_write_retry_backoff_ms: str,
        excel_write_final_status: str,
        excel_unavailable_recovered: bool,
        excel_operation_name: str,
        excel_com_attempt: int,
        excel_com_retry_count: int,
        excel_com_retry_backoff_ms: str,
        excel_com_retryable_error: bool,
        mapping_attempt_count: int,
        cleanup_retry_count: int,
        cleanup_final_status: str,
    ) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_attempt_log_header()
        write_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
        cells_written = ";".join(excel_cells_written)
        mode = "WRITE_NO_TRIGGER" if self.write_no_trigger_guard else ("PREVIEW" if self.preview else "REAL")
        write_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        stake_before_min_floor = _finite_float_or_none(
            intent.stake_original if intent.stake_original is not None else intent.stake
        )
        stake_final = _finite_float_or_none(intent.stake)
        stake_min_floor_applied = bool(
            stake_before_min_floor is not None
            and stake_final is not None
            and 0.0 < stake_before_min_floor < 1.0
            and stake_final >= 1.0
        )
        stake_after_min_floor = 1.0 if stake_min_floor_applied else stake_before_min_floor
        row = {
            "timestamp": write_timestamp,
            "dry_run_or_real": mode,
            "enabled": str(bool(self.enabled)).lower(),
            "provider": self.order_provider,
            "course": context.course or intent.course_id or "",
            "market_id": intent.market_id,
            "market_type": intent.market_type,
            "runner": intent.runner_name,
            "trap": intent.trap,
            "selection_id": intent.selection_id if intent.selection_id is not None else intent.trap,
            "side": intent.side,
            "order_type": intent.order_type,
            "execution_phase": _execution_phase(intent),
            "processed_key": _processed_key(intent, context),
            "post_processed_key": self._batch_log_context.get("post_processed_key", ""),
            "post_processed_key_scope": self._batch_log_context.get("post_processed_key_scope", ""),
            "parent_id": self._batch_log_context.get("parent_id", ""),
            "course_id": self._batch_log_context.get("course_id", ""),
            "win_market_id": self._batch_log_context.get("win_market_id", ""),
            "place_market_id": self._batch_log_context.get("place_market_id", ""),
            "processed_key_seen": str(bool(self._batch_log_context.get("processed_key_seen", False))),
            "processed_key_seen_matching_existing_key": self._batch_log_context.get(
                "processed_key_seen_matching_existing_key",
                "",
            ),
            "pre_post_independent": str(bool(self._batch_log_context.get("pre_post_independent", False))),
            "pre_existing_order_allowed": str(bool(self._batch_log_context.get("pre_existing_order_allowed", False))),
            "pre_cancel_required_before_post": str(
                bool(self._batch_log_context.get("pre_cancel_required_before_post", False))
            ),
            "stake_limit_scope": self._batch_log_context.get("stake_limit_scope", ""),
            "triggered_systems": intent.triggered_systems or intent.strategy_id,
            "triggered_prices": intent.triggered_prices or "",
            "pre_ladder": str(bool(getattr(intent, "pre_ladder", False))),
            "ladder_id": intent.ladder_id or "",
            "ladder_step": intent.ladder_step or "",
            "active_pre_ladder_id": active_pre_ladder_id_snapshot,
            "continuing_active_pre_ladder": str(
                _continuing_active_pre_ladder(intent, active_pre_ladder_id_snapshot)
            ),
            "active_pre_ladder_count": len(self.active_pre_ladders),
            "max_active_pre_ladders": _pre_ladder_real_max_ladders(),
            "configured_ladder_steps": ",".join(str(step) for step in _pre_ladder_steps_from_env()),
            "ladder_plan_frozen": str(bool(getattr(intent, "ladder_plan_frozen", False))),
            "ladder_plan_created_step": getattr(intent, "ladder_plan_created_step", "") or "",
            "ladder_prices_frozen": getattr(intent, "ladder_prices_frozen", "") or "",
            "current_ladder_price_from_frozen_plan": str(
                bool(getattr(intent, "current_ladder_price_from_frozen_plan", False))
            ),
            "computed_limit_price_raw": _blank_if_none(getattr(intent, "computed_limit_price_raw", None)),
            "computed_limit_price_effective": _blank_if_none(_pre_value_target_effective_from_intent(intent)),
            "min_price_floor_applied": str(bool(getattr(intent, "min_price_floor_applied", False))),
            "pre_value_target_price": _blank_if_none(_pre_value_target_effective_from_intent(intent)),
            "ladder_planned_price": _blank_if_none(getattr(intent, "ladder_planned_price", None)),
            "sent_price_before_value_clamp": _blank_if_none(
                getattr(intent, "sent_price_before_value_clamp", None)
            ),
            "sent_price_after_value_clamp": _blank_if_none(price_tick_rounded),
            "value_clamp_applied": str(
                bool(getattr(intent, "value_clamp_applied", False)) or _value_clamp_was_needed(intent)
            ),
            "value_limit_breached": str(
                bool(getattr(intent, "value_limit_breached", False)) or _value_clamp_was_needed(intent)
            ),
            "value_limit_skip_reason": getattr(intent, "value_limit_skip_reason", "") or "",
            "tick_rounding_direction": getattr(intent, "tick_rounding_direction", "") or price_tick_rounding_side,
            "best_same_side_offer_at_creation": (
                ""
                if getattr(intent, "best_same_side_offer_at_creation", None) is None
                else getattr(intent, "best_same_side_offer_at_creation")
            ),
            "best_back_displayed": (
                "" if getattr(intent, "best_back_displayed", None) is None else getattr(intent, "best_back_displayed")
            ),
            "best_lay_displayed": (
                "" if getattr(intent, "best_lay_displayed", None) is None else getattr(intent, "best_lay_displayed")
            ),
            "start_price_source": getattr(intent, "start_price_source", "") or "",
            "final_lim_price": "" if intent.price is None else intent.price,
            "ladder_direction": getattr(intent, "ladder_direction", "") or "",
            "ladder_disabled_lim_not_in_ladder_direction": str(
                bool(getattr(intent, "ladder_disabled_lim_not_in_ladder_direction", False))
            ),
            "direct_lim_order_planned": str(bool(direct_lim_order_planned)),
            "direct_lim_order_written": str(bool(direct_lim_order_written)),
            "direct_lim_candidates_count": self._batch_log_context.get("direct_lim_candidates_count", ""),
            "direct_lim_candidate_index": self._batch_log_context.get("direct_lim_candidate_index", ""),
            "direct_lim_provider_called": self._batch_log_context.get("direct_lim_provider_called", ""),
            "direct_lim_provider_skip_reason": self._batch_log_context.get("direct_lim_provider_skip_reason", ""),
            "direct_lim_batch_processed_count": self._batch_log_context.get("direct_lim_batch_processed_count", ""),
            "direct_lim_written_count": self._batch_log_context.get("direct_lim_written_count", ""),
            "direct_lim_rejected_count": self._batch_log_context.get("direct_lim_rejected_count", ""),
            "no_replace_steps_for_direct_lim": str(
                bool(getattr(intent, "no_replace_steps_for_direct_lim", False))
            ),
            "current_milestone": _current_milestone(context),
            "computed_step_index": _ladder_step_index(intent.ladder_step),
            "expected_ladder_step": _expected_ladder_step_for_context(context),
            "milestone_seen": (
                context.milestone_seen
                if context.milestone_seen is not None
                else context.countdown_seconds
            ),
            "next_ladder_step_due": intent.ladder_step or "",
            "skipped_step_reason": update_skipped_reason,
            "active_ladder_completed": str(bool(active_ladder_completed)),
            "active_ladder_release_reason": active_ladder_release_reason,
            "countdown_authorization_reason": _countdown_authorization_reason(
                intent,
                context,
                write_no_trigger_guard=self.write_no_trigger_guard,
            ),
            "signal_timestamp": intent.timestamp,
            "write_timestamp": write_timestamp,
            "write_delay_since_signal_seconds": _seconds_between_iso_timestamps(
                intent.timestamp,
                write_timestamp,
            ),
            "countdown_at_signal": intent.signal_countdown_seconds
            if intent.signal_countdown_seconds is not None
            else context.milestone_seen,
            "countdown_at_write": countdown_at_write
            if countdown_at_write is not None
            else context.countdown_seconds,
            "pre_batch_milestone_authorized": str(bool(context.pre_batch_milestone_authorized)),
            "pre_batch_milestone_seconds": _blank_if_none(context.pre_batch_milestone_seconds),
            "pre_batch_started_countdown_seconds": _blank_if_none(
                context.pre_batch_started_countdown_seconds
            ),
            "pre_batch_write_grace_seconds": _blank_if_none(
                context.pre_batch_write_grace_seconds
            ),
            "pre_batch_candidate_index": self._batch_log_context.get("pre_batch_candidate_index", ""),
            "pre_batch_candidates_count": self._batch_log_context.get("pre_batch_candidates_count", ""),
            "pre_batch_late_write_allowed": str(bool(pre_batch_late_write_allowed)),
            "pre_batch_late_write_seconds_after_start": _blank_if_none(
                pre_batch_late_write_seconds_after_start
            ),
            "no_stacking_check_passed": str(bool(no_stacking_check_passed)),
            "active_ladder_count": len(self.active_pre_ladders),
            "max_ladders_limit": _pre_ladder_real_max_ladders(),
            "market_reference_price_at_signal": "" if intent.market_reference_price_at_signal is None else intent.market_reference_price_at_signal,
            "current_market_price_at_write": "" if current_market_price_at_write is None else current_market_price_at_write,
            "stale_distance": "" if stale_distance is None else stale_distance,
            "stale_price_limit": "" if stale_price_limit is None else stale_price_limit,
            "stale_check_ignored_for_pre": str(bool(stale_check_ignored_for_pre)),
            "conflict_detected": str(bool(intent.conflict_detected)),
            "conflict_type": getattr(intent, "conflict_type", "") or "",
            "back_price": "" if intent.back_price is None else intent.back_price,
            "lay_price": "" if intent.lay_price is None else intent.lay_price,
            "market_reference_price": "" if intent.market_reference_price is None else intent.market_reference_price,
            "back_distance": "" if intent.back_distance is None else intent.back_distance,
            "lay_distance": "" if intent.lay_distance is None else intent.lay_distance,
            "selected_side": intent.selected_side or "",
            "rejected_side": intent.rejected_side or "",
            "conflict_group_key": getattr(intent, "conflict_group_key", "") or "",
            "conflict_candidates_count": getattr(intent, "conflict_candidates_count", "") or "",
            "winning_side": getattr(intent, "winning_side", "") or "",
            "losing_side": getattr(intent, "losing_side", "") or "",
            "winning_strategy_id": getattr(intent, "winning_strategy_id", "") or "",
            "losing_strategy_id": getattr(intent, "losing_strategy_id", "") or "",
            "winning_edge": _blank_if_none(getattr(intent, "winning_edge", None)),
            "losing_edge": _blank_if_none(getattr(intent, "losing_edge", None)),
            "winning_score": _blank_if_none(getattr(intent, "winning_score", None)),
            "losing_score": _blank_if_none(getattr(intent, "losing_score", None)),
            "winning_lim_price": _blank_if_none(getattr(intent, "winning_lim_price", None)),
            "losing_lim_price": _blank_if_none(getattr(intent, "losing_lim_price", None)),
            "back_systems": getattr(intent, "back_systems", "") or "",
            "lay_systems": getattr(intent, "lay_systems", "") or "",
            "conflict_resolution_reason": intent.conflict_resolution_reason or "",
            "pre_back_lay_conflict": str(bool(getattr(intent, "pre_back_lay_conflict", False))),
            "pre_conflict_resolution": getattr(intent, "pre_conflict_resolution", "") or "",
            "pre_conflict_chosen_side": getattr(intent, "pre_conflict_chosen_side", "") or "",
            "pre_conflict_rejected_side": getattr(intent, "pre_conflict_rejected_side", "") or "",
            "pre_conflict_reason": getattr(intent, "pre_conflict_reason", "") or "",
            "pre_conflict_group_key": getattr(intent, "pre_conflict_group_key", "") or "",
            "pre_conflict_course_id": getattr(intent, "pre_conflict_course_id", "") or "",
            "pre_conflict_market_id": getattr(intent, "pre_conflict_market_id", "") or "",
            "pre_conflict_market_type": getattr(intent, "pre_conflict_market_type", "") or "",
            "pre_conflict_selection_id": getattr(intent, "pre_conflict_selection_id", "") or "",
            "pre_conflict_runner_name": getattr(intent, "pre_conflict_runner_name", "") or "",
            "pre_back_target_price": _blank_if_none(getattr(intent, "pre_back_target_price", None)),
            "pre_lay_target_price": _blank_if_none(getattr(intent, "pre_lay_target_price", None)),
            "pre_current_best_lay": _blank_if_none(getattr(intent, "pre_current_best_lay", None)),
            "pre_current_best_back": _blank_if_none(getattr(intent, "pre_current_best_back", None)),
            "pre_back_distance_ticks": _blank_if_none(getattr(intent, "pre_back_distance_ticks", None)),
            "pre_lay_distance_ticks": _blank_if_none(getattr(intent, "pre_lay_distance_ticks", None)),
            "intended_trigger": intended_trigger,
            "trigger": intended_trigger,
            "stake": intent.stake,
            "stake_original": intent.stake_original if intent.stake_original is not None else intent.stake,
            "stake_used": intent.stake,
            "stake_forced": str(bool(intent.stake_forced)),
            "stake_min_floor_applied": str(bool(stake_min_floor_applied)),
            "stake_before_min_floor": _blank_if_none(stake_before_min_floor),
            "stake_after_min_floor": _blank_if_none(stake_after_min_floor),
            "stake_final": _blank_if_none(stake_final),
            "stake_capped": str(bool(stake_capped)),
            "stake_cap_value": "" if stake_cap_value is None else stake_cap_value,
            "staking_formula": getattr(intent, "staking_formula", "") or "",
            "staking_alpha": _blank_if_none(getattr(intent, "staking_alpha", None)),
            "staking_back_alpha": _blank_if_none(getattr(intent, "staking_back_alpha", None)),
            "staking_lay_alpha": _blank_if_none(getattr(intent, "staking_lay_alpha", None)),
            "stake_raw_before_caps": _blank_if_none(getattr(intent, "stake_raw_before_caps", None)),
            "stake_after_caps": _blank_if_none(getattr(intent, "stake_after_caps", None)),
            "lay_liability_after_sizing": _blank_if_none(getattr(intent, "lay_liability_after_sizing", None)),
            "lay_liability_cap": _blank_if_none(getattr(intent, "lay_liability_cap", None)),
            "lay_liability_cap_hit": str(bool(getattr(intent, "lay_liability_cap_hit", False))),
            "force_test_bsp_place": str(bool(intent.force_test_bsp_place)),
            "force_test_back_place_limit": str(bool(intent.force_test_back_place_limit)),
            "selected_reason": intent.selected_reason or "",
            "selected_runner": intent.selected_runner or "",
            "selected_trap": intent.selected_trap,
            "selected_place_odds": intent.selected_place_odds,
            "selected_place_back_odds": intent.selected_place_back_odds,
            "selected_place_lay_odds": intent.selected_place_lay_odds,
            "price_used": intent.price_used if intent.price_used is not None else intent.price,
            "price": intent.price,
            "price_raw_before_tick": _blank_if_none(price_raw_before_tick),
            "price_tick_rounded": _blank_if_none(price_tick_rounded),
            "price_tick_rounding_side": price_tick_rounding_side,
            "price_is_valid_betfair_tick": (
                "" if price_is_valid_betfair_tick is None else str(bool(price_is_valid_betfair_tick)).lower()
            ),
            "strategy_id": intent.strategy_id,
            "status": status,
            "reason": reason,
            "mapping_found": str(bool(mapping_found)),
            "mapping_reason": mapping_reason,
            "command_cells": command_cells,
            "total_runners_in_gruss_sheet": self._batch_log_context.get("total_runners_in_gruss_sheet", ""),
            "raw_gruss_runner_rows": self._batch_log_context.get("raw_gruss_runner_rows", ""),
            "raw_selection_ids_seen": self._batch_log_context.get("raw_selection_ids_seen", ""),
            "raw_runner_names_seen": self._batch_log_context.get("raw_runner_names_seen", ""),
            "mapped_runners_count": self._batch_log_context.get("mapped_runners_count", ""),
            "unmapped_runners_count": self._batch_log_context.get("unmapped_runners_count", ""),
            "mapped_selection_ids": self._batch_log_context.get("mapped_selection_ids", ""),
            "unmapped_selection_ids": self._batch_log_context.get("unmapped_selection_ids", ""),
            "ignored_runner_rows": self._batch_log_context.get("ignored_runner_rows", ""),
            "ignored_runner_reason": self._batch_log_context.get("ignored_runner_reason", ""),
            "mapped_excel_rows": self._batch_log_context.get("mapped_excel_rows", ""),
            "excel_sheet": excel_sheet,
            "excel_row": excel_row,
            "excel_cells_written": cells_written,
            "cells_written": cells_written,
            "excel_write_attempt": _blank_if_none(excel_write_attempt),
            "excel_write_retry_count": _blank_if_none(excel_write_retry_count),
            "excel_write_retry_backoff_ms": excel_write_retry_backoff_ms,
            "excel_write_final_status": excel_write_final_status,
            "excel_unavailable_recovered": str(bool(excel_unavailable_recovered)),
            "excel_operation_name": excel_operation_name,
            "excel_com_attempt": _blank_if_none(excel_com_attempt),
            "excel_com_retry_count": _blank_if_none(excel_com_retry_count),
            "excel_com_retry_backoff_ms": excel_com_retry_backoff_ms,
            "excel_com_retryable_error": str(bool(excel_com_retryable_error)),
            "mapping_attempt_count": _blank_if_none(mapping_attempt_count),
            "cleanup_retry_count": _blank_if_none(cleanup_retry_count),
            "cleanup_final_status": cleanup_final_status,
            "trigger_cell_address": trigger_cell_address,
            "trigger_cell_current_value": trigger_cell_current_value,
            "trigger_cell_expected_empty": (
                "" if trigger_cell_expected_empty is None else str(trigger_cell_expected_empty)
            ),
            "trigger_mapping_name": trigger_mapping_name or intended_trigger,
            "trigger_written": str(bool(trigger_written)),
            "trigger_value_written": trigger_value_written,
            "action": action or status,
            "bet_ref_before": bet_ref_before,
            "bet_ref_after": bet_ref_after,
            "bet_ref_poll_attempts": bet_ref_poll_attempts,
            "bet_ref_poll_duration_ms": bet_ref_poll_duration_ms,
            "pre_write_attempt_id": pre_write_attempt_id,
            "pre_bet_ref_required": str(bool(pre_bet_ref_required)),
            "pre_bet_ref_confirmed": str(bool(pre_bet_ref_confirmed)),
            "pre_bet_ref_found": str(bool(pre_bet_ref_confirmed)),
            "pre_bet_ref_missing": str(bool(pre_bet_ref_missing)),
            "pre_bet_ref_poll_attempts": pre_bet_ref_poll_attempts,
            "pre_bet_ref_poll_duration_ms": pre_bet_ref_poll_duration_ms,
            "pre_bet_ref_missing_retryable": str(bool(pre_bet_ref_missing_retryable)),
            "pre_bet_ref_late_detected": str(bool(pre_bet_ref_late_detected)),
            "pre_bet_ref_late_value": pre_bet_ref_late_value,
            "pre_retry_count": _blank_if_none(pre_retry_count),
            "pre_retry_allowed": str(bool(pre_retry_allowed)),
            "pre_retry_reason": pre_retry_reason,
            "pre_retry_block_reason": pre_retry_block_reason,
            "pre_unconfirmed_reason": pre_unconfirmed_reason,
            "bet_ref_lookup_sources": self._batch_log_context.get("bet_ref_lookup_sources", ""),
            "bet_ref_lookup_source_used": "",
            "bet_ref_lookup_source": bet_ref_lookup_source,
            "bet_ref_lookup_matched_runner": bet_ref_lookup_matched_runner,
            "row_t_value": row_t_value,
            "selections_rows_scanned": _blank_if_none(selections_rows_scanned),
            "selections_match_found": str(bool(selections_match_found)),
            "selections_match_reason": selections_match_reason,
            "selections_runner": selections_runner,
            "selections_side": selections_side,
            "selections_stake": "" if selections_stake is None else selections_stake,
            "selections_bet_ref": selections_bet_ref,
            "selections_req_odds": "" if selections_req_odds is None else selections_req_odds,
            "selections_market_name": selections_market_name,
            "selections_debug_recent_rows": selections_debug_recent_rows,
            "selections_top_candidates": selections_top_candidates,
            "bet_ref_row_t_dump": bet_ref_row_t_dump,
            "bet_ref_diagnostic_hold_after_batch": str(bool(bet_ref_diagnostic_hold_after_batch)),
            "selections_market_query": selections_market_query,
            "selections_current_market_rows": selections_current_market_rows,
            "selections_current_runner_rows": selections_current_runner_rows,
            "runner_qz_dump": runner_qz_dump,
            "selections_sheet_headers": selections_sheet_headers,
            "selections_full_recent_rows": selections_full_recent_rows,
            "workbook_sheet_names": workbook_sheet_names,
            "diagnostic_keep_triggers": str(bool(diagnostic_keep_triggers)),
            "active_ladder_bet_ref_stored": str(bool(active_ladder_bet_ref_stored)),
            "active_ladder_created": str(bool(active_ladder_created)),
            "pending_ladder_created": str(bool(pending_ladder_created)),
            "matched_evidence_found": str(bool(matched_evidence_found)),
            "selection_row_evidence_found": str(bool(selection_row_evidence_found)),
            "no_stacking_blocked_retry": str(bool(no_stacking_blocked_retry)),
            "replace_allowed": str(bool(replace_allowed)),
            "replace_trigger": replace_trigger,
            "bet_ref_suffix_n_handled": str(bool(bet_ref_suffix_n_handled)),
            "bet_ref_status_value": bet_ref_status_value,
            "replace_bet_ref_wait_attempted": str(bool(replace_bet_ref_wait_attempted)),
            "replace_bet_ref_wait_ms": replace_bet_ref_wait_ms,
            "replace_bet_ref_poll_ms": replace_bet_ref_poll_ms,
            "replace_bet_ref_wait_result": replace_bet_ref_wait_result,
            "bet_ref_before_wait": bet_ref_before_wait,
            "bet_ref_after_wait": bet_ref_after_wait,
            "active_ladder_bet_ref_updated": str(bool(active_ladder_bet_ref_updated)),
            "replace_skipped_bet_ref_still_pending": str(bool(replace_skipped_bet_ref_still_pending)),
            "pre_ladder_initial_order_failed": str(bool(pre_ladder_initial_order_failed)),
            "pre_ladder_disabled_after_initial_failure": str(bool(pre_ladder_disabled_after_initial_failure)),
            "no_replace_steps_for_failed_initial": str(bool(no_replace_steps_for_failed_initial)),
            "requested_price": _blank_if_none(requested_price),
            "requested_stake": _blank_if_none(requested_stake),
            "ladder_step_index": _blank_if_none(ladder_step_index),
            "ladder_step_count": _blank_if_none(ladder_step_count),
            "matched_after_step": str(bool(matched_after_step)),
            "matched_after_step_avg_odds": _blank_if_none(matched_after_step_avg_odds),
            "matched_after_step_stake": _blank_if_none(matched_after_step_stake),
            "avg_matched_odds_cell_address": avg_matched_odds_cell_address,
            "avg_matched_odds_cell_value": "" if avg_matched_odds_cell_value is None else avg_matched_odds_cell_value,
            "matched_stake_cell_address": matched_stake_cell_address,
            "matched_stake_cell_value": "" if matched_stake_cell_value is None else matched_stake_cell_value,
            "profit_loss_cell_address": profit_loss_cell_address,
            "profit_loss_cell_value": "" if profit_loss_cell_value is None else profit_loss_cell_value,
            "batch_size": self._batch_log_context.get("batch_size", ""),
            "batch_write_start_timestamp": self._batch_log_context.get(
                "batch_write_start_timestamp",
                "",
            ),
            "batch_write_end_timestamp": self._batch_log_context.get(
                "batch_write_end_timestamp",
                "",
            ),
            "batch_write_duration_ms": self._batch_log_context.get(
                "batch_write_duration_ms",
                "",
            ),
            "order_index_in_batch": self._batch_log_context.get("order_index_in_batch", ""),
            "bet_ref_collection_phase_start": "",
            "bet_ref_collection_phase_end": "",
            "bet_ref_collection_duration_ms": "",
            "bet_ref_found_count": "",
            "bet_ref_missing_count": "",
            "runner_row": excel_row if excel_row is not None else "",
            "runner_order_in_sheet": _runner_order_in_sheet(excel_row),
            "update_allowed": str(bool(update_allowed)),
            "update_skipped_reason": update_skipped_reason,
            "matched_stake": "" if intent.matched_stake is None else intent.matched_stake,
            "pre_cancel_attempted": str(bool(pre_cancel_attempted)),
            "pre_cancel_written": str(bool(pre_cancel_written)),
            "pre_cancel_skip_reason": pre_cancel_skip_reason,
            "pre_cancel_only_if_post_pending": str(bool(pre_cancel_only_if_post_pending)),
            "post_pending_for_runner": str(bool(post_pending_for_runner)),
            "post_after_pre_cancel_attempted": str(bool(post_after_pre_cancel_attempted)),
            "bet_ref_at_cancel": bet_ref_at_cancel,
            "matched_stake_at_cancel": _blank_if_none(matched_stake_at_cancel),
            "countdown_seconds_at_cancel": _blank_if_none(countdown_seconds_at_cancel),
            "trigger_clear_attempted": str(bool(trigger_clear_attempted)),
            "trigger_cleared": str(bool(trigger_cleared)),
            "trigger_clear_reason": trigger_clear_reason,
            "trigger_cell_value_before_clear": trigger_cell_value_before_clear,
            "trigger_clear_delay_ms": trigger_clear_delay_ms,
            "command_cells_clear_attempted": str(bool(command_cells_clear_attempted)),
            "command_cells_cleared": str(bool(command_cells_cleared)),
            "command_cells_clear_reason": command_cells_clear_reason,
            "command_cells_clear_addresses": command_cells_clear_addresses,
            "command_cells_clear_delay_ms": command_cells_clear_delay_ms,
            "command_cells_clear_scheduled": str(bool(command_cells_clear_scheduled)),
            "command_cells_clear_due_time": command_cells_clear_due_time,
            "command_cells_clear_non_blocking": str(bool(command_cells_clear_non_blocking)),
            "command_cells_clear_executed": str(bool(command_cells_clear_executed)),
            "command_cells_clear_lag_ms": _blank_if_none(command_cells_clear_lag_ms),
            "startup_command_cells_cleanup_attempted": str(bool(startup_command_cells_cleanup_attempted)),
            "startup_command_cells_cleanup_done": str(bool(startup_command_cells_cleanup_done)),
            "stale_command_cells_cleanup_attempted": str(bool(stale_command_cells_cleanup_attempted)),
            "stale_command_cells_cleanup_addresses": stale_command_cells_cleanup_addresses,
            "stale_command_cells_cleanup_reason": stale_command_cells_cleanup_reason,
            "post_write_odds_cell_address": (
                post_write_verification.odds_cell_address if post_write_verification else ""
            ),
            "post_write_odds_value": post_write_verification.odds_value if post_write_verification else "",
            "post_write_stake_cell_address": (
                post_write_verification.stake_cell_address if post_write_verification else ""
            ),
            "post_write_stake_value": (
                post_write_verification.stake_value if post_write_verification else ""
            ),
            "post_write_trigger_cell_address": (
                post_write_verification.trigger_cell_address if post_write_verification else ""
            ),
            "post_write_trigger_value": (
                post_write_verification.trigger_value if post_write_verification else ""
            ),
            "post_write_verified": (
                "" if post_write_verification is None else str(post_write_verification.verified)
            ),
            "post_provider_called": "" if post_provider_called is None else str(bool(post_provider_called)),
            "post_batch_id": post_batch_id,
            "post_batch_market_id": post_batch_market_id,
            "post_batch_market_name": post_batch_market_name,
            "post_batch_candidate_count": _blank_if_none(post_batch_candidate_count),
            "post_batch_written_count": _blank_if_none(post_batch_written_count),
            "post_batch_write_duration_ms": _blank_if_none(post_batch_write_duration_ms),
            "post_batch_confirmation_started": str(bool(post_batch_confirmation_started)),
            "post_batch_confirmation_duration_ms": _blank_if_none(post_batch_confirmation_duration_ms),
            "post_batch_runner_index": _blank_if_none(post_batch_runner_index),
            "post_batch_total_runners": _blank_if_none(post_batch_total_runners),
            "post_send_seconds_before_off": _blank_if_none(post_send_seconds_before_off),
            "post_allow_after_scheduled_off_seconds": _blank_if_none(post_allow_after_scheduled_off_seconds),
            "post_trigger_window_hit": "" if post_trigger_window_hit is None else str(bool(post_trigger_window_hit)),
            "post_write_attempted": "" if post_write_attempted is None else str(bool(post_write_attempted)),
            "post_write_status": post_write_status,
            "post_write_reason": post_write_reason,
            "post_bet_ref_required": str(bool(post_bet_ref_required)),
            "post_bet_ref_wait_attempted": str(bool(post_bet_ref_wait_attempted)),
            "post_bet_ref_wait_ms": _blank_if_none(post_bet_ref_wait_ms),
            "post_bet_ref_poll_ms": _blank_if_none(post_bet_ref_poll_ms),
            "post_existing_bet_ref_before": post_existing_bet_ref_before,
            "post_existing_pre_bet_ref": post_existing_pre_bet_ref or post_existing_bet_ref_before,
            "post_existing_matched_before": _blank_if_none(post_existing_matched_before),
            "post_existing_pre_matched_stake": _blank_if_none(
                post_existing_pre_matched_stake
                if post_existing_pre_matched_stake is not None
                else post_existing_matched_before
            ),
            "post_existing_avg_odds_before": _blank_if_none(post_existing_avg_odds_before),
            "post_existing_pre_avg_odds": _blank_if_none(
                post_existing_pre_avg_odds
                if post_existing_pre_avg_odds is not None
                else post_existing_avg_odds_before
            ),
            "post_independent_mode_enabled": str(bool(post_independent_mode_enabled)),
            "post_row_prepared_for_new_order": str(bool(post_row_prepared_for_new_order)),
            "post_pre_bet_ref_cleared_for_write": str(bool(post_pre_bet_ref_cleared_for_write)),
            "post_pre_bet_ref_preserved_in_state": str(bool(post_pre_bet_ref_preserved_in_state)),
            "post_new_bet_ref_expected": str(bool(post_new_bet_ref_expected)),
            "post_new_bet_ref_found": str(bool(post_new_bet_ref_found)),
            "post_new_bet_ref": post_new_bet_ref,
            "post_added_stake_confirmed": str(bool(post_added_stake_confirmed)),
            "post_added_stake_amount": _blank_if_none(post_added_stake_amount),
            "post_total_matched_before": _blank_if_none(post_total_matched_before),
            "post_total_matched_after": _blank_if_none(post_total_matched_after),
            "post_total_matched_delta": _blank_if_none(post_total_matched_delta),
            "post_expected_market_id": post_expected_market_id,
            "post_expected_market_type": post_expected_market_type,
            "post_expected_runner": post_expected_runner,
            "post_expected_selection_id": post_expected_selection_id,
            "post_expected_side": post_expected_side,
            "post_expected_stake": _blank_if_none(post_expected_stake),
            "post_expected_price": _blank_if_none(post_expected_price),
            "post_write_timestamp": post_write_timestamp,
            "post_order_write_timestamp": post_order_write_timestamp or post_write_timestamp,
            "post_bet_ref_after": post_bet_ref_after,
            "post_bet_ref_changed": str(bool(post_bet_ref_changed)),
            "post_bet_ref_confirmed_new": str(bool(post_bet_ref_confirmed_new)),
            "post_bet_ref_poll_attempts": _blank_if_none(post_bet_ref_poll_attempts),
            "post_bet_ref_poll_duration_ms": _blank_if_none(post_bet_ref_poll_duration_ms),
            "post_order_confirmed": str(bool(post_order_confirmed)),
            "post_order_confirmation_source": post_order_confirmation_source,
            "post_confirmation_source": post_confirmation_source or post_order_confirmation_source,
            "post_selections_lookup_attempted": str(bool(post_selections_lookup_attempted)),
            "post_selections_match_found": str(bool(post_selections_match_found)),
            "post_selections_match_reason": post_selections_match_reason,
            "post_selections_reject_reason": post_selections_reject_reason,
            "post_clear_after_bet_ref": str(bool(post_clear_after_bet_ref)),
            "post_cells_clear_delay_ms": _blank_if_none(post_cells_clear_delay_ms),
            "post_cells_cleared_after_confirmation": str(bool(post_cells_cleared_after_confirmation)),
            "post_cells_cleared_after_unconfirmed": str(bool(post_cells_cleared_after_unconfirmed)),
            "post_clear_reason": post_clear_reason,
            "post_write_unconfirmed_reason": post_write_unconfirmed_reason,
            "post_unconfirmed_reason": post_unconfirmed_reason or post_write_unconfirmed_reason,
            "post_reject_reason": post_reject_reason,
            "countdown_seconds_at_post_write": _blank_if_none(countdown_seconds_at_post_write),
            "market_status_at_post_write": _blank_if_none(market_status_at_post_write),
            "hold_trigger_for_visual_test": str(bool(hold_trigger_for_visual_test)),
        }
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=GRUSS_REAL_ATTEMPTS_HEADER)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _ensure_attempt_log_header(self) -> None:
        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            return
        with self.output_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == GRUSS_REAL_ATTEMPTS_HEADER:
                return
            rows = list(reader)
        with self.output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=GRUSS_REAL_ATTEMPTS_HEADER)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in GRUSS_REAL_ATTEMPTS_HEADER})

    def _append_command_cells_cleanup_attempt(
        self,
        *,
        reason: str,
        attempted: bool,
        done: bool,
        addresses: Iterable[str],
        cleanup_reason: str,
        stale_scan_attempt_count: int = 0,
        stale_scan_retry_count: int = 0,
        stale_scan_recovered: bool = False,
        stale_triggers_confirmed: bool = False,
        stale_cleanup_retry_count: int = 0,
        stale_cleanup_recovered: bool = False,
        stale_cleanup_final_status: str = "",
        unsafe_stop_reason: str = "",
    ) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_attempt_log_header()
        write_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
        reason_text = str(reason or "").strip().lower()
        is_startup = reason_text == "startup"
        is_shutdown = reason_text == "shutdown"
        is_market_change = reason_text.startswith(("course_change", "market_change"))
        row = {field: "" for field in GRUSS_REAL_ATTEMPTS_HEADER}
        row.update(
            {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "dry_run_or_real": "REAL" if self._is_true_real_mode() else "DIAGNOSTIC",
                "enabled": str(bool(self.enabled)).lower(),
                "provider": self.order_provider,
                "status": "STALE_COMMAND_CELLS_CLEANUP",
                "reason": cleanup_reason,
                "startup_command_cells_cleanup_attempted": str(bool(attempted and is_startup)),
                "startup_command_cells_cleanup_done": str(bool(done and is_startup)),
                "stale_command_cells_cleanup_attempted": str(bool(attempted)),
                "stale_command_cells_cleanup_addresses": ";".join(str(address) for address in addresses),
                "stale_command_cells_cleanup_reason": cleanup_reason,
                "stale_scan_attempt_count": _blank_if_none(stale_scan_attempt_count),
                "stale_scan_retry_count": _blank_if_none(stale_scan_retry_count),
                "stale_scan_recovered": str(bool(stale_scan_recovered)),
                "stale_triggers_confirmed": str(bool(stale_triggers_confirmed)),
                "stale_cleanup_retry_count": _blank_if_none(stale_cleanup_retry_count),
                "stale_cleanup_recovered": str(bool(stale_cleanup_recovered)),
                "stale_cleanup_final_status": stale_cleanup_final_status,
                "unsafe_stop_reason": unsafe_stop_reason,
                "shutdown_command_cells_cleanup_done": str(bool(done and is_shutdown)),
                "market_change_command_cells_cleanup_done": str(bool(done and is_market_change)),
                "command_cells_clear_delay_ms": self.command_cells_clear_delay_ms,
            }
        )
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=GRUSS_REAL_ATTEMPTS_HEADER)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _update_attempt_log_rows(
        self,
        processed_keys: Iterable[str],
        *,
        common_fields: dict[str, Any] | None = None,
        per_key_fields: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        keys = {str(key) for key in processed_keys if key}
        if not keys or not self.output_path.exists():
            return
        with self.output_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        changed = False
        common_fields = common_fields or {}
        per_key_fields = per_key_fields or {}
        for row in rows:
            key = str(row.get("processed_key") or "")
            if key not in keys:
                continue
            updates = {**common_fields, **per_key_fields.get(key, {})}
            for field, value in updates.items():
                if field in GRUSS_REAL_ATTEMPTS_HEADER:
                    row[field] = "" if value is None else str(value)
                    changed = True
        if not changed:
            return
        with self.output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=GRUSS_REAL_ATTEMPTS_HEADER)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in GRUSS_REAL_ATTEMPTS_HEADER})


def _processed_key(intent: OrderIntent, context: GrussRealOrderContext) -> str:
    race_id = str(context.course or intent.course_id or intent.parent_id or "").strip()
    selection_id = intent.selection_id if intent.selection_id is not None else intent.trap
    parts = [
        race_id,
        intent.market_id,
        selection_id,
        intent.side,
        intent.market_type,
        _execution_phase(intent),
    ]
    if _is_pre_ladder_intent(intent):
        parts.extend(
            [
                f"milestone={context.milestone_seen if context.milestone_seen is not None else context.countdown_seconds}",
                f"ladder_step={intent.ladder_step or ''}",
                f"ladder_id={intent.ladder_id or intent.ladder_tracking_key or ''}",
            ]
        )
    return "|".join(str(part) for part in parts)


def _pre_write_attempt_id(intent: OrderIntent, context: GrussRealOrderContext, attempt: int | None) -> str:
    if not _is_pre_ladder_intent(intent):
        return ""
    return f"{_processed_key(intent, context)}|write_attempt={int(attempt or 0)}"


def _pre_retry_key_for_intent(intent: OrderIntent) -> str:
    selection_id = intent.selection_id if intent.selection_id is not None else intent.trap
    return "|".join(
        str(part or "")
        for part in (
            intent.market_id,
            selection_id,
            str(intent.side or "").upper(),
            intent.strategy_id,
        )
    )


def _pre_retry_key_for_result(result: GrussRealOrderResult) -> str:
    selection_id = result.selection_id if result.selection_id is not None else result.trap
    return "|".join(
        str(part or "")
        for part in (
            result.market_id,
            selection_id,
            str(result.side or "").upper(),
            result.strategy_id,
        )
    )


def _pre_retry_count_for_intent(provider: Any, intent: OrderIntent) -> int:
    counts = getattr(provider, "pre_bet_ref_missing_retry_counts", {})
    return int(counts.get(_pre_retry_key_for_intent(intent), 0))


def _active_pre_ladder_state_from_intent(
    intent: OrderIntent,
    course_key: str,
    runner_row: int,
    *,
    bet_ref: str = "",
    pending_confirmation: bool = False,
) -> _ActivePreLadderState:
    return _ActivePreLadderState(
        course_key=course_key,
        market_type=str(intent.market_type or "").upper(),
        market_id=str(intent.market_id or ""),
        selection_id=str(intent.selection_id if intent.selection_id is not None else intent.trap),
        runner_name=normalize_runner_name(intent.runner_name),
        trap=intent.trap,
        side=str(intent.side or "").upper(),
        row=runner_row,
        bet_ref=normalise_gruss_bet_ref(bet_ref),
        pending_confirmation=bool(pending_confirmation),
    )


def _cancel_intent_from_state(
    ladder_id: str,
    state: _ActivePreLadderState,
    context: GrussRealOrderContext,
) -> OrderIntent:
    return OrderIntent(
        provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
        market_type=state.market_type or "PLACE",
        market_id=state.market_id,
        parent_id=context.course,
        runner_name=state.runner_name,
        trap=state.trap,
        side=state.side or "BACK",
        order_type="CANCEL",
        price=None,
        stake=None,
        strategy_id="PRE_CANCEL_BEFORE_POST",
        course_id=context.course,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        dry_run=False,
        selection_id=state.selection_id,
        execution_phase="POST",
        pre_ladder=True,
        ladder_id=ladder_id,
        ladder_step="CANCEL",
        ladder_tracking_key=ladder_id,
        gruss_planned_trigger="CANCEL",
    )


def _empty_cancel_intent(context: GrussRealOrderContext) -> OrderIntent:
    return OrderIntent(
        provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
        market_type="PLACE",
        market_id=context.place_market_id or "",
        parent_id=context.course,
        runner_name="",
        trap=None,
        side="BACK",
        order_type="CANCEL",
        price=None,
        stake=None,
        strategy_id="PRE_CANCEL_BEFORE_POST",
        course_id=context.course,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        dry_run=False,
        selection_id="",
        execution_phase="POST",
        pre_ladder=True,
        ladder_id="",
        ladder_step="CANCEL",
        ladder_tracking_key="",
        gruss_planned_trigger="CANCEL",
    )


def _active_ladder_post_key(state: _ActivePreLadderState) -> tuple[str, str, str]:
    return (
        str(state.market_id or "").strip(),
        str(state.selection_id if state.selection_id not in (None, "") else state.trap).strip(),
        str(state.market_type or "").strip().upper(),
    )


def _post_pending_key(intent: OrderIntent) -> tuple[str, str, str]:
    selection_id = intent.selection_id if intent.selection_id is not None else intent.trap
    return (
        str(intent.market_id or "").strip(),
        str(selection_id if selection_id not in (None, "") else intent.runner_name).strip(),
        str(intent.market_type or "").strip().upper(),
    )


def _post_pending_ladder_keys(intents: Iterable[OrderIntent]) -> set[tuple[str, str, str]]:
    return {
        _post_pending_key(intent)
        for intent in intents
        if _execution_phase(intent) == "POST"
    }


def _max_orders_key(intent: OrderIntent, context: GrussRealOrderContext) -> str:
    race_id = str(context.course or intent.course_id or intent.parent_id or intent.market_id).strip()
    return "|".join((race_id, _execution_phase(intent)))


def _execution_phase(intent: OrderIntent) -> str:
    phase = str(getattr(intent, "execution_phase", "") or "POST").strip().upper()
    return phase if phase in {"PRE", "POST"} else "POST"


def _is_pre_ladder_intent(intent: OrderIntent) -> bool:
    return bool(getattr(intent, "pre_ladder", False))


def _continuing_active_pre_ladder(intent: OrderIntent, active_ladder_id: str | None) -> bool:
    if not _is_pre_ladder_intent(intent) or not active_ladder_id:
        return False
    ladder_id = str(intent.ladder_id or intent.ladder_tracking_key or "").strip()
    return bool(ladder_id) and ladder_id == active_ladder_id and _ladder_step_index(intent.ladder_step) > 0


def _active_ladder_state_matches(
    state: _ActivePreLadderState,
    intent: OrderIntent,
    row: int,
) -> bool:
    selection_id = str(intent.selection_id if intent.selection_id is not None else intent.trap)
    if state.row and state.row != row:
        return False
    if state.market_type and state.market_type != str(intent.market_type or "").upper():
        return False
    if state.market_id and state.market_id != str(intent.market_id or ""):
        return False
    if state.selection_id and state.selection_id != selection_id:
        return False
    if state.trap is not None and intent.trap is not None and state.trap != intent.trap:
        return False
    if state.runner_name and normalize_runner_name(intent.runner_name) != state.runner_name:
        return False
    if state.side and state.side != str(intent.side or "").upper():
        return False
    return True


def _pre_ladder_course_key(intent: OrderIntent, context: GrussRealOrderContext) -> str:
    return str(context.course or intent.course_id or intent.parent_id or intent.market_id or "").strip()


def _is_final_ladder_step(intent: OrderIntent) -> bool:
    steps = _pre_ladder_steps_from_env()
    if not steps:
        return False
    return _ladder_step_index(intent.ladder_step) == len(steps) - 1


def _current_milestone(context: GrussRealOrderContext) -> int | None:
    value = context.milestone_seen if context.milestone_seen is not None else context.countdown_seconds
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _expected_ladder_step_for_context(context: GrussRealOrderContext) -> str:
    milestone = _current_milestone(context)
    steps = _pre_ladder_steps_from_env()
    if milestone not in steps:
        return ""
    return f"{steps.index(milestone) + 1}/{len(steps)}"


def _pre_ladder_steps_from_env() -> tuple[int, ...]:
    raw = os.getenv("DOGBOT_PRE_LADDER_STEPS", "52,38,26,16")
    steps: list[int] = []
    for chunk in raw.split(","):
        try:
            step = int(chunk.strip())
        except (TypeError, ValueError):
            continue
        if step >= 0:
            steps.append(step)
    return tuple(steps or (52, 38, 26, 16))


def _post_send_seconds_before_off() -> int:
    return _bounded_int_env("DOGBOT_POST_SEND_SECONDS_BEFORE_OFF", 1, 0, 60)


def _post_allow_after_scheduled_off_seconds() -> int:
    return _bounded_int_env("DOGBOT_POST_ALLOW_AFTER_SCHEDULED_OFF_SECONDS", 5, 0, 60)


def _post_bet_ref_required() -> bool:
    return _env_bool("DOGBOT_POST_BET_REF_REQUIRED", True)


def _post_independent_mode_enabled(batch_context: dict[str, Any] | None = None) -> bool:
    context = batch_context or {}
    raw = context.get("pre_post_independent")
    if raw not in (None, ""):
        return str(raw).strip().casefold() in {"1", "true", "yes", "on", "y"}
    return _env_bool("DOGBOT_PRE_POST_INDEPENDENT", False)


def _post_bet_ref_wait_ms() -> int:
    return _bounded_int_env("DOGBOT_POST_BET_REF_WAIT_MS", 8000, 0, 30000)


def _post_bet_ref_poll_ms() -> int:
    return _bounded_int_env("DOGBOT_POST_BET_REF_POLL_MS", 250, 10, 2000)


def _post_clear_after_bet_ref() -> bool:
    return _env_bool("DOGBOT_POST_CLEAR_AFTER_BET_REF", True)


def _post_command_cells_clear_delay_ms() -> int:
    if os.getenv("DOGBOT_POST_COMMAND_CELLS_CLEAR_DELAY_MS") in (None, ""):
        return _command_cells_clear_delay_ms()
    return _bounded_int_env("DOGBOT_POST_COMMAND_CELLS_CLEAR_DELAY_MS", 1000, 0, 30000)


def _post_countdown_allowed(countdown: int | None) -> bool:
    if countdown is None:
        return False
    try:
        value = int(countdown)
    except (TypeError, ValueError):
        return False
    return -_post_allow_after_scheduled_off_seconds() <= value <= _post_send_seconds_before_off()


def _pre_cancel_before_post_enabled() -> bool:
    return _env_bool("DOGBOT_PRE_CANCEL_BEFORE_POST", True)


def _pre_cancel_only_if_post_pending() -> bool:
    return _env_bool("DOGBOT_PRE_CANCEL_ONLY_IF_POST_PENDING", True)


def _pre_cancel_seconds_before_off() -> int:
    return _bounded_int_env("DOGBOT_PRE_CANCEL_SECONDS_BEFORE_OFF", 1, 0, 60)


def _pre_initial_batch_write_grace_seconds() -> int:
    return _bounded_int_env("DOGBOT_PRE_INITIAL_BATCH_WRITE_GRACE_SECONDS", 10, 0, 60)


def _replace_min_countdown_seconds() -> int:
    return _bounded_int_env("DOGBOT_GRUSS_REPLACE_MIN_COUNTDOWN_SECONDS", 10, 0, 300)


def _pre_ignore_stale_price_before_write() -> bool:
    return _env_bool("DOGBOT_PRE_IGNORE_STALE_PRICE_BEFORE_WRITE", True)


def _pre_batch_late_write_allowed(
    intent: OrderIntent,
    context: GrussRealOrderContext,
    milestone: int,
    countdown_at_write: int,
) -> bool:
    if not context.pre_batch_milestone_authorized:
        return False
    if context.pre_batch_milestone_seconds != milestone:
        return False
    if _ladder_step_index(intent.ladder_step) != 0:
        return False
    started = context.pre_batch_started_countdown_seconds
    if started is None:
        return False
    grace = (
        context.pre_batch_write_grace_seconds
        if context.pre_batch_write_grace_seconds is not None
        else _pre_initial_batch_write_grace_seconds()
    )
    try:
        started_value = int(started)
        grace_value = int(grace)
    except (TypeError, ValueError):
        return False
    if grace_value < 0:
        return False
    return started_value + 1 >= countdown_at_write >= max(1, started_value - grace_value)


def _pre_batch_late_write_seconds_after_start(
    started_countdown: int | None,
    countdown_at_write: int | None,
) -> int | None:
    if started_countdown is None or countdown_at_write is None:
        return None
    try:
        return max(0, int(started_countdown) - int(countdown_at_write))
    except (TypeError, ValueError):
        return None


def _is_valid_pre_ladder_milestone(
    intent: OrderIntent,
    context: GrussRealOrderContext,
) -> bool:
    if not _is_pre_ladder_intent(intent) or _execution_phase(intent) != "PRE":
        return False
    try:
        countdown = int(context.countdown_seconds)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return countdown in _pre_ladder_steps_from_env()


def _countdown_authorization_reason(
    intent: OrderIntent,
    context: GrussRealOrderContext,
    *,
    write_no_trigger_guard: bool,
) -> str:
    if context.countdown_seconds is None:
        return "countdown_seconds_unavailable"
    if context.countdown_seconds < 0:
        return "countdown_elapsed"
    if _is_pre_ladder_intent(intent):
        if _is_valid_pre_ladder_milestone(intent, context):
            return "pre_ladder_valid_milestone"
        return "pre_ladder_invalid_milestone"
    if _execution_phase(intent) == "POST":
        if _post_countdown_allowed(context.countdown_seconds):
            return "post_countdown_allowed"
        return "after_off_do_not_write" if context.countdown_seconds < 0 else "countdown_above_post_send_seconds"
    limit = 2 if write_no_trigger_guard else 3
    if context.countdown_seconds <= limit:
        return "write_no_trigger_countdown_lte_2" if write_no_trigger_guard else "post_countdown_lte_3"
    return (
        "countdown_above_2_seconds"
        if write_no_trigger_guard
        else "countdown_above_3_seconds"
    )


def _ladder_step_index(value: Any) -> int:
    text = str(value or "").strip()
    if "/" not in text:
        return 0
    try:
        return max(0, int(text.split("/", 1)[0]) - 1)
    except ValueError:
        return 0


def _clean_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _bet_ref_lookup_sources_from_env() -> tuple[str, ...]:
    raw = os.getenv("DOGBOT_GRUSS_BET_REF_LOOKUP_SOURCES", "ROW_T,SELECTIONS_SHEET")
    sources: list[str] = []
    for chunk in raw.split(","):
        source = chunk.strip().upper()
        if source in {"ROW_T", "SELECTIONS_SHEET"} and source not in sources:
            sources.append(source)
    return tuple(sources or ["ROW_T", "SELECTIONS_SHEET"])


def _detect_selection_columns(values: list[list[Any]]) -> tuple[int, dict[str, int]]:
    best_index = 0
    default_columns: dict[str, int] = {
        "runner": 0,
        "bet_ref": 1,
        "side": 2,
        "odds": 3,
        "stake": 4,
    }
    best_columns: dict[str, int] = dict(default_columns)
    best_score = 0
    for row_index, row in enumerate(values[:10]):
        columns: dict[str, int] = {}
        for column_index, value in enumerate(row):
            header = _normalise_header(value)
            if not header:
                continue
            if header in {"selection", "selection name", "runner", "runner name", "dog", "greyhound"}:
                columns["runner"] = column_index
            elif header in {"selection id", "selection_id", "selectionid", "runner id", "runner_id"}:
                columns["selection_id"] = column_index
            elif header in {"bet ref", "bet_ref", "betref", "bet id", "bet_id", "reference"}:
                columns["bet_ref"] = column_index
            elif header in {"bet type", "bet_type", "side", "back lay"}:
                columns["side"] = column_index
            elif header in {"odds", "price"}:
                columns["odds"] = column_index
            elif header in {"stake", "size"}:
                columns["stake"] = column_index
            elif header in {"amount"}:
                columns["amount"] = column_index
            elif header in {"average odds", "avg odds", "avg matched", "average matched odds"}:
                columns["average_odds"] = column_index
            elif header in {"result", "status"}:
                columns["result"] = column_index
            elif header in {"market name", "market", "market title"}:
                columns["market_name"] = column_index
            elif header in {"market id", "market_id", "marketid"}:
                columns["market_id"] = column_index
            elif header in {"market type", "market_type", "markettype"}:
                columns["market_type"] = column_index
            elif header in {"req odds", "requested odds", "request odds", "requested price", "req price"}:
                columns["req_odds"] = column_index
            elif header in {"req stake", "requested stake", "request stake"}:
                columns["req_stake"] = column_index
            elif header in {"matched odds", "matched price"}:
                columns["matched_odds"] = column_index
            elif header in {"matched stake", "matched amount"}:
                columns["matched_stake"] = column_index
            elif header in {"time", "timestamp", "bet time", "placed time", "submitted", "submitted time"}:
                columns["timestamp"] = column_index
        score = len(set(columns) & {"runner", "bet_ref", "side", "stake", "amount", "req_stake"})
        if score > best_score:
            best_index = row_index
            best_columns = dict(columns)
            best_score = score
    if best_score <= 0:
        return best_index, default_columns
    return best_index, best_columns


def _normalise_header(value: Any) -> str:
    return " ".join(_clean_text(value).replace("_", " ").lower().split())


def _value_at(row: list[Any], index: int | None) -> Any:
    if index is None or index < 0 or index >= len(row):
        return None
    return row[index]


def _normalise_selection_side(value: Any) -> str:
    text = _clean_text(value).upper()
    if text in {"B", "BACK"}:
        return "BACK"
    if text in {"L", "LAY"}:
        return "LAY"
    return text


def _post_selection_reject_reason(
    candidate: _SelectionsBetRefCandidate,
    context: GrussRealOrderContext,
    *,
    post_write_timestamp: str,
    existing_bet_ref_before: str,
    expected_market_id: str,
    expected_market_type: str,
    expected_runner: str,
    expected_trap: int | None,
    expected_selection_id: str,
    expected_side: str,
    expected_stake: float | None,
    expected_price: float | None,
) -> str:
    if not candidate.bet_ref:
        return "missing_bet_ref"
    if not is_valid_bet_ref(candidate.bet_ref):
        return "invalid_bet_ref"
    existing = normalise_gruss_bet_ref(existing_bet_ref_before)
    if is_valid_bet_ref(existing) and strip_gruss_ref_suffix(candidate.bet_ref) == strip_gruss_ref_suffix(existing):
        return "existing_pre_bet_ref"

    placed_at = _parse_iso_timestamp(candidate.timestamp)
    written_at = _parse_iso_timestamp(post_write_timestamp)
    if placed_at is None:
        return "timestamp_missing"
    if written_at is None:
        return "post_write_timestamp_missing"
    if placed_at < written_at:
        return "timestamp_before_post_write"

    market_type = _normalise_post_market_type(expected_market_type)
    candidate_market_type = _normalise_post_market_type(candidate.market_type)
    if candidate_market_type and market_type and candidate_market_type != market_type:
        return "market_type_mismatch"
    if market_type == "PLACE" and _market_name_is_clear_non_place(candidate.market_name):
        return "market_type_mismatch"
    if market_type == "WIN" and _market_name_is_place(candidate.market_name):
        return "market_type_mismatch"

    if expected_market_id:
        if candidate.market_id:
            if candidate.market_id != expected_market_id:
                return "market_id_mismatch"
        elif not _post_market_name_matches_course(candidate.market_name, context.course):
            return "market_id_missing"

    if expected_selection_id and candidate.selection_id and candidate.selection_id != expected_selection_id:
        return "selection_id_mismatch"

    if expected_trap is not None and candidate.trap is not None:
        if candidate.trap != expected_trap:
            return "runner_mismatch"
    else:
        expected_runner_key = _selection_runner_key(expected_runner)
        candidate_runner_key = _selection_runner_key(candidate.runner)
        if not expected_runner_key or not candidate_runner_key:
            return "runner_missing"
        if expected_runner_key != candidate_runner_key and not _selection_runner_loose_match(
            expected_runner_key,
            candidate_runner_key,
        ):
            return "runner_mismatch"

    if not expected_side:
        return "expected_side_missing"
    if not candidate.side:
        return "side_missing"
    if candidate.side != expected_side:
        return "side_mismatch"

    if expected_stake is None:
        return "expected_stake_missing"
    if candidate.stake is None:
        return "stake_missing"
    if not _stake_close(candidate.stake, expected_stake):
        return "stake_mismatch"

    if expected_price is None:
        return "expected_price_missing"
    candidate_prices = [
        price
        for price in (candidate.req_odds, candidate.matched_odds, candidate.average_odds)
        if price is not None
    ]
    if not candidate_prices:
        return "price_missing"
    if not any(_odds_within_tick_tolerance(price, expected_price) for price in candidate_prices):
        return "price_mismatch"
    return ""


def _normalise_post_market_type(value: Any) -> str:
    text = _normalise_header(value).upper()
    if not text:
        return ""
    if "PLACE" in text or "PLACED" in text:
        return "PLACE"
    if "WIN" in text:
        return "WIN"
    return text.replace(" ", "_")


def _post_market_name_matches_course(market_name: Any, course: Any) -> bool:
    market_text = _normalise_header(market_name)
    course_text = _clean_text(course)
    if not market_text or not course_text:
        return False
    course_parts = [part.strip() for part in re.split(r"[\\/]+", course_text) if part.strip()]
    candidates = [_normalise_header(course_text)]
    if course_parts:
        candidates.append(_normalise_header(course_parts[-1]))
    return any(candidate and candidate in market_text for candidate in candidates)


def _summarise_post_selection_rejections(reasons: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    for reason in reasons:
        key = reason or "unknown"
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "no_selection_candidates"
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ";".join(f"{reason}:{count}" for reason, count in ranked[:5])


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if _clean_text(value):
            return value
    return None


def _match_selection_bet_ref_candidate(
    state: _ActivePreLadderState,
    result: GrussRealOrderResult,
    candidates: Iterable[_SelectionsBetRefCandidate],
    *,
    used_bet_refs: set[str],
) -> tuple[_SelectionsBetRefCandidate | None, str, str]:
    scored: list[tuple[int, _SelectionsBetRefCandidate, str]] = []
    for candidate in candidates:
        score, reason = _score_selection_bet_ref_candidate(
            state,
            result,
            candidate,
            used_bet_refs=used_bet_refs,
        )
        scored.append((score, candidate, reason))
    scored.sort(key=lambda item: (item[0], item[1].row_number), reverse=True)
    top_candidates = _format_top_selection_candidates(scored)
    reliable = [(score, candidate, reason) for score, candidate, reason in scored if score >= 80]
    if not reliable:
        return None, _best_selection_rejection_reason(scored), top_candidates
    score, match, reason = reliable[0]
    if len(reliable) > 1 and reliable[1][0] == score:
        return None, "ambiguous_selection_row", top_candidates
    return match, reason, top_candidates


def _match_selection_activity_evidence_candidate(
    state: _ActivePreLadderState,
    result: GrussRealOrderResult,
    candidates: Iterable[_SelectionsBetRefCandidate],
) -> tuple[_SelectionsBetRefCandidate | None, str]:
    scored: list[tuple[int, _SelectionsBetRefCandidate, str]] = []
    for candidate in candidates:
        score, reason = _score_selection_activity_evidence_candidate(state, result, candidate)
        scored.append((score, candidate, reason))
    scored.sort(key=lambda item: (item[0], item[1].row_number), reverse=True)
    reliable = [(score, candidate, reason) for score, candidate, reason in scored if score >= 80]
    if not reliable:
        return None, _best_selection_rejection_reason(scored)
    score, match, reason = reliable[0]
    if len(reliable) > 1 and reliable[1][0] == score:
        return None, "ambiguous_selection_activity_evidence"
    return match, reason


def _score_selection_activity_evidence_candidate(
    state: _ActivePreLadderState,
    result: GrussRealOrderResult,
    candidate: _SelectionsBetRefCandidate,
) -> tuple[int, str]:
    score = 0
    reasons: list[str] = []
    if state.trap is not None and candidate.trap is not None:
        if candidate.trap != state.trap:
            return 0, "trap_mismatch"
        score += 55
        reasons.append("trap_match")
    else:
        state_runner = _selection_runner_key(state.runner_name)
        candidate_runner = _selection_runner_key(candidate.runner)
        if not state_runner or not candidate_runner:
            return 0, "runner_missing"
        if state_runner == candidate_runner:
            score += 55
            reasons.append("runner_match")
        elif _selection_runner_loose_match(state_runner, candidate_runner):
            score += 45
            reasons.append("runner_loose_match")
        else:
            return 0, "runner_mismatch"

    if state.side and candidate.side:
        if candidate.side != state.side:
            return 0, "side_mismatch"
        score += 20
        reasons.append("side_match")
    elif state.side and not candidate.side:
        reasons.append("side_missing")

    requested_stake = _result_requested_stake(result)
    if candidate.stake is None:
        reasons.append("stake_missing")
    elif requested_stake is None or _stake_close(candidate.stake, requested_stake):
        score += 15
        reasons.append("stake_match")
    else:
        return 0, "stake_mismatch"

    if _market_name_is_place(candidate.market_name):
        score += 10
        reasons.append("place_market_match")
    elif _market_name_is_clear_non_place(candidate.market_name):
        return 0, "market_mismatch"
    elif candidate.market_name:
        reasons.append("market_unknown")
    else:
        reasons.append("market_missing")

    requested_odds = _result_requested_odds(result)
    if candidate.req_odds is None:
        reasons.append("req_odds_missing")
    elif requested_odds is None or _odds_within_tick_tolerance(candidate.req_odds, requested_odds):
        score += 10
        reasons.append("req_odds_match")
    else:
        return 0, "req_odds_mismatch"

    activity_markers: list[str] = []
    if is_valid_bet_ref(candidate.bet_ref):
        activity_markers.append("valid_bet_ref")
    if candidate.matched_stake is not None and candidate.matched_stake > 0:
        activity_markers.append("matched_stake_positive")
    if candidate.average_odds is not None:
        activity_markers.append("average_odds_present")
    if candidate.matched_odds is not None:
        activity_markers.append("matched_odds_present")
    if candidate.result:
        activity_markers.append("result_present")
    if candidate.timestamp:
        activity_markers.append("timestamp_present")
    if activity_markers:
        score += 10
        reasons.extend(activity_markers)
    elif score < 90:
        return 0, "no_activity_marker"
    else:
        reasons.append("selection_row_identity_match")
    return score, "+".join(reasons)


def _score_selection_bet_ref_candidate(
    state: _ActivePreLadderState,
    result: GrussRealOrderResult,
    candidate: _SelectionsBetRefCandidate,
    *,
    used_bet_refs: set[str],
) -> tuple[int, str]:
    if not candidate.bet_ref:
        return 0, "missing_bet_ref"
    if not is_valid_bet_ref(candidate.bet_ref):
        return 0, "invalid_bet_ref"
    if candidate.bet_ref in used_bet_refs:
        return 0, "duplicate_bet_ref"

    score = 0
    reasons: list[str] = []
    if state.trap is not None and candidate.trap is not None:
        if candidate.trap != state.trap:
            return 0, "trap_mismatch"
        score += 55
        reasons.append("trap_match")
    else:
        state_runner = _selection_runner_key(state.runner_name)
        candidate_runner = _selection_runner_key(candidate.runner)
        if not state_runner or not candidate_runner:
            return 0, "runner_missing"
        if state_runner == candidate_runner:
            score += 55
            reasons.append("runner_match")
        elif _selection_runner_loose_match(state_runner, candidate_runner):
            score += 45
            reasons.append("runner_loose_match")
        else:
            return 0, "runner_mismatch"

    if state.side and candidate.side:
        if candidate.side != state.side:
            return 0, "side_mismatch"
        score += 20
        reasons.append("side_match")
    elif state.side and not candidate.side:
        reasons.append("side_missing")

    requested_stake = _result_requested_stake(result)
    if candidate.stake is None:
        reasons.append("stake_missing")
    elif requested_stake is None or _stake_close(candidate.stake, requested_stake):
        score += 15
        reasons.append("stake_match")
    else:
        return 0, "stake_mismatch"

    if _market_name_is_place(candidate.market_name):
        score += 10
        reasons.append("place_market_match")
    elif _market_name_is_clear_non_place(candidate.market_name):
        return 0, "market_mismatch"
    elif candidate.market_name:
        reasons.append("market_unknown")
    else:
        reasons.append("market_missing")

    requested_odds = _result_requested_odds(result)
    if candidate.req_odds is None:
        reasons.append("req_odds_missing")
    elif requested_odds is None or _odds_within_tick_tolerance(candidate.req_odds, requested_odds):
        score += 10
        reasons.append("req_odds_match")
    else:
        return 0, "req_odds_mismatch"

    reasons.append(f"result={candidate.result}" if candidate.result else "result_pending")
    return score, "+".join(reasons)


def _best_selection_rejection_reason(scored: list[tuple[int, _SelectionsBetRefCandidate, str]]) -> str:
    if not scored:
        return "no_selection_candidates"
    best_score, _candidate, reason = scored[0]
    if best_score <= 0:
        return f"no_matching_selection_row:{reason}"
    return f"selection_score_below_threshold:{best_score}:{reason}"


def _selection_runner_key(value: Any) -> str:
    text = normalize_runner_name(_clean_text(value)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _selection_runner_loose_match(left: str, right: str) -> bool:
    compact_left = left.replace(" ", "")
    compact_right = right.replace(" ", "")
    if len(compact_left) < 4 or len(compact_right) < 4:
        return False
    return compact_left in compact_right or compact_right in compact_left


def _stake_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=0.05)


def _matched_stake_delta(before: float | None, after: float | None) -> float | None:
    if after is None:
        return None
    base = before if before is not None else 0.0
    delta = after - base
    if delta <= 0.05:
        return 0.0
    return round(delta, 6)


def _result_requested_stake(result: GrussRealOrderResult) -> float | None:
    for value in (result.stake_used, result.post_write_stake_value, result.stake_original):
        number = _positive_float_or_none(value)
        if number is not None:
            return number
    return None


def _result_requested_odds(result: GrussRealOrderResult) -> float | None:
    for _address, value in result.write_plan:
        number = _positive_float_or_none(value)
        if number is not None:
            return number
    for value in (result.post_write_odds_value,):
        number = _positive_float_or_none(value)
        if number is not None:
            return number
    return None


def _result_matched_evidence_found(result: GrussRealOrderResult) -> bool:
    for value in (
        result.matched_after_step_stake,
        _matched_stake_value(result.matched_stake_cell_value),
    ):
        if value is not None and value > 0:
            return True
    return bool(
        result.matched_after_step_avg_odds is not None
        or _positive_float_or_none(result.avg_matched_odds_cell_value) is not None
    )


def _market_name_is_place(value: Any) -> bool:
    text = _normalise_header(value)
    return any(marker in text for marker in ("to be placed", "place", "placed"))


def _market_name_is_clear_non_place(value: Any) -> bool:
    text = _normalise_header(value)
    return bool(text) and "win" in text and not _market_name_is_place(text)


def _odds_within_tick_tolerance(left: float, right: float) -> bool:
    tick = max(_betfair_tick_size(left), _betfair_tick_size(right))
    return abs(left - right) <= (tick * 2.0 + 1e-9)


def _betfair_tick_size(price: float) -> float:
    if price < 2:
        return 0.01
    if price < 3:
        return 0.02
    if price < 4:
        return 0.05
    if price < 6:
        return 0.1
    if price < 10:
        return 0.2
    if price < 20:
        return 0.5
    if price < 30:
        return 1.0
    if price < 50:
        return 2.0
    if price < 100:
        return 5.0
    return 10.0


def _format_recent_selection_rows(candidates: Iterable[_SelectionsBetRefCandidate], *, limit: int = 8) -> str:
    recent = sorted(candidates, key=lambda candidate: candidate.row_number, reverse=True)[:limit]
    parts: list[str] = []
    for candidate in reversed(recent):
        parts.append(
            "|".join(
                (
                    f"row={candidate.row_number}",
                    f"time={candidate.timestamp}",
                    f"bet_ref={candidate.bet_ref}",
                    f"selection={candidate.runner}",
                    f"bet_type={candidate.side}",
                    f"amount={'' if candidate.stake is None else candidate.stake}",
                    f"average_odds={'' if candidate.average_odds is None else candidate.average_odds}",
                    f"result={candidate.result}",
                    f"market_name={candidate.market_name}",
                    f"req_odds={'' if candidate.req_odds is None else candidate.req_odds}",
                    f"req_stake={'' if candidate.stake is None else candidate.stake}",
                    f"matched_odds={'' if candidate.matched_odds is None else candidate.matched_odds}",
                    f"matched_stake={'' if candidate.matched_stake is None else candidate.matched_stake}",
                )
            )
        )
    return " ; ".join(parts)


def _format_selection_rows_for_market(
    candidates: Iterable[_SelectionsBetRefCandidate],
    market_query: str,
) -> str:
    fragments = _market_query_fragments(market_query)
    if not fragments:
        return ""
    matches = [
        candidate
        for candidate in candidates
        if any(fragment in _normalised_search_text(candidate.market_name) for fragment in fragments)
    ]
    return _format_recent_selection_rows(matches[-12:])


def _format_selection_rows_for_runner(
    candidates: Iterable[_SelectionsBetRefCandidate],
    state: _ActivePreLadderState,
) -> str:
    state_runner = _selection_runner_key(state.runner_name)
    matches: list[_SelectionsBetRefCandidate] = []
    for candidate in candidates:
        if state.trap is not None and candidate.trap is not None and candidate.trap == state.trap:
            matches.append(candidate)
            continue
        candidate_runner = _selection_runner_key(candidate.runner)
        if state_runner and candidate_runner and (
            state_runner == candidate_runner
            or _selection_runner_loose_match(state_runner, candidate_runner)
        ):
            matches.append(candidate)
    return _format_recent_selection_rows(matches[-12:])


def _format_top_selection_candidates(
    scored: Iterable[tuple[int, _SelectionsBetRefCandidate, str]]
) -> str:
    parts: list[str] = []
    for score, candidate, reason in list(scored)[:5]:
        parts.append(
            "|".join(
                (
                    f"row={candidate.row_number}",
                    f"time={candidate.timestamp}",
                    f"score={score}",
                    f"reason={reason}",
                    f"bet_ref={candidate.bet_ref}",
                    f"selection={candidate.runner}",
                    f"side={candidate.side}",
                    f"stake={'' if candidate.stake is None else candidate.stake}",
                    f"req_odds={'' if candidate.req_odds is None else candidate.req_odds}",
                    f"market_name={candidate.market_name}",
                )
            )
        )
    return " ; ".join(parts)


def _format_header_row(row: Iterable[Any]) -> str:
    parts: list[str] = []
    for index, value in enumerate(row, start=1):
        text = _clean_text(value)
        if text:
            parts.append(f"{_excel_column_name(index)}={text}")
    return ",".join(parts)


def _excel_column_name(index: int) -> str:
    result = ""
    value = max(1, int(index))
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _workbook_sheet_names(bridge: Any) -> str:
    try:
        if hasattr(bridge, "sheet_names"):
            names = bridge.sheet_names()
        else:
            workbook = getattr(bridge, "workbook", None)
            names = [str(sheet.name) for sheet in getattr(workbook, "sheets", [])]
    except Exception as exc:
        return f"sheet_names_failed:{exc}"
    return ",".join(str(name) for name in names)


def _selections_scan_rows() -> int:
    return _bounded_int_env("DOGBOT_GRUSS_SELECTIONS_SCAN_ROWS", 1000, 120, 5000)


def _diagnostic_keep_triggers() -> bool:
    return _env_bool("DOGBOT_GRUSS_DIAGNOSTIC_KEEP_TRIGGERS", False)


def _market_query_fragments(value: Any) -> tuple[str, ...]:
    text = _clean_text(value)
    if not text:
        return ()
    raw_parts = [text]
    raw_parts.extend(part for part in re.split(r"[\\/|>]+", text) if part.strip())
    fragments: list[str] = []
    for part in raw_parts:
        normalised = _normalised_search_text(part)
        if not normalised or normalised in {"greyhound racing", "pgr", "sis"}:
            continue
        fragments.append(normalised)
    fragments.sort(key=len, reverse=True)
    return tuple(dict.fromkeys(fragments))


def _normalised_search_text(value: Any) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _int_from_context(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _set_result_attr(result: GrussRealOrderResult, name: str, value: Any) -> None:
    object.__setattr__(result, name, value)


def _intent_from_result(result: GrussRealOrderResult) -> OrderIntent:
    return OrderIntent(
        provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
        market_type=str(result.market_type or result.excel_sheet or ""),
        market_id=str(result.market_id or result.post_expected_market_id or ""),
        parent_id=str(result.parent_id or "") or None,
        runner_name=str(result.runner_name or result.post_expected_runner or ""),
        trap=result.trap,
        side=str(result.side or result.post_expected_side or ""),
        order_type=str(result.order_type or "LIMIT"),
        price=(
            result.post_expected_price
            if result.post_expected_price is not None
            else result.final_lim_price if result.final_lim_price is not None else result.requested_price
        ),
        stake=result.post_expected_stake if result.post_expected_stake is not None else result.stake_used,
        strategy_id=str(result.strategy_id or result.triggered_systems or ""),
        course_id=str(result.course_id or "") or None,
        timestamp=str(result.signal_timestamp or result.write_timestamp or result.post_write_timestamp or ""),
        dry_run=False,
        selection_id=result.selection_id if result.selection_id is not None else result.post_expected_selection_id,
        execution_phase="POST",
    )


def normalise_gruss_bet_ref(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = _clean_text(value)
    if not text:
        return ""
    upper = text.upper()
    if upper in {"PENDING", "CANCELLED", "CANCELED", "LAPSED", "VARIOUS"} or upper.startswith("RESULT_"):
        return upper

    suffix_n = upper.endswith("N")
    core = text[:-1].strip() if suffix_n else text
    if re.fullmatch(r"\d+\.0", core):
        core = core[:-2]
    if not re.fullmatch(r"\d{8,}", core):
        return f"{core.upper()}N" if suffix_n else core
    return f"{core}N" if suffix_n else core


def is_valid_bet_ref(value: Any) -> bool:
    text = normalise_gruss_bet_ref(value).upper()
    if not text or is_terminal_bet_status(text):
        return False
    if is_pending_bet_ref_status(text) or text == "VARIOUS":
        return False
    return bool(re.fullmatch(r"[0-9]{8,}N?", text))


def strip_gruss_ref_suffix(value: Any) -> str:
    text = normalise_gruss_bet_ref(value).upper()
    if re.fullmatch(r"[0-9]{8,}N", text):
        return text[:-1]
    return text


def is_terminal_bet_status(value: Any) -> bool:
    text = normalise_gruss_bet_ref(value).upper()
    return text in {"CANCELLED", "CANCELED", "LAPSED"} or text.startswith("RESULT_")


def is_pending_bet_ref_status(value: Any) -> bool:
    return normalise_gruss_bet_ref(value).upper() in {"PENDING", "PENDINGR"}


def _is_suspended_or_closed_market_status(value: Any) -> bool:
    text = _clean_text(value).casefold()
    return "suspended" in text or "closed" in text


def _is_untradable_market_status(value: Any) -> bool:
    text = _clean_text(value).casefold()
    if not text:
        return False
    if "suspended" in text or "closed" in text:
        return True
    if "not in play" in text or "not-in-play" in text:
        return False
    return "in play" in text or "in-play" in text or text == "inplay"


def _matched_stake_value(value: Any) -> float | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(0.0, number)


def _runner_order_in_sheet(row: int | None) -> int | str:
    if row is None:
        return ""
    try:
        return max(0, int(row) - 4)
    except (TypeError, ValueError):
        return ""


def _read_cell_quiet(bridge: Any, sheet_name: str, address: str) -> Any:
    if not sheet_name or not address:
        return None
    try:
        return bridge.read_cell(sheet_name, address)
    except Exception:
        return None


def _blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def _ladder_step_count(value: Any) -> int | None:
    text = str(value or "").strip()
    if "/" not in text:
        return None
    try:
        return max(0, int(text.split("/", 1)[1]))
    except (TypeError, ValueError):
        return None


def _find_runner_row_in_values(row_values: Iterable[tuple[int, Any]], intent: OrderIntent) -> int | None:
    normalized_target = normalize_runner_name(intent.runner_name)
    for row_number, value in row_values:
        trap_matches = intent.trap is None or extract_trap(value) == intent.trap
        name_matches = not normalized_target or normalize_runner_name(value) == normalized_target
        if trap_matches and name_matches:
            return row_number
    return None


def _selection_id_for_log(intent: OrderIntent) -> str:
    value = intent.selection_id if intent.selection_id is not None else intent.trap
    if value in (None, ""):
        value = intent.runner_name
    return str(value)


def _strip_runner_trap_for_log(value: Any) -> str:
    text = _clean_text(value)
    return re.sub(r"^\s*[\[(]?[1-8][\]).:\-\s]+", "", text).strip() or text


def _join_unique(values: Iterable[Any]) -> str:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return "|".join(result)


def _normalise_identifier(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _flatten_single_column(values: list[list[Any]]) -> Iterable[Any]:
    if len(values) == 1 and values and len(values[0]) > 1:
        yield from values[0]
        return
    for row in values:
        yield row[0] if row else None


def _positive_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _valid_order_price(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 1.01


def _price_to_write_for_gruss(intent: OrderIntent, trigger: str) -> float:
    price = float(intent.price)
    if not _should_tick_round_limit_price(intent, trigger):
        return price
    try:
        rounded = float(round_final_lim_to_ladder_tick(str(intent.side or "").upper(), max(price, 1.01)))
    except (TypeError, ValueError):
        return price
    if not _is_pre_ladder_intent(intent):
        return rounded
    target = _pre_value_target_effective_from_intent(intent)
    if target is None:
        return rounded
    target_tick = float(round_final_lim_to_ladder_tick(str(intent.side or "").upper(), target))
    if str(intent.side or "").upper() == "BACK":
        return max(rounded, target_tick)
    return min(rounded, target_tick)


def _pre_value_target_effective_from_intent(intent: OrderIntent) -> float | None:
    for value in (
        getattr(intent, "pre_value_target_price", None),
        getattr(intent, "computed_limit_price_effective", None),
        getattr(intent, "computed_limit_price_raw", None),
        getattr(intent, "price", None),
    ):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return max(number, 1.01)
    return None


def _value_clamp_was_needed(intent: OrderIntent) -> bool:
    if not _is_pre_ladder_intent(intent):
        return False
    target = _pre_value_target_effective_from_intent(intent)
    if target is None:
        return False
    try:
        raw_price = float(getattr(intent, "price", None))
        rounded = float(round_final_lim_to_ladder_tick(str(intent.side or "").upper(), max(raw_price, 1.01)))
        target_tick = float(round_final_lim_to_ladder_tick(str(intent.side or "").upper(), target))
    except (TypeError, ValueError):
        return False
    if str(intent.side or "").upper() == "BACK":
        return rounded < target_tick
    return rounded > target_tick


def _should_tick_round_limit_price(intent: OrderIntent, trigger: str) -> bool:
    if str(getattr(intent, "order_type", "") or "").upper() != "LIMIT":
        return False
    if str(trigger or "").upper() in {"BACKSP", "LAYSP"}:
        return False
    return str(getattr(intent, "side", "") or "").upper() in {"BACK", "LAY"}


def _is_valid_betfair_tick(value: Any) -> bool:
    try:
        number = float(value)
        rounded = float(round_to_betfair_tick(number))
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and math.isclose(number, rounded, rel_tol=1e-9, abs_tol=1e-9)


def _odds_value_from_plan(
    plan: Iterable[tuple[str, Any]],
    odds_address: str,
) -> float | None:
    for address, value in plan:
        if str(address).upper() != str(odds_address).upper():
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None
    return None


def _values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            actual_number = float(actual)
        except (TypeError, ValueError):
            return False
        return math.isfinite(actual_number) and math.isclose(
            float(expected),
            actual_number,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    return expected == actual


def _float_or_infinity(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.inf


def _positive_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 1.0:
        return None
    return number


def _finite_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_positive(*values: float | None) -> float | None:
    for value in values:
        number = _positive_float_or_none(value)
        if number is not None:
            return number
    return None


def _relative_distance(value: float | None, reference: float | None) -> float | None:
    value_number = _positive_float_or_none(value)
    reference_number = _positive_float_or_none(reference)
    if value_number is None or reference_number is None:
        return None
    return abs(value_number - reference_number) / reference_number


def _safe_read_cell(bridge: Any, sheet_name: str, address: str) -> Any:
    try:
        return bridge.read_cell(sheet_name, address)
    except Exception:
        return None


def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return min(max(value, minimum), maximum)


def _stale_price_limit() -> float:
    raw = os.getenv("DOGBOT_PRE_LADDER_MAX_STALE_PRICE_DISTANCE_PCT", "0.25")
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return 0.25
    if not math.isfinite(value) or value < 0:
        return 0.25
    return value


def _countdown_in_pre_ladder_window(milestone: int, countdown: int) -> bool:
    if milestone == 5:
        return 0 < countdown <= 5
    return milestone - 5 <= countdown <= milestone


def _seconds_between_iso_timestamps(start: str | None, end: str | None) -> float | None:
    start_dt = _parse_iso_timestamp(start)
    end_dt = _parse_iso_timestamp(end)
    if start_dt is None or end_dt is None:
        return None
    return round(max(0.0, (end_dt - start_dt).total_seconds()), 3)


def _seconds_since_iso_timestamp(timestamp: str | None) -> float | None:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return _seconds_between_iso_timestamps(timestamp, now)


def _iso_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _real_max_orders(real_test_mode: bool) -> int | None:
    raw = os.getenv("DOGBOT_GRUSS_REAL_MAX_ORDERS")
    if raw in (None, ""):
        return 1 if real_test_mode else None
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _real_max_orders_by_phase(real_test_mode: bool, default_max_orders: int | None) -> dict[str, int | None]:
    return {
        "PRE": _real_max_orders_for_env("DOGBOT_GRUSS_REAL_MAX_ORDERS_PRE", real_test_mode, default_max_orders),
        "POST": _real_max_orders_for_env("DOGBOT_GRUSS_REAL_MAX_ORDERS_POST", real_test_mode, default_max_orders),
    }


def _real_max_orders_for_env(
    name: str,
    real_test_mode: bool,
    default_max_orders: int | None,
) -> int | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default_max_orders
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _real_max_stake(real_test_mode: bool) -> float | None:
    raw = os.getenv("DOGBOT_GRUSS_REAL_MAX_STAKE")
    if raw in (None, ""):
        return 1.0 if real_test_mode else None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _variable_stake_cap_config_error(real_max_stake: float | None) -> str:
    if real_max_stake is None or real_max_stake <= 0:
        return "pre_ladder_real_variable_requires_positive_max_stake"
    hard_cap = _variable_stake_hard_cap()
    if hard_cap <= 0 or real_max_stake > hard_cap:
        return "pre_ladder_real_variable_max_stake_exceeds_hard_cap"
    if real_max_stake > 5.0 and not _env_bool("DOGBOT_GRUSS_REAL_ALLOW_VARIABLE_STAKE_OVER_5", False):
        return "pre_ladder_real_variable_over_5_requires_explicit_allow"
    return ""


def _variable_stake_hard_cap() -> float:
    raw = os.getenv("DOGBOT_GRUSS_REAL_VARIABLE_STAKE_HARD_CAP")
    if raw in (None, ""):
        return 10.0
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _trigger_clear_delay_ms() -> int:
    raw = os.getenv("DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS")
    if raw in (None, ""):
        return 1500
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 1500


def _command_cells_clear_delay_ms() -> int:
    raw = os.getenv("DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_DELAY_MS")
    if raw not in (None, ""):
        try:
            return max(0, int(str(raw).strip()))
        except (TypeError, ValueError):
            return 500
    return 500


def _command_cells_clear_columns_from_env() -> tuple[str, ...]:
    raw = os.getenv("DOGBOT_GRUSS_CLEAR_COMMAND_CELLS_COLUMNS", "Q,R,S")
    allowed = {"Q", "R", "S"}
    columns: list[str] = []
    for chunk in str(raw or "").split(","):
        column = chunk.strip().upper()
        if column in allowed and column not in columns:
            columns.append(column)
    return tuple(columns or ("Q", "R", "S"))


def _normalize_command_cell_addresses(addresses: Any) -> tuple[str, ...]:
    if addresses in (None, ""):
        return ()
    if isinstance(addresses, str):
        parts = re.split(r"[;,]", addresses)
    else:
        try:
            iterator = iter(addresses)
        except TypeError:
            parts = [str(addresses)]
        else:
            parts = []
            for address in iterator:
                if isinstance(address, str):
                    parts.extend(re.split(r"[;,]", address))
                else:
                    parts.append(str(address))
    return tuple(part.strip().upper() for part in parts if part and part.strip())


def _pre_ladder_trigger_clear_delay_override_ms() -> int | None:
    return None


def _excel_write_retry_backoff_ms() -> tuple[int, ...]:
    return _excel_com_retry_backoff_ms(
        retries_name="DOGBOT_GRUSS_EXCEL_WRITE_RETRIES",
        backoff_name="DOGBOT_GRUSS_EXCEL_WRITE_RETRY_BACKOFF_MS",
    )


def _excel_com_retry_backoff_ms(
    *,
    retries_name: str = "DOGBOT_GRUSS_EXCEL_COM_RETRIES",
    backoff_name: str = "DOGBOT_GRUSS_EXCEL_COM_RETRY_BACKOFF_MS",
) -> tuple[int, ...]:
    if retries_name == "DOGBOT_GRUSS_EXCEL_COM_RETRIES" and os.getenv(retries_name) in (None, ""):
        retries_name = "DOGBOT_GRUSS_EXCEL_WRITE_RETRIES"
    if backoff_name == "DOGBOT_GRUSS_EXCEL_COM_RETRY_BACKOFF_MS" and os.getenv(backoff_name) in (None, ""):
        backoff_name = "DOGBOT_GRUSS_EXCEL_WRITE_RETRY_BACKOFF_MS"
    retries = max(0, _int_env(retries_name, 3))
    if retries == 0:
        return ()
    raw = os.getenv(backoff_name, "300,600,900")
    values: list[int] = []
    for chunk in str(raw or "").split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            values.append(max(0, int(text)))
        except (TypeError, ValueError):
            continue
    if not values:
        values = [300, 600, 900]
    while len(values) < retries:
        values.append(values[-1])
    return tuple(values[:retries])


def _is_temporary_excel_write_error(exc: Exception) -> bool:
    args = getattr(exc, "args", ())
    int_args = {arg for arg in args if isinstance(arg, int)}
    text = " ".join(str(part) for part in (type(exc).__name__, exc, *args) if part is not None)
    lowered = text.casefold()
    normalized = (
        lowered.replace("’", "'")
        .replace("`", "'")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
    )
    return (
        -2147418111 in int_args
        or "-2147418111" in normalized
        or "call was rejected by callee" in normalized
        or "rejected by callee" in normalized
        or "rpc_e_call_rejected" in normalized
        or "appel a ete rejete" in normalized
        or ("appel" in normalized and "rejet" in normalized)
        or "excel_unavailable" in normalized
        or "this object does not support enumeration" in normalized
        or "application is busy" in normalized
        or "application busy" in normalized
    )


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _pre_ladder_real_max_ladders() -> int:
    return max(1, _int_env("DOGBOT_PRE_LADDER_REAL_MAX_LADDERS", 1))


def _pre_ladder_real_max_ladders_config_error() -> str:
    raw = os.getenv("DOGBOT_PRE_LADDER_REAL_MAX_LADDERS")
    if raw in (None, ""):
        return "pre_ladder_real_max_ladders_required"
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return "pre_ladder_real_max_ladders_invalid"
    if value <= 0:
        return "pre_ladder_real_max_ladders_invalid"
    return ""


def _pre_bet_ref_missing_max_retries() -> int:
    return _bounded_int_env("DOGBOT_PRE_LADDER_BET_REF_MISSING_MAX_RETRIES", 2, 0, 20)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on", "y"}


def _column_env(name: str, default: str) -> str:
    value = os.getenv(name, default).strip().upper()
    if not value.isalpha():
        raise ValueError(f"invalid Excel column in {name}: {value!r}")
    return value


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
