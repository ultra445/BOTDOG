from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dogbot.config import ORDER_PROVIDER_GRUSS_EXCEL_DRYRUN


ORDERS_GRUSS_DRYRUN_HEADER = [
    "timestamp",
    "provider",
    "market_type",
    "market_id",
    "parent_id",
    "course_id",
    "runner_name",
    "trap",
    "side",
    "order_type",
    "price",
    "stake",
    "strategy_id",
    "selection_id",
    "execution_phase",
    "triggered_systems",
    "triggered_prices",
    "pre_ladder",
    "ladder_id",
    "ladder_step",
    "ladder_tracking_key",
    "gruss_planned_trigger",
    "matched_stake",
    "status",
    "reason",
]


@dataclass(frozen=True)
class OrderIntent:
    provider: str
    market_type: str
    market_id: str
    parent_id: str | None
    runner_name: str
    trap: int | None
    side: str
    order_type: str
    price: float | None
    stake: float | None
    strategy_id: str
    course_id: str | None
    timestamp: str
    dry_run: bool
    selection_id: str | int | None = None
    execution_phase: str = "POST"
    triggered_systems: str | None = None
    triggered_prices: str | None = None
    stake_original: float | None = None
    stake_forced: bool = False
    force_test_bsp_place: bool = False
    force_test_back_place_limit: bool = False
    selected_reason: str | None = None
    selected_runner: str | None = None
    selected_trap: int | None = None
    selected_place_odds: float | None = None
    selected_place_back_odds: float | None = None
    selected_place_lay_odds: float | None = None
    price_used: float | None = None
    pre_ladder: bool = False
    ladder_id: str | None = None
    ladder_step: str | None = None
    ladder_tracking_key: str | None = None
    gruss_planned_trigger: str | None = None
    matched_stake: float | None = None
    signal_countdown_seconds: int | None = None
    market_reference_price_at_signal: float | None = None
    best_same_side_back_offer: float | None = None
    best_same_side_lay_offer: float | None = None
    ladder_plan_frozen: bool = False
    ladder_plan_created_step: str | int | None = None
    ladder_prices_frozen: str | None = None
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
    value_limit_skip_reason: str | None = None
    tick_rounding_direction: str | None = None
    best_same_side_offer_at_creation: float | None = None
    best_back_displayed: float | None = None
    best_lay_displayed: float | None = None
    start_price_source: str | None = None
    ladder_direction: str | None = None
    ladder_disabled_lim_not_in_ladder_direction: bool = False
    direct_lim_order_planned: bool = False
    direct_lim_order_written: bool = False
    no_replace_steps_for_direct_lim: bool = False
    conflict_detected: bool = False
    conflict_type: str | None = None
    back_price: float | None = None
    lay_price: float | None = None
    market_reference_price: float | None = None
    back_distance: float | None = None
    lay_distance: float | None = None
    selected_side: str | None = None
    rejected_side: str | None = None
    conflict_resolution_reason: str | None = None
    strategy_edge: float | None = None
    strategy_score: float | None = None
    conflict_group_key: str | None = None
    conflict_candidates_count: int | None = None
    winning_side: str | None = None
    losing_side: str | None = None
    winning_strategy_id: str | None = None
    losing_strategy_id: str | None = None
    winning_edge: float | None = None
    losing_edge: float | None = None
    winning_score: float | None = None
    losing_score: float | None = None
    winning_lim_price: float | None = None
    losing_lim_price: float | None = None
    back_systems: str | None = None
    lay_systems: str | None = None
    pre_back_lay_conflict: bool = False
    pre_conflict_resolution: str | None = None
    pre_conflict_chosen_side: str | None = None
    pre_conflict_rejected_side: str | None = None
    pre_conflict_reason: str | None = None
    pre_conflict_group_key: str | None = None
    pre_conflict_course_id: str | None = None
    pre_conflict_market_id: str | None = None
    pre_conflict_market_type: str | None = None
    pre_conflict_selection_id: str | None = None
    pre_conflict_runner_name: str | None = None
    pre_back_target_price: float | None = None
    pre_lay_target_price: float | None = None
    pre_current_best_lay: float | None = None
    pre_current_best_back: float | None = None
    pre_back_distance_ticks: float | None = None
    pre_lay_distance_ticks: float | None = None
    staking_formula: str | None = None
    staking_alpha: float | None = None
    staking_back_alpha: float | None = None
    staking_lay_alpha: float | None = None
    stake_raw_before_caps: float | None = None
    stake_after_caps: float | None = None
    lay_liability_after_sizing: float | None = None
    lay_liability_cap: float | None = None
    lay_liability_cap_hit: bool = False


@dataclass(frozen=True)
class GrussOrderResult:
    status: str
    reason: str
    output_path: Path


class GrussOrderProvider:
    """Dry-run-only Gruss order provider.

    TODO: future live Gruss support may translate validated OrderIntent objects
    into Gruss trigger cells. This class intentionally does not write to Excel.
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        order_provider: str = ORDER_PROVIDER_GRUSS_EXCEL_DRYRUN,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.order_provider = order_provider
        self.output_path = self.data_dir / "orders_gruss_dryrun.csv"

    def place_order(self, intent: OrderIntent) -> GrussOrderResult:
        status = "GRUSS_DRYRUN"
        reason = "dry_run_logged"
        validation_errors = validate_order_intent(intent)
        if self.order_provider != ORDER_PROVIDER_GRUSS_EXCEL_DRYRUN:
            validation_errors.append(f"unsupported_order_provider={self.order_provider}")
        if not intent.dry_run:
            validation_errors.append("dry_run_required")
        if validation_errors:
            status = "REJECTED_DRYRUN"
            reason = "; ".join(validation_errors)

        self._append(intent, status, reason)
        return GrussOrderResult(status=status, reason=reason, output_path=self.output_path)

    def _append(self, intent: OrderIntent, status: str, reason: str) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        write_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
        row = _csv_row(intent, status, reason)
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ORDERS_GRUSS_DRYRUN_HEADER)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def validate_order_intent(intent: OrderIntent, *, minimum_stake: float = 2.0) -> list[str]:
    errors: list[str] = []
    if str(intent.market_type or "").upper() not in {"WIN", "PLACE"}:
        errors.append("invalid_market_type")
    if str(intent.side or "").upper() not in {"BACK", "LAY"}:
        errors.append("invalid_side")
    if not str(intent.runner_name or "").strip():
        errors.append("missing_runner_name")
    try:
        stake = float(intent.stake) if intent.stake is not None else 0.0
    except (TypeError, ValueError):
        stake = 0.0
    if stake < minimum_stake:
        errors.append("stake_below_minimum")
    if str(intent.order_type or "").upper() == "LIMIT":
        try:
            price = float(intent.price) if intent.price is not None else 0.0
        except (TypeError, ValueError):
            price = 0.0
        if price < 1.01:
            errors.append("invalid_limit_price")
    return errors


def make_order_intent(
    *,
    provider: str,
    market_type: str,
    market_id: str,
    parent_id: str | None,
    runner_name: str,
    trap: int | None,
    side: str,
    order_type: str,
    price: float | None,
    stake: float | None,
    strategy_id: str,
    course_id: str | None,
    timestamp: str | None = None,
    dry_run: bool = True,
    stake_original: float | None = None,
    stake_forced: bool = False,
    force_test_bsp_place: bool = False,
    force_test_back_place_limit: bool = False,
    selected_reason: str | None = None,
    selected_runner: str | None = None,
    selected_trap: int | None = None,
    selected_place_odds: float | None = None,
    selected_place_back_odds: float | None = None,
    selected_place_lay_odds: float | None = None,
    price_used: float | None = None,
    selection_id: str | int | None = None,
    execution_phase: str = "POST",
    triggered_systems: str | None = None,
    triggered_prices: str | None = None,
    pre_ladder: bool = False,
    ladder_id: str | None = None,
    ladder_step: str | None = None,
    ladder_tracking_key: str | None = None,
    gruss_planned_trigger: str | None = None,
    matched_stake: float | None = None,
    signal_countdown_seconds: int | None = None,
    market_reference_price_at_signal: float | None = None,
    best_same_side_back_offer: float | None = None,
    best_same_side_lay_offer: float | None = None,
    ladder_plan_frozen: bool = False,
    ladder_plan_created_step: str | int | None = None,
    ladder_prices_frozen: str | None = None,
    current_ladder_price_from_frozen_plan: bool = False,
    computed_limit_price_raw: float | None = None,
    computed_limit_price_effective: float | None = None,
    min_price_floor_applied: bool = False,
    pre_value_target_price: float | None = None,
    ladder_planned_price: float | None = None,
    sent_price_before_value_clamp: float | None = None,
    sent_price_after_value_clamp: float | None = None,
    value_clamp_applied: bool = False,
    value_limit_breached: bool = False,
    value_limit_skip_reason: str | None = None,
    tick_rounding_direction: str | None = None,
    best_same_side_offer_at_creation: float | None = None,
    best_back_displayed: float | None = None,
    best_lay_displayed: float | None = None,
    start_price_source: str | None = None,
    ladder_direction: str | None = None,
    ladder_disabled_lim_not_in_ladder_direction: bool = False,
    direct_lim_order_planned: bool = False,
    direct_lim_order_written: bool = False,
    no_replace_steps_for_direct_lim: bool = False,
    conflict_detected: bool = False,
    conflict_type: str | None = None,
    back_price: float | None = None,
    lay_price: float | None = None,
    market_reference_price: float | None = None,
    back_distance: float | None = None,
    lay_distance: float | None = None,
    selected_side: str | None = None,
    rejected_side: str | None = None,
    conflict_resolution_reason: str | None = None,
    strategy_edge: float | None = None,
    strategy_score: float | None = None,
    conflict_group_key: str | None = None,
    conflict_candidates_count: int | None = None,
    winning_side: str | None = None,
    losing_side: str | None = None,
    winning_strategy_id: str | None = None,
    losing_strategy_id: str | None = None,
    winning_edge: float | None = None,
    losing_edge: float | None = None,
    winning_score: float | None = None,
    losing_score: float | None = None,
    winning_lim_price: float | None = None,
    losing_lim_price: float | None = None,
    back_systems: str | None = None,
    lay_systems: str | None = None,
    pre_back_lay_conflict: bool = False,
    pre_conflict_resolution: str | None = None,
    pre_conflict_chosen_side: str | None = None,
    pre_conflict_rejected_side: str | None = None,
    pre_conflict_reason: str | None = None,
    pre_conflict_group_key: str | None = None,
    pre_conflict_course_id: str | None = None,
    pre_conflict_market_id: str | None = None,
    pre_conflict_market_type: str | None = None,
    pre_conflict_selection_id: str | None = None,
    pre_conflict_runner_name: str | None = None,
    pre_back_target_price: float | None = None,
    pre_lay_target_price: float | None = None,
    pre_current_best_lay: float | None = None,
    pre_current_best_back: float | None = None,
    pre_back_distance_ticks: float | None = None,
    pre_lay_distance_ticks: float | None = None,
    staking_formula: str | None = None,
    staking_alpha: float | None = None,
    staking_back_alpha: float | None = None,
    staking_lay_alpha: float | None = None,
    stake_raw_before_caps: float | None = None,
    stake_after_caps: float | None = None,
    lay_liability_after_sizing: float | None = None,
    lay_liability_cap: float | None = None,
    lay_liability_cap_hit: bool = False,
) -> OrderIntent:
    return OrderIntent(
        provider=provider,
        market_type=str(market_type or "").upper(),
        market_id=str(market_id or ""),
        parent_id=parent_id,
        runner_name=runner_name,
        trap=trap,
        side=str(side or "").upper(),
        order_type=str(order_type or "").upper(),
        price=price,
        stake=stake,
        strategy_id=str(strategy_id or ""),
        course_id=course_id,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        dry_run=bool(dry_run),
        selection_id=selection_id,
        execution_phase=str(execution_phase or "POST").strip().upper(),
        triggered_systems=triggered_systems,
        triggered_prices=triggered_prices,
        stake_original=stake_original,
        stake_forced=bool(stake_forced),
        force_test_bsp_place=bool(force_test_bsp_place),
        force_test_back_place_limit=bool(force_test_back_place_limit),
        selected_reason=selected_reason,
        selected_runner=selected_runner,
        selected_trap=selected_trap,
        selected_place_odds=selected_place_odds,
        selected_place_back_odds=selected_place_back_odds,
        selected_place_lay_odds=selected_place_lay_odds,
        price_used=price_used,
        pre_ladder=bool(pre_ladder),
        ladder_id=ladder_id,
        ladder_step=ladder_step,
        ladder_tracking_key=ladder_tracking_key,
        gruss_planned_trigger=gruss_planned_trigger,
        matched_stake=matched_stake,
        signal_countdown_seconds=signal_countdown_seconds,
        market_reference_price_at_signal=market_reference_price_at_signal,
        best_same_side_back_offer=best_same_side_back_offer,
        best_same_side_lay_offer=best_same_side_lay_offer,
        ladder_plan_frozen=bool(ladder_plan_frozen),
        ladder_plan_created_step=ladder_plan_created_step,
        ladder_prices_frozen=ladder_prices_frozen,
        current_ladder_price_from_frozen_plan=bool(current_ladder_price_from_frozen_plan),
        computed_limit_price_raw=computed_limit_price_raw,
        computed_limit_price_effective=computed_limit_price_effective,
        min_price_floor_applied=bool(min_price_floor_applied),
        pre_value_target_price=pre_value_target_price,
        ladder_planned_price=ladder_planned_price,
        sent_price_before_value_clamp=sent_price_before_value_clamp,
        sent_price_after_value_clamp=sent_price_after_value_clamp,
        value_clamp_applied=bool(value_clamp_applied),
        value_limit_breached=bool(value_limit_breached),
        value_limit_skip_reason=value_limit_skip_reason,
        tick_rounding_direction=tick_rounding_direction,
        best_same_side_offer_at_creation=best_same_side_offer_at_creation,
        best_back_displayed=best_back_displayed,
        best_lay_displayed=best_lay_displayed,
        start_price_source=start_price_source,
        ladder_direction=ladder_direction,
        ladder_disabled_lim_not_in_ladder_direction=bool(ladder_disabled_lim_not_in_ladder_direction),
        direct_lim_order_planned=bool(direct_lim_order_planned),
        direct_lim_order_written=bool(direct_lim_order_written),
        no_replace_steps_for_direct_lim=bool(no_replace_steps_for_direct_lim),
        conflict_detected=bool(conflict_detected),
        conflict_type=conflict_type,
        back_price=back_price,
        lay_price=lay_price,
        market_reference_price=market_reference_price,
        back_distance=back_distance,
        lay_distance=lay_distance,
        selected_side=selected_side,
        rejected_side=rejected_side,
        conflict_resolution_reason=conflict_resolution_reason,
        strategy_edge=strategy_edge,
        strategy_score=strategy_score,
        conflict_group_key=conflict_group_key,
        conflict_candidates_count=conflict_candidates_count,
        winning_side=winning_side,
        losing_side=losing_side,
        winning_strategy_id=winning_strategy_id,
        losing_strategy_id=losing_strategy_id,
        winning_edge=winning_edge,
        losing_edge=losing_edge,
        winning_score=winning_score,
        losing_score=losing_score,
        winning_lim_price=winning_lim_price,
        losing_lim_price=losing_lim_price,
        back_systems=back_systems,
        lay_systems=lay_systems,
        pre_back_lay_conflict=bool(pre_back_lay_conflict),
        pre_conflict_resolution=pre_conflict_resolution,
        pre_conflict_chosen_side=pre_conflict_chosen_side,
        pre_conflict_rejected_side=pre_conflict_rejected_side,
        pre_conflict_reason=pre_conflict_reason,
        pre_conflict_group_key=pre_conflict_group_key,
        pre_conflict_course_id=pre_conflict_course_id,
        pre_conflict_market_id=pre_conflict_market_id,
        pre_conflict_market_type=pre_conflict_market_type,
        pre_conflict_selection_id=pre_conflict_selection_id,
        pre_conflict_runner_name=pre_conflict_runner_name,
        pre_back_target_price=pre_back_target_price,
        pre_lay_target_price=pre_lay_target_price,
        pre_current_best_lay=pre_current_best_lay,
        pre_current_best_back=pre_current_best_back,
        pre_back_distance_ticks=pre_back_distance_ticks,
        pre_lay_distance_ticks=pre_lay_distance_ticks,
        staking_formula=staking_formula,
        staking_alpha=staking_alpha,
        staking_back_alpha=staking_back_alpha,
        staking_lay_alpha=staking_lay_alpha,
        stake_raw_before_caps=stake_raw_before_caps,
        stake_after_caps=stake_after_caps,
        lay_liability_after_sizing=lay_liability_after_sizing,
        lay_liability_cap=lay_liability_cap,
        lay_liability_cap_hit=bool(lay_liability_cap_hit),
    )


def _csv_row(intent: OrderIntent, status: str, reason: str) -> dict[str, Any]:
    row = asdict(intent)
    row["timestamp"] = intent.timestamp
    row["status"] = status
    row["reason"] = reason
    return {field: row.get(field) for field in ORDERS_GRUSS_DRYRUN_HEADER}
