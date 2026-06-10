from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dogbot.config import ORDER_PROVIDER_GRUSS_EXCEL_REAL
from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH, GrussExcelBridge
from dogbot.gruss.gruss_mapper import extract_trap, normalize_runner_name
from dogbot.gruss.gruss_orders import OrderIntent, validate_order_intent


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
    "triggered_systems",
    "triggered_prices",
    "intended_trigger",
    "trigger",
    "stake",
    "stake_original",
    "stake_used",
    "stake_forced",
    "stake_capped",
    "stake_cap_value",
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
    "strategy_id",
    "status",
    "reason",
    "excel_sheet",
    "excel_row",
    "excel_cells_written",
    "cells_written",
    "trigger_cell_address",
    "trigger_cell_current_value",
    "trigger_cell_expected_empty",
    "trigger_mapping_name",
    "trigger_written",
    "trigger_value_written",
    "trigger_clear_attempted",
    "trigger_cleared",
    "trigger_clear_reason",
    "trigger_cell_value_before_clear",
    "trigger_clear_delay_ms",
    "post_write_odds_cell_address",
    "post_write_odds_value",
    "post_write_stake_cell_address",
    "post_write_stake_value",
    "post_write_trigger_cell_address",
    "post_write_trigger_value",
    "post_write_verified",
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


@dataclass(frozen=True)
class GrussTriggerLayout:
    """Standard Gruss triggered-betting columns for a sheet starting at A1."""

    trigger_column: str = "Q"
    odds_column: str = "R"
    stake_column: str = "S"
    back_limit_trigger: str = "BACK"
    lay_limit_trigger: str = "LAY"
    back_sp_moc_trigger: str = "BACKSP"
    lay_sp_moc_trigger: str = "LAYSP"

    @classmethod
    def from_env(cls) -> GrussTriggerLayout:
        return cls(
            trigger_column=_column_env("DOGBOT_GRUSS_TRIGGER_COLUMN", "Q"),
            odds_column=_column_env("DOGBOT_GRUSS_ODDS_COLUMN", "R"),
            stake_column=_column_env("DOGBOT_GRUSS_STAKE_COLUMN", "S"),
            back_limit_trigger=os.getenv("DOGBOT_GRUSS_BACK_LIMIT_TRIGGER", "BACK").strip().upper(),
            lay_limit_trigger=os.getenv("DOGBOT_GRUSS_LAY_LIMIT_TRIGGER", "LAY").strip().upper(),
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
    stake_capped: bool = False
    stake_cap_value: float | None = None
    execution_phase: str = "POST"
    processed_key: str = ""
    trigger_cell_address: str = ""
    trigger_cell_current_value: Any = None
    trigger_cell_expected_empty: bool | None = None
    trigger_mapping_name: str = ""
    trigger_value_written: str = ""
    trigger_clear_attempted: bool = False
    trigger_cleared: bool = False
    trigger_clear_reason: str = ""
    trigger_cell_value_before_clear: Any = None
    trigger_clear_delay_ms: int = 0
    post_write_odds_cell_address: str = ""
    post_write_odds_value: Any = None
    post_write_stake_cell_address: str = ""
    post_write_stake_value: Any = None
    post_write_trigger_cell_address: str = ""
    post_write_trigger_value: Any = None
    post_write_verified: bool | None = None
    hold_trigger_for_visual_test: bool = False


@dataclass(frozen=True)
class _TriggerClearOutcome:
    attempted: bool = False
    cleared: bool = False
    reason: str = ""
    value_before_clear: Any = None
    delay_ms: int = 0


@dataclass(frozen=True)
class _PostWriteVerification:
    odds_cell_address: str
    odds_value: Any
    stake_cell_address: str
    stake_value: Any
    trigger_cell_address: str
    trigger_value: Any
    verified: bool


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
        self.write_no_trigger = _env_bool("DOGBOT_GRUSS_WRITE_NO_TRIGGER", False)
        self.real_test_mode = _env_bool("DOGBOT_GRUSS_REAL_TEST_MODE", False)
        self.real_max_orders = _real_max_orders(self.real_test_mode)
        self.real_max_orders_by_phase = _real_max_orders_by_phase(self.real_test_mode, self.real_max_orders)
        self.real_max_stake = _real_max_stake(self.real_test_mode)
        self.trigger_clear_delay_ms = _trigger_clear_delay_ms()
        self.hold_trigger_for_visual_test = _env_bool(
            "DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST",
            False,
        )

    def place_order(
        self,
        intent: OrderIntent,
        context: GrussRealOrderContext,
    ) -> GrussRealOrderResult:
        self._refresh_safety_flags()
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
            return self._finish(intent, context, "REJECTED_REAL", f"excel_unavailable: {exc}")

        if runner_row is None:
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                "runner_row_not_found",
                excel_sheet=sheet_name,
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
                trigger_value = self.bridge.read_cell(sheet_name, trigger_address)
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
            trigger_value = self.bridge.read_cell(sheet_name, trigger_address)
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

        try:
            written = tuple(self.bridge.write_cells(sheet_name, plan, allow_write=True))
        except Exception as exc:
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                f"excel_write_failed: {exc}",
                excel_sheet=sheet_name,
                excel_row=runner_row,
                write_plan=plan,
                trigger_cell_address=trigger_address,
                trigger_cell_current_value=trigger_value,
                trigger_cell_expected_empty=True,
                trigger_mapping_name=trigger_mapping_name,
                stake_capped=stake_capped,
                stake_cap_value=stake_cap_value,
            )

        verification = self._verify_real_write(sheet_name, runner_row, plan)
        trigger_written = _values_match(trigger_mapping_name, verification.trigger_value)
        clear_outcome = _TriggerClearOutcome()
        if trigger_written:
            clear_outcome = self._clear_written_trigger(
                sheet_name,
                trigger_address,
                trigger_mapping_name,
                hold_for_visual_test=self.hold_trigger_for_visual_test,
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
            "post_write_verification": verification,
            "hold_trigger_for_visual_test": self.hold_trigger_for_visual_test,
            "stake_capped": stake_capped,
            "stake_cap_value": stake_cap_value,
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
        return self._finish(
            intent,
            context,
            "GRUSS_REAL_WRITTEN",
            "excel_trigger_written",
            **finish_kwargs,
        )

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
        elif context.countdown_seconds > (2 if self.write_no_trigger_guard else 3):
            errors.append(
                "countdown_above_2_seconds"
                if self.write_no_trigger_guard
                else "countdown_above_3_seconds"
            )
        elif context.countdown_seconds < 0:
            errors.append("countdown_elapsed")
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
        if self._is_true_real_mode():
            max_orders = self._max_orders_for_intent(intent)
            max_key = _max_orders_key(intent, context)
            if max_orders is not None and self.real_order_counts.get(max_key, 0) >= max_orders:
                errors.append("max_orders_reached")

        minimum_stake = 0.01 if self._is_true_real_mode() and self.real_test_mode else 2.0
        errors.extend(validate_order_intent(intent, minimum_stake=minimum_stake))
        if str(intent.order_type or "").upper() not in {"LIMIT", "SP_MOC"}:
            errors.append("invalid_order_type")
        if not _positive_finite(intent.stake):
            errors.append("invalid_stake")
        if not _valid_order_price(intent.price):
            errors.append("invalid_price")
        return _dedupe(errors)

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
        values = self.bridge.read_range(sheet_name, "A5:A84")
        candidates = list(_flatten_single_column(values))
        normalized_target = normalize_runner_name(intent.runner_name)
        for offset, value in enumerate(candidates):
            if value in (None, ""):
                break
            trap_matches = intent.trap is None or extract_trap(value) == intent.trap
            name_matches = not normalized_target or normalize_runner_name(value) == normalized_target
            if trap_matches and name_matches:
                return 5 + offset
        return None

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
            current_id = _normalise_identifier(self.bridge.read_cell(sheet_name, "N3"))
            if not current_id:
                return f"current_market_id_unavailable={sheet_name}"
            if current_id != _normalise_identifier(expected_id):
                return f"current_market_id_mismatch={sheet_name}:{current_id}"
            suspend_status = str(self.bridge.read_cell(sheet_name, "F2") or "").strip().casefold()
            if "suspended" in suspend_status:
                return f"current_market_suspended={sheet_name}"
        return None

    def _build_write_plan(self, intent: OrderIntent, row: int) -> tuple[tuple[str, Any], ...]:
        trigger = self._trigger_for(intent)
        cells: list[tuple[str, Any]] = [
            (self.layout.odds_address(row), float(intent.price)),
        ]
        cells.append((self.layout.stake_address(row), float(intent.stake)))
        # Trigger is deliberately written last.
        cells.append((self.layout.trigger_address(row), trigger))
        return tuple(cells)

    def _cap_real_stake(self, intent: OrderIntent) -> tuple[OrderIntent, bool, float | None]:
        cap_value = self.real_max_stake
        if cap_value is None:
            return intent, False, None
        try:
            cap = float(cap_value)
            stake_to_write = float(intent.stake)
        except (TypeError, ValueError):
            return intent, False, cap_value
        if not math.isfinite(cap) or cap <= 0:
            return intent, False, cap
        if stake_to_write <= cap:
            return intent, False, cap
        stake_original = intent.stake_original if intent.stake_original is not None else stake_to_write
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
    ) -> _TriggerClearOutcome:
        delay_ms = self.trigger_clear_delay_ms
        try:
            if hold_for_visual_test:
                print(
                    "holding trigger for visual test "
                    f"{sheet_name}!{trigger_address}={trigger_value_written} "
                    f"delay_ms={delay_ms}"
                )
            if delay_ms:
                time.sleep(delay_ms / 1000)
            current_value = self.bridge.read_cell(sheet_name, trigger_address)
        except Exception as exc:
            return _TriggerClearOutcome(
                attempted=True,
                reason=f"trigger_clear_read_failed: {exc}",
                delay_ms=delay_ms,
            )

        if trigger_value_written not in {"BACK", "LAY", "BACKSP", "LAYSP"}:
            return _TriggerClearOutcome(
                attempted=True,
                reason="trigger_clear_skipped_unrecognized_trigger",
                value_before_clear=current_value,
                delay_ms=delay_ms,
            )
        if current_value != trigger_value_written:
            return _TriggerClearOutcome(
                attempted=True,
                reason="trigger_clear_skipped_value_changed",
                value_before_clear=current_value,
                delay_ms=delay_ms,
            )

        try:
            self.bridge.clear_trigger_cells(
                sheet_name,
                [trigger_address],
                trigger_column=self.layout.trigger_column,
                allow_clear=True,
            )
            value_after_clear = self.bridge.read_cell(sheet_name, trigger_address)
        except Exception as exc:
            return _TriggerClearOutcome(
                attempted=True,
                reason=f"trigger_clear_failed: {exc}",
                value_before_clear=current_value,
                delay_ms=delay_ms,
            )
        if value_after_clear not in (None, ""):
            return _TriggerClearOutcome(
                attempted=True,
                reason="trigger_clear_verify_failed",
                value_before_clear=current_value,
                delay_ms=delay_ms,
            )
        return _TriggerClearOutcome(
            attempted=True,
            cleared=True,
            reason="trigger_cleared",
            value_before_clear=current_value,
            delay_ms=delay_ms,
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
            odds_value = self.bridge.read_cell(sheet_name, odds_address)
            stake_value = self.bridge.read_cell(sheet_name, stake_address)
            trigger_value = self.bridge.read_cell(sheet_name, trigger_address)
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
        trigger_clear_attempted: bool = False,
        trigger_cleared: bool = False,
        trigger_clear_reason: str = "",
        trigger_cell_value_before_clear: Any = None,
        trigger_clear_delay_ms: int = 0,
        post_write_verification: _PostWriteVerification | None = None,
        hold_trigger_for_visual_test: bool = False,
        stake_capped: bool = False,
        stake_cap_value: float | None = None,
    ) -> GrussRealOrderResult:
        addresses = tuple(excel_cells_written)
        plan = tuple(write_plan)
        try:
            intended_trigger = self._trigger_for(intent)
        except Exception:
            intended_trigger = ""
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
            trigger_clear_attempted=trigger_clear_attempted,
            trigger_cleared=trigger_cleared,
            trigger_clear_reason=trigger_clear_reason,
            trigger_cell_value_before_clear=trigger_cell_value_before_clear,
            trigger_clear_delay_ms=trigger_clear_delay_ms,
            post_write_verification=post_write_verification,
            hold_trigger_for_visual_test=hold_trigger_for_visual_test,
            stake_capped=stake_capped,
            stake_cap_value=stake_cap_value,
        )
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
            stake_capped=stake_capped,
            stake_cap_value=stake_cap_value,
            execution_phase=_execution_phase(intent),
            processed_key=_processed_key(intent, context),
            trigger_cell_address=trigger_cell_address,
            trigger_cell_current_value=trigger_cell_current_value,
            trigger_cell_expected_empty=trigger_cell_expected_empty,
            trigger_mapping_name=trigger_mapping_name or intended_trigger,
            trigger_value_written=trigger_value_written,
            trigger_clear_attempted=trigger_clear_attempted,
            trigger_cleared=trigger_cleared,
            trigger_clear_reason=trigger_clear_reason,
            trigger_cell_value_before_clear=trigger_cell_value_before_clear,
            trigger_clear_delay_ms=trigger_clear_delay_ms,
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
        trigger_clear_attempted: bool,
        trigger_cleared: bool,
        trigger_clear_reason: str,
        trigger_cell_value_before_clear: Any,
        trigger_clear_delay_ms: int,
        post_write_verification: _PostWriteVerification | None,
        hold_trigger_for_visual_test: bool,
        stake_capped: bool,
        stake_cap_value: float | None,
    ) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_attempt_log_header()
        write_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
        cells_written = ";".join(excel_cells_written)
        mode = "WRITE_NO_TRIGGER" if self.write_no_trigger_guard else ("PREVIEW" if self.preview else "REAL")
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
            "triggered_systems": intent.triggered_systems or intent.strategy_id,
            "triggered_prices": intent.triggered_prices or "",
            "intended_trigger": intended_trigger,
            "trigger": intended_trigger,
            "stake": intent.stake,
            "stake_original": intent.stake_original if intent.stake_original is not None else intent.stake,
            "stake_used": intent.stake,
            "stake_forced": str(bool(intent.stake_forced)),
            "stake_capped": str(bool(stake_capped)),
            "stake_cap_value": "" if stake_cap_value is None else stake_cap_value,
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
            "strategy_id": intent.strategy_id,
            "status": status,
            "reason": reason,
            "excel_sheet": excel_sheet,
            "excel_row": excel_row,
            "excel_cells_written": cells_written,
            "cells_written": cells_written,
            "trigger_cell_address": trigger_cell_address,
            "trigger_cell_current_value": trigger_cell_current_value,
            "trigger_cell_expected_empty": (
                "" if trigger_cell_expected_empty is None else str(trigger_cell_expected_empty)
            ),
            "trigger_mapping_name": trigger_mapping_name or intended_trigger,
            "trigger_written": str(bool(trigger_written)),
            "trigger_value_written": trigger_value_written,
            "trigger_clear_attempted": str(bool(trigger_clear_attempted)),
            "trigger_cleared": str(bool(trigger_cleared)),
            "trigger_clear_reason": trigger_clear_reason,
            "trigger_cell_value_before_clear": trigger_cell_value_before_clear,
            "trigger_clear_delay_ms": trigger_clear_delay_ms,
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


def _processed_key(intent: OrderIntent, context: GrussRealOrderContext) -> str:
    race_id = str(context.course or intent.course_id or intent.parent_id or "").strip()
    selection_id = intent.selection_id if intent.selection_id is not None else intent.trap
    return "|".join(
        str(part)
        for part in (
            race_id,
            intent.market_id,
            selection_id,
            intent.side,
            intent.market_type,
            _execution_phase(intent),
        )
    )


def _max_orders_key(intent: OrderIntent, context: GrussRealOrderContext) -> str:
    race_id = str(context.course or intent.course_id or intent.parent_id or intent.market_id).strip()
    return "|".join((race_id, _execution_phase(intent)))


def _execution_phase(intent: OrderIntent) -> str:
    phase = str(getattr(intent, "execution_phase", "") or "POST").strip().upper()
    return phase if phase in {"PRE", "POST"} else "POST"


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
    return math.isfinite(number) and number > 1.01


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


def _trigger_clear_delay_ms() -> int:
    raw = os.getenv("DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS")
    if raw in (None, ""):
        return 1500
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 1500


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
