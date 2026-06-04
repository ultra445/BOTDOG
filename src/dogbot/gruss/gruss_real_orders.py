from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
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
    "side",
    "order_type",
    "intended_trigger",
    "stake",
    "price",
    "strategy_id",
    "status",
    "reason",
    "excel_sheet",
    "excel_row",
    "excel_cells_written",
    "cells_written",
    "trigger_written",
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
        self.real_max_stake = _real_max_stake(self.real_test_mode)

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
            trigger_address = f"{self.layout.trigger_column}{runner_row}"
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

        if not self.layout_confirmed:
            return self._finish(
                intent,
                context,
                "REJECTED_REAL",
                "trigger_layout_not_confirmed",
                excel_sheet=sheet_name,
                excel_row=runner_row,
                write_plan=plan,
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
            )

        real_key = _processed_key(intent, context)
        self.real_order_counts[real_key] = self.real_order_counts.get(real_key, 0) + 1
        return self._finish(
            intent,
            context,
            "GRUSS_REAL_WRITTEN",
            "excel_trigger_written",
            excel_sheet=sheet_name,
            excel_row=runner_row,
            excel_cells_written=written,
            write_plan=plan,
            trigger_written=True,
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
        guarded_unarmed_mode = (
            (self.preview_only_guard and self.preview)
            or (self.write_no_trigger_guard and self.write_no_trigger)
        )
        if not self.enabled and not guarded_unarmed_mode:
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
        if context.market_already_processed or _processed_key(intent, context) in self.processed_markets:
            errors.append("market_already_processed")
        if self._is_true_real_mode():
            real_key = _processed_key(intent, context)
            if (
                self.real_max_orders is not None
                and self.real_order_counts.get(real_key, 0) >= self.real_max_orders
            ):
                errors.append("max_orders_reached")
            if self.real_max_stake is not None and _float_or_infinity(intent.stake) > self.real_max_stake:
                errors.append("stake_above_real_test_limit")

        errors.extend(validate_order_intent(intent))
        if str(intent.order_type or "").upper() not in {"LIMIT", "SP_MOC"}:
            errors.append("invalid_order_type")
        if not _positive_finite(intent.stake):
            errors.append("invalid_stake")
        if not _valid_order_price(intent.price):
            errors.append("invalid_price")
        return _dedupe(errors)

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
            (f"{self.layout.odds_column}{row}", float(intent.price)),
        ]
        cells.append((f"{self.layout.stake_column}{row}", float(intent.stake)))
        # Trigger is deliberately written last.
        cells.append((f"{self.layout.trigger_column}{row}", trigger))
        return tuple(cells)

    def _trigger_for(self, intent: OrderIntent) -> str:
        key = (str(intent.side).upper(), str(intent.order_type).upper())
        triggers = {
            ("BACK", "LIMIT"): self.layout.back_limit_trigger,
            ("LAY", "LIMIT"): self.layout.lay_limit_trigger,
            ("BACK", "SP_MOC"): self.layout.back_sp_moc_trigger,
            ("LAY", "SP_MOC"): self.layout.lay_sp_moc_trigger,
        }
        return triggers[key]

    def _without_trigger(
        self,
        plan: Iterable[tuple[str, Any]],
        row: int,
    ) -> tuple[tuple[str, Any], ...]:
        trigger_address = f"{self.layout.trigger_column}{row}".upper()
        preparation = tuple(
            (address, value)
            for address, value in plan
            if str(address).upper() != trigger_address
        )
        if any(str(address).upper() == trigger_address for address, _ in preparation):
            raise RuntimeError("trigger_cell_present_in_no_trigger_plan")
        addresses = [str(address).upper() for address, _ in preparation]
        expected = {
            f"{self.layout.odds_column}{row}".upper(),
            f"{self.layout.stake_column}{row}".upper(),
        }
        if len(addresses) != 2 or len(set(addresses)) != 2 or set(addresses) != expected:
            raise RuntimeError("preparation_plan_must_contain_only_distinct_odds_and_stake_cells")
        return preparation

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
        self.real_max_stake = _real_max_stake(self.real_test_mode)

    def _is_true_real_mode(self) -> bool:
        return (
            self.order_provider == ORDER_PROVIDER_GRUSS_EXCEL_REAL
            and self.enabled
            and not self.preview_only_guard
            and not self.write_no_trigger_guard
            and not self.write_no_trigger
            and not self.preview
        )

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
            "side": intent.side,
            "order_type": intent.order_type,
            "intended_trigger": intended_trigger,
            "stake": intent.stake,
            "price": intent.price,
            "strategy_id": intent.strategy_id,
            "status": status,
            "reason": reason,
            "excel_sheet": excel_sheet,
            "excel_row": excel_row,
            "excel_cells_written": cells_written,
            "cells_written": cells_written,
            "trigger_written": str(bool(trigger_written)),
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
    return str(context.course or intent.course_id or intent.parent_id or intent.market_id)


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


def _real_max_stake(real_test_mode: bool) -> float | None:
    raw = os.getenv("DOGBOT_GRUSS_REAL_MAX_STAKE")
    if raw in (None, ""):
        return 1.0 if real_test_mode else None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


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
