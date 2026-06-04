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
    "stake",
    "price",
    "strategy_id",
    "status",
    "reason",
    "excel_sheet",
    "excel_row",
    "excel_cells_written",
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
        preview_only_guard: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_path = self.data_dir / "gruss_real_order_attempts.csv"
        self.bridge = bridge or GrussExcelBridge(DEFAULT_WORKBOOK_PATH)
        self.order_provider = os.getenv("DOGBOT_ORDER_PROVIDER", "").strip().lower()
        self.enabled = _env_bool("DOGBOT_GRUSS_ENABLE_REAL_ORDERS", False)
        self.preview = _env_bool("DOGBOT_GRUSS_REAL_PREVIEW", True)
        self.layout_confirmed = _env_bool("DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED", False)
        self.layout = layout or GrussTriggerLayout.from_env()
        self.processed_markets = processed_markets if processed_markets is not None else set()
        self.preview_only_guard = bool(preview_only_guard)

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

        self.processed_markets.add(_processed_key(intent, context))
        return self._finish(
            intent,
            context,
            "GRUSS_REAL_WRITTEN",
            "excel_trigger_written",
            excel_sheet=sheet_name,
            excel_row=runner_row,
            excel_cells_written=written,
            write_plan=plan,
        )

    def _preflight_errors(
        self,
        intent: OrderIntent,
        context: GrussRealOrderContext,
    ) -> list[str]:
        errors: list[str] = []
        if self.order_provider != ORDER_PROVIDER_GRUSS_EXCEL_REAL:
            errors.append(f"real_provider_not_selected={self.order_provider or 'unset'}")
        if self.preview_only_guard and self.enabled:
            errors.append("preview_only_refuses_real_orders_enabled")
        if self.preview_only_guard and not self.preview:
            errors.append("preview_only_requires_preview")
        if not self.enabled and not (self.preview_only_guard and self.preview):
            errors.append("real_orders_not_enabled: set DOGBOT_GRUSS_ENABLE_REAL_ORDERS=true")
        if not context.validation_ok:
            errors.append("win_place_validation_failed")
        if not context.tradable:
            errors.append("market_not_tradable")
        if str(context.region or "").upper() == "UNKNOWN":
            errors.append("unknown_region")
        if context.countdown_seconds is None:
            errors.append("countdown_seconds_unavailable")
        elif context.countdown_seconds > 3:
            errors.append("countdown_above_3_seconds")
        elif context.countdown_seconds < 0:
            errors.append("countdown_elapsed")
        if context.market_already_processed or _processed_key(intent, context) in self.processed_markets:
            errors.append("market_already_processed")

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
    ) -> GrussRealOrderResult:
        addresses = tuple(excel_cells_written)
        plan = tuple(write_plan)
        self._append_attempt(
            intent,
            context,
            status,
            reason,
            excel_sheet=excel_sheet,
            excel_row=excel_row,
            excel_cells_written=addresses,
        )
        return GrussRealOrderResult(
            status=status,
            reason=reason,
            output_path=self.output_path,
            excel_sheet=excel_sheet,
            excel_row=excel_row,
            excel_cells_written=addresses,
            write_plan=plan,
        )

    def _refresh_safety_flags(self) -> None:
        """Re-read every arming flag for each individual order attempt."""
        self.order_provider = os.getenv("DOGBOT_ORDER_PROVIDER", "").strip().lower()
        self.enabled = _env_bool("DOGBOT_GRUSS_ENABLE_REAL_ORDERS", False)
        self.preview = _env_bool("DOGBOT_GRUSS_REAL_PREVIEW", True)
        self.layout_confirmed = _env_bool("DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED", False)

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
    ) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        write_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "dry_run_or_real": "PREVIEW" if self.preview else "REAL",
            "enabled": str(bool(self.enabled)).lower(),
            "provider": self.order_provider,
            "course": context.course or intent.course_id or "",
            "market_id": intent.market_id,
            "market_type": intent.market_type,
            "runner": intent.runner_name,
            "trap": intent.trap,
            "side": intent.side,
            "stake": intent.stake,
            "price": intent.price,
            "strategy_id": intent.strategy_id,
            "status": status,
            "reason": reason,
            "excel_sheet": excel_sheet,
            "excel_row": excel_row,
            "excel_cells_written": ";".join(excel_cells_written),
        }
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=GRUSS_REAL_ATTEMPTS_HEADER)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


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
