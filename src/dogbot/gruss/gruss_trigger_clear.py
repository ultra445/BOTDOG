from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dogbot.gruss.gruss_real_orders import GrussTriggerLayout


TRIGGER_CLEAR_LOG_HEADER = [
    "timestamp",
    "sheet",
    "row",
    "runner",
    "trigger_cell",
    "old_value",
    "cleared",
    "mode",
    "status",
]


@dataclass(frozen=True)
class GrussTriggerClearTarget:
    sheet: str
    row: int
    runner: str
    trigger_cell: str
    previous_value: Any


@dataclass(frozen=True)
class GrussTriggerClearResult:
    target: GrussTriggerClearTarget
    mode: str
    cleared: bool
    status: str


def find_nonempty_runner_trigger_cells(
    bridge,
    *,
    sheets: Iterable[str] = ("WIN", "PLACE"),
    layout: GrussTriggerLayout | None = None,
) -> list[GrussTriggerClearTarget]:
    trigger_layout = layout or GrussTriggerLayout.from_env()
    _require_q_trigger_layout(trigger_layout)
    targets: list[GrussTriggerClearTarget] = []
    for sheet_name in sheets:
        sheet = str(sheet_name).upper()
        if sheet not in {"WIN", "PLACE"}:
            raise PermissionError(f"Trigger clearing forbidden for sheet: {sheet}")
        runner_values = bridge.read_range(sheet, "A5:A84")
        for offset, value in enumerate(_flatten_single_column(runner_values)):
            if value in (None, ""):
                break
            row = 5 + offset
            address = trigger_layout.trigger_address(row)
            current_value = bridge.read_cell(sheet, address)
            if current_value in (None, ""):
                continue
            targets.append(
                GrussTriggerClearTarget(
                    sheet=sheet,
                    row=row,
                    runner=str(value),
                    trigger_cell=address,
                    previous_value=current_value,
                )
            )
    return targets


def clear_runner_trigger_cells(
    bridge,
    targets: Iterable[GrussTriggerClearTarget],
    *,
    layout: GrussTriggerLayout | None = None,
    allow_clear: bool = False,
) -> list[GrussTriggerClearResult]:
    trigger_layout = layout or GrussTriggerLayout.from_env()
    _require_q_trigger_layout(trigger_layout)
    prepared = tuple(targets)
    for target in prepared:
        if target.sheet not in {"WIN", "PLACE"}:
            raise PermissionError(f"Trigger clearing forbidden for sheet: {target.sheet}")
        if target.trigger_cell.upper() != trigger_layout.trigger_address(target.row):
            raise PermissionError(f"Unsafe trigger clear target: {target.sheet}!{target.trigger_cell}")
    if not allow_clear:
        return [
            GrussTriggerClearResult(
                target=target,
                mode="preview",
                cleared=False,
                status="WOULD_CLEAR",
            )
            for target in prepared
        ]

    results: list[GrussTriggerClearResult] = []
    for sheet in dict.fromkeys(target.sheet for target in prepared):
        sheet_targets = [target for target in prepared if target.sheet == sheet]
        addresses = [target.trigger_cell for target in sheet_targets]
        bridge.clear_trigger_cells(
            sheet,
            addresses,
            trigger_column=trigger_layout.trigger_column,
            allow_clear=True,
        )
        for target in sheet_targets:
            current_value = bridge.read_cell(sheet, target.trigger_cell)
            cleared = current_value in (None, "")
            status = "CLEARED" if cleared else "CLEAR_VERIFY_FAILED"
            results.append(
                GrussTriggerClearResult(
                    target=target,
                    mode="real_clear",
                    cleared=cleared,
                    status=status,
                )
            )
    return results


def append_trigger_clear_log(
    output_path: str | Path,
    results: Iterable[GrussTriggerClearResult],
) -> Path:
    path = Path(output_path)
    rows = tuple(results)
    if not rows:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIGGER_CLEAR_LOG_HEADER)
        if write_header:
            writer.writeheader()
        for result in rows:
            target = result.target
            writer.writerow(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "sheet": target.sheet,
                    "row": target.row,
                    "runner": target.runner,
                    "trigger_cell": target.trigger_cell,
                    "old_value": target.previous_value,
                    "cleared": str(result.cleared),
                    "mode": result.mode,
                    "status": result.status,
                }
            )
    return path


def _flatten_single_column(values: list[list[Any]]):
    if len(values) == 1 and values and len(values[0]) > 1:
        yield from values[0]
        return
    for row in values:
        yield row[0] if row else None


def _require_q_trigger_layout(layout: GrussTriggerLayout) -> None:
    if layout.trigger_column.upper() != "Q":
        raise PermissionError("Trigger cleaner is restricted to column Q")
