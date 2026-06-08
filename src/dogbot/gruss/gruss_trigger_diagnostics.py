from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dogbot.gruss.gruss_real_orders import GrussTriggerLayout


@dataclass(frozen=True)
class GrussRunnerTriggerCells:
    row: int
    runner: str
    back_trigger_cell: str
    back_trigger_value: Any
    lay_trigger_cell: str
    lay_trigger_value: Any
    backsp_trigger_cell: str
    backsp_trigger_value: Any
    laysp_trigger_cell: str
    laysp_trigger_value: Any


def inspect_place_trigger_cells(
    bridge,
    layout: GrussTriggerLayout | None = None,
) -> list[GrussRunnerTriggerCells]:
    """Read configured PLACE trigger cells without performing any Excel write."""
    trigger_layout = layout or GrussTriggerLayout.from_env()
    runner_values = bridge.read_range("PLACE", "A5:A84")
    rows: list[GrussRunnerTriggerCells] = []
    for offset, value in enumerate(_flatten_single_column(runner_values)):
        if value in (None, ""):
            break
        row = 5 + offset
        address = trigger_layout.trigger_address(row)
        current_value = bridge.read_cell("PLACE", address)
        rows.append(
            GrussRunnerTriggerCells(
                row=row,
                runner=str(value),
                back_trigger_cell=address,
                back_trigger_value=current_value,
                lay_trigger_cell=address,
                lay_trigger_value=current_value,
                backsp_trigger_cell=address,
                backsp_trigger_value=current_value,
                laysp_trigger_cell=address,
                laysp_trigger_value=current_value,
            )
        )
    return rows


def _flatten_single_column(values: list[list[Any]]):
    if len(values) == 1 and values and len(values[0]) > 1:
        yield from values[0]
        return
    for row in values:
        yield row[0] if row else None
