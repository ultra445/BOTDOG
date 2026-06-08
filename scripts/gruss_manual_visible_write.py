from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH, GrussExcelBridge


SHEET = "PLACE"
TRIGGER_CELL = "Q5"
VISIBLE_WRITE_CELLS = ("R5", "S5")
TEST_WRITE_PLAN = (("R5", 9.99), ("S5", 1))
TRIGGER_COMMANDS = frozenset({"BACK", "LAY", "BACKSP", "LAYSP"})
WAIT_SECONDS = 10


def main() -> int:
    print("Gruss manual visible write test - R5/S5 ONLY")
    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")
    print("PLACE!Q5 ne sera jamais ecrite.")
    print("Aucune commande BACK/LAY/BACKSP/LAYSP ne sera ecrite.")

    bridge = GrussExcelBridge(DEFAULT_WORKBOOK_PATH)
    try:
        ensure_open_visible_place_sheet(bridge)
        run_visible_write_test(bridge)
    except Exception as exc:
        print(f"ERREUR: {exc}")
        return 1
    return 0


def ensure_open_visible_place_sheet(bridge) -> None:
    bridge.connect_open_workbook()
    if not bridge.is_workbook_visible():
        raise RuntimeError("workbook Excel Gruss non visible")
    if not bridge.has_sheet(SHEET):
        raise RuntimeError("onglet PLACE manquant")


def run_visible_write_test(
    bridge,
    *,
    wait_seconds: float = WAIT_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bool:
    trigger_value = bridge.read_cell(SHEET, TRIGGER_CELL)
    if trigger_value not in (None, ""):
        raise RuntimeError(f"{SHEET}!{TRIGGER_CELL} non vide; aucune ecriture R/S autorisee")
    old_values = tuple((address, bridge.read_cell(SHEET, address)) for address in VISIBLE_WRITE_CELLS)
    _validate_safe_plan(TEST_WRITE_PLAN)
    _validate_safe_plan(old_values)

    for (address, old_value), (_, new_value) in zip(old_values, TEST_WRITE_PLAN):
        print(f"{SHEET}!{address} old_value={old_value!r} new_value={new_value!r}")

    write_attempted = False
    try:
        write_attempted = True
        _write_rs_only(bridge, TEST_WRITE_PLAN)
        print(f"Valeurs visibles pendant {wait_seconds:g} secondes.")
        sleep_fn(wait_seconds)
    finally:
        if write_attempted:
            try:
                _write_rs_only(bridge, old_values)
                restored = all(
                    bridge.read_cell(SHEET, address) == old_value
                    for address, old_value in old_values
                )
            except Exception as exc:
                print(f"restored=False reason={exc}")
                raise RuntimeError(f"restauration R5/S5 impossible: {exc}") from exc
            print(f"restored={restored}")
            if not restored:
                raise RuntimeError("verification de restauration R5/S5 echouee")
    return True


def _write_rs_only(bridge, cells: Iterable[tuple[str, object]]) -> list[str]:
    plan = tuple(cells)
    _validate_safe_plan(plan)
    return bridge.write_cells_without_trigger(
        SHEET,
        plan,
        trigger_address=TRIGGER_CELL,
        allow_write=True,
    )


def _validate_safe_plan(cells: Iterable[tuple[str, object]]) -> None:
    plan = tuple(cells)
    addresses = tuple(str(address).strip().upper() for address, _ in plan)
    if addresses != VISIBLE_WRITE_CELLS:
        raise PermissionError("visible write test is restricted exactly to PLACE!R5 and PLACE!S5")
    for _, value in plan:
        if str(value or "").strip().upper() in TRIGGER_COMMANDS:
            raise PermissionError("trigger command values are forbidden in the visible write test")


if __name__ == "__main__":
    raise SystemExit(main())
