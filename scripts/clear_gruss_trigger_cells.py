from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH, GrussExcelBridge
from dogbot.gruss.gruss_real_orders import GrussTriggerLayout
from dogbot.gruss.gruss_trigger_clear import (
    append_trigger_clear_log,
    clear_runner_trigger_cells,
    find_nonempty_runner_trigger_cells,
)


LOG_PATH = ROOT / "data" / "gruss_trigger_clear_attempts.csv"
SHEETS = ("WIN", "PLACE")


def main() -> int:
    allow_clear = clear_enabled()
    mode = "real_clear" if allow_clear else "preview"
    print(f"Gruss trigger cleaner - {mode}")
    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")
    print("Cibles autorisees: cellules trigger Q des lignes runners WIN/PLACE uniquement.")
    print("R/S ne seront jamais modifies. Aucune commande trigger ne sera ecrite.")
    if not allow_clear:
        print("PREVIEW ONLY: definir DOGBOT_GRUSS_CLEAR_TRIGGERS=true pour effacer.")

    bridge = GrussExcelBridge(DEFAULT_WORKBOOK_PATH)
    try:
        ensure_open_visible_workbook_and_sheets(bridge)
        layout = GrussTriggerLayout.from_env()
        targets = find_nonempty_runner_trigger_cells(bridge, sheets=SHEETS, layout=layout)
        results = clear_runner_trigger_cells(
            bridge,
            targets,
            layout=layout,
            allow_clear=allow_clear,
        )
    except Exception as exc:
        print(f"ERREUR: {exc}")
        return 1

    for result in results:
        target = result.target
        print(
            f"{result.status}: {target.sheet}!{target.trigger_cell} "
            f"row={target.row} runner={target.runner!r} old_value={target.previous_value!r} "
            f"cleared={result.cleared} mode={result.mode}"
        )
    append_trigger_clear_log(LOG_PATH, results)
    print(f"cellules ciblees={len(results)}")
    print(f"log={LOG_PATH}")
    if allow_clear and any(not result.cleared for result in results):
        return 1
    return 0


def clear_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return str(values.get("DOGBOT_GRUSS_CLEAR_TRIGGERS", "")).strip().casefold() == "true"


def ensure_open_visible_workbook_and_sheets(bridge) -> None:
    bridge.connect_open_workbook()
    if not bridge.is_workbook_visible():
        raise RuntimeError("workbook Excel Gruss non visible")
    for sheet in SHEETS:
        if not bridge.has_sheet(sheet):
            raise RuntimeError(f"onglet {sheet} manquant")


if __name__ == "__main__":
    raise SystemExit(main())
