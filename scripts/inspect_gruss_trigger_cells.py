from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH, GrussExcelBridge
from dogbot.gruss.gruss_real_orders import GrussTriggerLayout
from dogbot.gruss.gruss_trigger_diagnostics import inspect_place_trigger_cells


def main() -> int:
    print("Gruss PLACE trigger-cell inspector - READ ONLY")
    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")
    print("Aucune cellule Excel ne sera ecrite.")

    bridge = GrussExcelBridge(DEFAULT_WORKBOOK_PATH)
    try:
        bridge.connect_open_workbook()
        if not bridge.is_workbook_visible():
            raise RuntimeError("workbook Excel non visible")
        if not bridge.has_sheet("PLACE"):
            raise RuntimeError("onglet PLACE manquant")
    except Exception as exc:
        print(f"ERREUR: {exc}")
        return 1

    layout = GrussTriggerLayout.from_env()
    print(
        "Preparation configuree: "
        f"Odds/SP={layout.odds_column}{{row}} Stake={layout.stake_column}{{row}}"
    )
    print(
        "Triggers configures: "
        f"BACK={layout.trigger_column}{{row}}:{layout.back_limit_trigger} | "
        f"LAY={layout.trigger_column}{{row}}:{layout.lay_limit_trigger} | "
        f"BACKSP={layout.trigger_column}{{row}}:{layout.back_sp_moc_trigger} | "
        f"LAYSP={layout.trigger_column}{{row}}:{layout.lay_sp_moc_trigger}"
    )

    try:
        rows = inspect_place_trigger_cells(bridge, layout)
    except Exception as exc:
        print(f"ERREUR: lecture trigger impossible: {exc}")
        return 1

    for item in rows:
        print(
            f"row={item.row} runner={item.runner!r} | "
            f"BACK {item.back_trigger_cell}={item.back_trigger_value!r} | "
            f"LAY {item.lay_trigger_cell}={item.lay_trigger_value!r} | "
            f"BACKSP {item.backsp_trigger_cell}={item.backsp_trigger_value!r} | "
            f"LAYSP {item.laysp_trigger_cell}={item.laysp_trigger_value!r}"
        )
    print(f"runners inspectes={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
