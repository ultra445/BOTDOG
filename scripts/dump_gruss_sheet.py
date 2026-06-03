from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH, GrussExcelBridge


ROWS = 80
COLUMNS = 50
OUTPUT_DIR = ROOT / "data" / "gruss_dumps"


def main() -> int:
    bridge = GrussExcelBridge(DEFAULT_WORKBOOK_PATH)

    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")
    try:
        found_open = bridge.connect()
    except Exception as exc:
        print("Workbook trouve: NON")
        print(f"Erreur connexion Excel: {exc}")
        return 1

    print(f"Workbook trouve ouvert: {'OUI' if found_open else 'NON'}")
    if not found_open:
        print("Workbook ouvert depuis le chemin cible.")

    exit_code = 0
    for sheet_name, output_name in (("WIN", "win_dump.csv"), ("PLACE", "place_dump.csv")):
        exists = bridge.has_sheet(sheet_name)
        print(f"Onglet {sheet_name} trouve: {'OUI' if exists else 'NON'}")
        if not exists:
            exit_code = 1
            continue

        output_path = OUTPUT_DIR / output_name
        try:
            result = bridge.export_csv_diagnostic(
                sheet_name=sheet_name,
                output_path=output_path,
                rows=ROWS,
                columns=COLUMNS,
            )
        except Exception as exc:
            print(f"Erreur dump onglet {sheet_name}: {exc}")
            exit_code = 1
            continue

        print(
            f"Onglet {sheet_name}: {result.rows_dumped} lignes x "
            f"{result.columns_dumped} colonnes dumpees"
        )
        print(f"CSV genere: {result.output_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
