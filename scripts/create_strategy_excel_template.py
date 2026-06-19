from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogbot.excel_strategy_loader import DEFAULT_STRATEGY_EXCEL_PATH, create_strategy_excel_template


def main() -> int:
    target = create_strategy_excel_template(ROOT / DEFAULT_STRATEGY_EXCEL_PATH)
    print(f"strategy Excel template created: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
