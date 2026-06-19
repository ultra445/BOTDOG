from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogbot.excel_strategy_loader import (
    DEFAULT_STRATEGY_EXCEL_MIGRATION_REPORT_PATH,
    DEFAULT_STRATEGY_EXCEL_PATH,
    export_strategy_slots_to_excel,
    load_excel_strategy_slots,
)
from dogbot.strategies import build_registry


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export current Python strategy registry to Excel.")
    parser.add_argument(
        "--output",
        default=str(ROOT / DEFAULT_STRATEGY_EXCEL_PATH),
        help="Excel workbook path to write.",
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / DEFAULT_STRATEGY_EXCEL_MIGRATION_REPORT_PATH),
        help="Migration report CSV path to write.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    previous_excel_enabled = os.environ.get("DOGBOT_STRATEGIES_EXCEL_ENABLED")
    os.environ["DOGBOT_STRATEGIES_EXCEL_ENABLED"] = "false"
    try:
        slots = build_registry()
    finally:
        if previous_excel_enabled is None:
            os.environ.pop("DOGBOT_STRATEGIES_EXCEL_ENABLED", None)
        else:
            os.environ["DOGBOT_STRATEGIES_EXCEL_ENABLED"] = previous_excel_enabled

    summary = export_strategy_slots_to_excel(
        slots,
        Path(args.output),
        migration_report_path=Path(args.report),
    )
    load_result = load_excel_strategy_slots(Path(args.output), write_report=False)
    print(f"strategy Excel exported: {Path(args.output)}")
    print(f"migration report: {Path(args.report)}")
    print(
        "summary: "
        f"python_strategies_detected={summary['python_strategies_detected']} "
        f"excel_strategies_exported={summary['excel_strategies_exported']} "
        f"active_strategies={summary['active_strategies']} "
        f"disabled_with_warning={summary['disabled_with_warning']} "
        f"loader_active={load_result.active_count} "
        f"loader_disabled={load_result.disabled_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
