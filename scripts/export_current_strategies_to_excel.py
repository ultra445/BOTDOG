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
    default_strategy_excel_export_path,
    export_strategy_slots_to_excel,
    load_excel_strategy_slots,
)
from dogbot.strategies import build_registry


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export current Python strategy registry to Excel.")
    parser.add_argument(
        "--output",
        default=None,
        help="Excel workbook path to write. Defaults to config/exports/dogbot_strategies_export_YYYYMMDD_HHMMSS.xlsx.",
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / DEFAULT_STRATEGY_EXCEL_MIGRATION_REPORT_PATH),
        help="Migration report CSV path to write.",
    )
    parser.add_argument("--overwrite-config", action="store_true", help="Allow writing config/dogbot_strategies.xlsx after creating a timestamped backup.")
    parser.add_argument("--allow-template", action="store_true", help="Allow writing a template-sized workbook when combined with a low --min-strategies value.")
    parser.add_argument("--min-strategies", type=int, default=20, help="Minimum strategy rows required before writing the active config.")
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

    output = Path(args.output) if args.output else ROOT / default_strategy_excel_export_path()
    summary = export_strategy_slots_to_excel(
        slots,
        output,
        migration_report_path=Path(args.report),
        overwrite_config=args.overwrite_config,
        allow_template=args.allow_template,
        min_strategies=args.min_strategies,
    )
    load_result = load_excel_strategy_slots(output, write_report=False)
    print(f"strategy Excel exported: {output}")
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
