from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogbot.excel_strategy_loader import create_strategy_excel_template, default_strategy_excel_export_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a strategy Excel template without touching the active config by default.")
    parser.add_argument(
        "--output",
        default=None,
        help="Template path to write. Defaults to config/exports/dogbot_strategies_template_YYYYMMDD_HHMMSS.xlsx.",
    )
    parser.add_argument("--overwrite-config", action="store_true", help="Allow writing config/dogbot_strategies.xlsx after creating a timestamped backup.")
    parser.add_argument("--allow-template", action="store_true", help="Allow a 3-strategy template to be written to the active config path.")
    parser.add_argument("--min-strategies", type=int, default=1, help="Minimum strategy rows required for the template output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = Path(args.output) if args.output else ROOT / default_strategy_excel_export_path("dogbot_strategies_template")
    target = create_strategy_excel_template(
        output,
        overwrite_config=args.overwrite_config,
        allow_template=args.allow_template,
        min_strategies=args.min_strategies,
    )
    print(f"strategy Excel template created: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
