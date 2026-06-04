from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogbot.config import ORDER_PROVIDER_GRUSS_EXCEL_REAL
from dogbot.gruss.gruss_dryrun_engine import (
    describe_state,
    gruss_region_for_snapshots,
    race_key,
    read_gruss_dryrun_state,
)
from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH
from dogbot.gruss.gruss_feed import GrussFeed
from dogbot.gruss.gruss_orders import make_order_intent
from dogbot.gruss.gruss_real_orders import GrussExcelOrderProvider, GrussRealOrderContext


DATA_DIR = ROOT / "data"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # This command is permanently preview-only, regardless of the caller's env.
    os.environ["DOGBOT_GRUSS_REAL_PREVIEW"] = "true"

    print("Gruss real-order preview once")
    print(f"Workbook cible: {DEFAULT_WORKBOOK_PATH}")
    print("PREVIEW ONLY: aucune cellule Excel ne sera ecrite.")

    feed = GrussFeed(DEFAULT_WORKBOOK_PATH)
    try:
        state = read_gruss_dryrun_state(feed)
    except Exception as exc:
        print(f"REFUS: lecture Gruss impossible: {exc}")
        return 1

    print(describe_state(state))
    snapshot = state.win_snapshot if args.market_type == "WIN" else state.place_snapshot
    runner = next((item for item in snapshot.runners if item.trap == args.trap), None)
    if runner is None:
        print(f"REFUS: runner trap {args.trap} introuvable dans {args.market_type}")
        return 1

    price = args.price
    if args.order_type == "LIMIT" and price is None:
        price = runner.best_back if args.side == "BACK" else runner.best_lay

    countdown_seconds = state.win_snapshot.metadata.countdown_seconds
    if countdown_seconds is None:
        countdown_seconds = state.place_snapshot.metadata.countdown_seconds
    course_key = race_key(state.win_snapshot, state.place_snapshot)
    intent = make_order_intent(
        provider=ORDER_PROVIDER_GRUSS_EXCEL_REAL,
        market_type=args.market_type,
        market_id=str(snapshot.metadata.market_id or ""),
        parent_id=snapshot.metadata.parent_id,
        runner_name=runner.runner_name,
        trap=runner.trap,
        side=args.side,
        order_type=args.order_type,
        price=price,
        stake=args.stake,
        strategy_id=args.strategy_id,
        course_id=course_key,
        dry_run=False,
    )
    context = GrussRealOrderContext(
        validation_ok=not state.validation_warnings,
        tradable=state.tradable,
        region=gruss_region_for_snapshots(state.win_snapshot, state.place_snapshot),
        countdown_seconds=countdown_seconds,
        course=str(snapshot.metadata.event_path or course_key),
        win_market_id=state.win_snapshot.metadata.market_id,
        place_market_id=state.place_snapshot.metadata.market_id,
    )
    provider = GrussExcelOrderProvider(DATA_DIR, bridge=feed.bridge)
    result = provider.place_order(intent, context)

    print(f"status: {result.status}")
    print(f"reason: {result.reason}")
    print(f"sheet/row: {result.excel_sheet or '-'} / {result.excel_row or '-'}")
    if result.write_plan:
        print("write plan:")
        for address, value in result.write_plan:
            print(f"  {result.excel_sheet}!{address} = {value!r}")
    print(f"log: {result.output_path}")
    return 0 if result.status == "GRUSS_REAL_PREVIEW" else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview one Gruss Excel order without writing to Excel.")
    parser.add_argument("--market-type", choices=("WIN", "PLACE"), default="PLACE")
    parser.add_argument("--trap", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--side", choices=("BACK", "LAY"), default="BACK")
    parser.add_argument("--order-type", choices=("LIMIT", "SP_MOC"), default="LIMIT")
    parser.add_argument("--price", type=float, default=None)
    parser.add_argument("--stake", type=float, default=2.0)
    parser.add_argument("--strategy-id", default="MANUAL_GRUSS_REAL_PREVIEW")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
