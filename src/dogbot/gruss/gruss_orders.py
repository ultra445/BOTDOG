from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dogbot.config import ORDER_PROVIDER_GRUSS_EXCEL_DRYRUN


ORDERS_GRUSS_DRYRUN_HEADER = [
    "timestamp",
    "provider",
    "market_type",
    "market_id",
    "parent_id",
    "course_id",
    "runner_name",
    "trap",
    "side",
    "order_type",
    "price",
    "stake",
    "strategy_id",
    "status",
    "reason",
]


@dataclass(frozen=True)
class OrderIntent:
    provider: str
    market_type: str
    market_id: str
    parent_id: str | None
    runner_name: str
    trap: int | None
    side: str
    order_type: str
    price: float | None
    stake: float | None
    strategy_id: str
    course_id: str | None
    timestamp: str
    dry_run: bool


@dataclass(frozen=True)
class GrussOrderResult:
    status: str
    reason: str
    output_path: Path


class GrussOrderProvider:
    """Dry-run-only Gruss order provider.

    TODO: future live Gruss support may translate validated OrderIntent objects
    into Gruss trigger cells. This class intentionally does not write to Excel.
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        order_provider: str = ORDER_PROVIDER_GRUSS_EXCEL_DRYRUN,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.order_provider = order_provider
        self.output_path = self.data_dir / "orders_gruss_dryrun.csv"

    def place_order(self, intent: OrderIntent) -> GrussOrderResult:
        status = "GRUSS_DRYRUN"
        reason = "dry_run_logged"
        validation_errors = validate_order_intent(intent)
        if self.order_provider != ORDER_PROVIDER_GRUSS_EXCEL_DRYRUN:
            validation_errors.append(f"unsupported_order_provider={self.order_provider}")
        if not intent.dry_run:
            validation_errors.append("dry_run_required")
        if validation_errors:
            status = "REJECTED_DRYRUN"
            reason = "; ".join(validation_errors)

        self._append(intent, status, reason)
        return GrussOrderResult(status=status, reason=reason, output_path=self.output_path)

    def _append(self, intent: OrderIntent, status: str, reason: str) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        write_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
        row = _csv_row(intent, status, reason)
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ORDERS_GRUSS_DRYRUN_HEADER)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def validate_order_intent(intent: OrderIntent) -> list[str]:
    errors: list[str] = []
    if str(intent.market_type or "").upper() not in {"WIN", "PLACE"}:
        errors.append("invalid_market_type")
    if str(intent.side or "").upper() not in {"BACK", "LAY"}:
        errors.append("invalid_side")
    if not str(intent.runner_name or "").strip():
        errors.append("missing_runner_name")
    try:
        stake = float(intent.stake) if intent.stake is not None else 0.0
    except (TypeError, ValueError):
        stake = 0.0
    if stake < 2.0:
        errors.append("stake_below_minimum")
    if str(intent.order_type or "").upper() == "LIMIT":
        try:
            price = float(intent.price) if intent.price is not None else 0.0
        except (TypeError, ValueError):
            price = 0.0
        if price <= 1.01:
            errors.append("invalid_limit_price")
    return errors


def make_order_intent(
    *,
    provider: str,
    market_type: str,
    market_id: str,
    parent_id: str | None,
    runner_name: str,
    trap: int | None,
    side: str,
    order_type: str,
    price: float | None,
    stake: float | None,
    strategy_id: str,
    course_id: str | None,
    timestamp: str | None = None,
    dry_run: bool = True,
) -> OrderIntent:
    return OrderIntent(
        provider=provider,
        market_type=str(market_type or "").upper(),
        market_id=str(market_id or ""),
        parent_id=parent_id,
        runner_name=runner_name,
        trap=trap,
        side=str(side or "").upper(),
        order_type=str(order_type or "").upper(),
        price=price,
        stake=stake,
        strategy_id=str(strategy_id or ""),
        course_id=course_id,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        dry_run=bool(dry_run),
    )


def _csv_row(intent: OrderIntent, status: str, reason: str) -> dict[str, Any]:
    row = asdict(intent)
    row["timestamp"] = intent.timestamp
    row["status"] = status
    row["reason"] = reason
    return {field: row.get(field) for field in ORDERS_GRUSS_DRYRUN_HEADER}
