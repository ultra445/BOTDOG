from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogbot.excel_strategy_loader import (  # noqa: E402
    DEFAULT_STRATEGY_EXCEL_PATH,
    _slot_market_type,
    _slot_order_fields,
    _slot_side,
    load_excel_strategy_slots,
)
from dogbot.strategies import EXECUTION_PHASE_POST, EXECUTION_PHASE_PRE, RunnerCtx, build_registry  # noqa: E402


REPORT_PATH = ROOT / "data" / "strategy_excel_equivalence_report.csv"
REPORT_COLUMNS = [
    "context_id",
    "python_strategy_id",
    "excel_strategy_id",
    "status",
    "mismatch_type",
    "python_value",
    "excel_value",
    "message",
]


@dataclass
class CompareSummary:
    contexts_tested: int = 0
    python_fires: int = 0
    excel_fires: int = 0
    matches: int = 0
    mismatches: int = 0
    skipped: int = 0
    errors: int = 0


def main() -> int:
    python_slots = _load_python_slots()
    excel_slots = load_excel_strategy_slots(ROOT / DEFAULT_STRATEGY_EXCEL_PATH, write_report=False).slots
    rows: list[dict[str, Any]] = []
    summary = CompareSummary()
    python_by_id = {str(slot.tag): slot for slot in python_slots}
    excel_by_id = {str(slot.tag): slot for slot in excel_slots}

    manual_review_ids = sorted(set(python_by_id) - set(excel_by_id))
    for strategy_id in manual_review_ids:
        rows.append(
            _row(
                context_id="manual_review",
                python_strategy_id=strategy_id,
                excel_strategy_id="",
                status="SKIPPED_MANUAL_REVIEW",
                mismatch_type="excel_strategy_disabled_or_missing",
                python_value="present",
                excel_value="missing",
                message="Strategy is not active in Excel.",
            )
        )
        summary.skipped += 1

    metadata_mismatches = _metadata_mismatches(python_by_id, excel_by_id)
    rows.extend(metadata_mismatches)
    summary.mismatches += len(metadata_mismatches)

    for context_id, ctx in _contexts():
        summary.contexts_tested += 1
        try:
            python_fired = _fired_ids(python_slots, ctx)
            excel_fired = _fired_ids(excel_slots, ctx)
        except Exception as exc:
            rows.append(
                _row(
                    context_id=context_id,
                    python_strategy_id="",
                    excel_strategy_id="",
                    status="ERROR",
                    mismatch_type="condition_exception",
                    python_value="",
                    excel_value="",
                    message=repr(exc),
                )
            )
            summary.errors += 1
            continue
        summary.python_fires += len(python_fired)
        summary.excel_fires += len(excel_fired)
        for strategy_id in sorted(python_fired & excel_fired):
            rows.append(
                _row(
                    context_id=context_id,
                    python_strategy_id=strategy_id,
                    excel_strategy_id=strategy_id,
                    status="MATCH",
                    mismatch_type="",
                    python_value="fire",
                    excel_value="fire",
                    message="",
                )
            )
            summary.matches += 1
        for strategy_id in sorted(python_fired - excel_fired):
            if strategy_id in manual_review_ids:
                rows.append(
                    _row(
                        context_id=context_id,
                        python_strategy_id=strategy_id,
                        excel_strategy_id="",
                        status="SKIPPED_MANUAL_REVIEW",
                        mismatch_type="excel_strategy_disabled_or_missing",
                        python_value="fire",
                        excel_value="not_loaded",
                        message="Strategy is not active in Excel.",
                    )
                )
                summary.skipped += 1
                continue
            rows.append(
                _row(
                    context_id=context_id,
                    python_strategy_id=strategy_id,
                    excel_strategy_id=strategy_id,
                    status="MISMATCH",
                    mismatch_type="python_only_fire",
                    python_value="fire",
                    excel_value="no_fire",
                    message="Python fired but Excel did not.",
                )
            )
            summary.mismatches += 1
        for strategy_id in sorted(excel_fired - python_fired):
            rows.append(
                _row(
                    context_id=context_id,
                    python_strategy_id=strategy_id,
                    excel_strategy_id=strategy_id,
                    status="MISMATCH",
                    mismatch_type="excel_only_fire",
                    python_value="no_fire",
                    excel_value="fire",
                    message="Excel fired but Python did not.",
                )
            )
            summary.mismatches += 1

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(
        "strategy Excel equivalence: "
        f"contexts_tested={summary.contexts_tested} "
        f"python_fires={summary.python_fires} "
        f"excel_fires={summary.excel_fires} "
        f"matches={summary.matches} "
        f"mismatches={summary.mismatches} "
        f"skipped={summary.skipped} "
        f"errors={summary.errors} "
        f"report={REPORT_PATH}"
    )
    return 0 if summary.mismatches == 0 and summary.errors == 0 else 1


def _load_python_slots() -> list[Any]:
    old = os.environ.get("DOGBOT_STRATEGIES_EXCEL_ENABLED")
    os.environ["DOGBOT_STRATEGIES_EXCEL_ENABLED"] = "false"
    try:
        return build_registry()
    finally:
        if old is None:
            os.environ.pop("DOGBOT_STRATEGIES_EXCEL_ENABLED", None)
        else:
            os.environ["DOGBOT_STRATEGIES_EXCEL_ENABLED"] = old


def _contexts() -> list[tuple[str, RunnerCtx]]:
    contexts: list[tuple[str, RunnerCtx]] = []
    place_prices = [1.2, 2.0, 3.5, 5.0, 8.0, 16.0, 30.0]
    win_prices = [3.0, 5.0, 8.0, 13.0, 25.0]
    ev_values = [-0.5, -0.2, 0.0, 0.1, 0.25]
    mom_values = [-0.2, 0.0, 0.2]
    milestones = [2, 45]
    index = 0
    for market_type in ["WIN", "PLACE"]:
        for region in ["UK", "ROW"]:
            for trap in range(1, 9):
                for place_price in place_prices:
                    for win_price in win_prices:
                        for ev_place in ev_values:
                            for mom45 in mom_values:
                                for milestone in milestones:
                                    index += 1
                                    price = win_price if market_type == "WIN" else place_price
                                    phase = EXECUTION_PHASE_POST if milestone == 2 else EXECUTION_PHASE_PRE
                                    contexts.append(
                                        (
                                            f"ctx-{index}",
                                            RunnerCtx(
                                                market_id=f"{market_type.lower()}-market",
                                                market_type=market_type,
                                                selection_id=trap,
                                                course_id="synthetic-course",
                                                ltp=price,
                                                milestone=milestone,
                                                secs_to_off=float(milestone),
                                                trap=trap,
                                                region=region,
                                                winbet=win_price,
                                                base_win=win_price,
                                                bsp_place=None,
                                                place_theo=place_price,
                                                ev_place=ev_place,
                                                mom45=mom45,
                                                bb=max(1.01, price - 0.1),
                                                bl=price + 0.1,
                                                execution_phase=phase,
                                            ),
                                        )
                                    )
    return contexts


def _fired_ids(slots: list[Any], ctx: RunnerCtx) -> set[str]:
    fired: set[str] = set()
    for slot in slots:
        if str(getattr(slot, "execution_phase", "")).upper() != str(ctx.execution_phase).upper():
            continue
        if _slot_market_type(slot) != str(ctx.market_type).upper():
            continue
        if getattr(slot, "requires_mom45", False) and ctx.mom45 is None:
            continue
        if bool(slot.condition(ctx)):
            fired.add(str(slot.tag))
    return fired


def _metadata_mismatches(python_by_id: dict[str, Any], excel_by_id: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy_id in sorted(set(python_by_id) & set(excel_by_id)):
        py = _metadata(python_by_id[strategy_id])
        xl = _metadata(excel_by_id[strategy_id])
        for key, py_value in py.items():
            xl_value = xl.get(key)
            if str(py_value) == str(xl_value):
                continue
            rows.append(
                _row(
                    context_id="metadata",
                    python_strategy_id=strategy_id,
                    excel_strategy_id=strategy_id,
                    status="MISMATCH",
                    mismatch_type=f"metadata:{key}",
                    python_value=py_value,
                    excel_value=xl_value,
                    message=f"Metadata mismatch for {key}.",
                )
            )
    return rows


def _metadata(slot: Any) -> dict[str, Any]:
    order_mode, price_mode, _, price_limit_variable, price_limit_factor = _slot_order_fields(slot)
    return {
        "side": _slot_side(slot),
        "market_type": _slot_market_type(slot),
        "phase": str(getattr(slot, "execution_phase", "")),
        "order_mode": order_mode,
        "price_mode": price_mode,
        "limit_style": str(getattr(getattr(slot, "limit_style", None), "value", getattr(slot, "limit_style", "")) or ""),
        "price_for_bounds": str(getattr(slot, "price_for_bounds", "") or ""),
        "price_limit_factor": "" if price_limit_factor is None else str(float(price_limit_factor)),
        "price_limit_variable": price_limit_variable,
        "edge_env": str(getattr(slot, "edge_env", "") or ""),
        "max_runner_stake_env": str(getattr(slot, "max_runner_stake_env", "") or ""),
        "requires_mom45": str(bool(getattr(slot, "requires_mom45", False))),
    }


def _row(
    *,
    context_id: str,
    python_strategy_id: str,
    excel_strategy_id: str,
    status: str,
    mismatch_type: str,
    python_value: Any,
    excel_value: Any,
    message: str,
) -> dict[str, Any]:
    return {
        "context_id": context_id,
        "python_strategy_id": python_strategy_id,
        "excel_strategy_id": excel_strategy_id,
        "status": status,
        "mismatch_type": mismatch_type,
        "python_value": python_value,
        "excel_value": excel_value,
        "message": message,
    }


if __name__ == "__main__":
    raise SystemExit(main())
