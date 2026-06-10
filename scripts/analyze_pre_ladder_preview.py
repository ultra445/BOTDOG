from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _default_trade_path() -> Path:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Path("data") / f"trades_{today}.csv"


def _as_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _step_number(row: dict[str, str]) -> int:
    text = str(row.get("ladder_step") or "").strip()
    if "/" not in text:
        return 999
    try:
        return int(text.split("/", 1)[0])
    except ValueError:
        return 999


def _row_run_id(row: dict[str, str]) -> str:
    return str(row.get("run_id") or row.get("evaluation_id") or "legacy_run")


def _parent_market_id(row: dict[str, str]) -> str:
    return str(row.get("parent_market_id") or row.get("course_id") or row.get("market_id") or "")


def _group_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (_row_run_id(row), _parent_market_id(row), str(row.get("ladder_id") or ""))


def _as_bool(value: object) -> bool | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _as_int(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parent_post_reached(rows: Iterable[dict[str, str]]) -> set[tuple[str, str]]:
    reached: set[tuple[str, str]] = set()
    for row in rows:
        phase = str(row.get("execution_phase") or "").upper()
        milestone = str(row.get("milestone") or "").strip()
        complete = str(row.get("complete_after_post") or "").strip().lower() in {"1", "true", "yes", "on"}
        if (phase == "POST" and milestone in {"0", "0.0"}) or complete:
            reached.add((_row_run_id(row), _parent_market_id(row)))
    return reached


def _post_status_by_parent(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], dict[str, object]]:
    status: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {
            "post_checked": False,
            "post_evaluated": False,
            "post_signal_count": 0,
            "post_missing_reason": "post_milestone_not_reached",
            "explicit": False,
        }
    )
    post_reached = _parent_post_reached(rows)
    for key in post_reached:
        status[key]["post_checked"] = True
        status[key]["post_missing_reason"] = "post_not_logged_or_no_signal"

    for row in rows:
        key = (_row_run_id(row), _parent_market_id(row))
        current = status[key]
        checked = _as_bool(row.get("post_checked"))
        evaluated = _as_bool(row.get("post_evaluated"))
        signal_count = _as_int(row.get("post_signal_count"))
        missing_reason = str(row.get("post_missing_reason") or "").strip()
        if checked is not None:
            current["post_checked"] = checked
            current["explicit"] = True
        if evaluated is not None:
            current["post_evaluated"] = evaluated
            current["explicit"] = True
        if signal_count is not None:
            current["post_signal_count"] = max(int(current["post_signal_count"]), signal_count)
            current["explicit"] = True
        if missing_reason:
            current["post_missing_reason"] = missing_reason
            current["explicit"] = True
        if str(row.get("execution_phase") or "").upper() == "POST":
            current["post_checked"] = True
            current["post_evaluated"] = True
            current["post_signal_count"] = int(current["post_signal_count"]) + 1
            if not missing_reason:
                current["post_missing_reason"] = "post_logged"

    for key, current in status.items():
        if not current["post_checked"]:
            current["post_missing_reason"] = "post_milestone_not_reached"
        elif current["post_evaluated"] and int(current["post_signal_count"]) == 0:
            current["post_missing_reason"] = "no_post_signal"
        elif current["post_evaluated"]:
            current["post_missing_reason"] = "post_logged"
        elif not current["post_missing_reason"]:
            current["post_missing_reason"] = "post_not_logged_or_no_signal"
    return dict(status)


def _parent_final_pre_reached(rows: Iterable[dict[str, str]]) -> set[tuple[str, str]]:
    reached: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("status") != "PRE_LADDER_PREVIEW":
            continue
        milestone = str(row.get("milestone") or row.get("ladder_seconds_before_off") or "").strip()
        if milestone in {"5", "5.0"} or str(row.get("ladder_step") or "") == "4/4":
            reached.add((_row_run_id(row), _parent_market_id(row)))
    return reached


def _has_no_better_range(row: dict[str, str]) -> bool:
    text = "|".join(
        str(row.get(name) or "")
        for name in ("no_better_ladder_range_reason", "reason")
    )
    return "no_better_back_ladder_range" in text or "no_better_lay_ladder_range" in text


def _issue(
    label: str,
    *,
    run_id: str,
    ladder_id: str,
    steps: list[str],
    prices: list[float | None],
    final_ticks: list[float | None],
    reasons: list[str],
) -> str:
    return (
        f"{label} | run_id/evaluation_id={run_id} ladder_id={ladder_id} "
        f"steps_seen={','.join(steps)} "
        f"current_ladder_price_by_step={','.join('' if price is None else str(price) for price in prices)} "
        f"final_lim_price_tick={','.join('' if tick is None else str(tick) for tick in final_ticks)} "
        f"reason={'|'.join(reasons)}"
    )


def analyze_rows(rows: list[dict[str, str]]) -> tuple[list[str], list[str], list[str]]:
    pre_rows = [row for row in rows if row.get("status") == "PRE_LADDER_PREVIEW" and row.get("ladder_id")]
    post_status = _post_status_by_parent(rows)
    final_pre_reached = _parent_final_pre_reached(rows)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in pre_rows:
        grouped[_group_key(row)].append(row)

    lines: list[str] = []
    warnings: list[str] = []
    issues: list[str] = []
    for group_key in sorted(grouped):
        run_id, parent_market_id, ladder_id = group_key
        group = sorted(grouped[group_key], key=_step_number)
        first = group[0]
        side = str(first.get("side") or "").upper()
        strategy = first.get("final_system") or first.get("strategy") or ""
        runner = first.get("runner") or first.get("runner_name") or first.get("selection_id") or ""
        steps = [str(row.get("ladder_step") or "") for row in group]
        prices = [_as_float(row.get("current_ladder_price")) for row in group]
        final_ticks = [_as_float(row.get("final_lim_price_tick")) for row in group]
        final_tick = next((price for price in final_ticks if price is not None), None)
        final_tick_values = {tick for tick in final_ticks if tick is not None}
        ladder_prices = first.get("ladder_prices") or ""
        reasons = [
            str(row.get("no_better_ladder_range_reason") or row.get("reason") or "")
            for row in group
            if row.get("no_better_ladder_range_reason") or row.get("reason")
        ]
        has_no_better = any(_has_no_better_range(row) for row in group)
        parent_post_status = post_status.get(
            (run_id, parent_market_id),
            {
                "post_checked": False,
                "post_evaluated": False,
                "post_signal_count": 0,
                "post_missing_reason": "post_milestone_not_reached",
                "explicit": False,
            },
        )
        post_checked = bool(parent_post_status["post_checked"])
        post_evaluated = bool(parent_post_status["post_evaluated"])
        post_signal_count = int(parent_post_status["post_signal_count"])
        post_missing_reason = str(parent_post_status["post_missing_reason"])

        lines.extend(
            [
                f"run_id/evaluation_id={run_id} parent_market_id={parent_market_id} ladder_id={ladder_id}",
                f"  runner={runner} side={side} strategy={strategy}",
                f"  steps_seen={','.join(steps)}",
                f"  current_ladder_price_by_step={','.join('' if price is None else str(price) for price in prices)}",
                f"  final_lim_price_tick={','.join('' if tick is None else str(tick) for tick in final_ticks)}",
                f"  ladder_prices={ladder_prices}",
                f"  reason={'|'.join(reasons)}",
                f"  post_checked={post_checked}",
                f"  post_evaluated={post_evaluated}",
                f"  post_signal_count={post_signal_count}",
                f"  post_missing_reason={post_missing_reason}",
            ]
        )

        numeric_prices = [price for price in prices if price is not None]
        if has_no_better:
            if any(
                price is not None and tick is not None and price != tick
                for price, tick in zip(prices, final_ticks)
            ):
                issues.append(
                    _issue(
                        "no_better range pas au final_lim_price_tick",
                        run_id=run_id,
                        ladder_id=ladder_id,
                        steps=steps,
                        prices=prices,
                        final_ticks=final_ticks,
                        reasons=reasons,
                    )
                )
        elif side == "BACK":
            if any(later > earlier for earlier, later in zip(numeric_prices, numeric_prices[1:])):
                issues.append(
                    _issue(
                        "BACK ladder non decroissant",
                        run_id=run_id,
                        ladder_id=ladder_id,
                        steps=steps,
                        prices=prices,
                        final_ticks=final_ticks,
                        reasons=reasons,
                    )
                )
            if any(
                price is not None and tick is not None and price < tick
                for price, tick in zip(prices, final_ticks)
            ):
                issues.append(
                    _issue(
                        "BACK prix hors limite",
                        run_id=run_id,
                        ladder_id=ladder_id,
                        steps=steps,
                        prices=prices,
                        final_ticks=final_ticks,
                        reasons=reasons,
                    )
                )
        elif side == "LAY":
            if any(later < earlier for earlier, later in zip(numeric_prices, numeric_prices[1:])):
                issues.append(
                    _issue(
                        "LAY ladder non croissant",
                        run_id=run_id,
                        ladder_id=ladder_id,
                        steps=steps,
                        prices=prices,
                        final_ticks=final_ticks,
                        reasons=reasons,
                    )
                )
            if any(
                price is not None and tick is not None and price > tick
                for price, tick in zip(prices, final_ticks)
            ):
                issues.append(
                    _issue(
                        "LAY prix hors limite",
                        run_id=run_id,
                        ladder_id=ladder_id,
                        steps=steps,
                        prices=prices,
                        final_ticks=final_ticks,
                        reasons=reasons,
                    )
                )

        if (
            not has_no_better
            and "4/4" not in steps
            and (run_id, parent_market_id) in final_pre_reached
        ):
            issues.append(
                _issue(
                    "step 4 manquant",
                    run_id=run_id,
                    ladder_id=ladder_id,
                    steps=steps,
                    prices=prices,
                    final_ticks=final_ticks,
                    reasons=reasons,
                )
            )
        if final_tick_values:
            for row, price, tick in zip(group, prices, final_ticks):
                if not has_no_better and row.get("ladder_step") == "4/4" and tick is not None and price != tick:
                    issues.append(
                        _issue(
                            "step 4 different de final_lim_price_tick",
                            run_id=run_id,
                            ladder_id=ladder_id,
                            steps=steps,
                            prices=prices,
                            final_ticks=final_ticks,
                            reasons=reasons,
                        )
                    )
        if "4/4" in steps:
            post_context_reasons = [*reasons, post_missing_reason]
            if not post_checked:
                warnings.append(
                    _issue(
                        "POST non verifie apres PRE",
                        run_id=run_id,
                        ladder_id=ladder_id,
                        steps=steps,
                        prices=prices,
                        final_ticks=final_ticks,
                        reasons=post_context_reasons,
                    )
                )
            elif not post_evaluated:
                target = issues if post_missing_reason == "expected_post_evaluation_missing" else warnings
                target.append(
                    _issue(
                        "POST evaluation absente apres PRE",
                        run_id=run_id,
                        ladder_id=ladder_id,
                        steps=steps,
                        prices=prices,
                        final_ticks=final_ticks,
                        reasons=post_context_reasons,
                    )
                )

    return lines, warnings, issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse les lignes PRE_LADDER_PREVIEW par ladder_id.")
    parser.add_argument("--file", type=Path, default=_default_trade_path())
    args = parser.parse_args()

    if not args.file.exists():
        print(f"file_missing={args.file}")
        return 2

    with args.file.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    lines, warnings, issues = analyze_rows(rows)
    print(f"file={args.file}")
    print(f"pre_ladder_groups={sum(1 for line in lines if ' ladder_id=' in line or line.startswith('run_id/evaluation_id='))}")
    for line in lines:
        print(line)
    print("WARNINGS")
    if warnings:
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("- none")
    print("ISSUES")
    if issues:
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("- none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
