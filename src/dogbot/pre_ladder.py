from __future__ import annotations

import math
from dataclasses import dataclass

PRE_LADDER_SYSTEM_IDS = {
    "BACK_PLACE_101",
    "BACK_PLACE_102",
    "BACK_PLACE_201",
    "BACK_PLACE_202",
    "BACK_PLACE_203",
    "LAY_PLACE_301",
    "LAY_PLACE_302",
    "LAY_PLACE_303",
    "LAY_PLACE_304",
    "LAY_PLACE_305",
    "LAY_PLACE_351",
}

BETFAIR_PRICE_BANDS: tuple[tuple[float, float, float], ...] = (
    (1.01, 2.0, 0.01),
    (2.0, 3.0, 0.02),
    (3.0, 4.0, 0.05),
    (4.0, 6.0, 0.1),
    (6.0, 10.0, 0.2),
    (10.0, 20.0, 0.5),
    (20.0, 30.0, 1.0),
    (30.0, 50.0, 2.0),
    (50.0, 100.0, 5.0),
    (100.0, 1000.0, 10.0),
)


@dataclass(frozen=True)
class PreLadderOrderState:
    status: str
    requested_stake: float
    matched_stake: float = 0.0
    cancel_succeeded: bool = True


@dataclass(frozen=True)
class PreLadderStepDecision:
    send_new_order: bool
    current_step_stake: float
    previous_order_status: str = ""
    matched_stake: float = 0.0
    remaining_stake: float = 0.0
    cancelled_previous: bool = False
    cancel_failed: bool = False
    stop_reason: str = ""


@dataclass(frozen=True)
class GrussPreLadderTriggerPlan:
    trigger: str
    allowed: bool
    bet_ref_required: bool
    bet_ref_present: bool
    no_stack: bool
    real_confirmed: bool
    reason: str


@dataclass(frozen=True)
class PreLadderPricePlan:
    start_price: float | None
    prices: list[float]
    reason: str = ""


def round_to_betfair_tick(price: float) -> float:
    if not _valid_price(price):
        raise ValueError(f"invalid_betfair_price={price!r}")
    candidates = _betfair_ticks()
    nearest = min(candidates, key=lambda tick: (abs(tick - float(price)), tick))
    return _clean_price(nearest)


def round_final_lim_to_ladder_tick(side: str, price: float) -> float:
    if not _valid_price(price):
        raise ValueError(f"invalid_betfair_price={price!r}")
    upper_side = str(side or "").upper()
    if upper_side not in {"BACK", "LAY"}:
        raise ValueError(f"invalid_side={side!r}")
    value = float(price)
    ticks = _betfair_ticks()
    if upper_side == "BACK":
        for tick in ticks:
            if tick >= value - 1e-9:
                return tick
    else:
        for tick in reversed(ticks):
            if tick <= value + 1e-9:
                return tick
    raise ValueError(f"invalid_betfair_price={price!r}")


def previous_tick(price: float) -> float:
    rounded = round_to_betfair_tick(price)
    ticks = _betfair_ticks()
    try:
        index = ticks.index(rounded)
    except ValueError as exc:
        raise ValueError(f"invalid_betfair_price={price!r}") from exc
    return ticks[max(0, index - 1)]


def next_tick(price: float) -> float:
    rounded = round_to_betfair_tick(price)
    ticks = _betfair_ticks()
    try:
        index = ticks.index(rounded)
    except ValueError as exc:
        raise ValueError(f"invalid_betfair_price={price!r}") from exc
    return ticks[min(len(ticks) - 1, index + 1)]


def build_pre_ladder_prices(
    side: str,
    start_price: float,
    final_lim_price: float,
    *,
    steps: int = 4,
) -> list[float]:
    if steps <= 0:
        raise ValueError("steps_must_be_positive")
    start = round_to_betfair_tick(start_price)
    upper_side = str(side or "").upper()
    if upper_side not in {"BACK", "LAY"}:
        raise ValueError(f"invalid_side={side!r}")
    final = round_final_lim_to_ladder_tick(upper_side, final_lim_price)
    if steps == 1:
        return [final]
    if upper_side == "BACK" and start <= final:
        raise ValueError("no_better_back_ladder_range")
    if upper_side == "LAY" and start >= final:
        raise ValueError("no_better_lay_ladder_range")

    raw_prices = [
        start + ((final - start) * (index / (steps - 1)))
        for index in range(steps)
    ]
    prices = [round_to_betfair_tick(price) for price in raw_prices]
    prices[0] = start
    prices[-1] = final
    if upper_side == "BACK":
        prices = _force_monotonic(prices, descending=True)
    else:
        prices = _force_monotonic(prices, descending=False)
    prices[-1] = final
    return prices


def build_pre_ladder_from_same_side_offer(
    side: str,
    best_same_side_offer: float,
    final_lim_price: float,
    *,
    steps: int = 4,
) -> PreLadderPricePlan:
    upper_side = str(side or "").upper()
    if upper_side not in {"BACK", "LAY"}:
        raise ValueError(f"invalid_side={side!r}")
    final = round_final_lim_to_ladder_tick(upper_side, final_lim_price)
    same_side = round_to_betfair_tick(best_same_side_offer)

    if upper_side == "BACK":
        if same_side <= final:
            return PreLadderPricePlan(
                start_price=None,
                prices=[final],
                reason="no_better_back_ladder_range",
            )
        start = previous_tick(same_side)
        if start <= final:
            return PreLadderPricePlan(
                start_price=None,
                prices=[final],
                reason="no_better_back_ladder_range",
            )
    else:
        if same_side >= final:
            return PreLadderPricePlan(
                start_price=None,
                prices=[final],
                reason="no_better_lay_ladder_range",
            )
        start = next_tick(same_side)
        if start >= final:
            return PreLadderPricePlan(
                start_price=None,
                prices=[final],
                reason="no_better_lay_ladder_range",
            )

    return PreLadderPricePlan(
        start_price=start,
        prices=build_pre_ladder_prices(upper_side, start, final, steps=steps),
    )


def decide_pre_ladder_step(
    previous_order: PreLadderOrderState | None,
    *,
    full_stake: float,
) -> PreLadderStepDecision:
    if previous_order is None:
        return PreLadderStepDecision(
            send_new_order=True,
            current_step_stake=round(float(full_stake), 2),
            remaining_stake=round(float(full_stake), 2),
        )

    status = str(previous_order.status or "").upper()
    matched_stake = round(max(0.0, float(previous_order.matched_stake)), 2)
    requested_stake = round(max(0.0, float(previous_order.requested_stake)), 2)
    remaining_stake = round(max(0.0, requested_stake - matched_stake), 2)

    if status == "FULLY_MATCHED" or remaining_stake <= 0:
        return PreLadderStepDecision(
            send_new_order=False,
            current_step_stake=0.0,
            previous_order_status=status,
            matched_stake=matched_stake,
            remaining_stake=0.0,
            stop_reason="fully_matched",
        )

    if not previous_order.cancel_succeeded:
        return PreLadderStepDecision(
            send_new_order=False,
            current_step_stake=0.0,
            previous_order_status=status,
            matched_stake=matched_stake,
            remaining_stake=remaining_stake,
            cancelled_previous=False,
            cancel_failed=True,
            stop_reason="cancel_failed_do_not_stack",
        )

    stake = remaining_stake if status == "PARTIALLY_MATCHED" else round(float(full_stake), 2)
    return PreLadderStepDecision(
        send_new_order=True,
        current_step_stake=round(stake, 2),
        previous_order_status=status,
        matched_stake=matched_stake,
        remaining_stake=round(stake, 2),
        cancelled_previous=True,
        stop_reason="",
    )


def plan_gruss_pre_ladder_trigger(
    *,
    side: str,
    step_index: int,
    bet_ref: str | None,
    replace_confirmed: bool = False,
) -> GrussPreLadderTriggerPlan:
    """Plan the Gruss trigger for diagnostics without authorising real ladder writes."""
    upper_side = str(side or "").upper()
    if upper_side not in {"BACK", "LAY"}:
        raise ValueError(f"invalid_side={side!r}")
    if step_index <= 0:
        return GrussPreLadderTriggerPlan(
            trigger=upper_side,
            allowed=True,
            bet_ref_required=False,
            bet_ref_present=False,
            no_stack=False,
            real_confirmed=False,
            reason="initial_ladder_trigger_preview_only",
        )

    has_bet_ref = bool(str(bet_ref or "").strip())
    if not has_bet_ref:
        return GrussPreLadderTriggerPlan(
            trigger="",
            allowed=False,
            bet_ref_required=True,
            bet_ref_present=False,
            no_stack=True,
            real_confirmed=False,
            reason="bet_ref_not_ready",
        )

    replace_trigger = "BACKR" if upper_side == "BACK" else "LAYR"
    return GrussPreLadderTriggerPlan(
        trigger=replace_trigger,
        allowed=True,
        bet_ref_required=True,
        bet_ref_present=True,
        no_stack=False,
        real_confirmed=bool(replace_confirmed),
        reason=(
            "replace_with_existing_bet_ref"
            if replace_confirmed
            else "replace_trigger_not_confirmed_preview_only"
        ),
    )


def _force_monotonic(prices: list[float], *, descending: bool) -> list[float]:
    if len(prices) <= 2:
        return prices
    adjusted = list(prices)
    for index in range(1, len(adjusted) - 1):
        previous = adjusted[index - 1]
        current = adjusted[index]
        if descending and current > previous:
            adjusted[index] = previous
        elif not descending and current < previous:
            adjusted[index] = previous
    return adjusted


def _valid_price(price: float) -> bool:
    try:
        value = float(price)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and 1.01 <= value <= 1000.0


def _betfair_ticks() -> list[float]:
    ticks: list[float] = []
    for low, high, step in BETFAIR_PRICE_BANDS:
        value = low
        while value < high - 1e-9:
            ticks.append(_clean_price(value))
            value += step
    ticks.append(1000.0)
    return sorted(set(ticks))


def _clean_price(price: float) -> float:
    return round(float(price) + 1e-9, 2)
