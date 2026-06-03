from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from dogbot.indexer import MarketIndex
from dogbot.types import MarketIndexEntry, RunnerMeta

from dogbot.gruss.gruss_mapper import GrussRunner, GrussSnapshot
from dogbot.gruss.gruss_region import (
    gruss_country_code_for_region,
    normalize_gruss_meeting_name,
    normalize_gruss_region,
)


@dataclass(frozen=True)
class GrussEngineBundle:
    market_index: MarketIndex
    win_book: Any
    place_book: Any


def build_engine_bundle(
    win_snapshot: GrussSnapshot,
    place_snapshot: GrussSnapshot,
    *,
    now_utc: datetime | None = None,
) -> GrussEngineBundle:
    now = now_utc or datetime.now(timezone.utc)
    countdown_seconds = _first_available(
        win_snapshot.metadata.countdown_seconds,
        place_snapshot.metadata.countdown_seconds,
    )
    if countdown_seconds is None:
        raise ValueError("countdown_seconds unavailable")
    start_utc = now + timedelta(seconds=int(countdown_seconds))

    win_id = _require_id(win_snapshot.metadata.market_id, "WIN market_id")
    place_id = _require_id(place_snapshot.metadata.market_id, "PLACE market_id")
    parent_id = _first_available(win_snapshot.metadata.parent_id, place_snapshot.metadata.parent_id)
    event_path = _first_available(win_snapshot.metadata.event_path, place_snapshot.metadata.event_path)
    meeting_name = normalize_gruss_meeting_name(event_path)
    region = normalize_gruss_region(
        event_path=event_path,
        market_title=_first_available(win_snapshot.metadata.market_title, place_snapshot.metadata.market_title),
        meeting_name=meeting_name,
    )
    country_code = gruss_country_code_for_region(region)

    win_entry = _build_index_entry(
        snapshot=win_snapshot,
        market_type="WIN",
        market_id=win_id,
        linked_win_id=win_id,
        linked_place_id=place_id,
        start_utc=start_utc,
        parent_id=parent_id,
        venue=meeting_name,
        country_code=country_code,
        normalized_region=region,
    )
    place_entry = _build_index_entry(
        snapshot=place_snapshot,
        market_type="PLACE",
        market_id=place_id,
        linked_win_id=win_id,
        linked_place_id=place_id,
        start_utc=start_utc,
        parent_id=parent_id,
        venue=meeting_name,
        country_code=country_code,
        normalized_region=region,
    )
    market_index = MarketIndex([win_entry, place_entry])
    return GrussEngineBundle(
        market_index=market_index,
        win_book=_build_book(win_snapshot, "WIN", win_id, start_utc, country_code, region, meeting_name),
        place_book=_build_book(place_snapshot, "PLACE", place_id, start_utc, country_code, region, meeting_name),
    )


def _build_index_entry(
    *,
    snapshot: GrussSnapshot,
    market_type: str,
    market_id: str,
    linked_win_id: str,
    linked_place_id: str,
    start_utc: datetime,
    parent_id: str | None,
    venue: str | None,
    country_code: str | None,
    normalized_region: str,
) -> MarketIndexEntry:
    runners_meta = {
        _selection_id(runner): RunnerMeta(
            selection_id=_selection_id(runner),
            runner_name=runner.runner_name,
            sort_priority=runner.trap,
            trap=str(runner.trap) if runner.trap is not None else None,
            draw=str(runner.trap) if runner.trap is not None else None,
        )
        for runner in snapshot.runners
        if runner.trap is not None
    }
    entry = MarketIndexEntry(
        market_id=market_id,
        market_type=market_type,
        event_id=parent_id,
        event_name=snapshot.metadata.event_path,
        event_open_utc=start_utc,
        venue=venue,
        country_code=country_code,
        event_local_date=None,
        race_number=None,
        course_id=parent_id,
        win_market_id=linked_win_id,
        place_market_id=linked_place_id,
        n_places=snapshot.metadata.winners,
        runners_meta=runners_meta,
    )
    entry.market_name = snapshot.metadata.market_title
    entry.event = SimpleNamespace(id=parent_id, name=snapshot.metadata.event_path, venue=venue, country_code=country_code)
    entry.market_start_time = start_utc
    entry.normalized_region = normalized_region
    entry.runners = [
        SimpleNamespace(
            selection_id=_selection_id(runner),
            runner_name=runner.runner_name,
            sort_priority=runner.trap,
            metadata={"TRAP": str(runner.trap), "CLOTH_NUMBER": str(runner.trap)},
        )
        for runner in snapshot.runners
        if runner.trap is not None
    ]
    return entry


def _build_book(
    snapshot: GrussSnapshot,
    market_type: str,
    market_id: str,
    start_utc: datetime,
    country_code: str | None,
    normalized_region: str,
    meeting_name: str | None,
) -> Any:
    return SimpleNamespace(
        market_id=market_id,
        market_definition=SimpleNamespace(
            status=snapshot.metadata.suspend_status or snapshot.metadata.market_status,
            market_type=market_type,
            number_of_winners=snapshot.metadata.winners,
            market_time=start_utc,
            country_code=country_code,
            normalized_region=normalized_region,
            venue=meeting_name,
        ),
        runners=[_build_runner(runner) for runner in snapshot.runners if runner.trap is not None],
        total_matched=snapshot.metadata.total_matched,
        inplay=False,
    )


def _build_runner(runner: GrussRunner) -> Any:
    return SimpleNamespace(
        selection_id=_selection_id(runner),
        status="ACTIVE",
        last_price_traded=_positive_or_none(runner.ltp),
        total_matched=runner.total_amount_matched,
        ex=SimpleNamespace(
            available_to_back=_ladder_point(runner.best_back, runner.back_stake_1),
            available_to_lay=_ladder_point(runner.best_lay, runner.lay_stake_1),
        ),
    )


def _ladder_point(price: float | None, size: float | None) -> list[Any]:
    price = _positive_or_none(price)
    if price is None:
        return []
    return [SimpleNamespace(price=price, size=float(size or 0.0))]


def _selection_id(runner: GrussRunner) -> int:
    if runner.trap is None:
        raise ValueError(f"runner has no trap: {runner.selection_raw}")
    return int(runner.trap)


def _positive_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def _require_id(value: str | None, label: str) -> str:
    if not value:
        raise ValueError(f"{label} unavailable")
    return str(value)


def _first_available(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None
