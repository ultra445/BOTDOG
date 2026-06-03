from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


_TRAP_PATTERNS = (
    re.compile(r"^\s*(?:trap|t)\s*([1-6])\b", re.IGNORECASE),
    re.compile(r"^\s*[\[(]?([1-6])[\]).:\-\s]+"),
)
_TRAP_PREFIX = re.compile(r"^\s*[\[(]?([1-6])[\]).:\-\s]+")
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GrussMarketMetadata:
    market_title: Any
    event_path: Any
    parent_id: str | None
    last_updated: Any
    countdown: Any
    countdown_seconds: int | None
    countdown_display: str | None
    market_status: Any
    suspend_status: Any
    total_matched: float | None
    market_id: str | None
    winners: int | None


@dataclass(frozen=True)
class GrussRunner:
    selection_raw: str
    trap: int | None
    runner_name: str
    back_odds_1: float | None
    back_stake_1: float | None
    back_odds_2: float | None
    back_stake_2: float | None
    back_odds_3: float | None
    back_stake_3: float | None
    lay_odds_1: float | None
    lay_stake_1: float | None
    lay_odds_2: float | None
    lay_stake_2: float | None
    lay_odds_3: float | None
    lay_stake_3: float | None
    reduction_factor: float | None
    last_price_matched: float | None
    total_amount_matched: float | None
    best_back: float | None
    best_lay: float | None
    ltp: float | None


@dataclass(frozen=True)
class GrussSnapshot:
    sheet_name: str
    metadata: GrussMarketMetadata
    runners: list[GrussRunner]


def normalize_runner_name(name: object) -> str:
    """Return a stable runner-name key for matching WIN and PLACE runners."""
    if name is None:
        return ""
    text = unicodedata.normalize("NFKC", str(name))
    text = _TRAP_PREFIX.sub("", text)
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )
    text = re.sub(r"[^0-9a-zA-Z]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def extract_trap(runner_name: object) -> int | None:
    """Extract a greyhound trap number from common Gruss runner labels."""
    if runner_name is None:
        return None
    name = unicodedata.normalize("NFKC", str(runner_name))
    for pattern in _TRAP_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def parse_gruss_sheet(rows: list[list[Any]], sheet_name: str) -> GrussSnapshot:
    """Parse a Gruss worksheet dump into market metadata and runner rows."""
    countdown = _cell(rows, 1, 3)
    countdown_seconds = parse_countdown_seconds(countdown)
    metadata = GrussMarketMetadata(
        market_title=_cell(rows, 0, 0),
        event_path=_cell(rows, 0, 5),
        parent_id=_find_label_value(rows, 0, "parent id"),
        last_updated=_cell(rows, 1, 1),
        countdown=countdown,
        countdown_seconds=countdown_seconds,
        countdown_display=format_countdown_display(countdown_seconds),
        market_status=_cell(rows, 1, 4),
        suspend_status=_cell(rows, 1, 5),
        total_matched=_as_float(_cell(rows, 2, 1)),
        market_id=_find_label_value(rows, 2, "market id"),
        winners=_as_int(_cell(rows, 2, 4)),
    )
    return GrussSnapshot(
        sheet_name=sheet_name.upper(),
        metadata=metadata,
        runners=_parse_runners(rows),
    )


def validate_win_place_pair(win_snapshot: GrussSnapshot, place_snapshot: GrussSnapshot) -> bool:
    """Return True only when WIN and PLACE snapshots look like the same race."""
    warnings = get_win_place_validation_warnings(win_snapshot, place_snapshot)

    for warning in warnings:
        _LOGGER.warning("Gruss WIN/PLACE validation failed: %s", warning)

    return not warnings


def get_win_place_validation_warnings(
    win_snapshot: GrussSnapshot,
    place_snapshot: GrussSnapshot,
) -> list[str]:
    """Return validation warnings for a WIN/PLACE pair without logging."""
    warnings: list[str] = []
    win_meta = win_snapshot.metadata
    place_meta = place_snapshot.metadata

    if win_meta.parent_id and place_meta.parent_id and win_meta.parent_id != place_meta.parent_id:
        warnings.append(f"parent_id mismatch: WIN={win_meta.parent_id} PLACE={place_meta.parent_id}")

    if win_meta.event_path and place_meta.event_path and win_meta.event_path != place_meta.event_path:
        warnings.append(f"event_path mismatch: WIN={win_meta.event_path!r} PLACE={place_meta.event_path!r}")

    trap_overlap = _overlap_ratio(
        {runner.trap for runner in win_snapshot.runners if runner.trap is not None},
        {runner.trap for runner in place_snapshot.runners if runner.trap is not None},
    )
    if trap_overlap is None:
        warnings.append("runner trap overlap unavailable")
    elif trap_overlap < 0.75:
        warnings.append(f"runner trap overlap too low: {trap_overlap:.2f}")

    name_overlap = _overlap_ratio(
        {normalize_runner_name(runner.runner_name) for runner in win_snapshot.runners if runner.runner_name},
        {normalize_runner_name(runner.runner_name) for runner in place_snapshot.runners if runner.runner_name},
    )
    if name_overlap is None:
        warnings.append("runner name overlap unavailable")
    elif name_overlap < 0.75:
        warnings.append(f"runner name overlap too low: {name_overlap:.2f}")

    return warnings


def require_valid_win_place_pair(win_snapshot: GrussSnapshot, place_snapshot: GrussSnapshot) -> None:
    """Raise before any downstream strategy evaluation can use mismatched sheets."""
    if not validate_win_place_pair(win_snapshot, place_snapshot):
        raise ValueError("Gruss WIN and PLACE snapshots do not describe the same race")


def parse_countdown_seconds(value: Any) -> int | None:
    """Convert Gruss countdown values to signed seconds.

    Gruss may expose D2 as an Excel day fraction or as text such as
    "00:00:41" / "-00:00:13".
    """
    value = _empty_to_none(value)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(round(float(value) * 86400))

    text = _clean_text(value)
    time_seconds = _parse_time_text_seconds(text)
    if time_seconds is not None:
        return time_seconds

    number = _as_float(text)
    if number is not None:
        return int(round(number * 86400))
    return None


def format_countdown_display(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    sign = "-" if seconds < 0 else ""
    absolute_seconds = abs(seconds)
    minutes = absolute_seconds // 60
    remaining_seconds = absolute_seconds % 60
    return f"{sign}{minutes:02d}:{remaining_seconds:02d}"


class GrussMapper:
    """Stateless mapper for Gruss Excel rows."""

    normalize_runner_name = staticmethod(normalize_runner_name)
    extract_trap = staticmethod(extract_trap)
    parse_sheet = staticmethod(parse_gruss_sheet)
    parse_countdown_seconds = staticmethod(parse_countdown_seconds)
    format_countdown_display = staticmethod(format_countdown_display)
    get_win_place_validation_warnings = staticmethod(get_win_place_validation_warnings)
    validate_win_place_pair = staticmethod(validate_win_place_pair)
    require_valid_win_place_pair = staticmethod(require_valid_win_place_pair)


def _parse_runners(rows: list[list[Any]]) -> list[GrussRunner]:
    runners: list[GrussRunner] = []
    for row in rows[4:]:
        selection_raw = _clean_text(_row_cell(row, 0))
        if not selection_raw:
            break

        trap = extract_trap(selection_raw)
        runner_name = _strip_trap_prefix(selection_raw)
        back_odds_1 = _as_float(_row_cell(row, 5))
        lay_odds_1 = _as_float(_row_cell(row, 7))
        last_price_matched = _as_float(_row_cell(row, 14))

        runners.append(
            GrussRunner(
                selection_raw=selection_raw,
                trap=trap,
                runner_name=runner_name,
                back_odds_1=back_odds_1,
                back_stake_1=_as_float(_row_cell(row, 6)),
                back_odds_2=_as_float(_row_cell(row, 3)),
                back_stake_2=_as_float(_row_cell(row, 4)),
                back_odds_3=_as_float(_row_cell(row, 1)),
                back_stake_3=_as_float(_row_cell(row, 2)),
                lay_odds_1=lay_odds_1,
                lay_stake_1=_as_float(_row_cell(row, 8)),
                lay_odds_2=_as_float(_row_cell(row, 9)),
                lay_stake_2=_as_float(_row_cell(row, 10)),
                lay_odds_3=_as_float(_row_cell(row, 11)),
                lay_stake_3=_as_float(_row_cell(row, 12)),
                reduction_factor=_as_float(_row_cell(row, 13)),
                last_price_matched=last_price_matched,
                total_amount_matched=_as_float(_row_cell(row, 15)),
                best_back=back_odds_1,
                best_lay=lay_odds_1,
                ltp=last_price_matched,
            )
        )
    return runners


def _find_label_value(rows: list[list[Any]], row_index: int, label: str) -> str | None:
    row = rows[row_index] if 0 <= row_index < len(rows) else []
    target = _label_key(label)
    for index, value in enumerate(row[:-1]):
        if _label_key(value) == target:
            return _as_identifier(row[index + 1])
    return None


def _cell(rows: list[list[Any]], row_index: int, column_index: int) -> Any:
    if row_index < 0 or row_index >= len(rows):
        return None
    return _empty_to_none(_row_cell(rows[row_index], column_index))


def _row_cell(row: list[Any], column_index: int) -> Any:
    if column_index < 0 or column_index >= len(row):
        return None
    return row[column_index]


def _strip_trap_prefix(value: object) -> str:
    return _TRAP_PREFIX.sub("", _clean_text(value)).strip()


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _label_key(value: object) -> str:
    return _clean_text(value).rstrip(":").strip().casefold()


def _empty_to_none(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _as_float(value: Any) -> float | None:
    value = _empty_to_none(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    return int(number)


def _as_identifier(value: Any) -> str | None:
    value = _empty_to_none(value)
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = _clean_text(value)
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text or None


def _parse_time_text_seconds(text: str) -> int | None:
    match = re.fullmatch(r"([+-]?)(\d{1,3}):(\d{2})(?::(\d{2}))?", text)
    if not match:
        return None
    sign_text, first, second, third = match.groups()
    if third is None:
        minutes = int(first)
        seconds = int(second)
        total_seconds = minutes * 60 + seconds
    else:
        hours = int(first)
        minutes = int(second)
        seconds = int(third)
        total_seconds = hours * 3600 + minutes * 60 + seconds
    if sign_text == "-":
        total_seconds *= -1
    return total_seconds


def _overlap_ratio(left: set[Any], right: set[Any]) -> float | None:
    if not left or not right:
        return None
    return len(left & right) / min(len(left), len(right))
