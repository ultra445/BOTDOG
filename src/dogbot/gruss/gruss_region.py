from __future__ import annotations

import logging
import re
from typing import Any


_LOGGER = logging.getLogger(__name__)

UK_GREYHOUND_MEETINGS = {
    "central park",
    "crayford",
    "doncaster",
    "dunstall park",
    "harlow",
    "hove",
    "kinsley",
    "monmore",
    "newcastle",
    "nottingham",
    "oxford",
    "pelaw grange",
    "perry barr",
    "romford",
    "sheffield",
    "sunderland",
    "swindon",
    "towcester",
    "valley",
    "yarmouth",
}

ROW_REGION_HINTS = {
    "australia",
    "new zealand",
    "nz",
}

RESULT_SCREEN_TOKENS = {
    "bet ref",
    "result",
}


def normalize_gruss_region(
    event_path: Any = None,
    market_title: Any = None,
    meeting_name: Any = None,
) -> str:
    """Return UK, ROW, or UNKNOWN for a Gruss greyhound market."""
    if is_gruss_result_screen(event_path=event_path, market_title=market_title, meeting_name=meeting_name):
        return "UNKNOWN"

    meeting = normalize_gruss_meeting_name(meeting_name or _meeting_from_event_path(event_path) or market_title)
    haystack = " ".join(
        _clean(value)
        for value in (event_path, market_title, meeting_name, meeting)
        if value not in (None, "")
    ).casefold()

    if any(hint in haystack for hint in ROW_REGION_HINTS):
        return "ROW"
    if meeting.casefold() in UK_GREYHOUND_MEETINGS:
        return "UK"
    if any(meeting_name in haystack for meeting_name in UK_GREYHOUND_MEETINGS):
        return "UK"

    if "pgr" in haystack or "sis" in haystack:
        _LOGGER.warning(
            "Unknown Gruss PGR/SIS meeting region: event_path=%r market_title=%r meeting_name=%r",
            event_path,
            market_title,
            meeting_name or meeting,
        )
        return "UNKNOWN"

    _LOGGER.warning(
        "Unknown Gruss meeting region: event_path=%r market_title=%r meeting_name=%r",
        event_path,
        market_title,
        meeting_name or meeting,
    )
    return "UNKNOWN"


def is_gruss_result_screen(
    event_path: Any = None,
    market_title: Any = None,
    meeting_name: Any = None,
) -> bool:
    values = {_clean(value).casefold() for value in (event_path, market_title, meeting_name) if _clean(value)}
    return bool(values & RESULT_SCREEN_TOKENS)


def normalize_gruss_meeting_name(value: Any) -> str:
    text = _clean(value)
    if "\\" in text:
        text = text.split("\\")[-1]
    text = re.sub(r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+-\s+.*$", "", text)
    text = re.sub(r"\b(?:A\d+|D\d+|S\d+|OR|B\d+)\s+\d{3,4}m\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bTo Be Placed\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def gruss_country_code_for_region(region: str) -> str | None:
    if region == "UK":
        return "GB"
    if region == "ROW":
        return "AU"
    return None


def _meeting_from_event_path(event_path: Any) -> str:
    return normalize_gruss_meeting_name(event_path)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
