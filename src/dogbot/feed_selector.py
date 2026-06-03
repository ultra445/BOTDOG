from __future__ import annotations

from typing import Any

from dogbot.config import DATA_PROVIDER_BETFAIR_API, DATA_PROVIDER_GRUSS_EXCEL, ProviderConfig, load_provider_config
from dogbot.gruss.gruss_excel_bridge import DEFAULT_WORKBOOK_PATH
from dogbot.gruss.gruss_feed import GrussFeed


def create_data_feed_from_config(config: ProviderConfig | None = None) -> Any:
    """Select the configured data provider without changing Betfair's default path."""
    config = config or load_provider_config()
    if config.data_provider == DATA_PROVIDER_BETFAIR_API:
        return None
    if config.data_provider == DATA_PROVIDER_GRUSS_EXCEL:
        return GrussFeed(DEFAULT_WORKBOOK_PATH)
    raise ValueError(f"unsupported data provider: {config.data_provider}")
