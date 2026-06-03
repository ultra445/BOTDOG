from __future__ import annotations

import unittest

from dogbot.config import (
    DATA_PROVIDER_BETFAIR_API,
    DATA_PROVIDER_GRUSS_EXCEL,
    ORDER_PROVIDER_DRY_RUN,
    ProviderConfig,
)
from dogbot.feed_selector import create_data_feed_from_config
from dogbot.gruss.gruss_feed import GrussFeed


class FeedSelectorTests(unittest.TestCase):
    def test_betfair_api_keeps_existing_path(self) -> None:
        feed = create_data_feed_from_config(
            ProviderConfig(DATA_PROVIDER_BETFAIR_API, ORDER_PROVIDER_DRY_RUN)
        )

        self.assertIsNone(feed)

    def test_gruss_excel_selects_gruss_feed(self) -> None:
        feed = create_data_feed_from_config(
            ProviderConfig(DATA_PROVIDER_GRUSS_EXCEL, ORDER_PROVIDER_DRY_RUN)
        )

        self.assertIsInstance(feed, GrussFeed)


if __name__ == "__main__":
    unittest.main()
