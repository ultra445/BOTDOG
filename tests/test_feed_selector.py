from __future__ import annotations

import unittest
from unittest.mock import patch

from dogbot.config import (
    DATA_PROVIDER_BETFAIR_API,
    DATA_PROVIDER_GRUSS_EXCEL,
    ORDER_PROVIDER_DRY_RUN,
    ORDER_PROVIDER_GRUSS_EXCEL_DRYRUN,
    ProviderConfig,
    load_provider_config,
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

    def test_default_betfair_provider_path_remains_dry_run(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            config = load_provider_config()

        self.assertEqual(config, ProviderConfig(DATA_PROVIDER_BETFAIR_API, ORDER_PROVIDER_DRY_RUN))

    def test_gruss_data_provider_defaults_to_gruss_dry_run_orders(self) -> None:
        with patch.dict("os.environ", {"DOGBOT_DATA_PROVIDER": DATA_PROVIDER_GRUSS_EXCEL}, clear=True):
            config = load_provider_config()

        self.assertEqual(
            config,
            ProviderConfig(DATA_PROVIDER_GRUSS_EXCEL, ORDER_PROVIDER_GRUSS_EXCEL_DRYRUN),
        )


if __name__ == "__main__":
    unittest.main()
