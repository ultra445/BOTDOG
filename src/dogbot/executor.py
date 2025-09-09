import os
import csv
import time
import logging
import datetime as dt
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class Executor:
    """
    Boucle de polling:
      - récupère régulièrement les MarketBooks,
      - appelle la stratégie via decide_all(market_book, market_index_entry, now_utc),
      - écrit des snapshots CSV périodiques.
    """

    def __init__(
        self,
        client,
        strategy,
        snapshot_enabled: bool = True,
        snapshot_path: str = "./data",
        snapshot_period: float = 5.0,
        dry_run: bool = True,
        poll_interval: float = 2.0,
    ):
        self.client = client
        self.strategy = strategy
        self._snapshot_enabled = bool(snapshot_enabled)
        self._snapshot_path = snapshot_path
        self._snapshot_period = float(snapshot_period)
        self._dry_run = bool(dry_run)
        self._poll_interval = float(poll_interval)

        os.makedirs(self._snapshot_path, exist_ok=True)

    # ---------- Helpers ----------

    @staticmethod
    def _utc_now() -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc)

    def _get_time_to_off(self, market_id: str) -> Optional[float]:
        try:
            return self.client.time_to_off_seconds(market_id)
        except Exception as e:
            logger.debug("[executor] time_to_off(%s) -> %s", market_id, repr(e))
            return None

    def _snapshot_filename(self) -> str:
        day = self._utc_now().strftime("%Y%m%d")
        return os.path.join(self._snapshot_path, f"{day}_snapshots.csv")

    def _append_snapshot_row(
        self,
        market_id: str,
        book: Any,
        index_entry: Optional[dict],
        t_to_off_s: Optional[float],
    ) -> None:
        if not self._snapshot_enabled:
            return

        filename = self._snapshot_filename()
        file_exists = os.path.isfile(filename)

        # Données simples et robustes
        inplay = getattr(book, "inplay", False)
        runners = getattr(book, "runners", None)
        n_runners = len(runners) if runners else 0
        total_matched = getattr(book, "total_matched", None)

        # Favori par last_price_traded > 0 min
        fav_lpt = None
        if runners:
            prices = [
                r.last_price_traded
                for r in runners
                if getattr(r, "last_price_traded", None) not in (None, 0)
            ]
            fav_lpt = min(prices) if prices else None

        row = {
            "ts_utc": self._utc_now().isoformat(),
            "market_id": market_id,
            "market_type": index_entry.get("market_type") if index_entry else None,
            "inplay": inplay,
            "runners": n_runners,
            "t_to_off_s": None if t_to_off_s is None else round(float(t_to_off_s), 3),
            "last_price_fav": fav_lpt,
            "total_matched": total_matched,
            "venue": index_entry.get("venue") if index_entry else None,
            "event_name": index_entry.get("event_name") if index_entry else None,
        }

        fieldnames = [
            "ts_utc",
            "market_id",
            "market_type",
            "inplay",
            "runners",
            "t_to_off_s",
            "last_price_fav",
            "total_matched",
            "venue",
            "event_name",
        ]

        with open(filename, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            w.writerow(row)

    # ---------- Strategy wrapper ----------

    def _call_strategy_decide_all(
        self,
        market_id: str,
        book: Any,
        index_entry: Optional[dict],
        now_utc: dt.datetime,
    ) -> List[dict]:
        """
        Appelle la stratégie de manière sûre.
        Signature attendue: decide_all(market_book, market_index_entry, now_utc) -> list[dikt]
        """
        try:
            return self.strategy.decide_all(book, index_entry, now_utc) or []
        except TypeError as e:
            logger.warning(
                "[strategy] decide_all type error on %s: %s", market_id, e
            )
            return []
        except Exception as e:
            logger.warning(
                "[strategy] decide_all error on %s: %s", market_id, e
            )
            return []

    # ---------- Main loop ----------

    def run(self, market_ids: List[str]) -> None:
        if not market_ids:
            logger.warning("No markets to run.")
            return

        logger.info("Starting polling loop… (%d markets)", len(market_ids))

        # Prefetch initial books
        books = self.client.prefetch_market_books(market_ids)

        last_snapshot = 0.0
        while True:
            start = time.perf_counter()

            # Rafraîchir les books
            books = self.client.get_market_books(market_ids)
            now_utc = self._utc_now()

            for mid in market_ids:
                book = books.get(mid)
                if not book:
                    continue

                index_entry = self.client.get_market_index_entry(mid)
                t_to_off = self._get_time_to_off(mid)

                inplay = getattr(book, "inplay", False)
                runners = getattr(book, "runners", None)
                n_runners = len(runners) if runners else 0

                logger.debug(
                    "[poll] %s t_to_off=%ss inplay=%s runners=%s",
                    mid,
                    "None" if t_to_off is None else int(t_to_off),
                    inplay,
                    n_runners,
                )

                # Stratégie
                instructions = self._call_strategy_decide_all(
                    mid, book, index_entry, now_utc
                )
                if instructions:
                    if self._dry_run:
                        logger.info(
                            "DRY-RUN: would place %d instruction(s) on %s",
                            len(instructions),
                            mid,
                        )
                    else:
                        # Intégration ordres si besoin, ici on log seulement
                        logger.info(
                            "LIVE: placing %d instruction(s) on %s (not implemented here)",
                            len(instructions),
                            mid,
                        )

                # Snapshots (périodiques)
                now_s = time.perf_counter()
                if self._snapshot_enabled and (now_s - last_snapshot) >= self._snapshot_period:
                    self._append_snapshot_row(mid, book, index_entry, t_to_off)
                    last_snapshot = now_s

            # Cadence
            elapsed = time.perf_counter() - start
            sleep_for = max(0.0, self._poll_interval - elapsed)
            time.sleep(sleep_for)
