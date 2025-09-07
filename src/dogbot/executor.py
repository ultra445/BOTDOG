from __future__ import annotations

import time
from typing import Iterable, List, Optional

from loguru import logger

# types hints légers
from .betfair_client import BetfairClient, MarketIndexEntry


class Executor:
    def __init__(
        self,
        client: BetfairClient,
        strategy,  # on ne touche pas à ta stratégie; on la laisse telle quelle
        poll_interval: float = 2.0,
        dry_run: bool = True,
    ) -> None:
        self.client = client
        self.strategy = strategy
        self.poll_interval = poll_interval
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    def _place_or_log(self, market_id: str, num_instr: int) -> None:
        if self.dry_run:
            logger.info("DRY-RUN: would place {} instruction(s) on {}", num_instr, market_id)
            return
        # TODO: appel réel de placement d’ordres si dry_run=False

    # ------------------------------------------------------------------
    def run(self, market_ids: Iterable[str]) -> None:
        """
        Boucle de polling simple :
        - récupère les books,
        - calcule t_to_off,
        - appelle la stratégie (protégée),
        - heartbeat DEBUG à chaque tour même si aucune donnée.
        """
        mlist: List[str] = list(market_ids)
        logger.info("Starting polling loop… ({} market ids)", len(mlist))

        loop = 0
        while True:
            loop += 1

            try:
                books = self.client.poll_market_books(mlist)
            except Exception as e:
                logger.error("poll_market_books raised: {}", e)
                time.sleep(self.poll_interval)
                continue

            # Heartbeat de boucle même si aucun book
            n_books = len(books) if isinstance(books, dict) else len(books or [])
            logger.debug("[loop {}] fetched {} books for {} ids", loop, n_books, len(mlist))

            # Normalisation de sécurité : si on reçoit une liste, on mappe -> dict
            if not isinstance(books, dict):
                books = {getattr(b, "market_id", getattr(b, "marketId", "")): b for b in (books or []) if b}

            if not books:
                time.sleep(self.poll_interval)
                continue

            for mid, book in books.items():
                # index
                idx: Optional[MarketIndexEntry] = self.client.get_market_index_entry(mid)
                if idx is None:
                    logger.debug("[poll] {}: no index entry", mid)

                # t_to_off
                try:
                    t_to_off = self.client.get_time_to_off(book, idx.start_time if idx else None)
                except Exception as e:
                    logger.debug("[poll] {}: get_time_to_off error: {}", mid, repr(e))
                    t_to_off = None

                inplay = bool(getattr(book, "inplay", False))
                runners = len(getattr(book, "runners", []) or [])
                logger.debug("[poll] {} t_to_off={}s inplay={} runners={}", mid, t_to_off, inplay, runners)

                # Appel stratégie, mais blindé pour ne jamais stopper la boucle
                try:
                    # Beaucoup de stratégies s’attendent à un 'ctx' dict-like avec .get(...)
                    ctx = {"index": idx, "book": book, "t_to_off": t_to_off}
                    # Adapte à ta signature (si différente), ici on reste simple :
                    decisions = []
                    if hasattr(self.strategy, "decide_all"):
                        decisions = self.strategy.decide_all(mid, book, ctx)  # type: ignore
                    elif hasattr(self.strategy, "decide"):
                        decisions = self.strategy.decide(mid, book, ctx)  # type: ignore
                    else:
                        decisions = []
                except Exception as e:
                    logger.warning("[strategy] decide_all error on {}: {}", mid, repr(e))
                    decisions = []

                # Logging/placement (ici, minimal : 1 instruction si la stratégie renvoie quelque chose)
                if decisions:
                    self._place_or_log(mid, len(decisions))

            time.sleep(self.poll_interval)
