# src/dogbot/executor.py
from __future__ import annotations

import time
import logging
from dataclasses import asdict
from typing import List, Sequence, Optional

from .betfair_client import BetfairClient
# ta/tes stratégies doivent exposer .decide_all(...)

logger = logging.getLogger(__name__)


class Executor:
    def __init__(
        self,
        client: BetfairClient,
        strategy,
        dry_run: bool = True,
        poll_interval: float = 1.5,
        **kwargs,  # pour tolérer des kwargs passés par run.py dans d'autres versions
    ) -> None:
        self.client = client
        self.strategy = strategy
        self.dry_run = dry_run
        self.poll_interval = poll_interval

    # --- tiny helper pour log/placer
    def _place_or_log(self, market_id: str, instructions: List[dict]) -> None:
        total_stake = 0.0
        try:
            for ins in instructions or []:
                size = float(ins.get("size", 0) or 0)
                total_stake += size
        except Exception:
            pass

        if not instructions:
            return

        if self.dry_run:
            logger.info(
                f"DRY-RUN: would place {len(instructions)} instruction(s) on {market_id} with total stake ~{total_stake:.2f}"
            )
        else:
            # ici tu brancheras ton placeOrders réel si nécessaire
            logger.info(
                f"PLACE: {len(instructions)} instruction(s) on {market_id} stake ~{total_stake:.2f}"
            )
            # TODO: self.client.place_orders(...)

    def run(self, market_ids: Sequence[str]) -> None:
        logger.info("Starting polling loop…")
        while True:
            try:
                books = self.client.poll_market_books(market_ids)
                for mb in books or []:
                    mid = mb.market_id

                    # t_to_off robuste (utilise MarketBook ou fallback index)
                    try:
                        t_to_off = self.client.get_time_to_off(mid, mb)
                    except Exception as e:
                        logger.debug(f"[poll] {mid}: get_time_to_off error: {e!r}")
                        t_to_off = None

                    inplay = getattr(mb, "inplay", False)
                    runners = len(getattr(mb, "runners", []) or [])
                    logger.debug(f"[poll] {mid} t_to_off={t_to_off}s inplay={inplay} runners={runners}")

                    # --- récupérer index / event (dataclasses) ---
                    idx_dc = self.client.get_market_index_entry(mid)
                    ev_dc = self.client.get_event_index_entry(idx_dc.event_id) if idx_dc else None

                    # convertir en dicts pour la stratégie (qui utilise .get)
                    idx_ctx = asdict(idx_dc) if idx_dc else {}
                    ev_ctx  = asdict(ev_dc)  if ev_dc  else {}

                    # --- APPEL STRATÉGIE ---
                    try:
                        instructions = self.strategy.decide_all(
                            market_id=mid,
                            book=mb,
                            index=idx_ctx,     # dict (safe pour .get)
                            event=ev_ctx,      # dict (safe pour .get)
                            t_to_off=t_to_off,
                            inplay=inplay,
                        )
                    except Exception as e:
                        logger.warning(f"[strategy] decide_all error on {mid}: {e}")
                        instructions = []

                    # --- Exécuter ou log ---
                    self._place_or_log(mid, instructions)

                time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                logger.info("Shutting down (CTRL+C)")
                break
            except Exception as e:
                logger.exception(f"Unhandled error in polling loop: {e}")
                time.sleep(1.0)
