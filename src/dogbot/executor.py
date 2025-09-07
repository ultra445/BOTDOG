# src/dogbot/executor.py
from __future__ import annotations

import time
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Sequence

from .betfair_client import BetfairClient

logger = logging.getLogger(__name__)


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Convertit un dataclass/objet quelconque en dict, sinon {}."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    try:
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass
    try:
        return dict(obj.__dict__)
    except Exception:
        return {}


class Executor:
    def __init__(
        self,
        client: BetfairClient,
        strategy,
        dry_run: bool = True,
        poll_interval: float = 1.5,
        **kwargs,  # tolère d'anciens kwargs (snapshot_enabled, features, …)
    ) -> None:
        self.client = client
        self.strategy = strategy
        self.dry_run = dry_run
        self.poll_interval = poll_interval

    # ---------- Helpers ----------
    def _calc_t_to_off(self, market_id: str, book) -> Optional[int]:
        """Essaie différentes signatures pour rester compatible."""
        try:
            # signature moderne: (market_id, book)
            return self.client.get_time_to_off(market_id, book)
        except TypeError:
            # ancienne signature: (market_id)
            try:
                return self.client.get_time_to_off(market_id)
            except Exception:
                return None
        except AttributeError:
            # aucun helper dispo -> on tente un fallback facultatif
            try:
                derive = getattr(self.client, "derive_time_to_off_from_book")
                return derive(book)
            except Exception as e:
                logger.debug(f"[poll] {market_id}: no time_to_off helper ({e!r})")
                return None
        except Exception as e:
            logger.debug(f"[poll] {market_id}: get_time_to_off error: {e!r}")
            return None

    def _place_or_log(self, market_id: str, instructions: Optional[List[dict]]) -> None:
        if not instructions:
            return
        total = 0.0
        for ins in instructions:
            try:
                total += float(ins.get("size", 0) or 0)
            except Exception:
                pass

        if self.dry_run:
            logger.info(
                f"DRY-RUN: would place {len(instructions)} instruction(s) on {market_id} with total stake ~{total:.2f}"
            )
            return

        # Ici tu brancheras ton appel réel à placeOrders si besoin
        try:
            logger.info(
                f"PLACE: {len(instructions)} instruction(s) on {market_id} stake ~{total:.2f}"
            )
            # TODO: self.client.place_orders(market_id, instructions)
        except Exception as e:
            logger.error(f"place_orders failed on {market_id}: {e}")

    # ---------- Boucle principale ----------
    def run(self, market_ids: Sequence[str]) -> None:
        logger.info("Starting polling loop…")
        while True:
            try:
                books = self.client.poll_market_books(market_ids) or []
                for mb in books:
                    mid = getattr(mb, "market_id", None) or getattr(mb, "marketId", None) or "?"

                    # t_to_off robuste
                    t_to_off = self._calc_t_to_off(mid, mb)

                    inplay = bool(getattr(mb, "inplay", False))
                    runners_count = len(getattr(mb, "runners", []) or [])
                    logger.debug(
                        f"[poll] {mid} t_to_off={t_to_off}s inplay={inplay} runners={runners_count}"
                    )

                    # Index marché/événement (facultatif si helpers absents)
                    idx_dc = None
                    try:
                        idx_dc = self.client.get_market_index_entry(mid)
                    except AttributeError as e:
                        logger.debug(f"[poll] {mid}: no index entry helper ({e!r})")
                    except Exception as e:
                        logger.debug(f"[poll] {mid}: get_market_index_entry error: {e!r}")

                    ev_dc = None
                    try:
                        ev_id = getattr(idx_dc, "event_id", None)
                        if ev_id:
                            ev_dc = self.client.get_event_index_entry(ev_id)
                    except Exception as e:
                        logger.debug(f"[poll] {mid}: get_event_index_entry error: {e!r}")

                    idx_ctx = _to_dict(idx_dc)
                    ev_ctx = _to_dict(ev_dc)

                    # Appel stratégie (reçoit des dicts -> .get() OK)
                    try:
                        instructions = self.strategy.decide_all(
                            market_id=mid,
                            book=mb,
                            index=idx_ctx,
                            event=ev_ctx,
                            t_to_off=t_to_off,
                            inplay=inplay,
                        )
                    except Exception as e:
                        logger.warning(f"[strategy] decide_all error on {mid}: {e}")
                        instructions = []

                    self._place_or_log(mid, instructions)

                time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                logger.info("Shutting down (CTRL+C)")
                break
            except Exception as e:
                logger.exception(f"Unhandled error in polling loop: {e}")
                time.sleep(1.0)
