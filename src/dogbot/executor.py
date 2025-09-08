# src/dogbot/executor.py

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

LOG = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Executor:
    """
    Boucle de polling simple :
      - récupère les MarketBook via client.poll_market_books(market_ids) -> dict {market_id: book}
      - calcule t_to_off via client.get_time_to_off(book)
      - appelle la stratégie via _call_strategy_decide_all(...) avec adaptation de signature
      - place (ou log en DRY-RUN) les instructions retournées

    Paramètres snapshot_* sont optionnels et sûrs : s'ils ne sont pas fournis, rien n'est écrit.
    """

    def __init__(
        self,
        client: Any,
        strategy: Any,
        dry_run: bool = True,
        poll_seconds: float = 1.5,
        snapshot_path: Optional[str] = None,
        snapshot_period: Optional[float] = None,
    ) -> None:
        self.client = client
        self.strategy = strategy
        self.dry_run = bool(dry_run)
        self.poll_seconds = float(poll_seconds)

        # Snapshot (CSV) optionnel
        self._snapshot_path = Path(snapshot_path) if snapshot_path else None
        self._snapshot_period = float(snapshot_period) if snapshot_period is not None else None
        self._last_snapshot_ts: float = 0.0
        self._snapshot_header_written = False

    # --------------------------------------------------------------------- #
    # API publique
    # --------------------------------------------------------------------- #
    def run(self, market_ids: List[str]) -> None:
        if not market_ids:
            LOG.warning("Aucun market_id fourni à Executor.run(); arrêt.")
            return

        LOG.info("Starting polling loop…")
        # Préfetch éventuel (no-op si non implémenté)
        try:
            if hasattr(self.client, "prefetch_market_books"):
                self.client.prefetch_market_books(market_ids)
        except Exception as e:
            LOG.debug("prefetch_market_books error (ignorée): %s", e)

        try:
            while True:
                books: Dict[str, Any] = {}
                try:
                    books = self.client.poll_market_books(market_ids) or {}
                    if isinstance(books, list):
                        # Par sécurité si un client tiers renvoie une liste
                        books = {getattr(b, "market_id", f"unknown-{i}"): b for i, b in enumerate(books)}
                except Exception as e:
                    LOG.error("poll_market_books a échoué: %s", e, exc_info=True)
                    time.sleep(self.poll_seconds)
                    continue

                snapshot_rows: List[Dict[str, Any]] = []

                for mid, book in books.items():
                    # in-play + runners
                    inplay = getattr(book, "inplay", False)
                    runners = getattr(book, "runners", None)
                    runner_count = len(runners) if runners is not None else 0

                    # t_to_off
                    t_to_off = None
                    try:
                        t_to_off = self.client.get_time_to_off(book)
                    except Exception as e_tto:
                        LOG.debug("[executor] get_time_to_off(%s) -> %s", mid, e_tto)

                    LOG.debug("[poll] %s t_to_off=%s inplay=%s runners=%s", mid, t_to_off, inplay, runner_count)

                    # Stratégie
                    instructions: List[Dict[str, Any]] = self._call_strategy_decide_all(mid, book)

                    # Placement / log
                    if instructions:
                        self._place_or_log(mid, instructions)

                    # Snapshot (facultatif)
                    if self._should_snapshot():
                        idx = None
                        try:
                            idx = self.client.get_market_index_entry(mid)
                        except Exception:
                            idx = None

                        snapshot_rows.append(
                            {
                                "ts": _now_utc().isoformat(),
                                "market_id": mid,
                                "event_id": idx.get("event_id") if idx else None,
                                "event_name": idx.get("event_name") if idx else None,
                                "country": idx.get("country") if idx else None,
                                "market_type": idx.get("market_type") if idx else None,
                                "number_of_winners": idx.get("number_of_winners") if idx else None,
                                "start_time": (idx.get("start_time").isoformat() if idx and idx.get("start_time") else None),
                                "inplay": bool(inplay),
                                "runners": runner_count,
                                "t_to_off": t_to_off,
                            }
                        )

                # Écriture snapshot si nécessaire
                if snapshot_rows and self._should_snapshot():
                    self._write_snapshot_rows(snapshot_rows)
                    self._last_snapshot_ts = time.time()

                time.sleep(self.poll_seconds)

        except KeyboardInterrupt:
            LOG.info("Shutting down (CTRL+C)")

    # --------------------------------------------------------------------- #
    # Stratégie — adaptation de signature
    # --------------------------------------------------------------------- #
    def _call_strategy_decide_all(self, market_id: str, market_book: Any) -> List[Dict[str, Any]]:
        """
        Essaie différentes signatures pour être compatible avec diverses implémentations:
          - decide_all(market_id=..., market_book=..., market_index_entry=...)
          - decide_all(market_book=..., market_index_entry=...)
          - decide_all(market_id, market_book, market_index_entry)
          - decide_all(market_id, market_book)
          - decide_all(market_book)
        Retourne une liste d'instructions (peut être vide).
        """
        idx = None
        try:
            idx = self.client.get_market_index_entry(market_id)
        except Exception:
            idx = None

        attempts = [
            {"kwargs": {"market_id": market_id, "market_book": market_book, "market_index_entry": idx}},
            {"kwargs": {"market_book": market_book, "market_index_entry": idx}},
            {"args": (market_id, market_book, idx)},
            {"args": (market_id, market_book)},
            {"args": (market_book,)},
        ]

        for i, call in enumerate(attempts, 1):
            try:
                if "kwargs" in call:
                    res = self.strategy.decide_all(**call["kwargs"])  # type: ignore[arg-type]
                else:
                    res = self.strategy.decide_all(*call["args"])  # type: ignore[arg-type]
                if res is None:
                    return []
                if isinstance(res, list):
                    return res
                # tolère un seul dict
                if isinstance(res, dict):
                    return [res]
                # sinon on ignore
                LOG.debug("[strategy] decide_all tentative %d: type retour inattendu (%s)", i, type(res).__name__)
            except TypeError as te:
                # mismatch de signature -> on tente la suivante
                LOG.debug("[strategy] decide_all tentative %d: TypeError %s", i, te)
                continue
            except Exception as e:
                LOG.warning("[strategy] decide_all erreur: %s", e, exc_info=True)
                return []
        # aucune signature n'a fonctionné
        LOG.warning("[strategy] aucune signature decide_all compatible trouvée")
        return []

    # --------------------------------------------------------------------- #
    # Placement (ou log en dry-run)
    # --------------------------------------------------------------------- #
    def _place_or_log(self, market_id: str, instructions: List[Dict[str, Any]]) -> None:
        total_stake = 0.0
        for ins in instructions:
            try:
                s = ins.get("size") or ins.get("stake") or 0.0
                total_stake += float(s)
            except Exception:
                pass

        if self.dry_run:
            LOG.info("DRY-RUN: would place %d instruction(s) on %s with total stake ~%.2f",
                     len(instructions), market_id, total_stake)
            return

        # Exemple de placement réel (à adapter selon le format d'instruction exact)
        try:
            if hasattr(self.client, "place_orders"):
                self.client.place_orders(market_id, instructions)  # si dispo dans ton client
            else:
                LOG.warning("Placement réel non implémenté côté client. Instructions ignorées.")
        except Exception as e:
            LOG.warning("Erreur placement ordres sur %s: %s", market_id, e, exc_info=True)

    # --------------------------------------------------------------------- #
    # Snapshot CSV
    # --------------------------------------------------------------------- #
    def _should_snapshot(self) -> bool:
        return self._snapshot_path is not None and self._snapshot_period is not None

    def _write_snapshot_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not self._snapshot_path:
            return

        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        # entêtes
        fieldnames = [
            "ts",
            "market_id",
            "event_id",
            "event_name",
            "country",
            "market_type",
            "number_of_winners",
            "start_time",
            "inplay",
            "runners",
            "t_to_off",
        ]
        write_header = not self._snapshot_path.exists() or not self._snapshot_header_written

        try:
            with self._snapshot_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                    self._snapshot_header_written = True
                for r in rows:
                    w.writerow({k: r.get(k) for k in fieldnames})
        except Exception as e:
            LOG.warning("Écriture snapshot échouée (%s): %s", self._snapshot_path, e)
