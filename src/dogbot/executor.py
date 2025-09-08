# src/dogbot/executor.py
from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from loguru import logger

try:
    # On n'échoue pas si StrategyManager n'est pas présent.
    from .strategies import StrategyManager  # type: ignore
except Exception:  # pragma: no cover
    StrategyManager = Any  # typing fallback


class _NoOpStrategy:
    """Stratégie neutre : ne renvoie aucune instruction."""
    def decide_all(self, *args, **kwargs):
        return []


def _coerce_seconds(val: Any) -> Optional[float]:
    """Transforme divers formats (list/tuple/str/int/float/None) en float|None."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, (list, tuple)) and len(val) > 0:
        try:
            return float(val[0])
        except Exception:
            return None
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        try:
            return float(val)
        except Exception:
            return None
    return None


def _today_snapshot_path() -> str:
    # Fichier au format 20250908_snapshots.csv dans la racine du projet courant
    fname = datetime.now(timezone.utc).strftime("%Y%m%d") + "_snapshots.csv"
    return os.path.abspath(os.path.join(os.getcwd(), fname))


def _ensure_header(path: str, header: Iterable[str]) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(list(header))


def _instantiate_strategy(obj: Any) -> Any:
    """Si on nous passe une classe, on l'instancie; sinon on renvoie tel quel."""
    if obj is None:
        return _NoOpStrategy()
    try:
        if isinstance(obj, type):
            return obj()  # StrategyManager -> StrategyManager()
    except Exception:
        pass
    return obj


class Executor:
    def __init__(
        self,
        client,  # BetfairClient
        strategy: Optional[Any] = None,
        *,
        dry_run: bool = True,
        poll_interval: float | int | str | list | tuple = 2.0,
        snapshot_seconds: float | int | str | list | tuple | None = None,
    ) -> None:
        self.client = client
        self.strategy = _instantiate_strategy(strategy)
        self.dry_run = bool(dry_run)

        # conversions robustes
        self._poll_interval = _coerce_seconds(poll_interval) or 2.0
        self._snapshot_period = _coerce_seconds(snapshot_seconds)

        # snapshot
        self._snapshot_path = _today_snapshot_path()
        self._next_snapshot_at = (
            time.monotonic() + self._snapshot_period
            if self._snapshot_period is not None
            else None
        )

        # header minimal (une ligne par marché)
        self._snapshot_header = [
            "ts_utc",
            "market_id",
            "course_uid",
            "inplay",
            "t_to_off_s",
            "runners",
            "event_name",
            "market_type",
            "venue",
        ]

        logger.info(
            "Executor ready (dry_run={}, poll_interval={}s, snapshot_period={}s)",
            self.dry_run,
            self._poll_interval,
            self._snapshot_period,
        )

    # ------------------------------------------------------------------ #

    def run(self, market_ids: list[str]) -> None:
        if not market_ids:
            logger.warning("No market_ids to poll — exiting run().")
            return

        logger.info("Starting polling loop…")
        try:
            while True:
                books = self.client.get_market_books(market_ids)  # dict[mid] = MarketBook
                now = datetime.now(timezone.utc)

                # éventuel snapshot
                if self._snapshot_period is not None and self._should_snapshot():
                    try:
                        self._write_snapshot(now, books)
                    except Exception as e:
                        logger.warning("snapshot write error: {}", e)
                    self._next_snapshot_at = time.monotonic() + self._snapshot_period

                for mid, book in books.items():
                    try:
                        t_to_off = self.client.get_time_to_off(book)
                    except TypeError as e:
                        logger.debug("[executor] get_time_to_off(book) -> {}", e)
                        t_to_off = None
                    except Exception as e:
                        logger.debug(
                            "[executor] get_time_to_off exception on {}: {}", mid, e
                        )
                        t_to_off = None

                    inplay = bool(getattr(book, "inplay", False))
                    runners = len(getattr(book, "runners", []) or [])

                    logger.debug(
                        "[poll] {} t_to_off={} inplay={} runners={}",
                        mid,
                        None if t_to_off is None else int(t_to_off),
                        inplay,
                        runners,
                    )

                    entry = self.client.get_market_index_entry(mid)
                    decisions = self._safe_decide_all(
                        market_book=book,
                        market_index_entry=entry,
                        now=now,
                    )

                    if decisions:
                        self._place_or_log(mid, decisions)

                time.sleep(self._poll_interval)

        except KeyboardInterrupt:
            logger.info("Polling loop stopped by user (Ctrl+C).")

    # ------------------------------------------------------------------ #

    def _safe_decide_all(self, **kwargs) -> list[dict] | list[Any]:
        """Essaye plusieurs signatures pour decide_all afin d'être compatible avec ton StrategyManager."""
        strat = self.strategy
        if not hasattr(strat, "decide_all"):
            return []

        mb = kwargs.get("market_book")
        mie = kwargs.get("market_index_entry")
        now = kwargs.get("now")

        # 1) Essai en kwargs (idéal si les noms correspondent)
        try:
            out = strat.decide_all(**kwargs)
            return out or []
        except TypeError as te:
            # continue avec des variantes
            logger.debug("[strategy] kwargs call failed: {}", te)

        # 2) Positionnel (book, entry, now)
        try:
            out = strat.decide_all(mb, mie, now)
            return out or []
        except TypeError:
            pass

        # 3) Positionnel (entry, book, now)
        try:
            out = strat.decide_all(mie, mb, now)
            return out or []
        except TypeError:
            pass

        # 4) Positionnel (book, entry)
        try:
            out = strat.decide_all(mb, mie)
            return out or []
        except TypeError:
            pass

        # 5) Positionnel (book, now)
        try:
            out = strat.decide_all(mb, now)
            return out or []
        except TypeError:
            pass

        # 6) Positionnel (book)
        try:
            out = strat.decide_all(mb)
            return out or []
        except TypeError as te:
            logger.warning("[strategy] decide_all type error: {}", te)
            return []

    def _place_or_log(self, market_id: str, instructions: list[dict] | list[Any]) -> None:
        if self.dry_run:
            total = 0.0
            for ins in instructions:
                try:
                    total += float(getattr(ins, "size", None) or ins.get("size", 0.0) or 0.0)
                except Exception:
                    pass
            logger.info(
                "DRY-RUN: would place {} instruction(s) on {} with total stake ~{:.2f}",
                len(instructions),
                market_id,
                total,
            )
            return

        logger.warning("Live placing not implemented in this snippet.")

    # ------------------------------------------------------------------ #
    # Snapshots

    def _should_snapshot(self) -> bool:
        if self._next_snapshot_at is None:
            return False
        return time.monotonic() >= self._next_snapshot_at

    def _write_snapshot(self, now_utc: datetime, books: Dict[str, Any]) -> None:
        path = self._snapshot_path
        _ensure_header(path, self._snapshot_header)

        rows = []
        for mid, book in books.items():
            entry = self.client.get_market_index_entry(mid)
            course_uid = getattr(entry, "course_uid", None)
            event_name = getattr(entry, "event_name", None)
            market_type = getattr(entry, "market_type", None)
            venue = getattr(entry, "venue", None)

            inplay = bool(getattr(book, "inplay", False))
            runners = len(getattr(book, "runners", []) or [])

            try:
                t_to_off = self.client.get_time_to_off(book)
            except Exception:
                t_to_off = None

            rows.append(
                [
                    now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                    mid,
                    course_uid,
                    inplay,
                    None if t_to_off is None else int(t_to_off),
                    runners,
                    event_name,
                    market_type,
                    venue,
                ]
            )

        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(rows)
