import os
import csv
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

from .betfair_client import BetfairClient
from .risk import RiskManager
from .strategies import StrategyManager, StrategyBase

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_list_seconds(name: str, default_csv: str) -> List[int]:
    raw = os.getenv(name, default_csv)
    out: List[int] = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            pass
    return out


class Executor:
    """
    Boucle de polling + déclenchement des stratégies.
    Rétro-compatible avec d’anciens run.py (kwargs ignorés proprement).
    """

    def __init__(
        self,
        client: BetfairClient,
        risk: Optional[RiskManager] = None,
        strategy_manager: Optional[StrategyManager] = None,
        *,
        strategy: Optional[StrategyBase] = None,
        strategies: Optional[List[StrategyBase]] = None,
        poll_interval_secs: Optional[int] = None,
        phase_seconds: Optional[List[int]] = None,
        phase_tolerance: Optional[int] = None,
        dry_run: Optional[bool] = None,
        snapshot_enabled: Optional[bool] = None,
        snapshot_path: Optional[str] = None,
        snapshot_fields: Optional[List[str]] = None,
        **kwargs,
    ):
        self.client = client
        self.risk = risk if risk is not None else RiskManager()

        # StrategyManager à partir de strategy/strategies si non fourni
        if strategy_manager is None:
            bundle: List[StrategyBase] = []
            if strategies:
                bundle.extend(strategies)
            if strategy:
                bundle.append(strategy)
            self.strategy_manager = StrategyManager(bundle)
        else:
            self.strategy_manager = strategy_manager

        # Fréquence de polling & phases
        if poll_interval_secs is None:
            poll_interval_secs = _env_int("POLL_INTERVAL_SECS", 1)
        self.poll_interval_secs = int(poll_interval_secs)

        if phase_seconds is None:
            phase_seconds = _env_list_seconds("PHASE_SECONDS", "300,150,80,45,5")
        self.phase_seconds = sorted([int(x) for x in phase_seconds], reverse=True)

        if phase_tolerance is None:
            phase_tolerance = _env_int("PHASE_TOLERANCE", 2)
        self.phase_tolerance = int(phase_tolerance)

        # Options dry-run & snapshot
        self.dry_run = True if dry_run is None else bool(dry_run)
        self.snapshot_enabled = False if snapshot_enabled is None else bool(snapshot_enabled)
        self.snapshot_path = snapshot_path
        self.snapshot_fields = snapshot_fields

        # Anti double-tir : (market_id, phase, strategy_name)
        self._fired: Dict[Tuple[str, int, str], bool] = {}

    # -----------------------------
    # Helpers
    # -----------------------------
    def _seconds_to_off(self, idx: Dict[str, Any]) -> Optional[int]:
        """Calcule T- (en s) à partir de l'index catalogue (clé 'market_time')."""
        try:
            mt: Optional[datetime] = None
            if isinstance(idx, dict):
                mt = idx.get("market_time") or idx.get("start_time")
            if not mt:
                return None
            now = _utcnow()
            return int((mt - now).total_seconds())
        except Exception:
            return None

    def _within_phase(self, t_to_off: int, phase: int) -> bool:
        """Vérifie si t_to_off est dans la fenêtre [phase ± tolérance]."""
        return abs(t_to_off - phase) <= self.phase_tolerance

    def _snapshot_header(self) -> List[str]:
        base = ["ts_utc", "market_id", "inplay", "t_to_off", "runners"]
        if self.snapshot_fields:
            for f in self.snapshot_fields:
                if f not in base:
                    base.append(f)
        return base

    def _snapshot_row(
        self,
        market_id: str,
        book: Any,
        idx: Dict[str, Any],
        t_to_off: Optional[int],
    ) -> List[Any]:
        ts = _utcnow().isoformat()
        inplay = getattr(book, "inplay", False)
        runners = len(getattr(book, "runners", []) or [])
        row: Dict[str, Any] = {
            "ts_utc": ts,
            "market_id": market_id,
            "inplay": bool(inplay),
            "t_to_off": t_to_off if t_to_off is not None else "",
            "runners": runners,
        }
        if self.snapshot_fields:
            for f in self.snapshot_fields:
                if f in row:
                    continue
                val = None
                if isinstance(idx, dict):
                    val = idx.get(f)
                if val is None and hasattr(book, f):
                    val = getattr(book, f)
                row[f] = val if val is not None else ""
        return [row[c] for c in self._snapshot_header()]

    def _maybe_write_snapshot(
        self,
        market_id: str,
        book: Any,
        idx: Dict[str, Any],
        t_to_off: Optional[int],
    ) -> None:
        if not self.snapshot_enabled:
            return
        out_path = self.snapshot_path or os.getenv("SNAPSHOT_CSV_PATH", "snapshots.csv")
        exists = os.path.exists(out_path)
        try:
            with open(out_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(self._snapshot_header())
                writer.writerow(self._snapshot_row(market_id, book, idx, t_to_off))
        except Exception as e:
            log.warning("snapshot write failed: %s", e)

    def _place_or_log(self, market_id: str, instructions: List[Dict[str, Any]]) -> None:
        """Place les ordres… ou log en dry-run."""
        if not instructions:
            return
        if self.dry_run:
            total = sum(float(i.get("size") or 0.0) for i in instructions)
            log.info(
                "DRY-RUN: would place %d instruction(s) on %s with total stake ~%.2f",
                len(instructions),
                market_id,
                total,
            )
            return
        try:
            self.client.place_orders(market_id, instructions)
        except Exception as e:
            log.warning("place_orders error on %s: %s", market_id, e)

    def _books_to_dict(self, books_any: Any) -> Dict[str, Any]:
        """
        Normalise le retour de poll_market_books :
        - si dict: le retourne tel quel
        - si list: construit {market_id: book}
        - sinon: dict vide
        """
        if isinstance(books_any, dict):
            return books_any
        out: Dict[str, Any] = {}
        if isinstance(books_any, list):
            for b in books_any:
                mid = getattr(b, "market_id", None)
                if mid:
                    out[str(mid)] = b
        return out

    # -----------------------------
    # Boucle principale
    # -----------------------------
    def run(self, market_ids: List[str]) -> None:
        log.info("Starting polling loop…")

        # Pré-chauffage si disponible
        prefetch = getattr(self.client, "prefetch_market_books", None)
        if callable(prefetch):
            try:
                prefetch(market_ids)
            except Exception as e:
                log.debug("prefetch_market_books failed (ignored): %s", e)

        try:
            while True:
                # 1) Lire les MarketBook (liste ou dict selon l’implémentation)
                books_any = self.client.poll_market_books(market_ids)
                books = self._books_to_dict(books_any)

                # 2) Pour chaque marché, décider/agir selon phase
                market_index = getattr(self.client, "market_index", {}) or {}

                for mid, book in books.items():
                    idx = market_index.get(mid)
                    if not idx:
                        log.debug("[poll] %s: no index entry", mid)
                        # on peut quand même continuer (snapshot minimal, etc.)

                    # T- avant le départ
                    t_to_off = self._seconds_to_off(idx or {})
                    if t_to_off is None:
                        # secours via client.get_time_to_off(market_id, book) si dispo
                        get_tto = getattr(self.client, "get_time_to_off", None)
                        if callable(get_tto):
                            try:
                                t_to_off = get_tto(market_id=mid, book=book)
                            except Exception:
                                pass

                    inplay = getattr(book, "inplay", False)
                    runners = len(getattr(book, "runners", []) or [])

                    log.debug(
                        "[poll] %s t_to_off=%s inplay=%s runners=%d",
                        mid,
                        t_to_off,
                        inplay,
                        runners,
                    )

                    # Ecrire snapshot si demandé
                    self._maybe_write_snapshot(mid, book, idx or {}, t_to_off)

                    # Pas d'heure ou déjà en live => pas de phases
                    if t_to_off is None or inplay:
                        continue

                    # Pour chaque phase configurée
                    for phase in self.phase_seconds:
                        if not self._within_phase(t_to_off, phase):
                            continue

                        # Décisions de toutes les stratégies
                        decisions = self.strategy_manager.decide_all(mid, book, idx or {})

                        for strat_name, instrs in decisions.items():
                            key = (mid, phase, strat_name)
                            if self._fired.get(key):
                                continue
                            self._fired[key] = True

                            final_instrs = self.risk.filter_instructions(mid, strat_name, instrs)
                            self._place_or_log(mid, final_instrs)

                time.sleep(self.poll_interval_secs)

        except KeyboardInterrupt:
            log.info("Shutting down (CTRL+C)")
        except Exception as e:
            log.exception("Fatal error in polling loop: %s", e)
            raise
