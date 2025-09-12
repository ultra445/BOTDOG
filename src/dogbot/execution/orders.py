from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Deque, List
from collections import deque
import csv
import time

import betfairlightweight
from betfairlightweight import filters
from betfairlightweight.exceptions import APIError

# ----- types simples -----
@dataclass
class OrderAttempt:
    ts: str
    market_id: str
    selection_id: int
    side: str         # "BACK" | "LAY"
    price: float
    size: float
    persistence: str  # "LAPSE" | "PERSIST" | "MARKET_ON_CLOSE" | "LIMIT_ON_CLOSE"
    strategy: str
    idempotency_key: str

@dataclass
class OrderResult:
    ok: bool
    status: str
    bet_id: Optional[str]
    matched_size: float
    avg_price_matched: Optional[float]
    error_code: Optional[str]
    instruction_report: Optional[dict]

class OrderExecutor:
    """
    Exécuteur d’ordres Betfair avec :
      - LIMIT (persistence LAPSE/PERSIST)
      - MARKET_ON_CLOSE (BSP sans limite)
      - LIMIT_ON_CLOSE (BSP avec limite)
    + idempotence, retry/backoff, throttling,
      logs CSV : orders_YYYYMMDD.csv + fills_YYYYMMDD.csv
    """
    def __init__(
        self,
        client: betfairlightweight.Exchange,
        data_dir: str = "./data",
        throttle_max_per_minute: int = 25,
        default_persistence: str = "LAPSE",
        fok_ms: Optional[int] = None,   # Fill-or-Kill (LIMIT uniquement)
    ):
        self.client = client
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.throttle_max_per_minute = int(throttle_max_per_minute)
        self.default_persistence = default_persistence
        self.fok_ms = fok_ms

        self._bucket: Deque[float] = deque(maxlen=self.throttle_max_per_minute)
        self._idem: set[str] = set()

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        self.orders_csv = self.data_dir / f"orders_{today}.csv"
        self.fills_csv  = self.data_dir / f"fills_{today}.csv"
        self._ensure_header(self.orders_csv, [
            "ts","market_id","selection_id","side","price","size","persistence",
            "strategy","idempotency_key","ok","status","bet_id","matched_size",
            "avg_price_matched","error_code"
        ])
        self._ensure_header(self.fills_csv, [
            "ts","market_id","selection_id","side","price","size","bet_id","strategy"
        ])

    # ---------- public ----------

    def place_limit(
        self,
        *,
        market_id: str,
        selection_id: int,
        side: str,                 # "BACK"|"LAY"
        price: float,
        size: float,
        strategy: str,
        persistence: Optional[str] = None,  # "LAPSE"|"PERSIST" (pour LIMIT)
        idem_key: Optional[str] = None,
        retries: int = 2,
        backoff_ms: int = 250,
    ) -> OrderResult:
        """Place un ordre LIMIT avec blindages et logging."""
        persistence = persistence or self.default_persistence
        idem_key = idem_key or self._mk_idem_key(market_id, selection_id, side, price, size, strategy)
        if idem_key in self._idem:
            # Idempotence: on ne replace pas le même ordre
            return OrderResult(True, "IDEMPOTENT_SKIPPED", None, 0.0, None, None, None)

        self._throttle_wait()

        attempt = OrderAttempt(
            ts=self._now(), market_id=market_id, selection_id=int(selection_id),
            side=side, price=float(price), size=float(size),
            persistence=persistence, strategy=strategy, idempotency_key=idem_key,
        )
        self._log_order_attempt(attempt)

        instruction = filters.place_instruction(
            order_type="LIMIT",
            selection_id=int(selection_id),
            side=side,
            limit_order=filters.limit_order(
                price=float(price),
                size=float(size),
                persistence_type=persistence
            )
        )
        req = dict(
            market_id=str(market_id),
            instructions=[instruction],
            customer_strategy_ref=str(strategy)[:15],   # suffixe audit
        )

        # LIMIT → FOK applicable éventuellement
        return self._do_place_with_retries(attempt, req, retries, backoff_ms, apply_fok=True)

    def place_sp_market_on_close(
        self,
        *,
        market_id: str,
        selection_id: int,
        side: str,                 # "BACK"|"LAY"
        size_or_liability: float,  # BACK: stake ; LAY: liability
        strategy: str,
        idem_key: Optional[str] = None,
        retries: int = 2,
        backoff_ms: int = 250,
    ) -> OrderResult:
        """
        Betfair BSP sans limite (MARKET_ON_CLOSE).
        NB: Betfair nomme le champ "liability" même pour BACK ; pour BACK, c'est en pratique la taille du pari (stake).
        """
        idem_key = idem_key or f"{market_id}:{selection_id}:{side}:SP_MOC:{round(size_or_liability,2)}:{strategy}"
        if idem_key in self._idem:
            return OrderResult(True, "IDEMPOTENT_SKIPPED", None, 0.0, None, None, None)

        self._throttle_wait()

        attempt = OrderAttempt(
            ts=self._now(), market_id=market_id, selection_id=int(selection_id),
            side=side, price=0.0, size=float(size_or_liability),
            persistence="MARKET_ON_CLOSE", strategy=strategy, idempotency_key=idem_key,
        )
        self._log_order_attempt(attempt)

        instruction = filters.place_instruction(
            order_type="MARKET_ON_CLOSE",
            selection_id=int(selection_id),
            side=side,
            market_on_close_order=filters.market_on_close_order(
                liability=float(size_or_liability)
            )
        )
        req = dict(
            market_id=str(market_id),
            instructions=[instruction],
            customer_strategy_ref=str(strategy)[:15],
        )

        # SP MOC → pas de FOK (il n'y a pas de match avant l'off)
        return self._do_place_with_retries(attempt, req, retries, backoff_ms, apply_fok=False)

    def place_sp_limit_on_close(
        self,
        *,
        market_id: str,
        selection_id: int,
        side: str,                 # "BACK"|"LAY"
        size_or_liability: float,  # BACK: stake ; LAY: liability
        sp_limit_price: float,     # BACK: min SP ; LAY: max SP
        strategy: str,
        idem_key: Optional[str] = None,
        retries: int = 2,
        backoff_ms: int = 250,
    ) -> OrderResult:
        """
        Betfair BSP avec limite (LIMIT_ON_CLOSE).
        BACK → sp_limit_price = SP minimum acceptable ; LAY → maximum acceptable.
        """
        idem_key = idem_key or f"{market_id}:{selection_id}:{side}:SP_LOC:{round(sp_limit_price,2)}:{round(size_or_liability,2)}:{strategy}"
        if idem_key in self._idem:
            return OrderResult(True, "IDEMPOTENT_SKIPPED", None, 0.0, None, None, None)

        self._throttle_wait()

        attempt = OrderAttempt(
            ts=self._now(), market_id=market_id, selection_id=int(selection_id),
            side=side, price=float(sp_limit_price), size=float(size_or_liability),
            persistence="LIMIT_ON_CLOSE", strategy=strategy, idempotency_key=idem_key,
        )
        self._log_order_attempt(attempt)

        instruction = filters.place_instruction(
            order_type="LIMIT_ON_CLOSE",
            selection_id=int(selection_id),
            side=side,
            limit_on_close_order=filters.limit_on_close_order(
                price=float(sp_limit_price),
                liability=float(size_or_liability)
            )
        )
        req = dict(
            market_id=str(market_id),
            instructions=[instruction],
            customer_strategy_ref=str(strategy)[:15],
        )

        # SP LOC → pas de FOK (match seulement à l'off)
        return self._do_place_with_retries(attempt, req, retries, backoff_ms, apply_fok=False)

    def cancel_market_orders(self, *, market_id: str, bet_ids: Optional[List[str]] = None) -> None:
        """Annule des ordres ouverts sur un marché (tous si bet_ids=None)."""
        try:
            if bet_ids:
                instructions = [filters.cancel_instruction(bet_id=b) for b in bet_ids]
                self.client.betting.cancel_orders(market_id=market_id, instructions=instructions)
            else:
                # récupère puis annule tous les ordres ouverts du marché
                cur = self.client.betting.list_current_orders(market_ids=[market_id])
                ids = [c.bet_id for c in getattr(cur, "current_orders", [])]
                if ids:
                    instructions = [filters.cancel_instruction(bet_id=b) for b in ids]
                    self.client.betting.cancel_orders(market_id=market_id, instructions=instructions)
        except Exception:
            # on ne casse jamais l'exécuteur pour un échec d'annulation
            pass

    # ---------- internals ----------

    def _do_place_with_retries(
        self,
        attempt: OrderAttempt,
        req: dict,
        retries: int,
        backoff_ms: int,
        apply_fok: bool,
    ) -> OrderResult:
        """Cœur commun de place_orders avec retry/backoff + logging."""
        last_exc: Optional[Exception] = None
        for attempt_idx in range(retries + 1):
            try:
                rep = self.client.betting.place_orders(**req)
                instr_rep = rep.instruction_reports[0] if rep.instruction_reports else None
                status = getattr(instr_rep, "status", None)
                bet_id = getattr(instr_rep, "bet_id", None)
                matched_size = float(getattr(instr_rep, "size_matched", 0.0) or 0.0)
                avg_price = getattr(instr_rep, "average_price_matched", None)
                error_code = getattr(getattr(instr_rep, "error_code", None), "value", None) if status != "SUCCESS" else None

                ok = (status == "SUCCESS")
                result = OrderResult(ok, status or "UNKNOWN", bet_id, matched_size, avg_price, error_code, getattr(instr_rep, "__dict__", None))
                self._log_order_result(attempt, result)
                self._idem.add(attempt.idempotency_key)

                # FOK : uniquement pour LIMIT (LAPSE/PERSIST)
                if ok and apply_fok and self.fok_ms and matched_size <= 0.0:
                    self._sleep_ms(self.fok_ms)
                    self.cancel_market_orders(market_id=attempt.market_id, bet_ids=[bet_id] if bet_id else None)
                return result

            except APIError as e:
                last_exc = e
                # Certaines erreurs sont "dures" -> stop direct
                if "INSUFFICIENT_FUNDS" in str(e) or "INVALID_ORDER" in str(e):
                    self._log_order_result(attempt, OrderResult(False, "ERROR", None, 0.0, None, str(e), None))
                    return OrderResult(False, "ERROR", None, 0.0, None, str(e), None)
            except Exception as e:
                last_exc = e

            if attempt_idx < retries:
                self._sleep_ms(backoff_ms)

        # après retries
        self._log_order_result(attempt, OrderResult(False, "ERROR", None, 0.0, None, str(last_exc) if last_exc else "unknown_error", None))
        return OrderResult(False, "ERROR", None, 0.0, None, str(last_exc) if last_exc else "unknown_error", None)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _ensure_header(self, path: Path, header: list[str]) -> None:
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

    def _log_order_attempt(self, a: OrderAttempt) -> None:
        with self.orders_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([a.ts, a.market_id, a.selection_id, a.side, a.price, a.size, a.persistence, a.strategy, a.idempotency_key, "", "", "", "", ""])

    def _log_order_result(self, a: OrderAttempt, r: OrderResult) -> None:
        with self.orders_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([self._now(), a.market_id, a.selection_id, a.side, a.price, a.size, a.persistence, a.strategy, a.idempotency_key,
                        "1" if r.ok else "0", r.status, r.bet_id or "", f"{r.matched_size:.2f}", f"{r.avg_price_matched or ''}", r.error_code or ""])

        # si match, log un "fill" simplifié
        if r.ok and r.matched_size and r.matched_size > 0:
            with self.fills_csv.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([self._now(), a.market_id, a.selection_id, a.side, a.price, r.matched_size, r.bet_id or "", a.strategy])

    def _mk_idem_key(self, market_id: str, selection_id: int, side: str, price: float, size: float, strategy: str) -> str:
        base = f"{market_id}:{selection_id}:{side}:{round(price,2)}:{round(size,2)}:{strategy}"
        return base

    def _throttle_wait(self) -> None:
        now = time.time()
        if len(self._bucket) == self._bucket.maxlen:
            # si rempli: attendre jusqu’à ce que la fenêtre de 60s glissante libère une place
            while self._bucket and now - self._bucket[0] < 60.0:
                time.sleep(0.05)
                now = time.time()
        self._bucket.append(now)

    def _sleep_ms(self, ms: int) -> None:
        time.sleep(max(0.0, ms / 1000.0))
