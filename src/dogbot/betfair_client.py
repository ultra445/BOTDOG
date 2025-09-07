from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

import betfairlightweight
from betfairlightweight import filters as bf_filters
from betfairlightweight.resources.bettingresources import MarketBook, MarketCatalogue

logger = logging.getLogger(__name__)


# ---------- Dataclasses d'index ----------
@dataclass
class EventIndexEntry:
    event_id: str
    venue: Optional[str]
    country_code: Optional[str]
    open_date: Optional[datetime]      # UTC
    course_uuid: str                   # = event_id


@dataclass
class MarketIndexEntry:
    market_id: str
    market_type: Optional[str]         # "WIN", "PLACE", etc.
    event_id: str
    start_time: Optional[datetime]     # UTC
    venue: Optional[str]
    country_code: Optional[str]
    runner_count: Optional[int]
    course_uuid: str                   # = event_id


class BetfairClient:
    """
    Wrapper léger autour de betfairlightweight avec :
      - login
      - scan du catalogue (et construction d'index marché/événement)
      - polling de MarketBook en chunks
      - utilitaires : t_to_off, accès aux index
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        app_key: Optional[str] = None,
        cert_dir: Optional[str] = None,
        books_chunk: Optional[int] = None,
        best_price_depth: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Rend le client compatible avec les anciens appels comme:
          BetfairClient(user=..., password=..., app_key=..., certs=..., books_chunk=8)
        Tout est optionnel : si absent, on lit l'environnement.
        """
        # aliases possibles depuis run.py
        if username is None:
            username = kwargs.get("user") or os.getenv("BETFAIR_USERNAME", "")
        if password is None:
            password = os.getenv("BETFAIR_PASSWORD", "")
        if app_key is None:
            app_key = kwargs.get("appKey") or os.getenv("BETFAIR_APP_KEY", "")
        if cert_dir is None:
            cert_dir = kwargs.get("certs") or os.getenv("BETFAIR_CERT_DIR", "")

        # limites requêtes
        if books_chunk is None:
            try:
                books_chunk = int(os.getenv("BETFAIR_BOOKS_CHUNK", "8"))
            except Exception:
                books_chunk = 8
        if best_price_depth is None:
            try:
                best_price_depth = int(os.getenv("BETFAIR_PRICE_DEPTH", "3"))
            except Exception:
                best_price_depth = 3

        self.username = username or ""
        self.password = password or ""
        self.app_key = app_key or ""
        self.cert_dir = cert_dir or ""
        self.books_chunk = max(1, int(books_chunk))
        self.best_price_depth = max(1, int(best_price_depth))

        # Index internes
        self.market_index: Dict[str, MarketIndexEntry] = {}
        self.event_index: Dict[str, EventIndexEntry] = {}

        # Client BFLW
        if self.cert_dir:
            self.client = betfairlightweight.APIClient(
                username=self.username,
                password=self.password,
                app_key=self.app_key,
                certs=self.cert_dir,
            )
        else:
            self.client = betfairlightweight.APIClient(
                username=self.username,
                password=self.password,
                app_key=self.app_key,
            )

    # ---------- Auth ----------
    def login(self) -> None:
        self.client.login()
        logger.info("Logged in to Betfair API")

    # ---------- Utilitaires datetime ----------
    @staticmethod
    def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
        if dt is None:
            return None
        if dt.tzinfo is None:
            # on assume UTC si naïf
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    # ---------- Scan catalogue ----------
    def scan_catalogue(
        self,
        countries: Optional[Sequence[str]] = None,
        market_types: Optional[Sequence[str]] = None,
        lookahead_minutes: int = 10,
        event_type_ids: Optional[Sequence[str]] = None,  # 4339 greyhounds, 7 horse racing
        max_results: int = 200,
    ) -> List[str]:
        """
        Récupère les marchés à venir et construit les index.
        Retourne la liste des market_ids.
        """
        countries = countries or ["GB", "IE"]
        market_types = market_types or ["WIN", "PLACE"]
        event_type_ids = event_type_ids or ["4339"]  # greyhounds par défaut

        frm = self._now_utc()
        to = frm + timedelta(minutes=lookahead_minutes)

        market_filter = bf_filters.market_filter(
            event_type_ids=list(event_type_ids),
            market_countries=list(countries),
            market_type_codes=list(market_types),
            market_start_time=bf_filters.time_range(from_=frm, to=to),
        )

        market_projection = [
            "EVENT",
            "MARKET_START_TIME",
            "RUNNER_DESCRIPTION",
            "MARKET_DESCRIPTION",
        ]

        catalogues: List[MarketCatalogue] = self.client.betting.list_market_catalogue(
            filter=market_filter,
            market_projection=market_projection,
            max_results=max_results,
        )

        # Reset minimal des index (on reconstruit)
        self.market_index.clear()
        self.event_index.clear()

        start_present = 0
        market_ids: List[str] = []

        for mc in catalogues:
            market_ids.append(mc.market_id)

            # Event
            ev = mc.event
            event_id = getattr(ev, "id", None)
            venue = getattr(ev, "venue", None)
            country_code = getattr(ev, "country_code", None)
            open_date = self._to_utc(getattr(ev, "open_date", None))
            if event_id and event_id not in self.event_index:
                self.event_index[event_id] = EventIndexEntry(
                    event_id=event_id,
                    venue=venue,
                    country_code=country_code,
                    open_date=open_date,
                    course_uuid=event_id,
                )

            # Market
            mdesc = getattr(mc, "description", None)
            market_type = getattr(mdesc, "market_type", None)
            start_time = self._to_utc(getattr(mc, "market_start_time", None))
            if start_time:
                start_present += 1

            self.market_index[mc.market_id] = MarketIndexEntry(
                market_id=mc.market_id,
                market_type=market_type,
                event_id=event_id or "",
                start_time=start_time,
                venue=venue,
                country_code=country_code,
                runner_count=len(getattr(mc, "runners", []) or []),
                course_uuid=(event_id or ""),
            )

        logger.info(
            f"Found {len(market_ids)} greyhound markets in next {lookahead_minutes} minutes "
            f"(start_time present for {start_present}/{len(market_ids)})"
        )
        return market_ids

    # ---------- Poll MarketBook ----------
    def poll_market_books(self, market_ids: Sequence[str]) -> List[MarketBook]:
        """
        Récupère les MarketBooks en chunks pour éviter TOO_MUCH_DATA.
        """
        if not market_ids:
            return []

        all_books: List[MarketBook] = []
        chunk = max(1, int(self.books_chunk))
        depth = max(1, int(self.best_price_depth))

        price_proj = bf_filters.price_projection(
            price_data=["EX_BEST_OFFERS"],  # volontairement léger
            virtualise=True,
            ex_best_offers_overrides=bf_filters.ex_best_offers_overrides(
                best_prices_depth=depth
            ),
        )

        for i in range(0, len(market_ids), chunk):
            subset = market_ids[i : i + chunk]
            try:
                books = self.client.betting.list_market_book(
                    market_ids=list(subset),
                    price_projection=price_proj,
                    order_projection=None,
                    match_projection=None,
                )
                if books:
                    all_books.extend(books)
            except Exception as e:
                # log et on continue (le poll suivant réessaiera)
                logger.warning(
                    f"list_market_book chunk error on {len(subset)} ids: {e!r}"
                )
                time.sleep(0.25)

        return all_books

    # ---------- Utilitaires index ----------
    def get_market_index_entry(self, market_id: str) -> Optional[MarketIndexEntry]:
        return self.market_index.get(market_id)

    def get_event_index_entry(self, event_id: str) -> Optional[EventIndexEntry]:
        return self.event_index.get(event_id)

    def get_course_uuid_for_market(self, market_id: str) -> Optional[str]:
        idx = self.get_market_index_entry(market_id)
        return idx.course_uuid if idx else None

    # ---------- t_to_off ----------
    def get_time_to_off(self, market_id: str, book: Optional[MarketBook]) -> Optional[int]:
        """
        Calcule t_to_off (secondes) depuis, par ordre de priorité :
          1) book.market_definition.market_time
          2) index.market.start_time (scan catalogue)
        """
        start_dt: Optional[datetime] = None

        # 1) market_definition.market_time du MarketBook
        try:
            if book and getattr(book, "market_definition", None):
                md = book.market_definition
                start_dt = getattr(md, "market_time", None)
                start_dt = self._to_utc(start_dt)
        except Exception:
            start_dt = None

        # 2) fallback via l'index
        if start_dt is None:
            idx = self.get_market_index_entry(market_id)
            if idx and idx.start_time:
                start_dt = idx.start_time

        if start_dt is None:
            return None

        now = self._now_utc()
        return int(round((start_dt - now).total_seconds()))
