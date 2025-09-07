from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import betfairlightweight as bflw
from loguru import logger


@dataclass
class MarketIndexEntry:
    market_id: str
    race_id: str
    start_time: Optional[datetime]
    market_type: Optional[str]
    number_of_winners: int = 1


class BetfairClient:
    """
    Minimise les changements : on garde la logique existante
    et on ajoute les utilitaires attendus par l'executor.
    """

    def __init__(
        self,
        app_key: str,
        username: str,
        password: str,
        certs_path: Optional[str] = None,
        locale: str = "en",
    ) -> None:
        self.app_key = app_key
        self.username = username
        self.password = password
        self.certs_path = certs_path
        self.locale = locale

        # betfairlightweight client
        self.client = bflw.APIClient(
            username=self.username,
            password=self.password,
            app_key=self.app_key,
            certs=self.certs_path,
            locale=self.locale,
        )

        # Index {market_id -> MarketIndexEntry}
        self.market_index: Dict[str, MarketIndexEntry] = {}

    # ---------------------------------------------------------------------
    # Auth
    # ---------------------------------------------------------------------
    def login(self) -> None:
        self.client.login()
        logger.info("Logged in to Betfair API")

    # ---------------------------------------------------------------------
    # Catalogue
    # ---------------------------------------------------------------------
    def scan_catalogue(
        self,
        countries: Iterable[str],
        market_types: Iterable[str],
        lookahead_minutes: int = 10,
        max_results: int = 200,
    ) -> List[str]:
        """
        Scanne les marchés et alimente self.market_index.
        Retourne la liste des market_ids (filtrés par market_types).
        """
        now_utc = datetime.now(timezone.utc)
        to_utc = now_utc.replace(microsecond=0)  # keep tz-aware
        # betfairlightweight attend des timestamps; on laisse le filtre faire par défaut (FIRST_TO_START)

        # On s'appuie sur l'endpoint haut-niveau; pas de custom JSON ici.
        cats = self.client.betting.list_market_catalogue(
            filter={
                "eventTypeIds": ["4339"],  # Greyhound Racing
                "marketCountries": list(countries),
                "marketTypeCodes": list(market_types),
            },
            market_projection=[
                "MARKET_START_TIME",
                "RUNNER_DESCRIPTION",
                "EVENT",
                "COMPETITION",
                "MARKET_DESCRIPTION",
            ],
            sort="FIRST_TO_START",
            max_results=max_results,
        )

        market_ids: List[str] = []
        with_start = 0

        for cat in cats or []:
            mid = getattr(cat, "market_id", None) or getattr(cat, "marketId", None)
            if not mid:
                continue

            # start_time
            start_time: Optional[datetime] = getattr(cat, "market_start_time", None)
            if start_time and start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            if start_time:
                with_start += 1

            # race_id : on privilégie l'id d'event s'il est dispo
            event = getattr(cat, "event", None)
            venue = getattr(event, "venue", None)
            event_id = getattr(event, "id", None)

            if event_id:
                race_id = str(event_id)
            else:
                # fallback stable : venue + horodatage
                stamp = start_time.astimezone(timezone.utc).strftime("%Y%m%d%H%M") if start_time else "unknown"
                race_id = f"{(venue or 'UNKNOWN').strip()}_{stamp}"

            # market_type
            mtype = getattr(cat, "market_type", None) or getattr(cat, "marketName", None)

            # nb de places gagnantes (si dispo)
            desc = getattr(cat, "description", None)
            nwin = getattr(desc, "number_of_winners", None)
            if not isinstance(nwin, int) or nwin <= 0:
                nwin = 1

            self.market_index[mid] = MarketIndexEntry(
                market_id=mid,
                race_id=race_id,
                start_time=start_time,
                market_type=str(mtype) if mtype else None,
                number_of_winners=nwin,
            )

            # On ne renvoie que ce qui correspond au market_types demandés
            # (Betfair a déjà filtré, mais on reste défensif)
            if mtype and str(mtype).upper() in set(mt.upper() for mt in market_types):
                market_ids.append(mid)

        logger.info(
            "Found {} greyhound markets in next {} minutes (start_time present for {}/{})",
            len(market_ids),
            lookahead_minutes,
            with_start,
            len(market_ids),
        )
        return market_ids

    # ---------------------------------------------------------------------
    # Books
    # ---------------------------------------------------------------------
    def poll_market_books(self, market_ids: List[str]) -> Dict[str, object]:
        """
        Récupère les MarketBook par paquets et **retourne toujours un dict**
        {market_id: MarketBook} pour correspondre à l'executor.
        """
        if not market_ids:
            return {}

        out: Dict[str, object] = {}
        chunk = 25  # confortable sous les limites
        for i in range(0, len(market_ids), chunk):
            part = market_ids[i : i + chunk]
            books = self.client.betting.list_market_book(
                market_ids=part,
                price_projection={"priceData": ["EX_BEST_OFFERS"]},
                order_projection=None,
                match_projection=None,
            )
            for b in books or []:
                mid = getattr(b, "market_id", None) or getattr(b, "marketId", None)
                if mid:
                    out[mid] = b

        return out

    # ---------------------------------------------------------------------
    # Utilitaires attendus par l'executor
    # ---------------------------------------------------------------------
    def get_market_index_entry(self, market_id: str) -> Optional[MarketIndexEntry]:
        return self.market_index.get(market_id)

    def get_time_to_off(self, book: object, fallback_start: Optional[datetime]) -> Optional[int]:
        """
        Renvoie le t_to_off en secondes (peut être négatif si déjà parti),
        ou None si on ne sait pas.
        """
        market_time: Optional[datetime] = None

        # market_definition.market_time si dispo dans le book
        mdef = getattr(book, "market_definition", None)
        if mdef is None:
            # certains wrappers utilisent camelCase
            mdef = getattr(book, "marketDefinition", None)
        if mdef is not None:
            market_time = getattr(mdef, "market_time", None) or getattr(mdef, "marketTime", None)

        start_time = market_time or fallback_start
        if not start_time:
            return None

        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return int((start_time - now).total_seconds())
