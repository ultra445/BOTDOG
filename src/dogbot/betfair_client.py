import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import betfairlightweight
from betfairlightweight import filters
from betfairlightweight.exceptions import APIError

LOG = logging.getLogger(__name__)

# EventTypeId officiel pour greyhounds
GREYHOUND_EVENT_TYPE_ID = "4339"


def _env_list(name: str, default_csv: str) -> List[str]:
    raw = os.getenv(name, default_csv)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    # Format attendu par Betfair "YYYY-MM-DDTHH:MM:SSZ"
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class MarketIndexEntry:
    market_id: str
    event_id: Optional[str]
    market_type: Optional[str]
    start_time: Optional[datetime]
    is_place: bool


class BetfairClient:
    """
    Client Betfair :
      - login via certificats
      - scan des catalogues WIN/PLACE (+ index marché)
      - polling MarketBook (chunking + retry)
      - calcul t_to_off depuis MarketBook ou index

    Paramètres acceptés (alias inclus pour compatibilité avec run.py) :
      - app_key (ou env BETFAIR_APP_KEY)
      - username | user (ou env BETFAIR_USERNAME)
      - password | pwd (ou env BETFAIR_PASSWORD)
      - certs_dir | certs_path (ou env BETFAIR_CERTS_DIR, défaut C:\\betfair-certs)
      - lookahead_minutes (ou env BETFAIR_LOOKAHEAD_MINUTES, défaut 120)
      - countries (ou env BETFAIR_COUNTRIES, défaut "GB,IE")
      - market_book_chunk (ou env BETFAIR_MARKET_BOOK_CHUNK, défaut 5)
    """

    def __init__(
        self,
        app_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        # Aliases de compat
        user: Optional[str] = None,
        pwd: Optional[str] = None,
        certs_dir: Optional[str] = None,
        certs_path: Optional[str] = None,  # alias accepté pour run.py existant
        lookahead_minutes: Optional[int] = None,
        countries: Optional[List[str]] = None,
        market_book_chunk: Optional[int] = None,
        **_ignore,  # avale tout autre kwarg inattendu pour éviter les plantages
    ):
        self.app_key = app_key or os.getenv("BETFAIR_APP_KEY")
        self.username = username or user or os.getenv("BETFAIR_USERNAME")
        self.password = password or pwd or os.getenv("BETFAIR_PASSWORD")

        # Choix du dossier certificats : priorité à certs_path si fourni par run.py
        self.certs_dir = (
            certs_path
            or certs_dir
            or os.getenv("BETFAIR_CERTS_DIR")
            or r"C:\betfair-certs"
        )

        if not (self.app_key and self.username and self.password):
            raise RuntimeError(
                "Identifiants Betfair manquants : fournissez app_key / username / password "
                "(ou variables BETFAIR_APP_KEY / BETFAIR_USERNAME / BETFAIR_PASSWORD)."
            )

        self.lookahead_minutes = int(lookahead_minutes or os.getenv("BETFAIR_LOOKAHEAD_MINUTES", "120"))
        self.countries = countries or _env_list("BETFAIR_COUNTRIES", "GB,IE")

        self.market_book_chunk = int(market_book_chunk or os.getenv("BETFAIR_MARKET_BOOK_CHUNK", "5"))
        self.market_book_chunk = max(1, self.market_book_chunk)

        # betfairlightweight client
        self.client = betfairlightweight.APIClient(
            username=self.username,
            password=self.password,
            app_key=self.app_key,
            certs=self.certs_dir,
        )

        # Index des marchés scannés
        self.market_index: Dict[str, MarketIndexEntry] = {}

    # ------------------------- Auth ------------------------- #
    def login(self) -> None:
        LOG.debug("Login Betfair (certs_dir=%s)", self.certs_dir)
        self.client.login()
        LOG.info("Logged in to Betfair API")

    # -------------------- Catalogue & index ----------------- #
    def scan_catalogue(
        self,
        lookahead_minutes: Optional[int] = None,
        market_types: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        max_results: int = 200,
    ) -> List[str]:
        mins = int(lookahead_minutes or self.lookahead_minutes)
        country_list = countries or self.countries
        types = market_types or ["WIN", "PLACE"]

        now = _utc_now()
        time_from = _iso_utc(now)
        time_to = _iso_utc(now + timedelta(minutes=mins))

        market_filter = {
            "eventTypeIds": [GREYHOUND_EVENT_TYPE_ID],
            "marketTypeCodes": types,
            "marketCountries": country_list,
            "marketStartTime": {"from": time_from, "to": time_to},
        }

        projection = ["MARKET_START_TIME", "RUNNER_METADATA", "EVENT", "MARKET_DESCRIPTION"]

        catalogues = self._call_with_retry(
            self.client.betting.list_market_catalogue,
            filter=market_filter,
            market_projection=projection,
            max_results=max_results,
            sort="FIRST_TO_START",
        )

        found = 0
        with_start = 0
        self.market_index.clear()

        for cat in catalogues or []:
            found += 1
            market_id = cat.market_id
            event_id = getattr(cat.event, "id", None)

            # ✅ FIX: le type de marché est dans cat.description.market_type
            mtype = None
            try:
                desc = getattr(cat, "description", None)
                if desc is not None:
                    mtype = getattr(desc, "market_type", None)
            except Exception:
                mtype = None

            # Fallback si description absente : on infère via le nom du marché
            if not mtype:
                name = (getattr(cat, "market_name", None) or "").lower()
                if "place" in name or "to be placed" in name:
                    mtype = "PLACE"
                elif "win" in name:
                    mtype = "WIN"

            start_time = getattr(cat, "market_start_time", None)
            if start_time:
                with_start += 1

            is_place = (str(mtype).upper() == "PLACE") if mtype else False

            self.market_index[market_id] = MarketIndexEntry(
                market_id=market_id,
                event_id=event_id,
                market_type=mtype,
                start_time=start_time,
                is_place=is_place,
            )

        LOG.info(
            "Found %d greyhound markets in next %d minutes (start_time present for %d/%d)",
            found, mins, with_start, found,
        )
        return list(self.market_index.keys())

    def get_market_index_entry(self, market_id: str) -> Optional[MarketIndexEntry]:
        return self.market_index.get(market_id)

    def get_time_to_off(self, market_id: str, book=None) -> Optional[int]:
        """
        Retourne le temps jusqu’au départ (secondes).
        Priorité aux infos du MarketBook (market_definition.market_time),
        sinon retombe sur l’index catalogue (start_time).
        """
        try:
            if book and getattr(book, "market_definition", None):
                mdef = book.market_definition
                mt = getattr(mdef, "market_time", None)
                if isinstance(mt, datetime):
                    return int((mt.astimezone(timezone.utc) - _utc_now()).total_seconds())
        except Exception:
            pass

        entry = self.market_index.get(market_id)
        if entry and isinstance(entry.start_time, datetime):
            return int((entry.start_time.astimezone(timezone.utc) - _utc_now()).total_seconds())
        return None

    # ------------------- MarketBook polling ----------------- #
    def poll_market_books(self, market_ids: List[str]):
        """Récupère les MarketBooks en chunks avec retry."""
        if not market_ids:
            return []

        pp = filters.price_projection(
            price_data=["EX_BEST_OFFERS", "EX_TRADED", "SP_AVAILABLE", "SP_TRADED"],
            virtualise=True,
            rollover_stakes=False,
        )

        all_books = []
        chunk = max(1, self.market_book_chunk)

        for i in range(0, len(market_ids), chunk):
            part = market_ids[i : i + chunk]
            try:
                books = self._call_with_retry(
                    self.client.betting.list_market_book,
                    market_ids=part,
                    price_projection=pp,
                )
                if books:
                    all_books.extend(books)
            except APIError as e:
                LOG.warning("list_market_book chunk error on %d ids: %s", len(part), e)
            except Exception as e:
                LOG.warning("list_market_book chunk unexpected error on %d ids: %s", len(part), e)

        return all_books

    def prefetch_market_books(self, market_ids: List[str]) -> None:
        """Appel de warm-up, échec toléré."""
        if not market_ids:
            return
        try:
            self.poll_market_books(market_ids)
        except Exception as e:
            LOG.debug("prefetch_market_books ignored error: %s", e)

    # ------------------------ Retry util -------------------- #
    def _call_with_retry(self, fn, attempts: int = 3, **kwargs):
        last_err = None
        for k in range(1, attempts + 1):
            try:
                return fn(**kwargs)
            except Exception as e:
                last_err = e
                LOG.warning(
                    "%s error '%s' on attempt %d → retrying…",
                    getattr(fn, "__name__", "call"),
                    e,
                    k,
                )
        if last_err:
            raise last_err
