# src/dogbot/betfair_client.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from loguru import logger
import betfairlightweight as bflw
from betfairlightweight.resources import MarketBook, MarketCatalogue


# --- Helpers ---------------------------------------------------------------

def _absnorm(path: str) -> str:
    path = path.strip().strip('"').strip("'")
    return os.path.abspath(os.path.normpath(path))


def _chunked(seq: Sequence[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), n):
        yield list(seq[i : i + n])


def _to_iso_z(dt: datetime) -> str:
    """Format datetime UTC en ISO 8601 se terminant par 'Z' (sans microsecondes)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    # remove microseconds for cleaner payloads
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class MarketIndexEntry:
    market_id: str
    market_type: str | None
    event_id: str | None
    event_name: str | None
    country: str | None
    venue: str | None
    start_time: datetime | None
    course_uid: str | None   # identifiant commun WIN/PLACE (simple, robuste)


# --- Client ----------------------------------------------------------------

class BetfairClient:
    """
    Enveloppe autour de betfairlightweight qui:
      - gère proprement les certificats (Windows OK, versions anciennes de la lib)
      - expose des méthodes simples utilisées par run.py / executor.py
    """

    def __init__(
        self,
        username: str,
        password: str,
        app_key: str,
        certs_path: str | None = None,
        *,
        list_book_chunk: int = 20,  # 25 max conseillé par l'API
    ) -> None:
        self.username = username
        self.password = password
        self.app_key = app_key

        # 1) Résolution du dossier certs (priorité: argument > env > ./certs)
        env_dir = os.getenv("BETFAIR_CERTS_PATH") or os.getenv("BF_CERTS_PATH")
        base_dir = certs_path or env_dir or os.path.join(os.getcwd(), "certs")
        self._certs_dir = _absnorm(base_dir)

        # 2) Vérification .crt/.key (la lib veut un DOSSIER qui contient les fichiers)
        if not os.path.isdir(self._certs_dir):
            raise FileNotFoundError(f"Dossier des certificats introuvable: {self._certs_dir}")

        files = os.listdir(self._certs_dir)
        has_crt = any(f.lower().endswith(".crt") for f in files)
        has_key = any(f.lower().endswith(".key") for f in files)
        if not (has_crt and has_key):
            raise FileNotFoundError(
                f"Aucun couple .crt/.key trouvé dans {self._certs_dir}. "
                f"Assure-toi d'y mettre tes fichiers client-2048.crt et client-2048.key (ou équivalents)."
            )

        crt_name = next((f for f in files if f.lower().endswith(".crt")), "?")
        key_name = next((f for f in files if f.lower().endswith(".key")), "?")
        logger.info("Certs: dir='{}' crt='{}' key='{}'", self._certs_dir, crt_name, key_name)

        # 3) Création du client — ta version attend un CHEMIN DE DOSSIER dans 'certs'
        self.client = bflw.APIClient(
            username=self.username,
            password=self.password,
            app_key=self.app_key,
            certs=self._certs_dir,  # dossier, pas (crt, key)
        )

        self._list_book_chunk = int(list_book_chunk)
        self._index: dict[str, MarketIndexEntry] = {}

    # --- Auth --------------------------------------------------------------

    def login(self) -> None:
        self.client.login()
        logger.info("Logged in to Betfair API")

    # --- Catalogue ---------------------------------------------------------

    def scan_catalogue(
        self,
        *,
        countries: list[str],
        market_types: list[str],
        lookahead_minutes: int,
    ) -> list[str]:
        """
        Retourne la liste des marketIds (WIN/PLACE) dans [now, now+lookahead].
        Remplit aussi un index (MarketIndexEntry) exploitable par l'executor.
        """
        now_utc = datetime.now(timezone.utc)
        to_utc = now_utc + timedelta(minutes=int(lookahead_minutes))

        # IMPORTANT: convertir en chaînes ISO pour éviter "datetime is not JSON serializable"
        start_filter = {
            "from": _to_iso_z(now_utc),
            "to": _to_iso_z(to_utc),
        }

        flt = bflw.filters.market_filter(
            market_countries=countries or None,
            market_type_codes=market_types or None,
            market_start_time=start_filter,
        )
        projections = [
            "MARKET_START_TIME",
            "RUNNER_DESCRIPTION",
            "MARKET_DESCRIPTION",
            "EVENT",
        ]

        catalogues: list[MarketCatalogue] = self.client.betting.list_market_catalogue(
            filter=flt,
            market_projection=projections,
            sort="FIRST_TO_START",
            max_results=1000,
        )

        self._index.clear()
        market_ids: list[str] = []
        for cat in catalogues:
            mid = cat.market_id
            market_ids.append(mid)

            start = getattr(cat, "market_start_time", None)
            event = getattr(cat, "event", None)
            evt_id = getattr(event, "id", None)
            evt_name = getattr(event, "name", None)
            country = getattr(event, "country_code", None)
            venue = getattr(cat, "market_name", None)

            mtype = getattr(cat, "market_type", None)
            if mtype is None:
                desc = getattr(cat, "description", None)
                mtype = getattr(desc, "market_type", None)

            if isinstance(start, datetime):
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                course_uid = f"{evt_id}@{start.strftime('%Y%m%dT%H%M')}"
            else:
                course_uid = None

            self._index[mid] = MarketIndexEntry(
                market_id=mid,
                market_type=mtype,
                event_id=evt_id,
                event_name=evt_name,
                country=country,
                venue=venue,
                start_time=start,
                course_uid=course_uid,
            )

        logger.info(
            "Found {} greyhound markets in next {} minutes (start_time present for {}/{})",
            len(market_ids),
            lookahead_minutes,
            sum(1 for e in self._index.values() if isinstance(e.start_time, datetime)),
            len(market_ids),
        )
        return market_ids

    # --- Books -------------------------------------------------------------

    def get_market_books(self, market_ids: list[str]) -> dict[str, MarketBook]:
        """
        Récupère les MarketBook pour les market_ids donnés (dict {mid: MarketBook}).
        """
        if not market_ids:
            return {}

        price_projection = bflw.filters.price_projection(
            price_data=["EX_BEST_OFFERS"]  # léger pour le polling
        )

        out: dict[str, MarketBook] = {}
        for chunk in _chunked(market_ids, self._list_book_chunk):
            try:
                books: list[MarketBook] = self.client.betting.list_market_book(
                    market_ids=chunk,
                    price_projection=price_projection,
                    order_projection=None,
                    match_projection=None,
                )
                for b in books:
                    out[b.market_id] = b
            except Exception as e:
                logger.warning(
                    "list_market_book chunk error on {} ids: {}", len(chunk), e
                )
        return out

    # Alias backward-compat
    def prefetch_market_books(self, market_ids: list[str]) -> dict[str, MarketBook]:
        return self.get_market_books(market_ids)

    # --- Infos utiles à l’executor ----------------------------------------

    def get_market_index_entry(self, market_id: str) -> MarketIndexEntry | None:
        return self._index.get(market_id)

    def get_time_to_off(self, book: MarketBook) -> float | None:
        """
        Calcule T-to-OFF en secondes depuis le MarketBook.
        Essaye marketDefinition.marketTime, sinon retombe sur l’index.
        """
        start: datetime | None = None

        mdef = getattr(book, "market_definition", None)
        if mdef is not None:
            start = getattr(mdef, "market_time", None)

        if start is None:
            entry = self._index.get(book.market_id)
            if entry:
                start = entry.start_time

        if not isinstance(start, datetime):
            return None

        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        return (start - now).total_seconds()
