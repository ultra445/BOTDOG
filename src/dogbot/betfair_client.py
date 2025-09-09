import os
import logging
import datetime as dt
from typing import Dict, List, Optional

from betfairlightweight import APIClient
from betfairlightweight.filters import market_filter, price_projection

logger = logging.getLogger(__name__)


class BetfairClient:
    """
    Enveloppe minimale et robuste autour de betfairlightweight pour:
      - Login avec certificats (Windows OK)
      - Scan catalogue (construction d'un index marché -> métadonnées)
      - Lecture des MarketBooks (dict {market_id: MarketBook})
    """

    def __init__(
        self,
        username: str,
        password: str,
        app_key: str,
        certs_dir: Optional[str] = None,
    ):
        # Gestion des certificats :
        # - betfairlightweight attend un "certs" = chemin D'UN DOSSIER
        #   contenant client-2048.crt et client-2048.key
        # - On laisse la lib composer les chemins.
        if certs_dir:
            certs_dir = os.path.normpath(certs_dir)

        self.client = APIClient(
            username=username,
            password=password,
            app_key=app_key,
            certs=certs_dir or None,
            locale="en",
        )

        self._market_index: Dict[str, dict] = {}

        logger.info(
            "Certs: dir='%s' crt='%s' key='%s'",
            certs_dir if certs_dir else "(library default)",
            "client-2048.crt",
            "client-2048.key",
        )

    # ---------- Auth ----------

    def login(self) -> None:
        self.client.login()
        logger.info("Logged in to Betfair API")

    def logout(self) -> None:
        try:
            self.client.logout()
        except Exception:
            pass

    # ---------- Catalogue / Index ----------

    @staticmethod
    def _dt_utc_now() -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc)

    @staticmethod
    def _to_utc(dt_like: Optional[dt.datetime]) -> Optional[dt.datetime]:
        if not dt_like:
            return None
        if dt_like.tzinfo is None:
            # Betfair renvoie parfois naive => on considère que c'est déjà UTC
            return dt_like.replace(tzinfo=dt.timezone.utc)
        return dt_like.astimezone(dt.timezone.utc)

    def scan_catalogue(
        self,
        countries: Optional[List[str]] = None,
        market_types: Optional[List[str]] = None,
        lookahead_minutes: int = 60,
        max_results: int = 1000,
    ) -> List[str]:
        """
        Récupère les marchés (par défaut Greyhound WIN de 0 à lookahead_minutes)
        et construit un index {market_id: infos}.
        Retourne la liste des market_ids.
        """
        countries = countries or ["GB", "IE", "AU", "NZ"]
        market_types = market_types or ["WIN"]

        now_utc = self._dt_utc_now()
        to_utc = now_utc + dt.timedelta(minutes=lookahead_minutes)

        # IMPORTANT: on passe des chaînes ISO "Z" pour éviter tout souci de sérialisation
        time_from = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        time_to = to_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        f = market_filter(
            market_countries=countries,
            market_type_codes=market_types,
            market_start_time={"from": time_from, "to": time_to},
            # event_type_ids=[4339],  # Greyhound Racing (optionnel)
        )

        projection = ["MARKET_START_TIME", "RUNNER_DESCRIPTION", "MARKET_DESCRIPTION", "EVENT"]

        catalogues = self.client.betting.list_market_catalogue(
            filter=f,
            market_projection=projection,
            sort="FIRST_TO_START",
            max_results=max_results,
        )

        self._market_index.clear()
        market_ids: List[str] = []

        start_present = 0
        for cat in catalogues:
            mid = getattr(cat, "market_id", None)
            if not mid:
                continue

            start_utc = self._to_utc(getattr(cat, "market_start_time", None))
            if start_utc:
                start_present += 1

            desc = getattr(cat, "description", None)
            mtype = getattr(desc, "market_type", None)

            event = getattr(cat, "event", None)
            event_id = getattr(event, "id", None)
            event_name = getattr(event, "name", None)
            venue = getattr(event, "venue", None)
            country = getattr(event, "country_code", None)

            runners = getattr(cat, "runners", None)
            n_runners = len(runners) if runners else None

            self._market_index[mid] = {
                "market_id": mid,
                "market_type": mtype,
                "market_start_time_utc": start_utc,
                "event_id": event_id,
                "event_name": event_name,
                "venue": venue,
                "country": country,
                "number_of_runners": n_runners,
            }
            market_ids.append(mid)

        logger.info(
            "Found %d greyhound markets in next %d minutes (start_time present for %d/%d)",
            len(market_ids),
            lookahead_minutes,
            start_present,
            len(market_ids),
        )
        return market_ids

    def get_market_index_entry(self, market_id: str) -> Optional[dict]:
        return self._market_index.get(market_id)

    # ---------- Market books ----------

    def prefetch_market_books(self, market_ids: List[str]) -> Dict[str, object]:
        return self.get_market_books(market_ids)

    def get_market_books(self, market_ids: List[str]) -> Dict[str, object]:
        if not market_ids:
            return {}
        proj = price_projection(price_data=["EX_BEST_OFFERS", "EX_TRADED", "SP_TRADED", "SP_PROJECTED"])
        books = self.client.betting.list_market_book(
            market_ids=market_ids,
            price_projection=proj,
            order_projection=None,
            match_projection=None,
        )
        result: Dict[str, object] = {}
        for b in books:
            try:
                mid = b.market_id
                result[mid] = b
            except Exception:
                continue
        return result

    # ---------- Utilitaires ----------

    def time_to_off_seconds(self, market_id: str) -> Optional[float]:
        """Calcul T-TO-OFF à partir de l'index catalogue (UTC)."""
        entry = self.get_market_index_entry(market_id)
        if not entry:
            return None
        mstart = entry.get("market_start_time_utc")
        if not isinstance(mstart, dt.datetime):
            return None
        return (mstart - self._dt_utc_now()).total_seconds()
