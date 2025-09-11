# src/dogbot/feeds.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _midpoint(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        return (float(a) + float(b)) / 2.0 if (a is not None and b is not None) else None
    except Exception:
        return None


class PriceFeed:
    """
    Interface minimale pour enrichir une ligne runner avec les champs SP (Projected)
    attendus par les CSV snapshots :
      - NEAR_SP_WIN / FAR_SP_WIN / SP_EST_WIN / BSPMOY_WIN / SP_AVAILABLE_WIN
      - NEAR_SP_PLACE / FAR_SP_PLACE / SP_EST_PLACE / BSPMOY_PLACE / SP_AVAILABLE_PLACE
    """

    def enrich_runner_row(
        self,
        row: Dict[str, Any],
        market_type: str,
        runner_obj: Any,
    ) -> None:
        raise NotImplementedError


class RestFeed(PriceFeed):
    """
    Lit les champs depuis listMarketBook (runner.sp.near_price/far_price).
    IMPORTANT : pour recevoir ces valeurs, l'appel REST doit inclure priceProjection SP_AVAILABLE.
    (Si jamais near/far n'arrivent pas, on retombe sur un midpoint back/lay pour SP_EST_*.)
    """

    def enrich_runner_row(
        self,
        row: Dict[str, Any],
        market_type: str,
        runner_obj: Any,
    ) -> None:
        sp = getattr(runner_obj, "sp", None)
        spn = getattr(sp, "near_price", None) if sp is not None else None
        spf = getattr(sp, "far_price", None) if sp is not None else None

        # Choix des colonnes selon WIN/PLACE
        if market_type == "WIN":
            row["NEAR_SP_WIN"] = spn
            row["FAR_SP_WIN"] = spf
            row["SP_EST_WIN"] = spn  # Estimated BSP = near
            row["BSPMOY_WIN"] = (spn + spf) / 2.0 if (spn is not None and spf is not None) else None
            row["SP_AVAILABLE_WIN"] = 1 if (spn is not None or spf is not None) else 0

            # Fallback: si near absent, approx via midpoint best back/lay
            if row.get("SP_EST_WIN") is None:
                row["SP_EST_WIN"] = _midpoint(row.get("BEST_BACK_PRICE_1_WIN"), row.get("BEST_LAY_PRICE_1_WIN"))

        else:  # PLACE
            row["NEAR_SP_PLACE"] = spn
            row["FAR_SP_PLACE"] = spf
            row["SP_EST_PLACE"] = spn
            row["BSPMOY_PLACE"] = (spn + spf) / 2.0 if (spn is not None and spf is not None) else None
            row["SP_AVAILABLE_PLACE"] = 1 if (spn is not None or spf is not None) else 0

            if row.get("SP_EST_PLACE") is None:
                row["SP_EST_PLACE"] = _midpoint(row.get("BEST_BACK_PRICE_1_PLACE"), row.get("BEST_LAY_PRICE_1_PLACE"))


class StreamFeed(PriceFeed):
    """
    Stub pour plus tard : on y mettra les valeurs poussées par le Streaming API (spn/spf).
    L’idée est que l’executor ne change pas : même enrichissement de row avec les mêmes clés.
    """

    def __init__(self, cache_getter=None) -> None:
        # cache_getter: callable (market_id, selection_id) -> dict avec 'spn', 'spf', etc.
        self.cache_getter = cache_getter

    def enrich_runner_row(
        self,
        row: Dict[str, Any],
        market_type: str,
        runner_obj: Any,
    ) -> None:
        # On suppose que row a déjà MARKET_ID/SELECTION_ID mis par l'executor.
        market_id = row.get("MARKET_ID")
        selection_id = row.get("SELECTION_ID")

        spn = None
        spf = None
        if self.cache_getter and market_id and selection_id:
            data = self.cache_getter(market_id, selection_id) or {}
            spn = data.get("spn")
            spf = data.get("spf")

        # On peuple les mêmes colonnes que RestFeed
        if market_type == "WIN":
            row["NEAR_SP_WIN"] = spn
            row["FAR_SP_WIN"] = spf
            row["SP_EST_WIN"] = spn
            row["BSPMOY_WIN"] = (spn + spf) / 2.0 if (spn is not None and spf is not None) else None
            row["SP_AVAILABLE_WIN"] = 1 if (spn is not None or spf is not None) else 0

            if row.get("SP_EST_WIN") is None:
                row["SP_EST_WIN"] = _midpoint(row.get("BEST_BACK_PRICE_1_WIN"), row.get("BEST_LAY_PRICE_1_WIN"))
        else:
            row["NEAR_SP_PLACE"] = spn
            row["FAR_SP_PLACE"] = spf
            row["SP_EST_PLACE"] = spn
            row["BSPMOY_PLACE"] = (spn + spf) / 2.0 if (spn is not None and spf is not None) else None
            row["SP_AVAILABLE_PLACE"] = 1 if (spn is not None or spf is not None) else 0

            if row.get("SP_EST_PLACE") is None:
                row["SP_EST_PLACE"] = _midpoint(row.get("BEST_BACK_PRICE_1_PLACE"), row.get("BEST_LAY_PRICE_1_PLACE"))


def create_price_feed_from_env(stream_cache_getter=None) -> PriceFeed:
    """
    USE_STREAMING=true -> StreamFeed, sinon RestFeed.
    Pour l’instant on utilisera RestFeed; quand tu activeras le stream, on passera USE_STREAMING=true
    et on fournira un cache_getter qui lit spn/spf du listener.
    """
    use_stream = os.getenv("USE_STREAMING", "false").lower() in ("1", "true", "yes", "y")
    if use_stream:
        return StreamFeed(cache_getter=stream_cache_getter)
    return RestFeed()
