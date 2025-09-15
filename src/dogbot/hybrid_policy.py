from __future__ import annotations

import json, os
from typing import Dict, Any
from .strategies import ExecMode, LimitStyle
from .staking import Side

DEFAULT_POLICY = {
    "default": "LIMIT_LTP",
    "limit_style": "AGGRESSIVE",
    "rules": [
        # Exemples :
        # {"if": {"side":"BACK","market_type":"WIN","mom45_ge":0.02}, "then":"SP_MOC"},
        # {"if": {"side":"LAY","market_type":"WIN","mom45_le":-0.01}, "then":"SP_LOC", "sp_limit_mult":1.05},
        # {"if": {"side":"BACK","market_type":"PLACE","mom45p_ge":0.00,"mom45p_le":0.10}, "then":"LIMIT_LTP", "limit_price":"CROSS"},
        # {"if": {"side":"BACK","market_type":"WIN","d5_ge":0.01}, "then":"LIMIT_LTP", "limit_price":"CROSS"},
        # {"if": {"side":"LAY","market_type":"WIN","vol_ge":0.05}, "then":"SP_MOC"}
    ]
}

def _load_policy() -> Dict[str, Any]:
    """
    Charge la policy depuis HYB_POLICY_PATH (ou ./config/hybrid_policy.json).
    Si introuvable, retourne DEFAULT_POLICY.
    """
    path = os.getenv("HYB_POLICY_PATH", "./config/hybrid_policy.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_POLICY.copy()

def _match(ctx, cond: Dict[str, Any]) -> bool:
    """
    Clés supportées dans cond:
      - market_type: "WIN" | "PLACE"
      - mom45_ge / mom45_le         (WIN;   ctx.mom45)
      - mom45p_ge / mom45p_le       (PLACE; ctx.mom45_place)
      - d5_ge / d5_le               (WIN;   ctx.d5)
      - d30_ge / d30_le             (WIN;   ctx.d30)
      - vol_ge / vol_le             (WIN;   ctx.vol60)
      - fav_rank_ltp_eq / _ge / _le
      - secs_to_off_ge / _le
    Le filtre 'side' est géré en amont dans choose_action (via slot.side).
    """
    def ge(val, thr):
        return (val is not None) and (val >= thr)
    def le(val, thr):
        return (val is not None) and (val <= thr)

    # market type
    if "market_type" in cond and str(ctx.market_type).upper() != str(cond["market_type"]).upper():
        return False

    # WIN momentum (BASE_WIN 45s -> 2s)
    if "mom45_ge" in cond and not ge(getattr(ctx, "mom45", None), float(cond["mom45_ge"])):
        return False
    if "mom45_le" in cond and not le(getattr(ctx, "mom45", None), float(cond["mom45_le"])):
        return False

    # PLACE momentum (LTP 45s -> 2s)
    if "mom45p_ge" in cond and not ge(getattr(ctx, "mom45_place", None), float(cond["mom45p_ge"])):
        return False
    if "mom45p_le" in cond and not le(getattr(ctx, "mom45_place", None), float(cond["mom45p_le"])):
        return False

    # Micro-momentum & volatilité (WIN)
    if "d5_ge" in cond and not ge(getattr(ctx, "d5", None), float(cond["d5_ge"])):
        return False
    if "d5_le" in cond and not le(getattr(ctx, "d5", None), float(cond["d5_le"])):
        return False
    if "d30_ge" in cond and not ge(getattr(ctx, "d30", None), float(cond["d30_ge"])):
        return False
    if "d30_le" in cond and not le(getattr(ctx, "d30", None), float(cond["d30_le"])):
        return False
    if "vol_ge" in cond and not ge(getattr(ctx, "vol60", None), float(cond["vol_ge"])):
        return False
    if "vol_le" in cond and not le(getattr(ctx, "vol60", None), float(cond["vol_le"])):
        return False

    # Fav rank LTP
    if "fav_rank_ltp_eq" in cond:
        if ctx.fav_rank_ltp is None or int(ctx.fav_rank_ltp) != int(cond["fav_rank_ltp_eq"]):
            return False
    if "fav_rank_ltp_ge" in cond and not ge(None if ctx.fav_rank_ltp is None else float(ctx.fav_rank_ltp), float(cond["fav_rank_ltp_ge"])):
        return False
    if "fav_rank_ltp_le" in cond and not le(None if ctx.fav_rank_ltp is None else float(ctx.fav_rank_ltp), float(cond["fav_rank_ltp_le"])):
        return False

    # time to off
    if "secs_to_off_le" in cond and not le(getattr(ctx, "secs_to_off", None), float(cond["secs_to_off_le"])):  # noqa
        return False
    if "secs_to_off_ge" in cond and not ge(getattr(ctx, "secs_to_off", None), float(cond["secs_to_off_ge"])):  # noqa
        return False

    return True

def choose_action(ctx, slot) -> Dict[str, Any]:
    """
    Retourne un dict:
      {
        "mode": ExecMode,
        "limit_style": LimitStyle,
        "sp_limit": float|None,
        "limit_price": "CROSS"|"MID"|"OWN"|None
      }
    - CROSS: BACK au best LAY ; LAY au best BACK (cross du spread)
    - MID  : (bb+bl)/2
    - OWN  : BACK au best BACK ; LAY au best LAY (poster côté nôtre)
    """
    pol = _load_policy()
    default_mode = str(pol.get("default", "LIMIT_LTP")).upper()
    default_style = str(pol.get("limit_style", "AGGRESSIVE")).upper()

    # normalize defaults
    try:
        def_mode = ExecMode[default_mode]
    except Exception:
        def_mode = ExecMode.LIMIT_LTP
    try:
        def_style = LimitStyle[default_style]
    except Exception:
        def_style = LimitStyle.AGGRESSIVE

    # Scan des règles
    for rule in pol.get("rules", []):
        cond = rule.get("if", {})
        # 'side' : filtré par rapport au slot courant
        if "side" in cond:
            try:
                if Side(str(cond["side"]).upper()) != slot.side:
                    continue
            except Exception:
                continue

        if _match(ctx, cond):
            # Action
            then = str(rule.get("then", default_mode)).upper()
            try:
                mode = ExecMode[then]
            except Exception:
                mode = def_mode

            # Limit style (optionnel dans la règle, sinon défaut global)
            style = def_style
            if "limit_style" in rule:
                try:
                    style = LimitStyle[str(rule["limit_style"]).upper()]
                except Exception:
                    pass

            # SP_LOC options (sp_limit fixe ou via multiplicateur)
            sp_limit = None
            if mode == ExecMode.SP_LOC:
                if "sp_limit" in rule:
                    try:
                        sp_limit = float(rule["sp_limit"])
                    except Exception:
                        sp_limit = None
                elif "sp_limit_mult" in rule:
                    try:
                        mult = float(rule["sp_limit_mult"])
                        price_ref = getattr(ctx, "base_win", None) or getattr(ctx, "ltp", None)
                        if price_ref is not None:
                            sp_limit = price_ref * mult
                    except Exception:
                        sp_limit = None

            # Prix limite demandé par la règle (CROSS/MID/OWN)
            limit_price = rule.get("limit_price")
            out = {"mode": mode, "limit_style": style, "sp_limit": sp_limit}
            if isinstance(limit_price, str):
                out["limit_price"] = limit_price.upper()

            return out

    # Fallback: default
    return {"mode": def_mode, "limit_style": def_style, "sp_limit": None}
