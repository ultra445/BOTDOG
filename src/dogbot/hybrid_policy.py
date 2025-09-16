from __future__ import annotations
import json, os
from dataclasses import dataclass
from typing import Any, Dict, Optional

CONFIG_PATH = os.environ.get("HYBRID_POLICY_PATH", "./config/hybrid_policy.json")

def _load_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"default": "LIMIT_LTP", "limit_style": "AGGRESSIVE", "rules": []}

def _eq(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None: return False
    return str(a).upper() == str(b).upper()

def _num(val) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None

def _effective_mom(ctx) -> Optional[float]:
    # Preferred source: WIN momentum 'mom45' (even when market_type == PLACE)
    for name in ("mom45", "mom45_win", "mom_win45", "mom45p"):  # last is fallback
        if hasattr(ctx, name):
            v = getattr(ctx, name)
            try:
                return float(v) if v is not None else None
            except Exception:
                continue
    return None

\
        def _match(ctx, cond: Dict[str, Any]) -> bool:
    # side
    if "side" in cond and not _eq(cond["side"], getattr(ctx, "side", None)):
        return False
    # market_type
    if "market_type" in cond and not _eq(cond["market_type"], getattr(ctx, "market_type", None)):
        return False
    # region ("UK" or "ROW")
            if "region" in cond:
                want = str(cond["region"]).upper()
                have = str(getattr(ctx, "region", "") or "").upper()
                if want and have and want != have:
                    return False

    mom = _effective_mom(ctx)

    # Numeric range checks (we accept mom45_* and mom45p_* but map both to mom)
    for key, val in cond.items():
        if key.endswith("_ge"):
            if key.startswith("mom45") and mom is not None:
                if not (mom >= _num(val)):
                    return False
            else:
                pass
        if key.endswith("_le"):
            if key.startswith("mom45") and mom is not None:
                if not (mom <= _num(val)):
                    return False
            else:
                pass
    return True

def choose_action(ctx, slot) -> dict:
    cfg = _load_config()
    rules = cfg.get("rules") or []

    for rule in rules:
        cond = rule.get("if", {})
        if _match(ctx, cond):
            then = str(rule.get("then", "LIMIT_LTP")).upper()
            lp = rule.get("limit_price", None)
            splim = rule.get("sp_limit")
            splimm = rule.get("sp_limit_mult")
            return {"mode": then, "limit_price": lp, "sp_limit": splim, "sp_limit_mult": splimm}

    # default
    default_mode = str(cfg.get("default", "LIMIT_LTP")).upper()
    limit_style = cfg.get("limit_style", None)
    lp = "CROSS" if (default_mode == "LIMIT_LTP" and str(limit_style or "").upper() == "AGGRESSIVE") else None
    return {"mode": default_mode, "limit_price": lp, "sp_limit": None, "sp_limit_mult": None}


    # default
    default_mode = str(cfg.get("default", "LIMIT_LTP")).upper()
    limit_style = cfg.get("limit_style", None)
    # 'AGGRESSIVE' -> translate to CROSS for LIMIT_LTP convenience
    lp = "CROSS" if (default_mode == "LIMIT_LTP" and str(limit_style or "").upper() == "AGGRESSIVE") else None
    return Action(default_mode, lp)