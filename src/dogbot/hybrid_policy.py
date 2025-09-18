from __future__ import annotations
import json, os
from typing import Any, Dict, Optional

CONFIG_PATH = os.environ.get("HYBRID_POLICY_PATH", "./config/hybrid_policy.json")

def _load_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"default": "LIMIT_LTP", "limit_style": "AGGRESSIVE", "rules": []}

def _eq(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    return str(a).upper() == str(b).upper()

def _num(val) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None

def _effective_mom(ctx) -> Optional[float]:
    """Prefer WIN momentum mom45 even when market_type == PLACE; fallback to mom45p if needed."""
    for name in ("mom45", "mom45_win", "mom_win45", "mom45p"):
        if hasattr(ctx, name):
            v = getattr(ctx, name)
            try:
                return float(v) if v is not None else None
            except Exception:
                continue
    return None

def _match(ctx, cond: Dict[str, Any]) -> bool:
    # side / market_type
    if "side" in cond and not _eq(cond["side"], getattr(ctx, "side", None)):
        return False
    if "market_type" in cond and not _eq(cond["market_type"], getattr(ctx, "market_type", None)):
        return False
    # optional region filter ("UK" / "ROW")
    if "region" in cond:
        want = str(cond["region"]).upper()
        have = str(getattr(ctx, "region", "") or "").upper()
        if want and have and want != have:
            return False

    mom = _effective_mom(ctx)

    # Numeric thresholds on momentum
    for key, val in cond.items():
        if key.endswith("_ge") and key.startswith("mom45"):
            thr = _num(val)
            if thr is not None and mom is not None and not (mom >= thr):
                return False
        if key.endswith("_le") and key.startswith("mom45"):
            thr = _num(val)
            if thr is not None and mom is not None and not (mom <= thr):
                return False
    return True

def choose_action(ctx, slot) -> dict:
    cfg = _load_config()
    rules = cfg.get("rules") or []
    for rule in rules:
        cond = rule.get("if", {})
        if _match(ctx, cond):
            return {
                "mode": str(rule.get("then", "LIMIT_LTP")).upper(),
                "limit_price": rule.get("limit_price"),
                "sp_limit": rule.get("sp_limit"),
                "sp_limit_mult": rule.get("sp_limit_mult"),
            }

    # default
    default_mode = str(cfg.get("default", "LIMIT_LTP")).upper()
    limit_style = str(cfg.get("limit_style", "")).upper()
    # convenience: if default is LIMIT_LTP + AGGRESSIVE, set CROSS
    lp = "CROSS" if (default_mode == "LIMIT_LTP" and limit_style == "AGGRESSIVE") else None
    return {"mode": default_mode, "limit_price": lp, "sp_limit": None, "sp_limit_mult": None}
