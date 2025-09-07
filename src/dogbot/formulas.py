# src/dogbot/formulas.py
from __future__ import annotations
import os, math
from typing import Optional

# Petit set de fonctions autorisées dans les formules .env
SAFE_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "pow": pow,
}

def _get_env_float(key: str) -> Optional[float]:
    v = os.getenv(key, "").strip()
    if not v:
        return None
    try:
        return float(v)
    except Exception:
        return None

def _get_env_formula(key: str) -> Optional[str]:
    f = os.getenv(key, "").strip()
    return f or None

def _eval_formula(expr: str, *, odds: float, side: str, market_type: str) -> Optional[float]:
    # Évalue une expression Python simple, sans builtins
    try:
        return float(eval(
            expr,
            {"__builtins__": None},
            {**SAFE_FUNCS, "odds": odds, "side": side, "market_type": market_type},
        ))
    except Exception:
        return None

def compute_stake(*, side: str, market_type: str, odds: float, default_unit: float = 2.0) -> float:
    """
    Calcule la mise à partir de :
      1) STAKE_FORMULA_{BACK/LAY}_{WIN/PLACE}, si présent et valide
      2) STAKE_{BACK/LAY}_{WIN/PLACE}, si présent
      3) default_unit
    Arguments passés par nom (keyword-only).
    """
    # Choix des clés en fonction (side, market_type)
    key_suffix = None
    sm = (side.upper(), market_type.upper())
    if sm == ("BACK", "WIN"):
        key_suffix = "BACK_WIN"
    elif sm == ("LAY", "WIN"):
        key_suffix = "LAY_WIN"
    elif sm == ("BACK", "PLACE"):
        key_suffix = "BACK_PLACE"
    elif sm == ("LAY", "PLACE"):
        key_suffix = "LAY_PLACE"
    else:
        # Cas non prévu : repli
        return max(0.0, float(default_unit))

    # 1) Formule dynamique
    fkey = f"STAKE_FORMULA_{key_suffix}"
    expr = _get_env_formula(fkey)
    if expr:
        val = _eval_formula(expr, odds=odds, side=side.upper(), market_type=market_type.upper())
        if val is not None and val > 0:
            return float(val)

    # 2) Mise fixe de repli
    ffix = _get_env_float(f"STAKE_{key_suffix}")
    if ffix is not None and ffix > 0:
        return float(ffix)

    # 3) Dernier repli
    return max(0.0, float(default_unit))
