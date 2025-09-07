from __future__ import annotations
import datetime as dt
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional
import math

# ===== MarketState =====

class MarketState:
    """
    Stocke les séries de LTP par selection_id, avec timestamps, pour un marché.
    start_time est l'heure prévue (UTC). market_type: "WIN" ou "PLACE".
    Garde aussi un mapping selection_id -> trap (trap map) et le nombre de partants prévus.
    """
    def __init__(self, market_id: str, start_time: Any, market_type: str, meta_index: Dict[str, Any] | None = None):
        self.market_id = market_id
        self.market_type = (market_type or "").upper()
        self.start_time = self._norm_ts(start_time)
        self.meta_index = meta_index or {}
        # time series: selection_id -> deque[(ts, ltp)]
        self.ltp_ts: Dict[int, deque] = defaultdict(lambda: deque(maxlen=5000))
        # runner status (ACTIVE/REMOVED)
        self.status: Dict[int, str] = {}
        # trap map et info partants prévus
        self.trap_map: Dict[int, int] = {}   # selection_id -> trap
        self.planned_count: Optional[int] = None

        # hydrate depuis meta_index si dispo
        self._maybe_init_traps_from_meta(meta_index or {})

    @staticmethod
    def _norm_ts(ts: Any) -> Optional[dt.datetime]:
        if ts is None:
            return None
        if isinstance(ts, dt.datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=dt.timezone.utc)
            return ts
        try:
            s = str(ts)
            if s.endswith("Z"):
                return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            d = dt.datetime.fromisoformat(s)
            return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
        except Exception:
            return None

    def _maybe_init_traps_from_meta(self, meta: Dict[str, Any]) -> None:
        """
        Essaie de remplir trap_map & planned_count depuis le catalogue (meta_index).
        Format attendu (souple) :
          meta["runners"] = [{"selection_id": int, "trap": int? , "sort_priority": int?}, ...]
          meta["planned_count"] = int (facultatif)
        """
        try:
            runners = meta.get("runners") or []
            for r in runners:
                sid = r.get("selection_id")
                if sid is None:
                    continue
                trap = r.get("trap")
                if trap is None:
                    sp = r.get("sort_priority")
                    if sp is not None:
                        trap = int(sp)
                if trap is not None:
                    self.trap_map[int(sid)] = int(trap)
            if not self.planned_count:
                pc = meta.get("planned_count")
                if isinstance(pc, int) and pc > 0:
                    self.planned_count = pc
                else:
                    if runners:
                        self.planned_count = len(runners)
        except Exception:
            pass

    def _maybe_init_traps_from_market_def(self, book) -> None:
        """
        Si on n'a pas de trap_map, essaye de la déduire du market_definition du book :
          - runner.metadata["trap"] si présent (greyhounds)
          - sinon runner.sort_priority
        """
        try:
            md = getattr(book, "market_definition", None)
            if not md:
                return
            mdrunners = getattr(md, "runners", None) or []
            if not mdrunners:
                return
            tmp_map: Dict[int, int] = {}
            for r in mdrunners:
                sid = getattr(r, "selection_id", None)
                if sid is None:
                    continue
                trap = None
                # metadata.trap (string parfois)
                meta = getattr(r, "metadata", None) or {}
                t = meta.get("trap")
                if t is not None:
                    try:
                        trap = int(t)
                    except Exception:
                        trap = None
                if trap is None:
                    sp = getattr(r, "sort_priority", None)
                    if sp is not None:
                        try:
                            trap = int(sp)
                        except Exception:
                            trap = None
                if trap is not None:
                    tmp_map[int(sid)] = int(trap)
            if tmp_map:
                self.trap_map.update(tmp_map)
            if self.planned_count is None:
                self.planned_count = len(mdrunners)
        except Exception:
            pass

    def get_trap(self, sel_id: int, book=None) -> Optional[int]:
        """
        Renvoie le TRAP connu pour selection_id. Essaie de le déduire si absent.
        """
        if sel_id in self.trap_map:
            return self.trap_map[sel_id]
        if book is not None:
            self._maybe_init_traps_from_market_def(book)
            return self.trap_map.get(sel_id)
        return None

    def update_from_book(self, book) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        # on profite du book pour initialiser traps si besoin
        if not self.trap_map:
            self._maybe_init_traps_from_market_def(book)
        for r in getattr(book, "runners", []) or []:
            sel_id = getattr(r, "selection_id", None)
            if sel_id is None:
                continue
            self.status[sel_id] = getattr(r, "status", "ACTIVE") or "ACTIVE"
            # LTP si dispo, sinon meilleur back (approx)
            ltp = getattr(r, "last_price_traded", None)
            if ltp is None:
                ex = getattr(r, "ex", None)
                if ex and getattr(ex, "available_to_back", None):
                    try:
                        ltp = ex.available_to_back[0].price
                    except Exception:
                        ltp = None
            if ltp is not None and ltp > 1.0:
                self.ltp_ts[sel_id].append((now, float(ltp)))

    # === utilitaires LTP à t-Δ ===

    def _ltp_at(self, sel_id: int, target_ts: dt.datetime) -> Optional[float]:
        """Dernière valeur <= target_ts (sinon None)."""
        series = self.ltp_ts.get(sel_id)
        if not series:
            return None
        for ts, val in reversed(series):
            if ts <= target_ts:
                return val
        return None

    def _ltp_now(self, sel_id: int) -> Optional[float]:
        series = self.ltp_ts.get(sel_id)
        return series[-1][1] if series else None

    def ltp_offset_seconds(self, sel_id: int, seconds_before: int) -> Optional[float]:
        if not self.start_time:
            return None
        target = self.start_time - dt.timedelta(seconds=seconds_before)
        return self._ltp_at(sel_id, target)

    # === P et K (places payées) ===

    def active_runners_count(self) -> int:
        return sum(1 for s in self.status.values() if (s or "ACTIVE") == "ACTIVE")

    def k_places_used(self) -> int:
        P = self.active_runners_count()
        return 3 if P >= 8 else 2

    # === Virtual Trap helpers ===

    def _active_traps(self, book=None) -> List[int]:
        """
        Retourne la liste triée des traps des coureurs ACTIFS (via status).
        """
        traps: List[int] = []
        for sid, st in self.status.items():
            if (st or "ACTIVE") != "ACTIVE":
                continue
            tr = self.get_trap(sid, book=book)
            if tr is not None:
                traps.append(int(tr))
        traps.sort()
        return traps

    def virtual_trap_for(self, sel_id: int, book=None) -> Optional[int]:
        """
        Calcule VIRTUALTRAP pour sel_id selon les règles:
          - VIRTUALTRAP=1 = plus petit trap ACTIF (toujours)
          - VIRTUALTRAP=8 = plus grand trap ACTIF, mais seulement si planned_count >= 7
          - sinon VIRTUALTRAP = TRAP
        """
        tr = self.get_trap(sel_id, book=book)
        if tr is None:
            return None
        active_trs = self._active_traps(book=book)
        if not active_trs:
            return tr  # fallback
        min_active = active_trs[0]
        max_active = active_trs[-1]
        # bord gauche
        if tr == min_active:
            return 1
        # bord droit (seulement si >=7 partants prévus)
        planned = self.planned_count or len(active_trs)
        if planned >= 7 and tr == max_active:
            return 8
        # milieu inchangé
        return tr


# ===== FeatureStore =====

class FeatureStore:
    """
    Gère les états de marché et expose compute_selection_features(...)
    pour produire les variables LTP_tΔ, mom, diff, PLACE_THEORIQUE / Q_PLACE_THEORIQUE,
    et TRAP / VIRTUALTRAP.
    """
    def __init__(self):
        self.states: Dict[str, MarketState] = {}

    def prime_market(self, market_id: str, start_time: Any, market_type: str, index_entry: Dict[str, Any] | None = None):
        if market_id not in self.states:
            self.states[market_id] = MarketState(market_id, start_time, market_type, index_entry or {})

    def ingest_book(self, book) -> None:
        mid = getattr(book, "market_id", None)
        if mid is None:
            return
        st = self.states.get(mid)
        if st is None:
            # fallback si non primé
            mtype = ""
            stime = None
            try:
                md = getattr(book, "market_definition", None)
                if md:
                    mtype = getattr(md, "market_type", "") or ""
                    stime = getattr(md, "market_time", None)
            except Exception:
                pass
            st = MarketState(mid, stime, mtype, {})
            self.states[mid] = st
        st.update_from_book(book)

    # ====== Top-K (PLACE_THÉORIQUE) sous Plackett–Luce ======
    @staticmethod
    def _esp(weights: List[float], k: int) -> float:
        if k < 0:
            return 0.0
        e = [0.0] * (k + 1)
        e[0] = 1.0
        for w in weights:
            for j in range(k, 0, -1):
                e[j] += w * e[j - 1]
        return e[k]

    @staticmethod
    def _topk_prob_pl(weights: List[float], idx: int, K: int) -> Optional[float]:
        if not weights or idx < 0 or idx >= len(weights) or K <= 0 or K > len(weights):
            return None
        wi = weights[idx]
        if wi <= 0:
            return 0.0
        den = FeatureStore._esp(weights, K)
        if den <= 0:
            return None
        others = [w for j, w in enumerate(weights) if j != idx and w > 0]
        num = wi * FeatureStore._esp(others, K - 1)
        return max(0.0, min(1.0, num / den))

    @staticmethod
    def _implied_win_probs_from_ltp(ltps: List[Optional[float]]) -> List[float]:
        raw = []
        for x in ltps:
            if x is None or x <= 1.0:
                raw.append(0.0)
            else:
                raw.append(1.0 / float(x))
        s = sum(raw)
        if s <= 0:
            return [0.0 for _ in raw]
        return [r / s for r in raw]

    def compute_selection_features(self, market_book, selection_id: int, market_index_entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Renvoie un dict avec, entre autres :
          - LTP_t80_WIN, LTP_t45_WIN, LTP_t150_WIN, LTP_t300_WIN, LTP_at_market_WIN,
          - mom45_WIN, mom80_WIN, DIFF_45_80, DIFF_80_150, DIFF_150_300,
          - P (nb actifs), K_PLACE_USED,
          - PLACE_THEORIQUE, Q_PLACE_THEORIQUE (calculés à partir des LTP WIN courants),
          - TRAP, VIRTUALTRAP (selon règles).
        """
        feats: Dict[str, Any] = {}
        mid = getattr(market_book, "market_id", None)
        st = self.states.get(mid)
        if st is None:
            return feats

        # ---- TRAP / VIRTUALTRAP d'abord (on en profite pour hydrater trap_map via book si besoin) ----
        trap = st.get_trap(selection_id, book=market_book)
        vtrap = st.virtual_trap_for(selection_id, book=market_book)
        feats["TRAP"] = trap
        feats["VIRTUALTRAP"] = vtrap

        # ---- LTP à offsets ----
        if st.start_time is not None:
            ltp_t80 = st.ltp_offset_seconds(selection_id, 80)
            ltp_t45 = st.ltp_offset_seconds(selection_id, 45)
            ltp_t150 = st.ltp_offset_seconds(selection_id, 150)
            ltp_t300 = st.ltp_offset_seconds(selection_id, 300)
            ltp_t0 = st.ltp_offset_seconds(selection_id, 2) or st.ltp_offset_seconds(selection_id, 0)

            feats["LTP_t80_WIN"] = ltp_t80
            feats["LTP_t45_WIN"] = ltp_t45
            feats["LTP_t150_WIN"] = ltp_t150
            feats["LTP_t300_WIN"] = ltp_t300
            feats["LTP_at_market_WIN"] = ltp_t0

            def ratio(a, b):
                try:
                    if a is None or b is None or b <= 0:
                        return None
                    return (a / b) - 1.0
                except Exception:
                    return None

            feats["mom45_WIN"] = ratio(ltp_t0, ltp_t45)
            feats["mom80_WIN"] = ratio(ltp_t0, ltp_t80)
            feats["DIFF_45_80"] = ratio(ltp_t45, ltp_t80)
            feats["DIFF_80_150"] = ratio(ltp_t80, ltp_t150)
            feats["DIFF_150_300"] = ratio(ltp_t150, ltp_t300)

        # ---- P et K ----
        P = st.active_runners_count()
        K = st.k_places_used()
        feats["P"] = P
        feats["K_PLACE_USED"] = K

        # ---- PLACE_THEORIQUE (Plackett–Luce) ----
        runners = getattr(market_book, "runners", []) or []
        sel_ids: List[int] = []
        ltps_now: List[Optional[float]] = []
        for r in runners:
            sid = getattr(r, "selection_id", None)
            if sid is None:
                continue
            if st.status.get(sid, "ACTIVE") != "ACTIVE":
                continue
            sel_ids.append(sid)
            ltps_now.append(st._ltp_now(sid))

        weights = self._implied_win_probs_from_ltp(ltps_now)
        try:
            idx = sel_ids.index(selection_id)
        except ValueError:
            idx = -1

        place_th = None
        if idx >= 0 and 1 <= K <= len(weights):
            place_th = self._topk_prob_pl(weights, idx, K)

        feats["PLACE_THEORIQUE"] = place_th
        feats["Q_PLACE_THEORIQUE"] = (1.0 / place_th) if (place_th and place_th > 0.0) else None

        return feats
