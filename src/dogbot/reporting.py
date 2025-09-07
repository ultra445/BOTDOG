from __future__ import annotations
import csv, os, datetime as dt, threading
from typing import Dict, Any

class ReportWriter:
    """
    Écrit un CSV par jour dans reports/trades_YYYY-MM-DD.csv
    On peut surcharger le dossier via env REPORT_DIR.
    """
    def __init__(self):
        self.dir = os.getenv("REPORT_DIR", "reports")
        os.makedirs(self.dir, exist_ok=True)
        self._lock = threading.Lock()
        self._header = [
            "ts_iso","dry_run","placed","status","error",
            "market_id","market_type","event","scheduled_start",
            "selection_id","selection_name","trap",
            "side","price","size",
            "strategy","category","slot",
            "bet_id","customer_order_ref"
        ]

    def _path_today(self) -> str:
        d = dt.datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.dir, f"trades_{d}.csv")

    def _ensure_header(self, path: str):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(self._header)

    def write_trade(self, row: Dict[str, Any]):
        path = self._path_today()
        with self._lock:
            self._ensure_header(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([row.get(k, "") for k in self._header])
