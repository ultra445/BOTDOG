from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Any
from ..types import Instruction, MarketIndexEntry

class Strategy(ABC):
    name: str = "BACK_WIN_1"

    @abstractmethod
    def decide_all(self, market_book: Any, market_index_entry: MarketIndexEntry, now_utc: datetime) -> List[Instruction]:
        """Retourne une liste d'instructions pour un MarketBook donné.
        Doit être robuste (ne pas crasher le caller).
        """
        ...
