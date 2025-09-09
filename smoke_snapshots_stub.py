# smoke_snapshots_stub.py
from datetime import datetime, timezone, timedelta
import types
from src.dogbot.executor import Executor
from src.dogbot.indexer import MarketIndex
from src.dogbot.types import MarketIndexEntry
from src.dogbot.strategy.back_win_1 import BackWin1

# Fake MarketBook minimal
book = types.SimpleNamespace(
    market_id="1.2345678",
    inplay=False,
    total_matched=1234.56,
    runners=[types.SimpleNamespace(selection_id=1, last_price_traded=2.6),
             types.SimpleNamespace(selection_id=2, last_price_traded=3.1)],
    market_definition=types.SimpleNamespace(market_type="WIN"),
)

# Index: départ dans 5 minutes
mie = MarketIndexEntry(
    market_id="1.2345678",
    market_type="WIN",
    event_id="999",
    event_open_utc=datetime.now(timezone.utc) + timedelta(minutes=5),
    venue="SHEFFIELD",
    country_code="GB",
    event_local_date=None,
    race_number=None,
    course_id="SHEFFIELD:2025-09-09:R00",
)
index = MarketIndex([mie])

executor = Executor(client=None, strategy=BackWin1(), market_index=index, dry_run=True)
executor.process_book(book)
print("OK — regarde le CSV dans ./data/<date>_snapshots.csv")
