from dotenv import load_dotenv
load_dotenv()
from src.dogbot.config import Settings
from src.dogbot.strategies_catalog import load_strategies_from_env
cfg = Settings.load()
strats = load_strategies_from_env(cfg.max_market_stake, per_cat=10)
print("Loaded:", [s.name for s in strats], "stakes:", [s.unit_stake for s in strats])
