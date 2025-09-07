import os, time, threading, datetime as dt
from dotenv import load_dotenv
import betfairlightweight as bflw
from betfairlightweight import filters

load_dotenv()

user  = os.getenv("BF_USER")
pwd   = os.getenv("BF_PASS")
app   = os.getenv("BF_APP_KEY")
certs = os.getenv("BF_CERTS_PATH")
countries = [s.strip() for s in os.getenv("DOG_COUNTRIES", "GB,IE,AU,NZ").split(",") if s.strip()]

client = bflw.APIClient(user, pwd, app_key=app, certs=certs)
client.login()

now = dt.datetime.now(dt.timezone.utc)
mf = filters.market_filter(
    event_type_ids=["4339"],    # Greyhounds
    market_countries=countries,
    market_type_codes=["WIN"],
    market_start_time={"from": now.isoformat(), "to": (now + dt.timedelta(minutes=60)).isoformat()},
)

cats = client.betting.list_market_catalogue(
    filter=mf,
    max_results=1,
    market_projection=["EVENT","RUNNER_DESCRIPTION"]
)

if not cats:
    print("X Aucun marché trouvé dans la prochaine heure")
    client.logout()
    raise SystemExit(0)

mid = cats[0].market_id
print("OK Debug sur le marché:", mid, "| event:", cats[0].event.name)

listener = bflw.StreamListener(max_latency=0.5)
stream = client.streaming.create_stream(unique_id=42, listener=listener)
stream.subscribe_to_markets(
    market_filter=filters.streaming_market_filter(market_ids=[mid]),
    market_data_filter=filters.streaming_market_data_filter(
    fields=["EX_BEST_OFFERS","EX_TRADED","LAST_PRICE_TRADED","MARKET_DEF"],
    ladder_levels=3,
),

    
)

# Thread de stream
t = threading.Thread(target=stream.start, daemon=True)
t.start()

start = time.time()
while time.time() - start < 20:  # écoute 20 secondes
    updates = listener.snap()
    if updates:
        mb = updates[0]
        print("---- MarketBook ----")
        for r in mb.runners:
            print(f"Runner {r.selection_id}: LTP={r.last_price_traded}")
        print("--------------------")
    time.sleep(2)

try:
    stream.stop()
except Exception:
    pass
client.logout()
print("OK Debug terminé")
