import os, time, threading, datetime as dt, queue
from dotenv import load_dotenv
import betfairlightweight as bflw
from betfairlightweight import filters

load_dotenv()

user  = os.getenv("BF_USER"); pwd = os.getenv("BF_PASS")
app   = os.getenv("BF_APP_KEY"); certs = os.getenv("BF_CERTS_PATH")
countries = [s.strip() for s in os.getenv("DOG_COUNTRIES", "GB,IE").split(",") if s.strip()]
client = bflw.APIClient(user, pwd, app_key=app, certs=certs); client.login()

now = dt.datetime.now(dt.timezone.utc)
mf = filters.market_filter(
    event_type_ids=["4339"],
    market_countries=countries,
    market_type_codes=["WIN"],
    market_start_time={"from": now.isoformat(), "to": (now + dt.timedelta(minutes=45)).isoformat()},
)
cats = client.betting.list_market_catalogue(filter=mf, max_results=1, market_projection=["EVENT","RUNNER_DESCRIPTION"])
if not cats:
    print("No markets found"); client.logout(); raise SystemExit(0)

mid = cats[0].market_id
print("Subscribing to market:", mid, "| countries =", countries)

out_q = queue.Queue()
listener = bflw.StreamListener(output_queue=out_q, max_latency=0.5)
stream = client.streaming.create_stream(unique_id=99, listener=listener)
stream.subscribe_to_markets(
    market_filter=filters.streaming_market_filter(market_ids=[mid]),
    market_data_filter=filters.streaming_market_data_filter(
        fields=["EX_BEST_OFFERS","EX_BEST_OFFERS_DISP","EX_TRADED","EX_LTP","EX_MARKET_DEF"],
        ladder_levels=3,
    ),
)
t = threading.Thread(target=stream.start, daemon=True); t.start()

start = time.time(); count = 0
while time.time() - start < 20:
    try:
        u = out_q.get(timeout=0.5)
    except queue.Empty:
        continue
    if u:
        count += len(u)
        mb = u[0]
        lpt = getattr(mb.runners[0], "last_price_traded", None) if mb.runners else None
        print("Update:", "market", mb.market_id, "| inplay=", getattr(mb, "inplay", False),
              "| runners=", len(mb.runners) if mb.runners else 0, "| first LPT=", lpt)

try:
    stream.stop()
except Exception:
    pass
client.logout()
print("Done. Updates received =", count)
