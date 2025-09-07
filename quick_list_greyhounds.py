# -*- coding: utf-8 -*-
import os, datetime as dt
from dotenv import load_dotenv
import betfairlightweight as bflw
from betfairlightweight import filters

load_dotenv()

client = bflw.APIClient(
    os.getenv('BF_USER'),
    os.getenv('BF_PASS'),
    app_key=os.getenv('BF_APP_KEY'),
    certs=os.getenv('BF_CERTS_PATH'),
)
client.login()

now = dt.datetime.utcnow()
mfilter = filters.market_filter(
    event_type_ids=['4339'],               # Greyhounds
    market_countries=['GB','IE'],          # countries
    market_type_codes=['WIN'],             # WIN (or PLACE later)
    market_start_time={
        'from': now.isoformat()+'Z',
        'to':   (now+dt.timedelta(hours=2)).isoformat()+'Z'
    }
)

cats = client.betting.list_market_catalogue(
    filter=mfilter,
    max_results=20,
    market_projection=['MARKET_START_TIME','EVENT','RUNNER_DESCRIPTION']
)

print('Markets found:', len(cats))
for c in cats[:5]:
    venue = getattr(c.event, 'venue', None)
    print(c.market_id, '|', venue, '|', c.market_start_time)

client.logout()
