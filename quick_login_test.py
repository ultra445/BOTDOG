import os
from dotenv import load_dotenv
import betfairlightweight as bflw

load_dotenv()

user  = os.getenv('BF_USER')
pwd   = os.getenv('BF_PASS')
app   = os.getenv('BF_APP_KEY')
certs = os.getenv('BF_CERTS_PATH')

print('Certs path =', certs)

client = bflw.APIClient(user, pwd, app_key=app, certs=certs)
try:
    client.login()
    print('LOGIN OK  | session token:', (client.session_token or '')[:12], '...')
    client.keep_alive()
    print('KEEP-ALIVE OK')
finally:
    try:
        client.logout()
        print('LOGOUT OK')
    except Exception:
        pass
