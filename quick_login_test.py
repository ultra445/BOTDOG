# quick_login_test.py — certs en MODE DOSSIER (pas tuple)
from __future__ import annotations
import os
from pathlib import Path
import betfairlightweight as bflw

# Charge explicitement le .env à la racine du projet (facultatif si tu as déjà les $env:)
try:
    from dotenv import load_dotenv  # type: ignore
    ROOT = Path(__file__).parent
    load_dotenv(ROOT / ".env")
except Exception:
    pass

def need(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    raise SystemExit("Variable d'env manquante: " + " / ".join(names))

def main() -> None:
    username  = need("BF_USER", "BETFAIR_USERNAME", "BETFAIR_USER")
    password  = need("BF_PASS", "BETFAIR_PASSWORD")
    app_key   = need("BF_APP_KEY", "BETFAIR_APP_KEY")
    certs_dir = need("BF_CERTS_PATH", "BETFAIR_CERTS_PATH")  # ex: C:\betfair-certs

    p = Path(certs_dir)
    if not p.is_dir():
        raise SystemExit(f"Le chemin certs n'est pas un dossier: {p}")

    crt = p / "client-2048.crt"
    key = p / "client-2048.key"
    if not (crt.exists() and key.exists()):
        raise SystemExit(
            "Certificats introuvables dans le dossier:\n"
            f"  {crt}\n  {key}\n"
            "➡️ Renomme (ou copie) tes fichiers en ces noms exacts."
        )

    print("Certs (dossier) ->", certs_dir)

    # IMPORTANT: passer un CHEMIN DE DOSSIER, pas un tuple
    client = bflw.APIClient(username=username, password=password, app_key=app_key, certs=certs_dir)
    try:
        print("[LOGIN] connecting...")
        client.login()
        print("[LOGIN] OK  | token:", (client.session_token or "")[:12], "...")
        client.keep_alive()
        print("[KEEP-ALIVE] OK")
    finally:
        try:
            client.logout()
            print("[LOGOUT] OK")
        except Exception:
            pass

if __name__ == "__main__":
    main()
