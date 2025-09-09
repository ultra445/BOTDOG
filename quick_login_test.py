# quick_login_test.py
from __future__ import annotations
import os, sys
from pathlib import Path

# Charger explicitement le .env à la racine du projet
ROOT = Path(__file__).parent
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:
    pass

import betfairlightweight as bflw


def _pick(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def _need(*names: str) -> str:
    v = _pick(*names)
    if not v:
        raise SystemExit(
            "Variable d'env manquante: " + " / ".join(names)
        )
    return v


def _resolve_certs(certs_path: str) -> str | tuple[str, str]:
    """Accepte un dossier contenant .crt/.key (noms standards ou premiers trouvés),
    ou bien un chemin direct vers un .crt/.key (on renvoie tel quel)."""
    p = Path(certs_path)
    if p.is_dir():
        # Noms standard
        crt = p / "client-2048.crt"
        key = p / "client-2048.key"
        if crt.exists() and key.exists():
            return (str(crt), str(key))
        # Premier .crt / premier .key trouvés
        crt_files = list(p.glob("*.crt"))
        key_files = list(p.glob("*.key"))
        if crt_files and key_files:
            return (str(crt_files[0]), str(key_files[0]))
        # À défaut, on renvoie le dossier (bflw sait gérer certains cas)
        return str(p)
    else:
        # Fichier direct
        return str(p)


def main() -> None:
    username = _need("BF_USER", "BETFAIR_USERNAME", "BETFAIR_USER")
    password = _need("BF_PASS", "BETFAIR_PASSWORD")
    app_key  = _need("BF_APP_KEY", "BETFAIR_APP_KEY")
    certs_in = _need("BF_CERTS_PATH", "BETFAIR_CERTS_PATH")

    certs = _resolve_certs(certs_in)
    print("Certs ->", certs_in)

    client = bflw.APIClient(username=username, password=password, app_key=app_key, certs=certs)
    try:
        print("[LOGIN] connecting...")
        client.login()
        print("[LOGIN] OK  | session token:", (client.session_token or "")[:12], "...")
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
