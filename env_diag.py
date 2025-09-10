from pathlib import Path
from dotenv import load_dotenv, dotenv_values
import os, sys, binascii

root = Path(__file__).resolve().parent
p = root / ".env"

print("[DIAG] Working dir:", root)
print("[DIAG] .env exists?:", p.exists(), "| path:", p)

if not p.exists():
    sys.exit(0)

b = p.read_bytes()
print("[DIAG] File size (bytes):", len(b))
print("[DIAG] First bytes (hex):", binascii.hexlify(b[:4]).decode())

if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
    print("[DIAG] Looks like UTF-16 -> re-save as UTF-8 (sans BOM)")
elif b[:3] == b"\xef\xbb\xbf":
    print("[DIAG] UTF-8 BOM detected (OK)")
else:
    print("[DIAG] No BOM (OK)")

load_dotenv(p, override=True)
vals = dotenv_values(p)
keys = ["BETFAIR_USERNAME","BETFAIR_PASSWORD","BETFAIR_APP_KEY","BETFAIR_CERTS_PATH"]
print("[DIAG] Keys in file:", sorted(k for k in vals if k and not k.strip().startswith("#")))
print("[DIAG] getenv flags:", {k: bool(os.getenv(k)) for k in keys})
