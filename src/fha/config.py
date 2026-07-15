from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
VALIDATION = DATA / "validation"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"

for path in (PROCESSED, TABLES):
    path.mkdir(parents=True, exist_ok=True)
