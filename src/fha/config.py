"""
Repo-relative paths and secret resolution (explicit arg -> env var -> key
file). Never prints a secret.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo root = two levels up from this file (src/fha/config.py -> repo root).
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
CONFIG = ROOT / "config"

for _p in (RAW, INTERIM, PROCESSED, EXTERNAL, FIGURES, TABLES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)


def _resolve_secret(explicit: str | None, env_var: str,
                    *file_candidates: Path) -> str | None:
    """First non-empty of: explicit arg, env var, contents of a key file."""
    if explicit:
        return explicit.strip()
    val = os.environ.get(env_var)
    if val:
        return val.strip()
    for fc in file_candidates:
        try:
            if fc and fc.is_file():
                txt = fc.read_text().strip()
                if txt:
                    return txt
        except OSError:
            continue
    return None


def courtlistener_token(explicit: str | None = None) -> str | None:
    """CourtListener REST API token (optional but raises rate limits a lot).

    Drop your token in config/.cl_token, or `export COURTLISTENER_TOKEN=...`.
    Get one free at https://www.courtlistener.com/help/api/rest/ (register).
    """
    return _resolve_secret(
        explicit, "COURTLISTENER_TOKEN",
        CONFIG / ".cl_token",
        Path.home() / ".config" / "courtlistener" / "token",
    )


def census_key(explicit: str | None = None) -> str | None:
    """Census API key (free: https://api.census.gov/data/key_signup.html).

    Drop it in config/.census_key, or `export CENSUS_API_KEY=...`.
    """
    return _resolve_secret(
        explicit, "CENSUS_API_KEY",
        CONFIG / ".census_key",
        Path.home() / ".census_api_key",
    )


@dataclass
class Settings:
    """Pipeline parameters. Override per-run rather than editing in place."""
    # collection
    cl_base_url: str = "https://www.courtlistener.com/api/rest/v4"
    search_query: str = '"fair housing act"'
    # CourtListener court ids to sweep (district + appellate). Empty = all courts.
    courts: list[str] = field(default_factory=list)
    date_min: str = "1990-01-01"
    date_max: str = "2024-12-31"
    max_pages: int = 5            # page size 20; raise once a token is set
    polite_delay_s: float = 1.1   # anonymous rate-limit courtesy

    # embeddings / clustering
    embedding_backend: str = "tfidf"   # "tfidf" (runs now) | "legalbert" (GPU)
    legalbert_model: str = "nlpaueb/legal-bert-base-uncased"
    n_doctrinal_regimes: int = 3       # strict / moderate / weak

    # /7: panels
    unit: str = "circuit"              # "circuit" | "metro"
    year_min: int = 1990
    year_max: int = 2024

    # robustness
    lag_years: tuple[int, ...] = (1, 2, 3, 4, 5)
    placebo_statutes: tuple[str, ...] = ("42 U.S.C. 1983", "Title VII", "ADA Title II")


SETTINGS = Settings()
