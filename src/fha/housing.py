"""
Housing outcomes linked to circuits by geography: Census ACS Black-White
dissimilarity (segregation) and HMDA denial rates, aggregated at the county
level and population-weighted up to circuits.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config

# --- state -> circuit crosswalk (derived from the district court geography) ---
STATE_TO_CIRCUIT = {
    "ME": "1", "MA": "1", "NH": "1", "RI": "1", "PR": "1",
    "CT": "2", "NY": "2", "VT": "2",
    "DE": "3", "NJ": "3", "PA": "3", "VI": "3",
    "MD": "4", "NC": "4", "SC": "4", "VA": "4", "WV": "4",
    "LA": "5", "MS": "5", "TX": "5",
    "KY": "6", "MI": "6", "OH": "6", "TN": "6",
    "IL": "7", "IN": "7", "WI": "7",
    "AR": "8", "IA": "8", "MN": "8", "MO": "8", "NE": "8", "ND": "8", "SD": "8",
    "AK": "9", "AZ": "9", "CA": "9", "HI": "9", "ID": "9", "MT": "9",
    "NV": "9", "OR": "9", "WA": "9", "GU": "9",
    "CO": "10", "KS": "10", "NM": "10", "OK": "10", "UT": "10", "WY": "10",
    "AL": "11", "FL": "11", "GA": "11",
    "DC": "DC",
}
STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
    # territories in STATE_TO_CIRCUIT that ACS covers (PR); others (VI/GU/NMI)
    # are not in the standard acs5 tract product and are skipped gracefully.
    "PR": "72",
}
FIPS_STATE = {v: k for k, v in STATE_FIPS.items()}


class CensusClient:
    """Minimal ACS 5-year client."""

    BASE = "https://api.census.gov/data"

    def __init__(self, key: str | None = None):
        self.key = config.census_key(key)

    def fetch(self, year: int, variables: list[str], for_geo: str,
              in_geo: str | None = None, dataset: str = "acs/acs5") -> pd.DataFrame:
        params = {"get": ",".join(variables), "for": for_geo}
        if in_geo:
            params["in"] = in_geo
        if self.key:
            params["key"] = self.key
        r = requests.get(f"{self.BASE}/{year}/{dataset}", params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data[1:], columns=data[0])


# ACS variables: non-Hispanic White alone, non-Hispanic Black alone, totals
SEG_VARS = ["B03002_003E", "B03002_004E", "B03002_001E"]
RENT_VAR = "B25064_001E"        # median gross rent
VALUE_VAR = "B25077_001E"       # median home value


def dissimilarity_from_tracts(tracts: pd.DataFrame, area_col: str = "county") -> pd.DataFrame:
    """Black/White dissimilarity index per area from tract counts.

    D = 0.5 * sum_i | b_i/B - w_i/W |, computed within each area over its tracts.
    """
    df = tracts.copy()
    df["white"] = pd.to_numeric(df["B03002_003E"], errors="coerce").clip(lower=0)
    df["black"] = pd.to_numeric(df["B03002_004E"], errors="coerce").clip(lower=0)
    df["pop"] = pd.to_numeric(df["B03002_001E"], errors="coerce").clip(lower=0)
    out = []
    for area, g in df.groupby(area_col):
        B, W = g["black"].sum(), g["white"].sum()
        if B <= 0 or W <= 0:
            continue
        D = 0.5 * (np.abs(g["black"] / B - g["white"] / W)).sum()
        out.append({area_col: area, "dissimilarity_index": round(float(D), 4),
                    "population": int(g["pop"].sum())})
    return pd.DataFrame(out)


def fetch_segregation_panel(years: list[int], states: list[str] | None = None,
                            key: str | None = None) -> pd.DataFrame:
    """Live: circuit x year Black/White dissimilarity, pop-weighted from counties."""
    cc = CensusClient(key)
    states = states or list(STATE_TO_CIRCUIT)
    rows = []
    for year in years:
        for st in states:
            fips = STATE_FIPS.get(st)
            if not fips:
                continue
            try:
                tr = cc.fetch(year, SEG_VARS, for_geo="tract:*",
                              in_geo=f"state:{fips} county:*")
            except requests.HTTPError:
                continue
            tr["county"] = tr["state"] + tr["county"]
            county_d = dissimilarity_from_tracts(tr, area_col="county")
            county_d["circuit"] = STATE_TO_CIRCUIT[st]
            rows.append(county_d.assign(year=year, state=st))
    if not rows:
        return pd.DataFrame(columns=["circuit", "year", "dissimilarity_index"])
    allc = pd.concat(rows, ignore_index=True)
    # population-weighted mean dissimilarity up to circuit x year
    def wavg(g):
        w = g["population"].clip(lower=1)
        return np.average(g["dissimilarity_index"], weights=w)
    panel = (allc.groupby(["circuit", "year"])
             .apply(wavg, include_groups=False)
             .reset_index(name="dissimilarity_index"))
    return panel


class HMDAClient:
    """CFPB HMDA aggregate API (no key). Denial rate as a lending-bias proxy."""

    BASE = "https://ffiec.cfpb.gov/v2/data-browser-api/view/aggregations"

    def denial_rates(self, year: int, states: list[str]) -> pd.DataFrame:
        out = []
        for st in states:
            params = {"years": year, "states": st,
                      "actions_taken": "1,3"}   # 1=originated, 3=denied
            try:
                r = requests.get(self.BASE, params=params, timeout=60)
                r.raise_for_status()
                aggs = r.json().get("aggregations", [])
            except (requests.HTTPError, requests.ConnectionError, ValueError):
                continue
            orig = sum(a["count"] for a in aggs if str(a.get("actions_taken")) == "1")
            den = sum(a["count"] for a in aggs if str(a.get("actions_taken")) == "3")
            tot = orig + den
            if tot:
                out.append({"circuit": STATE_TO_CIRCUIT.get(st), "year": year,
                            "state": st, "denial_rate": round(den / tot, 4)})
        df = pd.DataFrame(out)
        if df.empty:
            return df
        return (df.groupby(["circuit", "year"])["denial_rate"].mean()
                .reset_index())


def load_housing_panel(real: bool = False, years: list[int] | None = None,
                       states: list[str] | None = None) -> pd.DataFrame:
    """Return the circuit x year housing panel. real=False uses the synthetic
    panel written by `synth`; real=True hits Census (+HMDA) live."""
    if not real:
        p = config.EXTERNAL / "synthetic_housing_panel.csv"
        if not p.exists():
            from . import synth
            synth.generate()
        return pd.read_csv(p)
    years = years or list(range(2012, 2023))
    seg = fetch_segregation_panel(years, states)
    try:
        hm = HMDAClient().denial_rates(years[-1], states or list(STATE_TO_CIRCUIT))
        if not hm.empty:
            seg = seg.merge(hm, on=["circuit", "year"], how="left")
    except Exception as e:                       # noqa: BLE001 - logged, not hidden
        logging.getLogger("fha.housing").warning(
            "HMDA enrichment skipped: %s", e)
    out = config.EXTERNAL / "housing_panel.csv"
    seg.to_csv(out, index=False)
    return seg
