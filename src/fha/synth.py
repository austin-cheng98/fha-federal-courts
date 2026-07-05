"""
Synthetic data generator for offline validation (outputs namespaced
synthetic_*). Embeds a known causal chain with BETA_TRUE so the estimators
can be checked against ground truth.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from . import config

CIRCUITS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "DC"]
BETA_TRUE = -0.70          # stronger enforcement -> LOWER segregation (the truth)
SHOCK_YEAR = 2015          # Inclusive Communities

# A few realistic district/appellate court ids per circuit for flavor.
_COURT_BY_CIRCUIT = {
    "1": ["mad", "ca1"], "2": ["nysd", "ca2"], "3": ["njd", "ca3"],
    "4": ["mdd", "ca4"], "5": ["txsd", "ca5"], "6": ["ohnd", "ca6"],
    "7": ["ilnd", "ca7"], "8": ["mnd", "ca8"], "9": ["cand", "ca9"],
    "10": ["cod", "ca10"], "11": ["flsd", "ca11"], "DC": ["dcd", "cadc"],
}

# Opinion text building blocks; selected conditional on latent labels so the
# rule extractors in extract.py fire correctly.
_INTRO = ("Plaintiff brings this action under the Fair Housing Act, 42 U.S.C. "
          "{sec}, alleging that defendant {who}. The Court has jurisdiction "
          "pursuant to 28 U.S.C. 1331.")
_CLAIM_TEXT = {
    "disparate_impact": ("Plaintiff advances a disparate impact theory, contending that "
        "defendant's facially neutral policy has a discriminatory effect on a protected "
        "class. Under 24 C.F.R. 100.500 and Texas Department of Housing v. Inclusive "
        "Communities Project, the burden-shifting framework requires plaintiff to make a "
        "prima facie showing of a robust causal connection. "),
    "disparate_treatment": ("Plaintiff alleges intentional discrimination, i.e. disparate "
        "treatment because of race. Applying the McDonnell Douglas burden-shifting framework, "
        "plaintiff must establish a prima facie case, after which defendant must articulate a "
        "legitimate, nondiscriminatory reason, and plaintiff may show pretext. "),
    "reasonable_accommodation": ("Plaintiff, who is handicapped within the meaning of the Act, "
        "seeks a reasonable accommodation under 42 U.S.C. 3604(f). The requested accommodation "
        "is necessary to afford an equal opportunity to use and enjoy the dwelling. "),
    "zoning_exclusionary": ("Plaintiff challenges the municipality's zoning ordinance and denial "
        "of a special use permit as exclusionary land use that makes housing unavailable. "),
    "refusal_rent_sell": ("Plaintiff alleges defendant refused to rent and engaged in steering "
        "and redlining, otherwise making a dwelling unavailable. "),
}
_WIN = ("Accordingly, plaintiff's motion for summary judgment is granted and defendant's "
        "motion is denied. Judgment is entered in favor of the plaintiff. ")
_LOSE = ("Accordingly, defendant's motion for summary judgment is granted and plaintiff's "
         "complaint is dismissed. Judgment is entered in favor of the defendant. ")
_REVERSE = "We reverse the judgment of the district court and remand for further proceedings. "
_AFFIRM = "We affirm the judgment of the district court. "
_REMEDY = {
    "injunction": "The Court orders permanent injunctive relief enjoining the challenged policy. ",
    "damages": "Plaintiff is awarded compensatory and punitive damages. ",
    "declaratory": "The Court issues declaratory relief declaring the policy unlawful. ",
}
_FILLER = ("See 42 U.S.C. 3604; 42 U.S.C. 3605. The Court has considered the parties' briefs "
           "and the record. 994 F. Supp. 2d 1121. 135 S. Ct. 2507. ") * 4


def _latent_strictness(circuit: str, year: int, rng: random.Random,
                       circ_fe: dict, shock: dict) -> float:
    base = circ_fe[circuit]
    trend = 0.015 * (year - 2000)
    post = shock[circuit] if year >= SHOCK_YEAR else 0.0
    s = base + trend + post + rng.gauss(0, 0.05)
    return 1 / (1 + math.exp(-4 * (s - 0.5)))   # squash to (0,1)


def _synth_case_record(i: int, court_id: str, circuit: str, level: str,
                       year: int, s: float, rng: random.Random) -> dict:
    """Build one synthetic case dict from a latent strictness s. Shared by the
    circuit- and district-level generators."""
    claims = []
    if rng.random() < 0.25 + 0.45 * s:
        claims.append("disparate_impact")
    if rng.random() < 0.30:
        claims.append("disparate_treatment")
    if rng.random() < 0.20 + 0.25 * s:
        claims.append("reasonable_accommodation")
    if rng.random() < 0.18:
        claims.append("zoning_exclusionary")
    if rng.random() < 0.15:
        claims.append("refusal_rent_sell")
    if not claims:
        claims = ["disparate_treatment"]
    # win probability rises with latent strictness AND observable claim type
    p_win = (0.20 + 0.40 * s + 0.12 * ("disparate_impact" in claims)
             + 0.06 * ("reasonable_accommodation" in claims))
    win = rng.random() < min(p_win, 0.95)
    text = [_INTRO.format(sec="3604", who="discriminated in housing")]
    for c in claims:
        text.append(_CLAIM_TEXT[c])
    if level == "appellate":
        text.append(_REVERSE if win else _AFFIRM)
    text.append(_WIN if win else _LOSE)
    if win:
        if rng.random() < 0.4 + 0.4 * s:
            text.append(_REMEDY["injunction"])
        if rng.random() < 0.5:
            text.append(_REMEDY["damages"])
        if rng.random() < 0.3:
            text.append(_REMEDY["declaratory"])
    text.append(_FILLER)
    full = " ".join(text)
    return {
        "cluster_id": 90_000_000 + i,
        "case_name": f"Synthetic Plaintiff {i} v. Defendant Housing Auth.",
        "court_id": court_id, "circuit": circuit, "court_level": level,
        "court_jurisdiction": "F", "date_filed": f"{year}-0{rng.randint(1,9)}-15",
        "year": year, "docket_number": f"{year}-cv-{1000+i}",
        "citations": ["994 F. Supp. 2d 1121"], "cite_count": rng.randint(0, 80),
        "judges": rng.choice(["Smith", "Johnson", "Lee", "Garcia", "Nguyen"]),
        "panel_names": [], "precedential_status": "Published",
        "nature_of_suit": "443", "posture": "On Motion for Summary Judgment",
        "procedural_history": "", "opinion_ids": [80_000_000 + i],
        "text": full, "text_len": len(full), "text_source": "synthetic",
        "source": "synthetic", "_latent_strictness": round(s, 4),
    }


def generate_district(n_cases: int = 16000, year_min: int = 2000,
                      year_max: int = 2023, seed: int = 7) -> dict:
    """District-level synthetic harness (~94 federal districts as the unit).

    The methodological point: 12 circuits is too few clusters for well-powered
    cluster-robust inference; the district x year panel has ~90 clusters and
    recovers the embedded effect under VALID inference. Same DGP, finer unit:
    each district has its own base doctrine; the post-2015 shock is shared within
    a circuit (Inclusive Communities binds the circuit); housing responds with
    the known BETA_TRUE. Cases land in real district court ids; the housing panel
    is keyed by district (court_id).
    """
    from .reference import _CIRCUIT_DISTRICTS
    districts = [(d, c) for c, ds in _CIRCUIT_DISTRICTS.items() for d in ds]
    d2c = {d: c for d, c in districts}
    dlist = [d for d, _ in districts]

    seed_rng = random.Random(seed)
    circ_fe = {c: seed_rng.uniform(0.30, 0.70) for c in CIRCUITS}
    shock = {c: max(0.0, 0.10 + 0.6 * (circ_fe[c] - 0.5) + seed_rng.uniform(0, 0.08))
             for c in CIRCUITS}
    dist_base = {d: min(0.95, max(0.05, circ_fe[d2c[d]] + seed_rng.gauss(0, 0.10)))
                 for d in dlist}
    corpus_rng = random.Random(seed_rng.getrandbits(64))
    housing_rng = random.Random(seed_rng.getrandbits(64))

    def dlatent(d, y, rng):
        c = d2c[d]
        x = (dist_base[d] + 0.015 * (y - 2000)
             + (shock[c] if y >= SHOCK_YEAR else 0.0) + rng.gauss(0, 0.05))
        return 1 / (1 + math.exp(-4 * (x - 0.5)))

    cases = []
    for i in range(n_cases):
        d = corpus_rng.choice(dlist)
        year = corpus_rng.randint(year_min, year_max)
        s = dlatent(d, year, corpus_rng)
        cases.append(_synth_case_record(i, d, d2c[d], "district", year, s, corpus_rng))
    corpus_path = config.RAW / "synthetic_district_corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    import pandas as pd
    a_d = {d: housing_rng.gauss(0, 0.12) for d in dlist}
    d_t = {y: 0.01 * (y - year_min) + housing_rng.gauss(0, 0.03)
           for y in range(year_min, year_max + 1)}
    rows = []
    for d in dlist:
        for y in range(year_min, year_max + 1):
            s = dlatent(d, y, housing_rng)
            seg = 0.55 + a_d[d] + d_t[y] + BETA_TRUE * (s - 0.5) + housing_rng.gauss(0, 0.025)
            rows.append({"court_id": d, "circuit": d2c[d], "year": y,
                         "dissimilarity_index": round(seg, 4),
                         "_latent_strictness": round(s, 4)})
    panel_path = config.EXTERNAL / "synthetic_district_housing_panel.csv"
    pd.DataFrame(rows).to_csv(panel_path, index=False)
    return {"BETA_TRUE": BETA_TRUE, "n_cases": n_cases, "n_districts": len(dlist),
            "corpus": str(corpus_path), "housing_panel": str(panel_path)}


def generate(n_cases: int = 9000, year_min: int = 2000, year_max: int = 2023,
             seed: int = 7) -> dict:
    """Write a synthetic corpus + housing panel + ground-truth file.

    n_cases is sized so each circuit x year cell holds ~30 decisions. The FEII is
    a noisy proxy for latent doctrine, so it suffers
    measurement-error attenuation; ~30 cases/cell shrinks the sampling noise
    enough that the within-circuit TWFE recovers the embedded effect at p<0.05 by
    default. (On the *true* latent regressor the estimator returns exactly
    BETA_TRUE.) Lower n_cases for a faster, noisier smoke test.
    """
    # Independent RNG streams so that editing the corpus branch cannot perturb
    # the housing ground truth. The shared structural truth (circ_fe, shock) is
    # drawn from seed_rng and is identical regardless of n_cases.
    seed_rng = random.Random(seed)
    circ_fe = {c: seed_rng.uniform(0.30, 0.70) for c in CIRCUITS}  # circuit base
    # Post-2015 bump intensity correlates with pre-shock base doctrine, so the
    # EX-ANTE (pre-shock) treatment split in derive_treatment is informative
    # (circuits already inclined toward broad FHA doctrine responded more to
    # Inclusive Communities). This is what makes the *exogenous* DiD identify.
    shock = {c: max(0.0, 0.10 + 0.6 * (circ_fe[c] - 0.5) + seed_rng.uniform(0, 0.08))
             for c in CIRCUITS}                                    # post-2015 bump
    corpus_rng = random.Random(seed_rng.getrandbits(64))
    housing_rng = random.Random(seed_rng.getrandbits(64))
    rng = corpus_rng                      # the per-case loop below uses `rng`

    corpus_path = config.RAW / "synthetic_corpus.jsonl"
    cases = []
    for i in range(n_cases):
        circuit = rng.choice(CIRCUITS)
        year = rng.randint(year_min, year_max)
        court_id = rng.choice(_COURT_BY_CIRCUIT[circuit])
        level = "appellate" if court_id.startswith("ca") else "district"
        s = _latent_strictness(circuit, year, rng, circ_fe, shock)
        cases.append(_synth_case_record(i, court_id, circuit, level, year, s, rng))
    with corpus_path.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    # --- housing panel with KNOWN beta on latent strictness ---
    # Uses the INDEPENDENT housing_rng so corpus-size changes leave this fixed.
    import pandas as pd
    rows = []
    a_c = {c: housing_rng.gauss(0, 0.12) for c in CIRCUITS}
    d_t = {y: 0.01 * (y - year_min) + housing_rng.gauss(0, 0.03)
           for y in range(year_min, year_max + 1)}
    for c in CIRCUITS:
        for y in range(year_min, year_max + 1):
            s = _latent_strictness(c, y, housing_rng, circ_fe, shock)
            # baseline segregation index ~0.55, pushed down by enforcement
            seg = 0.55 + a_c[c] + d_t[y] + BETA_TRUE * (s - 0.5) + housing_rng.gauss(0, 0.025)
            rows.append({"circuit": c, "year": y,
                         "dissimilarity_index": round(seg, 4),
                         "median_rent": round(900 + 600 * (y - year_min) / 23
                                              + housing_rng.gauss(0, 60), 1),
                         "_latent_strictness": round(s, 4)})
    panel = pd.DataFrame(rows)
    panel_path = config.EXTERNAL / "synthetic_housing_panel.csv"
    panel.to_csv(panel_path, index=False)

    truth = {"BETA_TRUE": BETA_TRUE, "SHOCK_YEAR": SHOCK_YEAR,
             "n_cases": n_cases, "circ_fe": circ_fe, "shock": shock,
             "corpus": str(corpus_path), "housing_panel": str(panel_path)}
    truth_path = config.PROCESSED / "_synth_truth.json"
    truth_path.write_text(json.dumps(truth, indent=2))
    return truth


if __name__ == "__main__":
    t = generate()
    print(f"wrote {t['n_cases']} synthetic cases; BETA_TRUE={t['BETA_TRUE']}")
