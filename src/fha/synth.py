import json
import math
import random

import pandas as pd

from . import config

CIRCUITS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "DC"]
BETA_TRUE = -0.70
SHOCK_YEAR = 2015
COURTS = {
    "1": ["mad", "ca1"], "2": ["nysd", "ca2"], "3": ["njd", "ca3"],
    "4": ["mdd", "ca4"], "5": ["txsd", "ca5"], "6": ["ohnd", "ca6"],
    "7": ["ilnd", "ca7"], "8": ["mnd", "ca8"], "9": ["cand", "ca9"],
    "10": ["cod", "ca10"], "11": ["flsd", "ca11"], "DC": ["dcd", "cadc"],
}
INTRO = ("Plaintiff brings this action under the Fair Housing Act, 42 U.S.C. "
         "{sec}, alleging that defendant discriminated in housing. ")
CLAIMS = {
    "disparate_impact": ("Plaintiff advances a disparate impact theory, contending "
        "that defendant's facially neutral policy has a discriminatory effect on a "
        "protected class. Under 24 C.F.R. 100.500 and Texas Department of Housing v. "
        "Inclusive Communities Project, the burden-shifting framework requires plaintiff "
        "to make a prima facie showing of a robust causal connection. "),
    "disparate_treatment": ("Plaintiff alleges intentional discrimination, i.e. "
        "disparate treatment because of race. Applying the McDonnell Douglas "
        "burden-shifting framework, plaintiff must establish a prima facie case, after "
        "which defendant must articulate a legitimate, nondiscriminatory reason, and "
        "plaintiff may show pretext. "),
    "reasonable_accommodation": ("Plaintiff, who is handicapped within the meaning "
        "of the Act, seeks a reasonable accommodation under 42 U.S.C. 3604(f). The "
        "requested accommodation is necessary to afford an equal opportunity to use "
        "and enjoy the dwelling. "),
    "zoning_exclusionary": ("Plaintiff challenges the municipality's zoning ordinance "
        "and denial of a special use permit as exclusionary land use that makes housing "
        "unavailable. "),
    "refusal_rent_sell": ("Plaintiff alleges defendant refused to rent and engaged in "
        "steering and redlining, otherwise making a dwelling unavailable. "),
}
WIN = ("Accordingly, plaintiff's motion for summary judgment is granted and defendant's "
       "motion is denied. Judgment is entered in favor of the plaintiff. ")
LOSE = ("Accordingly, defendant's motion for summary judgment is granted and plaintiff's "
        "complaint is dismissed. Judgment is entered in favor of the defendant. ")
REMEDIES = {
    "injunction": "The Court orders permanent injunctive relief enjoining the challenged policy. ",
    "damages": "Plaintiff is awarded compensatory and punitive damages. ",
    "declaratory": "The Court issues declaratory relief declaring the policy unlawful. ",
}
FILLER = ("See 42 U.S.C. 3604; 42 U.S.C. 3605. The Court has considered the parties' "
          "briefs and the record. 994 F. Supp. 2d 1121. 135 S. Ct. 2507. ") * 4


def _latent(circuit, year, rng, bases, shocks):
    value = bases[circuit] + 0.015 * (year - 2000)
    value += shocks[circuit] if year >= SHOCK_YEAR else 0
    value += rng.gauss(0, 0.05)
    return 1 / (1 + math.exp(-4 * (value - 0.5)))


def _record(i, court, circuit, level, year, strictness, rng):
    claims = []
    if rng.random() < 0.25 + 0.45 * strictness:
        claims.append("disparate_impact")
    if rng.random() < 0.30:
        claims.append("disparate_treatment")
    if rng.random() < 0.20 + 0.25 * strictness:
        claims.append("reasonable_accommodation")
    if rng.random() < 0.18:
        claims.append("zoning_exclusionary")
    if rng.random() < 0.15:
        claims.append("refusal_rent_sell")
    if not claims:
        claims = ["disparate_treatment"]
    win = rng.random() < min(0.20 + 0.40 * strictness
                             + 0.12 * ("disparate_impact" in claims)
                             + 0.06 * ("reasonable_accommodation" in claims), 0.95)
    text = [INTRO.format(sec="3604")]
    text += [CLAIMS[claim] for claim in claims]
    text.append(WIN if win else LOSE)
    if win:
        for name, probability in (("injunction", 0.4 + 0.4 * strictness),
                                  ("damages", 0.5), ("declaratory", 0.3)):
            if rng.random() < probability:
                text.append(REMEDIES[name])
    text.append(FILLER)
    joined = " ".join(text)
    return {
        "cluster_id": 90_000_000 + i,
        "case_name": f"Synthetic Plaintiff {i} v. Defendant Housing Auth.",
        "court_id": court, "circuit": circuit, "court_level": level,
        "court_jurisdiction": "F", "date_filed": f"{year}-0{rng.randint(1, 9)}-15",
        "year": year, "docket_number": f"{year}-cv-{1000 + i}",
        "citations": ["994 F. Supp. 2d 1121"], "judges": "Smith",
        "precedential_status": "Published", "nature_of_suit": "443",
        "opinion_ids": [80_000_000 + i], "text": joined, "text_len": len(joined),
        "text_source": "synthetic", "source": "synthetic",
        "_latent_strictness": round(strictness, 4),
    }


def generate(n_cases=9000, year_min=2000, year_max=2023, seed=7):
    seed_rng = random.Random(seed)
    bases = {c: seed_rng.uniform(0.30, 0.70) for c in CIRCUITS}
    shocks = {c: max(0.0, 0.10 + 0.6 * (bases[c] - 0.5)
                       + seed_rng.uniform(0, 0.08)) for c in CIRCUITS}
    corpus_rng = random.Random(seed_rng.getrandbits(64))
    housing_rng = random.Random(seed_rng.getrandbits(64))
    cases = []
    for i in range(n_cases):
        circuit = corpus_rng.choice(CIRCUITS)
        year = corpus_rng.randint(year_min, year_max)
        court = corpus_rng.choice(COURTS[circuit])
        level = "appellate" if court.startswith("ca") else "district"
        cases.append(_record(i, court, circuit, level, year,
                             _latent(circuit, year, corpus_rng, bases, shocks), corpus_rng))
    corpus = config.RAW / "synthetic_corpus.jsonl"
    with corpus.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case) + "\n")

    rows = []
    unit_effect = {c: housing_rng.gauss(0, 0.12) for c in CIRCUITS}
    time_effect = {y: 0.01 * (y - year_min) + housing_rng.gauss(0, 0.03)
                   for y in range(year_min, year_max + 1)}
    for circuit in CIRCUITS:
        for year in range(year_min, year_max + 1):
            strictness = _latent(circuit, year, housing_rng, bases, shocks)
            outcome = (0.55 + unit_effect[circuit] + time_effect[year]
                       + BETA_TRUE * (strictness - 0.5)
                       + housing_rng.gauss(0, 0.025))
            rows.append({"circuit": circuit, "year": year,
                         "dissimilarity_index": round(outcome, 4),
                         "median_rent": round(900 + 600 * (year - year_min) / 23
                                              + housing_rng.gauss(0, 60), 1),
                         "_latent_strictness": round(strictness, 4)})
    housing = config.EXTERNAL / "synthetic_housing_panel.csv"
    pd.DataFrame(rows).to_csv(housing, index=False)
    truth = {"BETA_TRUE": BETA_TRUE, "corpus": str(corpus),
             "housing_panel": str(housing)}
    (config.PROCESSED / "_synth_truth.json").write_text(json.dumps(truth, indent=2))
    return truth
