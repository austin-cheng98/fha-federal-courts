"""
Shared reference data: court-id -> circuit map, FHA citation patterns, and
the claim / reasoning / outcome / remedy lexicons used by the rule-based
extractors.
"""
from __future__ import annotations

import re

# CourtListener court id  ->  circuit
# Circuit is a string ("DC"/"Federal" stay readable). District ids are
# CourtListener slugs; appellate ids are "ca1".."ca11", "cadc", "cafc", "scotus".

_CIRCUIT_DISTRICTS: dict[str, list[str]] = {
    "1":  ["med", "mad", "nhd", "rid", "prd"],
    "2":  ["ctd", "nynd", "nyed", "nysd", "nywd", "vtd"],
    "3":  ["ded", "njd", "paed", "pamd", "pawd", "vid"],
    "4":  ["mdd", "nced", "ncmd", "ncwd", "scd", "vaed", "vawd", "wvnd", "wvsd"],
    "5":  ["laed", "lamd", "lawd", "msnd", "mssd", "txnd", "txed", "txsd", "txwd"],
    "6":  ["kyed", "kywd", "mied", "miwd", "ohnd", "ohsd", "tned", "tnmd", "tnwd"],
    "7":  ["ilnd", "ilcd", "ilsd", "innd", "insd", "wied", "wiwd"],
    "8":  ["ared", "arwd", "iand", "iasd", "mnd", "moed", "mowd", "ned", "ndd", "sdd"],
    "9":  ["akd", "azd", "cand", "caed", "cacd", "casd", "hid", "idd", "mtd",
           "nvd", "ord", "waed", "wawd", "gud", "nmid"],
    "10": ["cod", "ksd", "nmd", "oknd", "oked", "okwd", "utd", "wyd"],
    "11": ["alnd", "almd", "alsd", "flnd", "flmd", "flsd", "gand", "gamd", "gasd"],
    "DC": ["dcd"],
}

# Build district -> circuit lookup.
COURT_TO_CIRCUIT: dict[str, str] = {}
for _circ, _courts in _CIRCUIT_DISTRICTS.items():
    for _c in _courts:
        COURT_TO_CIRCUIT[_c] = _circ
# Appellate courts.
for _n in range(1, 12):
    COURT_TO_CIRCUIT[f"ca{_n}"] = str(_n)
COURT_TO_CIRCUIT["cadc"] = "DC"
COURT_TO_CIRCUIT["cafc"] = "Federal"
COURT_TO_CIRCUIT["scotus"] = "SCOTUS"

APPELLATE_COURTS = {f"ca{n}" for n in range(1, 12)} | {"cadc", "cafc"}
SUPREME_COURT = "scotus"


def court_to_circuit(court_id: str | None) -> str | None:
    """Map a CourtListener court id to a circuit label, else None."""
    if not court_id:
        return None
    return COURT_TO_CIRCUIT.get(court_id.strip().lower())


def court_level(court_id: str | None) -> str | None:
    """Return 'district', 'appellate', 'supreme', or None."""
    if not court_id:
        return None
    cid = court_id.strip().lower()
    if cid == SUPREME_COURT:
        return "supreme"
    if cid in APPELLATE_COURTS:
        return "appellate"
    if cid in COURT_TO_CIRCUIT:
        return "district"
    return None


# FHA statutory citation patterns
# FHA = 42 U.S.C. 3601-3619 plus 3631; matched by section number or named act.
# A bare "FHA" is not a hit (in housing-finance opinions it usually means the
# Federal Housing Administration).

# Section numbers that make a case substantively about FHA enforcement.
FHA_CORE_SECTIONS = ["3601", "3602", "3603", "3604", "3605", "3606",
                     "3607", "3608", "3613", "3614", "3617", "3631"]

# "42 U.S.C. (section) 3604", "42 USC 3604(a)", "Section 3604", "sec. 3617"
_SECTION_NUMS = "|".join(FHA_CORE_SECTIONS)
CITE_USC = re.compile(
    r"42\s*U\.?\s?S\.?\s?C\.?\s*(?:section|sec\.?|§{1,2})?\s*(36\d{2})",
    re.IGNORECASE,
)
CITE_SECTION_BARE = re.compile(
    rf"(?:§{{1,2}}|section|sec\.)\s*({_SECTION_NUMS})\b", re.IGNORECASE
)
NAMED_ACT = re.compile(
    r"fair\s+housing\s+(?:act|amendments\s+act)|\btitle\s+viii\b|"
    r"\bF\.?H\.?A\.?A\.?\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Fold typographic quotes/apostrophes to ASCII so the cue regexes match.
    CourtListener opinions routinely use U+2019 etc., which break possessive
    cues like r"plaintiff'?s?". Call once before any extraction."""
    if not text:
        return text
    return text.translate({0x2019: "'", 0x2018: "'", 0x02BC: "'",
                           0x201C: '"', 0x201D: '"'})


def find_fha_citations(text: str) -> list[str]:
    """Return the sorted set of FHA section numbers cited in `text`.

    The USC branch is filtered to FHA_CORE_SECTIONS. The bare-section branch
    ("section 3604") only counts when a statutory anchor (U.S.C. / the named Act)
    is nearby, so a stray "section 3617" from a different title or a state code is
    not mistaken for an FHA citation.
    """
    if not text:
        return []
    hits: set[str] = set()
    for m in CITE_USC.finditer(text):
        if m.group(1) in FHA_CORE_SECTIONS:
            hits.add(m.group(1))
    named = NAMED_ACT.search(text) is not None
    for m in CITE_SECTION_BARE.finditer(text):
        lo, hi = max(0, m.start() - 40), min(len(text), m.end() + 40)
        window = text[lo:hi]
        if named or re.search(r"u\.?\s?s\.?\s?c\.?", window, re.IGNORECASE):
            hits.add(m.group(1))
    return sorted(hits)


def mentions_fha(text: str) -> bool:
    """True if the text names the Act or cites a core FHA section."""
    if not text:
        return False
    return bool(NAMED_ACT.search(text)) or bool(find_fha_citations(text))


# claim-type lexicons (multi-label)
# Each pattern is compiled case-insensitive. A case can carry several labels.
CLAIM_LEXICON: dict[str, list[str]] = {
    "disparate_treatment": [
        r"disparate\s+treatment", r"intentional\s+discrimination",
        r"discriminatory\s+intent", r"because\s+of\s+(?:his|her|their)\s+race",
    ],
    "disparate_impact": [
        r"disparate\s+impact", r"discriminatory\s+effect",
        r"inclusive\s+communities", r"\bdisproportionate\w*\s+(?:impact|effect)",
    ],
    "zoning_exclusionary": [
        r"\bzoning\b", r"exclusionary", r"\bland\s+use\b", r"special\s+(?:use\s+)?permit",
        r"\bvariance\b", r"\bgroup\s+home\b", r"comprehensive\s+plan",
    ],
    "refusal_rent_sell": [
        r"refus\w*\s+to\s+(?:rent|sell|lease|negotiate)", r"\bsteering\b",
        r"\bredlining\b", r"made\s+unavailable", r"otherwise\s+made\s+unavailable",
    ],
    "reasonable_accommodation": [
        r"reasonable\s+accommodation", r"reasonable\s+modification",
        r"\bhandicap\w*\b", r"\bdisabilit\w+\b", r"\b3604\s*\(\s*f\s*\)",
        r"emotional\s+support\s+animal", r"assistance\s+animal",
    ],
}


# legal-reasoning lexicons
REASONING_LEXICON: dict[str, list[str]] = {
    "mcdonnell_douglas": [
        r"mcdonnell\s+douglas", r"burden[-\s]shifting", r"prima\s+facie\s+case",
        r"legitimate,?\s+nondiscriminatory", r"\bpretext\w*\b",
    ],
    "hud_burden_shifting": [  # 24 CFR 100.500 three-step impact framework
        r"24\s*C\.?F\.?R\.?\s*(?:section|§)?\s*100\.500",
        r"robust\s+caus\w+", r"legitimate,?\s+substantial,?\s+nondiscriminatory\s+interest",
    ],
    "summary_judgment": [
        r"summary\s+judgment", r"genuine\s+(?:issue|dispute)\s+of\s+material\s+fact",
        r"rule\s+56", r"no\s+genuine\s+dispute",
    ],
    "preponderance": [r"preponderance\s+of\s+the\s+evidence"],
    "heightened_proof": [r"clear\s+and\s+convincing"],
    "standing_threshold": [r"\bstanding\b", r"injury[-\s]in[-\s]fact", r"article\s+iii"],
}


# outcome & remedy lexicons
# Plaintiff- vs defendant-favorable disposition cues, scored (not hard-matched)
# and weighted toward the opinion tail. Affirm/reverse are excluded here: a
# reversal favors whoever lost below, so direction cannot map to a prevailing
# party without appellant identity (captured separately via REVERSAL_CUES/AFFIRM_CUES).
PLAINTIFF_WIN_CUES = [
    r"judgment\s+(?:is\s+)?(?:entered\s+)?(?:for|in\s+favor\s+of)\s+(?:the\s+)?plaintiff",
    r"plaintiff'?s?\s+motion\s+(?:for\s+summary\s+judgment\s+)?is\s+granted",
    r"defendant'?s?\s+motion\s+(?:to\s+dismiss|for\s+summary\s+judgment)\s+is\s+denied",
    r"find\w*\s+(?:for|in\s+favor\s+of)\s+(?:the\s+)?plaintiff",
]
DEFENDANT_WIN_CUES = [
    r"judgment\s+(?:is\s+)?(?:entered\s+)?(?:for|in\s+favor\s+of)\s+(?:the\s+)?defendant",
    r"defendant'?s?\s+motion\s+(?:to\s+dismiss|for\s+summary\s+judgment)\s+is\s+granted",
    r"plaintiff'?s?\s+(?:complaint|claims?)\s+(?:(?:is|are|was|were)\s+)?(?:hereby\s+)?dismissed",
    r"motion\s+to\s+dismiss\s+is\s+granted",
    r"summary\s+judgment\s+(?:is\s+)?granted\s+(?:to|in\s+favor\s+of)\s+defendant",
]
# Operative appellate action only -- anchored so they don't fire on lower-court
# history ("the district court affirmed") or string-cites to other opinions.
REVERSAL_CUES = [r"we\s+(?:hereby\s+)?(?:reverse|vacate)",
                 r"(?:judgment|order)\s+(?:is|are)\s+(?:reversed|vacated)",
                 r"reversed\s+and\s+remanded"]
AFFIRM_CUES = [r"we\s+(?:hereby\s+)?affirm",
               r"(?:judgment|order|decision)\s+(?:is|are)\s+affirmed"]

REMEDY_LEXICON: dict[str, list[str]] = {
    "damages": [r"\bdamages\b", r"compensatory", r"punitive\s+damages",
                r"monetary\s+(?:relief|award)"],
    "injunction": [r"\binjunction\b", r"injunctive\s+relief", r"enjoin\w*",
                   r"permanent\s+injunction", r"preliminary\s+injunction"],
    "declaratory": [r"declaratory\s+(?:relief|judgment)", r"declare\w*\s+that"],
    "attorneys_fees": [r"attorney'?s?'?\s+fees", r"\bcosts\s+and\s+fees\b"],
    "civil_penalty": [r"civil\s+penalt\w+"],
}

# Dismissal types that often imply private settlement.
SETTLEMENT_INFERENCE_CUES = [
    r"stipulat\w+\s+(?:of\s+)?dismissal", r"dismiss\w*\s+with\s+prejudice\s+pursuant\s+to",
    r"settlement\s+agreement", r"consent\s+decree", r"voluntar\w+\s+dismiss\w+",
    r"rule\s+41\s*\(\s*a\s*\)",
]


# landmark FHA shocks for event studies
# (year, label, scope): scope = "national" for SCOTUS, else the circuit string.
FHA_SHOCKS: list[dict] = [
    {"year": 2015, "label": "Inclusive Communities (disparate impact upheld)",
     "scope": "national", "kind": "scotus"},
    {"year": 1988, "label": "Fair Housing Amendments Act (disability + families)",
     "scope": "national", "kind": "statute"},
    {"year": 2013, "label": "HUD disparate-impact rule (24 CFR 100.500)",
     "scope": "national", "kind": "regulation"},
    {"year": 2020, "label": "HUD 2020 disparate-impact rule revision",
     "scope": "national", "kind": "regulation"},
]


def compile_lexicon(lex: dict[str, list[str]]) -> dict[str, re.Pattern]:
    """Compile a {label: [patterns]} lexicon into {label: combined regex}."""
    return {
        label: re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE)
        for label, pats in lex.items()
    }


def score_cues(text: str, patterns: list[str]) -> int:
    """Count how many distinct cue patterns fire in `text`."""
    if not text:
        return 0
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
