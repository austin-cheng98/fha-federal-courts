"""
Structured legal-variable extraction: textual indicators, burden-shifting
framework, directional outcome cues, remedies, and derived enforcement and
legal-signal scores. Deterministic lexicon/rule based, with an optional
negation-scoping mode and inspectable match spans in the feature output.
"""
from __future__ import annotations

import pickle
import re
import json
from pathlib import Path

import pandas as pd

from . import config
from .reference import (
    CLAIM_LEXICON, REASONING_LEXICON, REMEDY_LEXICON,
    PLAINTIFF_WIN_CUES, DEFENDANT_WIN_CUES, REVERSAL_CUES, AFFIRM_CUES,
    SETTLEMENT_INFERENCE_CUES, find_fha_citations, score_cues, compile_lexicon,
    normalize_text,
)

_CLAIM_RX = compile_lexicon(CLAIM_LEXICON)
_REASON_RX = compile_lexicon(REASONING_LEXICON)
_REMEDY_RX = compile_lexicon(REMEDY_LEXICON)

# Optional negation scoping (off by default): a lexicon hit is discarded when a
# negator appears within the same clause, up to ~70 chars upstream.
_NEG_RX = re.compile(
    r"\b(?:not|no|never|without|nor|neither|fail(?:s|ed)?\s+to|"
    r"d(?:oes|id|o)\s+not|cannot|declin(?:e|es|ed)\s+to|waived?|"
    r"abandon(?:s|ed)?|withdr(?:ew|awn)|no\s+longer|absent)"
    r"\b[^.;:\n]{0,70}$", re.IGNORECASE)


def _hit(rx, text: str, negation: bool = False) -> bool:
    """Presence test; with negation=True, a match preceded by a same-clause
    negator does not count, and the flag fires only on a non-negated match."""
    if not text:
        return False
    if not negation:
        return bool(rx.search(text))
    for m in rx.finditer(text):
        if not _NEG_RX.search(text[max(0, m.start() - 90):m.start()]):
            return True
    return False


def _span_hits(rx, text: str, negation: bool = False) -> list[dict]:
    """Return inspectable character offsets for the rule hits used in a flag."""
    out = []
    for m in rx.finditer(text or ""):
        if negation and _NEG_RX.search(text[max(0, m.start() - 90):m.start()]):
            continue
        out.append({"start": m.start(), "end": m.end(),
                    "text": m.group(0)[:160]})
    return out


def _json_spans(spans: list[dict]) -> str:
    return json.dumps(spans, ensure_ascii=False, separators=(",", ":"))


# Textual indicators. These are deliberately multi-label and are separated
# below into proof/duty indicators and conduct indicators; they are not a
# mutually exclusive cause-of-action taxonomy.
def extract_claims(text: str, negation: bool = False) -> dict[str, int]:
    out = {}
    for k, rx in _CLAIM_RX.items():
        out[f"claim_{k}"] = int(_hit(rx, text, negation))
        out[f"claim_{k}_spans"] = _json_spans(_span_hits(rx, text, negation))
    return out


# reasoning structure
def extract_reasoning(text: str, negation: bool = False) -> dict:
    flags = {}
    evidence = {}
    for k, rx in _REASON_RX.items():
        flags[f"reason_{k}"] = int(_hit(rx, text, negation))
        evidence[f"reason_{k}_spans"] = _json_spans(_span_hits(rx, text, negation))
    # burden-shifting framework actually invoked
    if flags["reason_hud_burden_shifting"]:
        framework = "hud_three_step"
    elif flags["reason_mcdonnell_douglas"]:
        framework = "mcdonnell_douglas"
    else:
        framework = "none_explicit"
    # proof standard
    if flags["reason_heightened_proof"]:
        standard = "clear_and_convincing"
    elif flags["reason_preponderance"]:
        standard = "preponderance"
    else:
        standard = "unstated"
    # precedent treatment: density of reporter citations in the opinion body
    # count reporter citations across the common federal reporters
    n_cites = len(re.findall(
        r"\b\d{1,4}\s+(?:F\.(?:\s?(?:2d|3d|4th|App'?x|Supp\.?(?:\s?[23]d)?))?|"
        r"U\.?\s?S\.?|S\.?\s?Ct\.|L\.?\s?Ed\.(?:\s?2d)?|Fed\.?\s?App)",
        text or "", re.IGNORECASE))
    if n_cites >= 25:
        precedent = "explicit_heavy"
    elif n_cites >= 5:
        precedent = "explicit_moderate"
    else:
        precedent = "sparse"
    return {
        **flags,
        "burden_framework": framework,
        "proof_standard": standard,
        "precedent_treatment": precedent,
        "n_precedent_cites": n_cites,
        **evidence,
    }


# outcomes & remedies
def _tail(text: str, frac: float = 0.35) -> str:
    """Holdings cluster at the end; weight cue scoring toward the tail."""
    if not text:
        return ""
    cut = int(len(text) * (1 - frac))
    return text[cut:]


def extract_outcomes(text: str, court_level: str | None = None,
                     negation: bool = False) -> dict:
    tail = _tail(text)
    pro_p = score_cues(tail, PLAINTIFF_WIN_CUES) + 0.5 * score_cues(text, PLAINTIFF_WIN_CUES)
    pro_d = score_cues(tail, DEFENDANT_WIN_CUES) + 0.5 * score_cues(text, DEFENDANT_WIN_CUES)
    # Mixed dispositions ("granted in part and denied in part") are common in FHA
    # rulings; scoring them as a clean win is wrong, so flag + leave undetermined.
    mixed = bool(re.search(r"\bin\s+part\b", tail, re.IGNORECASE)) and pro_p > 0 and pro_d > 0
    if mixed or pro_p == pro_d == 0:
        outcome_cue = None               # undetermined / genuinely split
    else:
        outcome_cue = int(pro_p > pro_d)

    affirm = score_cues(tail, AFFIRM_CUES)
    reverse = score_cues(tail, REVERSAL_CUES)
    reversal = None
    if court_level == "appellate" and (affirm or reverse):
        if affirm and reverse:           # affirmed in part / reversed in part
            reversal = None
        else:
            reversal = int(reverse > affirm)

    remedies = {}
    for k, rx in _REMEDY_RX.items():
        remedies[f"remedy_{k}"] = int(_hit(rx, text, negation))
        remedies[f"remedy_{k}_spans"] = _json_spans(_span_hits(rx, text, negation))
    settlement = int(score_cues(text, SETTLEMENT_INFERENCE_CUES) > 0)
    return {
        # This is a directional text cue, not a party-role-resolved judgment.
        "outcome_cue": outcome_cue,
        "plaintiff_win": outcome_cue,   # legacy alias; not used by FEII
        "disposition_mixed": int(mixed),
        "pro_plaintiff_cues": round(pro_p, 2),
        "pro_defendant_cues": round(pro_d, 2),
        "appellate_reversal": reversal,
        **remedies,
        "settlement_inferred": settlement,
        "outcome_evidence": json.dumps({
            "plaintiff": [x for pats in PLAINTIFF_WIN_CUES
                          for x in _span_hits(re.compile(pats, re.IGNORECASE), tail)],
            "defendant": [x for pats in DEFENDANT_WIN_CUES
                          for x in _span_hits(re.compile(pats, re.IGNORECASE), tail)],
        }, ensure_ascii=False, separators=(",", ":")),
    }


# derived scalars
def enforcement_strength(row: dict) -> float:
    """Per-case pro-enforcement proxy in [0,1]. Feeds FEII."""
    s = 0.0
    if row.get("outcome_cue") == 1:
        s += 0.45
    # broader remedies => stronger enforcement
    s += 0.12 * row.get("remedy_injunction", 0)
    s += 0.08 * row.get("remedy_damages", 0)
    s += 0.05 * row.get("remedy_declaratory", 0)
    s += 0.05 * row.get("remedy_civil_penalty", 0)
    # recognizing disparate impact is the strong-enforcement doctrine
    s += 0.15 * row.get("claim_disparate_impact", 0)
    s += 0.10 * row.get("reason_hud_burden_shifting", 0)
    return round(min(s, 1.0), 3)


def doctrinal_breadth_score(row: dict) -> float:
    """Rule-based legal-signal score, not a court-favorability measure.

    The score summarizes detected liability theories, burden-framework terms,
    accommodation language, and precedent engagement. It is correlated with
    procedural posture and is therefore described as a breadth/legal-signal
    score rather than pure doctrinal strictness.
    """
    s = 0.0
    s += 0.35 * row.get("claim_disparate_impact", 0)     # broad liability theory
    s += 0.25 * row.get("reason_hud_burden_shifting", 0)
    s += 0.20 * row.get("claim_reasonable_accommodation", 0)
    s += 0.10 * (1 if row.get("precedent_treatment") == "explicit_heavy" else 0)
    s += 0.10 * (1 if row.get("burden_framework") != "none_explicit" else 0)
    return round(min(s, 1.0), 3)


def doctrinal_strictness(row: dict) -> float:
    """Backward-compatible alias for :func:`doctrinal_breadth_score`."""
    return doctrinal_breadth_score(row)


# main entry points
def extract_case(rec: dict, negation: bool = False) -> dict:
    """Full Step-3 feature row for one case."""
    # normalize typographic quotes once so every cue regex matches real text
    text = normalize_text(rec.get("text", "") or "")
    out = {
        # metadata
        "cluster_id": rec.get("cluster_id"),
        "case_name": rec.get("case_name", ""),
        "court_id": rec.get("court_id", ""),
        "circuit": rec.get("circuit"),
        "court_level": rec.get("court_level"),
        "year": rec.get("year"),
        "judges": rec.get("judges", ""),
        "precedential_status": rec.get("precedential_status", ""),
        "nature_of_suit": rec.get("nature_of_suit", ""),
        "fha_sections": ",".join(find_fha_citations(text)),
        "text_source": rec.get("text_source", "none"),
        "text_len": rec.get("text_len", len(text)),
    }
    out.update(extract_claims(text, negation))
    # Keep the analytic layers explicit even though the source regexes remain
    # backward-compatible under claim_* names.
    out["proof_treatment"] = out["claim_disparate_treatment"]
    out["proof_impact"] = out["claim_disparate_impact"]
    out["duty_accommodation"] = out["claim_reasonable_accommodation"]
    out["conduct_refusal_steering"] = out["claim_refusal_rent_sell"]
    out["conduct_zoning_land_use"] = out["claim_zoning_exclusionary"]
    out.update(extract_reasoning(text, negation))
    out.update(extract_outcomes(text, rec.get("court_level"), negation))
    out["enforcement_strength"] = enforcement_strength(out)
    out["doctrinal_breadth"] = doctrinal_breadth_score(out)
    out["legal_signal_score"] = out["doctrinal_breadth"]
    # Retain the legacy column so existing downstream scripts continue to run.
    out["doctrinal_strictness"] = out["doctrinal_breadth"]
    return out


def extract_corpus(records: list[dict], negation: bool = False) -> pd.DataFrame:
    return pd.DataFrame(extract_case(r, negation) for r in records)


# ML version -- OneVsRest multi-label claim classifier
class ClaimClassifier:
    """TF-IDF + OneVsRest logistic regression for the 5 claim labels."""

    def __init__(self):
        self.labels = [f"claim_{k}" for k in CLAIM_LEXICON]
        self._vec = None
        self._clf = None

    def fit(self, texts: list[str], Y) -> "ClaimClassifier":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        self._vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                    min_df=2, stop_words="english", max_features=40_000)
        X = self._vec.fit_transform(texts)
        self._clf = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, class_weight="balanced"))
        self._clf.fit(X, Y)
        return self

    def predict(self, texts: list[str]):
        return self._clf.predict(self._vec.transform(texts))

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or config.MODELS / "claim_classifier.pkl")
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path
