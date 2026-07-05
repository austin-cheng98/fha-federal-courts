"""
FHA case identification. rule_is_fha() is the deterministic rule used for the
released corpus; an optional TF-IDF + logistic-regression classifier is
available but off by default (use_ml=False).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from . import config
from .reference import mentions_fha, find_fha_citations, NAMED_ACT

# Federal civil cover-sheet code for "Civil Rights: Housing/Accommodations".
HOUSING_NOS_CODES = {"443"}


def rule_is_fha(rec: dict) -> bool:
    """Layer A: high-precision rule. True if the case is plausibly FHA-substantive.

    A case qualifies if it (a) cites a core FHA section, OR (b) names the Act
    AND is not merely a passing mention -- we require either a housing cover
    sheet (NOS 443) or >=2 distinct FHA cues in the text.
    """
    text = rec.get("text", "") or ""
    nos = str(rec.get("nature_of_suit", "") or "")
    cites = find_fha_citations(text)
    if cites:
        return True
    named = bool(NAMED_ACT.search(text))
    if named and any(code in nos for code in HOUSING_NOS_CODES):
        return True
    # named + corroborating signal (claim/remedy language density)
    if named:
        from .reference import CLAIM_LEXICON, score_cues
        hits = sum(score_cues(text, pats) for pats in CLAIM_LEXICON.values())
        return hits >= 2
    return False


def weak_label_corpus(records: list[dict]) -> list[int]:
    """Apply the rule to a corpus -> 0/1 weak labels for training Layer B."""
    return [int(rule_is_fha(r)) for r in records]


@dataclass
class FHAClassifier:
    """TF-IDF + calibrated logistic regression for FHA-relevance (Layer B)."""
    threshold: float = 0.5
    _vec: object = None
    _clf: object = None

    def fit(self, texts: list[str], labels: list[int]) -> "FHAClassifier":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self._vec = TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_df=0.9,
            stop_words="english", max_features=50_000,
        )
        X = self._vec.fit_transform(texts)
        self._clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=4.0)
        self._clf.fit(X, labels)
        return self

    def predict_proba(self, texts: list[str]):
        X = self._vec.transform(texts)
        return self._clf.predict_proba(X)[:, 1]

    def predict(self, texts: list[str]) -> list[int]:
        return [int(p >= self.threshold) for p in self.predict_proba(texts)]

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or config.MODELS / "fha_classifier.pkl")
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: Path | None = None) -> "FHAClassifier":
        path = Path(path or config.MODELS / "fha_classifier.pkl")
        with path.open("rb") as fh:
            return pickle.load(fh)


def build_clean_corpus(records: list[dict], *, use_ml: bool = True,
                       min_text_len: int = 200) -> tuple[list[dict], dict]:
    """End-to-end: produce the cleaned FHA-only dataset + a report.

    With enough labeled data we train Layer B and keep cases the rule OR the
    model flags (union recall) but require the model's probability for borderline
    rule-negatives. With too little data we fall back to the rule alone.
    """
    rule = weak_label_corpus(records)
    report = {"n_input": len(records), "n_rule_positive": sum(rule)}
    texts = [r.get("text", "") or "" for r in records]

    kept = []
    if use_ml and sum(rule) >= 20 and (len(rule) - sum(rule)) >= 20:
        clf = FHAClassifier().fit(texts, rule)
        clf.save()
        proba = clf.predict_proba(texts)
        report["ml_trained"] = True
        for r, rl, p in zip(records, rule, proba):
            keep = bool(rl) or p >= 0.6
            if keep and len(r.get("text", "") or "") >= min_text_len:
                r = {**r, "fha_proba": round(float(p), 4), "fha_rule": rl}
                kept.append(r)
    else:
        report["ml_trained"] = False
        for r, rl in zip(records, rule):
            if rl and len(r.get("text", "") or "") >= min_text_len:
                kept.append({**r, "fha_proba": None, "fha_rule": rl})

    report["n_kept"] = len(kept)
    return kept, report
