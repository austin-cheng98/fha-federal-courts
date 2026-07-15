from .reference import (
    CLAIM_LEXICON,
    NAMED_ACT,
    REMEDY_LEXICON,
    find_fha_citations,
    is_nos443,
    score_cues,
)


def rule_is_fha(record):
    text = record.get("text", "") or ""
    nos = record.get("nature_of_suit", "")
    if find_fha_citations(text):
        return True
    if not NAMED_ACT.search(text):
        return False
    if is_nos443(nos):
        return True
    hits = sum(score_cues(text, patterns) for patterns in CLAIM_LEXICON.values())
    hits += sum(score_cues(text, patterns) for patterns in REMEDY_LEXICON.values())
    return hits >= 2


def build_clean_corpus(records, min_text_len=200, require_nos443=True):
    labels = [int(rule_is_fha(record)) for record in records]
    nos = [int(is_nos443(record.get("nature_of_suit"))) for record in records]
    if require_nos443:
        labels = [label and code for label, code in zip(labels, nos)]
    clean = []
    for record, label in zip(records, labels):
        if label and len(record.get("text", "") or "") >= min_text_len:
            clean.append({**record, "fha_rule": int(label)})
    return clean, {
        "n_input": len(records),
        "n_nos443": sum(nos),
        "n_rule_positive": sum(labels),
        "n_kept": len(clean),
    }
