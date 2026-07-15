import json
from pathlib import Path

import pandas as pd

from . import config, doctrine, econometrics, feii, reports, synth
from .classify import build_clean_corpus
from .extract import extract_corpus
from .housing import load_housing_panel


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")
            if line.strip()]


def run(source="paper"):
    if source == "paper":
        records = _load_jsonl(config.PROCESSED / "paper_corpus.jsonl")
        housing_path = config.EXTERNAL / "housing_panel.csv"
        require_nos443 = True
    elif source == "synthetic":
        truth = synth.generate()
        records = _load_jsonl(truth["corpus"])
        housing_path = truth["housing_panel"]
        require_nos443 = False
    else:
        raise ValueError("source must be 'paper' or 'synthetic'")

    clean, identification = build_clean_corpus(
        records, require_nos443=require_nos443)
    if not clean:
        raise ValueError("no usable opinion records")
    features = extract_corpus(clean)
    features = features[features.circuit.notna() & features.year.notna()].reset_index(drop=True)
    features.to_csv(config.PROCESSED / "case_features.csv", index=False)

    texts = [r.get("text", "") for r in clean]
    step4 = doctrine.run_step4(features, texts)
    step4["features"].to_csv(config.PROCESSED / "case_features.csv", index=False)
    step4["doctrine_map"].to_csv(config.PROCESSED / "doctrine_map.csv", index=False)
    step4["transitions"].to_csv(config.PROCESSED / "doctrine_transitions.csv", index=False)
    step4["divergence"].to_csv(config.PROCESSED / "doctrine_divergence.csv", index=False)

    panel_feii = feii.aggregate(step4["features"], unit="circuit")
    panel_feii.to_csv(config.PROCESSED / "feii_panel.csv", index=False)
    housing = load_housing_panel(housing_path)
    panel = econometrics.build_panel(panel_feii, housing)
    panel.to_csv(config.PROCESSED / "analysis_panel.csv", index=False)
    twfe = econometrics.twfe(panel, y="dissimilarity_index")

    step4["doctrine_map"].to_csv(config.TABLES / "doctrinal_map.csv", index=False)
    step4["divergence"].to_csv(config.TABLES / "doctrinal_divergence.csv", index=False)
    reports.write_summary(len(features), len(panel_feii), len(housing), twfe)
    return {
        "source": source,
        "identification": identification,
        "cases": len(features),
        "feii_cells": len(panel_feii),
        "housing_rows": len(housing),
        "twfe": twfe,
        "outputs": str(config.OUTPUTS),
    }
