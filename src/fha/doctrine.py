"""
Doctrinal regimes: TF-IDF + LSA embeddings, KMeans clustering (clusters
named weak/moderate/strict by mean legal-signal breadth), and the circuit-year
doctrine map, transitions, and cross-circuit divergence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# embeddings
def embed(texts: list[str], backend: str | None = None, dim: int = 100) -> np.ndarray:
    """Return a dense doctrinal embedding matrix (n_docs x dim)."""
    backend = backend or config.SETTINGS.embedding_backend
    if backend == "legalbert":
        return _embed_legalbert(texts)
    return _embed_tfidf_lsa(texts, dim=dim)


def _embed_tfidf_lsa(texts: list[str], dim: int = 100) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                          max_df=0.9, stop_words="english", max_features=50_000)
    X = vec.fit_transform(texts)
    k = min(dim, X.shape[1] - 1, max(2, X.shape[0] - 1))
    svd = TruncatedSVD(n_components=k, random_state=0)
    return normalize(svd.fit_transform(X))


def _embed_legalbert(texts: list[str]) -> np.ndarray:
    """Optional high-capacity backend. Needs torch + transformers (+ a GPU to be
    fast). Mean-pools legal-BERT token states over 512-token windows."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "legal-BERT backend needs `pip install torch transformers`. "
            "Use embedding_backend='tfidf' to run without them."
        ) from e
    name = config.SETTINGS.legalbert_model
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = texts[i:i + 16]
            enc = tok(batch, padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(pooled.cpu().numpy())
    return np.vstack(out)


# doctrinal clustering -> named regimes
REGIME_NAMES = ["weak", "moderate", "strict"]


def cluster_regimes(emb: np.ndarray, legal_signal: np.ndarray,
                    n_regimes: int | None = None, seed: int = 0
                    ) -> tuple[np.ndarray, dict]:
    """KMeans on embeddings; order clusters by mean legal-signal breadth.

    ``legal_signal`` is the primary name; the historical strictness naming is
    retained only in the compatibility key returned below.
    """
    from sklearn.cluster import KMeans
    n = n_regimes or config.SETTINGS.n_doctrinal_regimes
    n = min(n, max(2, emb.shape[0]))
    km = KMeans(n_clusters=n, n_init=10, random_state=seed)
    raw = km.fit_predict(emb)
    # order clusters by mean legal-signal score, assign names from weak->strict
    order = (pd.Series(legal_signal).groupby(raw).mean().sort_values().index.tolist())
    names = REGIME_NAMES if n == 3 else [f"regime_{i}" for i in range(n)]
    mapping = {cl: (names[i] if n == 3 else f"regime_{i}")
               for i, cl in enumerate(order)}
    labels = np.array([mapping[c] for c in raw])
    info = {"mapping": {int(k): v for k, v in mapping.items()},
            "sizes": pd.Series(labels).value_counts().to_dict(),
            "regime_legal_signal":
                pd.Series(legal_signal).groupby(labels).mean().round(3).to_dict()}
    info["mean_strictness_by_regime"] = info["regime_legal_signal"]
    return labels, info


# circuit doctrinal map
def doctrine_map(df: pd.DataFrame) -> pd.DataFrame:
    """circuit x year regime composition. df needs columns circuit, year, regime."""
    g = (df.groupby(["circuit", "year", "regime"]).size()
           .unstack("regime", fill_value=0))
    shares = g.div(g.sum(axis=1), axis=0)
    shares["n_cases"] = g.sum(axis=1)
    if "strict" in shares:
        shares["share_strict"] = shares["strict"]
    # within-cell doctrinal entropy = dispersion of regimes
    regime_cols = [c for c in g.columns]
    p = shares[regime_cols].clip(lower=1e-12)
    shares["regime_entropy"] = (-(p * np.log(p)).sum(axis=1)).round(3)
    return shares.reset_index()


def regime_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year transitions of each circuit's dominant regime (4.3 edges)."""
    dom = (df.groupby(["circuit", "year", "regime"]).size()
             .reset_index(name="n")
             .sort_values("n").drop_duplicates(["circuit", "year"], keep="last")
             .sort_values(["circuit", "year"]))
    dom["prev_regime"] = dom.groupby("circuit")["regime"].shift(1)
    dom["transition"] = (dom["prev_regime"].fillna("") + " -> " + dom["regime"])
    dom["changed"] = (dom["prev_regime"].notna() &
                      (dom["prev_regime"] != dom["regime"])).astype(int)
    return dom[["circuit", "year", "regime", "prev_regime", "transition", "changed"]]


def cross_circuit_divergence(map_df: pd.DataFrame) -> pd.DataFrame:
    """Per-year divergence across circuits: variance in share_strict + mean
    entropy. High divergence = circuits interpret the FHA differently (4.3)."""
    if "share_strict" not in map_df:
        map_df = map_df.assign(share_strict=0.0)
    out = (map_df.groupby("year")
           .agg(share_strict_var=("share_strict", "var"),
                share_strict_mean=("share_strict", "mean"),
                mean_entropy=("regime_entropy", "mean"),
                n_circuits=("circuit", "nunique"))
           .reset_index())
    return out.round(4)


def run_step4(features: pd.DataFrame, texts: list[str],
              backend: str | None = None) -> dict:
    """Glue: embed -> cluster -> attach regime -> build maps. Returns frames."""
    emb = embed(texts, backend=backend)
    signal_col = "legal_signal_score" if "legal_signal_score" in features else "doctrinal_strictness"
    labels, info = cluster_regimes(emb, features[signal_col].to_numpy())
    feat = features.copy()
    feat["regime"] = labels
    cmap = doctrine_map(feat)
    trans = regime_transitions(feat)
    diverge = cross_circuit_divergence(cmap)
    return {"features": feat, "embeddings": emb, "cluster_info": info,
            "doctrine_map": cmap, "transitions": trans, "divergence": diverge}
