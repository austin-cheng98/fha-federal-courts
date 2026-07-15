import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def embed(texts, dim=100):
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, ngram_range=(1, 2), min_df=2,
        max_df=0.9, stop_words="english", max_features=50_000)
    matrix = vectorizer.fit_transform(texts)
    components = min(dim, matrix.shape[1] - 1, max(2, matrix.shape[0] - 1))
    return normalize(TruncatedSVD(n_components=components, random_state=0)
                     .fit_transform(matrix))


def cluster_regimes(embeddings, signal, n_regimes=3):
    model = KMeans(n_clusters=n_regimes, n_init=10, random_state=0)
    raw = model.fit_predict(embeddings)
    order = pd.Series(signal).groupby(raw).mean().sort_values().index.tolist()
    names = ["weak", "moderate", "strict"]
    mapping = {cluster: names[i] for i, cluster in enumerate(order)}
    labels = np.array([mapping[cluster] for cluster in raw])
    means = pd.Series(signal).groupby(labels).mean().round(3).to_dict()
    return labels, {"sizes": pd.Series(labels).value_counts().to_dict(),
                    "regime_legal_signal": means}


def doctrine_map(features):
    counts = (features.groupby(["circuit", "year", "regime"]).size()
              .unstack("regime", fill_value=0))
    shares = counts.div(counts.sum(axis=1), axis=0)
    shares["n_cases"] = counts.sum(axis=1)
    shares["share_strict"] = shares.get("strict", 0)
    columns = [c for c in counts.columns]
    probabilities = shares[columns].clip(lower=1e-12)
    shares["regime_entropy"] = -(probabilities * np.log(probabilities)).sum(axis=1)
    return shares.reset_index()


def regime_transitions(features):
    dominant = (features.groupby(["circuit", "year", "regime"]).size()
                .reset_index(name="n").sort_values("n")
                .drop_duplicates(["circuit", "year"], keep="last")
                .sort_values(["circuit", "year"]))
    dominant["prev_regime"] = dominant.groupby("circuit")["regime"].shift()
    dominant["transition"] = (dominant["prev_regime"].fillna("")
                               + " -> " + dominant["regime"])
    dominant["changed"] = (dominant["prev_regime"].notna()
                            & dominant["prev_regime"].ne(dominant["regime"])).astype(int)
    return dominant[["circuit", "year", "regime", "prev_regime", "transition", "changed"]]


def cross_circuit_divergence(mapping):
    return (mapping.groupby("year")
            .agg(share_strict_var=("share_strict", "var"),
                 share_strict_mean=("share_strict", "mean"),
                 mean_entropy=("regime_entropy", "mean"),
                 n_circuits=("circuit", "nunique"))
            .reset_index().round(4))


def run_step4(features, texts):
    embeddings = embed(texts)
    labels, info = cluster_regimes(embeddings, features["legal_signal_score"].to_numpy())
    output = features.copy()
    output["regime"] = labels
    mapping = doctrine_map(output)
    return {
        "features": output,
        "cluster_info": info,
        "doctrine_map": mapping,
        "transitions": regime_transitions(output),
        "divergence": cross_circuit_divergence(mapping),
    }
