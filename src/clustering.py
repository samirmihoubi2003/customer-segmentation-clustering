from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


@dataclass
class ClusterOutputs:
    df_with_labels: pd.DataFrame
    profile: pd.DataFrame
    best_k: int
    silhouette_table: pd.DataFrame
    pca_2d: np.ndarray


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep numeric spending features for clustering.
    Keep Channel & Region for interpretation.
    """
    meta = df[["Channel", "Region"]].copy()
    X = df.drop(columns=["Channel", "Region"]).copy()

    # Spending variables are skewed -> log1p stabilizes variance
    X = np.log1p(X)

    return X, meta


def choose_k_by_silhouette(
    X_scaled: np.ndarray, k_min: int = 2, k_max: int = 10, seed: int = 42
):
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed)
        labels = km.fit_predict(X_scaled)
        s = silhouette_score(X_scaled, labels)
        rows.append({"k": k, "silhouette": float(s)})

    tab = pd.DataFrame(rows).sort_values("k")
    best_k = int(tab.loc[tab["silhouette"].idxmax(), "k"])
    return best_k, tab


def run_kmeans_segmentation(df: pd.DataFrame, seed: int = 42) -> ClusterOutputs:
    X, _meta = prepare_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, sil_tab = choose_k_by_silhouette(X_scaled)

    km = KMeans(n_clusters=best_k, n_init=50, random_state=seed)
    labels = km.fit_predict(X_scaled)

    out = df.copy()
    out["cluster"] = labels

    # Segment profile: mean spending by cluster (on original scale)
    profile = out.groupby("cluster")[X.columns.tolist()].mean().sort_index()

    # PCA for 2D visualization (on scaled features)
    pca = PCA(n_components=2, random_state=seed)
    pca_2d = pca.fit_transform(X_scaled)

    return ClusterOutputs(
        df_with_labels=out,
        profile=profile,
        best_k=best_k,
        silhouette_table=sil_tab,
        pca_2d=pca_2d,
    )
