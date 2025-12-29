from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    numeric = df.select_dtypes(include=["number"]).drop(columns=["Channel", "Region"], errors="ignore")
    corr = numeric.corr()

    plt.figure()
    sns.heatmap(corr, annot=False)
    plt.title("Correlation heatmap (spending features)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_silhouette(tab: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    plt.figure()
    plt.plot(tab["k"], tab["silhouette"], marker="o")
    plt.title("Silhouette score vs K")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_pca_clusters(pca_2d: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    plt.figure()
    plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=labels)
    plt.title("PCA 2D projection of clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
