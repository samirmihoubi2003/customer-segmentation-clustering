from __future__ import annotations

from pathlib import Path

from src.data import Paths, load_wholesale
from src.clustering import run_kmeans_segmentation
from src.viz import plot_correlation_heatmap, plot_silhouette, plot_pca_clusters


def write_recommendations(profile, out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Business recommendations by segment\n\n")
    lines.append("Recommendations derived from the **average spending profile** of each cluster.\n\n")

    for cluster_id, row in profile.iterrows():
        top = row.sort_values(ascending=False).head(3)

        lines.append(f"## Segment {cluster_id}\n\n")
        lines.append("Top 3 categories:\n")
        for feat, val in top.items():
            lines.append(f"- **{feat}** (avg ≈ {val:.1f})\n")

        lines.append("\nSuggested actions:\n")
        lines.append("- Build targeted promotions around the dominant categories.\n")
        lines.append("- Cross-sell complementary categories (bundles).\n")
        lines.append("- Adapt retention strategy based on segment value.\n\n")

    out_md.write_text("".join(lines), encoding="utf-8")


def main():
    results = Path("results")
    plots = results / "plots"
    results.mkdir(parents=True, exist_ok=True)

    df = load_wholesale(Paths())

    # EDA
    plot_correlation_heatmap(df, plots / "heatmap_corr.png")

    # Clustering
    out = run_kmeans_segmentation(df)
    out.df_with_labels.to_csv(results / "wholesale_with_clusters.csv", index=False)
    out.profile.to_csv(results / "segments_profile.csv", index=True)

    # Plots
    plot_silhouette(out.silhouette_table, plots / "silhouette_vs_k.png")
    plot_pca_clusters(out.pca_2d, out.df_with_labels["cluster"].to_numpy(), plots / "pca_clusters.png")

    # Recommendations
    write_recommendations(out.profile, results / "recommendations.md")

    print(f"[done] Best K = {out.best_k}")
    print("[done] Check results/ folder")


if __name__ == "__main__":
    main()
