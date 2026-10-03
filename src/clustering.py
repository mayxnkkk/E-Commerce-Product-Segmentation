"""
Clustering Pipeline Module.

Implements:
1. Feature scaling with StandardScaler
2. K-Means evaluation (K=2 to 10) via Elbow method (WCSS) and Silhouette Score
3. Optimal K selection
4. Final K-Means training, cluster assignment, and cluster profiling
5. Automated commercial cluster interpretation and inventory strategy mapping
6. Dimensionality reduction and 2D visualization via PCA
7. Agglomerative (Hierarchical) clustering with Ward linkage and Dendrogram
8. Algorithmic comparative analysis (K-Means vs. Hierarchical) with Adjusted Rand Index
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    FEATURE_COLUMNS,
    setup_matplotlib_style,
    save_figure,
)


def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Standardize the 6 clustering features using StandardScaler.
    Fit and persist the scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLUMNS])

    os.makedirs(MODELS_DIR, exist_ok=True)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"StandardScaler fit on {len(df)} samples and saved to: {scaler_path}")
    return X_scaled, scaler


def evaluate_kmeans_k_range(
    X_scaled: np.ndarray, k_range: range = range(2, 11), random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluate K-Means for K in k_range, computing WCSS (inertia) and Silhouette Score.
    Generates Elbow curve and Silhouette score plots.
    """
    setup_matplotlib_style()
    wcss = []
    silhouette_scores = []
    k_values = list(k_range)

    print("Evaluating K-Means for K in range 2 to 10...")
    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        score = silhouette_score(X_scaled, km.labels_)
        silhouette_scores.append(score)
        print(f"  K = {k:2d} | WCSS = {km.inertia_:10.2f} | Silhouette Score = {score:.4f}")

    eval_df = pd.DataFrame(
        {
            "k": k_values,
            "wcss": wcss,
            "silhouette_score": silhouette_scores,
        }
    )

    # Persist evaluation table
    os.makedirs(TABLES_DIR, exist_ok=True)
    eval_table_path = os.path.join(TABLES_DIR, "kmeans_k_evaluation.csv")
    eval_df.to_csv(eval_table_path, index=False)
    print(f"Saved K-Means evaluation table: {eval_table_path}")

    # Plot 1: Elbow Curve (WCSS)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, wcss, marker="o", linewidth=2.2, markersize=8, color="#1f77b4")
    ax.set_title("Elbow Method for Optimal K (WCSS / Inertia)", pad=12)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
    ax.set_xticks(k_values)
    ax.grid(True, linestyle="--", alpha=0.6)
    save_figure(fig, "elbow_curve.png")

    # Plot 2: Silhouette Scores Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, silhouette_scores, marker="s", linewidth=2.2, markersize=8, color="#2ca02c")
    ax.set_title("Silhouette Score vs. Number of Clusters (K)", pad=12)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(k_values)
    ax.grid(True, linestyle="--", alpha=0.6)
    save_figure(fig, "silhouette_scores.png")

    return eval_df


def determine_optimal_k(eval_df: pd.DataFrame) -> int:
    """
    Select optimal K using Elbow inflection point supported by Silhouette analysis.
    Uses second derivative / curvature of WCSS curve.
    """
    wcss = eval_df["wcss"].values
    k_vals = eval_df["k"].values

    # Compute second difference (discrete acceleration/curvature)
    second_diff = np.diff(np.diff(wcss))
    # Optimal elbow corresponds to maximum curvature (second difference)
    elbow_idx = np.argmax(second_diff) + 1  # Offset by 1 due to double diff
    optimal_k_elbow = int(k_vals[elbow_idx])

    # Silhouette support check
    best_k_sil = int(eval_df.loc[eval_df["silhouette_score"].idxmax(), "k"])

    print(f"Optimal K Analysis:\n  - Elbow Curvature peak: K = {optimal_k_elbow}\n  - Max Silhouette score: K = {best_k_sil}")
    # Academic decision: If elbow suggests 4 (or 3/4) with strong silhouette, pick elbow
    optimal_k = optimal_k_elbow
    return optimal_k


def train_final_kmeans(
    X_scaled: np.ndarray, optimal_k: int, random_state: int = 42
) -> tuple[KMeans, np.ndarray]:
    """Train and persist final KMeans model."""
    kmeans = KMeans(
        n_clusters=optimal_k,
        init="k-means++",
        n_init=10,
        random_state=random_state,
    )
    cluster_labels = kmeans.fit_predict(X_scaled)

    model_path = os.path.join(MODELS_DIR, "kmeans_model.pkl")
    joblib.dump(kmeans, model_path)
    print(f"Trained final KMeans (K={optimal_k}) and saved to: {model_path}")
    return kmeans, cluster_labels


def analyze_clusters(
    df: pd.DataFrame, cluster_labels: np.ndarray, optimal_k: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cluster sizes, detailed profiles (mean and median), and commercial labels.
    """
    df_clustered = df.copy()
    df_clustered["cluster"] = cluster_labels

    # 1. Cluster Sizes
    size_counts = df_clustered["cluster"].value_counts().sort_index()
    size_pcts = (size_counts / len(df_clustered) * 100).round(2)
    cluster_sizes_df = pd.DataFrame(
        {
            "cluster_id": size_counts.index,
            "product_count": size_counts.values,
            "percentage": size_pcts.values,
        }
    )

    # Save cluster sizes table
    sizes_table_path = os.path.join(TABLES_DIR, "cluster_sizes.csv")
    cluster_sizes_df.to_csv(sizes_table_path, index=False)
    print(f"Saved cluster sizes table: {sizes_table_path}")

    # Plot Cluster Size Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = sns.color_palette("Set2", n_colors=optimal_k)
    bars = ax.bar(
        [f"Cluster {i}" for i in cluster_sizes_df["cluster_id"]],
        cluster_sizes_df["product_count"],
        color=palette,
        edgecolor="black",
        alpha=0.85,
    )
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height}\n({height/len(df_clustered)*100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    ax.set_title("Product Distribution Across Clusters", pad=12)
    ax.set_xlabel("Product Segment")
    ax.set_ylabel("Number of Products")
    save_figure(fig, "cluster_size_distribution.png")

    # 2. Cluster Profiles: Calculate Mean and Median for all features
    profile_rows = []
    for cid in range(optimal_k):
        c_sub = df_clustered[df_clustered["cluster"] == cid]
        row = {
            "cluster_id": cid,
            "product_count": len(c_sub),
            "percentage": round(len(c_sub) / len(df_clustered) * 100, 2),
            "mean_price": round(float(c_sub["price"].mean()), 2),
            "median_price": round(float(c_sub["price"].median()), 2),
            "mean_rating": round(float(c_sub["rating"].mean()), 2),
            "median_rating": round(float(c_sub["rating"].median()), 2),
            "mean_review_count": round(float(c_sub["review_count"].mean()), 1),
            "median_review_count": round(float(c_sub["review_count"].median()), 1),
            "mean_return_rate": round(float(c_sub["return_rate"].mean()), 2),
            "median_return_rate": round(float(c_sub["return_rate"].median()), 2),
            "mean_sales_volume": round(float(c_sub["sales_volume"].mean()), 1),
            "median_sales_volume": round(float(c_sub["sales_volume"].median()), 1),
            "mean_discount_frequency": round(float(c_sub["discount_frequency"].mean()), 2),
            "median_discount_frequency": round(float(c_sub["discount_frequency"].median()), 2),
        }
        profile_rows.append(row)

    profiles_df = pd.DataFrame(profile_rows)

    # 3. Dynamic Commercial Labeling and Strategy Generation
    # Evaluates the actual cluster centroids to assign commercially meaningful names
    names = []
    descriptions = []
    strategies = []
    inventory_recs = []
    pricing_recs = []
    marketing_recs = []
    quality_recs = []

    overall_mean_price = df_clustered["price"].mean()
    overall_mean_sales = df_clustered["sales_volume"].mean()
    overall_mean_return = df_clustered["return_rate"].mean()
    overall_mean_discount = df_clustered["discount_frequency"].mean()
    overall_mean_rating = df_clustered["rating"].mean()

    for _, p in profiles_df.iterrows():
        p_price = p["mean_price"]
        p_sales = p["mean_sales_volume"]
        p_return = p["mean_return_rate"]
        p_disc = p["mean_discount_frequency"]
        p_rate = p["mean_rating"]

        # Classification based on empirical metrics
        if p_price > 1.4 * overall_mean_price and p_rate >= 4.0:
            c_name = "Premium High Performers"
            desc = "High-ticket products with excellent customer satisfaction, substantial review depth, and strong revenue contribution."
            inv = "High stock priority with controlled safety stock; avoid stockouts of high-margin SKUs."
            prc = "Maintain premium price point; avoid dilution with heavy discounting."
            mkt = "Showcase prestige, premium build quality, verified buyer reviews, and brand heritage."
            qlt = "Maintain rigorous quality assurance and white-glove customer care."
        elif p_sales > overall_mean_sales and p_price <= 1.2 * overall_mean_price and p_return < overall_mean_return:
            c_name = "Value High-Sellers"
            desc = "High-velocity everyday essentials with strong customer ratings, moderate pricing, and high turnover."
            inv = "Maximum inventory allocation; automate re-order points to avoid stockouts."
            prc = "Competitive everyday pricing; use volume bundling and cross-sell discounts."
            mkt = "Promote best-seller badges, fast shipping, and top-of-funnel search visibility."
            qlt = "Maintain reliable supply chain packaging and fast fulfilment SLA."
        elif p_return > 1.2 * overall_mean_return or (p_disc > 1.2 * overall_mean_discount and p_sales < overall_mean_sales):
            c_name = "High-Return / Discount-Dependent Products"
            desc = "Products that require deep discounts to convert or suffer from uncharacteristically high customer return rates."
            inv = "Strictly controlled inventory; limit forward stock commitments until return causes are resolved."
            prc = "Review promotional dependency; avoid margin erosion from recurring 50%+ discounts."
            mkt = "Address catalog misrepresentation; update sizing guides, specifications, and realistic imagery."
            qlt = "Initiate mandatory return-root-cause inspection; audit vendor manufacturing or packaging defects."
        else:
            c_name = "Low-Performance Products"
            desc = "Sluggish sales velocity, below-average customer ratings, and low consumer engagement."
            inv = "Minimize stock; initiate aggressive clearance sales to free up working capital and warehouse space."
            prc = "Test tactical markdowns to liquidate dormant stock without cannibalizing core lines."
            mkt = "Reposition listings with improved keywords or phase out bottom-decile SKUs."
            qlt = "Investigate low customer ratings (reviews < 3.5); evaluate supplier quality or delist."

        names.append(c_name)
        descriptions.append(desc)
        inventory_recs.append(inv)
        pricing_recs.append(prc)
        marketing_recs.append(mkt)
        quality_recs.append(qlt)
        strategies.append(f"Inventory: {inv} | Pricing: {prc} | Marketing: {mkt} | Quality: {qlt}")

    profiles_df["cluster_name"] = names
    profiles_df["description"] = descriptions
    profiles_df["commercial_strategy"] = strategies
    profiles_df["inventory_strategy"] = inventory_recs
    profiles_df["pricing_strategy"] = pricing_recs
    profiles_df["marketing_strategy"] = marketing_recs
    profiles_df["quality_strategy"] = quality_recs

    # Save cluster profiles in models/ and results/tables/
    profiles_model_path = os.path.join(MODELS_DIR, "cluster_profiles.csv")
    profiles_table_path = os.path.join(TABLES_DIR, "cluster_profiles.csv")
    profiles_df.to_csv(profiles_model_path, index=False)
    profiles_df.to_csv(profiles_table_path, index=False)
    print(f"Saved cluster profiles to:\n  - {profiles_model_path}\n  - {profiles_table_path}")

    # Save cluster assignments
    assignments_path = os.path.join(TABLES_DIR, "kmeans_cluster_assignments.csv")
    df_clustered[["product_id", "cluster"]].to_csv(assignments_path, index=False)
    print(f"Saved cluster assignments: {assignments_path}")

    return df_clustered, profiles_df


def visualize_pca(
    X_scaled: np.ndarray, cluster_labels: np.ndarray, optimal_k: int, profiles_df: pd.DataFrame
):
    """
    Project 6D feature space onto 2 principal components for visualization.
    """
    setup_matplotlib_style()
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 6.5))
    palette = sns.color_palette("Set2", n_colors=optimal_k)

    for cid in range(optimal_k):
        mask = cluster_labels == cid
        c_name = profiles_df.loc[profiles_df["cluster_id"] == cid, "cluster_name"].values[0]
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=f"Cluster {cid}: {c_name}",
            color=palette[cid],
            alpha=0.6,
            edgecolor="none",
            s=35,
        )

    ax.set_title(
        f"2D PCA Visualization of K-Means Clusters\n(PC1: {var_exp[0]*100:.1f}%, PC2: {var_exp[1]*100:.1f}% Variance Explained)",
        pad=12,
    )
    ax.set_xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% Variance)")
    ax.legend(title="Product Segments", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    save_figure(fig, "pca_kmeans_clusters.png")
    print(f"PCA Total Explained Variance (2 components): {sum(var_exp)*100:.2f}%")
    return X_pca, pca


def run_hierarchical_clustering(
    X_scaled: np.ndarray, optimal_k: int, df: pd.DataFrame, sample_size_dendrogram: int = 200
) -> tuple[np.ndarray, float]:
    """
    Perform Agglomerative Clustering with Ward linkage.
    Generate publication-ready dendrogram using scipy.
    """
    setup_matplotlib_style()
    print("Running Hierarchical (Agglomerative) Clustering with Ward Linkage...")

    # Hierarchical model
    agg = AgglomerativeClustering(n_clusters=optimal_k, metric="euclidean", linkage="ward")
    hier_labels = agg.fit_predict(X_scaled)
    hier_sil = float(silhouette_score(X_scaled, hier_labels))
    print(f"Agglomerative Clustering Silhouette Score: {hier_sil:.4f}")

    # Generate Dendrogram on a representative stratified subsample for visual clarity
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_scaled), size=min(sample_size_dendrogram, len(X_scaled)), replace=False)
    X_sub = X_scaled[sample_indices]

    Z = linkage(X_sub, method="ward")

    fig, ax = plt.subplots(figsize=(11, 6))
    dendrogram(
        Z,
        ax=ax,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90.0,
        leaf_font_size=10.0,
        show_contracted=True,
    )
    ax.set_title(f"Hierarchical Clustering Dendrogram (Ward's Linkage, p=30 cuts)", pad=12)
    ax.set_xlabel("Sample Clusters / Leaf Index")
    ax.set_ylabel("Ward Distance / Dissimilarity")
    ax.grid(True, linestyle="--", alpha=0.5)

    save_figure(fig, "dendrogram.png")
    return hier_labels, hier_sil


def compare_algorithms(
    kmeans_labels: np.ndarray,
    hier_labels: np.ndarray,
    kmeans_sil: float,
    hier_sil: float,
    optimal_k: int,
) -> pd.DataFrame:
    """
    Compare K-Means and Agglomerative Clustering across cluster sizes,
    Silhouette Score, and Adjusted Rand Index (ARI).
    """
    ari = float(adjusted_rand_score(kmeans_labels, hier_labels))
    print(f"Adjusted Rand Index (ARI) between K-Means and Agglomerative: {ari:.4f}")

    comparison_data = {
        "Metric": [
            "Algorithm Type",
            "Optimal Clusters (K)",
            "Silhouette Score",
            "Cluster 0 Size",
            "Cluster 1 Size",
            "Cluster 2 Size",
            "Cluster 3 Size",
            "Adjusted Rand Index (ARI)",
        ],
        "K-Means": [
            "Centroid-based (Partitional)",
            optimal_k,
            round(kmeans_sil, 4),
            int((kmeans_labels == 0).sum()),
            int((kmeans_labels == 1).sum()),
            int((kmeans_labels == 2).sum()),
            int((kmeans_labels == 3).sum()) if optimal_k >= 4 else "N/A",
            round(ari, 4),
        ],
        "Agglomerative": [
            "Connectivity-based (Hierarchical)",
            optimal_k,
            round(hier_sil, 4),
            int((hier_labels == 0).sum()),
            int((hier_labels == 1).sum()),
            int((hier_labels == 2).sum()),
            int((hier_labels == 3).sum()) if optimal_k >= 4 else "N/A",
            round(ari, 4),
        ],
    }

    comp_df = pd.DataFrame(comparison_data)
    table_path = os.path.join(TABLES_DIR, "algorithm_comparison.csv")
    comp_df.to_csv(table_path, index=False)
    print(f"Saved algorithm comparison table: {table_path}")
    return comp_df


def run_full_clustering_pipeline(after_csv: str = None):
    """Execute complete end-to-end clustering pipeline."""
    if after_csv is None:
        after_csv = os.path.join(DATA_PROCESSED_DIR, "products_after_outlier_treatment.csv")

    df = pd.read_csv(after_csv)
    print(f"Loaded dataset for clustering: {after_csv} (Shape: {df.shape})")

    # 1. Feature scaling
    X_scaled, scaler = scale_features(df)

    # 2. Evaluate K range (2 to 10)
    eval_df = evaluate_kmeans_k_range(X_scaled)

    # 3. Determine optimal K
    optimal_k = determine_optimal_k(eval_df)

    # 4. Train final K-Means
    kmeans_model, kmeans_labels = train_final_kmeans(X_scaled, optimal_k)
    kmeans_sil = float(silhouette_score(X_scaled, kmeans_labels))
    kmeans_wcss = float(kmeans_model.inertia_)

    # 5. Cluster profiling and size analysis
    df_clustered, profiles_df = analyze_clusters(df, kmeans_labels, optimal_k)

    # 6. PCA 2D visualization
    visualize_pca(X_scaled, kmeans_labels, optimal_k, profiles_df)

    # 7. Hierarchical clustering and dendrogram
    hier_labels, hier_sil = run_hierarchical_clustering(X_scaled, optimal_k, df)

    # 8. Compare K-Means vs Agglomerative
    comp_df = compare_algorithms(kmeans_labels, hier_labels, kmeans_sil, hier_sil, optimal_k)

    return {
        "optimal_k": optimal_k,
        "kmeans_model": kmeans_model,
        "scaler": scaler,
        "kmeans_labels": kmeans_labels,
        "hier_labels": hier_labels,
        "kmeans_sil": kmeans_sil,
        "kmeans_wcss": kmeans_wcss,
        "hier_sil": hier_sil,
        "profiles_df": profiles_df,
        "comp_df": comp_df,
    }


if __name__ == "__main__":
    run_full_clustering_pipeline()
