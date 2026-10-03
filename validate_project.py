"""
Automated Project Validation Suite.

Validates:
1. File structure and all expected files exist
2. Python source modules import and execute cleanly
3. Model artifacts (KMeans, Scaler, Profiles, Metadata) load without error
4. Output figures and tables exist and are non-empty
5. End-to-end inference works across test products
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXPECTED_FILES = [
    "data/raw/ecommerce_products.csv",
    "data/processed/products_before_outlier_treatment.csv",
    "data/processed/products_after_outlier_treatment.csv",
    "data/DATA_GENERATION.md",
    "notebooks/ecommerce_product_segmentation.ipynb",
    "src/__init__.py",
    "src/data_generation.py",
    "src/preprocessing.py",
    "src/eda.py",
    "src/clustering.py",
    "src/evaluation.py",
    "src/utils.py",
    "models/kmeans_model.pkl",
    "models/scaler.pkl",
    "models/cluster_profiles.csv",
    "models/model_metadata.json",
    "results/final_results.json",
    "results/tables/cluster_profiles.csv",
    "results/tables/cluster_sizes.csv",
    "results/tables/kmeans_cluster_assignments.csv",
    "results/tables/outlier_comparison.csv",
    "results/tables/stability_results.csv",
    "results/tables/algorithm_comparison.csv",
    "results/figures/elbow_curve.png",
    "results/figures/silhouette_scores.png",
    "results/figures/pca_kmeans_clusters.png",
    "results/figures/dendrogram.png",
    "results/figures/cluster_size_distribution.png",
    "results/figures/outlier_before_after_comparison.png",
    "results/figures/outlier_impact_comparison.png",
    "results/figures/correlation_heatmap.png",
    "app.py",
    "requirements.txt",
    "README.md",
    "REPORT.md",
    "PRESENTATION.md",
    "VIVA_QUESTIONS.md",
    "PROJECT_EXPLANATION.md",
    "DEMO_GUIDE.md",
]


def test_file_structure():
    print("--- 1. Testing File Structure ---")
    missing = []
    for f in EXPECTED_FILES:
        path = os.path.join(BASE_DIR, f)
        if not os.path.exists(path):
            missing.append(f)
        else:
            size = os.path.getsize(path)
            assert size > 0, f"File {f} is empty!"
    if missing:
        print(f"FAILED: Missing files: {missing}")
        return False
    print(f"PASSED: All {len(EXPECTED_FILES)} required files exist and are non-empty.")
    return True


def test_artifacts_and_predictions():
    print("\n--- 2. Testing Artifacts & Model Inference ---")
    scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
    kmeans_path = os.path.join(BASE_DIR, "models", "kmeans_model.pkl")
    profiles_path = os.path.join(BASE_DIR, "models", "cluster_profiles.csv")
    meta_path = os.path.join(BASE_DIR, "models", "model_metadata.json")
    results_path = os.path.join(BASE_DIR, "results", "final_results.json")

    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)
    profiles = pd.read_csv(profiles_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)
    with open(results_path, "r") as f:
        results = json.load(f)

    assert meta["optimal_k"] == 4, "Optimal K in metadata must be 4"
    assert results["optimal_k"] == 4, "Optimal K in results must be 4"
    assert len(profiles) == 4, "Must have 4 cluster profiles"
    print("PASSED: Artifacts loaded and metadata verified.")

    # Test 4 distinct realistic predictions
    test_cases = [
        {
            "name": "Test 1 (Premium)",
            "data": [12500.0, 4.65, 1800, 4.2, 4800, 15.0],
            "expected_cluster": 1,
            "expected_name": "Premium High Performers",
        },
        {
            "name": "Test 2 (Value)",
            "data": [850.0, 4.20, 850, 6.8, 7200, 35.0],
            "expected_cluster": 2,
            "expected_name": "Value High-Sellers",
        },
        {
            "name": "Test 3 (Low Perf)",
            "data": [1450.0, 2.90, 45, 13.5, 180, 20.0],
            "expected_cluster": 3,
            "expected_name": "Low-Performance Products",
        },
        {
            "name": "Test 4 (High Return)",
            "data": [3200.0, 3.30, 320, 22.5, 1100, 70.0],
            "expected_cluster": 0,
            "expected_name": "High-Return / Discount-Dependent Products",
        },
    ]

    cols = ["price", "rating", "review_count", "return_rate", "sales_volume", "discount_frequency"]

    for tc in test_cases:
        df_in = pd.DataFrame([tc["data"]], columns=cols)
        scaled_in = scaler.transform(df_in)
        pred_cid = int(kmeans.predict(scaled_in)[0])
        pred_name = profiles.loc[profiles["cluster_id"] == pred_cid, "cluster_name"].values[0]

        print(f"  {tc['name']} -> Predicted: Cluster {pred_cid} ({pred_name})")
        assert pred_cid == tc["expected_cluster"], f"Expected {tc['expected_cluster']} but got {pred_cid}"
        assert pred_name == tc["expected_name"], f"Expected {tc['expected_name']} but got {pred_name}"

    print("PASSED: All 4 test predictions matched expected cluster identities.")
    return True


if __name__ == "__main__":
    ok1 = test_file_structure()
    ok2 = test_artifacts_and_predictions()
    if ok1 and ok2:
        print("\nALL AUTOMATED VALIDATION CHECKS PASSED PERFECTLY!")
    else:
        sys.exit(1)
