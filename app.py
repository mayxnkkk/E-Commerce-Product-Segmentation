"""
Streamlit Web Application for E-Commerce Product Segmentation.

Provides an interactive user interface to:
1. Input product commercial attributes
2. Scale inputs with the pre-trained StandardScaler
3. Predict the product segment using the pre-trained KMeans model
4. Display comparative metrics between the input product and the segment average
5. Provide actionable commercial recommendations (Inventory, Pricing, Marketing, Quality)
6. Inspect dataset segment profiles and model performance metrics
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="E-Commerce Product Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .strategy-card {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .strategy-title {
        font-weight: 600;
        color: #15803D;
        margin-bottom: 0.3rem;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #EFF6FF;
        color: #1D4ED8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants & Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
KMEANS_PATH = os.path.join(MODELS_DIR, "kmeans_model.pkl")
PROFILES_PATH = os.path.join(MODELS_DIR, "cluster_profiles.csv")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

FEATURE_COLUMNS = [
    "price",
    "rating",
    "review_count",
    "return_rate",
    "sales_volume",
    "discount_frequency",
]


@st.cache_resource
def load_model_artifacts():
    """Load pre-trained artifacts once and cache them."""
    if not os.path.exists(SCALER_PATH) or not os.path.exists(KMEANS_PATH):
        st.error("Model artifacts not found. Please train models first using the pipeline.")
        return None, None, None

    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    profiles_df = pd.read_csv(PROFILES_PATH)
    return scaler, kmeans, profiles_df


scaler, kmeans, profiles_df = load_model_artifacts()

# App Header
st.markdown('<div class="main-header">E-Commerce Product Segmentation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header"><b>Machine Learning Based Product Segmentation Using K-Means</b></div>',
    unsafe_allow_html=True,
)

# ----------------- SIDEBAR -----------------
st.sidebar.header("Enter Product Details")
st.sidebar.markdown("Specify product attributes to determine its commercial cluster.")

# Quick Demo Preset Selectors
preset_option = st.sidebar.selectbox(
    "Quick-Load Sample Preset:",
    [
        "Custom Input",
        "Test 1: Premium Flagship",
        "Test 2: High-Volume Value Seller",
        "Test 3: Sluggish / Low Performance",
        "Test 4: High-Return / Clearance SKU",
    ],
)

# Default values based on preset
if preset_option == "Test 1: Premium Flagship":
    def_price = 12500.0
    def_rating = 4.65
    def_reviews = 1800
    def_returns = 4.2
    def_sales = 4800
    def_discount = 15.0
elif preset_option == "Test 2: High-Volume Value Seller":
    def_price = 850.0
    def_rating = 4.20
    def_reviews = 850
    def_returns = 6.8
    def_sales = 7200
    def_discount = 35.0
elif preset_option == "Test 3: Sluggish / Low Performance":
    def_price = 1450.0
    def_rating = 2.90
    def_reviews = 45
    def_returns = 13.5
    def_sales = 180
    def_discount = 20.0
elif preset_option == "Test 4: High-Return / Clearance SKU":
    def_price = 3200.0
    def_rating = 3.30
    def_reviews = 320
    def_returns = 22.5
    def_sales = 1100
    def_discount = 70.0
else:
    def_price = 1800.0
    def_rating = 4.10
    def_reviews = 450
    def_returns = 7.5
    def_sales = 3200
    def_discount = 25.0

# Sidebar Inputs (Part 32)
input_price = st.sidebar.number_input(
    "Price (₹ / INR)",
    min_value=1.0,
    max_value=100000.0,
    value=float(def_price),
    step=50.0,
    help="Product retail selling price.",
)

input_rating = st.sidebar.slider(
    "Average Rating",
    min_value=1.0,
    max_value=5.0,
    value=float(def_rating),
    step=0.05,
    help="Customer satisfaction rating on 1.0 to 5.0 scale.",
)

input_reviews = st.sidebar.number_input(
    "Review Count",
    min_value=0,
    max_value=100000,
    value=int(def_reviews),
    step=10,
    help="Total customer reviews submitted.",
)

input_returns = st.sidebar.slider(
    "Return Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=float(def_returns),
    step=0.1,
    help="Percentage of delivered units returned by customers.",
)

input_sales = st.sidebar.number_input(
    "Sales Volume (Units)",
    min_value=0,
    max_value=100000,
    value=int(def_sales),
    step=50,
    help="Total units sold over the observation timeframe.",
)

input_discount = st.sidebar.slider(
    "Discount Frequency (%)",
    min_value=0.0,
    max_value=100.0,
    value=float(def_discount),
    step=0.5,
    help="Percentage of active listings days with promotional markdowns.",
)

predict_clicked = st.sidebar.button("Predict Product Segment", type="primary", use_container_width=True)

# State management: Blue prediction box only appears upon clicking Predict
current_inputs = {
    "price": float(input_price),
    "rating": float(input_rating),
    "review_count": int(input_reviews),
    "return_rate": float(input_returns),
    "sales_volume": int(input_sales),
    "discount_frequency": float(input_discount),
}

if "prediction_state" not in st.session_state:
    st.session_state.prediction_state = "initial"
    st.session_state.last_inputs = None
    st.session_state.last_prediction = None

if predict_clicked:
    user_data = pd.DataFrame([current_inputs])
    user_scaled = scaler.transform(user_data[FEATURE_COLUMNS])
    pred_cluster = int(kmeans.predict(user_scaled)[0])
    cluster_info = profiles_df[profiles_df["cluster_id"] == pred_cluster].iloc[0]
    cluster_name = cluster_info["cluster_name"]

    st.session_state.prediction_state = "predicted"
    st.session_state.last_inputs = current_inputs
    st.session_state.last_prediction = {
        "pred_cluster": pred_cluster,
        "cluster_info": cluster_info,
        "cluster_name": cluster_name,
        "user_data": user_data,
        "inputs": current_inputs,
    }
    st.toast(f"Prediction computed: {cluster_name} (Cluster {pred_cluster})", icon="🎯")
    st.sidebar.success(f"✓ Classified: **{cluster_name}**")
elif st.session_state.prediction_state == "predicted" and st.session_state.last_inputs != current_inputs:
    st.session_state.prediction_state = "modified"

# ----------------- MAIN CONTENT TABS -----------------
tab_predict, tab_catalog, tab_methodology = st.tabs(
    ["🎯 Product Segment Predictor", "📊 Segment Intelligence & Overview", "🔬 Model Architecture & Validation"]
)

# ----------------- TAB 1: PREDICTION & RESULTS -----------------
with tab_predict:
    if st.session_state.prediction_state == "initial":
        st.markdown("### 🎯 Product Segment Predictor")
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.6); border: 2px dashed rgba(148, 163, 184, 0.3); border-radius: 12px; padding: 3rem 2rem; text-align: center; margin: 1.5rem 0;">
                <div style="font-size: 2.8rem; margin-bottom: 0.8rem;">🎯</div>
                <h3 style="color: #F8FAFC; margin-bottom: 0.5rem; font-size: 1.4rem;">Ready to Classify Product</h3>
                <p style="color: #94A3B8; font-size: 1.05rem; max-width: 580px; margin: 0 auto 1.5rem auto; line-height: 1.6;">
                    Select a <b>Quick-Load Sample Preset</b> or customize the 6 parameters in the sidebar, 
                    then click the red <b>"Predict Product Segment"</b> button on the bottom left to compute the cluster.
                </p>
                <div style="display: inline-block; background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.35); padding: 0.5rem 1.2rem; border-radius: 20px; color: #93C5FD; font-size: 0.9rem;">
                    ⚡ Real-time inference using pre-trained K-Means centroids & StandardScaler
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif st.session_state.prediction_state == "modified":
        st.markdown("### 🎯 Product Segment Predictor")
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.6); border: 2px dashed rgba(245, 158, 11, 0.4); border-radius: 12px; padding: 2.5rem 2rem; text-align: center; margin: 1.5rem 0;">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">🔄</div>
                <h3 style="color: #F8FAFC; margin-bottom: 0.5rem; font-size: 1.35rem;">Parameters Updated</h3>
                <p style="color: #CBD5E1; font-size: 1.05rem; max-width: 580px; margin: 0 auto 1.5rem auto; line-height: 1.6;">
                    Product attributes were modified in the sidebar. Click the red <b>"Predict Product Segment"</b> button on the bottom left to evaluate this updated SKU profile.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # prediction_state == "predicted"
        pred_cluster = st.session_state.last_prediction["pred_cluster"]
        cluster_info = st.session_state.last_prediction["cluster_info"]
        cluster_name = st.session_state.last_prediction["cluster_name"]
        eval_inputs = st.session_state.last_prediction["inputs"]

        # Display Result Banner (Part 34)
        st.markdown("### Predicted Product Segment")
        col_banner1, col_banner2 = st.columns([3, 1])

        with col_banner1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #1E3A8A, #3B82F6); padding: 1.2rem; border-radius: 10px; color: white;">
                    <span style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.85;">Product Classification</span>
                    <h2 style="margin: 0.2rem 0; font-size: 1.8rem; color: #FFFFFF;">{cluster_name}</h2>
                    <p style="margin: 0; opacity: 0.95; font-size: 0.95rem;">{cluster_info['description']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_banner2:
            st.metric(
                label="Assigned Cluster ID",
                value=f"Cluster {pred_cluster}",
                help="Optimal cluster index derived from K-Means centroids.",
            )
            st.metric(
                label="Segment Catalog Share",
                value=f"{cluster_info['percentage']}%",
                delta=f"{cluster_info['product_count']} SKUs",
            )

        st.markdown("---")

        # Segment Profile Comparison Table (Part 34)
        st.markdown("### Segment Profile Comparison")
        st.markdown("How this individual product compares against the segment benchmark average:")

        comparison_df = pd.DataFrame(
            {
                "Feature": [
                    "Price (₹)",
                    "Average Rating (1-5)",
                    "Review Count",
                    "Return Rate (%)",
                    "Sales Volume (Units)",
                    "Discount Frequency (%)",
                ],
                "Segment Average": [
                    f"₹{cluster_info['mean_price']:,.2f}",
                    f"{cluster_info['mean_rating']:.2f}",
                    f"{cluster_info['mean_review_count']:,.1f}",
                    f"{cluster_info['mean_return_rate']:.2f}%",
                    f"{cluster_info['mean_sales_volume']:,.1f}",
                    f"{cluster_info['mean_discount_frequency']:.2f}%",
                ],
                "User Product": [
                    f"₹{eval_inputs['price']:,.2f}",
                    f"{eval_inputs['rating']:.2f}",
                    f"{eval_inputs['review_count']:,}",
                    f"{eval_inputs['return_rate']:.2f}%",
                    f"{eval_inputs['sales_volume']:,}",
                    f"{eval_inputs['discount_frequency']:.2f}%",
                ],
                "Variance vs. Benchmark": [
                    f"{((eval_inputs['price'] - cluster_info['mean_price']) / cluster_info['mean_price'] * 100):+.1f}%",
                    f"{((eval_inputs['rating'] - cluster_info['mean_rating']) / cluster_info['mean_rating'] * 100):+.1f}%",
                    f"{((eval_inputs['review_count'] - cluster_info['mean_review_count']) / max(1, cluster_info['mean_review_count']) * 100):+.1f}%",
                    f"{((eval_inputs['return_rate'] - cluster_info['mean_return_rate']) / cluster_info['mean_return_rate'] * 100):+.1f}%",
                    f"{((eval_inputs['sales_volume'] - cluster_info['mean_sales_volume']) / max(1, cluster_info['mean_sales_volume']) * 100):+.1f}%",
                    f"{((eval_inputs['discount_frequency'] - cluster_info['mean_discount_frequency']) / max(1, cluster_info['mean_discount_frequency']) * 100):+.1f}%",
                ],
            }
        )

        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Recommended Strategy (Part 34)
        st.markdown("### Recommended Commercial Strategy")
        strat_col1, strat_col2 = st.columns(2)

        with strat_col1:
            st.markdown(
                f"""
                <div class="strategy-card">
                    <div class="strategy-title">📦 Inventory & Supply Chain</div>
                    <div>{cluster_info['inventory_strategy']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="strategy-card">
                    <div class="strategy-title">🏷️ Pricing & Promotions</div>
                    <div>{cluster_info['pricing_strategy']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with strat_col2:
            st.markdown(
                f"""
                <div class="strategy-card">
                    <div class="strategy-title">🚀 Marketing & Merchandising</div>
                    <div>{cluster_info['marketing_strategy']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="strategy-card">
                    <div class="strategy-title">🛡️ Quality & Returns Management</div>
                    <div>{cluster_info['quality_strategy']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ----------------- TAB 2: CATALOG SEGMENT INTELLIGENCE -----------------
with tab_catalog:
    st.markdown("### Catalog-Wide Segment Distribution & Benchmarks")
    st.markdown("Overview of the discovered clusters across all 5,000 catalog products:")

    # Summary metrics across clusters
    summary_cols = st.columns(4)
    for i, (_, row) in enumerate(profiles_df.iterrows()):
        with summary_cols[i]:
            st.metric(
                label=f"Cluster {row['cluster_id']}: {row['cluster_name']}",
                value=f"{row['percentage']}%",
                delta=f"{row['product_count']} SKUs",
            )
            st.caption(f"Avg Price: ₹{row['mean_price']:,.0f} | Avg Sales: {row['mean_sales_volume']:,.0f} units")

    st.markdown("#### Comprehensive Cluster Profile Matrix")
    display_profile_df = profiles_df[
        [
            "cluster_id",
            "cluster_name",
            "product_count",
            "percentage",
            "mean_price",
            "median_price",
            "mean_rating",
            "mean_review_count",
            "mean_sales_volume",
            "mean_return_rate",
            "mean_discount_frequency",
        ]
    ].rename(
        columns={
            "cluster_id": "Cluster",
            "cluster_name": "Segment Name",
            "product_count": "Count",
            "percentage": "Catalog %",
            "mean_price": "Mean Price (₹)",
            "median_price": "Median Price (₹)",
            "mean_rating": "Mean Rating",
            "mean_review_count": "Mean Reviews",
            "mean_sales_volume": "Mean Sales",
            "mean_return_rate": "Return %",
            "mean_discount_frequency": "Discount %",
        }
    )
    st.dataframe(display_profile_df, use_container_width=True, hide_index=True)

    # Figures display
    st.markdown("#### Segment Visualization (PCA & Size Distribution)")
    fig_col1, fig_col2 = st.columns(2)
    pca_fig_path = os.path.join(FIGURES_DIR, "pca_kmeans_clusters.png")
    size_fig_path = os.path.join(FIGURES_DIR, "cluster_size_distribution.png")

    if os.path.exists(pca_fig_path):
        with fig_col1:
            st.image(pca_fig_path, caption="2D PCA Space with Cluster Projections", use_container_width=True)
    if os.path.exists(size_fig_path):
        with fig_col2:
            st.image(size_fig_path, caption="Cluster Population Distribution", use_container_width=True)


# ----------------- TAB 3: MODEL METHODOLOGY & VALIDATION -----------------
with tab_methodology:
    st.markdown("### Machine Learning Model Architecture & Validation")

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    with col_meta1:
        st.metric(label="Clustering Algorithm", value="K-Means (k-means++)")
    with col_meta2:
        st.metric(label="Optimal Clusters (K)", value="4")
    with col_meta3:
        st.metric(label="WCSS / Inertia", value="8,298.79")
    with col_meta4:
        st.metric(label="Silhouette Score", value="0.4559")

    # Model Information Expander (Part 35)
    with st.expander("ℹ️ Model Architecture & Technical Specifications", expanded=True):
        st.markdown(
            """
            - **Algorithm:** K-Means with `k-means++` centroid initialization
            - **Number of Clusters:** $K = 4$ (selected via Elbow Method inflection supported by Silhouette peak)
            - **Features Utilized (6):** `price`, `rating`, `review_count`, `return_rate`, `sales_volume`, `discount_frequency`
            - **Scaling Method:** `StandardScaler` (Z-score standardization: $\\mu=0, \\sigma=1$) fit on model development set
            - **Outlier Treatment:** Interquartile Range (IQR) capping (Winsorization) on `price` and `sales_volume`
            - **Random State:** 42 (ensures deterministic reproducibility)
            
            > **Note on Deployment vs. Validation:**
            > *Hierarchical (Agglomerative) clustering was used for comparison and structural validation ($ARI = 0.9752$), while K-Means is used for production deployment because it can assign new unseen products to existing clusters in real time using its learned centroids.*
            """
        )

    # Outlier Impact Comparison
    st.markdown("#### Impact of Outlier Treatment (Before vs. After Capping)")
    outlier_fig_path = os.path.join(FIGURES_DIR, "outlier_before_after_comparison.png")
    outlier_impact_path = os.path.join(FIGURES_DIR, "outlier_impact_comparison.png")

    if os.path.exists(outlier_impact_path):
        st.image(outlier_impact_path, caption="Quantitative Impact of Outlier Capping on Clustering Quality", use_container_width=True)

    # Evaluation Curves
    st.markdown("#### Elbow & Silhouette Evaluation Curves")
    curve_col1, curve_col2 = st.columns(2)
    elbow_path = os.path.join(FIGURES_DIR, "elbow_curve.png")
    sil_path = os.path.join(FIGURES_DIR, "silhouette_scores.png")

    if os.path.exists(elbow_path):
        with curve_col1:
            st.image(elbow_path, caption="Elbow Method Curve (WCSS vs. K)", use_container_width=True)
    if os.path.exists(sil_path):
        with curve_col2:
            st.image(sil_path, caption="Silhouette Score Curve vs. K", use_container_width=True)

    # Dendrogram
    st.markdown("#### Hierarchical Clustering Dendrogram (Ward Linkage)")
    dendro_path = os.path.join(FIGURES_DIR, "dendrogram.png")
    if os.path.exists(dendro_path):
        st.image(dendro_path, caption="Hierarchical Agglomerative Dendrogram showing 4 dominant cut branches", use_container_width=True)


# ----------------- ABOUT SECTION (Part 37) -----------------
st.markdown("---")
st.markdown("### About This Project")
st.info(
    "This application uses unsupervised machine learning to group products based on their commercial characteristics. "
    "K-Means clustering identifies product segments, and the resulting segment profiles can support inventory and pricing decisions."
)
