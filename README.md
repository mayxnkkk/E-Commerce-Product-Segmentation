# E-Commerce Product Segmentation Using Machine Learning

> **A Complete, Presentation-Ready Machine Learning Project for Academic & Enterprise Evaluation**

---

## 1. Project Title
**E-Commerce Product Segmentation Using Clustering**  
*Unsupervised Machine Learning for Data-Driven Inventory Allocation and Pricing Strategy*

---

## 2. Problem Statement
Modern multi-category e-commerce retailers manage thousands of product listings (SKUs). Treating all products uniformly leads to severe operational inefficiencies: stockouts of high-velocity items, excess holding costs for stagnant inventory, margin erosion from untargeted discounting, and high reverse-logistics friction from defect-prone items. 

Retailers require an unsupervised analytical system that partitions the product catalog based on multidimensional commercial traits—including unit pricing, customer satisfaction, review volume, return rates, sales velocity, and promotional frequency—without relying on manual or subjective rules.

---

## 3. Objectives
1. **Analyze E-Commerce Product Data:** Discover commercial distributions and cross-variable correlations.
2. **Detect & Treat Outliers:** Use the Interquartile Range (IQR) technique and winsorization/capping to bound extreme variance while preserving genuine product records.
3. **Feature Standardization:** Scale features using `StandardScaler` to ensure Euclidean distances in clustering are unbiased.
4. **Determine Optimal Granularity:** Determine optimal $K$ via the Elbow Method (WCSS) supported by the Silhouette Coefficient.
5. **Train K-Means Model:** Partition products into mathematically dense, commercially interpretable clusters.
6. **Apply Hierarchical Clustering:** Validate structural groupings using Agglomerative Clustering with Ward linkage and dendrogram cuts.
7. **Dimensionality Reduction:** Apply Principal Component Analysis (PCA) to project the 6-dimensional feature space onto 2D for visual validation.
8. **Comparative Analysis:** Compare K-Means and Agglomerative Clustering using the Adjusted Rand Index (ARI).
9. **Quantify Outlier Impact:** Systematically contrast clustering metrics before and after IQR capping.
10. **Interactive Deployment:** Deploy the pre-trained K-Means model into an intuitive, responsive Streamlit application for real-time SKU classification.

---

## 4. Dataset Overview
- **Source:** Synthetic E-Commerce Product Dataset modeled after Amazon Marketplace commercial dynamics.
- **Sample Size:** 5,000 product records.
- **Random Seed:** `42` for complete mathematical reproducibility.
- **Storage Location:** `data/raw/ecommerce_products.csv`.

---

## 5. Synthetic Data Explanation
Publicly available e-commerce datasets (e.g., standard Kaggle Amazon scraped data) frequently omit proprietary merchant metrics such as actual unit sales volumes and reverse-logistics return rates. 

To bridge this gap realistically:
- Synthetic distributions were modeled using right-skewed lognormal and gamma curves.
- Four realistic commercial archetypes were simulated as underlying tendencies (Premium Performers, Value Sellers, Low-Performance Products, High-Return/Discount-Dependent Products).
- Realistic cross-feature interactions and Gaussian noise were introduced so that clusters overlap naturally and require true unsupervised learning discovery.
- Deliberate luxury price outliers (up to ₹99,500) and blockbuster viral sales volume outliers (up to 26,000 units) were injected to test outlier treatment robustness.

Full documentation is available in [`data/DATA_GENERATION.md`](data/DATA_GENERATION.md).

---

## 6. Feature Set
Clustering relies exclusively on **six numerical commercial features**:

| Feature Name | Type | Range in Catalog | Description |
|---|---|---|---|
| `price` | Float | ₹108.84 – ₹99,323.79 | Retail selling price in INR (₹) |
| `rating` | Float | 1.09 – 5.00 | Average customer star rating |
| `review_count` | Integer | 10 – 3,745 reviews | Total verified customer reviews |
| `return_rate` | Float | 0.40% – 29.56% | Percentage of shipped units returned |
| `sales_volume` | Integer | 1 – 25,943 units | Total units sold during timeframe |
| `discount_frequency`| Float | 0.0% – 100.0% | Percentage of active listing time discounted |

*Note: `product_id` is an administrative identifier and is strictly excluded from feature scaling and clustering.*

---

## 7. Technologies & Libraries
- **Language:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Machine Learning:** `scikit-learn` (`StandardScaler`, `KMeans`, `AgglomerativeClustering`, `PCA`, `silhouette_score`, `adjusted_rand_score`)
- **Hierarchical Clustering:** `scipy` (`scipy.cluster.hierarchy.dendrogram`, `linkage`)
- **Model Serialization:** `joblib`
- **Interactive UI:** `streamlit`

---

## 8. Project Architecture & Directory Structure
```text
ecommerce-product-segmentation/
│
├── data/
│   ├── raw/
│   │   └── ecommerce_products.csv
│   │
│   ├── processed/
│   │   ├── products_before_outlier_treatment.csv
│   │   └── products_after_outlier_treatment.csv
│   │
│   └── DATA_GENERATION.md
│
├── notebooks/
│   └── ecommerce_product_segmentation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── clustering.py
│   ├── evaluation.py
│   └── utils.py
│
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── cluster_profiles.csv
│   └── model_metadata.json
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── final_results.json
│
├── app.py
├── requirements.txt
├── README.md
├── REPORT.md
├── PRESENTATION.md
├── VIVA_QUESTIONS.md
├── PROJECT_EXPLANATION.md
└── DEMO_GUIDE.md
```

---

## 9. Preprocessing Pipeline
1. **Hygiene Verification:** Confirmed 0 missing entries, 0 duplicate rows, and valid numeric boundaries across all 5,000 records.
2. **Defensive Imputation:** Configured median imputation fallback for any production missing inputs.
3. **Partitioning:** Preserved raw and processed intermediate states to guarantee pipeline auditability.

---

## 10. Outlier Treatment (IQR Winsorization)
Clustering algorithms utilizing Euclidean distance calculations are exceptionally vulnerable to extreme values.
- **Detection Method:** Interquartile Range ($IQR = Q_3 - Q_1$).
  - Lower Bound: $\max(0, Q_1 - 1.5 \times IQR)$
  - Upper Bound: $Q_3 + 1.5 \times IQR$
- **Identified Outliers:**
  - `price`: 674 outliers (13.48% of catalog) due to heavy positive lognormal skew and luxury items. Upper bound capped at ₹8,593.50.
  - `sales_volume`: 55 extreme blockbuster outliers (1.10% of catalog). Upper bound capped at 13,792 units.
- **Treatment Policy:** **Capping (Winsorization)** rather than deletion. Extreme sales and prices represent genuine business phenomena; discarding them would remove critical business records. Capping preserves the observation count ($N=5,000$) while bounding distance distortion.

---

## 11. Feature Scaling (`StandardScaler`)
Because features possess vastly different scales (e.g., `price` up to ₹8,593 vs. `rating` from 1.0 to 5.0), unscaled Euclidean distance would be dominated by `price` and `sales_volume`.
$$z = \frac{x - \mu}{\sigma}$$
`StandardScaler` shifts each feature to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$). The fitted scaler is saved to `models/scaler.pkl` to transform new incoming inference data identically.

---

## 12. K-Means Clustering
- **Algorithm:** K-Means with `k-means++` initialization and `n_init=10`.
- **Evaluated Range:** $K \in [2, 10]$.
- **Evaluation Criteria:** Within-Cluster Sum of Squares (WCSS / Inertia) and Silhouette Coefficient.
- **Reproducibility:** Seed fixed at `random_state=42`.

---

## 13. Optimal $K$ Selection (Elbow Method)
- **Primary Evidence:** The Elbow Curve exhibits its sharpest inflection point at $K = 4$.
- **Supporting Evidence:** The Silhouette Score peaks at $K = 4$ ($0.4559$), dropping markedly for $K \ge 5$ ($0.3913$ at $K=5$, $0.3470$ at $K=6$).
- **Conclusion:** $K = 4$ was selected as the optimal cluster count.

---

## 14. Hierarchical (Agglomerative) Clustering
- **Linkage Criterion:** Ward's minimum variance linkage (`linkage="ward"`).
- **Distance Metric:** Euclidean distance on the standardized 6D space.
- **Purpose:** Provide an independent, non-centroid topological validation of catalog groupings.

---

## 15. Dendrogram Analysis
- Generated via `scipy.cluster.hierarchy.linkage` and `dendrogram`.
- Cutting the dendrogram horizontally at a Ward dissimilarity threshold between 25 and 35 clearly reveals **four major vertical tree stems**, providing decisive structural confirmation of $K = 4$.

---

## 16. Principal Component Analysis (PCA)
- PCA reduces the 6 standardized features into 2 orthogonal components for visualization.
- **Explained Variance:** PC1 explains 46.2% and PC2 explains 22.5% (Total 2D Explained Variance: **68.69%**).
- The resulting 2D projection demonstrates clear separation between the four clusters, validating cluster cohesion.

---

## 17. Commercial Cluster Profiles
Based on empirical centroids from `models/cluster_profiles.csv`:

| Cluster ID | Segment Name | Catalog Count | Catalog Share | Mean Price | Mean Rating | Mean Sales | Mean Return | Mean Discount |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | **High-Return / Discount-Dependent** | 994 | 19.88% | ₹2,832.93 | 3.47 | 1,201.5 | **20.02%** | **64.94%** |
| **1** | **Premium High Performers** | 1,007 | 20.14% | **₹7,880.65** | **4.46** | 4,747.0 | **5.43%** | **17.99%** |
| **2** | **Value High-Sellers** | 1,769 | 35.38% | ₹1,095.72 | 4.14 | **6,612.6** | 7.60% | 35.14% |
| **3** | **Low-Performance Products** | 1,230 | 24.60% | ₹1,839.28 | **3.06** | **223.1** | 12.39% | 24.09% |

---

## 18. Outlier Treatment Impact Analysis
Direct experiment comparing raw (uncapped) vs. capped data at $K = 4$:

| Metric | Before Outlier Capping | After Outlier Capping | Change / Impact |
|---|:---:|:---:|:---|
| **Outliers Treated** | 729 | 0 (all bounded) | Variance bounded |
| **WCSS (Inertia)** | 11,288.74 | 8,298.79 | **-26.49% (Tighter clusters)** |
| **Silhouette Score** | 0.4228 | 0.4559 | **+7.82% (Better separation)** |
| **Largest Cluster Size** | 2,748 (55.0%) | 1,769 (35.4%) | Balanced catalog distribution |
| **Smallest Cluster Size** | 46 (0.9%) | 994 (19.9%) | **Eliminated distorted micro-cluster** |

*Takeaway: Before capping, the 46 extreme luxury items formed an isolated micro-cluster, distorting the remaining 99.1% of the catalog. Capping restored balanced commercial segment discovery.*

---

## 19. Deployment Architecture
- **Framework:** Streamlit web application (`app.py`).
- **Inference Mechanism:** Accepts 6 real-time SKU inputs $\to$ validates constraints $\to$ scales via saved `models/scaler.pkl` $\to$ predicts cluster via saved `models/kmeans_model.pkl` $\to$ retrieves commercial interpretation and strategy.
- **Inference Latency:** $< 5$ milliseconds (no model retraining).

---

## 20. Experimental Results Summary
- **Optimal K:** 4
- **Final WCSS:** 8,298.79
- **Final Silhouette Score:** 0.4559
- **Agglomerative Silhouette Score:** 0.4532
- **Adjusted Rand Index (K-Means vs Hierarchical):** 0.9752 (97.5% assignment concordance)
- **Multi-Seed Stability (Seeds 0, 10, 20, 30, 40):** Standard deviation = 0.000 (deterministic convergence)

---

## 21. Limitations
1. **Unsupervised Nature:** Lacks external ground-truth labels; evaluation relies on mathematical heuristic validation.
2. **Feature Dependency:** Does not account for category hierarchy, gross margin percentages, seasonality, or supplier lead times.
3. **Synthetic Assumption:** Real marketplace datasets contain complex missingness, text feedback nuances, and platform-specific fraud dynamics.

---

## 22. Future Scope
- Integration of natural language sentiment analysis on customer review text.
- Inclusion of profit margin data to calculate customer lifetime value (LTV) per segment.
- Experimentation with density-based clustering (DBSCAN) and probabilistic clustering (Gaussian Mixture Models).

---

## 23. Installation & Setup
```bash
# 1. Clone repository or navigate to workspace
cd ecommerce-product-segmentation

# 2. Install required dependencies
pip install -r requirements.txt
```

---

## 24. Running Instructions

### Run the Data & Modeling Pipeline:
```bash
# 1. Generate synthetic dataset
python3 src/data_generation.py

# 2. Run preprocessing & outlier treatment
python3 src/preprocessing.py

# 3. Generate exploratory data analysis charts
python3 src/eda.py

# 4. Train K-Means, Hierarchical, PCA, and Profiling
python3 src/clustering.py

# 5. Run evaluation, stability, and outlier comparison
python3 src/evaluation.py
```

### Launch the Streamlit Web Application:
```bash
streamlit run app.py
```
*Access the application in your browser at `http://localhost:8501`.*

### Run the Jupyter Capstone Notebook:
```bash
jupyter notebook notebooks/ecommerce_product_segmentation.ipynb
```
