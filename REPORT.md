# E-Commerce Product Segmentation Using Clustering
## Comprehensive Machine Learning Capstone Technical Report

---

## 1. Introduction
In competitive digital retail environments, catalog assortment management is central to operational profitability and customer retention. Enterprise merchants list tens of thousands of stock-keeping units (SKUs) spanning diverse price points, velocity brackets, and quality tiers. Traditional merchant operations often rely on coarse administrative rules (e.g., broad merchandise categories such as "Electronics" or "Home Goods") or subjective human intuition. However, category labels alone fail to differentiate high-velocity cash generators from margin-destroying, high-return SKUs within the same merchandise vertical.

Unsupervised machine learning provides an objective, data-driven approach to discover natural commercial product segments. By analyzing behavioral patterns across sales velocity, customer satisfaction, pricing sensitivity, and returns, clustering algorithms can isolate cohesive cohorts of products. These algorithmic segments enable retailers to dynamically calibrate replenishment schedules, promotional budgets, and supply chain safeguards.

---

## 2. Problem Statement
An online retailer managing a multi-category product catalog requires an automated Machine Learning pipeline to group products into actionable segments based on six fundamental commercial features:
1. **Unit Price (`price`)**
2. **Customer Satisfaction (`rating`)**
3. **Review Count / Social Proof (`review_count`)**
4. **Return Rate (`return_rate`)**
5. **Demand Velocity (`sales_volume`)**
6. **Promotional Exposure (`discount_frequency`)**

The resulting segmentation must directly inform executive decision-making across **inventory allocation**, **pricing strategies**, **promotional dependency management**, and **quality assurance**.

---

## 3. Objectives
The technical objectives of this project are:
- Ingest and preprocess 5,000 product observations with zero unhandled missingness or duplicate distortion.
- Implement robust Interquartile Range (IQR) outlier detection on skewed variables (`price` and `sales_volume`) and apply winsorization/capping.
- Standardize the 6-dimensional feature space using `StandardScaler` to equalize Euclidean distance contributions.
- Identify the optimal number of clusters ($K$) using the Elbow Method (WCSS) and Silhouette analysis across $K \in [2, 10]$.
- Train and evaluate a production-ready $K$-Means clustering model.
- Conduct structural validation using Hierarchical Agglomerative Clustering (Ward's linkage) and inspect hierarchical dendrogram partitions.
- Project high-dimensional cluster geometries into 2D space using Principal Component Analysis (PCA).
- Empirically quantify the impact of outlier capping by comparing clustering quality on raw versus capped data.
- Assess algorithm convergence and stability across multiple random initializations.
- Deploy the final K-Means pipeline into an interactive Streamlit web application capable of classifying new product inputs in real time.

---

## 4. Dataset
- **Total Records:** 5,000 product observations.
- **Total Columns:** 7 (1 SKU identifier + 6 numerical clustering attributes).
- **Format:** Comma-Separated Values (`data/raw/ecommerce_products.csv`).
- **Reproducibility Seed:** `42` (`numpy.random.seed(42)`).

---

## 5. Data Generation / Collection
In real-world e-commerce, proprietary merchant metrics—specifically **shipped sales volume** and **reverse-logistics return rate**—are highly guarded trade secrets rarely published on open repositories like Kaggle. To address this while maintaining strict empirical rigor, a commercially realistic synthetic dataset was constructed.

The generation logic models four underlying commercial archetypes, with realistic Gaussian noise and inter-variable interactions (e.g., positive rating lift on sales, return friction penalties, and discount-driven velocity boosts). Crucially, underlying profiles were used solely to create naturalistic distributions; no synthetic class labels were retained, ensuring that the machine learning models discovered clusters entirely through unsupervised analysis.

Additionally, 45 luxury price outliers (up to ₹99,323.79) and 55 blockbuster sales volume outliers (up to 25,943 units) were injected to test outlier detection and capping algorithms.

---

## 6. Data Preprocessing
Data quality auditing yielded the following metrics:
- **Missing Values:** 0 across all features.
- **Duplicate Rows:** 0 exact duplicate observations.
- **Duplicate Identifiers:** 5,000 unique sequential `product_id` strings (`PROD_00001` to `PROD_05000`).
- **Range Validity:** All ratings fell strictly within $[1.0, 5.0]$, return rates within $[0.0\%, 30.0\%]$, and discount frequencies within $[0.0\%, 100.0\%]$.
- **Defensive Measures:** A median-imputation fallback was embedded in `src/preprocessing.py` to ensure resilience against missing values in future streaming pipelines.

---

## 7. Exploratory Data Analysis (EDA)
Exploratory analysis revealed characteristic e-commerce distributional behaviors:
- **Price:** Exhibited extreme right-skewness (Mean: ₹4,288.90, Median: ₹1,722.58, Min: ₹108.84, Max: ₹99,323.79). Over 75% of products list below ₹4,015.
- **Rating:** Followed a bell-shaped distribution centered around moderate-to-good satisfaction (Mean: 3.80, Median: 3.95, Standard Deviation: 0.67).
- **Review Count:** Heavily right-skewed (Mean: 531.4, Median: 279.0, Max: 3,745 reviews).
- **Return Rate:** Averaged 10.37% (Median: 7.91%, Range: 0.40% – 29.56%).
- **Sales Volume:** Exhibited strong right-skewness (Mean: 3,666.0 units, Median: 3,686.0 units, Max: 25,943 units).
- **Discount Frequency:** Centered at 34.89% (Median: 30.95%, Range: 0.0% – 100.0%).

Correlation analysis highlighted moderate positive synergy between `rating` and `sales_volume` ($r = +0.31$), and a notable positive relationship between `discount_frequency` and `sales_volume` ($r = +0.28$), confirming that customer ratings and pricing promotions actively influence sales velocity in the catalog.

---

## 8. Outlier Detection and Treatment
Outlier analysis was focused on `price` and `sales_volume`. Using the Interquartile Range method ($IQR = Q_3 - Q_1$):
- **Price:**
  - $Q_1 = ₹962.06$, $Q_3 = ₹4,014.64$, $IQR = ₹3,052.58$
  - Lower Bound: $\max(0, ₹962.06 - 1.5 \times ₹3,052.58) = ₹0.00$
  - Upper Bound: $₹4,014.64 + 1.5 \times ₹3,052.58 = ₹8,593.50$
  - **Detected Outliers:** 674 products (13.48% of catalog).
- **Sales Volume:**
  - $Q_1 = 546.0$ units, $Q_3 = 5,844.5$ units, $IQR = 5,298.5$ units
  - Lower Bound: $\max(0, 546.0 - 1.5 \times 5,298.5) = 0.0$ units
  - Upper Bound: $5,844.5 + 1.5 \times 5,298.5 = 13,792.25$ units
  - **Detected Outliers:** 55 products (1.10% of catalog).

### Treatment Strategy: IQR Capping (Winsorization)
Rather than deleting 729 rows (which would discard 14.58% of the commercial dataset, including genuine blockbuster and luxury items), values exceeding the upper bounds were capped:
$$\text{price}_{\text{capped}} = \min(\text{price}, 8593.50), \quad \text{sales}_{\text{capped}} = \min(\text{sales}, 13792.25)$$
This bounded variance for distance calculations while preserving sample size ($N=5,000$).

---

## 9. Feature Scaling
Clustering algorithms calculate geometric proximity using Euclidean distance:
$$d(x, y) = \sqrt{\sum_{i=1}^{p} (x_i - y_i)^2}$$
If features with large magnitudes (e.g., `price` up to ₹8,593.50 or `sales_volume` up to 13,792 units) remained unscaled, they would completely dominate features with smaller scales (e.g., `rating` between 1.0 and 5.0).

`StandardScaler` was fit on the capped dataset, standardizing each variable to mean $\mu = 0$ and standard deviation $\sigma = 1$:
$$z = \frac{x - \mu}{\sigma}$$
The fitted scaler was serialized to `models/scaler.pkl`.

---

## 10. K-Means Clustering
K-Means with `k-means++` initialization was executed over candidate values $K \in [2, 10]$ with $n\_init = 10$ and `random_state = 42`. The resulting metrics are summarized below:

| $K$ | Within-Cluster Sum of Squares (WCSS) | Silhouette Coefficient |
|:---:|:---:|:---:|
| 2 | 17,718.60 | 0.3841 |
| 3 | 11,984.53 | 0.4191 |
| **4** | **8,298.79** | **0.4559** |
| 5 | 7,726.35 | 0.3913 |
| 6 | 7,171.12 | 0.3470 |
| 7 | 6,697.14 | 0.2621 |
| 8 | 6,390.22 | 0.2280 |
| 9 | 6,038.26 | 0.2090 |
| 10 | 5,796.64 | 0.2103 |

---

## 11. Optimal Number of Clusters
Selecting optimal $K$ combined both the Elbow Method and Silhouette analysis:
1. **Elbow Method:** The WCSS reduction decreases substantially after $K = 4$. The transition from $K=2 \to 3$ yields $\Delta WCSS = 5,734.07$; $K=3 \to 4$ yields $\Delta WCSS = 3,685.74$; whereas $K=4 \to 5$ drops by only $572.44$. The point of maximum curvature (second derivative) occurs distinctly at **$K = 4$**.
2. **Silhouette Coefficient:** The silhouette curve reaches its global maximum at **$K = 4$** with a score of **$0.4559$**. For $K \ge 5$, cluster cohesion deteriorates into overlapping sub-partitions.

Thus, **$K = 4$** was selected as the optimal cluster count.

---

## 12. Hierarchical Clustering
Agglomerative Hierarchical Clustering was performed on the standardized 6D dataset using Ward's minimum variance criterion (`linkage='ward'`). Ward's linkage minimizes the total within-cluster variance at each merge step:
$$\Delta ESS_{AB} = \frac{n_A n_B}{n_A + n_B} \|\mu_A - \mu_B\|^2$$
At $K = 4$, the Agglomerative model achieved a Silhouette Coefficient of **$0.4532$**, providing strong independent validation of the K-Means partition ($0.4559$).

---

## 13. PCA Visualization
Principal Component Analysis (PCA) was fitted on the 6-dimensional standardized space:
- **PC1 Variance Explained:** 46.20%
- **PC2 Variance Explained:** 22.49%
- **Cumulative 2D Explained Variance:** **68.69%**

The resulting 2D scatter plot demonstrates clear separation among the four clusters:
- **PC1 (Horizontal Axis):** Captures general product vitality, separating high-velocity/high-rating items from low-performing/high-return SKUs.
- **PC2 (Vertical Axis):** Captures the price-volume trade-off, separating premium high-ticket items from mass-market value sellers.

---

## 14. Comparative Study (K-Means vs. Hierarchical)
The two clustering paradigms were compared across several dimensions:

| Evaluation Metric | K-Means Clustering | Agglomerative Clustering |
|---|:---:|:---:|
| **Algorithmic Paradigm** | Partitional / Centroid-based | Agglomerative / Connectivity-based |
| **Optimization Target** | Intra-cluster squared Euclidean distance | Ward's minimum variance criterion |
| **Silhouette Score** | **0.4559** | **0.4532** |
| **Cluster 0 Size** | 994 (19.88%) | 988 (19.76%) |
| **Cluster 1 Size** | 1,007 (20.14%) | 1,012 (20.24%) |
| **Cluster 2 Size** | 1,769 (35.38%) | 1,765 (35.30%) |
| **Cluster 3 Size** | 1,230 (24.60%) | 1,235 (24.70%) |
| **Adjusted Rand Index (ARI)** | \multicolumn{2}{c|}{**0.9752**} |
| **Online Inference Capability** | **Yes** (Instant centroid mapping) | **No** (Requires full re-clustering) |

An **Adjusted Rand Index of 0.9752** demonstrates that 97.5% of pairwise product assignments are identical between the two algorithms. This confirms that the discovered segments reflect genuine data density rather than algorithmic bias.

---

## 15. Cluster Profiles
Detailed centroid metrics computed on the capped catalog:

| Cluster | Segment Name | Products | Share | Mean Price | Median Price | Mean Rating | Mean Reviews | Mean Sales | Mean Return | Mean Discount |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | **High-Return / Discount-Dependent** | 994 | 19.88% | ₹2,832.93 | ₹2,498.24 | 3.47 | 258.9 | 1,201.5 | **20.02%** | **64.94%** |
| **1** | **Premium High Performers** | 1,007 | 20.14% | **₹7,880.65** | **₹8,593.50** | **4.46** | **1,114.7** | 4,747.0 | **5.43%** | **17.99%** |
| **2** | **Value High-Sellers** | 1,769 | 35.38% | ₹1,095.72 | ₹914.96 | 4.14 | 557.4 | **6,612.6** | 7.60% | 35.14% |
| **3** | **Low-Performance Products** | 1,230 | 24.60% | ₹1,839.28 | ₹1,519.65 | **3.06** | **68.2** | **223.1** | 12.39% | 24.09% |

---

## 16. Outlier Treatment Impact
An explicit experimental comparison was conducted by running K-Means ($K=4$) on both datasets:

| Metric | Dataset A (Raw / Uncapped) | Dataset B (IQR Capped) | Empirical Variance |
|---|:---:|:---:|:---|
| **Total Outliers Treated** | 729 | 0 | All extreme values bounded |
| **Optimal K** | 4 | 4 | Invariant structure |
| **WCSS (Inertia)** | 11,288.74 | 8,298.79 | **-26.49% variance reduction** |
| **Silhouette Score** | 0.4228 | 0.4559 | **+7.82% cluster tightness gain** |
| **Largest Cluster Size** | 2,748 (55.0%) | 1,769 (35.4%) | Restored catalog balance |
| **Smallest Cluster Size** | 46 (0.9%) | 994 (19.9%) | **Eliminated non-actionable micro-cluster** |

### Analytical Conclusion on Outlier Treatment:
Before outlier treatment, the 46 extreme luxury items formed an isolated micro-cluster, collapsing the remaining 99.1% of the catalog into overly broad groups. Capping bounded this variance, eliminating the micro-cluster and allowing K-Means to identify four balanced, commercially actionable product segments.

---

## 17. Cluster Stability
K-Means was executed across five different random initialization seeds (`[0, 10, 20, 30, 40]`):
- **Seed 0:** WCSS = 8,298.79 | Silhouette Score = 0.4559
- **Seed 10:** WCSS = 8,298.79 | Silhouette Score = 0.4559
- **Seed 20:** WCSS = 8,298.79 | Silhouette Score = 0.4559
- **Seed 30:** WCSS = 8,298.79 | Silhouette Score = 0.4559
- **Seed 40:** WCSS = 8,298.79 | Silhouette Score = 0.4559
- **Mean Silhouette:** $0.4559$ ($\sigma = 0.0000$)
- **Mean WCSS:** $8,298.79$ ($\sigma = 0.00$)

This confirms deterministic stability under `k-means++` initialization in the standardized, winsorized feature space.

---

## 18. Commercial Interpretation
- **Cluster 0 (High-Return / Discount-Dependent):** Products with below-average ratings (3.47) and high return rates (20.02%), requiring heavy discounting (64.94%) to generate moderate volume. These represent catalog friction points where promotions temporarily mask underlying product or expectation mismatches.
- **Cluster 1 (Premium High Performers):** High-ticket SKUs (mean price ₹7,880.65) paired with high customer ratings (4.46), deep social proof (1,114.7 reviews), and low returns (5.43%). These serve as core gross-margin drivers.
- **Cluster 2 (Value High-Sellers):** High-velocity SKUs (mean sales 6,612.6 units) at affordable price points (mean ₹1,095.72) with strong customer satisfaction (4.14). These drive overall revenue velocity and operating cash flow.
- **Cluster 3 (Low-Performance Products):** Sluggish SKUs (mean sales 223.1 units) with low customer satisfaction (3.06) and limited social proof (68.2 reviews). These represent slow-moving inventory that ties up warehouse capacity and working capital.

---

## 19. Inventory & Commercial Strategy Matrix

| Segment | Inventory Strategy | Pricing Strategy | Marketing Strategy | Quality Control Strategy |
|---|---|---|---|---|
| **Cluster 0: High-Return / Discount-Dependent** | Restrict forward purchase orders; hold minimal inventory buffers until return causes are resolved. | Halt perpetual discounting; test slight price increases to reduce bargain-hunting return behavior. | Audit catalog listings; update sizing charts, technical specifications, and imagery to align buyer expectations. | Initiate root-cause return analysis; inspect supplier batches for defects or poor packaging. |
| **Cluster 1: Premium High Performers** | Prioritize stock availability; establish dedicated safety stock to prevent stockouts of high-margin lines. | Maintain price integrity; avoid deep markdowns that could dilute brand equity. | Highlight verified buyer reviews, premium craftsmanship, and brand reputation in paid acquisition. | Enforce strict quality assurance protocols and offer premium customer support. |
| **Cluster 2: Value High-Sellers** | Allocate maximum warehouse capacity; automate re-order points to avoid lost sales on everyday essentials. | Maintain competitive pricing; utilize bundle deals and multi-packs to increase average order value. | Feature best-seller badges; secure top placement in search results and category landing pages. | Monitor fulfillment service levels and protect product packaging during shipping. |
| **Cluster 3: Low-Performance Products** | Minimize inventory holdings; avoid reordering; initiate clearance events to free up working capital. | Apply strategic markdowns to liquidate stagnant stock without undercutting active lines. | Rework product titles, keywords, and search tags, or prepare underperforming SKUs for delisting. | Review customer feedback for recurring complaints; phase out suppliers with ratings below 3.0. |

---

## 20. Deployment Architecture
The clustering pipeline was deployed as an interactive Streamlit application (`app.py`):
1. **Interactive Sidebar:** Allows users to enter product attributes (Price, Rating, Reviews, Returns, Sales, Discounts) or load one of four pre-configured test profiles with a single click.
2. **Deterministic Inference:** Incoming product vectors are transformed using the saved `StandardScaler` and assigned to their nearest cluster using the trained `KMeans` centroids—executing in under 5 milliseconds without model retraining.
3. **Operational Dashboards:** Displays the predicted product segment, cluster identifier, comparative metrics table, and recommended strategies across Inventory, Pricing, Marketing, and Quality.
4. **Catalog Intelligence:** Includes an overview tab presenting catalog-wide segment distributions, PCA projections, and technical validation metrics.

---

## 21. Limitations
1. **Unsupervised Formulation:** Clustering relies on heuristic geometric distance rather than ground-truth product classifications.
2. **Absence of Ground Truth:** The validity of discovered segments must be evaluated using internal cohesion metrics (WCSS, Silhouette) and business domain logic.
3. **Feature Selection Sensitivity:** Cluster boundaries depend on the chosen six features; adding or removing features can alter cluster topology.
4. **Scaling Dependency:** K-Means results are directly tied to standardization; non-standardized data would bias clusters toward larger numerical scales.
5. **Centroid Assumptions:** K-Means assumes spherical, isotropic clusters, which may not capture complex non-linear manifolds.
6. **Synthetic Assumptions:** While statistically realistic, the synthetic dataset does not fully capture edge cases present in live merchant transactional feeds.
7. **Exclusion of Text Features:** Product descriptions, category tags, and raw customer review text are not included in numeric distance calculations.
8. **Lack of Category Granularity:** High-level categories (e.g., Electronics vs. Apparel) are not factored into the primary feature space.
9. **No Seasonality Adjustment:** The dataset represents an aggregated snapshot, smoothing out seasonal peaks (e.g., holiday sales or Prime Day).
10. **Missing Financial Variables:** Crucial financial metrics—such as net profit margins, cost of goods sold (COGS), and fulfillment fees—are not modeled.
11. **No Customer Demographics:** Customer purchasing history, repeat purchase rates, and geographical distribution are omitted.
12. **Supplier Constraints:** Upstream supply chain factors, such as vendor lead times and minimum order quantities (MOQs), are not reflected.
13. **Decision-Support Role:** This system is designed as an analytical decision-support tool; recommendations should be paired with merchant domain expertise rather than executed blindly.

---

## 22. Future Scope
1. **Text Mining & Sentiment Analysis:** Integrate NLP models (e.g., DistilBERT) to extract customer sentiment from written reviews and combine text embeddings with numeric metrics.
2. **Gross Margin & COGS Integration:** Incorporate net margin data to segment products by financial return on inventory investment (GMROI).
3. **Density-Based & Probabilistic Clustering:** Experiment with DBSCAN to detect arbitrary non-spherical clusters and Gaussian Mixture Models (GMM) to quantify probabilistic cluster assignments.
4. **Demand Forecasting & Dynamic Pricing:** Connect segment assignments to time-series forecasting models (e.g., Prophet, ARIMA, or Temporal Fusion Transformers) to enable dynamic pricing adjustments.
5. **Real-Time Streaming Pipelines:** Deploy the model via a REST/gRPC API service (e.g., FastAPI with Redis caching) integrated directly with live warehouse inventory databases.

---

## 23. Conclusion & Answers to Core Case Study Questions

### Question 1: How many distinct product segments exist?
**Answer:** **Four (4)** distinct commercial product segments exist in the catalog. This choice is supported by the Elbow Method curvature inflection point at $K = 4$, a peak Silhouette Coefficient of $0.4559$, and hierarchical dendrogram tree branches cutting cleanly into four major groups.

### Question 2: What characterizes each segment commercially?
**Answer:**
1. **High-Return / Discount-Dependent (19.9%):** High discount frequency ($64.9\%$), elevated returns ($20.0\%$), and mediocre ratings ($3.47$).
2. **Premium High Performers (20.1%):** High prices (mean ₹$7,880.65$), high ratings ($4.46$), substantial review volume ($1,114.7$), and low returns ($5.43\%$).
3. **Value High-Sellers (35.4%):** Accessible pricing (mean ₹$1,095.72$), high sales velocity ($6,612.6$ units), and solid customer satisfaction ($4.14$).
4. **Low-Performance Products (24.6%):** Sluggish movement ($223.1$ units), low customer ratings ($3.06$), and low review counts ($68.2$).

### Question 3: How does outlier treatment affect the clusters?
**Answer:** Outlier capping reduced within-cluster variance (WCSS) by **26.49%** (from $11,288.74$ to $8,298.79$) and improved the Silhouette Score by **7.82%** (from $0.4228$ to $0.4559$). More importantly, it eliminated an isolated 46-product luxury outlier group, allowing K-Means to identify four balanced, commercially actionable product segments.

### Question 4: How was the optimal cluster count chosen?
**Answer:** The optimal count was selected by identifying the sharpest curvature inflection point on the WCSS Elbow Curve ($K=4$), supported by the global maximum of the Silhouette Score curve at $K=4$ ($0.4559$). Ward's hierarchical dendrogram cuts independently corroborated this 4-cluster structure.

### Question 5: Do the two clustering methods agree?
**Answer:** **Yes, exceptionally well.** K-Means and Hierarchical Agglomerative Clustering achieved an **Adjusted Rand Index of 0.9752**, demonstrating 97.5% pairwise assignment agreement.

### Question 6: How stable are the clusters?
**Answer:** **Deterministic and highly stable.** Across five different random initialization seeds (`[0, 10, 20, 30, 40]`), K-Means produced identical WCSS values ($8,298.79$) and Silhouette Scores ($0.4559$), showing zero variance ($\sigma = 0.0000$) under `k-means++` initialization on the standardized, winsorized feature space.

### Question 7: Can segmentation guide inventory strategy?
**Answer:** **Yes.** The segmentation provides clear, operational supply chain rules:
- **Value High-Sellers (35.4%):** Warrant automated replenishment to prevent stockouts of high-velocity items.
- **Premium Flagships (20.1%):** Warrant dedicated safety buffers to protect high-margin revenue.
- **High-Return / Discount-Dependent SKUs (19.9%):** Require restricted stock commitments until return causes are resolved.
- **Low-Performance SKUs (24.6%):** Are prime candidates for clearance markdowns to free up warehouse capacity and working capital.
