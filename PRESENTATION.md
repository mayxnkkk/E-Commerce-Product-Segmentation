# Presentation: E-Commerce Product Segmentation Using Clustering
## 12-Slide Academic Capstone Defense Deck

---

## Slide 1 — Title
### Slide Content
- **Project Title:** E-Commerce Product Segmentation Using Machine Learning
- **Subtitle:** Unsupervised Clustering for Data-Driven Inventory Allocation and Pricing Strategy
- **Student Name:** Machine Learning Capstone Team
- **Tools & Tech:** Python 3.10+, Scikit-Learn, Pandas, SciPy, Matplotlib, Streamlit

### Speaker Notes
> "Good morning, respected professors and evaluators. Today, we present our project on 'E-Commerce Product Segmentation Using Machine Learning'. In modern e-commerce, retailers manage thousands of product listings, but treating every product identically leads to stockouts, bloated warehouse holding costs, and margin erosion. Our project builds an unsupervised clustering system to automatically segment an online retailer's catalog based on multidimensional commercial performance metrics to guide inventory and pricing strategies."

### Important Point
**Clustering transforms raw catalog metrics into actionable inventory and pricing decisions without manual rules.**

---

## Slide 2 — Problem Statement
### Slide Content
- **Business Challenge:** Treating diverse products uniformly leads to capital misallocation and stockouts.
- **Traditional Limitation:** Broad category tags (e.g., 'Electronics') fail to distinguish profitable flagships from high-return items.
- **Core Question:** How can an e-commerce merchant algorithmically segment thousands of products using commercial behavior (price, rating, reviews, returns, sales volume, and discounts)?
- **Goal:** Drive four operational pillars: Inventory Replenishment, Pricing Integrity, Promotional Efficiency, and Return Mitigation.

### Speaker Notes
> "On major platforms like Amazon, two products in the exact same merchandise category can perform completely differently. One might be a rapid cash generator, while another suffers from high return rates and eats up warehouse space. Our goal is to segment products by their true operational behaviors—balancing demand velocity, price sensitivity, customer satisfaction, and reverse logistics."

### Important Point
**Category tags tell you what a product is; machine learning clustering tells you how a product performs.**

---

## Slide 3 — Project Objectives
### Slide Content
1. **Analyze Commercial Data:** Ingest 5,000 product records and model real-world relationships.
2. **Treat Skewed Outliers:** Implement IQR detection and capping (winsorization) on `price` and `sales_volume`.
3. **Feature Scaling:** Apply `StandardScaler` to prevent high-magnitude features from dominating Euclidean distance.
4. **Determine Optimal $K$:** Use the Elbow Method (WCSS) and Silhouette Score to identify natural cluster counts.
5. **Model Training & Comparison:** Train K-Means and compare with Ward's Hierarchical Clustering.
6. **2D PCA Visualization:** Project the 6D space onto 2 principal components to validate cluster cohesion.
7. **Production Deployment:** Build an interactive Streamlit application for real-time SKU segment classification.

### Speaker Notes
> "Our objectives span the full machine learning lifecycle: data generation, statistical preprocessing, distance scaling, optimal cluster selection, hierarchical validation, PCA visualization, and live web deployment. We focus on keeping the approach rigorous, explainable, and reproducible."

### Important Point
**Every phase—from preprocessing to real-time deployment—is mathematically grounded and production-ready.**

---

## Slide 4 — Dataset & Features
### Slide Content
- **Dataset Size:** 5,000 product records (`ecommerce_products.csv`).
- **Reproducibility:** Deterministic seed `random_state = 42`.
- **6 Numerical Features:**
  - `price`: ₹108.84 to ₹99,323.79 (right-skewed lognormal)
  - `rating`: 1.09 to 5.00 (average star rating)
  - `review_count`: 10 to 3,745 (social proof depth)
  - `return_rate`: 0.40% to 29.56% (reverse-logistics friction)
  - `sales_volume`: 1 to 25,943 units (demand velocity)
  - `discount_frequency`: 0.0% to 100.0% (promotional exposure)
- *Note: `product_id` is retained strictly as an administrative key and excluded from model inputs.*

### Speaker Notes
> "Public e-commerce datasets typically lack proprietary merchant numbers such as sales volume and return rates. To solve this realistically, we modeled 5,000 products reflecting authentic marketplace dynamics: right-skewed pricing and sales distributions, four underlying commercial tendencies, and natural noise. We also intentionally injected luxury price and viral sales volume outliers to evaluate outlier treatment."

### Important Point
**We cluster on 6 core commercial dimensions; `product_id` is excluded to prevent spurious distance indexing.**

---

## Slide 5 — Exploratory Data Analysis & Outlier Detection
### Slide Content
- **EDA Insights:**
  - `price` and `sales_volume` display pronounced right-skewness.
  - Moderate positive correlation between customer rating and unit sales ($r = +0.31$).
- **IQR Outlier Detection ($1.5 \times IQR$):**
  - **Price:** $Q_1 = ₹962.06$, $Q_3 = ₹4,014.64$, $IQR = ₹3,052.58$. Upper Bound = **₹8,593.50** (674 outliers, 13.48%).
  - **Sales Volume:** $Q_1 = 546$, $Q_3 = 5,845$, $IQR = 5,299$. Upper Bound = **13,792 units** (55 outliers, 1.10%).
- **Treatment Strategy:** Capping (Winsorization) to upper bounds rather than deletion.

### Speaker Notes
> "During exploratory data analysis, we confirmed that e-commerce data is heavily right-skewed. Using the 1.5 times IQR rule, we detected 674 price outliers and 55 sales volume outliers. Instead of deleting these 729 products—which would discard genuine luxury lines and blockbuster sellers—we capped them at their respective upper bounds. This controls variance while retaining all 5,000 products."

### Important Point
**We cap rather than delete: extreme sales and luxury prices are genuine business events, not data entry errors.**

---

## Slide 6 — Preprocessing & Feature Scaling
### Slide Content
- **Data Hygiene:** 0 missing values, 0 duplicate rows, all boundaries verified.
- **The Distance Scaling Problem:**
  - Euclidean distance formula: $d(x, y) = \sqrt{\sum (x_i - y_i)^2}$
  - An unscaled price range (₹100 to ₹8,593) would completely overpower a customer rating scale (1.0 to 5.0).
- **The Solution:** `StandardScaler` (Z-score standardization):
  $$z = \frac{x - \mu}{\sigma}$$
- **Artifact:** Fitted scaler saved as `models/scaler.pkl` to transform new incoming inference data.

### Speaker Notes
> "Because K-Means uses Euclidean distance, features with larger numeric values naturally dominate the clustering. Without scaling, a difference of ₹1,000 in price would completely drown out a 1-star difference in customer rating. By applying `StandardScaler`, every feature is standardized to zero mean and unit variance, giving all six dimensions equal mathematical footing."

### Important Point
**Without standardization, clustering is just a price-and-volume sorter; `StandardScaler` ensures all six features contribute equally.**

---

## Slide 7 — K-Means & The Elbow Method
### Slide Content
- **Evaluation Range:** $K \in [2, 10]$ with `k-means++` initialization and `random_state = 42`.
- **Elbow Curve (WCSS / Inertia):**
  - $K=2 \to 3$: WCSS drops by 5,734 (17,718 $\to$ 11,985)
  - $K=3 \to 4$: WCSS drops by 3,686 (11,985 $\to$ 8,298)
  - $K=4 \to 5$: WCSS drops by only 572 (8,298 $\to$ 7,726)
  - **Sharp elbow inflection point at $K = 4$.**
- **Silhouette Coefficient Curve:**
  - $K=2: 0.3841$ | $K=3: 0.4191$ | **$K=4: 0.4559$ (Peak)** | $K=5: 0.3913$
- **Decision:** **$K = 4$** is unambiguously selected as the optimal cluster count.

### Speaker Notes
> "To select the optimal number of clusters, we evaluated K from 2 to 10. The WCSS elbow curve exhibits its sharpest bend at K=4. Beyond K=4, the marginal reduction in variance drops off significantly. Supporting this, the Silhouette score reaches its global maximum of 0.4559 at K=4. Both metrics clearly point to K=4 as the optimal cluster count."

### Important Point
**$K = 4$ is confirmed by both the second derivative of the WCSS Elbow curve and the peak of the Silhouette score.**

---

## Slide 8 — Hierarchical Clustering & Dendrogram
### Slide Content
- **Algorithm:** Agglomerative Hierarchical Clustering (`linkage = 'ward'`, Euclidean metric).
- **Silhouette Score:** **0.4532** (virtually identical to K-Means: $0.4559$).
- **Dendrogram Analysis:**
  - Constructed using SciPy with Ward's minimum variance criterion.
  - Horizontal cuts at distance threshold 25–35 reveal **four major vertical tree stems**.
- **Adjusted Rand Index (ARI):** **0.9752** (97.5% pairwise agreement with K-Means).

### Speaker Notes
> "To ensure our results weren't just an artifact of K-Means' spherical assumptions, we ran Agglomerative Hierarchical Clustering using Ward's linkage. The dendrogram clearly branches into four primary groups. The hierarchical model achieved a Silhouette score of 0.4532, and the Adjusted Rand Index between the two algorithms was 0.9752, showing that both methods reach essentially the same conclusion."

### Important Point
**An Adjusted Rand Index of 0.9752 proves that the 4-cluster structure reflects genuine data density, not algorithm bias.**

---

## Slide 9 — 2D PCA Visualization
### Slide Content
- **Technique:** Principal Component Analysis (`PCA(n_components=2)`).
- **Variance Explained:**
  - **PC1:** 46.20% (Product Performance & Health)
  - **PC2:** 22.49% (Price-Volume Tradeoff)
  - **Total 2D Explained Variance:** **68.69%**
- **Interpretation:**
  - Clusters form distinct, well-separated groupings in 2D space.
  - PCA is used strictly for visual validation; it was not used to generate the clusters.

### Speaker Notes
> "Because humans cannot visualize six dimensions directly, we used Principal Component Analysis to project the data onto a 2D plane. PC1 and PC2 together account for 68.69% of the total variance. As shown in the scatter plot, the four K-Means clusters form clear, distinct groupings with minimal overlap, validating that our segments are geometrically well-separated."

### Important Point
**PCA is a 2D projection tool used to verify cluster separation visually; the actual clustering occurred in 6D space.**

---

## Slide 10 — Discovered Commercial Cluster Profiles
### Slide Content

| Cluster | Segment Name | Share | Mean Price | Rating | Sales | Return Rate | Discount |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | **High-Return / Discount-Dependent** | 19.88% | ₹2,833 | 3.47 | 1,202 | **20.02%** | **64.94%** |
| **1** | **Premium High Performers** | 20.14% | **₹7,881** | **4.46** | 4,747 | **5.43%** | **17.99%** |
| **2** | **Value High-Sellers** | 35.38% | ₹1,096 | 4.14 | **6,613** | 7.60% | 35.14% |
| **3** | **Low-Performance Products** | 24.60% | ₹1,839 | **3.06** | **223** | 12.39% | 24.09% |

### Speaker Notes
> "Here are the four clusters discovered by K-Means:
> - **Cluster 0 (19.9%):** High-Return / Discount-Dependent products with a 20% return rate and 65% discount dependency.
> - **Cluster 1 (20.1%):** Premium High Performers with high prices (₹7,881), high ratings (4.46), and a low 5.4% return rate.
> - **Cluster 2 (35.4%):** Value High-Sellers—our catalog backbone—driving 6,613 units on average at an accessible ₹1,096 price point.
> - **Cluster 3 (24.6%):** Low-Performance SKUs averaging only 223 units with low 3.06 ratings."

### Important Point
**The clusters map naturally to real commercial archetypes: Flagships, Workhorses, At-Risk SKUs, and Stagnant Lines.**

---

## Slide 11 — Outlier Impact & Cluster Stability
### Slide Content
- **Outlier Capping Experiment (Dataset A vs. Dataset B):**
  - **WCSS:** Dropped from 11,288.74 $\to$ **8,298.79** (-26.49% variance reduction).
  - **Silhouette Score:** Increased from 0.4228 $\to$ **0.4559** (+7.82% tightness gain).
  - **Cluster Balance:** Eliminated a tiny, non-actionable 46-product luxury outlier group (0.9%), restoring balanced commercial discovery across the catalog.
- **Random Initialization Stability (Seeds 0, 10, 20, 30, 40):**
  - WCSS is identical across all seeds: **8,298.79** ($\sigma = 0.00$).
  - Silhouette is identical: **0.4559** ($\sigma = 0.0000$).

### Speaker Notes
> "To test the importance of outlier handling, we clustered both the raw and capped datasets. Outlier capping reduced within-cluster variance by 26.5% and improved the silhouette score by 7.8%. Crucially, before capping, 46 extreme luxury items formed an isolated micro-cluster, distorting the remaining 99% of the catalog. Capping resolved this distortion. Furthermore, running the model across five random seeds produced identical metrics, demonstrating high model stability."

### Important Point
**Outlier capping prevented 46 luxury products from monopolizing an entire cluster, restoring balanced segment discovery.**

---

## Slide 12 — Deployment, Strategy & Future Scope
### Slide Content
- **Interactive Streamlit Deployment (`app.py`):**
  - Instant classification of user-entered product metrics in $<5$ ms using pre-trained centroids.
  - Displays cluster number, benchmark comparisons, and actionable strategy recommendations.
- **Commercial Action Plan:**
  - **Value Sellers:** Automated replenishment to prevent stockouts.
  - **Premium Flagships:** Safety stock buffers and protected brand pricing.
  - **High-Return:** Quality inspections and catalog accuracy updates.
  - **Low-Performance:** Clearance markdowns to free up warehouse capacity.
- **Future Work:** Integrate customer review NLP, factor in net profit margins, and test GMM / DBSCAN models.

### Speaker Notes
> "Finally, we deployed the model in a lightweight Streamlit application. Users can enter a product's price, rating, reviews, return rate, sales, and discounts, and the app assigns it to a segment in under 5 milliseconds. The app then provides tailored recommendations across inventory, pricing, marketing, and quality assurance. In the future, we plan to incorporate review text sentiment and net profit margin data."

### Important Point
**We built a complete, production-ready pipeline: from data ingestion and modeling to real-time interactive business decision support.**
