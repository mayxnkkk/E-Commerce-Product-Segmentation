# Live Project Demonstration Guide
## 5-Minute Step-by-Step Viva & Presentation Walkthrough

---

## 1. Preparation Before Demonstration
Open your terminal in the project directory:
```bash
cd /Users/mayanknegi/Desktop/ml_final
```
Ensure all pipeline files, trained artifacts, and plots are generated.

---

## 2. Step-by-Step Live Demo Flow (5 Minutes Total)

### Step 1: Launch the Application (0:00 – 0:30)
Run the following command in your terminal:
```bash
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

### Step 2: Show the Application Interface (0:30 – 1:00)
- Highlight the title: **E-Commerce Product Segmentation**.
- Briefly explain the sidebar with the six product input attributes: Price (₹), Average Rating (1–5), Review Count, Return Rate (%), Sales Volume, and Discount Frequency (%).
- Note that the application loads pre-trained artifacts (`scaler.pkl` and `kmeans_model.pkl`) to make predictions in real time without retraining.

### Step 3: Enter Sample Products & Predict Segments (1:00 – 2:30)
Use the **Quick-Load Sample Preset** dropdown in the sidebar to test realistic product scenarios:

#### Test Product 1: Premium Flagship
- **Inputs:** Price: ₹12,500 | Rating: 4.65 | Reviews: 1,800 | Return Rate: 4.2% | Sales: 4,800 | Discount: 15.0%
- **Click:** `Predict Product Segment`
- **Output:** **Cluster 1 — Premium High Performers**
- **Explanation:** Highlight the high rating and sales velocity alongside low returns and minimal discounting. Explain the recommendations: prioritize stock availability and maintain price integrity.

#### Test Product 2: High-Volume Value Seller
- **Inputs:** Price: ₹850 | Rating: 4.20 | Reviews: 850 | Return Rate: 6.8% | Sales: 7,200 | Discount: 35.0%
- **Click:** `Predict Product Segment`
- **Output:** **Cluster 2 — Value High-Sellers**
- **Explanation:** Point out the high sales volume (7,200 units) at an accessible price point. Explain the recommendations: automate reorder points to sustain supply continuity.

#### Test Product 3: Sluggish / Low Performance
- **Inputs:** Price: ₹1,450 | Rating: 2.90 | Reviews: 45 | Return Rate: 13.5% | Sales: 180 | Discount: 20.0%
- **Click:** `Predict Product Segment`
- **Output:** **Cluster 3 — Low-Performance Products**
- **Explanation:** Point out the low ratings (2.9) and minimal sales movement (180 units). Explain the recommendations: plan clearance events to free up warehouse capacity.

#### Test Product 4: High-Return / Clearance SKU
- **Inputs:** Price: ₹3,200 | Rating: 3.30 | Reviews: 320 | Return Rate: 22.5% | Sales: 1,100 | Discount: 70.0%
- **Click:** `Predict Product Segment`
- **Output:** **Cluster 0 — High-Return / Discount-Dependent Products**
- **Explanation:** Highlight the 22.5% return rate and 70% discount reliance. Explain the recommendations: restrict forward purchase orders and audit listings for misleading specifications.

### Step 4: Show Catalog Intelligence & Profiles (2:30 – 3:30)
- Click the second tab: **📊 Segment Intelligence & Overview**.
- Show the catalog distribution: Value High-Sellers (35.4%), Low-Performance (24.6%), Premium High Performers (20.1%), and High-Return (19.9%).
- Show the 2D PCA cluster visualization plot and explain that PC1 and PC2 capture 68.7% of catalog variance.

### Step 5: Walk Through Notebook & Methodology (3:30 – 5:00)
Switch to the browser or IDE tab showing `notebooks/ecommerce_product_segmentation.ipynb`:
1. **EDA & Distributions:** Show the right-skewed histograms for Price and Sales Volume.
2. **Outlier Box Plot:** Show the before-and-after box plots, explaining that capping preserved all 5,000 observations while bounding variance.
3. **Elbow Curve:** Point to the sharp bend at $K = 4$ and the corresponding peak in the Silhouette score ($0.4559$).
4. **Dendrogram:** Show that cutting the dendrogram tree yields four clean vertical stems.
5. **Comparative Metrics:** Show the Adjusted Rand Index of 0.9752, proving strong agreement between K-Means and Hierarchical Clustering.
6. **Concluding Takeaway:** Summarize that outlier capping reduced within-cluster variance by 26.5% and eliminated an isolated 46-item micro-cluster, ensuring balanced segments.

---

## 3. Four Sample Test Products Cheat-Sheet

| Field | Test 1: Premium Flagship | Test 2: Value Seller | Test 3: Low Performance | Test 4: High Return / Discount |
|---|:---:|:---:|:---:|:---:|
| **Price** | ₹12,500.00 | ₹850.00 | ₹1,450.00 | ₹3,200.00 |
| **Rating** | 4.65 | 4.20 | 2.90 | 3.30 |
| **Reviews** | 1,800 | 850 | 45 | 320 |
| **Return Rate** | 4.2% | 6.8% | 13.5% | 22.5% |
| **Sales Volume** | 4,800 units | 7,200 units | 180 units | 1,100 units |
| **Discount Freq** | 15.0% | 35.0% | 20.0% | 70.0% |
| **Predicted Segment** | **Cluster 1: Premium High Performers** | **Cluster 2: Value High-Sellers** | **Cluster 3: Low-Performance Products** | **Cluster 0: High-Return / Discount-Dependent** |
