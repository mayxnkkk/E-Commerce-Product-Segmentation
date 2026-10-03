# Data Generation Methodology & Documentation

## 1. Executive Summary & Purpose
This document describes the generation process, theoretical foundations, mathematical distributions, and commercial assumptions used to construct the synthetic e-commerce product dataset (`data/raw/ecommerce_products.csv`).

### Why Synthetic Data Was Used
In real-world public repositories (such as the Kaggle Amazon Sales Dataset), variables like **sales volume** (actual units sold) and **product return rate** are proprietary merchant metrics that retailers rarely disclose publicly. To simulate a genuine enterprise inventory and pricing optimization problem, we generated a mathematically coherent, commercially realistic synthetic dataset of **5,000 products** with controlled yet realistic properties.

---

## 2. Dataset Overview
- **Total Records:** 5,000 products
- **Total Columns:** 7 (1 identifier + 6 numerical clustering features)
- **Random Seed:** `42` (`np.random.seed(42)` for deterministic reproducibility)
- **Target Currency:** Indian Rupee (INR - ₹)

| Feature | Data Type | Units / Range | Commercial Meaning | Distribution Type |
|---|---|---|---|---|
| `product_id` | String | `PROD_00001` - `PROD_05000` | Unique SKU identifier | Sequential Identifier |
| `price` | Float | ₹100.00 – ₹100,000.00 | Retail unit selling price | Right-skewed (Lognormal) |
| `rating` | Float | 1.00 – 5.00 | Average customer star rating | Truncated Gaussian |
| `review_count` | Integer | 1 – 5,000+ | Total submitted customer reviews | Right-skewed (Gamma + Exponential) |
| `return_rate` | Float | 0.00% – 30.00% | Percentage of shipped units returned | Gaussian / Bounded |
| `sales_volume` | Integer | 1 – 26,000+ units | Total units sold over period | Right-skewed (Gamma) |
| `discount_frequency`| Float | 0.0% – 100.0% | Percentage of active time on discount | Gaussian / Bounded |

---

## 3. Commercial Logic & Underlying Profiles
Rather than generating six statistically independent columns, the generator models **four underlying commercial profiles (tendencies)**. These profiles define natural commercial behavioral archetypes, but realistic random noise is injected so clusters are organic, overlapping, and not artificially pre-labeled.

### Profile A: Premium Performers (~20% of catalog)
- **Characteristics:** High prices (₹6,000 – ₹35,000), superior customer ratings (4.2 – 4.9), strong review depth, healthy unit velocity, low return rates (2% – 8%), minimal reliance on heavy discounts.
- **Commercial Role:** High gross-margin drivers; brand flagships.

### Profile B: Value High-Sellers (~35% of catalog)
- **Characteristics:** Accessible budget-to-mid pricing (₹300 – ₹3,500), consistent customer satisfaction (3.8 – 4.6), high unit sales volumes (2,000 – 9,000 units), moderate discounting.
- **Commercial Role:** Volume drivers; cash-flow generators requiring high inventory availability.

### Profile C: Low-Performance Products (~25% of catalog)
- **Characteristics:** Modest pricing, lower consumer ratings (2.0 – 3.7), negligible reviews, sluggish sales (50 – 600 units), moderate return rate.
- **Commercial Role:** Slow-moving or dead stock candidates requiring clearance or delisting.

### Profile D: High-Return / Discount-Dependent (~20% of catalog)
- **Characteristics:** Mid-range prices, mediocre satisfaction (2.8 – 4.0), high return rates (15% – 28%), heavily reliant on deep promotions (50% – 90% discount frequency).
- **Commercial Role:** Operational risk SKUs requiring quality audits and promotional re-evaluation.

---

## 4. Cross-Feature Interdependencies
To mirror actual marketplace dynamics:
1. **Rating Boost:** Higher ratings moderately lift consumer conversion ($\Delta \text{Sales} \propto (\text{rating} - 3.5) \times 250$).
2. **Promotional Velocity:** Frequent discounting increases unit movement ($\Delta \text{Sales} \propto \frac{\text{discount}}{100} \times 400$).
3. **Friction Penalty:** Elevated return rates negatively impact organic sales velocity ($\Delta \text{Sales} \propto -(\text{return\_rate} - 10) \times 40$).

---

## 5. Deliberate Outlier Injection
To test **robust outlier detection (IQR)** and examine the comparative effect of **outlier capping (winsorization)** on clustering:
- **Extreme Luxury Prices:** 45 products deliberately assigned prices between ₹65,000 and ₹99,500.
- **Extreme Viral Blockbusters:** 55 products deliberately assigned sales volumes between 14,000 and 26,000 units.

These represent genuine high-value items and viral hits commonly found on platforms like Amazon and Flipkart.

---

## 6. Limitations of the Synthetic Dataset
1. **Unobserved Covariates:** Does not incorporate seasonal spikes (e.g., Diwali or Big Billion Days), supplier lead time, or net profit margins.
2. **Category Agnostic:** Lacks hierarchical product taxonomies (e.g., Electronics vs. Apparel).
3. **No Time-Series Dynamics:** Aggregated snapshot representing an annualized view rather than day-to-day fluctuations.

---

## 7. How to Explain in a Viva (30-Second Script)
> "Because real e-commerce sales volumes and return rates are confidential company data, we synthesized 5,000 products using seed 42. We modeled four realistic commercial archetypes using lognormal and gamma distributions to capture real-world skewness. We also deliberately injected luxury price and viral sales outliers so we could scientifically test whether IQR capping improves K-Means cluster stability."
