"""
Synthetic E-Commerce Product Data Generator.

Generates a realistic commercial dataset of 5,000 products with right-skewed
distributions, non-trivial inter-variable relationships, realistic noise, and
extreme outliers to evaluate clustering and outlier-treatment strategies.
"""

import os
import sys
import numpy as np
import pandas as pd

# Allow direct execution from root or src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import DATA_RAW_DIR, PROJECT_ROOT


def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic e-commerce product dataset.

    Parameters:
        n_samples (int): Number of product records (default: 5000)
        random_state (int): Random seed for reproducibility (default: 42)

    Returns:
        pd.DataFrame: E-commerce product dataset containing:
            - product_id: Unique identifier
            - price: Product unit price in INR (₹100 to ₹100,000)
            - rating: Average customer rating (1.0 to 5.0)
            - review_count: Number of customer reviews (right-skewed)
            - return_rate: Product return percentage (0.0% to 30.0%)
            - sales_volume: Number of units sold (right-skewed)
            - discount_frequency: Percentage of time discounted (0% to 100%)
    """
    np.random.seed(random_state)

    # Underlying profile allocation proportions:
    # A: Premium Performers (~20%)
    # B: Value Sellers (~35%)
    # C: Low-Performance Products (~25%)
    # D: High-Return / Discount-Dependent (~20%)
    n_a = int(0.20 * n_samples)
    n_b = int(0.35 * n_samples)
    n_c = int(0.25 * n_samples)
    n_d = n_samples - (n_a + n_b + n_c)

    # 1. Profile A: Premium Performers (High Price, High Rating, High Reviews, Strong Sales, Low Returns)
    price_a = np.random.lognormal(mean=9.2, sigma=0.45, size=n_a)  # ~ ₹6,000 - ₹30,000
    rating_a = np.random.normal(loc=4.45, scale=0.25, size=n_a)
    review_a = np.random.gamma(shape=4.0, scale=350, size=n_a) + 500
    sales_a = np.random.gamma(shape=5.0, scale=600, size=n_a) + 1200
    return_a = np.random.normal(loc=5.5, scale=2.0, size=n_a)
    discount_a = np.random.normal(loc=18.0, scale=8.0, size=n_a)

    # 2. Profile B: Value Sellers (Moderate/Lower Price, Good Rating, Moderate Reviews, Very High Sales)
    price_b = np.random.lognormal(mean=6.8, sigma=0.55, size=n_b)  # ~ ₹400 - ₹3,500
    rating_b = np.random.normal(loc=4.15, scale=0.30, size=n_b)
    review_b = np.random.gamma(shape=3.0, scale=250, size=n_b) + 200
    sales_b = np.random.gamma(shape=6.5, scale=700, size=n_b) + 1500
    return_b = np.random.normal(loc=7.5, scale=2.5, size=n_b)
    discount_b = np.random.normal(loc=35.0, scale=12.0, size=n_b)

    # 3. Profile C: Low-Performance Products (Budget/Moderate Price, Lower Rating, Low Reviews, Low Sales)
    price_c = np.random.lognormal(mean=7.3, sigma=0.60, size=n_c)  # ~ ₹600 - ₹4,500
    rating_c = np.random.normal(loc=3.10, scale=0.55, size=n_c)
    review_c = np.random.exponential(scale=60, size=n_c) + 10
    sales_c = np.random.gamma(shape=2.0, scale=120, size=n_c) + 30
    return_c = np.random.normal(loc=12.5, scale=4.0, size=n_c)
    discount_c = np.random.normal(loc=25.0, scale=10.0, size=n_c)

    # 4. Profile D: High-Return / Discount-Dependent (Moderate Price, Moderate Rating, High Returns, High Discounts)
    price_d = np.random.lognormal(mean=7.8, sigma=0.50, size=n_d)  # ~ ₹1,200 - ₹6,500
    rating_d = np.random.normal(loc=3.45, scale=0.45, size=n_d)
    review_d = np.random.gamma(shape=2.5, scale=150, size=n_d) + 80
    sales_d = np.random.gamma(shape=3.2, scale=320, size=n_d) + 300
    return_d = np.random.normal(loc=20.0, scale=3.5, size=n_d)
    discount_d = np.random.normal(loc=65.0, scale=12.0, size=n_d)

    # Combine profiles
    price = np.concatenate([price_a, price_b, price_c, price_d])
    rating = np.concatenate([rating_a, rating_b, rating_c, rating_d])
    review_count = np.concatenate([review_a, review_b, review_c, review_d])
    sales_volume = np.concatenate([sales_a, sales_b, sales_c, sales_d])
    return_rate = np.concatenate([return_a, return_b, return_c, return_d])
    discount_frequency = np.concatenate([discount_a, discount_b, discount_c, discount_d])

    # Add realistic cross-feature correlations and noise
    # - Higher rating slightly boosts sales
    rating_effect = (rating - 3.5) * 250
    # - Heavy discounting moderately boosts sales
    discount_effect = (discount_frequency / 100.0) * 400
    # - Excessive return rate dampens sales
    return_effect = -(return_rate - 10.0) * 40

    sales_volume = sales_volume + rating_effect + discount_effect + return_effect

    # Inject deliberate extreme outliers (Part 6 requirements: extreme price & sales outliers)
    # 45 extreme luxury price outliers
    extreme_price_indices = np.random.choice(n_samples, size=45, replace=False)
    price[extreme_price_indices] = np.random.uniform(65000, 99500, size=45)

    # 55 extreme blockbuster sales volume outliers
    extreme_sales_indices = np.random.choice(n_samples, size=55, replace=False)
    sales_volume[extreme_sales_indices] = np.random.uniform(14000, 26000, size=55)

    # Boundary enforcement and logical rounding
    price = np.clip(np.round(price, 2), 100.0, 100000.0)
    rating = np.clip(np.round(rating, 2), 1.0, 5.0)
    review_count = np.clip(np.round(review_count).astype(int), 1, None)
    return_rate = np.clip(np.round(return_rate, 2), 0.0, 30.0)
    sales_volume = np.clip(np.round(sales_volume).astype(int), 1, None)
    discount_frequency = np.round(np.clip(discount_frequency, 0.0, 100.0), 1)
    discount_frequency[discount_frequency < 0.0] = 0.0

    # Generate sequential unique Product IDs
    product_ids = [f"PROD_{i+1:05d}" for i in range(n_samples)]

    df = pd.DataFrame(
        {
            "product_id": product_ids,
            "price": price,
            "rating": rating,
            "review_count": review_count,
            "return_rate": return_rate,
            "sales_volume": sales_volume,
            "discount_frequency": discount_frequency,
        }
    )

    # Shuffle dataset so profiles are uniformly dispersed
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def save_raw_dataset(df: pd.DataFrame, output_path: str = None) -> str:
    """Save raw dataset to CSV."""
    if output_path is None:
        os.makedirs(DATA_RAW_DIR, exist_ok=True)
        output_path = os.path.join(DATA_RAW_DIR, "ecommerce_products.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to: {output_path} (Shape: {df.shape})")
    return output_path


if __name__ == "__main__":
    df = generate_synthetic_data()
    save_raw_dataset(df)
