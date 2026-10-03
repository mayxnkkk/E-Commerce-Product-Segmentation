"""
Exploratory Data Analysis (EDA) Module.

Generates comprehensive academic-grade visualizations:
1. Histograms for all 6 features (distributions)
2. Box plots before and after outlier treatment
3. Correlation heatmap
4. Scatter plots exploring commercial relationships
All figures are saved under results/figures/ with proper labels and titles.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import (
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    FEATURE_COLUMNS,
    setup_matplotlib_style,
    save_figure,
)


def generate_all_eda_visualizations(
    before_csv: str = None, after_csv: str = None
):
    """Generate and save all required EDA visualizations."""
    setup_matplotlib_style()

    if before_csv is None:
        before_csv = os.path.join(DATA_PROCESSED_DIR, "products_before_outlier_treatment.csv")
    if after_csv is None:
        after_csv = os.path.join(DATA_PROCESSED_DIR, "products_after_outlier_treatment.csv")

    df_before = pd.read_csv(before_csv)
    df_after = pd.read_csv(after_csv)

    print("Generating EDA visualizations...")

    # 1. Feature Histograms (6 individual or unified grid)
    # Individual histograms as requested by Part 14
    hist_specs = [
        ("price", "Distribution of Product Price", "Price (INR)", "Frequency", 50, "#1f77b4"),
        ("rating", "Distribution of Customer Ratings", "Average Star Rating (1-5)", "Frequency", 30, "#2ca02c"),
        ("review_count", "Distribution of Customer Review Counts", "Number of Reviews", "Frequency", 40, "#ff7f0e"),
        ("return_rate", "Distribution of Product Return Rates", "Return Rate (%)", "Frequency", 35, "#d62728"),
        ("sales_volume", "Distribution of Sales Volumes", "Units Sold", "Frequency", 45, "#9467bd"),
        ("discount_frequency", "Distribution of Discount Frequencies", "Discount Frequency (%)", "Frequency", 35, "#8c564b"),
    ]

    for col, title, xlabel, ylabel, bins, color in hist_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_before[col], bins=bins, kde=True, color=color, ax=ax, edgecolor="black", alpha=0.6)
        ax.set_title(title, pad=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        save_figure(fig, f"hist_{col}.png")

    # Combined 6-panel Histograms for quick comprehensive viewing
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, (col, title, xlabel, ylabel, bins, color) in enumerate(hist_specs):
        sns.histplot(df_before[col], bins=bins, kde=True, color=color, ax=axes[i], edgecolor="black", alpha=0.6)
        axes[i].set_title(title)
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)
    save_figure(fig, "histograms_all_features.png")

    # 2. Box Plots for Outlier Inspection (Price, Sales Volume, Rating, Return Rate)
    box_cols = ["price", "sales_volume", "rating", "return_rate"]
    box_labels = ["Price (INR)", "Sales Volume (Units)", "Star Rating", "Return Rate (%)"]

    for col, label in zip(box_cols, box_labels):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.boxplot(x=df_before[col], color="#aec7e8", ax=ax, flierprops=dict(marker="o", color="red", alpha=0.5))
        ax.set_title(f"Box Plot: {col.replace('_', ' ').title()} (Before Outlier Treatment)", pad=12)
        ax.set_xlabel(label)
        save_figure(fig, f"boxplot_{col}_before.png")

    # Combined Box Plot Comparison: Before vs After Outlier Treatment for Price and Sales Volume
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    sns.boxplot(x=df_before["price"], ax=axes[0, 0], color="#ff9896", flierprops=dict(marker="o", color="red", alpha=0.5))
    axes[0, 0].set_title("Price: Before IQR Capping (Extreme Outliers Visible)")
    axes[0, 0].set_xlabel("Price (INR)")

    sns.boxplot(x=df_after["price"], ax=axes[0, 1], color="#98df8a")
    axes[0, 1].set_title("Price: After IQR Capping (Outliers Capped at Upper Bound)")
    axes[0, 1].set_xlabel("Price (INR)")

    sns.boxplot(x=df_before["sales_volume"], ax=axes[1, 0], color="#ff9896", flierprops=dict(marker="o", color="red", alpha=0.5))
    axes[1, 0].set_title("Sales Volume: Before IQR Capping (Extreme Blockbusters)")
    axes[1, 0].set_xlabel("Sales Volume (Units)")

    sns.boxplot(x=df_after["sales_volume"], ax=axes[1, 1], color="#98df8a")
    axes[1, 1].set_title("Sales Volume: After IQR Capping (Outliers Capped at Upper Bound)")
    axes[1, 1].set_xlabel("Sales Volume (Units)")

    save_figure(fig, "outlier_before_after_comparison.png")

    # 3. Correlation Heatmap
    corr_matrix = df_before[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix (E-Commerce Products)", pad=15)
    save_figure(fig, "correlation_heatmap.png")

    # 4. Scatter Plots (Part 14 specifications)
    # 1. Price vs Sales Volume
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=df_before, x="price", y="sales_volume", alpha=0.4, color="#1f77b4", ax=ax)
    ax.set_title("Relationship: Price vs. Sales Volume", pad=12)
    ax.set_xlabel("Price (INR)")
    ax.set_ylabel("Sales Volume (Units Sold)")
    save_figure(fig, "scatter_price_vs_sales.png")

    # 2. Rating vs Sales Volume
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=df_before, x="rating", y="sales_volume", alpha=0.4, color="#2ca02c", ax=ax)
    ax.set_title("Relationship: Average Rating vs. Sales Volume", pad=12)
    ax.set_xlabel("Average Rating (1.0 to 5.0)")
    ax.set_ylabel("Sales Volume (Units Sold)")
    save_figure(fig, "scatter_rating_vs_sales.png")

    # 3. Review Count vs Sales Volume
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=df_before, x="review_count", y="sales_volume", alpha=0.4, color="#ff7f0e", ax=ax)
    ax.set_title("Relationship: Review Count vs. Sales Volume", pad=12)
    ax.set_xlabel("Customer Review Count")
    ax.set_ylabel("Sales Volume (Units Sold)")
    save_figure(fig, "scatter_review_vs_sales.png")

    # 4. Discount Frequency vs Sales Volume
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=df_before, x="discount_frequency", y="sales_volume", alpha=0.4, color="#9467bd", ax=ax)
    ax.set_title("Relationship: Discount Frequency vs. Sales Volume", pad=12)
    ax.set_xlabel("Discount Frequency (%)")
    ax.set_ylabel("Sales Volume (Units Sold)")
    save_figure(fig, "scatter_discount_vs_sales.png")

    # 5. Price vs Rating
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=df_before, x="price", y="rating", alpha=0.4, color="#e377c2", ax=ax)
    ax.set_title("Relationship: Price vs. Customer Rating", pad=12)
    ax.set_xlabel("Price (INR)")
    ax.set_ylabel("Customer Star Rating (1.0 to 5.0)")
    save_figure(fig, "scatter_price_vs_rating.png")

    print(f"All EDA visualizations successfully saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_eda_visualizations()
