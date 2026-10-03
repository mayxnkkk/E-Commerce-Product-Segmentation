"""
Utility functions for file paths, visualization styling, and result serialization.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Base project path (repository root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

FEATURE_COLUMNS = [
    "price",
    "rating",
    "review_count",
    "return_rate",
    "sales_volume",
    "discount_frequency",
]


def setup_matplotlib_style():
    """Configure clean, modern aesthetics for academic figures."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "semibold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
        }
    )


def save_figure(fig, filename: str, dpi: int = 300):
    """Save matplotlib figure with high DPI and tight bounding box."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {filepath}")
    return filepath


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy scalars and arrays."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def save_json(data: dict, filepath: str):
    """Save dictionary to JSON file with indentation and numpy support."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"Saved JSON: {filepath}")
