"""
Script to execute Segment 3: Exploratory Data Analysis & Financial Pattern Discovery.
Generates publication-quality charts and metrics without data leakage.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("eda_pipeline")

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "eda"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"


def run_eda():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training dataset for EDA (ensuring test set isolation)...")
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed train data not found at {TRAIN_DATA_PATH}. Run preprocessing first.")

    df_train = pd.read_csv(TRAIN_DATA_PATH)
    target_col = "Bankrupt?"
    feature_cols = [c for c in df_train.columns if c != target_col]
    logger.info(f"Loaded train data with shape {df_train.shape}")

    # Set aesthetic theme
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Target Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    counts = df_train[target_col].value_counts()
    colors = ["#2b5c8f", "#d9534f"]
    bars = ax.bar(["Non-Bankrupt (0)", "Bankrupt (1)"], counts.values, color=colors, width=0.45)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df_train)) * 100
        ax.text(bar.get_x() + bar.get_width()/2., h + 40, f"{h:,}\n({pct:.2f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Companies", fontsize=11)
    ax.set_title("Training Set Target Distribution (Severe Class Imbalance)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, len(df_train) * 1.15)
    plt.tight_layout()
    target_dist_path = FIGURES_DIR / "target_distribution.png"
    plt.savefig(target_dist_path)
    plt.close()
    logger.info(f"Saved target distribution figure to: {target_dist_path}")

    # 2. Key Financial Dimensions Selection for Comparison
    # Group informative features by financial concept
    key_features = [
        "ROA(C) before interest and depreciation before interest",
        "Net Income to Total Assets",
        "Debt ratio %",
        "Current Liability to Assets",
        "Borrowing dependency",
        "Working Capital to Total Assets",
        "Cash Flow to Total Assets",
        "Operating Profit Growth Rate",
    ]
    # Filter only those that exist in df_train
    available_key_features = [f for f in key_features if f in feature_cols]
    if len(available_key_features) < 4:
        # Fallback to top variance features
        available_key_features = df_train[feature_cols].var().nlargest(8).index.tolist()

    # Boxplot comparison
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9), dpi=300)
    axes = axes.flatten()
    for i, col in enumerate(available_key_features[:8]):
        sns.boxplot(
            x=target_col,
            y=col,
            data=df_train,
            ax=axes[i],
            palette=["#2b5c8f", "#d9534f"],
            showfliers=False,  # Hide extreme outliers for readability
        )
        # Shorten title for display
        short_title = col if len(col) <= 28 else col[:25] + "..."
        axes[i].set_title(short_title, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("0: Healthy | 1: Bankrupt", fontsize=9)
        axes[i].set_ylabel("Value", fontsize=9)

    plt.suptitle("Key Financial Ratio Distributions: Bankrupt vs Non-Bankrupt Companies", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    dist_plot_path = FIGURES_DIR / "top_feature_distributions.png"
    plt.savefig(dist_plot_path)
    plt.close()
    logger.info(f"Saved key feature distributions figure to: {dist_plot_path}")

    # 3. Correlation Analysis
    corr_matrix = df_train[feature_cols].corr()
    
    # Identify high correlation pairs (|corr| >= 0.90)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            val = corr_matrix.iloc[i, j]
            if abs(val) >= 0.90:
                high_corr_pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": float(val),
                    "abs_correlation": float(abs(val))
                })

    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(by="abs_correlation", ascending=False)
    high_corr_path = METRICS_DIR / "high_correlation_pairs.csv"
    high_corr_df.to_csv(high_corr_path, index=False)
    logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| >= 0.90). Saved to: {high_corr_path}")

    # Correlation Matrix Heatmap of representative features
    selected_for_corr = available_key_features[:10]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sub_corr = df_train[selected_for_corr].corr()
    sns.heatmap(sub_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap for Core Financial Predictors", fontsize=12, fontweight="bold", pad=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    corr_plot_path = FIGURES_DIR / "correlation_matrix.png"
    plt.savefig(corr_plot_path)
    plt.close()
    logger.info(f"Saved correlation matrix heatmap to: {corr_plot_path}")

    # 4. Generate Comprehensive EDA Summary Table
    eda_summary = []
    for col in feature_cols:
        non_bankrupt_vals = df_train[df_train[target_col] == 0][col]
        bankrupt_vals = df_train[df_train[target_col] == 1][col]
        
        eda_summary.append({
            "feature": col,
            "overall_mean": float(df_train[col].mean()),
            "overall_median": float(df_train[col].median()),
            "overall_std": float(df_train[col].std()),
            "non_bankrupt_mean": float(non_bankrupt_vals.mean()),
            "non_bankrupt_median": float(non_bankrupt_vals.median()),
            "bankrupt_mean": float(bankrupt_vals.mean()),
            "bankrupt_median": float(bankrupt_vals.median()),
            "mean_difference": float(bankrupt_vals.mean() - non_bankrupt_vals.mean()),
        })

    eda_summary_df = pd.DataFrame(eda_summary)
    summary_path = METRICS_DIR / "eda_summary.csv"
    eda_summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved EDA Summary Metrics to: {summary_path}")

    logger.info("EDA Pipeline completed successfully!")


if __name__ == "__main__":
    run_eda()
