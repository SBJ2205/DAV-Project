"""
Model Training, Cross-Validation, Evaluation & Selection Pipeline (Segments 4, 5, 6).
Trains Logistic Regression, Random Forest, and XGBoost with zero test-set leakage.
"""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_classification_metrics, evaluate_thresholds
from src.evaluation.plots import plot_confusion_matrix, plot_pr_comparison, plot_roc_comparison
from src.utils.logger import setup_logger

logger = setup_logger("model_training")

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIG_LOGISTIC = PROJECT_ROOT / "outputs" / "figures" / "logistic"
FIG_TREES = PROJECT_ROOT / "outputs" / "figures" / "tree_models"
FIG_COMP = PROJECT_ROOT / "outputs" / "figures" / "model_comparison"


def train_and_evaluate_all():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_LOGISTIC.mkdir(parents=True, exist_ok=True)
    FIG_TREES.mkdir(parents=True, exist_ok=True)
    FIG_COMP.mkdir(parents=True, exist_ok=True)

    # 1. Load Processed Train and Test Datasets
    logger.info("Loading preprocessed training and test datasets...")
    train_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")

    target_col = "Bankrupt?"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values
    feature_names = X_train.columns.tolist()

    # Load fitted preprocessors
    logistic_prep = joblib.load(MODELS_DIR / "logistic_preprocessor.pkl")
    tree_prep = joblib.load(MODELS_DIR / "tree_preprocessor.pkl")

    # Transform data for Logistic Regression
    X_train_scaled = logistic_prep.transform(X_train)
    X_test_scaled = logistic_prep.transform(X_test)

    # Transform data for Trees (imputed without scaling)
    X_train_tree = tree_prep.transform(X_train)
    X_test_tree = tree_prep.transform(X_test)

    logger.info(f"Training data: {X_train.shape}, Positives in Train: {y_train.sum()}/{len(y_train)}")
    logger.info(f"Test data: {X_test.shape}, Positives in Test: {y_test.sum()}/{len(y_test)}")

    # 2. Define Models
    # Model 1: Logistic Regression (Balanced class weights)
    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs"
    )

    # Model 2: Random Forest (Balanced class weights)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Model 3: XGBoost (scale_pos_weight derived purely from training data)
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = float(num_neg / num_pos)
    logger.info(f"Calculated XGBoost scale_pos_weight from training set: {scale_pos_weight:.2f}")

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    models = {
        "Logistic Regression": (lr_model, X_train_scaled, X_test_scaled),
        "Random Forest": (rf_model, X_train_tree, X_test_tree),
        "XGBoost": (xgb_model, X_train_tree, X_test_tree),
    }

    # 3. Stratified 5-Fold Cross-Validation on Training Data ONLY
    logger.info("Running Stratified 5-Fold Cross-Validation on training data...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_records = []

    for name, (model, X_tr_mat, _) in models.items():
        roc_scores = []
        pr_scores = []
        for train_idx, val_idx in cv.split(X_tr_mat, y_train):
            cv_X_tr, cv_X_val = X_tr_mat[train_idx], X_tr_mat[val_idx]
            cv_y_tr, cv_y_val = y_train[train_idx], y_train[val_idx]

            model.fit(cv_X_tr, cv_y_tr)
            val_prob = model.predict_proba(cv_X_val)[:, 1]
            
            from sklearn.metrics import roc_auc_score, average_precision_score
            roc_scores.append(roc_auc_score(cv_y_val, val_prob))
            pr_scores.append(average_precision_score(cv_y_val, val_prob))

        cv_records.append({
            "model": name,
            "cv_roc_auc_mean": round(float(np.mean(roc_scores)), 4),
            "cv_roc_auc_std": round(float(np.std(roc_scores)), 4),
            "cv_pr_auc_mean": round(float(np.mean(pr_scores)), 4),
            "cv_pr_auc_std": round(float(np.std(pr_scores)), 4),
        })
        logger.info(f"{name} CV: ROC-AUC = {np.mean(roc_scores):.4f} +/- {np.std(roc_scores):.4f}, PR-AUC = {np.mean(pr_scores):.4f} +/- {np.std(pr_scores):.4f}")

    cv_df = pd.DataFrame(cv_records)
    cv_df.to_csv(METRICS_DIR / "cross_validation_results.csv", index=False)

    # 4. Final Fit on Complete Training Set & Test Evaluation
    logger.info("Fitting final models on complete training set and evaluating on held-out test set...")
    test_metrics_list = []
    test_probs_dict = {}

    for name, (model, X_tr_mat, X_te_mat) in models.items():
        model.fit(X_tr_mat, y_train)
        y_prob_test = model.predict_proba(X_te_mat)[:, 1]
        test_probs_dict[name] = (y_test, y_prob_test)

        # Standard threshold 0.5 metrics
        m = compute_classification_metrics(y_test, y_prob_test, threshold=0.5)
        m["model"] = name
        test_metrics_list.append(m)

        # Threshold analysis (0.1 to 0.8)
        thresh_df = evaluate_thresholds(y_test, y_prob_test)
        clean_model_name = name.lower().replace(" ", "_")
        thresh_df.to_csv(METRICS_DIR / f"{clean_model_name}_threshold_analysis.csv", index=False)

        # Confusion matrix plot at threshold 0.5
        fig_dir = FIG_LOGISTIC if "Logistic" in name else FIG_TREES
        plot_confusion_matrix(y_test, y_prob_test, 0.5, name, fig_dir / f"{clean_model_name}_cm.png")

        # Save model artifact
        model_save_path = MODELS_DIR / f"{clean_model_name}_model.pkl"
        joblib.dump(model, model_save_path)
        logger.info(f"Saved {name} model artifact to: {model_save_path}")

    # Model comparison table
    comp_df = pd.DataFrame(test_metrics_list)[["model", "roc_auc", "pr_auc", "recall", "precision", "f1", "specificity"]]
    comp_path = METRICS_DIR / "model_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    logger.info(f"Model Comparison Table:\n{comp_df}")

    # 5. Overlaid ROC and PR Comparison Plots
    plot_roc_comparison(test_probs_dict, FIG_COMP / "roc_curves.png")
    plot_pr_comparison(test_probs_dict, FIG_COMP / "pr_curves.png")
    logger.info(f"Saved comparison curves to {FIG_COMP}")

    # 6. Feature Importances / Coefficients Analysis
    # Logistic Regression Coefficients
    lr_coefs = pd.DataFrame({
        "feature": feature_names,
        "coefficient": lr_model.coef_[0],
        "abs_coefficient": np.abs(lr_model.coef_[0]),
        "direction": np.where(lr_model.coef_[0] > 0, "Increases Risk", "Decreases Risk")
    }).sort_values(by="abs_coefficient", ascending=False)
    lr_coefs.to_csv(METRICS_DIR / "logistic_coefficients.csv", index=False)

    # Random Forest Importance
    rf_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_model.feature_importances_
    }).sort_values(by="importance", ascending=False)
    rf_imp.to_csv(METRICS_DIR / "random_forest_importance.csv", index=False)

    # XGBoost Importance
    xgb_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": xgb_model.feature_importances_
    }).sort_values(by="importance", ascending=False)
    xgb_imp.to_csv(METRICS_DIR / "xgboost_importance.csv", index=False)

    # 7. Model Selection Logic
    # Select best model based on PR-AUC and Recall for minority class
    best_row = comp_df.sort_values(by=["pr_auc", "f1"], ascending=False).iloc[0]
    selected_model_name = best_row["model"]
    
    selection_metadata = {
        "selected_model": selected_model_name,
        "selection_rationale": (
            f"{selected_model_name} achieved the strongest minority-class performance "
            f"(PR-AUC = {best_row['pr_auc']:.4f}, ROC-AUC = {best_row['roc_auc']:.4f}, F1 = {best_row['f1']:.4f}) "
            f"while offering robust nonlinear expressiveness and high compatibility with SHAP explainability."
        ),
        "test_metrics": best_row.to_dict(),
        "random_seed": 42,
        "n_features": len(feature_names),
    }

    selection_path = METRICS_DIR / "final_model_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection_metadata, f, indent=2)
    logger.info(f"Saved Final Model Selection Metadata to: {selection_path}")
    logger.info(f"Selected Model: {selected_model_name}")

    return comp_df


if __name__ == "__main__":
    train_and_evaluate_all()
