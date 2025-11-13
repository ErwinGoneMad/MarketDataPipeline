"""Model evaluation."""

import logging
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import xgboost as xgb

logger = logging.getLogger(__name__)


def evaluate_model(
    model: xgb.XGBModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "classification",
) -> dict:
    """
    Evaluates a model on test data.

    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test targets
        model_type: Model type ('classification' or 'regression')

    Returns:
        Dictionary with metrics
    """
    y_pred = model.predict(X_test)

    if model_type == "classification":
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        logger.info("Métriques de classification:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-score: {metrics['f1']:.4f}")

        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))

    elif model_type == "regression":
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

        logger.info("Métriques de régression:")
        logger.info(f"  MSE: {metrics['mse']:.6f}")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        logger.info(f"  MAE: {metrics['mae']:.6f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return metrics


def plot_feature_importance(
    model: xgb.XGBModel,
    feature_names: List[str],
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots feature importance.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of features to display
        save_path: Path to save the plot
    """
    importance = model.feature_importances_

    df_importance = (
        pl.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
            }
        )
        .sort("importance", descending=True)
        .head(top_n)
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=df_importance.to_pandas(),
        x="importance",
        y="feature",
        palette="viridis",
    )
    plt.title(f"Top {top_n} Features par Importance")
    plt.xlabel("Importance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Graphique sauvegardé: {save_path}")
    else:
        plt.show()

    plt.close()
