"""XGBoost model training."""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.config import config

logger = logging.getLogger(__name__)


def prepare_dataset(
    df: pl.DataFrame,
    feature_columns: Optional[List[str]] = None,
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
    datetime_col: str = "datetime",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepares the dataset for training.
    
    Args:
        df: DataFrame with features and target
        feature_columns: List of columns to use as features (None = all except target and datetime)
        target_col: Name of the target column
        test_size: Proportion of dataset for testing
        random_state: Seed for reproducibility
        datetime_col: Name of the datetime column
    
    Returns:
        Tuple (X_train, X_test, y_train, y_test, feature_names)
    """
    if feature_columns is None:
        exclude_cols = {target_col, datetime_col, "open", "high", "low", "close", "volume"}
        feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    df_features = df.select(feature_columns + [target_col])
    
    df_features = df_features.drop_nulls()
    
    if df_features.height == 0:
        raise ValueError("No valid data after removing NaN values")
    
    X = df_features.select(feature_columns).to_numpy()
    y = df_features.select(target_col).to_numpy().ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    logger.info(
        f"Dataset préparé: {X_train.shape[0]} train, {X_test.shape[0]} test, "
        f"{len(feature_columns)} features"
    )
    
    return X_train, X_test, y_train, y_test, feature_columns


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "classification",
    hyperparameters: Optional[dict] = None,
    save_path: Optional[Path] = None,
) -> xgb.XGBModel:
    """
    Trains an XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        model_type: Model type ('classification' or 'regression')
        hyperparameters: Dictionary of XGBoost hyperparameters
        save_path: Path to save the model
    
    Returns:
        Trained XGBoost model
    """
    default_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    
    if hyperparameters:
        default_params.update(hyperparameters)
    
    if model_type == "classification":
        default_params["objective"] = "binary:logistic"
        default_params["eval_metric"] = "logloss"
    elif model_type == "regression":
        default_params["objective"] = "reg:squarederror"
        default_params["eval_metric"] = "rmse"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.info(f"Entraînement du modèle {model_type} avec {X_train.shape[0]} échantillons...")
    
    model = xgb.XGBClassifier(**default_params) if model_type == "classification" else xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    
    logger.info("Modèle entraîné avec succès")
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Modèle sauvegardé: {save_path}")
    
    return model

