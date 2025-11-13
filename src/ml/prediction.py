"""Predictions with saved models."""

import logging
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl
import xgboost as xgb

logger = logging.getLogger(__name__)


def load_model(model_path: Path) -> xgb.XGBModel:
    """
    Loads a saved XGBoost model.
    
    Args:
        model_path: Path to the model file
    
    Returns:
        Loaded XGBoost model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    logger.info(f"Modèle chargé: {model_path}")
    return model


def predict(
    model: xgb.XGBModel,
    df: pl.DataFrame,
    feature_columns: list,
    return_proba: bool = False,
) -> pl.DataFrame:
    """
    Makes predictions on a DataFrame.
    
    Args:
        model: XGBoost model
        df: DataFrame with features
        feature_columns: List of feature columns to use
        return_proba: If True, also returns probabilities (classification only)
    
    Returns:
        DataFrame with 'prediction' column (and 'probability' if return_proba=True)
    """
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    X = df.select(feature_columns).to_numpy()
    
    predictions = model.predict(X)
    
    result = df.with_columns([
        pl.Series("prediction", predictions),
    ])
    
    if return_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            result = result.with_columns([
                pl.Series("probability", proba[:, 1]),
            ])
    
    return result

