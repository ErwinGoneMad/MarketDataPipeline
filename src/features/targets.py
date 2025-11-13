"""Target generation for machine learning."""

import polars as pl
from typing import Optional


def generate_targets(
    df: pl.DataFrame,
    horizon: int = 1,
    target_type: str = "classification",
    datetime_col: str = "datetime",
    close_col: str = "close",
) -> pl.DataFrame:
    """
    Generates targets for prediction.
    
    Args:
        df: DataFrame with OHLCV data and features
        horizon: Number of periods ahead to predict (default: 1)
        target_type: Type of target ('classification' or 'regression')
        datetime_col: Name of the datetime column
        close_col: Name of the close column
    
    Returns:
        DataFrame with 'target' column added
    """
    if close_col not in df.columns:
        raise ValueError(f"The '{close_col}' column is required")
    
    future_close = pl.col(close_col).shift(-horizon)
    forward_return = (future_close - pl.col(close_col)) / pl.col(close_col)
    
    if target_type == "classification":
        target = (forward_return > 0).cast(pl.Int8)
        target = target.alias("target")
    elif target_type == "regression":
        target = forward_return.alias("target")
    else:
        raise ValueError(f"Unsupported target type: {target_type}")
    
    df = df.with_columns([target])
    
    df = df.filter(pl.col("target").is_not_null())
    
    return df

