"""Utilities for data storage management."""

import polars as pl
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.config import config


def get_raw_data_path(
    symbol: str,
    interval: str,
    format: str = "parquet",
    timestamp: Optional[datetime] = None
) -> Path:
    """
    Generates the path to save raw data.
    
    Args:
        symbol: Ticker symbol (e.g., 'AAPL')
        interval: Time interval (e.g., '1min')
        format: File format ('parquet' or 'csv')
        timestamp: Optional timestamp for filename
    
    Returns:
        Complete file path
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_{interval}_{timestamp_str}.{format}"
    return config.RAW_DATA_DIR / filename


def get_processed_data_path(
    symbol: str,
    interval: str,
    suffix: str = "",
    format: str = "parquet"
) -> Path:
    """
    Generates the path to save processed data.
    
    Args:
        symbol: Ticker symbol
        interval: Time interval
        suffix: Optional suffix to differentiate transformations
        format: File format
    
    Returns:
        Complete file path
    """
    suffix_str = f"_{suffix}" if suffix else ""
    filename = f"{symbol}_{interval}{suffix_str}.{format}"
    return config.PROCESSED_DATA_DIR / filename


def save_dataframe(
    df: pl.DataFrame,
    filepath: Path,
    format: str = "parquet"
) -> None:
    """
    Saves a Polars DataFrame to a file.

    Args:
        df: Polars DataFrame
        filepath: Destination file path
        format: File format ('parquet' or 'csv')
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Unsupported DataFrame type: {type(df)}. Expected: pl.DataFrame")
    
    if format == "parquet":
        df.write_parquet(filepath)
    elif format == "csv":
        df.write_csv(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataframe(
    filepath: Path,
    format: Optional[str] = None
) -> pl.DataFrame:
    """
    Loads a Polars DataFrame from a file.

    Args:
        filepath: File path
        format: File format (auto-detected if None)

    Returns:
        Polars DataFrame
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if format is None:
        format = filepath.suffix[1:]
    
    if format == "parquet":
        return pl.read_parquet(filepath)
    elif format == "csv":
        return pl.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def find_latest_data_file(
    symbol: str,
    interval: str,
    format: str = "parquet"
) -> Optional[Path]:
    """
    Finds the most recent data file for a given symbol and interval.
    
    Args:
        symbol: Ticker symbol
        interval: Time interval
        format: File format
    
    Returns:
        Path of the most recent file or None if no file is found
    """
    pattern = f"{symbol}_{interval}_*.{format}"
    files = list(config.RAW_DATA_DIR.glob(pattern))
    
    if not files:
        return None
    
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def data_exists(symbol: str, interval: str, format: str = "parquet") -> bool:
    """
    Checks if data already exists for a given symbol and interval.
    
    Args:
        symbol: Ticker symbol
        interval: Time interval
        format: File format
    
    Returns:
        True if data exists, False otherwise
    """
    return find_latest_data_file(symbol, interval, format) is not None

