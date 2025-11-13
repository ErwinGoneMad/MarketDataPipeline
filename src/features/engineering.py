"""Feature engineering with Polars."""

import polars as pl
from typing import List, Optional


def compute_technical_features(
    df: pl.DataFrame,
    datetime_col: str = "datetime",
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pl.DataFrame:
    """
    Computes technical features: RSI, MACD, Bollinger Bands.
    
    Args:
        df: DataFrame with OHLCV columns
        datetime_col: Name of the datetime column
        rsi_period: Period for RSI
        macd_fast: Fast period for MACD
        macd_slow: Slow period for MACD
        macd_signal: Signal period for MACD
        bb_period: Period for Bollinger Bands
        bb_std: Number of standard deviations for Bollinger Bands
    
    Returns:
        DataFrame with technical features added
    """
    if "close" not in df.columns:
        raise ValueError("The 'close' column is required")
    
    expressions = []
    
    delta = pl.col("close").diff()
    gain = (delta > 0).cast(pl.Float64) * delta
    loss = (delta < 0).cast(pl.Float64) * (-delta)
    
    avg_gain = gain.ewm_mean(span=rsi_period, adjust=False)
    avg_loss = loss.ewm_mean(span=rsi_period, adjust=False)
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    expressions.append(rsi.alias("rsi"))
    
    ema_fast = pl.col("close").ewm_mean(span=macd_fast)
    ema_slow = pl.col("close").ewm_mean(span=macd_slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm_mean(span=macd_signal)
    histogram = macd_line - signal_line
    
    expressions.append(macd_line.alias("macd"))
    expressions.append(signal_line.alias("macd_signal"))
    expressions.append(histogram.alias("macd_histogram"))
    
    sma = pl.col("close").rolling_mean(window_size=bb_period)
    std = pl.col("close").rolling_std(window_size=bb_period)
    upper_band = sma + (std * bb_std)
    lower_band = sma - (std * bb_std)
    bb_width = (upper_band - lower_band) / sma
    
    expressions.append(sma.alias("bb_middle"))
    expressions.append(upper_band.alias("bb_upper"))
    expressions.append(lower_band.alias("bb_lower"))
    expressions.append(bb_width.alias("bb_width"))
    
    return df.with_columns(expressions)


def compute_momentum_features(
    df: pl.DataFrame,
    periods: List[int] = [5, 10, 20, 50],
) -> pl.DataFrame:
    """
    Computes momentum features.
    
    Args:
        df: DataFrame with 'close' column
        periods: List of periods to calculate returns
    
    Returns:
        DataFrame with momentum features added
    """
    if "close" not in df.columns:
        raise ValueError("The 'close' column is required")
    
    expressions = []
    
    for period in periods:
        # Manual calculation because pct_change() doesn't accept periods parameter
        close_shifted = pl.col("close").shift(period)
        expressions.append(
            ((pl.col("close") - close_shifted) / (close_shifted + 1e-10)).alias(f"return_{period}")
        )
        
        expressions.append(
            (pl.col("close") - pl.col("close").shift(period)).alias(f"momentum_{period}")
        )
        
        expressions.append(
            ((pl.col("close") - close_shifted) / 
             (close_shifted + 1e-10) * 100).alias(f"roc_{period}")
        )
        
        expressions.append(
            pl.col("close").rolling_mean(window_size=period).alias(f"sma_{period}")
        )
        
        sma = pl.col("close").rolling_mean(window_size=period)
        expressions.append(
            ((pl.col("close") - sma) / (sma + 1e-10) * 100).alias(f"distance_sma_{period}")
        )
    
    return df.with_columns(expressions)


def compute_volatility_features(
    df: pl.DataFrame,
    windows: List[int] = [10, 20, 60],
) -> pl.DataFrame:
    """
    Computes volatility features.
    
    Args:
        df: DataFrame with 'close' column
        windows: Windows for volatility calculation
    
    Returns:
        DataFrame with volatility features added
    """
    if "close" not in df.columns:
        raise ValueError("The 'close' column is required")
    
    expressions = []
    
    returns = pl.col("close").pct_change()
    expressions.append(returns.alias("returns"))
    
    for window in windows:
        expressions.append(
            returns.rolling_std(window_size=window).alias(f"volatility_{window}")
        )
        
        expressions.append(
            (returns.pow(2).rolling_sum(window_size=window)).alias(f"realized_vol_{window}")
        )
        
        if "high" in df.columns and "low" in df.columns:
            hl_range = (pl.col("high") - pl.col("low")) / pl.col("close")
            expressions.append(
                hl_range.rolling_mean(window_size=window).alias(f"hl_range_{window}")
            )
    
    return df.with_columns(expressions)


def compute_temporal_features(
    df: pl.DataFrame,
    datetime_col: str = "datetime",
) -> pl.DataFrame:
    """
    Computes temporal features.
    
    Args:
        df: DataFrame with datetime column
        datetime_col: Name of the datetime column
    
    Returns:
        DataFrame with temporal features added
    """
    if datetime_col not in df.columns:
        raise ValueError(f"The '{datetime_col}' column is required")
    
    df = df.with_columns([
        pl.col(datetime_col).dt.hour().alias("hour"),
        pl.col(datetime_col).dt.minute().alias("minute"),
        pl.col(datetime_col).dt.weekday().alias("day_of_week"),
        pl.col(datetime_col).dt.day().alias("day"),
        pl.col(datetime_col).dt.month().alias("month"),
    ])
    
    # Cyclic encoding for temporal features (hour, day_of_week, month)
    df = df.with_columns([
        (pl.col("hour") * 2 * 3.14159 / 24).sin().alias("hour_sin"),
        (pl.col("hour") * 2 * 3.14159 / 24).cos().alias("hour_cos"),
        (pl.col("day_of_week") * 2 * 3.14159 / 7).sin().alias("day_of_week_sin"),
        (pl.col("day_of_week") * 2 * 3.14159 / 7).cos().alias("day_of_week_cos"),
        (pl.col("month") * 2 * 3.14159 / 12).sin().alias("month_sin"),
        (pl.col("month") * 2 * 3.14159 / 12).cos().alias("month_cos"),
    ])
    
    return df


def compute_all_features(
    df: pl.DataFrame,
    datetime_col: str = "datetime",
    momentum_periods: Optional[List[int]] = None,
    volatility_windows: Optional[List[int]] = None,
) -> pl.DataFrame:
    """
    Computes all features.
    
    Args:
        df: DataFrame with OHLCV data
        datetime_col: Name of the datetime column
        momentum_periods: Periods for momentum (default: [5, 10, 20, 50])
        volatility_windows: Windows for volatility (default: [10, 20, 60])
    
    Returns:
        DataFrame with all features
    """
    if momentum_periods is None:
        momentum_periods = [5, 10, 20, 50]
    if volatility_windows is None:
        volatility_windows = [10, 20, 60]
    
    df = compute_technical_features(df, datetime_col=datetime_col)
    df = compute_momentum_features(df, periods=momentum_periods)
    df = compute_volatility_features(df, windows=volatility_windows)
    df = compute_temporal_features(df, datetime_col=datetime_col)
    
    return df

