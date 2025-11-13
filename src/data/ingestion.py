"""Data ingestion module from Alpha Vantage API."""

import time
import logging
from typing import Dict, List
import polars as pl
from alpha_vantage.timeseries import TimeSeries

from src.config import config
from src.data.storage import (
    save_dataframe,
    get_raw_data_path,
    data_exists,
    find_latest_data_file,
    load_dataframe,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Exception raised on rate limit."""

    pass


def handle_rate_limit(error: Exception, retry_count: int) -> bool:
    """
    Handles rate limit errors with exponential backoff.

    Args:
        error: The raised exception
        retry_count: Number of attempts already made

    Returns:
        True if should retry, False otherwise
    """
    error_str = str(error).lower()

    # Detect rate limit errors
    if "rate limit" in error_str or "api call frequency" in error_str:
        if retry_count < config.MAX_RETRIES:
            wait_time = config.RATE_LIMIT_SLEEP_SECONDS * (
                config.RETRY_BACKOFF_FACTOR**retry_count
            )
            logger.warning(
                f"Rate limit détecté. Attente de {wait_time:.1f} secondes "
                f"avant retry {retry_count + 1}/{config.MAX_RETRIES}"
            )
            time.sleep(wait_time)
            return True
        else:
            raise RateLimitError(
                f"Rate limit atteint après {config.MAX_RETRIES} tentatives"
            )

    return False


def normalize_datetime_column(df: pl.DataFrame, target_name: str = "datetime") -> pl.DataFrame:
    """
    Normalizes the datetime column name in a Polars DataFrame.
    
    Searches for common datetime columns (date, datetime, timestamp, etc.)
    and renames them to 'datetime' for consistency.
    
    Args:
        df: Polars DataFrame
        target_name: Target name for the datetime column (default: "datetime")
    
    Returns:
        DataFrame with normalized datetime column
    
    Raises:
        ValueError: If no datetime column is found
    """
    possible_names = ["date", "datetime", "timestamp", "time", "dt"]
    
    datetime_col = None
    for col_name in possible_names:
        if col_name in df.columns:
            if df[col_name].dtype in (pl.Datetime, pl.Date):
                datetime_col = col_name
                break
    
    if datetime_col is None:
        for col_name in df.columns:
            if df[col_name].dtype in (pl.Datetime, pl.Date):
                datetime_col = col_name
                break
    
    if datetime_col is None:
        raise ValueError(
            f"No datetime column found. Available columns: {df.columns}"
        )
    
    if datetime_col != target_name:
        df = df.rename({datetime_col: target_name})
        logger.debug(f"Column '{datetime_col}' renamed to '{target_name}'")
    
    return df


def fetch_intraday_data(
    symbol: str,
    interval: str = "1min",
    outputsize: str = "full",
    save: bool = True,
    use_cache: bool = True,
) -> pl.DataFrame:
    """
    Fetches intraday data from Alpha Vantage API.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL')
        interval: Time interval ('1min', '5min', '15min', '30min', '60min')
        outputsize: 'compact' (last 100) or 'full' (complete data)
        save: If True, saves data to disk
        use_cache: If True, uses cached data if available

    Returns:
        Polars DataFrame with OHLCV data
    """
    if use_cache and data_exists(symbol, interval):
        logger.info(f"Données en cache trouvées pour {symbol} ({interval})")
        cached_file = find_latest_data_file(symbol, interval)
        if cached_file:
            data = load_dataframe(cached_file)
            data = normalize_datetime_column(data)
            return data

    logger.info(f"Récupération des données pour {symbol} ({interval})...")

    ts = TimeSeries(
        key=config.ALPHA_VANTAGE_API_KEY, output_format="pandas", indexing_type="date"
    )

    retry_count = 0
    while retry_count < config.MAX_RETRIES:
        try:
            data_pd, meta_data = ts.get_intraday(
                symbol=symbol, interval=interval, outputsize=outputsize
            )

            if data_pd is None or data_pd.empty:
                raise ValueError(f"Aucune donnée reçue pour {symbol}")

            # Alpha Vantage uses prefixed column names: '1. open', '2. high', etc.
            column_mapping = {}
            for col in data_pd.columns:
                if "open" in col.lower():
                    column_mapping[col] = "open"
                elif "high" in col.lower():
                    column_mapping[col] = "high"
                elif "low" in col.lower():
                    column_mapping[col] = "low"
                elif "close" in col.lower():
                    column_mapping[col] = "close"
                elif "volume" in col.lower():
                    column_mapping[col] = "volume"

            data_pd = data_pd.rename(columns=column_mapping)

            import pandas as pd
            if not isinstance(data_pd.index, pd.DatetimeIndex):
                data_pd.index = pd.to_datetime(data_pd.index)

            data_pd_reset = data_pd.reset_index()
            datetime_col_name = "datetime"
            data_pd_reset = data_pd_reset.rename(columns={data_pd_reset.columns[0]: datetime_col_name})
            
            data = pl.from_pandas(data_pd_reset)
            
            if datetime_col_name in data.columns:
                data = data.with_columns(
                    pl.col(datetime_col_name).cast(pl.Datetime)
                )
            
            data = normalize_datetime_column(data, target_name="datetime")
            
            data = data.sort(datetime_col_name)

            logger.info(
                f"Données récupérées: {data.height} lignes pour {symbol} "
                f"({data[datetime_col_name].min()} à {data[datetime_col_name].max()})"
            )

            if save:
                filepath = get_raw_data_path(symbol, interval)
                save_dataframe(data, filepath, format=config.DEFAULT_STORAGE_FORMAT)
                logger.info(f"Données sauvegardées: {filepath}")

            return data

        except (ValueError, Exception) as e:
            if handle_rate_limit(e, retry_count):
                retry_count += 1
                continue
            else:
                logger.error(f"Erreur lors de la récupération des données: {e}")
                raise

    raise Exception(f"Échec après {config.MAX_RETRIES} tentatives pour {symbol}")


def fetch_multiple_symbols(
    symbols: List[str],
    interval: str = "1min",
    outputsize: str = "full",
    save: bool = True,
    use_cache: bool = True,
) -> Dict[str, pl.DataFrame]:
    """
    Fetches data for multiple symbols with rate limiting management.

    Args:
        symbols: List of symbols to fetch
        interval: Time interval
        outputsize: 'compact' or 'full'
        save: If True, saves data
        use_cache: If True, uses cache

    Returns:
        Dictionary {symbol: Polars DataFrame}
    """
    results = {}

    for i, symbol in enumerate(symbols):
        try:
            if i > 0:
                wait_time = config.RATE_LIMIT_SLEEP_SECONDS
                logger.info(
                    f"Attente de {wait_time} secondes avant le prochain appel "
                    f"({i + 1}/{len(symbols)})"
                )
                time.sleep(wait_time)

            data = fetch_intraday_data(
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                save=save,
                use_cache=use_cache,
            )
            results[symbol] = data

        except Exception as e:
            logger.error(f"Erreur pour {symbol}: {e}")
            continue

    logger.info(
        f"Récupération terminée: {len(results)}/{len(symbols)} symboles réussis"
    )
    return results


def fetch_daily_data(
    symbol: str, outputsize: str = "full", save: bool = True, use_cache: bool = True
) -> pl.DataFrame:
    """
    Fetches daily data from Alpha Vantage API.

    Args:
        symbol: Ticker symbol
        outputsize: 'compact' or 'full'
        save: If True, saves data
        use_cache: If True, uses cache

    Returns:
        Polars DataFrame with daily data
    """
    if use_cache and data_exists(symbol, "daily"):
        logger.info(f"Données daily en cache trouvées pour {symbol}")
        cached_file = find_latest_data_file(symbol, "daily")
        if cached_file:
            data = load_dataframe(cached_file)
            data = normalize_datetime_column(data)
            return data

    logger.info(f"Récupération des données daily pour {symbol}...")

    ts = TimeSeries(
        key=config.ALPHA_VANTAGE_API_KEY, output_format="pandas", indexing_type="date"
    )

    retry_count = 0
    while retry_count < config.MAX_RETRIES:
        try:
            data_pd, meta_data = ts.get_daily(symbol=symbol, outputsize=outputsize)

            if data_pd is None or data_pd.empty:
                raise ValueError(f"Aucune donnée reçue pour {symbol}")

            column_mapping = {}
            for col in data_pd.columns:
                if "open" in col.lower():
                    column_mapping[col] = "open"
                elif "high" in col.lower():
                    column_mapping[col] = "high"
                elif "low" in col.lower():
                    column_mapping[col] = "low"
                elif "close" in col.lower():
                    column_mapping[col] = "close"
                elif "volume" in col.lower():
                    column_mapping[col] = "volume"

            data_pd = data_pd.rename(columns=column_mapping)

            import pandas as pd
            if not isinstance(data_pd.index, pd.DatetimeIndex):
                data_pd.index = pd.to_datetime(data_pd.index)

            data_pd_reset = data_pd.reset_index()
            datetime_col_name = "datetime"
            data_pd_reset = data_pd_reset.rename(columns={data_pd_reset.columns[0]: datetime_col_name})
            
            data = pl.from_pandas(data_pd_reset)
            
            if datetime_col_name in data.columns:
                data = data.with_columns(
                    pl.col(datetime_col_name).cast(pl.Datetime)
                )
            
            data = normalize_datetime_column(data, target_name="datetime")
            
            data = data.sort(datetime_col_name)

            if save:
                filepath = get_raw_data_path(symbol, "daily")
                save_dataframe(data, filepath, format=config.DEFAULT_STORAGE_FORMAT)
                logger.info(f"Données daily sauvegardées: {filepath}")

            return data

        except (ValueError, Exception) as e:
            if handle_rate_limit(e, retry_count):
                retry_count += 1
                continue
            else:
                logger.error(f"Erreur lors de la récupération des données daily: {e}")
                raise

    raise Exception(f"Échec après {config.MAX_RETRIES} tentatives pour {symbol}")
