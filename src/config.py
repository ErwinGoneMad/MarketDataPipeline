"""Centralized configuration for the market data pipeline."""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralized configuration for the application."""
    
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHAVANTAGE_API_KEY", "")
    ALPHA_VANTAGE_BASE_URL: str = "https://www.alphavantage.co/query"
    
    RATE_LIMIT_CALLS_PER_MINUTE: int = 5
    RATE_LIMIT_CALLS_PER_DAY: int = 500
    RATE_LIMIT_SLEEP_SECONDS: float = 12.0  # Minimum 12 seconds between calls (60/5)
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_FACTOR: float = 2.0
    
    DEFAULT_SYMBOLS: List[str] = os.getenv(
        "DEFAULT_SYMBOLS", "AAPL,MSFT,TSLA"
    ).split(",")
    DEFAULT_INTERVAL: str = os.getenv("DEFAULT_INTERVAL", "1min")
    
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    
    DEFAULT_STORAGE_FORMAT: str = "parquet"
    
    MODEL_DIR: Path = BASE_DIR / "models"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    MOMENTUM_PERIODS: List[int] = [5, 10, 20, 50]
    VOLATILITY_WINDOWS: List[int] = [10, 20, 60]
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    
    PREDICTION_HORIZON: int = 1
    TARGET_TYPE: str = "classification"
    
    XGB_N_ESTIMATORS: int = 100
    XGB_MAX_DEPTH: int = 6
    XGB_LEARNING_RATE: float = 0.1
    XGB_SUBSAMPLE: float = 0.8
    XGB_COLSAMPLE_BYTREE: float = 0.8
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Creates necessary directories if they don't exist."""
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate(cls) -> None:
        """Validates the configuration."""
        if not cls.ALPHA_VANTAGE_API_KEY:
            raise ValueError(
                "ALPHAVANTAGE_API_KEY is not defined. "
                "Please define it in the .env file"
            )
        
        if not cls.DEFAULT_SYMBOLS:
            raise ValueError("No default symbols are defined")
        
        cls.ensure_directories()


config = Config()

