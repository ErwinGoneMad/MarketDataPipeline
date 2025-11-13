"""Main script to orchestrate the complete ML pipeline."""

import logging
from pathlib import Path
import polars as pl

from src.config import config
from src.data.ingestion import fetch_multiple_symbols
from src.features.engineering import compute_all_features
from src.features.targets import generate_targets
from src.ml.training import prepare_dataset, train_model
from src.ml.evaluation import evaluate_model, plot_feature_importance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Orchestrates the complete ML pipeline."""
    logger.info("="*60)
    logger.info("HIGH-VOLUME MARKET FEATURE PIPELINE - POLARS + XGBOOST")
    logger.info("="*60)
    
    try:
        config.validate()
        logger.info("Configuration validée")
    except ValueError as e:
        logger.error(f"Erreur de configuration: {e}")
        return
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: INGESTION DES DONNÉES")
    logger.info("="*60)
    
    symbols = config.DEFAULT_SYMBOLS
    interval = config.DEFAULT_INTERVAL
    
    logger.info(f"Récupération des données pour {symbols} ({interval})...")
    
    try:
        data_dict = fetch_multiple_symbols(
            symbols=symbols,
            interval=interval,
            outputsize='compact',  # Use 'compact' to avoid too many API calls
            save=True,
            use_cache=True
        )
        logger.info(f"Données récupérées pour {len(data_dict)} symboles")
    except Exception as e:
        logger.error(f"Erreur lors de l'ingestion: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: FEATURE ENGINEERING")
    logger.info("="*60)
    
    features_dict = {}
    for symbol, df in data_dict.items():
        logger.info(f"Calcul des features pour {symbol}...")
        try:
            df_features = compute_all_features(
                df,
                datetime_col="datetime",
                momentum_periods=config.MOMENTUM_PERIODS,
                volatility_windows=config.VOLATILITY_WINDOWS,
            )
            
            df_features = generate_targets(
                df_features,
                horizon=config.PREDICTION_HORIZON,
                target_type=config.TARGET_TYPE,
                datetime_col="datetime",
            )
            
            features_dict[symbol] = df_features
            logger.info(f"  {symbol}: {df_features.height} lignes, {len(df_features.columns)} colonnes")
            
        except Exception as e:
            logger.error(f"Erreur lors du feature engineering pour {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not features_dict:
        logger.error("Aucune donnée avec features générées")
        return
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 3: PRÉPARATION DU DATASET ML")
    logger.info("="*60)
    
    logger.info("Combinaison des données de tous les symboles...")
    # Ensure all DataFrames have columns in the same order and same types
    if features_dict:
        first_df = next(iter(features_dict.values()))
        column_order = first_df.columns
        schema = first_df.schema
        
        normalized_dfs = []
        for symbol, df in features_dict.items():
            df_reordered = df.select(column_order)
            
            cast_expressions = []
            for col_name in column_order:
                expected_dtype = schema[col_name]
                actual_dtype = df_reordered[col_name].dtype
                
                if actual_dtype != expected_dtype:
                    # Normalize datetime to match the first DataFrame's type
                    if isinstance(actual_dtype, pl.Datetime) and isinstance(expected_dtype, pl.Datetime):
                        cast_expressions.append(pl.col(col_name).cast(expected_dtype).alias(col_name))
                    elif actual_dtype != expected_dtype:
                        cast_expressions.append(pl.col(col_name).cast(expected_dtype).alias(col_name))
            
            if cast_expressions:
                df_reordered = df_reordered.with_columns(cast_expressions)
            
            normalized_dfs.append(df_reordered)
        
        all_data = pl.concat(normalized_dfs)
    else:
        raise ValueError("Aucune donnée à combiner")
    logger.info(f"Dataset combiné: {all_data.height} lignes, {len(all_data.columns)} colonnes")
    
    try:
        X_train, X_test, y_train, y_test, feature_names = prepare_dataset(
            all_data,
            feature_columns=None,  # Use all features except target and datetime
            target_col="target",
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
        )
    except Exception as e:
        logger.error(f"Erreur lors de la préparation du dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 4: ENTRAÎNEMENT DU MODÈLE")
    logger.info("="*60)
    
    try:
        hyperparameters = {
            "n_estimators": config.XGB_N_ESTIMATORS,
            "max_depth": config.XGB_MAX_DEPTH,
            "learning_rate": config.XGB_LEARNING_RATE,
            "subsample": config.XGB_SUBSAMPLE,
            "colsample_bytree": config.XGB_COLSAMPLE_BYTREE,
        }
        
        model_path = config.MODEL_DIR / f"xgboost_{config.TARGET_TYPE}_{config.PREDICTION_HORIZON}.pkl"
        
        model = train_model(
            X_train,
            y_train,
            model_type=config.TARGET_TYPE,
            hyperparameters=hyperparameters,
            save_path=model_path,
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 5: ÉVALUATION DU MODÈLE")
    logger.info("="*60)
    
    try:
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            model_type=config.TARGET_TYPE,
        )
        
        importance_path = config.PROCESSED_DATA_DIR / "feature_importance.png"
        plot_feature_importance(
            model,
            feature_names,
            top_n=20,
            save_path=str(importance_path),
        )
        
        logger.info(f"Métriques sauvegardées et graphiques générés")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE TERMINÉ AVEC SUCCÈS")
    logger.info("="*60)
    logger.info(f"Modèle sauvegardé: {model_path}")
    logger.info(f"Résultats disponibles dans: {config.PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
