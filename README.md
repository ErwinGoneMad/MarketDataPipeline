# High-Volume Market Feature Pipeline: Polars + XGBoost

Un pipeline robuste de données de marché qui utilise Polars pour ingérer et traiter de grandes quantités de données intraday, génère des features prédictives à grande échelle, et entraîne un modèle XGBoost pour prédire les retours à court terme (direction up/down).

## Objectifs

- Ingestion efficace de données intraday depuis Alpha Vantage avec Polars
- Feature engineering à grande échelle avec Polars (RSI, MACD, Bollinger Bands, momentum, volatilité)
- Entraînement de modèles XGBoost pour prédiction de direction (classification) ou valeur (régression)
- Pipeline complète et robuste pour le machine learning sur données de marché

## Installation

Ce projet utilise `uv` pour la gestion des dépendances.

```bash
# Installer uv si ce n'est pas déjà fait
curl -LsSf https://astral.sh/uv/install.sh | sh

# Installer les dépendances
uv pip install -e .

# Ou avec uv sync (si vous utilisez uv.lock)
uv sync
```

## Configuration

1. Copiez `.env.example` vers `.env` et ajoutez votre clé API Alpha Vantage:
```bash
cp .env.example .env
```

2. Éditez `.env` et ajoutez votre clé API:
```
ALPHAVANTAGE_API_KEY=your_api_key_here
```

Optionnellement, vous pouvez configurer les symboles et l'intervalle:
```
DEFAULT_SYMBOLS=AAPL,MSFT,TSLA
DEFAULT_INTERVAL=1min
```

## Structure du projet

```
src/
  config.py              # Configuration centralisée (ML, features, etc.)
  main.py                # Script principal orchestrant la pipeline
  data/
    ingestion.py         # Ingestion Alpha Vantage (Polars uniquement)
    storage.py           # Gestion du stockage Parquet (Polars)
  features/
    engineering.py       # Feature engineering (RSI, MACD, momentum, etc.)
    targets.py           # Génération des targets (up/down ou retour)
  ml/
    training.py          # Entraînement XGBoost
    evaluation.py        # Métriques et évaluation
    prediction.py        # Prédictions sur nouvelles données
data/
  raw/                   # Données brutes
  processed/             # Données transformées et résultats
models/                  # Modèles XGBoost sauvegardés
notebooks/
  benchmark_presentation.ipynb  # Notebook de visualisation (à transformer)
```

## Utilisation

### Exécuter le pipeline complet

```bash
python -m src.main
```

Le pipeline exécute automatiquement :
1. **Ingestion** : Récupération des données intraday depuis Alpha Vantage
2. **Feature Engineering** : Calcul de toutes les features techniques et temporelles
3. **Target Generation** : Génération des targets (classification up/down par défaut)
4. **Training** : Entraînement d'un modèle XGBoost
5. **Evaluation** : Métriques et visualisation de l'importance des features

### Utiliser les modules individuellement

```python
import polars as pl
from src.data.ingestion import fetch_intraday_data
from src.features.engineering import compute_all_features
from src.features.targets import generate_targets
from src.ml.training import prepare_dataset, train_model
from src.ml.evaluation import evaluate_model

# Ingérer des données
data = fetch_intraday_data('AAPL', interval='1min')

# Calculer les features
features = compute_all_features(data)

# Générer les targets (classification)
features_with_targets = generate_targets(features, horizon=1, target_type='classification')

# Préparer le dataset
X_train, X_test, y_train, y_test, feature_names = prepare_dataset(features_with_targets)

# Entraîner un modèle
model = train_model(X_train, y_train, model_type='classification')

# Évaluer
metrics = evaluate_model(model, X_test, y_test, model_type='classification')
```

## Configuration ML

Les paramètres ML peuvent être ajustés dans `src/config.py` :

- **Feature Engineering** : Périodes pour momentum, fenêtres de volatilité, paramètres RSI/MACD
- **Target Generation** : Horizon de prédiction (nombre de périodes à l'avance)
- **XGBoost** : Hyperparamètres (n_estimators, max_depth, learning_rate, etc.)
- **Train/Test Split** : Proportion pour le test set

## Rate Limits Alpha Vantage

- Free tier: 5 appels/minute, 500 appels/jour
- Le module d'ingestion gère automatiquement les rate limits avec des retries et backoff exponentiel

## Format de données

- Format principal: Parquet (efficace, compressé)
- Colonnes standardisées: datetime, open, high, low, close, volume
- Toutes les opérations utilisent Polars pour une performance optimale

## Features générées

### Features techniques
- **RSI** (Relative Strength Index)
- **MACD** (Moving Average Convergence Divergence) avec signal et histogram
- **Bollinger Bands** (moyenne, bandes supérieure/inférieure, largeur)

### Features de momentum
- Retours sur différentes périodes (5, 10, 20, 50)
- Momentum (différence de prix)
- Rate of Change (ROC)
- Distances aux moyennes mobiles

### Features de volatilité
- Volatilité rolling (écart-type des retours)
- Realized volatility
- High-Low range

### Features temporelles
- Heure, minute, jour de la semaine, jour, mois
- Encodage cyclique (sin/cos) pour les features temporelles

## Modèles

Les modèles XGBoost sont sauvegardés dans `models/` avec le format :
`xgboost_{target_type}_{horizon}.pkl`
