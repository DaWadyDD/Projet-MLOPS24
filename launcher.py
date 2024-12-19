import sys
import os
import pandas as pd

# Ajouter le chemin pour accéder aux modules
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Importer les fonctions nécessaires
from settings.base import RAW_DATA_DIR
from infrastructure.extract import extract
from domain.features_engineering import feature_engineering
from domain.model_training import train_and_evaluate_model, export_predictions

# Appeler extract pour récupérer le DataFrame
df = extract(RAW_DATA_DIR, 'prefix')

# Appliquer le feature engineering
df = feature_engineering(df)

print(df)

# Appliquer l'entrainement, la prediction et la visualisation du meilleur modèle
y_test, y_pred, y_pred_prob = train_and_evaluate_model(df)

# Appliquer l'export des prédictions en csv
export_predictions(y_test, y_pred, y_pred_prob)
