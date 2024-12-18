import sys
import os
import pandas as pd

# Ajouter le chemin pour accéder aux modules
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Importer les fonctions nécessaires
from settings.base import RAW_DATA_DIR
from infrastructure.extract import extract
from domain.features_engineering import feature_engineering
from domain.model_training import train_and_evaluate_model

# Appeler extract pour récupérer le DataFrame
df = extract(RAW_DATA_DIR, 'prefix')

# Appliquer le feature engineering
df = feature_engineering(df)  # Passer df à feature_engineering

print(df)

# Appliquer l'entrainement et prediction
train_and_evaluate_model(df)
