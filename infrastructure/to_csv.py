import os
from datetime import datetime
import pandas as pd

# Définir les chemins de base
base_dir = os.path.dirname(os.path.abspath(__file__))  # Répertoire racine du projet

# Chemins des répertoires "formated" et "models"
formatted_dir = os.path.join(base_dir, 'data', 'formated')
models_dir = os.path.join(base_dir, 'data', 'models')

# Vérifier si les répertoires existent, sinon les créer
os.makedirs(formatted_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Générer un nom de fichier basé sur la date et l'heure actuelle pour éviter les conflits
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Sauvegarde du DataFrame formaté
formatted_filename = f"credits_formated_{timestamp}.csv"
formatted_filepath = os.path.join(formatted_dir, formatted_filename)
df.to_csv(formatted_filepath, index=False)
print(f"Le fichier formaté a été enregistré sous: {formatted_filepath}")

# Créer un DataFrame avec les prédictions et les vraies valeurs
predictions_df = pd.DataFrame({
    'True Values': y_test,  # Valeurs réelles
    'Predictions': y_pred,  # Prédictions de classes
    'Prediction Probabilities': y_pred_prob  # Probabilités de la classe positive
})

# Sauvegarde des prédictions
predictions_filename = f"predictions_on_true_{timestamp}.csv"
predictions_filepath = os.path.join(models_dir, predictions_filename)
predictions_df.to_csv(predictions_filepath, index=False)
print(f"Les prédictions ont été enregistrées sous: {predictions_filepath}")
