import logging
import pandas as pd
import sys
import os

# Ajouter le chemin relatif pour accéder aux tests
sys.path.append('../tests')

# Importer les modules de configuration
from settings.base import MODELS_DIR  # Importer MODELS_DIR depuis base.py

# Importer toutes les fonctions du module domain.forcast
from domain.forcast import encode_target, split_data, cross_validate, preprocess_data, generate_predictions, evaluate_model, plot_random_forest_tree, plot_feature_importances, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve


# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_evaluate_model(df: pd.DataFrame) -> None:
    """
    Fonction pour entraîner et évaluer un modèle de RandomForest sur les données passées en paramètre.
    Elle effectue l'encodage, la séparation des données, l'entraînement, l'évaluation et la visualisation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Le DataFrame contenant les données d'entrée avec la colonne cible 'Risk'.
    """
    try:
        # Séparation des données
        X = df.drop(columns=['Risk'])  # Remplace 'Risk' par le nom de ta colonne cible
        y = df['Risk']  # Colonne cible

        # Applique l'encodage one-hot
        X_processed = preprocess_data(X)  # Applique le prétraitement des données

        # Encoder la variable cible
        y_encoded = encode_target(y)  # Encodage des labels

        # Séparer les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = split_data(X_processed, y_encoded)  # Séparation des données

        # Définir le modèle RandomForest et la grille de recherche
        grid_search = cross_validate()  # Validation croisée

        # Entraîner le modèle et évaluer
        evaluate_model(grid_search, X_train, y_train, X_test, y_test)  # Évaluation du modèle

        # Obtenir le meilleur modèle
        best_model = grid_search.best_estimator_

        # Visualisation de l'arbre de décision (arbre 0)
        plot_random_forest_tree(best_model, X_train, y_train, tree_index=0)

        # Visualiser l'importance des caractéristiques
        plot_feature_importances(best_model)

        # Prédictions sur le jeu de test
        y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]

        # Visualiser la matrice de confusion
        plot_confusion_matrix(y_test, y_pred)

        # Visualiser la courbe ROC
        plot_roc_curve(y_test, y_pred_prob)

        # Visualiser la courbe Précision/Rappel
        plot_precision_recall_curve(y_test, y_pred_prob)
        
        return y_test, y_pred, y_pred_prob

    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement et de l'évaluation du modèle : {e}")
        raise

# Fonction pour exporter les prédictions
def export_predictions(y_true, y_pred, y_pred_prob):
    """
    Fonction pour exporter les prédictions dans un fichier CSV avec un nom unique.
    
    Parameters
    ----------
    y_true : pd.Series
        Les vraies valeurs de la cible.
    y_pred : pd.Series
        Les prédictions du modèle.
    y_pred_prob : pd.Series
        Les probabilités de la classe positive.
    """
    try:
        logger.info("Début de l'exportation des prédictions.")

        base_file_name = "predictions_on_true"
        i = 1
        file_name = f"{base_file_name}_{i}.csv"
        output_path = os.path.join(MODELS_DIR, file_name)

        # Vérifier si le fichier existe déjà, et incrémenter si nécessaire
        while os.path.exists(output_path):
            i += 1
            file_name = f"{base_file_name}_{i}.csv"
            output_path = os.path.join(MODELS_DIR, file_name)
        
        logger.info(f"Nom du fichier généré : {file_name}")
        
        # Créer un DataFrame avec les prédictions
        predictions_df = pd.DataFrame({
            'True Values': y_true,
            'Predictions': y_pred,
            'Predicted Probabilities': y_pred_prob
        })
        
        # Sauvegarder le DataFrame dans un fichier CSV
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Prédictions exportées dans {output_path}.")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exportation des prédictions : {e}")
        raise