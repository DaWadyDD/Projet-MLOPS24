import logging
import pandas as pd
import sys
sys.path.append('../tests')  # Ajustez le chemin relatif si nécessaire

from forcast_test import encode_target, split_data, cross_validate, preprocess_data, generate_predictions, evaluate_model, plot_random_forest_tree, plot_feature_importances, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve;

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_evaluate_model(df: pd.DataFrame) -> None:
    """
    Fonction pour entraîner et évaluer un modèle de RandomForest sur les données passées en paramètre.
    Elle effectue l'encodage, la séparation des données, l'entraînement et l'évaluation.
    
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

    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement et de l'évaluation du modèle : {e}")
        raise