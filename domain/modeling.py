import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.tree import plot_tree

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fonction pour générer les prédictions
def generate_predictions(model, X: pd.DataFrame) -> np.array:
    """
    Générer les prédictions avec le modèle.

    Parameters
    ----------
    model : sklearn model
        Modèle préalablement entraîné.
    X : pd.DataFrame
        Données d'entrée pour faire les prédictions.

    Returns
    -------
    np.array
        Prédictions générées par le modèle.
    """
    try:
        logger.info("Génération des prédictions.")
        if model is None:
            raise ValueError("Le modèle n'est pas initialisé.")
        
        if X.shape[0] == 0:
            raise ValueError("Les données d'entrée sont vides.")
        
        predictions = model.predict(X)
        logger.info(f"Prédictions générées pour {X.shape[0]} exemples.")
        return predictions
    except ValueError as e:
        logger.error(f"Erreur de valeur lors de la génération des prédictions : {e}")
        raise
    except Exception as e:
        logger.error(f"Erreur inconnue lors de la génération des prédictions : {e}")
        raise

# Fonction pour évaluer le modèle
def evaluate_model(grid_search: GridSearchCV, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Exécuter la recherche des meilleurs hyperparamètres et évaluer le modèle sur le jeu de test.

    Parameters
    ----------
    grid_search : GridSearchCV
        Le processus de recherche.
    X_train : pd.DataFrame
        Données d'entraînement.
    y_train : pd.Series
        Labels d'entraînement.
    X_test : pd.DataFrame
        Données de test.
    y_test : pd.Series
        Labels de test.
    """
    try:
        # Exécution de la recherche des meilleurs paramètres
        grid_search.fit(X_train, y_train)
        
        logger.info("Best Parameters: %s", grid_search.best_params_)
        logger.info("Best Cross-Validation Accuracy: %.4f", grid_search.best_score_)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        logger.info("Test Set Performance:\n%s", classification_report(y_test, y_pred))
    
    except Exception as e:
        logger.error("Erreur lors de l'évaluation du modèle: %s", e)
