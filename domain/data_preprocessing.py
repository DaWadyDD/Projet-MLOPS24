import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fonction pour encoder les labels
def encode_target(y: pd.Series) -> np.array:
    """
    Encoder les classes cibles en entiers.

    Parameters
    ----------
    y : pd.Series
        Labels à encoder.

    Returns
    -------
    np.array
        Labels encodés.
    """
    try:
        logger.info("Encodage des labels.")
        le = LabelEncoder()
        return le.fit_transform(y)
    except Exception as e:
        logger.error(f"Erreur lors de l'encodage des labels : {e}")
        raise

# Fonction pour séparer les données
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Séparer les données en ensembles d'entraînement et de test.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Labels.
    test_size : float
        Proportion des données de test, par défaut 0.25.

    Returns
    -------
    Tuple[np.array, np.array, np.array, np.array]
        Données d'entraînement et de test pour X et y.
    """
    try:
        logger.info("Séparation des données en ensembles d'entraînement et de test.")
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    except Exception as e:
        logger.error(f"Erreur lors de la séparation des données : {e}")
        raise

# Fonction pour prétraiter les données
def preprocess_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    Applique l'encodage one-hot pour les variables catégorielles.
    
    Parameters
    ----------
    X : pd.DataFrame
        Données d'entrée contenant des variables catégorielles.
    
    Returns
    -------
    pd.DataFrame
        Données prétraitées avec variables numériques.
    """
    try:
        logger.info("Prétraitement des données par encodage one-hot.")
        return pd.get_dummies(X, drop_first=True)
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement des données : {e}")
        raise

# Fonction pour la recherche par validation croisée
def cross_validate() -> GridSearchCV:
    """
    Définir le modèle RandomForest et la grille de recherche pour GridSearchCV.

    Returns
    -------
    GridSearchCV
        Le processus de recherche sur les meilleurs hyperparamètres.
    """
    try:
        logger.info("Initialisation de la recherche par validation croisée pour RandomForest.")
        model = RandomForestClassifier()
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        return GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    except Exception as e:
        logger.error(f"Erreur lors de la configuration de la validation croisée : {e}")
        raise
