import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.tree import plot_tree
import logging
from sklearn.metrics import confusion_matrix

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fonction pour visualiser un arbre de décision
def plot_random_forest_tree(model: RandomForestClassifier, X_train: pd.DataFrame, y_train: pd.Series, tree_index: int = 0) -> None:
    """
    Visualiser l'un des arbres de décision du modèle RandomForest entraîné.

    Parameters
    ----------
    model : RandomForestClassifier
        Le modèle RandomForest entraîné.
    X_train : pd.DataFrame
        Données d'entraînement utilisées pour le modèle.
    y_train : pd.Series
        Labels d'entraînement utilisés pour le modèle.
    tree_index : int
        L'index de l'arbre à afficher, par défaut 0.
    """
    try:
        logger.info(f"Visualisation de l'arbre de décision à l'index {tree_index}.")
        tree = model.estimators_[tree_index]
        plt.figure(figsize=(20, 10))
        plot_tree(tree, 
                  filled=True, 
                  feature_names=X_train.columns, 
                  class_names=np.unique(y_train).astype(str), 
                  rounded=True)
        plt.show()
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation de l'arbre : {e}")
        raise

# Fonction pour visualiser l'importance des caractéristiques
def plot_feature_importances(model: RandomForestClassifier) -> None:
    """
    Visualiser l'importance des caractéristiques du modèle RandomForest.

    Parameters
    ----------
    model : RandomForestClassifier
        Le modèle RandomForest entraîné.
    """
    try:
        logger.info("Affichage de l'importance des caractéristiques.")
        feature_importances = model.feature_importances_
        feature_names = model.feature_names_in_
        
        # Créer un graphique avec Plotly
        fig = px.bar(x=feature_names, y=feature_importances, labels={'x': 'Features', 'y': 'Importance'},
                     title="Importance des caractéristiques du modèle RandomForest")
        fig.show()
    except Exception as e:
        logger.error(f"Erreur lors de l'affichage de l'importance des caractéristiques : {e}")
        raise

# Fonction pour afficher la matrice de confusion
def plot_confusion_matrix(y_true: np.array, y_pred: np.array) -> None:
    """
    Afficher la matrice de confusion sous forme de heatmap avec seaborn.

    Parameters
    ----------
    y_true : np.array
        Valeurs réelles.
    y_pred : np.array
        Valeurs prédites.
    """
    try:
        logger.info("Affichage de la matrice de confusion.")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
        plt.title('Matrice de confusion')
        plt.xlabel('Prédictions')
        plt.ylabel('Réel')
        plt.show()
    except Exception as e:
        logger.error(f"Erreur lors de l'affichage de la matrice de confusion : {e}")
        raise
