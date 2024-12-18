import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from typing import Tuple, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        plt.title("Matrice de confusion")
        plt.xlabel("Prédictions")
        plt.ylabel("Vraies valeurs")
        plt.show()
    except Exception as e:
        logger.error(f"Erreur lors de l'affichage de la matrice de confusion : {e}")
        raise

def plot_roc_curve(y_true: np.array, y_pred_prob: np.array) -> None:
    """
    Afficher la courbe ROC pour évaluer la performance du modèle.

    Parameters
    ----------
    y_true : np.array
        Valeurs réelles.
    y_pred_prob : np.array
        Probabilités prédites par le modèle.
    """
    try:
        from sklearn.metrics import roc_curve, auc
        
        logger.info("Affichage de la courbe ROC.")
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Courbe ROC')
        plt.legend(loc='lower right')
        plt.show()
    except Exception as e:
        logger.error(f"Erreur lors de l'affichage de la courbe ROC : {e}")
        raise

def plot_precision_recall_curve(y_true, y_pred_prob):
    """
    Affiche la courbe précision/rappel et logge les informations relatives à la performance.

    Parameters
    ----------
    y_true : pd.Series
        Labels réels.
    y_pred_prob : np.array
        Probabilités prédites.
    """
    try:
        # Calcul de la courbe précision/rappel
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        
        # Tracer la courbe
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.show()
        
        # Log de l'information
        logger.info("Precision-Recall Curve générée avec succès.")
    
    except Exception as e:
        # Log de l'erreur
        logger.error("Erreur lors de la génération de la courbe Precision-Recall: %s", e)