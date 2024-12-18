import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
    le = LabelEncoder()
    return le.fit_transform(y)

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
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

def cross_validate() -> GridSearchCV:
    """
    Définir le modèle RandomForest et la grille de recherche pour GridSearchCV.

    Returns
    -------
    GridSearchCV
        Le processus de recherche sur les meilleurs hyperparamètres.
    """
    model = RandomForestClassifier()
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    return GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

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
    return pd.get_dummies(X, drop_first=True)


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
    grid_search.fit(X_train, y_train)
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("\nTest Set Performance:\n", classification_report(y_test, y_pred))

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
    # Extraire l'arbre spécifique du modèle RandomForest
    tree = model.estimators_[tree_index]
    
    # Dessiner l'arbre
    plt.figure(figsize=(20, 10))
    plot_tree(tree, 
              filled=True, 
              feature_names=X_train.columns, 
              class_names=np.unique(y_train).astype(str), 
              rounded=True)
    plt.show()

def list_model_results(models: List[GridSearchCV], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Lister les résultats des modèles dans un DataFrame avec les meilleurs paramètres et performances.

    Parameters
    ----------
    models : List[GridSearchCV]
        Liste de processus de recherche pour les modèles.
    X_test : pd.DataFrame
        Données de test.
    y_test : pd.Series
        Labels de test.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les résultats des modèles.
    """
    results = []
    for model in models:
        best_model = model.best_estimator_
        y_pred = best_model.predict(X_test)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        best_params = model.best_params_
        results.append({
            'Best Parameters': best_params,
            'Accuracy': classification_rep['accuracy'],
            'Precision (class 0)': classification_rep['0']['precision'],
            'Recall (class 0)': classification_rep['0']['recall'],
            'F1-score (class 0)': classification_rep['0']['f1-score']
        })
    return pd.DataFrame(results)

def generate_predictions_df(model: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Générer un DataFrame avec les prédictions du modèle.

    Parameters
    ----------
    model : GridSearchCV
        Le processus de recherche pour le modèle.
    X_test : pd.DataFrame
        Données de test.
    y_test : pd.Series
        Labels de test.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les vraies valeurs et les prédictions.
    """
    best_model = model.best_estimator_
    y_pred = best_model.predict(X_test)
    pred_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred
    })
    return pred_df

import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_importances(model: RandomForestClassifier) -> None:
    """
    Visualiser l'importance des caractéristiques du modèle RandomForest.

    Parameters
    ----------
    model : RandomForestClassifier
        Le modèle RandomForest entraîné.
    """
    feature_importances = model.feature_importances_
    feature_names = model.feature_names_in_
    
    # Créer un graphique avec Plotly
    fig = px.bar(x=feature_names, y=feature_importances, labels={'x': 'Features', 'y': 'Importance'},
                 title="Importance des caractéristiques du modèle RandomForest")
    fig.show()

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
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies valeurs")
    plt.show()

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
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    plt.show()

def plot_precision_recall_curve(y_true: np.array, y_pred_prob: np.array) -> None:
    """
    Afficher la courbe de précision et rappel.

    Parameters
    ----------
    y_true : np.array
        Valeurs réelles.
    y_pred_prob : np.array
        Probabilités prédites par le modèle.
    """
    from sklearn.metrics import precision_recall_curve
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', lw=2)
    plt.xlabel('Rappel')
    plt.ylabel('Précision')
    plt.title('Courbe Précision/Rappel')
    plt.show()

