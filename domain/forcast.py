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

def plot_random_forest_tree(model: RandomForestClassifier, tree_index: int = 0) -> None:
    """
    Visualiser l'un des arbres de décision du modèle RandomForest entraîné.

    Paramètres
    ----------
    model : RandomForestClassifier
        Le modèle RandomForest entraîné.
    tree_index : int
        L'index de l'arbre à afficher, par défaut 0.
    """
    # Extract the specific tree from the random forest
    tree = model.estimators_[tree_index]
    
    # Plot the tree
    plt.figure(figsize=(20,10))
    plot_tree(tree, filled=True, feature_names=X_train.columns, class_names=np.unique(y_train).astype(str), rounded=True)
    plt.show()
