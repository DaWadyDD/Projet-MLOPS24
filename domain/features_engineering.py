import pandas as pd
import logging

# Configuration de base pour le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remplir les valeurs manquantes qui signifient le non-abonnement à un service bancaire."""
    try:
        logging.info("Début du traitement des valeurs manquantes.")
        
        if 'Saving accounts' in df.columns:
            df['Saving accounts'] = df['Saving accounts'].fillna('No savings')
            logging.info("Valeurs manquantes dans 'Saving accounts' remplies par 'No savings'.")
        
        if 'Checking account' in df.columns:
            df['Checking account'] = df['Checking account'].fillna('No checking')
            logging.info("Valeurs manquantes dans 'Checking account' remplies par 'No checking'.")
        
        return df
    except Exception as e:
        logging.error(f"Erreur lors du traitement des valeurs manquantes: {e}")
        raise

def convert_columns_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """Convertir certaines colonnes en type float pour éviter des conflits avec pd.cut()"""
    try:
        logging.info("Conversion des colonnes 'Age' et 'Credit amount' en float.")
        
        if 'Age' in df.columns:
            df['Age'] = df['Age'].astype(float)
            logging.info("Colonne 'Age' convertie en float.")
        
        if 'Credit amount' in df.columns:
            df['Credit amount'] = df['Credit amount'].astype(float)
            logging.info("Colonne 'Credit amount' convertie en float.")
        
        return df
    except Exception as e:
        logging.error(f"Erreur lors de la conversion des colonnes en float: {e}")
        raise

def create_interactive_column(df: pd.DataFrame) -> pd.DataFrame:
    """Créer une nouvelle colonne interactive entre le montant du crédit et la durée du crédit"""
    try:
        logging.info("Création de la colonne 'Credit_Duration_Ratio'.")
        
        if 'Credit amount' in df.columns and 'Duration' in df.columns:
            df['Credit_Duration_Ratio'] = df['Credit amount'] / df['Duration']
            logging.info("Colonne 'Credit_Duration_Ratio' créée.")
        
        return df
    except Exception as e:
        logging.error(f"Erreur lors de la création de la colonne interactive: {e}")
        raise

def create_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Définir des tranches d'âge"""
    try:
        logging.info("Création de la colonne 'Age_Group' avec des tranches d'âge.")
        
        if 'Age' in df.columns:
            df['Age_Group'] = pd.cut(df['Age'], bins=[20, 30, 40, float('inf')], labels=['Young', 'Adults', 'Mature-Seniors'], right=False)
            logging.info("Colonne 'Age_Group' créée avec les tranches d'âge.")
        
        return df
    except Exception as e:
        logging.error(f"Erreur lors de la création de la colonne 'Age_Group': {e}")
        raise

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprimer les colonnes inutiles"""
    try:
        logging.info("Suppression des colonnes inutiles : 'Age', 'Duration', 'Credit amount'.")
        
        columns_to_drop = ['Age', 'Duration', 'Credit amount']
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        logging.info(f"Colonnes supprimées : {columns_to_drop}.")
        return df
    except Exception as e:
        logging.error(f"Erreur lors de la suppression des colonnes inutiles: {e}")
        raise

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Appliquer toutes les étapes de feature engineering sur le DataFrame."""
    try:
        logging.info("Début du processus de feature engineering.")
        
        # Remplir les valeurs manquantes
        df = handle_missing_values(df)
        
        # Convertir certaines colonnes en float
        df = convert_columns_to_float(df)
        
        # Créer une nouvelle colonne interactive entre le montant du crédit et la durée du crédit
        df = create_interactive_column(df)
        
        # Définir des tranches d'âge
        df = create_age_group(df)
        
        # Supprimer les colonnes inutiles
        df = drop_unnecessary_columns(df)
        
        logging.info("Fin du processus de feature engineering.")
        return df
    except Exception as e:
        logging.error(f"Erreur dans le processus de feature engineering: {e}")
        raise