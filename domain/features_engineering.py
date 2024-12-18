import pandas as pd

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remplir les valeurs manquantes qui signifient le non abonnenement à un service bancaire."""
    df['Saving accounts'] = df['Saving accounts'].fillna('No savings')
    df['Checking account'] = df['Checking account'].fillna('No checking')
    return df

def convert_columns_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """Convertir certaines colonnes en type float pour éviter des conflits avec pd.cut()"""
    df['Age'] = df['Age'].astype(float)
    df['Credit amount'] = df['Credit amount'].astype(float)
    return df

def create_interactive_column(df: pd.DataFrame) -> pd.DataFrame:
    """Créer une nouvelle colonne interactive entre le montant du crédit et la durée du crédit"""
    df['Credit_Duration_Ratio'] = df['Credit amount'] / df['Duration']
    return df

def create_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Définir des tranches d'âge"""
    df['Age_Group'] = pd.cut(df['Age'], bins=[20, 30, 40, float('inf')], labels=['Young', 'Adults', 'Mature-Seniors'], right=False)
    return df

def categorize_credit_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Catégoriser le montant du crédit"""
    df['Credit amount'] = pd.cut(
        df['Credit amount'], 
        bins=[0, 2500, 5000, float('inf')], 
        labels=['Small', 'Moderate', 'Big']
    )
    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprimer les colonnes inutiles"""
    df.drop(columns=['Checking account', 'Saving accounts', 'Age'], inplace=True)
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Appliquer toutes les étapes de feature engineering sur le DataFrame."""
    df = handle_missing_values(df)
    df = convert_columns_to_float(df)
    df = create_interactive_column(df)
    df = create_age_group(df)
    df = categorize_credit_amount(df)
    df = drop_unnecessary_columns(df)
    return df