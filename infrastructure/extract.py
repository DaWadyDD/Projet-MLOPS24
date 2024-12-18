import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from domain.decorators import log_function_call  # Importer le décorateur


@log_function_call  # Appliquer le décorateur ici
def extract(data_dir: str, prefix: str) -> pd.DataFrame:
    """
    Charger et concaténer tous les fichiers CSV dont le nom commence par 'german' 
    dans le répertoire 'raw', et supprimer la colonne 'Unnamed: 0' si elle existe.

    La fonction parcourt le répertoire 'raw', identifie tous les fichiers 
    dont le nom commence par 'german' et les concatène en un seul DataFrame. 
    La colonne 'Unnamed: 0' est supprimée si elle est présente.

    Parameters
    ----------
    data_dir : str
        Chemin du répertoire où se trouvent les fichiers de données.
    prefix : str
        Préfixe des fichiers à charger. Seuls les fichiers correspondant à ce préfixe 
        (ex : 'german_credit_X.csv') seront traités.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les données concaténées après suppression de la colonne 'Unnamed: 0' (si présente).
    """
    df = pd.DataFrame()

    # Vérification du répertoire 'raw' et construction du chemin correct
    raw_dir = os.path.join(data_dir)
    
    # Vérification si le répertoire existe
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Le répertoire spécifié n'a pas été trouvé : {raw_dir}")

    # Liste des fichiers correspondant au préfixe 'german' dans le répertoire 'raw'
    files = [f for f in os.listdir(raw_dir) if f.startswith('german') and f.endswith('.csv')]

    for file in files:
        file_path = os.path.join(raw_dir, file)
        if os.path.isfile(file_path):
            # Log de chaque fichier chargé
            df_temp = pd.read_csv(file_path)
            # Suppression de la colonne 'Unnamed: 0' si elle existe
            if 'Unnamed: 0' in df_temp.columns:
                df_temp.drop(columns=['Unnamed: 0'], inplace=True)
            df = pd.concat([df, df_temp], ignore_index=True, sort=True)

    return df
