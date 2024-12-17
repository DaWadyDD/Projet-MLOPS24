import os
import pandas as pd
from domain.decorators import log_return_shape
from typing import List

@log_return_shape
def extract(data_dir: str, prefix: str, x_values: List[int]) -> pd.DataFrame:
    """
    Préparer et concaténer les fichiers CSV au format 'german_credit_X.csv' 
    où X est spécifié dans la liste 'x_values'.

    Parameters
    ----------
    data_dir : str
        Chemin du répertoire des données.
    prefix : str
        Identifiant de la source des données.
    x_values : List[int]
        Liste des valeurs de X pour les fichiers à concaténer.

    Returns
    -------
    pd.DataFrame
        DataFrame prête à être remplie avec des fichiers CSV.
    """
    df = pd.DataFrame()
    for x in x_values:
        file_name = f'{prefix}_german_credit_{x}.csv'
        file_path = os.path.join(data_dir, 'batchs', file_name)
        if os.path.isfile(file_path):
            batch = pd.read_csv(file_path)
            df = pd.concat([df, batch], ignore_index=True, sort=True)
    return df

