import os

# Répertoire racine du projet
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Chemins vers les répertoires de données
RAW_DATA_DIR = os.path.join(REPO_DIR, 'data', 'raw')
FORMATTED_DATA_DIR = os.path.join(REPO_DIR, 'data', 'formated')
MODELS_DIR = os.path.join(REPO_DIR, 'data', 'models')

# Chemin d'accès au fichier de configuration pour le logging
LOGGING_CONFIGURATION_FILE = os.path.join(os.path.dirname(__file__), 'logging.yaml')
