import os

# Définir le répertoire principal du projet
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Définir le chemin d'accès aux données brutes (raw)
DATA_DIR = os.path.join(REPO_DIR, 'data', 'raw')

# Définir le chemin vers le fichier de configuration de logging
LOGGING_CONFIGURATION_FILE = os.path.join(os.path.dirname(__file__), 'logging.yaml')