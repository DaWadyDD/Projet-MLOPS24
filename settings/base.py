import os

# Répertoire racine du projet
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Chemins vers les répertoires de données
RAW_DATA_DIR = os.path.normpath(os.path.join(REPO_DIR, 'data', 'raw'))
FORMATTED_DATA_DIR = os.path.normpath(os.path.join(REPO_DIR, 'data', 'formated'))
MODELS_DIR = os.path.normpath(os.path.join(REPO_DIR, 'data', 'models'))

# Chemin d'accès au fichier de configuration pour le logging
LOGGING_CONFIGURATION_FILE = os.path.normpath(os.path.join(os.path.dirname(__file__), 'logging.yaml'))

# Vérification des répertoires
for dir_path in [RAW_DATA_DIR, FORMATTED_DATA_DIR, MODELS_DIR]:
    if not os.path.exists(dir_path):
        print(f"Attention : Le répertoire n'existe pas -> {dir_path}")
    else:
        print(f"Répertoire confirmé -> {dir_path}")

# Vérification du fichier de configuration
if not os.path.exists(LOGGING_CONFIGURATION_FILE):
    print(f"Attention : Le fichier logging.yaml est introuvable à l'emplacement -> {LOGGING_CONFIGURATION_FILE}")
else:
    print(f"Fichier logging confirmé -> {LOGGING_CONFIGURATION_FILE}")

