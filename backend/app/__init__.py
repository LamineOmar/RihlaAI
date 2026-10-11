# Charge un fichier .env (dans le dossier courant ou un parent) avant toute lecture de variables.
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:  # python-dotenv absent : on utilise les variables d'environnement du système
    pass
