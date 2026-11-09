"""
BE1 Data Science : "Qui utilise mon appli ?" . Groupe TD 4
1. CONFIGURATION : constantes communes, style des graphiques, recherche des fichiers de données.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from joblib import Memory
from sklearn.exceptions import ConvergenceWarning

# Le TF-IDF est recalculé à chaque essai de la recherche sur grille, ce qui prenait
# beaucoup de temps chez nous. Avec ce cache, la préparation déjà calculée est réutilisée.
CACHE = Memory("cache", verbose=0)

# Sans ça, le MLP et la régression logistique remplissent le notebook d'avertissements
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# On fixe la graine pour retrouver les mêmes découpages et les mêmes scores à chaque exécution
GRAINE = 42

# Kaggle évalue avec le F1 score. Comme chaque session a un seul utilisateur,
# le F1 micro revient à l'accuracy. On regarde aussi le F1 macro dans les tableaux,
# parce qu'il compte autant les utilisateurs rares que les autres.
METRIQUE = "f1_micro"

# Les statistiques numériques qu'on calcule pour chaque session (voir decrire_session)
COLONNES_NUMERIQUES = ["log_actions", "log_duree", "cadence", "nb_types",
                       "ratio_distinct", "entropie", "nb_ecrans"]

BLEU, ORANGE, GRIS = "#68baed", "#fa7e4d", "#a8a692"
COULEURS = ["#388cf2", "#f25313", "#1fc428", "#fabc2a", "#fc79bf", "#2cc72c", "#8271e3", "#eb6160"]


def configurer_graphiques():
    """Même style pour toutes nos figures, pour que le rapport soit homogène."""
    plt.rcParams.update({
        "figure.figsize": (9.5, 4.5), "figure.dpi": 70,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.edgecolor": "#bccffb", "grid.color": "#c6dbdd",
    })


def trouver(nom):
    """Retrouve un fichier de données, qu'on lance le code depuis la racine ou depuis src/.
    On ignore les majuscules : selon les ordinateurs du groupe, on avait Data/ ou data/,
    TRAIN.csv ou train.csv, et sous Linux ça faisait planter la lecture."""
    for dossier in [Path("Data"), Path("data"), Path("../Data"), Path("../data")]:
        if dossier.is_dir():
            for f in dossier.iterdir():
                if f.name.lower() == nom.lower():
                    return f
    raise FileNotFoundError(f"{nom} introuvable dans Data/ ou ../Data/")
