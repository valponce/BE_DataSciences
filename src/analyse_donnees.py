"""
BE1 Data Science : "Qui utilise mon appli ?"
Groupe TD 4 : Hajar Zarguan, Valentine Ponce, Guillaume Sumian

Le code est réparti en cinq fichiers, un par étape du traitement :

    configuration.py   1. constantes, style des graphiques, recherche des données
    preparation.py     2. lecture, nettoyage et variables de chaque session
    visualisation.py   3. graphiques de l'exploration et de l'analyse des modèles
    modeles.py         4. préparation des variables, modèles comparés, évaluation
    optimisation.py    5. recherche sur grille, modèle final, soumission

Ce fichier les regroupe : le notebook n'a qu'un seul import à faire,
`import analyse_donnees as AN`, puis il appelle AN.read_ds(...), AN.creer_modeles(), etc.
"""

from configuration import *
from preparation import *
from visualisation import *
from modeles import *
from optimisation import *
