"""
BE1 Data Science : "Qui utilise mon appli ?" . Groupe TD 4
5. OPTIMISATION DU MODÈLE ET SOUMISSION : recherches sur grille, entraînement final
et écriture du fichier pour Kaggle.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from configuration import GRAINE, METRIQUE, trouver
from modeles import creer_pipeline

# Les valeurs testées pour chaque modèle. On a gardé des grilles assez petites,
# sinon la recherche durait plusieurs heures sur nos ordinateurs.
GRILLES = {
    "Régression logistique": {"modele__C": [1, 10, 100],
                              "modele__class_weight": [None, "balanced"]},
    "SVM linéaire": {"modele__C": [0.1, 0.5, 1, 5],
                     "modele__class_weight": [None, "balanced"]},
    "Perceptron multicouche (MLP)": {"modele__hidden_layer_sizes": [(128,), (256,), (512,)],
                                     "modele__alpha": [1e-4, 1e-2]},
    "Arbre de décision": {"modele__max_depth": [None, 20, 50],
                          "modele__min_samples_leaf": [1, 2, 5],
                          "modele__class_weight": [None, "balanced"]},
    "Forêt aléatoire": {"modele__n_estimators": [300, 600],
                        "modele__max_features": ["sqrt", "log2"],
                        "modele__class_weight": [None, "balanced_subsample"]},
}

# Une fois le meilleur modèle trouvé, on règle aussi la préparation des variables
GRILLE_PREPARATION = {
    "preparation__actions__min_df": [1, 2, 3],
    "preparation__enchainements__ngram_range": [(1, 1), (1, 2), (1, 3)],
}


def optimiser_modele(pipeline, grille, app, plis=3):
    """Recherche sur grille avec validation croisée à 3 plis, sur la partie apprentissage uniquement.
    n_jobs=3 plutôt que -1 : avec tous les cœurs, certains d'entre nous manquaient de mémoire."""
    recherche = GridSearchCV(pipeline, grille, scoring=METRIQUE, n_jobs=3, refit=True,
                             cv=StratifiedKFold(plis, shuffle=True, random_state=GRAINE))
    recherche.fit(app, app["util"])
    return recherche


def resume_recherche(recherche, n=5):
    """Les n meilleures combinaisons testées, avec des noms de colonnes raccourcis pour la lecture."""
    resultats = pd.DataFrame(recherche.cv_results_)
    colonnes = [c for c in resultats.columns if c.startswith("param_")]
    resultats = resultats.sort_values("rank_test_score")[colonnes + ["mean_test_score", "std_test_score"]]
    resultats.columns = [c.replace("param_", "").replace("modele__", "").replace("preparation__", "")
                         for c in resultats.columns]
    return resultats.head(n).reset_index(drop=True)


def optimiser_tous_les_modeles(modeles, app, val, resultats_initiaux):
    """Optimise chaque modèle puis compare son score avant / après sur la validation.
    Comme la validation n'a servi à régler aucun paramètre, le gain affiché est honnête.
    Pour choisir le meilleur modèle, on se base sur la validation croisée, pas sur la validation."""
    lignes, meilleurs = {}, {}
    for nom, grille in GRILLES.items():
        debut = time.time()
        recherche = optimiser_modele(creer_pipeline(modeles[nom]), grille, app)
        f1_avant = resultats_initiaux.loc[nom, "F1 validation"]
        f1_apres = f1_score(val["util"], recherche.best_estimator_.predict(val), average="micro")
        parametres = {k.replace("modele__", ""): v for k, v in recherche.best_params_.items()}
        lignes[nom] = {"F1 validation croisée": recherche.best_score_,
                       "F1 validation avant": f1_avant, "F1 validation après": f1_apres,
                       "Gain": f1_apres - f1_avant, "Meilleurs paramètres": parametres}
        meilleurs[nom] = recherche.best_estimator_
        print(f"{nom:<32} {f1_avant:.3f} -> {f1_apres:.3f}  ({time.time() - debut:.0f} s)")
    tableau = pd.DataFrame(lignes).T.sort_values("F1 validation croisée", ascending=False)
    return tableau, meilleurs


def entrainer_modele_final(pipeline, table_train):
    """Dernière étape avant Kaggle : on réentraîne le modèle choisi sur tout train.csv,
    puisque plus de données donnent un meilleur modèle."""
    return pipeline.fit(table_train, table_train["util"])


def ecrire_soumission(predictions, chemin_sortie="submission.csv"):
    """Écrit le fichier pour Kaggle. On reprend les noms de colonnes de sample_submission.csv
    pour être sûrs d'avoir le bon format (RowId, prediction)."""
    try:
        exemple = pd.read_csv(trouver("sample_submission.csv"))
        colonne_id, colonne_prediction = exemple.columns[:2]
        if len(exemple) != len(predictions):
            print(f"Attention : {len(predictions)} prédictions pour {len(exemple)} lignes attendues.")
    except FileNotFoundError:
        colonne_id, colonne_prediction = "RowId", "prediction"
    soumission = pd.DataFrame({colonne_id: np.arange(1, len(predictions) + 1),
                               colonne_prediction: predictions})
    soumission.to_csv(chemin_sortie, index=False)
    print(f"{len(soumission)} prédictions écrites dans {chemin_sortie}")
    return soumission
