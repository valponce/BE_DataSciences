"""
BE1 Data Science : "Qui utilise mon appli ?" . Groupe TD 4
4. MODÈLES DE CLASSIFICATION : préparation des variables (TF-IDF, one-hot, statistiques),
modèles comparés, découpage apprentissage / validation et évaluation.
"""

import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from configuration import CACHE, COLONNES_NUMERIQUES, GRAINE, METRIQUE

def construire_preparation(min_df=2, ngram_enchainements=(1, 2)):
    """Transforme le tableau des sessions en une seule matrice de nombres pour les modèles.
    Chaque bloc correspond à une idée qu'on a eue pendant l'exploration :
    - actions : ce que fait l'utilisateur et sur quel écran (TF-IDF des jetons complets) ;
    - enchainements : dans quel ordre il le fait (paires d'actions qui se suivent) ;
    - composants : les écrans, configurations et chaînes qu'il ouvre ;
    - navigateur : en one-hot, puisque chacun garde toujours le même ;
    - statistiques : son rythme de travail, centré réduit.
    On était partis sur des n-grammes jusqu'à 3, mais c'était beaucoup plus long
    pour presque rien de gagné, donc on s'arrête à 2 par défaut."""
    def tfidf(**options):
        # token_pattern \S+ : on découpe seulement sur les espaces, une action = un mot
        # lowercase=False : on garde les majuscules, les noms d'écrans y sont sensibles
        # sublinear_tf : une action répétée 500 fois ne doit pas peser 500 fois plus
        return TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True, **options)

    return ColumnTransformer([
        ("actions", tfidf(min_df=min_df), "doc_actions"),
        ("enchainements", tfidf(min_df=min_df, ngram_range=ngram_enchainements), "doc_types"),
        ("composants", tfidf(min_df=2), "doc_composants"),
        ("navigateur", OneHotEncoder(handle_unknown="ignore"), ["navigateur"]),
        ("statistiques", StandardScaler(), COLONNES_NUMERIQUES),
    ])


def creer_modeles():
    """Les cinq modèles qu'on compare, plus une référence naïve pour avoir un point de comparaison.
    Les paramètres de départ sont des valeurs classiques, on les optimise en section 5."""
    return {
        "Référence (utilisateur le plus fréquent)": DummyClassifier(strategy="most_frequent"),
        "Régression logistique": LogisticRegression(C=10, max_iter=2000),
        "SVM linéaire": LinearSVC(C=1, max_iter=5000),
        # Le MLP était le plus lent de tous : on a réduit la couche cachée et arrêté
        # l'entraînement dès que le score ne progresse plus
        "Perceptron multicouche (MLP)": MLPClassifier(hidden_layer_sizes=(128,), max_iter=40, tol=1e-3,
                                              n_iter_no_change=5, random_state=GRAINE),
        "Arbre de décision": DecisionTreeClassifier(random_state=GRAINE),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=GRAINE),
    }


def creer_pipeline(modele, **options_preparation):
    """Préparation + modèle dans un même Pipeline. L'intérêt : le TF-IDF et le StandardScaler
    sont appris seulement sur les données d'apprentissage, jamais sur la validation."""
    return Pipeline([("preparation", construire_preparation(**options_preparation)),
                 ("modele", modele)], memory=CACHE)

def decouper_apprentissage_validation(table, part_validation=0.25):
    """75 % pour apprendre, 25 % pour valider, en stratifiant sur l'utilisateur
    pour que chacun soit présent des deux côtés. Un utilisateur qui n'a plus qu'une seule
    session après le nettoyage ne peut pas être coupé en deux : on le met en apprentissage."""
    effectifs = table["util"].value_counts()
    uniques = table["util"].isin(effectifs[effectifs < 2].index)
    app, val = train_test_split(table[~uniques], test_size=part_validation,
                                stratify=table.loc[~uniques, "util"], random_state=GRAINE)
    app = pd.concat([app, table[uniques]])
    print(f"Apprentissage : {len(app)} sessions | Validation : {len(val)} sessions")
    return app.reset_index(drop=True), val.reset_index(drop=True)


def calculer_scores(y_vrai, y_predit):
    """Les quatre scores qu'on affiche dans nos tableaux."""
    return {
        "Accuracy": accuracy_score(y_vrai, y_predit),
        "F1 micro": f1_score(y_vrai, y_predit, average="micro"),
        "F1 macro": f1_score(y_vrai, y_predit, average="macro", zero_division=0),
        "F1 pondéré": f1_score(y_vrai, y_predit, average="weighted", zero_division=0),
    }


def evaluer_modele(pipeline, app, val):
    """Entraîne sur l'apprentissage, puis mesure le score sur les deux parties.
    Le score sur l'apprentissage sert seulement à repérer le sur-apprentissage."""
    debut = time.time()
    pipeline.fit(app, app["util"])
    duree = time.time() - debut
    y_predit = pipeline.predict(val)
    scores = calculer_scores(val["util"], y_predit)
    scores["F1 apprentissage"] = f1_score(app["util"], pipeline.predict(app), average="micro")
    scores["F1 validation"] = scores["F1 micro"]
    scores["Temps (s)"] = duree
    return scores, y_predit


def comparer_modeles(modeles, app, val):
    """Même préparation et même découpage pour tout le monde, pour que la comparaison soit juste."""
    lignes, predictions, pipelines = {}, {}, {}
    for nom, modele in modeles.items():
        pipeline = creer_pipeline(modele)
        scores, y_predit = evaluer_modele(pipeline, app, val)
        lignes[nom], predictions[nom], pipelines[nom] = scores, y_predit, pipeline
        print(f"{nom:<42} F1 validation = {scores['F1 validation']:.3f}  ({scores['Temps (s)']:.1f} s)")
    resultats = pd.DataFrame(lignes).T.sort_values("F1 validation", ascending=False)
    return resultats, predictions, pipelines


def validation_croisee(pipeline, table, plis=5):
    """On refait l'évaluation sur 5 découpages différents, pour vérifier que notre score
    ne vient pas d'un découpage chanceux. Renvoie la moyenne et l'écart type."""
    cv = StratifiedKFold(plis, shuffle=True, random_state=GRAINE)
    scores = cross_val_score(pipeline, table, table["util"], cv=cv, scoring=METRIQUE, n_jobs=-1)
    return scores.mean(), scores.std()
