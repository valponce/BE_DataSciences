

# 1. CONFIGURATION


import math
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from seaborn import heatmap

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_score,
                                     train_test_split, validation_curve)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Graine aléatoire : rend les découpages et les modèles reproductibles
GRAINE = 42

# Métrique du challenge : F1 score. Une session n'a qu'un seul utilisateur,
# donc le F1 "micro" est égal à l'accuracy. On suit aussi le F1 "macro",
# qui donne le même poids à chaque utilisateur, même s'il a peu de sessions.
METRIQUE = "f1_micro"

# Variables numériques construites pour chaque session (voir decrire_session)
COLONNES_NUMERIQUES = ["log_actions", "log_duree", "cadence", "nb_types",
                       "ratio_distinct", "entropie", "nb_ecrans"]

BLEU, ORANGE, GRIS = "#68baed", "#fa7e4d", "#a8a692"
COULEURS = ["#388cf2", "#f25313", "#1fc428", "#fabc2a", "#fc79bf", "#2cc72c", "#8271e3", "#eb6160"]


def configurer_graphiques():
    """Style commun à toutes les figures du notebook."""
    plt.rcParams.update({
        "figure.figsize": (9.5, 4.5), "figure.dpi": 70,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.edgecolor": "#bccffb", "grid.color": "#c6dbdd",
    })


def trouver(nom):
    """Cherche un fichier de données dans Data/ ou data/ (depuis la racine du dépôt
    ou depuis src/), sans tenir compte des majuscules dans les noms."""
    for dossier in [Path("Data"), Path("data"), Path("../Data"), Path("../data")]:
        if dossier.is_dir():
            for f in dossier.iterdir():
                if f.name.lower() == nom.lower():
                    return f
    raise FileNotFoundError(f"{nom} introuvable dans Data/ ou ../Data/")



# 2. PRÉPARATION ET NETTOYAGE DES DONNÉES


# 2.1 Lecture des fichiers 

def read_ds(ds_name: str):
    """Lit train.csv ou test.csv au format "large" : une ligne par session,
    une colonne par jeton (les lignes n'ont pas toutes la même longueur)."""
    fichier = trouver(f"{ds_name}.csv")
    with open(fichier, encoding="utf-8-sig") as f:
        max_actions = max(len(str(c).split(",")) for c in f.readlines())
        f.seek(0)
        _names = ["util", "navigateur"] if "train" in ds_name else ["navigateur"]
        _names.extend(range(max_actions - len(_names)))
        return pd.read_csv(f, names=_names, dtype=str)


def lire_sessions(fichier, avec_utilisateur):
    """Lit le fichier ligne par ligne. Renvoie la liste des sessions et le nombre de lignes vides."""
    sessions, lignes_vides = [], 0
    with open(fichier, encoding="utf-8-sig") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                lignes_vides += 1
                continue
            champs = ligne.split(",")
            if avec_utilisateur:
                sessions.append({"util": champs[0], "navigateur": champs[1], "jetons": champs[2:]})
            else:
                sessions.append({"navigateur": champs[0], "jetons": champs[1:]})
    return sessions, lignes_vides


# 2.2 Découpage d'un jeton d'action 

RE_TEMPS = re.compile(r"^t(\d+)$")
RE_ACTION = re.compile(r"^(?P<nom>[^(<$]*?)(?:\((?P<ecran>[^)]*)\))?(?:<(?P<conf>[^>]*)>)?"
                       r"(?:\$(?P<chaine>[^$]*)\$)?(?P<modif>1)?$")


def filter_action(value: str):
    """Garde seulement le type d'action : retire l'écran (...), la configuration <...>,
    la chaîne $...$ et le 1 final qui indique une modification."""
    nom = re.split(r"[(<$]", value)[0]
    if nom.endswith("1"):
        nom = nom[:-1]
    return nom.strip()


def apply_filter(val):
    """Type d'action pour un jeton d'action, NaN pour un marqueur de temps tXX."""
    if isinstance(val, str):
        return np.nan if RE_TEMPS.match(val) else filter_action(val)
    return val


def module(ecran):
    """Module du logiciel, lu dans le nom de l'écran : infologic.crm.modules... -> crm."""
    morceaux = ecran.split(".")
    if "infologic" in morceaux[:-1]:
        return morceaux[morceaux.index("infologic") + 1]
    return "(code court)"


# 2.3 Construction des variables d'une session 

def decrire_session(jetons):
    """Transforme la liste des jetons d'une session en quelques indicateurs simples."""
    types, actions, ecrans, configs, chaines = [], [], [], [], []
    fenetres = Counter()                 # nombre d'actions dans chaque fenêtre de 5 secondes
    t, n_marqueurs, n_vides, n_non_conformes = 0, 0, 0, 0
    for jeton in jetons:
        jeton = jeton.strip()
        if not jeton:
            n_vides += 1
            continue
        m = RE_TEMPS.match(jeton)
        if m:                            # marqueur temporel
            n_marqueurs += 1
            t = max(t, int(m.group(1)))
            continue
        m = RE_ACTION.match(jeton)       # action
        if m is None or not m["nom"].strip():
            n_non_conformes += 1
            nom = jeton
        else:
            nom = m["nom"].strip()
            if m["ecran"]:
                ecrans.append(m["ecran"])
            if m["conf"]:
                configs.append(m["conf"])
            if m["chaine"]:
                chaines.append(m["chaine"])
        types.append(nom.replace(" ", "_"))          # un symbole sans espace par action
        actions.append(jeton.replace(" ", "_"))
        fenetres[t] += 1

    n = len(types)
    n_fenetres = t // 5 + 1
    frequences = Counter(types)
    return {
        "n_actions": n,
        "n_marqueurs": n_marqueurs,
        "duree_s": t,                                            # durée estimée = dernier marqueur
        "cadence": n / max(t, 5),                                # actions par seconde
        "nb_types": len(frequences),
        "ratio_distinct": len(set(actions)) / max(n, 1),         # part d'actions différentes
        "entropie": -sum(c / n * math.log2(c / n) for c in frequences.values()) if n else 0.0,
        "nb_ecrans": len(set(ecrans)),
        "actions_par_fenetre": list(fenetres.values()) + [0] * (n_fenetres - len(fenetres)),
        "ecrans": ecrans, "configs": configs, "chaines": chaines,
        "doc_actions": " ".join(actions),                        # textes pour le TF-IDF
        "doc_types": " ".join(types),
        "doc_composants": " ".join(["ecran=" + e.replace(" ", "_") for e in ecrans]
                                   + ["conf=" + c.replace(" ", "_") for c in configs]
                                   + ["chaine=" + c.replace(" ", "_") for c in chaines]),
        "n_champs_vides": n_vides, "n_non_conformes": n_non_conformes,
    }


def construire_table_sessions(nom_fichier, avec_utilisateur):
    """Une ligne par session avec toutes les variables utilisées par les modèles."""
    sessions, lignes_vides = lire_sessions(trouver(nom_fichier), avec_utilisateur)
    lignes = []
    for s in sessions:
        ligne = {"navigateur": s["navigateur"], "empreinte": hash(tuple(s["jetons"]))}
        if avec_utilisateur:
            ligne["util"] = s["util"]
        ligne.update(decrire_session(s["jetons"]))
        lignes.append(ligne)
    table = pd.DataFrame(lignes)
    # Les longueurs et durées vont de quelques unités à plusieurs milliers :
    # le logarithme rend leur distribution plus symétrique
    table["log_actions"] = np.log1p(table["n_actions"])
    table["log_duree"] = np.log1p(table["duree_s"])
    return table, lignes_vides


# 2.4 Contrôle qualité et nettoyage 

def describe_data(df):
    """Affiche une description globale de la structure du DataFrame
    (sessions, utilisateurs, doublons, etc.)."""
    effectifs = df["util"].value_counts()
    n_sessions = len(df)
    sans_action = df.iloc[:, 2].isna().sum()
    doublons = df.duplicated().sum()

    audit = pd.DataFrame([
        ("Nombre de sessions", f"{n_sessions:,}"),
        ("Nombre d'utilisateurs", f"{len(effectifs)}"),
        ("Sessions par utilisateur (min / médiane / max)",
         f"{effectifs.min()} / {effectifs.median():.0f} / {effectifs.max()}"),
        ("Navigateurs distincts", f"{df['navigateur'].nunique()}"),
        ("Sessions sans aucune action", f"{sans_action}"),
        ("Sessions en double (même utilisateur, même trace)", f"{doublons}"),
    ])
    display(audit.style.hide(axis="index").hide(axis="columns").set_properties(**{"text-align": "left"}))


def controler_qualite(table, lignes_vides):
    """Tableau de contrôle : valeurs manquantes, sessions vides, doublons, jetons mal formés."""
    controles = [
        ("Lignes vides dans le fichier", lignes_vides),
        ("Sessions sans aucune action", int((table["n_actions"] == 0).sum())),
        ("Jetons vides (deux virgules consécutives)", int(table["n_champs_vides"].sum())),
        ("Jetons qui ne suivent pas le format décrit", int(table["n_non_conformes"].sum())),
    ]
    if "util" in table:
        controles.append(("Sessions en double (même utilisateur, même trace)",
                          int(table.duplicated(["util", "empreinte"]).sum())))
        traces_ambigues = table.groupby("empreinte")["util"].nunique()
        controles.append(("Traces identiques attribuées à plusieurs utilisateurs",
                          int((traces_ambigues > 1).sum())))
    resultat = pd.DataFrame(controles, columns=["Contrôle", "Résultat"])
    display(resultat.style.hide(axis="index"))
    return resultat


def nettoyer_sessions(table):
    """Retire les sessions sans action et les doublons exacts (même utilisateur,
    même trace) : un doublon pourrait se retrouver à la fois dans la partie
    apprentissage et dans la partie validation et fausser l'évaluation."""
    avant = len(table)
    table = table[table["n_actions"] > 0]
    if "util" in table:
        table = table.drop_duplicates(["util", "empreinte"])
    table = table.reset_index(drop=True)
    print(f"{avant - len(table)} sessions retirées, {len(table)} sessions conservées.")
    return table


def to_categories(df, col="util"):
    """Convertit une colonne textuelle en codes catégoriels numériques."""
    df[col] = pd.Categorical(df[col]).codes
    return df


def retirer_sessions_aberrantes(df, colonne="max_t"):
    """Retire, utilisateur par utilisateur, les sessions dont la valeur sort de
    [Q1 - 1.5 IQR ; Q3 + 1.5 IQR] (règle des boîtes à moustaches)."""
    groupes = df.groupby("util", observed=True)[colonne]
    q1 = groupes.transform(lambda s: s.quantile(0.25))
    q3 = groupes.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    garder = df[colonne].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return df[garder]


# 2.5 Indicateurs par utilisateur (format large) 
def colonnes_actions(df):
    """Colonnes qui contiennent les jetons (leurs noms sont des entiers)."""
    return [c for c in df.columns if isinstance(c, (int, np.integer))]


def jetons_longs(df):
    """Tous les jetons dans une seule série (index = (session, position)).
    Beaucoup plus rapide que de traiter les milliers de colonnes une par une."""
    return df[colonnes_actions(df)].stack().dropna().astype(str)


def extract_pattern_frequencies(df, critere, prefixe=""):
    """Fréquence, pour chaque utilisateur, des motifs capturés par l'expression régulière."""
    motifs = jetons_longs(df).str.extract(critere, expand=False).dropna()
    motifs = prefixe + motifs
    comptes = motifs.groupby(level=0).value_counts().unstack(fill_value=0)
    par_util = comptes.groupby(df.loc[comptes.index, "util"].astype(str)).sum()
    freq_critere = par_util.div(par_util.sum(axis=1), axis=0).fillna(0)
    return freq_critere.rename_axis("util").reset_index()


def temps_par_session(df, updated_df):
    """Durée de chaque session (dernier marqueur tXX) puis min / max / moyenne / écart type par utilisateur.
    Ajoute aussi la colonne max_t au DataFrame des sessions."""
    temps = jetons_longs(df).str.extract(r"^t(\d+)$", expand=False).dropna().astype(float)
    df["max_t"] = temps.groupby(level=0).max().reindex(df.index).fillna(0)

    new_features = df.groupby(df["util"].astype(str))["max_t"].agg(
        min_t="min", max_t="max", moyenne_t="mean", std_t="std"
    ).reset_index()
    updated_df = updated_df.assign(util=updated_df["util"].astype(str))
    return updated_df.merge(new_features, on="util", how="left")


def rapidite_util(df, updated_df):
    """Cadence moyenne de chaque utilisateur : nombre d'actions divisé par la durée de la session."""
    jetons = jetons_longs(df)
    est_action = ~jetons.str.match(r"^t\d+$")
    n_actions = est_action.groupby(level=0).sum().reindex(df.index).fillna(0)
    rapidite = n_actions / df["max_t"].clip(lower=5)
    par_util = rapidite.groupby(df["util"].astype(str)).mean().rename("rapidite").reset_index()
    updated_df = updated_df.assign(util=updated_df["util"].astype(str))
    return updated_df.merge(par_util, on="util", how="left")


def stat_actions(df, updated_df):
    """Par session : nombre d'actions en modification, sur une fiche, avec un écran,
    avec un changement de configuration. Puis min / max / moyenne / écart type par utilisateur."""
    jetons = jetons_longs(df)
    motifs = {
        "modif": r"1$",
        "fiche": r"\$.*\$",
        "ecran": r"\(.*\)",
        "changement_ecran": r"<.*>",
    }
    updated_df = updated_df.assign(util=updated_df["util"].astype(str))
    for key, motif in motifs.items():
        df[key] = jetons.str.contains(motif).groupby(level=0).sum().reindex(df.index).fillna(0)
        it = df.groupby(df["util"].astype(str))[key].agg(["min", "max", "mean", "std"])
        it.columns = [f"{statistique}_{key}" for statistique in it.columns]
        updated_df = updated_df.merge(it.reset_index(), on="util", how="left")
    return updated_df



# 3. VISUALISATION


def plot_distribution(effectifs, title, xlabel, ylabel, bins=30, couleur_barres=BLEU, couleur_mediane=ORANGE):
    """Histogramme avec la médiane en pointillés."""
    plt.figure()
    plt.hist(effectifs.dropna().values, bins=bins, color=couleur_barres, edgecolor="white")
    plt.axvline(effectifs.median(), color=couleur_mediane, linestyle="--",
                label=f"médiane : {effectifs.median():.2f}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


def barres_top(compteur, titre, xlabel, n=15, couleur=BLEU):
    """Diagramme en barres horizontales des n valeurs les plus fréquentes."""
    top = compteur.most_common(n)[::-1]
    etiquettes = [v if len(v) <= 45 else "…" + v[-44:] for v, _ in top]   # garde la fin des noms longs
    plt.figure(figsize=(9, 5.5))
    plt.barh([e.replace("$", r"\$") for e in etiquettes], [c for _, c in top], color=couleur)
    plt.title(titre)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.show()


def browsers_analysis(df) -> None:
    """Sessions et utilisateurs par navigateur, et fidélité des utilisateurs à leur navigateur."""
    nb_browsers_per_user = df.groupby("util", observed=True)["navigateur"].nunique()
    print(f"Nombre maximum de navigateurs utilisés par un utilisateur : {nb_browsers_per_user.max()}")

    browsers_used = df["navigateur"].value_counts()
    nb_users_per_browsers = df.groupby("navigateur")["util"].nunique()

    ax = browsers_used.plot.bar(color=BLEU, edgecolor="black", rot=45)
    ax.set_title("Nombre de sessions par navigateur")
    ax.set_xlabel("Navigateurs")
    ax.set_ylabel("Nombre de sessions")
    percentages = (browsers_used / len(df) * 100).round(2)
    ax.bar_label(ax.containers[0], labels=[f"{pct}%" for pct in percentages], padding=3)
    plt.tight_layout()
    plt.show()

    display(pd.DataFrame({"sessions": browsers_used, "part (%)": percentages,
                          "utilisateurs l'ayant utilisé": nb_users_per_browsers}))


def sessions_analysis(raw_df, updated_df):
    """Distribution du nombre de sessions par utilisateur ; ajoute la colonne nb_sessions."""
    sessions_per_users = raw_df.groupby(raw_df["util"].astype(str)).size().rename("nb_sessions").reset_index()
    updated_df = updated_df.assign(util=updated_df["util"].astype(str)).merge(sessions_per_users, on="util", how="left")
    plot_distribution(updated_df["nb_sessions"], "Nombre de sessions par utilisateur",
                      "Nombre de sessions", "Nombre d'utilisateurs")
    print(updated_df["nb_sessions"].describe().round(1))
    return updated_df


def _analyze_chi2_and_select(X, y, titre, seuil_relatif=0.25, n_affiches=35):
    """Score du χ² de chaque colonne de X par rapport à l'utilisateur y.
    Trace les meilleurs scores et renvoie les colonnes dont le score dépasse seuil_relatif * max."""
    scores_chi2, _ = chi2(X, y)
    scores = pd.Series(np.nan_to_num(scores_chi2), index=X.columns).sort_values(ascending=False)

    top = scores.head(n_affiches)[::-1]
    plt.figure(figsize=(9, 6.5))
    plt.barh([str(c).replace("$", r"\$") for c in top.index], top.values, color=BLEU)
    plt.title(titre)
    plt.xlabel("Score du χ²")
    plt.tight_layout()
    plt.show()

    return list(scores[scores >= scores.max() * seuil_relatif].index)


def tracer_correlations(table, colonnes, titre, n_max=20):
    """Matrice de corrélation entre les colonnes retenues (limitée aux n_max premières)."""
    colonnes = colonnes[:n_max]
    corr = table[colonnes].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title(titre)
    fig.tight_layout()
    plt.show()


def actions_analysis(df):
    """Tableau des types d'action, test du χ² et corrélations entre les types les plus liés à l'utilisateur.
    Renvoie la liste des types d'action retenus."""
    jetons = jetons_longs(df)
    types = jetons[~jetons.str.match(r"^t\d+$")].map(filter_action)

    # Nombre de chaque type d'action dans chaque session
    comptes = types.groupby(level=0).value_counts().unstack(fill_value=0)
    utilisateurs = df.loc[comptes.index, "util"].astype(str)
    total_actions = comptes.to_numpy().sum()
    print(f"Nombre total d'actions : {total_actions:,}")

    tableau_actions = pd.DataFrame({
        "Occurrences": comptes.sum(),
        "Part des actions (%)": (comptes.sum() / total_actions * 100).round(2),
        "Sessions où il apparaît": (comptes > 0).sum(),
        "Utilisateurs qui l'emploient": (comptes.groupby(utilisateurs).sum() > 0).sum(),
    }).rename_axis("Type d'action").sort_values("Occurrences", ascending=False)
    display(tableau_actions)

    # Test du χ² sur les fréquences par session
    frequences = comptes.div(comptes.sum(axis=1), axis=0)
    selected_actions = _analyze_chi2_and_select(
        frequences, utilisateurs, "Les types d'action les plus liés aux utilisateurs (test du χ²)")
    tracer_correlations(frequences, selected_actions, "Corrélation entre les types d'action retenus")
    return selected_actions


def analyser_composants_logiciels(df):
    """Même démarche que actions_analysis pour les chaînes $...$, écrans (...) et configurations <...>."""
    regles = {"chaine=": r"\$(.*?)\$", "ecran=": r"\((.*?)\)", "conf=": r"<(.*?)>"}
    jetons = jetons_longs(df)
    morceaux = []
    for prefixe, regle in regles.items():
        motifs = jetons.str.extract(regle, expand=False).dropna()
        morceaux.append((prefixe + motifs).groupby(level=0).value_counts().unstack(fill_value=0))
    comptes = pd.concat(morceaux, axis=1).fillna(0)
    utilisateurs = df.loc[comptes.index, "util"].astype(str)
    frequences = comptes.div(comptes.sum(axis=1), axis=0)

    selected_cols = _analyze_chi2_and_select(
        frequences, utilisateurs,
        "Écrans, chaînes et configurations les plus liés aux utilisateurs (test du χ²)")
    tracer_correlations(frequences, selected_cols, "Corrélation entre les composants retenus")
    return selected_cols


def tracer_comparaison_modeles(resultats):
    """Barres : F1 sur la partie apprentissage (gris) et sur la partie validation (bleu)."""
    resultats = resultats.sort_values("F1 validation")
    y = np.arange(len(resultats))
    plt.figure(figsize=(9, 4.5))
    plt.barh(y + 0.2, resultats["F1 apprentissage"], height=0.4, color=GRIS, label="apprentissage")
    plt.barh(y - 0.2, resultats["F1 validation"], height=0.4, color=BLEU, label="validation")
    for i, v in enumerate(resultats["F1 validation"]):
        plt.text(v + 0.01, i - 0.2, f"{v:.3f}", va="center", fontsize=9)
    plt.yticks(y, resultats.index)
    plt.xlim(0, 1.1)
    plt.xlabel("F1 score (micro)")
    plt.title("Comparaison des modèles")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def tracer_matrice_confusion(y_vrai, y_predit, utilisateurs, titre="Matrice de confusion"):
    """Matrice de confusion restreinte à une liste d'utilisateurs (sinon illisible avec 247 classes)."""
    utilisateurs = list(utilisateurs)
    masque = pd.Series(y_vrai).isin(utilisateurs).to_numpy()
    matrice = confusion_matrix(np.asarray(y_vrai)[masque], np.asarray(y_predit)[masque], labels=utilisateurs)
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(matrice, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=utilisateurs, yticklabels=utilisateurs, annot_kws={"size": 8})
    plt.xlabel("Utilisateur prédit")
    plt.ylabel("Utilisateur réel")
    plt.title(titre)
    plt.tight_layout()
    plt.show()


def tracer_f1_par_utilisateur(y_vrai, y_predit, y_apprentissage):
    """F1 de chaque utilisateur en fonction de son nombre de sessions d'apprentissage."""
    utilisateurs = np.unique(y_vrai)
    f1_util = f1_score(y_vrai, y_predit, labels=utilisateurs, average=None, zero_division=0)
    n_sessions = pd.Series(y_apprentissage).value_counts().reindex(utilisateurs).fillna(0)
    par_util = pd.DataFrame({"n_sessions": n_sessions.values, "f1": f1_util}, index=utilisateurs)

    plt.figure()
    plt.scatter(par_util["n_sessions"], par_util["f1"], color=BLEU, alpha=0.7, edgecolor="white")
    plt.xlabel("Nombre de sessions de l'utilisateur dans la partie apprentissage")
    plt.ylabel("F1 de l'utilisateur (validation)")
    plt.title("Qualité de la reconnaissance selon le nombre d'exemples")
    plt.tight_layout()
    plt.show()
    return par_util.sort_values("f1")


def tracer_importance_variables(pipeline, n=20, titre="Variables les plus importantes"):
    """Importance des variables pour un modèle à base d'arbres (arbre de décision ou forêt)."""
    noms = pipeline.named_steps["preparation"].get_feature_names_out()
    importances = pd.Series(pipeline.named_steps["modele"].feature_importances_, index=noms)
    top = importances.sort_values(ascending=False).head(n)[::-1]
    plt.figure(figsize=(9, 6))
    etiquettes = [nom.split("__", 1)[-1][-55:].replace("$", r"\$") for nom in top.index]
    plt.barh(etiquettes, top.values, color=BLEU)
    plt.xlabel("Importance (diminution moyenne de l'impureté de Gini)")
    plt.title(titre)
    plt.tight_layout()
    plt.show()
    return importances.sort_values(ascending=False)


def importance_par_groupe(importances):
    """Somme des importances par groupe de variables (actions, enchaînements, composants, ...)."""
    groupes = importances.groupby(importances.index.str.split("__").str[0]).sum()
    return groupes.sort_values(ascending=False).round(3).rename("importance totale")


def tracer_arbre(pipeline, profondeur=3):
    """Affiche le haut de l'arbre de décision (les premières questions posées par le modèle)."""
    noms = [n.split("__", 1)[-1][-35:] for n in pipeline.named_steps["preparation"].get_feature_names_out()]
    arbre = pipeline.named_steps["modele"]
    plt.figure(figsize=(18, 7))
    plot_tree(arbre, max_depth=profondeur, feature_names=noms, class_names=list(arbre.classes_),
              filled=True, impurity=False, proportion=True, fontsize=7)
    plt.title(f"Arbre de décision : {profondeur} premiers niveaux")
    plt.show()


def tracer_courbe_validation(pipeline, X, y, nom_parametre, valeurs, titre, echelle_log=True, plis=3):
    """Score en apprentissage et en validation croisée selon la valeur d'un paramètre."""
    score_app, score_val = validation_curve(
        pipeline, X, y, param_name=nom_parametre, param_range=valeurs,
        cv=StratifiedKFold(plis, shuffle=True, random_state=GRAINE), scoring=METRIQUE, n_jobs=-1)
    plt.figure()
    plt.plot(valeurs, score_app.mean(axis=1), "o-", color=GRIS, label="apprentissage")
    plt.plot(valeurs, score_val.mean(axis=1), "o-", color=BLEU, label="validation croisée")
    plt.fill_between(valeurs, score_val.mean(axis=1) - score_val.std(axis=1),
                     score_val.mean(axis=1) + score_val.std(axis=1), color=BLEU, alpha=0.15)
    if echelle_log:
        plt.xscale("log")
    plt.xlabel(nom_parametre.replace("modele__", ""))
    plt.ylabel("F1 score (micro)")
    plt.title(titre)
    plt.legend()
    plt.tight_layout()
    plt.show()
    meilleure = valeurs[int(np.argmax(score_val.mean(axis=1)))]
    print(f"Meilleure valeur : {meilleure} (F1 = {score_val.mean(axis=1).max():.3f})")
    return meilleure



# 4. MODÈLES DE CLASSIFICATION


def construire_preparation(min_df=2, ngram_enchainements=(1, 3)):
    """Transforme le tableau des sessions en une matrice numérique :
    - TF-IDF des jetons d'action complets (type + écran + configuration + chaîne) ;
    - TF-IDF des enchaînements de types d'action (n-grammes = actions consécutives) ;
    - TF-IDF des composants logiciels (écrans, configurations, chaînes) ;
    - navigateur codé en one-hot ;
    - statistiques de la session centrées réduites."""
    def tfidf(**options):
        return TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True, **options)

    return ColumnTransformer([
        ("actions", tfidf(min_df=min_df), "doc_actions"),
        ("enchainements", tfidf(min_df=min_df, ngram_range=ngram_enchainements), "doc_types"),
        ("composants", tfidf(min_df=1), "doc_composants"),
        ("navigateur", OneHotEncoder(handle_unknown="ignore"), ["navigateur"]),
        ("statistiques", StandardScaler(), COLONNES_NUMERIQUES),
    ])


def creer_modeles():
    """Les modèles comparés, avec des paramètres raisonnables avant optimisation."""
    return {
        "Référence (utilisateur le plus fréquent)": DummyClassifier(strategy="most_frequent"),
        "Régression logistique": LogisticRegression(C=10, max_iter=2000),
        "SVM linéaire": LinearSVC(C=1, max_iter=5000),
        "Perceptron multicouche (MLP)": MLPClassifier(hidden_layer_sizes=(256,), max_iter=100,
                                                      random_state=GRAINE),
        "Arbre de décision": DecisionTreeClassifier(random_state=GRAINE),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=GRAINE),
    }


def creer_pipeline(modele, **options_preparation):
    """Enchaîne la préparation des variables et le modèle. La préparation (vocabulaire du TF-IDF,
    moyennes du StandardScaler) est apprise uniquement sur les données d'apprentissage."""
    return Pipeline([("preparation", construire_preparation(**options_preparation)), ("modele", modele)])


def decouper_apprentissage_validation(table, part_validation=0.25):
    """Découpage stratifié : chaque utilisateur garde la même proportion de sessions dans les deux parties.
    Un utilisateur qui n'a qu'une seule session est placé en apprentissage."""
    effectifs = table["util"].value_counts()
    uniques = table["util"].isin(effectifs[effectifs < 2].index)
    app, val = train_test_split(table[~uniques], test_size=part_validation,
                                stratify=table.loc[~uniques, "util"], random_state=GRAINE)
    app = pd.concat([app, table[uniques]])
    print(f"Apprentissage : {len(app)} sessions | Validation : {len(val)} sessions")
    return app.reset_index(drop=True), val.reset_index(drop=True)


def calculer_scores(y_vrai, y_predit):
    """Accuracy et F1 (micro, macro, pondéré)."""
    return {
        "Accuracy": accuracy_score(y_vrai, y_predit),
        "F1 micro": f1_score(y_vrai, y_predit, average="micro"),
        "F1 macro": f1_score(y_vrai, y_predit, average="macro", zero_division=0),
        "F1 pondéré": f1_score(y_vrai, y_predit, average="weighted", zero_division=0),
    }


def evaluer_modele(pipeline, app, val):
    """Entraîne sur la partie apprentissage, évalue sur les deux parties."""
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
    """Évalue chaque modèle avec la même préparation et le même découpage.
    Renvoie le tableau des scores, les prédictions et les pipelines entraînés."""
    lignes, predictions, pipelines = {}, {}, {}
    for nom, modele in modeles.items():
        pipeline = creer_pipeline(modele)
        scores, y_predit = evaluer_modele(pipeline, app, val)
        lignes[nom], predictions[nom], pipelines[nom] = scores, y_predit, pipeline
        print(f"{nom:<42} F1 validation = {scores['F1 validation']:.3f}  ({scores['Temps (s)']:.1f} s)")
    resultats = pd.DataFrame(lignes).T.sort_values("F1 validation", ascending=False)
    return resultats, predictions, pipelines


def validation_croisee(pipeline, table, plis=5):
    """F1 en validation croisée stratifiée : moyenne et écart type entre les plis."""
    cv = StratifiedKFold(plis, shuffle=True, random_state=GRAINE)
    scores = cross_val_score(pipeline, table, table["util"], cv=cv, scoring=METRIQUE, n_jobs=-1)
    return scores.mean(), scores.std()



# 5. OPTIMISATION DU MODÈLE ET SOUMISSION


# Paramètres testés pour chaque modèle (recherche sur grille en validation croisée)
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

# Paramètres de la préparation des variables, testés ensuite sur le meilleur modèle
GRILLE_PREPARATION = {
    "preparation__actions__min_df": [1, 2, 3],
    "preparation__enchainements__ngram_range": [(1, 1), (1, 2), (1, 3)],
}


def optimiser_modele(pipeline, grille, app, plis=3):
    """Recherche sur grille en validation croisée, uniquement sur la partie apprentissage."""
    recherche = GridSearchCV(pipeline, grille, scoring=METRIQUE, n_jobs=-1, refit=True,
                             cv=StratifiedKFold(plis, shuffle=True, random_state=GRAINE))
    recherche.fit(app, app["util"])
    return recherche


def resume_recherche(recherche, n=5):
    """Les n meilleures combinaisons de paramètres testées."""
    resultats = pd.DataFrame(recherche.cv_results_)
    colonnes = [c for c in resultats.columns if c.startswith("param_")]
    resultats = resultats.sort_values("rank_test_score")[colonnes + ["mean_test_score", "std_test_score"]]
    resultats.columns = [c.replace("param_", "").replace("modele__", "").replace("preparation__", "")
                         for c in resultats.columns]
    return resultats.head(n).reset_index(drop=True)


def optimiser_tous_les_modeles(modeles, app, val, resultats_initiaux):
    """Optimise chaque modèle de GRILLES puis compare avant / après sur la partie validation.
    La partie validation ne sert à aucun réglage : le gain mesuré est donc une estimation honnête."""
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
    # Le classement se fait sur la validation croisée (partie apprentissage seulement)
    tableau = pd.DataFrame(lignes).T.sort_values("F1 validation croisée", ascending=False)
    return tableau, meilleurs


def entrainer_modele_final(pipeline, table_train):
    """Réentraîne le pipeline retenu sur toutes les sessions de train.csv."""
    return pipeline.fit(table_train, table_train["util"])


def ecrire_soumission(predictions, chemin_sortie="submission.csv"):
    """Écrit le fichier au format attendu par Kaggle (mêmes colonnes que sample_submission.csv)."""
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
