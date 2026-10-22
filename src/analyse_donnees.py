import numpy as np
import pandas as pd
import re
from IPython.display import display
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from seaborn import heatmap

from pathlib import Path
import re
import math
from collections import Counter

BLEU, ORANGE, GRIS = "#68baed", "#fa7e4d", "#a8a692"
COULEURS = ["#388cf2", "#f25313", "#1fc428", "#fabc2a", "#fc79bf", "#2cc72c", "#8271e3", "#eb6160"]

def trouver(nom):
    """Cherche le fichier dans ../data (notebook lancé depuis scr/) ou data/, sans tenir compte des majuscules."""
    for dossier in [Path("../data"), Path("data")]:
        if dossier.is_dir():
            for f in dossier.iterdir():
                if f.name.lower() == nom.lower():
                    return f
    raise FileNotFoundError(f"{nom} introuvable dans ../data ni dans data")


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


RE_TEMPS = re.compile(r"^t(\d+)$")
RE_ACTION = re.compile(r"^(?P<nom>[^(<$]*?)(?:\((?P<ecran>[^)]*)\))?(?:<(?P<conf>[^>]*)>)?(?:\$(?P<chaine>[^$]*)\$)?(?P<modif>1)?$")


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
        "doc_actions": " ".join(actions),                        # texte pour le TF-IDF
        "doc_types": " ".join(types),
        "n_champs_vides": n_vides, "n_non_conformes": n_non_conformes,
    }

def read_ds(ds_name: str):
    with open(f'Data/{ds_name}.csv') as f:
        max_actions = max(( len( str(c).split(",")) for c in f.readlines() ))
        f.seek(0)
        _names = ["util", "navigateur"] if "train" in ds_name else ["navigateur"]
        _names.extend(range(max_actions - len(_names)))
        return pd.read_csv(f, names=_names, dtype= str)


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
        ("Sessions par utilisateur (min / médiane / max)", f"{effectifs.min()} / {effectifs.median():.0f} / {effectifs.max()}"),
        ("Navigateurs distincts", f"{df['navigateur'].nunique()}"),
        ("Sessions sans aucune action", f"{sans_action}"),
        ("Sessions en double (même utilisateur, même trace)", f"{doublons}"),
    ])

    display(audit.style.hide(axis="index").hide(axis="columns").set_properties(**{"text-align": "left"}))
    
def to_categories(df, col="util"):
    """Convertit une colonne textuelle en codes catégoriels numériques."""
    df[col] = pd.Categorical(df[col]).codes
    return df
    
def filter_action(value: str):
    """Nettoie une chaîne d'action en supprimant les métadonnées entre parenthèses,
    chevrons ou dollars."""
    
    for delim in [ "(", "<", "$", "1"]:
        if delim in value and (low_ind := value.index(delim)):
            value = value[:low_ind]
    return value

def browsers_analysis(df) -> None: 
    # Nombre de navigateurs différents utilisés par chaque utilisateur
    nb_browsers_per_user = df.groupby('util')['navigateur'].nunique().reset_index()
    print(f"Nombre maximum de navigateurs utilisés par un utilisateur: {nb_browsers_per_user['navigateur'].max()}")

    # Compte le nombre de sessions par navigateur
    browsers_used = df['navigateur'].value_counts()

    # Compte le nombre d'utilisateur par navigateur
    nb_users_per_browsers = df.groupby('util')['navigateur'].first().reset_index()['navigateur'].value_counts()

    # Tracer du graphique en barres
    ax = browsers_used.plot.bar(edgecolor="black", rot=45)
    ax.set_title("Nombre de sessions par navigateur")
    ax.set_xlabel("Navigateurs")
    ax.set_ylabel("Nombre de sessions")

    # Ajout des pourcentages sur les barres
    total_sessions = len(df)
    percentages = (browsers_used / total_sessions * 100).round(2)
    labels = [f"{pct}%" for pct in percentages]
    ax.bar_label(ax.containers[0], labels=labels, padding=3)

    display(pd.DataFrame({"sessions": browsers_used, "part (%)": percentages,"utilisateurs l'ayant utilisé": nb_users_per_browsers }))

def sessions_analysis(raw_df,updated_df):
    # Nombre de sessions par users
    sessions_per_users = raw_df.groupby("util").size().reset_index(name="nb_sessions")
    sessions_per_users.plot.hist(bins=30,edgecolor="black")

    updated_df["nb_sessions"] = sessions_per_users["nb_sessions"]
    print(f"Nombre maximum de sessions effectuées par un même utilisateur: {updated_df["nb_sessions"].max()}")
    print(f"Nombre minimum de sessions effectuées par un même utilisateur: {updated_df["nb_sessions"].min()}")
    print(f"Nombre moyen de sessions pour un utilisateur: {updated_df["nb_sessions"].mean()}")
    print(f"Ecart-type du nombre de sessions pour un utilisateur: {updated_df["nb_sessions"].std()}")

def apply_filter(val):
    # Vérifie si c'est une chaîne de caractères et si elle commence par "t"
    if isinstance(val, str):
        if not val.startswith("t"):
            return filter_action(val)
        else:
            return np.nan
    return val

def extract_pattern_frequencies(df,critere):
   # Sélection des colonnes d'actions 
    cols_actions = df.columns.difference(["util", "navigateur"])
    
    # 1. Extrait uniquement ce qu'il y a entre les '$' pour chaque cellule
    df_extraites = df[cols_actions].apply(
        lambda col: col.astype(str).str.extract(critere, expand=False)
    )
    
    counts_per_row = df_extraites.stack().groupby(level=0).value_counts()
    
    df_filtered = counts_per_row.unstack(fill_value=0)
    df_grouped = df_filtered.groupby(df["util"]).sum()
    freq_critere = df_grouped.div(df_grouped.sum(axis=1), axis=0).fillna(0)
    freq_critere = freq_critere.reset_index()    
    return freq_critere

def actions_analysis(df,actions_list):
    
    cols_actions = df.columns.difference(["util", "navigateur"])

    # Filtre les traces pour ne récupérer que les actions
    actions_filtered = df[cols_actions].map(apply_filter)

    # Compte pour chaque action son nombre par ligne, pour cela on crée une grande série avec stack
    counts_per_row = actions_filtered.stack().groupby(level=0).value_counts()

    # Reforme le tableau (unstack) et rajoute les utilisateurs
    df_filtered = counts_per_row.unstack(fill_value=0)
    total_actions = df_filtered.sum().sum()
    print(total_actions)

 
    #updated_df = updated_df.merge(new_features, on="util", how="left")
    utilisateurs_par_action = df_filtered.groupby(df["util"]).sum().gt(0).sum()

    tableau_actions = pd.DataFrame(
        {"Type d'action": actions_list,
         "Occurrences": df_filtered.sum(),
         "Part des actions": df_filtered.sum() / total_actions *100,
         "Sessions où il apparaît": (df_filtered > 0).sum(),
         "Utilisateurs qui l'emploient": utilisateurs_par_action,
        })
    display(tableau_actions.sort_values(by="Part des actions").style.hide(axis="index"))

    
    # Calcul la fréquence d'utilisation de chaque actions par les différents utilisateurs
    df_grouped = df_filtered.groupby(df["util"]).sum()
    freq_actions = df_grouped.div(df_grouped.sum(axis=1), axis=0)
    freq_actions = freq_actions.reset_index()

    selected_actions = _analyze_chi2_and_select(freq_actions,"Les types d'action les plus liés aux utilisateurs (test du χ²)")
    
    corr = freq_actions[selected_actions].corr()

    fig, ax = plt.subplots(figsize=(20, 20))
    heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    fig.tight_layout()

    return selected_actions

def _analyze_chi2_and_select(
        freq_df, titre="Les types d'action les plus liés aux utilisateurs (test du χ²)"
    ):

    """Calcule le Chi2, trace le diagramme en barres des 40 meilleurs scores, 
       et retourne les colonnes sélectionnées par le seuil.
    """
    X = freq_df.drop(columns=["util"])
    y = freq_df["util"]

    scores_chi2, _ = chi2(X, y)

    top35 = np.argsort(-np.nan_to_num(scores_chi2))[:35]
    
    # Sélection du top 35 pour l'affichage graphique

    plt.figure(figsize=(9, 6.5))
    plt.barh([freq_df.columns[i] for i in top35[::-1]], scores_chi2[top35[::-1]])
    plt.title(titre)
    plt.xlabel("Score du χ²")
    plt.tight_layout()
    plt.show()
  
    # Seuil de sélection (25% du max)
    seuil = max(scores_chi2) * 0.25
    selected_cols = [X.columns[k] for k in range(len(scores_chi2)) if scores_chi2[k] >= seuil]
        
    return selected_cols

def fun(df,critere):
    # 1. Initialisation avec la liste unique des utilisateurs pour démarrer les fusions
    freq_critere = df[["util"]].drop_duplicates().reset_index(drop=True)
    
    # 2. Fusion successive des résultats pour chaque règle/critère
    for regle in critere:
        freq_critere = freq_critere.merge(extract_pattern_frequencies(df, regle), on="util", how="left")

    freq_critere = freq_critere.fillna(0)

    selected_cols = _analyze_chi2_and_select(freq_critere,"Les types d'écran, fiches ou configurations les plus liés aux utilisateurs (test du χ²)") 
   
    corr = freq_critere[selected_cols].corr()

    fig, ax = plt.subplots(figsize=(20, 20))
    heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    fig.tight_layout()

    return selected_cols

def temps_par_session(df,updated_df):
   
    # Sélection des colonnes d'actions 
    cols_actions = df.columns.difference(["util", "navigateur"])

    # Extraction du nombre de chaque tXX pour toutes les colonnes d'actions
    # \d+ capture la partie numérique après le 't'
    # expend =false -> renvoie serie au lieu df
    df_t_nums = df[cols_actions].apply(
        lambda col: col.astype(str).str.extract(r"t(\d+)", expand=False).astype(float)
    )
    df["max_t"] = df_t_nums.max(axis=1) 
    
    new_features = df.groupby("util")["max_t"].agg(
        min_t="min",
        max_t="max",
        moyenne_t="mean",
        std_t="std"
    ).reset_index()

    updated_df = updated_df.merge(new_features, on="util", how="left")

    return updated_df

def rapidite_util(df,updated_df):
    rapidite = []

    for index, row in df.iterrows():
        index_NaN = df.columns[row.isna()]  
        if not(index_NaN.empty):
            index_firstNaN = index_NaN[0]
        else:
            index_firstNaN = df.columns[-2]
        
        max_t = df.loc[index,"max_t"].item()
        nb_interval = int(max_t)/5
        rapidite.append((index_firstNaN - nb_interval)/max_t)

    df.insert(1, column='rapidite', value=rapidite)
    
    df = df.groupby("util")["rapidite"].mean().reset_index()
    updated_df['rapidite'] = df['rapidite']
    return updated_df
    
def stat_actions(df,updated_df):
    # Sélection des colonnes d'actions 
    cols_actions = df.columns.difference(["util", "navigateur"])
    
    # Extraction des données d'actions pour toutes les colonnes d'actions
    dfs_actions = {
        "modif": df[cols_actions].apply(
            lambda col: col.astype(str).str.extract(r"(.*1)$", expand=False)
        ),
        "fiche": df[cols_actions].apply(
            lambda col: col.astype(str).str.extract(r"(.*\$.*\$.*)", expand=False)
        ),
        "ecran": df[cols_actions].apply(
            lambda col: col.astype(str).str.extract(r"(.*\(.*\).*)$", expand=False)
        ),
        "changement_ecran": df[cols_actions].apply(
            lambda col: col.astype(str).str.extract(r"(.*\<.*\>.*)$", expand=False)
        )
    }


    
    df_fiche = dfs_actions["fiche"].stack().unique()
    print(df_fiche)
    for key, df_extracted in dfs_actions.items():
         df[key] = df_extracted.notna().sum(axis=1)
         it = df.groupby("util")[key].agg(
             min="min",
             max="max",
             mean="mean",
             std="std"
         ).reset_index()
         it = it.rename(columns={
             "min": f"min_{key}",
             "max": f"max_{key}",
             "mean": f"mean_{key}",
             "std": f"std_{key}"
         })
         updated_df = updated_df.merge(it, on="util", how="left")

    return updated_df


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

def module(ecran):
    """Module du logiciel, lu dans le nom de l'écran : infologic.crm.modules... -> crm."""
    morceaux = ecran.split(".")
    if "infologic" in morceaux[:-1]:
        return morceaux[morceaux.index("infologic") + 1]
    return "(code court)"

def remove_outliers_per_user(group):
    # Calcul de Q1 (25%), Q3 (75%) et l'IQR pour l'utilisateur
    Q1 = group['max_t'].quantile(0.25)
    Q3 = group['max_t'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Définition des bornes (1.5 * IQR est la règle standard des boxplots)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # On ne garde que les lignes comprises entre les bornes
    return group[(group['max_t'] >= lower_bound) & (group['max_t'] <= upper_bound)]

def plot_distribution(effectifs, title, xlabel, ylabel, bins = 30,couleur_barres= BLEU, couleur_mediane=ORANGE):
    plt.figure()
    plt.hist(effectifs.values, bins=bins, color=couleur_barres, edgecolor="white")
    plt.axvline(effectifs.median(), color=couleur_mediane, linestyle="--", label=f"médiane : {effectifs.median():.0f} sessions")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()