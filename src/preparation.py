"""
BE1 Data Science : "Qui utilise mon appli ?" . Groupe TD 4
2. PRÉPARATION ET NETTOYAGE DES DONNÉES : lecture des fichiers, découpage des jetons,
variables de chaque session, contrôle qualité et indicateurs par utilisateur.
"""

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from IPython.display import display

from configuration import trouver

# 2.1 Lecture des fichiers

def read_ds(ds_name: str):
    """Lecture au format "large" : une ligne par session, une colonne par jeton.
    Les lignes n'ont pas la même longueur, donc on cherche d'abord la plus longue
    pour savoir combien de colonnes créer (les cases en trop restent vides)."""
    fichier = trouver(f"{ds_name}.csv")
    with open(fichier, encoding="utf-8-sig") as f:
        max_actions = max(len(str(c).split(",")) for c in f.readlines())
        f.seek(0)   # on revient au début du fichier pour la vraie lecture
        _names = ["util", "navigateur"] if "train" in ds_name else ["navigateur"]
        _names.extend(range(max_actions - len(_names)))
        return pd.read_csv(f, names=_names, dtype=str)


def lire_sessions(fichier, avec_utilisateur):
    """Deuxième façon de lire, plus légère : chaque session devient un petit dictionnaire
    avec la liste de ses jetons. On compte au passage les lignes vides."""
    sessions, lignes_vides = [], 0
    with open(fichier, encoding="utf-8-sig") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                lignes_vides += 1
                continue
            champs = ligne.split(",")
            if avec_utilisateur:     # train.csv : le premier champ est l'utilisateur
                sessions.append({"util": champs[0], "navigateur": champs[1], "jetons": champs[2:]})
            else:                    # test.csv : pas d'utilisateur, on commence au navigateur
                sessions.append({"navigateur": champs[0], "jetons": champs[1:]})
    return sessions, lignes_vides


# 2.2 Découpage d'un jeton d'action

# Un marqueur de temps ressemble à t5, t10, t15...
RE_TEMPS = re.compile(r"^t(\d+)$")
# Une action suit le format Nom(écran)<configuration>$chaîne$1, et chaque morceau est facultatif.
# Les groupes nommés permettent ensuite de récupérer directement m["ecran"], m["conf"], etc.
RE_ACTION = re.compile(r"^(?P<nom>[^(<$]*?)(?:\((?P<ecran>[^)]*)\))?(?:<(?P<conf>[^>]*)>)?"
                       r"(?:\$(?P<chaine>[^$]*)\$)?(?P<modif>1)?$")


def filter_action(value: str):
    """Garde seulement le nom de l'action, par exemple "Saisie dans un champ".
    On coupe au premier ( < ou $, puis on enlève le 1 final (mode modification)."""
    nom = re.split(r"[(<$]", value)[0]
    if nom.endswith("1"):
        nom = nom[:-1]
    return nom.strip()


def apply_filter(val):
    """Nom de l'action pour un jeton d'action ; NaN pour un marqueur tXX, qu'on ne veut pas compter."""
    if isinstance(val, str):
        return np.nan if RE_TEMPS.match(val) else filter_action(val)
    return val


def module(ecran):
    """Le module du logiciel est le mot qui suit "infologic" dans le nom de l'écran
    (infologic.crm.modules... donne crm). Certains écrans ont un nom court sans ce préfixe."""
    morceaux = ecran.split(".")
    if "infologic" in morceaux[:-1]:
        return morceaux[morceaux.index("infologic") + 1]
    return "(code court)"


# 2.3 Construction des variables d'une session

def decrire_session(jetons):
    """C'est ici qu'on transforme une session brute en variables utilisables.
    On parcourt les jetons un par un : les tXX font avancer le temps, les autres sont des actions
    qu'on découpe en nom / écran / configuration / chaîne."""
    types, actions, ecrans, configs, chaines = [], [], [], [], []
    fenetres = Counter()                 # combien d'actions dans chaque fenêtre de 5 s
    t, n_marqueurs, n_vides, n_non_conformes = 0, 0, 0, 0
    for jeton in jetons:
        jeton = jeton.strip()
        if not jeton:                    # deux virgules qui se suivent
            n_vides += 1
            continue
        m = RE_TEMPS.match(jeton)
        if m:                            # marqueur de temps : on garde le plus grand vu jusqu'ici
            n_marqueurs += 1
            t = max(t, int(m.group(1)))
            continue
        m = RE_ACTION.match(jeton)
        if m is None or not m["nom"].strip():
            # jeton qui ne suit pas le format annoncé : on le garde tel quel mais on le compte
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
        # On remplace les espaces par des _ pour que le TF-IDF voie chaque action comme un seul "mot"
        types.append(nom.replace(" ", "_"))
        actions.append(jeton.replace(" ", "_"))
        fenetres[t] += 1

    n = len(types)
    n_fenetres = t // 5 + 1
    frequences = Counter(types)
    return {
        "n_actions": n,
        "n_marqueurs": n_marqueurs,
        "duree_s": t,                                     # on n'a pas mieux que le dernier marqueur pour la durée
        "cadence": n / max(t, 5),                         # actions par seconde (minimum 5 s pour éviter de diviser par 0)
        "nb_types": len(frequences),
        "ratio_distinct": len(set(actions)) / max(n, 1),  # proche de 1 = l'utilisateur fait rarement deux fois la même chose
        # entropie : faible si la session répète toujours les mêmes actions, élevée si elle est variée
        "entropie": -sum(c / n * math.log2(c / n) for c in frequences.values()) if n else 0.0,
        "nb_ecrans": len(set(ecrans)),
        "actions_par_fenetre": list(fenetres.values()) + [0] * (n_fenetres - len(fenetres)),
        "ecrans": ecrans, "configs": configs, "chaines": chaines,
        # Les trois "textes" qu'on donne ensuite au TF-IDF
        "doc_actions": " ".join(actions),
        "doc_types": " ".join(types),
        # On préfixe chaque composant pour ne pas confondre un écran et une chaîne qui auraient le même nom
        "doc_composants": " ".join(["ecran=" + e.replace(" ", "_") for e in ecrans]
                                   + ["conf=" + c.replace(" ", "_") for c in configs]
                                   + ["chaine=" + c.replace(" ", "_") for c in chaines]),
        "n_champs_vides": n_vides, "n_non_conformes": n_non_conformes,
    }


def construire_table_sessions(nom_fichier, avec_utilisateur):
    """Le tableau qu'on utilise pour les modèles : une ligne par session, une colonne par variable."""
    sessions, lignes_vides = lire_sessions(trouver(nom_fichier), avec_utilisateur)
    lignes = []
    for s in sessions:
        # L'empreinte nous sert juste à repérer les traces identiques (doublons)
        ligne = {"navigateur": s["navigateur"], "empreinte": hash(tuple(s["jetons"]))}
        if avec_utilisateur:
            ligne["util"] = s["util"]
        ligne.update(decrire_session(s["jetons"]))
        lignes.append(ligne)
    table = pd.DataFrame(lignes)
    # Certaines sessions ont 10 actions, d'autres plusieurs milliers. On passe au log
    # pour que ces valeurs extrêmes n'écrasent pas tout le reste.
    table["log_actions"] = np.log1p(table["n_actions"])
    table["log_duree"] = np.log1p(table["duree_s"])
    return table, lignes_vides


# 2.4 Contrôle qualité et nettoyage

def describe_data(df):
    """Premier coup d'œil sur le train : sessions, utilisateurs, navigateurs, doublons."""
    effectifs = df["util"].value_counts()
    n_sessions = len(df)
    sans_action = df.iloc[:, 2].isna().sum()   # si la 1re colonne d'action est vide, la session est vide
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
    """Les vérifications vues dans le cours de méthodologie, réunies dans un seul tableau."""
    controles = [
        ("Lignes vides dans le fichier", lignes_vides),
        ("Sessions sans aucune action", int((table["n_actions"] == 0).sum())),
        ("Jetons vides (deux virgules consécutives)", int(table["n_champs_vides"].sum())),
        ("Jetons qui ne suivent pas le format décrit", int(table["n_non_conformes"].sum())),
    ]
    if "util" in table:   # ces deux contrôles n'ont de sens que pour le train
        controles.append(("Sessions en double (même utilisateur, même trace)",
                          int(table.duplicated(["util", "empreinte"]).sum())))
        traces_ambigues = table.groupby("empreinte")["util"].nunique()
        controles.append(("Traces identiques attribuées à plusieurs utilisateurs",
                          int((traces_ambigues > 1).sum())))
    resultat = pd.DataFrame(controles, columns=["Contrôle", "Résultat"])
    display(resultat.style.hide(axis="index"))
    return resultat


def nettoyer_sessions(table):
    """On enlève les sessions vides et les doublons exacts du train.
    Un doublon pourrait tomber à la fois en apprentissage et en validation :
    le modèle "reconnaîtrait" alors une session qu'il a déjà vue, et le score serait trop beau."""
    avant = len(table)
    table = table[table["n_actions"] > 0]
    if "util" in table:
        table = table.drop_duplicates(["util", "empreinte"])
    table = table.reset_index(drop=True)
    print(f"{avant - len(table)} sessions retirées, {len(table)} sessions conservées.")
    return table


def to_categories(df, col="util"):
    """Remplace les trigrammes par des numéros (0, 1, 2...)."""
    df[col] = pd.Categorical(df[col]).codes
    return df


def retirer_sessions_aberrantes(df, colonne="max_t"):
    """Règle des boîtes à moustaches, appliquée utilisateur par utilisateur :
    on enlève ce qui sort de [Q1 - 1.5 IQR ; Q3 + 1.5 IQR].
    On ne s'en sert que pour calculer la cadence moyenne, pas pour entraîner les modèles."""
    groupes = df.groupby("util", observed=True)[colonne]
    q1 = groupes.transform(lambda s: s.quantile(0.25))
    q3 = groupes.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    garder = df[colonne].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return df[garder]


# 2.5 Indicateurs par utilisateur (à partir du format large)

def colonnes_actions(df):
    """Dans le format large, les colonnes de jetons sont numérotées 0, 1, 2...
    Les autres (util, navigateur, max_t...) ont des noms en texte."""
    return [c for c in df.columns if isinstance(c, (int, np.integer))]


def jetons_longs(df):
    """Met tous les jetons les uns sous les autres dans une seule colonne.
    Au début on traitait les 14 000 colonnes une par une, c'était beaucoup trop lent."""
    return df[colonnes_actions(df)].stack().dropna().astype(str)


def extract_pattern_frequencies(df, critere, prefixe=""):
    """Pour chaque utilisateur, la part de chaque motif trouvé par l'expression régulière
    (par exemple toutes les chaînes entre $...$)."""
    motifs = jetons_longs(df).str.extract(critere, expand=False).dropna()
    motifs = prefixe + motifs
    comptes = motifs.groupby(level=0).value_counts().unstack(fill_value=0)
    par_util = comptes.groupby(df.loc[comptes.index, "util"].astype(str)).sum()
    freq_critere = par_util.div(par_util.sum(axis=1), axis=0).fillna(0)
    return freq_critere.rename_axis("util").reset_index()


def temps_par_session(df, updated_df):
    """Durée de chaque session (dernier tXX), puis résumé par utilisateur.
    Attention : la fonction ajoute aussi la colonne max_t dans df, on en a besoin plus loin."""
    temps = jetons_longs(df).str.extract(r"^t(\d+)$", expand=False).dropna().astype(float)
    # une session sans aucun marqueur a duré moins de 5 s : on met 0
    df["max_t"] = temps.groupby(level=0).max().reindex(df.index).fillna(0)

    new_features = df.groupby(df["util"].astype(str))["max_t"].agg(
        min_t="min", max_t="max", moyenne_t="mean", std_t="std"
    ).reset_index()
    updated_df = updated_df.assign(util=updated_df["util"].astype(str))
    return updated_df.merge(new_features, on="util", how="left")


def rapidite_util(df, updated_df):
    """Vitesse moyenne de chaque utilisateur : nombre d'actions par seconde de session."""
    jetons = jetons_longs(df)
    est_action = ~jetons.str.match(r"^t\d+$")
    n_actions = est_action.groupby(level=0).sum().reindex(df.index).fillna(0)
    rapidite = n_actions / df["max_t"].clip(lower=5)
    par_util = rapidite.groupby(df["util"].astype(str)).mean().rename("rapidite").reset_index()
    updated_df = updated_df.assign(util=updated_df["util"].astype(str))
    return updated_df.merge(par_util, on="util", how="left")


def stat_actions(df, updated_df):
    """Dans chaque session, on compte les actions en modification, sur une fiche,
    sur un écran et avec un changement de configuration. Puis min / max / moyenne / écart type par utilisateur."""
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
        it.columns = [f"{statistique}_{key}" for statistique in it.columns]   # ex. mean_fiche
        updated_df = updated_df.merge(it.reset_index(), on="util", how="left")
    return updated_df
