"""
BE1 Data Science : "Qui utilise mon appli ?" . Groupe TD 4
3. VISUALISATION : graphiques de l'exploration (navigateurs, actions, composants, χ²)
et graphiques d'analyse des modèles (comparaison, confusion, importances, courbe de validation).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from seaborn import heatmap
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, validation_curve
from sklearn.tree import plot_tree

from configuration import BLEU, GRAINE, GRIS, METRIQUE, ORANGE
from preparation import filter_action, jetons_longs

def plot_distribution(effectifs, title, xlabel, ylabel, bins=30, couleur_barres=BLEU, couleur_mediane=ORANGE):
    """Histogramme avec la médiane en pointillés (on l'utilise un peu partout)."""
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
    """Les n valeurs les plus fréquentes en barres horizontales."""
    top = compteur.most_common(n)[::-1]
    # Les noms d'écrans sont très longs ; la fin est la partie la plus parlante, donc on garde la fin
    etiquettes = [v if len(v) <= 45 else "…" + v[-44:] for v, _ in top]
    plt.figure(figsize=(9, 5.5))
    # le $ est interprété comme du LaTeX par matplotlib, il faut l'échapper
    plt.barh([e.replace("$", r"\$") for e in etiquettes], [c for _, c in top], color=couleur)
    plt.title(titre)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.show()


def browsers_analysis(df) -> None:
    """Combien de sessions et d'utilisateurs par navigateur, et est-ce qu'un utilisateur
    change parfois de navigateur ? (Réponse : non, ce qui en fait une très bonne variable.)"""
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
    """Répartition du nombre de sessions par utilisateur ; ajoute la colonne nb_sessions au profil."""
    sessions_per_users = raw_df.groupby(raw_df["util"].astype(str)).size().rename("nb_sessions").reset_index()
    updated_df = updated_df.assign(util=updated_df["util"].astype(str)).merge(sessions_per_users, on="util", how="left")
    plot_distribution(updated_df["nb_sessions"], "Nombre de sessions par utilisateur",
                      "Nombre de sessions", "Nombre d'utilisateurs")
    print(updated_df["nb_sessions"].describe().round(1))
    return updated_df


def _analyze_chi2_and_select(X, y, titre, seuil_relatif=0.25, n_affiches=35):
    """Test du χ² entre chaque colonne et l'utilisateur. On trace les meilleurs scores
    et on garde les colonnes qui atteignent au moins 25 % du meilleur score
    (seuil choisi en regardant le graphique, pas de justification théorique)."""
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
    """Heatmap des corrélations. On se limite à 20 colonnes, au-delà on ne lit plus rien."""
    colonnes = colonnes[:n_max]
    corr = table[colonnes].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title(titre)
    fig.tight_layout()
    plt.show()


def actions_analysis(df):
    """Tableau récapitulatif des types d'action, puis test du χ² et corrélations.
    Renvoie les types d'action les plus liés à l'utilisateur."""
    jetons = jetons_longs(df)
    types = jetons[~jetons.str.match(r"^t\d+$")].map(filter_action)

    # une ligne par session, une colonne par type d'action, et dans chaque case le nombre d'occurrences
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

    # On travaille en fréquences plutôt qu'en comptes, sinon les longues sessions dominent le test
    frequences = comptes.div(comptes.sum(axis=1), axis=0)
    selected_actions = _analyze_chi2_and_select(
        frequences, utilisateurs, "Les types d'action les plus liés aux utilisateurs (test du χ²)")
    tracer_correlations(frequences, selected_actions, "Corrélation entre les types d'action retenus")
    return selected_actions


def analyser_composants_logiciels(df):
    """Même analyse que pour les actions, mais sur les chaînes, les écrans et les configurations."""
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
    """Gris = score sur les sessions déjà vues, bleu = score sur la validation.
    Un grand écart entre les deux barres, c'est du sur-apprentissage."""
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
    """Avec 247 utilisateurs, la matrice complète est illisible : on n'affiche que ceux qu'on choisit."""
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
    """On voulait vérifier une intuition : les utilisateurs avec peu de sessions
    sont-ils moins bien reconnus ? Un point = un utilisateur."""
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
    """Importance des variables pour l'arbre ou la forêt (les modèles linéaires n'ont pas cet attribut)."""
    noms = pipeline.named_steps["preparation"].get_feature_names_out()
    importances = pd.Series(pipeline.named_steps["modele"].feature_importances_, index=noms)
    top = importances.sort_values(ascending=False).head(n)[::-1]
    plt.figure(figsize=(9, 6))
    # scikit-learn nomme les variables "groupe__nom" : on enlève le groupe et on coupe les noms trop longs
    etiquettes = [nom.split("__", 1)[-1][-55:].replace("$", r"\$") for nom in top.index]
    plt.barh(etiquettes, top.values, color=BLEU)
    plt.xlabel("Importance (diminution moyenne de l'impureté de Gini)")
    plt.title(titre)
    plt.tight_layout()
    plt.show()
    return importances.sort_values(ascending=False)


def importance_par_groupe(importances):
    """Somme des importances par famille de variables, pour savoir quelle famille compte le plus."""
    groupes = importances.groupby(importances.index.str.split("__").str[0]).sum()
    return groupes.sort_values(ascending=False).round(3).rename("importance totale")


def tracer_arbre(pipeline, profondeur=3):
    """Les premières questions que pose l'arbre de décision. Au-delà de 3 niveaux, c'est illisible."""
    noms = [n.split("__", 1)[-1][-35:] for n in pipeline.named_steps["preparation"].get_feature_names_out()]
    arbre = pipeline.named_steps["modele"]
    plt.figure(figsize=(18, 7))
    plot_tree(arbre, max_depth=profondeur, feature_names=noms, class_names=list(arbre.classes_),
              filled=True, impurity=False, proportion=True, fontsize=7)
    plt.title(f"Arbre de décision : {profondeur} premiers niveaux")
    plt.show()


def tracer_courbe_validation(pipeline, X, y, nom_parametre, valeurs, titre, echelle_log=True, plis=3):
    """Score en apprentissage et en validation croisée pour plusieurs valeurs d'un paramètre.
    Ça nous a servi à voir à partir de quand le modèle commence à sur-apprendre."""
    score_app, score_val = validation_curve(
        pipeline, X, y, param_name=nom_parametre, param_range=valeurs,
        cv=StratifiedKFold(plis, shuffle=True, random_state=GRAINE), scoring=METRIQUE, n_jobs=-1)
    plt.figure()
    plt.plot(valeurs, score_app.mean(axis=1), "o-", color=GRIS, label="apprentissage")
    plt.plot(valeurs, score_val.mean(axis=1), "o-", color=BLEU, label="validation croisée")
    # la zone colorée montre l'écart type entre les plis
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
