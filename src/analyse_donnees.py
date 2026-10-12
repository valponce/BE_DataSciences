import numpy as np
import pandas as pd
import re

def read_ds(ds_name: str):
    with open(f'Data/{ds_name}.csv') as f:
        max_actions = max(( len( str(c).split(",")) for c in f.readlines() ))
        f.seek(0)
        _names = ["util", "navigateur"] if "train" in ds_name else ["navigateur"]
        _names.extend(range(max_actions - len(_names)))
        return pd.read_csv(f, names=_names, dtype= str)
    
def to_categories(df, col="util"):
    df[col] = pd.Categorical(df[col]).codes
    return df
    
def filter_action(value: str):
    for delim in [ "(", "<", "$", "1"]:
        if delim in value and (low_ind := value.index(delim)):
            value = value[:low_ind]
    return value

def browsers_analysis(df) -> None: 
    nb_browsers_per_user = df.groupby('util')['navigateur'].nunique().reset_index()
    print(f"Nombre maximum de navigateurs utilisés par un utilisateur: {nb_browsers_per_user['navigateur'].max()}")

    # Compte le nombre de sessions par navigateur
    browsers_used = df['navigateur'].value_counts()

    # Compte le nombre de sessions par navigateur
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

    return pd.DataFrame({"sessions": browsers_used, "part (%)": percentages,"utilisateurs l'ayant utilisé": nb_users_per_browsers })

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

def actions_analysis(df,updated_df, actions_list):
    pattern_ecran = re.compile(r"\((.*?)\)")
    pattern_conf_ecran = re.compile(r"<(.*?)>")
    pattern_chaine = re.compile(r"\$(.*?)\$")

    cols_actions = df.columns.difference(["util", "navigateur"])
    
    actions_filtered = df[cols_actions].map(apply_filter)

    counts_per_row = actions_filtered.stack().groupby(level=0).value_counts()

    df_filtered = counts_per_row.unstack(fill_value=0)
    df_filtered['util'] = df["util"]

    #new_features = df_filtered.groupby("util").agg(['min', 'max', 'mean', 'std'])
    new_features = df_filtered.groupby("util").agg(['mean'])

    new_features.columns = [f"{stat}_{col}" for col, stat in new_features.columns]
    new_features = new_features.reset_index()

    updated_df = updated_df.merge(new_features, on="util", how="left")
    
    return updated_df



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
