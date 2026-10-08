import pandas as pd

def read_ds(ds_name: str):
    with open(f'Data/{ds_name}.csv') as f:
        max_actions = max(( len( str(c).split(",")) for c in f.readlines() ))
        f.seek(0)
        _names = ["util", "navigateur"] if "train" in ds_name else ["navigateur"]
        _names.extend(range(max_actions - len(_names)))
        return pd.read_csv(f, names=_names, dtype= str)


def browsers_per_player(df):
    navigateurs_par_user = df.groupby('util')['navigateur'].nunique().reset_index()
    return navigateurs_par_user['navigateur'].max()

def transform_navig(df,updated_df):
    nav_transform = []
    df = df.groupby('util')['navigateur'].first().reset_index()
    dic = {"Firefox":0,"Google Chrome":1,"Microsoft Edge":2,"Opera":3}
    for value in df['navigateur']:
        nav_transform.append(dic[value])
    updated_df['navigateur'] = nav_transform
    return updated_df

def temps_par_session(df):
   
    # Sélection des colonnes d'actions 
    cols_actions = df.columns.difference(["util", "navigateur"])

    # Extraction du nombre de chaque tXX pour toutes les colonnes d'actions
    # \d+ capture la partie numérique après le 't'
    # expend =false -> renvoie serie au lieu df
    df_t_nums = df[cols_actions].apply(
        lambda col: col.astype(str).str.extract(r"t(\d+)", expand=False).astype(float)
    )
    df["max_t"] = df_t_nums.max(axis=1) 

    df = df.groupby("util")["max_t"].agg(
        min_t="min",
        max_t="max",
        moyenne_t="mean",
        std_t="std"
    ).reset_index()

    return df

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
        rapidite.append((index_firstNaN - nb_interval)/nb_interval)
        if ((index_firstNaN - nb_interval)/nb_interval<0):
            print(index_firstNaN)
            print(nb_interval)
            print(max_t)

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
