import pandas as pd
from sklearn.preprocessing import StandardScaler

import analyse_donnees as AN

def standardize(df_input):
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df_input),columns=df_input.columns)
    return df_standardized

def transform_train_data(features_train,p_select):

    features_train["util"] = pd.Categorical(features_train["util"])

    # Nouvelle Dataframe
    updated_df = pd.DataFrame({'util':features_train["util"].unique()})
    updated_df["util"] = pd.Categorical(updated_df["util"])

    # Création de la colonne des navigateurs
    updated_df = features_train[["util","navigateur"]].groupby("util")["navigateur"].first().reset_index()
    updated_df = AN.to_categories(updated_df,"navigateur")


    # Création des colonnes d'actions
    selected_features, high_corr, df_freq = AN.configuration_analysis(features_train,[r'\$(.*?)\$',r'\((.*?)\)',r'\<(.*?)\>'],p_select)
    selected_features = [ value for value in selected_features if value not in high_corr['Feature_2']]+['util']
    
    # Ajout de grandeurs retenues à la nouvelle df
    updated_df = updated_df.merge(df_freq[selected_features], on="util", how="left")

    updated_df = AN.temps_par_session(features_train,updated_df)
    updated_df = AN.rapidite_util(features_train,updated_df)
    #updated_df.iloc[:,2:] = standardize(updated_df.iloc[:,2:])

    return updated_df, selected_features[:-1] # enlève util

def transform_test_data(features_test,selected_features):

    updated_df = pd.DataFrame()
    updated_df ['navigateur'] = pd.Categorical(features_test.iloc[:,0]).codes

    # Création des colonnes d'actions
    df = AN.configuration_analysis_test(features_test,[r'\$(.*?)\$',r'\((.*?)\)',r'\<(.*?)\>'])
    
    # Ajout de grandeurs retenues à la nouvelle df
    #updated_df.iloc[:,1:]= updated_df[selected_features]
    for col in selected_features:
        if col in updated_df.columns:
            # Si la colonne existe dans df_freq, on la récupère
            updated_df[col] = df[col].values
        else:
            # Sinon, on crée une colonne remplie de 0
            updated_df[col] = 0.0
    
    df_t_nums = features_test.iloc[:,1:].apply(
        lambda col: col.astype(str).str.extract(r"t(\d+)", expand=False).astype(float)
    )

    max_t_values = df_t_nums.max(axis=1)
    updated_df["moyenne_t"] =  max_t_values
    features_test["max_t"] = max_t_values

    updated_df = AN.rapidite_util(features_test,updated_df,False)
    #updated_df = updated_df.loc[:, ~updated_df.columns.duplicated()]
    #updated_df.iloc[:,1:] = standardize(updated_df.iloc[:,1:])

    return updated_df