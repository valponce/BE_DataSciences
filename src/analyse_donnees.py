import pandas as pd

def read_ds(ds_name: str):
    with open(f'../Data/{ds_name}.csv') as f:
        max_actions = max(( len( str(c).split(",")) for c in f.readlines() ))
        f.seek(0)
        _names = ["util", "navigateur"] if "train" in ds_name else ["navigateur"]
        _names.extend(range(max_actions - len(_names)))
        return pd.read_csv(f, names=_names, dtype= str)

if __name__ == "__main__":
    features_train = read_ds("train")
    features_test = read_ds("test")
    print(features_train.shape, features_test.shape)